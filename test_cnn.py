
import torch

import torch._dynamo  # force torch's heavy lazy imports to load before TF/grpc/protobuf
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Import from our project
from src.embeddings import (
    pca_embedding, lle_embedding, umap_embedding, 
    wavelet_embedding, fft_embedding, AE_embedding,
    graph_embedding, tda_embedding, 
    nnclr_cnn_embedding, nnclr_lstm_embedding
)
from src.classifiers import (
    optimize_LOGRG, optimize_DT, optimize_RF, optimize_KNN, 
    optimize_XGBOOST, optimize_SVM, optimize_MLP, optimize_NB
)
from src.utils.preprocessing import load_dataset, prepare_dataset
from src.utils.evaluation import (
    evaluate_embeddings, visualize_embeddings, 
    compare_classifiers, generate_classification_report
)
from src.utils.visualization import umap_plt, plot_embedding_comparison

import sys
sys.path.insert(0, "src")
from run_experiments import *
from sklearn.model_selection import train_test_split
# Get the list of countries and their corresponding continents from the provided URL.
url_data = (r'https://raw.githubusercontent.com/oderofrancis/rona/main/Countries-Continents.csv')
data_countries = pd.read_csv(url_data)

ntl_data = pd.read_csv('final_pre365/ntl_cases.csv')[['case_id', 'time', 'ntl_filled']]
ntl_data['time'] = pd.to_datetime(ntl_data['time'])
ntl_data = ntl_data.sort_values(['case_id', 'time']).reset_index(drop=True)
ntl_data['time_step'] = ntl_data.groupby('case_id').cumcount()

cases = pd.read_csv('final_pre365/cases.csv')
cases['event_date'] = pd.to_datetime(cases['event_date'])


missing_countries = {
    "Continent": [
        "Africa",
        "Africa",
        "Asia",
        "Asia",
        "North America",
        "Asia",
        "North America",
        "Africa",
        "Asia",
        "North America",
    ],
    "Country": [
        "DR Congo",
        "Mayotte",
        "Myanmar",
        "Myanmar (post-coup)",
        "México",
        "Palestine (Israel)",
        "Puerto Rico",
        "Réunion",
        "Taiwan",
        "United States",
    ],
}

data_countries = pd.concat([
    data_countries,
    pd.DataFrame(missing_countries)
], ignore_index=True)
data_countries.rename(columns={"Country": "country"}, inplace=True)

cases = cases.merge(
    data_countries[['country', 'Continent']],
    on='country',
    how='left'
).set_index('case_id')

event_types = sorted(cases['event_type'].unique())

for date_col in ['event_start', 'event_end', 'window_start', 'window_end']:
    cases[date_col] = pd.to_datetime(cases[date_col] )

feature_list = ['event_type', 'Continent']
feature_dict = dict()
for feature in feature_list:
    feature_value_dict = {feature_value : i for i, feature_value in enumerate(cases[feature].unique())}
    
    cases[feature + '_class'] = cases[feature].map(feature_value_dict)
    feature_dict[feature] = feature_value_dict

colors = {'Continent': {
            'Africa':'green', 
            'Asia': 'red',
            'Europe': 'darkviolet',
            'North America': 'blue',
            'South America': 'darkorange',
            'Oceania': 'mediumturquoise'}, 
            'event_type': {'conflict': 'tomato', 'cyclone': 'cadetblue', 'earthquake': 'mediumseagreen', 'flood': 'dodgerblue'}}

from sklearn.preprocessing import StandardScaler
def calculate_window(ntl_data, cases, d_window):
    case_ids = dict()
    frames = []                       # collect pieces, concat once at the end
    cases_window = cases.copy()

    # plain Python float -> no NumPy "generic unit" timedelta warning
    window = pd.Timedelta(days=float(d_window))

    def _rescale(frame):
        frame = frame.copy()          # avoids SettingWithCopyWarning
        frame['time_step'] = frame.groupby('case_id').cumcount()
        frame['ntl_std'] = StandardScaler().fit_transform(
            frame['ntl_filled'].values.reshape(-1, 1)
        ).ravel()
        return frame

    for case_id, df in ntl_data.groupby('case_id'):
        if cases.loc[case_id, 'n_events'] == 1:
            event_date = cases.loc[case_id, 'event_date']
            start_date = event_date - window
            end_date = event_date + window

            cases_window.loc[case_id, 'window_start'] = start_date
            cases_window.loc[case_id, 'window_end'] = end_date

            df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            case_ids[case_id] = case_id

            if not df.empty and not df.isna().all().all():
                frames.append(_rescale(df))

        else:
            event_dates = pd.to_datetime(
                list(set(cases.loc[case_id, 'event_dates'].split(';')))
            )
            cases_window.drop(index=case_id, inplace=True)

            for i, event_date in enumerate(event_dates):
                new_case_id = f"{case_id}_event_{i+1}"
                start_date = event_date - window
                end_date = event_date + window

                cases_window.loc[new_case_id] = cases.loc[case_id]
                cases_window.loc[new_case_id, 'event_date'] = event_date
                cases_window.loc[new_case_id, 'window_start'] = start_date
                cases_window.loc[new_case_id, 'window_end'] = end_date

                case_ids[new_case_id] = case_id

                df_event = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

                if not df_event.empty and not df_event.isna().all().all():
                    df_event = df_event.copy()
                    df_event['case_id'] = new_case_id   # set before cumcount
                    frames.append(_rescale(df_event))

    new_ntl = (pd.concat(frames, ignore_index=True)
               if frames else pd.DataFrame(columns=ntl_data.columns))
    
    cases_window['original_id'] = cases_window.index.map(case_ids)

    return case_ids, new_ntl, cases_window

embedding_methods = ['nnclr_lstm']

for feature in feature_list[:1]:
    train_idx, val_idx = train_test_split(
        cases.index,
        test_size=0.2,
        stratify=cases[feature],
        random_state=42,
    )

    cases[f'split_{feature}'] = 'train'
    cases.loc[val_idx, f'split_{feature}'] = 'val'

    ntl_data = ntl_data.merge(
    cases[[f'split_{feature}']].reset_index(inplace=False),
    on='case_id',
    how='left')
            
    
    for d_window in [ 30]:
        case_ids,  ntl_window, cases_window = calculate_window(ntl_data, cases, d_window)
        results = {
                'dataset': d_window,
                'embedding_results': {},
                'classifier_results': {},
                'combined_results': {}
            }
        X_train_norm = ntl_window[ntl_window[f'split_{feature}'] == 'train'].pivot(index='case_id', columns='time_step', values='ntl_std').dropna()
        X_val_norm = ntl_window[ntl_window[f'split_{feature}'] == 'val'].pivot(index='case_id', columns='time_step', values='ntl_std').dropna()
        X_test_norm = X_val_norm
        #X_full = ntl_window.pivot(index='case_id', columns='time_step', values='ntl_std').dropna()
        
        

        dataset_path = f'results/{feature}/data_{str(d_window)}/data_{str(d_window)}'
        os.makedirs(f'results/{feature}/data_{str(d_window)}',exist_ok=True)

        output_dir = f'plots/{feature}/data_{str(d_window)}'
        os.makedirs(output_dir ,exist_ok=True)

        y_train = cases_window.loc[X_train_norm.index, feature + '_class'].values
        y_val = cases_window.loc[X_val_norm.index, feature + '_class'].values
        y_test =y_val
        
        y_train_labels = cases_window.loc[X_train_norm.index, feature].values
        y_val_labels = cases_window.loc[X_val_norm.index, feature ].values
        y_test_labels =y_val_labels

        for embed_method in embedding_methods:
            print(f"\nEvaluating embedding method: {embed_method}")
            # Run embedding
            try:
                train_emb, val_emb, test_emb, embed_time = run_embedding_experiment(
            X_train_norm.values, X_val_norm.values, X_test_norm, y_train, y_val, y_test, embed_method
                )

                # Store embedding results
                results['embedding_results'][embed_method] = {
                'time': embed_time,
                'shape': train_emb.shape if hasattr(train_emb, 'shape') else None
                }

                # Visualize embeddings
                try:
                    if embed_method in ['nnclr_cnn', 'nnclr_lstm']:
                        # For NNCLR methods, use umap_plt from visualization module
                        umap_plt(
                        train_emb, val_emb, test_emb, 
                        y_train_labels, y_val_labels, y_test_labels, 
                        f"{embed_method.upper()} Embedding (window = {d_window})", colors= colors[feature], filename=os.path.join(output_dir, f'{embed_method}_visualization.png')
                            )
                    else:
                        # For other methods, use visualize_embeddings from evaluation module
                        fig = visualize_embeddings(
                        train_emb,labels= y_train_labels, method='tsne', colors= colors[feature],
                        title=f'{embed_method.upper()} Embedding Visualization (window = {d_window})'
                            )
                        fig.savefig(os.path.join(output_dir, f'{embed_method}_visualization.png'))
                        plt.close(fig)
                except Exception as viz_error:
                    print(f"  - Warning: Visualization failed: {viz_error}")

                # Run classifier experiments
                for clf_name in classifiers:
                    print(f"  - Testing classifier: {clf_name}")

                    experiment_name = f"{dataset_path}_{embed_method}_{clf_name}"

                    try:
                        best_params, test_score, clf_time = run_classifier_experiment(
                train_emb, val_emb, test_emb, y_train, y_val, y_test, 
                clf_name, experiment_name
                    )

                        # Store results
                        if embed_method not in results['combined_results']:
                            results['combined_results'][embed_method] = {}

                        results['combined_results'][embed_method][clf_name] = {
                    'accuracy': test_score,
                    'params': best_params,
                    'time': clf_time
                        }

                        print(f"    Accuracy: {test_score:.4f}, Time: {clf_time:.2f}s")

                    except Exception as e:
                        print(f"    Error with classifier {clf_name}: {e}")
                        results['combined_results'][embed_method][clf_name] = {
                    'error': str(e)
                    }
                    

            except Exception as e:
                print(f"Error with embedding method {embed_method}: {e}")
                results['embedding_results'][embed_method] = {
                'error': str(e)
                    }
        # Save results
        rows = []

        for embedding, classifiers in results["combined_results"].items():
            for classifier, metrics in classifiers.items():
                rows.append({
                    "embedding": embedding,
                    "classifier": classifier,
                    "accuracy": metrics["accuracy"],
                    "time": metrics["time"],
                    "params": metrics["params"]
                })

        results_df = pd.DataFrame(rows)    
        results_df.to_csv(dataset_path+'_results.csv')