#%%
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
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import from our project
from src.embeddings import (
    pca_embedding, lle_embedding, umap_embedding, 
    wavelet_embedding, fft_embedding, AE_embedding,
    graph_embedding, tda_embedding, 
    nnclr_cnn_embedding, nnclr_lstm_embedding, mantis_embedding
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


#%%
train_val_ratio = 0.5

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
        "Oceania",
        "North America",
        "North America",
        "North America",
        "South America",
        "Africa",
        "Europe",
        "Africa",
        "Africa",
        "Europe",
        "Africa",
        "Asia",
        "Asia",
        "Asia",
        "Asia",
        "Asia",
        "Oceania",
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
        "French Polynesia",
        "Aruba",
        "Curaçao",
        "Martinique",
        "French Guiana",
        "Western Sahara",
        "Jersey",
        "São Tomé and Príncipe",
        "Republic of the Congo",
        "Kosovo",
        "Swaziland",  # now officially Eswatini
        "Northern Cyprus",
        "Palestine",
        "North Korea",
        "South Korea",
        "Timor-Leste",
        "New Caledonia",
    ],
}
#%%
# Get the list of countries and their corresponding continents from the provided URL.
url_data = (r'https://raw.githubusercontent.com/oderofrancis/rona/main/Countries-Continents.csv')
data_countries = pd.read_csv(url_data)


data_countries = pd.concat([
    data_countries,
    pd.DataFrame(missing_countries)
], ignore_index=True)
data_countries.rename(columns={"Country": "country"}, inplace=True)


ntl_data = pd.read_csv('final_pre365/ntl_cases.csv')[['case_id', 'time', 'ntl_filled']]
ntl_data['time'] = pd.to_datetime(ntl_data['time'])
ntl_data = ntl_data.sort_values(['case_id', 'time']).reset_index(drop=True)
ntl_data['time_step'] = ntl_data.groupby('case_id').cumcount()

cases = pd.read_csv('final_pre365/cases.csv').drop(columns=[ 'source_cluster_id',
         'catalog_event_date', 'catalog_event_dates',
       'shift_reason', 'window_anchor',
        'notes', 'raster_path'])

for date_col in ['event_date', 'event_start', 'event_end', 'window_start', 'window_end']:
    cases[date_col] = pd.to_datetime(cases[date_col] )
# Add the continent/

cases = cases.merge(
    data_countries[['country', 'Continent']],
    on='country',
    how='left'
).set_index('case_id')

    
    
ntl_cities = pd.read_csv('peaceful_cities/ntl_cities.csv').drop(columns= ['ntl', 'ntl_valid_frac','ntl_missing'])
ntl_cities['time'] = pd.to_datetime(ntl_cities['time'])
ntl_cities.rename(columns={'city_id':'case_id'}, inplace=True)

# Clean data
ntl_cities = ntl_cities.groupby('case_id').filter(lambda group: not group.isna().any().any()) ## remove these from cases
ntl_cities = ntl_cities.sort_values(['case_id','time'])
def is_consecutive(s):
    s = s.sort_values()
    # no duplicates, and every gap is exactly 1 day
    return s.is_unique and (s.diff().dropna() == pd.Timedelta(days=1)).all()

valid = ntl_cities.groupby("case_id")["time"].apply(is_consecutive)
ntl_cities = ntl_cities[ntl_cities["case_id"].isin(valid[valid].index)]

# keep cities with at least 60 days of data
ntl_cities['time_step'] = ntl_cities.groupby('case_id').cumcount()
n_days = ntl_cities.groupby('case_id')['time_step'].transform('size')
ntl_cities = ntl_cities[n_days >= 60]


cases_cities = gpd.read_file('peaceful_cities/cities.geojson')[['city_id', 'city', 'country']]
cases_cities.rename(columns={'city_id':'case_id'}, inplace=True)
cases_cities = cases_cities[cases_cities["case_id"].isin(ntl_cities['case_id'].unique())]
cases_cities['event_type'] = 'peace'
cases_cities.loc[cases_cities['country'] == 'Democratic Republic of the Congo', 'country'] = 'DR Congo'
cases_cities.loc[cases_cities['country'] == "Côte d'Ivoire", 'country'] = "Cote d'Ivoire"


cases_cities = cases_cities.merge(
    data_countries[['country', 'Continent']],
    on='country',
    how='left'
)

# Randomly select cities
peaceful_cities, _ = train_test_split(
        cases_cities,
        train_size=len(cases[cases['event_type'] == 'conflict']),
        stratify=cases_cities['Continent'],
        random_state=42,
    )

ntl_peaceful_cities = ntl_cities[ntl_cities['case_id'].isin(peaceful_cities['case_id'].unique())]

# Randomly select event date in peaceful cities

random_dates = {}
for case_id, grp in ntl_peaceful_cities.groupby("case_id"):
    start, end = grp["time"].min(), grp["time"].max()

    lower = start + pd.DateOffset(months=1)   # must be more than 1 month after the start
    upper = end - pd.Timedelta(days=30)       # must be at least 30 days before the end

    candidates = grp.loc[(grp["time"] > lower) & (grp["time"] <= upper), "time"]

    if candidates.empty:                      # city too short for this window
        continue

    random_dates[case_id] = candidates.iloc[np.random.default_rng(42).integers(len(candidates))]
    
# Concatenate dataframes
peaceful_cities['n_events'] = 1
event = peaceful_cities['case_id'].map(random_dates)
peaceful_cities[['event_date', 'event_start', 'event_end', 'event_dates']] = pd.concat([event] * 4, axis=1).values
window_start = ntl_peaceful_cities.groupby('case_id')['time'].min()
peaceful_cities['window_start'] = peaceful_cities['case_id'].map(window_start)
window_end = ntl_peaceful_cities.groupby('case_id')['time'].max()
peaceful_cities['window_end'] = peaceful_cities['case_id'].map(window_end)

cases =  pd.concat([cases, peaceful_cities.set_index('case_id')],ignore_index=False)

ntl_data = pd.concat([ntl_data, ntl_peaceful_cities], ignore_index=True)

cases['event_type'] = cases['event_type'].apply(lambda x : 'disaster' if x in ['earthquake', 'cyclone', 'flood'] else x)

#%%

feature_list = ['event_type', 'Continent']
feature_dict = {'event_type': {'disaster': 0, 'conflict': 1, 'peace': 2},
 'Continent': {'Asia': 0,
  'South America': 1,
  'North America': 2,
  'Oceania': 3,
  'Africa': 4,
  'Europe': 5}}
for feature in feature_list:
    cases[feature + '_class'] = cases[feature].map(feature_dict[feature])
    
#%%
colors = {'Continent': {
            'Africa':'green', 
            'Asia': 'red',
            'Europe': 'darkviolet',
            'North America': 'blue',
            'South America': 'darkorange',
            'Oceania': 'mediumturquoise'}, 
            'event_type': {'conflict': 'tomato',  'disaster': 'mediumseagreen',  'peace': 'deeppink'}}
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

#embedding_methods = ['nnclr_lstm']
classifiers = ['logreg', 'dt', 'rf', 'knn','xgboost', 'svm','nb']
all_results = {}   # {"<feature>_<window>": results_dict}
all_rows = []  
for feature in feature_list[:1]:
    train_idx, val_idx = train_test_split(
        cases.index,
        test_size=0.5,
        stratify=cases[feature],
        random_state=42,
    )

    cases[f'split_{feature}'] = 'train'
    cases.loc[val_idx, f'split_{feature}'] = 'val'

    ntl_data = ntl_data.merge(
    cases[[f'split_{feature}']].reset_index(inplace=False),
    on='case_id',
    how='left')
            
    
    for d_window in [ 14]:
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
        
        y_train = cases_window.loc[X_train_norm.index, feature + '_class'].values
        y_val = cases_window.loc[X_val_norm.index, feature + '_class'].values
        y_test =y_val
        
        y_train_labels = cases_window.loc[X_train_norm.index, feature].values
        y_val_labels = cases_window.loc[X_val_norm.index, feature ].values
        y_test_labels =y_val_labels

embedding_methods = ['Mantis-8M_pretrain_continued']
embed_method = 'Mantis-8M_pretrain_continued'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F
from mantis.architecture import Mantis8M
from mantis.architecture import MantisV2   
from mantis.trainer import MantisTrainer

def to_mantis_input(X, target_len=512):
    X = torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)
    return F.interpolate(X, size=target_len, mode="linear", align_corners=False).numpy()

start_time = time.time()
#cls, ckpt, rev = CHECKPOINTS[name]
net = Mantis8M(device=device)

state = torch.load('mantis_checkpoints/mantis_14d_continued_100_epoch.pt', map_location=device)
net.load_state_dict(state)  

net.load_state_dict(state)      

trainer = MantisTrainer(device=device, network=net)
target_len=512
batch_size=64
with torch.no_grad():
    train_emb = trainer.transform(to_mantis_input(X_train_norm, target_len), batch_size=batch_size)
    val_emb = trainer.transform(to_mantis_input(X_val_norm, target_len), batch_size=batch_size)
    test_emb = trainer.transform(to_mantis_input(X_test_norm, target_len), batch_size=batch_size)
    
end_time = time.time()
embed_time = end_time - start_time
results['embedding_results'][embed_method] = {
                'time': embed_time,
                'shape': train_emb.shape if hasattr(train_emb, 'shape') else None
                }
del trainer
torch.cuda.empty_cache()

fig = visualize_embeddings(
                        train_emb,labels= y_train_labels, method='umap', colors= colors[feature],
                        title=f'Mantis-8M_pretrain_continued Embedding Visualization (window = {d_window})'
                            )
fig.savefig( f'mantis_14d_continued_100_epoch_umap_visualization.png')
plt.close(fig)

for clf_name in classifiers:
    print(f"  - Testing classifier: {clf_name}")

    experiment_name = f"{embed_method}_{clf_name}"

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
                    
                
        # Save results
    all_results[feature + '_' +str( d_window)] = results
    for emb, clfs in results['combined_results'].items():
        for clf, m in clfs.items():
            all_rows.append({
                            'feature': feature,
                            'window': d_window,
                            'embedding': emb,
                            'classifier': clf,
                            'accuracy': m.get('accuracy'),
                            'time': m.get('time'),
                            'params': m.get('params'),
                            'error': m.get('error'),
                        })  
        




pd.DataFrame(all_rows).to_csv(f'mantis_14d_continued_100_epoch_results.csv', index=False)


"""
for embed_method in embedding_methods:
    
    if embed_method == 'No-embed':
        train_emb, val_emb, test_emb, embed_time = X_train_norm, X_val_norm, X_test_norm, 0
    
    elif 'Mantis' in embed_method:
        name = embed_method + '_' + str(d_window) if 'pretrain' in embed_method else embed_method
        train_emb, val_emb, test_emb, embed_time = run_embedding_experiment(
                            X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, name)
    else:
        train_emb, val_emb, test_emb, embed_time = run_embedding_experiment(
                            X_train_norm.values, X_val_norm.values, X_test_norm.values,
                            y_train, y_val, y_test, embed_method)
  
    
                      
    clf_name = 'knn'
    dataset_path = f'test_plot'
    experiment_name = f"{dataset_path}/{embed_method}_{clf_name}"
    fig = visualize_embeddings(
                            train_emb,labels= y_train_labels, method='umap', colors= colors[feature],
                            title=f'{embed_method.upper()} Embedding Visualization (window = {d_window})'
                                )
    fig.savefig(os.path.join(dataset_path, f'{embed_method}_umap_visualization.png'))
    plt.close(fig)

    best_params, test_score, clf_time = run_classifier_experiment(
                    train_emb, val_emb, test_emb, y_train, y_val, y_test, 
                    clf_name, experiment_name
                        )

    print(f"    Accuracy: {test_score:.4f}, Time: {clf_time:.2f}s")
    """
# %%