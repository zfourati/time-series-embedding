"""
Run experiments on time series embedding methods.

This module provides functions to run comprehensive
evaluation experiments comparing various embedding methods
and classification algorithms on time series datasets.
"""

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
    nnclr_cnn_embedding, nnclr_lstm_embedding, mantis_embedding, chronos_embedding,
    tivit_embedding, concat_embeddings, ts2vec_embedding
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


MANTIS_METHODS = {
    'Mantis-8M':    'mantis',
    'MantisPlus':   'mantis_plus',
    'Mantis-Utica': 'utica',
    'MantisV2':     'mantis_v2',
    'Mantis-8M_pretrain_continued_7' : 'Mantis-8M_pretrain_continued_7',
    'MantisPlus_pretrain_continued_7' : 'MantisPlus_pretrain_continued_7',
    'Mantis-Utica_pretrain_continued_7' : 'Mantis-Utica_pretrain_continued_7',
    'MantisV2_pretrain_continued_7' : 'MantisV2_pretrain_continued_7',
    'Mantis-8M_pretrain_scratch_7' : 'Mantis-8M_pretrain_scratch_7',
    'MantisV2_pretrain_scratch_7' : 'MantisV2_pretrain_scratch_7',
    'Mantis-8M_pretrain_continued_14' : 'Mantis-8M_pretrain_continued_14',
    'MantisPlus_pretrain_continued_14' : 'MantisPlus_pretrain_continued_14',
    'Mantis-Utica_pretrain_continued_14' : 'Mantis-Utica_pretrain_continued_14',
    'MantisV2_pretrain_continued_14' : 'MantisV2_pretrain_continued_14',
    'Mantis-8M_pretrain_scratch_14' : 'Mantis-8M_pretrain_scratch_14',
    'MantisV2_pretrain_scratch_14' : 'MantisV2_pretrain_scratch_14',
    'Mantis-8M_pretrain_continued_30' : 'Mantis-8M_pretrain_continued_30',
    'MantisPlus_pretrain_continued_30' : 'MantisPlus_pretrain_continued_30',
    'Mantis-Utica_pretrain_continued_30' : 'Mantis-Utica_pretrain_continued_30',
    'MantisV2_pretrain_continued_30' : 'MantisV2_pretrain_continued_30',
    'Mantis-8M_pretrain_scratch_30' : 'Mantis-8M_pretrain_scratch_30',
    'MantisV2_pretrain_scratch_30' : 'MantisV2_pretrain_scratch_30'
}

CHRONOS_CHECKPOINTS = {
    'Chronos-T5-small':  'amazon/chronos-t5-small',
    'Chronos-Bolt-base': 'amazon/chronos-bolt-base',
}

TIVIT_CHECKPOINTS = {
    'TiViT-CLIP-B16': ('openai/clip-vit-base-patch16', 7),   # (checkpoint, layer)
    'TiViT-CLIP-H14': ('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 14),  # layer used in the repo's example
}


def run_embedding_experiment(X_train, X_val, X_test, y_train, y_val, y_test, embedding_method, **kwargs):
    """
    Run an experiment for a specific embedding method.
    
    Parameters:
    -----------
    X_train : array-like
        Training data
    X_val : array-like
        Validation data
    X_test : array-like
        Test data
    y_train : array-like
        Training labels (needed for some embedding methods)
    y_val : array-like
        Validation labels (needed for some embedding methods)
    y_test : array-like
        Test labels (needed for some embedding methods)
    embedding_method : str
        Name of the embedding method to use
    **kwargs : dict
        Additional parameters for the embedding method
        
    Returns:
    --------
    tuple
        (train_embedded, val_embedded, test_embedded, time_taken)
    """
    start_time = time.time()
    
    # Apply the selected embedding method
    if embedding_method == 'pca':
        n_components = kwargs.get('n_components', min(X_train.shape[1], 10))
        train_emb, val_emb, test_emb = pca_embedding(X_train, X_val, X_test, n_components)
        
    elif embedding_method == 'lle':
        lle_params = {
            'n_neighbors': kwargs.get('n_neighbors', 10),
            'n_components': kwargs.get('n_components', 10),
            'random_state': kwargs.get('random_state', 42)
        }
        train_emb, val_emb, test_emb = lle_embedding(X_train, X_val, X_test, lle_params)
        
    elif embedding_method == 'umap':
        train_emb, val_emb, test_emb = umap_embedding(X_train, X_val, X_test)
        
    elif embedding_method == 'wavelet':
        train_emb, val_emb, test_emb = wavelet_embedding(X_train, X_val, X_test)
        
    elif embedding_method == 'fft':
        train_emb, val_emb, test_emb = fft_embedding(X_train, X_val, X_test)
        
    elif embedding_method == 'autoencoder':
        train_emb, val_emb, test_emb = AE_embedding(X_train, X_val, X_test)
        
    elif embedding_method == 'graph':
        train_emb, val_emb, test_emb = graph_embedding(X_train, X_val, X_test)
        
    elif embedding_method == 'tda':
        train_emb, val_emb, test_emb = tda_embedding(X_train, X_val, X_test)
    
    elif embedding_method == 'nnclr_cnn':
        # Convert to pandas DataFrame if necessary
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        if not isinstance(y_train, pd.DataFrame) and not isinstance(y_train, pd.Series):
            y_train = pd.DataFrame(y_train)
        if not isinstance(y_val, pd.DataFrame) and not isinstance(y_val, pd.Series):
            y_val = pd.DataFrame(y_val)
        if not isinstance(y_test, pd.DataFrame) and not isinstance(y_test, pd.Series):
            y_test = pd.DataFrame(y_test)
            
        width = kwargs.get('width', 128)
        n_classes = kwargs.get('n_classes', len(np.unique(y_train)))
        
        train_emb, val_emb, test_emb = nnclr_cnn_embedding(
            X_train, X_val, X_test, 
            y_train, y_val, y_test,
            width, n_classes
        )
    
    elif embedding_method == 'nnclr_lstm':
        # Convert to pandas DataFrame if necessary
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        if not isinstance(y_train, pd.DataFrame) and not isinstance(y_train, pd.Series):
            y_train = pd.DataFrame(y_train)
        if not isinstance(y_val, pd.DataFrame) and not isinstance(y_val, pd.Series):
            y_val = pd.DataFrame(y_val)
        if not isinstance(y_test, pd.DataFrame) and not isinstance(y_test, pd.Series):
            y_test = pd.DataFrame(y_test)
            
        width = kwargs.get('width', 128)
        n_classes = kwargs.get('n_classes', len(np.unique(y_train)))
        
        train_emb, val_emb, test_emb = nnclr_lstm_embedding(
            X_train, X_val, X_test, 
            y_train, y_val, y_test,
            width, n_classes
        )
    elif embedding_method in MANTIS_METHODS:
        train_emb, val_emb, test_emb = mantis_embedding(
            MANTIS_METHODS[embedding_method],
            X_train, X_val, X_test
        )
    elif embedding_method in CHRONOS_CHECKPOINTS:
        train_emb, val_emb, test_emb = chronos_embedding(
            CHRONOS_CHECKPOINTS[embedding_method], X_train, X_val, X_test)
    elif embedding_method in TIVIT_CHECKPOINTS:
        ckpt, layer = TIVIT_CHECKPOINTS[embedding_method]
        train_emb, val_emb, test_emb = tivit_embedding(
            ckpt, kwargs.get('layer', layer), X_train, X_val, X_test)
        
    elif embedding_method == 'TiViT+MantisV2':
        ckpt, layer = TIVIT_CHECKPOINTS['TiViT-CLIP-B16']
        blocks = [
            tivit_embedding(ckpt, kwargs.get('layer', layer), X_train, X_val, X_test),
            mantis_embedding('mantis_v2', X_train, X_val, X_test
                             ),
        ]
        train_emb, val_emb, test_emb = concat_embeddings(blocks)
    elif embedding_method == 'TS2Vec':
        train_emb, val_emb, test_emb = ts2vec_embedding(
            X_train, X_val, X_test,
            output_dims=kwargs.get('output_dims', 320),
            n_epochs=kwargs.get('n_epochs'),
            n_iters=kwargs.get('n_iters'),
        )
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    return train_emb, val_emb, test_emb, time_taken


def run_classifier_experiment(train_emb, val_emb, test_emb, y_train, y_val, y_test, 
                            classifier_name, experiment_name,labels=None, confusion_matrix=False ):
    """
    Run an experiment for a specific classifier on embedded data.
    
    Parameters:
    -----------
    train_emb : array-like
        Embedded training data
    val_emb : array-like
        Embedded validation data
    test_emb : array-like
        Embedded test data
    y_train : array-like
        Training labels
    y_val : array-like
        Validation labels
    y_test : array-like
        Test labels
    classifier_name : str
        Name of the classifier to use
    experiment_name : str
        Name for output files
        
    Returns:
    --------
    tuple
        (best_params, test_score, time_taken)
    """
    start_time = time.time()
    
    # Apply the selected classifier
    if classifier_name == 'logreg':
        best_params, test_score = optimize_LOGRG(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    elif classifier_name == 'dt':
        best_params, test_score = optimize_DT(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    elif classifier_name == 'rf':
        best_params, test_score = optimize_RF(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    elif classifier_name == 'knn':
        best_params, test_score = optimize_KNN(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    elif classifier_name == 'xgboost':
        best_params, test_score = optimize_XGBOOST(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    elif classifier_name == 'svm':
        best_params, test_score = optimize_SVM(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    elif classifier_name == 'mlp':
        best_params, test_score = optimize_MLP(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    elif classifier_name == 'nb':
        best_params, test_score = optimize_NB(
            train_emb, val_emb, test_emb, y_train, y_val, y_test, experiment_name, labels=labels, confusion_matrix=confusion_matrix
        )
        
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    return best_params, test_score, time_taken


def run_full_evaluation(dataset_path, target_column, embedding_methods, 
                      classifiers, output_dir='results', random_state=42):
    """
    Run a full evaluation of embedding methods and classifiers on a dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file
    target_column : str or int
        Name or index of the target column
    embedding_methods : list of str
        List of embedding methods to evaluate
    classifiers : list of str
        List of classifiers to evaluate
    output_dir : str, default='results'
        Directory to save results
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with all results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y = load_dataset(dataset_path, target_column)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.2, random_state=random_state
    )
    
    # Normalize data
    X_train_norm, X_val_norm, X_test_norm = normalize_data(X_train, X_val, X_test, method='standard')
    
    # Prepare results dictionary
    results = {
        'dataset': os.path.basename(dataset_path),
        'embedding_results': {},
        'classifier_results': {},
        'combined_results': {}
    }
    
    # Run experiments for each embedding method
    for embed_method in embedding_methods:
        print(f"\nEvaluating embedding method: {embed_method}")
        
        # Run embedding
        try:
            train_emb, val_emb, test_emb, embed_time = run_embedding_experiment(
                X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, embed_method
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
                        y_train, y_val, y_test, 
                        f"{embed_method.upper()} Embedding"
                    )
                else:
                    # For other methods, use visualize_embeddings from evaluation module
                    fig = visualize_embeddings(
                        train_emb, y_train, method='tsne', 
                        title=f'{embed_method.upper()} Embedding Visualization'
                    )
                    fig.savefig(os.path.join(output_dir, f'{embed_method}_visualization.png'))
                    plt.close(fig)
            except Exception as viz_error:
                print(f"  - Warning: Visualization failed: {viz_error}")
            
            # Run classifier experiments
            for clf_name in classifiers:
                print(f"  - Testing classifier: {clf_name}")
                
                experiment_name = f"{os.path.basename(dataset_path)}_{embed_method}_{clf_name}"
                
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{os.path.basename(dataset_path)}_{timestamp}_results.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate comparative visualizations
    if len(embedding_methods) > 0 and len(classifiers) > 0:
        # Extract accuracy results for comparison
        accuracy_data = {}
        for embed_method in results['combined_results']:
            accuracy_data[embed_method] = {}
            for clf_name in results['combined_results'][embed_method]:
                if 'accuracy' in results['combined_results'][embed_method][clf_name]:
                    accuracy_data[embed_method][clf_name] = {
                        'accuracy': results['combined_results'][embed_method][clf_name]['accuracy']
                    }
        
        # Plot comparison
        fig = compare_classifiers(
            accuracy_data, metric='accuracy',
            title=f'Classifier Comparison on {os.path.basename(dataset_path)}'
        )
        fig.savefig(os.path.join(output_dir, f'{os.path.basename(dataset_path)}_comparison.png'))
        plt.close(fig)
    
    print(f"\nEvaluation complete! Results saved to {result_file}")
    return results


def run_embedding_comparison(dataset_path, target_column, embedding_methods, 
                           output_dir='results', random_state=42, 
                           visualization_method='tsne'):
    """
    Compare multiple embedding methods on the same dataset visually.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file
    target_column : str or int
        Name or index of the target column
    embedding_methods : list of str
        List of embedding methods to compare
    output_dir : str, default='results'
        Directory to save results
    random_state : int, default=42
        Random seed for reproducibility
    visualization_method : str, default='tsne'
        Method for visualization: 'tsne', 'umap', or 'pca'
        
    Returns:
    --------
    dict
        Dictionary with embedding results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y = load_dataset(dataset_path, target_column)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.2, random_state=random_state
    )
    
    # Normalize data
    X_train_norm, X_val_norm, X_test_norm = normalize_data(X_train, X_val, X_test, method='standard')
    
    # Prepare results dictionary
    results = {
        'dataset': os.path.basename(dataset_path),
        'embedding_results': {}
    }
    
    # Generate embeddings for all methods
    embeddings = {}
    
    for embed_method in embedding_methods:
        print(f"\nGenerating {embed_method} embedding...")
        
        try:
            train_emb, val_emb, test_emb, embed_time = run_embedding_experiment(
                X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, embed_method
            )
            
            # Store embedding results
            results['embedding_results'][embed_method] = {
                'time': embed_time,
                'shape': train_emb.shape if hasattr(train_emb, 'shape') else None
            }
            
            # Add to embeddings dictionary for comparison
            embeddings[embed_method] = test_emb
            
        except Exception as e:
            print(f"Error with embedding method {embed_method}: {e}")
            results['embedding_results'][embed_method] = {
                'error': str(e)
            }
    
    # Create comparison visualization if we have embeddings
    if embeddings:
        print("\nCreating embedding comparison visualization...")
        fig = plot_embedding_comparison(
            embeddings, y_test, method=visualization_method,
            figsize=(15, 10 * (len(embeddings) + 1) // 2)
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(
            output_dir, 
            f"{os.path.basename(dataset_path)}_{timestamp}_embedding_comparison.png"
        )
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Visualization saved to {viz_path}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{os.path.basename(dataset_path)}_{timestamp}_embedding_results.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison complete! Results saved to {result_file}")
    return results


def run_robustness_evaluation(dataset_path, target_column, embedding_methods, 
                            noise_levels=[0.0, 0.05, 0.1, 0.2], 
                            output_dir='results', random_state=42):
    """
    Evaluate the robustness of embedding methods to noise.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file
    target_column : str or int
        Name or index of the target column
    embedding_methods : list of str
        List of embedding methods to evaluate
    noise_levels : list of float, default=[0.0, 0.05, 0.1, 0.2]
        Levels of Gaussian noise to add
    output_dir : str, default='results'
        Directory to save results
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with robustness results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y = load_dataset(dataset_path, target_column)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.0, random_state=random_state
    )
    
    # Normalize data
    X_train_norm, X_test_norm = normalize_data(X_train, X_test, method='standard')
    
    # Prepare results dictionary
    results = {
        'dataset': os.path.basename(dataset_path),
        'noise_levels': noise_levels,
        'robustness_results': {}
    }
    
    # Use a simple classifier for evaluation
    classifier = 'rf'  # Random Forest - good balance of speed and accuracy
    
    # For each embedding method
    for embed_method in embedding_methods:
        print(f"\nEvaluating robustness of {embed_method} embedding")
        
        results['robustness_results'][embed_method] = {}
        
        # Get clean embedding as baseline
        try:
            train_clean_emb, _, test_clean_emb, _ = run_embedding_experiment(
                X_train_norm, None, X_test_norm, y_train, None, y_test, embed_method
            )
            
            # Test classifier on clean embedding
            experiment_name = f"{os.path.basename(dataset_path)}_{embed_method}_clean"
            _, clean_score, _ = run_classifier_experiment(
                train_clean_emb, None, test_clean_emb, y_train, None, y_test, 
                classifier, experiment_name
            )
            
            results['robustness_results'][embed_method]['clean'] = clean_score
            print(f"  - Clean accuracy: {clean_score:.4f}")
            
            # For each noise level
            for noise_level in noise_levels:
                if noise_level == 0.0:
                    continue  # Skip clean case (already done)
                
                print(f"  - Adding {noise_level:.2f} noise")
                
                # Add noise to test data
                np.random.seed(random_state)
                noise = np.random.normal(0, noise_level, X_test_norm.shape)
                X_test_noisy = X_test_norm + noise
                
                # Get embedding with noise
                _, _, test_noisy_emb, _ = run_embedding_experiment(
                    X_train_norm, None, X_test_noisy, y_train, None, y_test, embed_method
                )
                
                # Test classifier on noisy embedding
                experiment_name = f"{os.path.basename(dataset_path)}_{embed_method}_noise{noise_level}"
                _, noisy_score, _ = run_classifier_experiment(
                    train_clean_emb, None, test_noisy_emb, y_train, None, y_test, 
                    classifier, experiment_name
                )
                
                results['robustness_results'][embed_method][f'noise_{noise_level}'] = noisy_score
                print(f"    - Accuracy with noise: {noisy_score:.4f}")
            
        except Exception as e:
            print(f"Error with embedding method {embed_method}: {e}")
            results['robustness_results'][embed_method]['error'] = str(e)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{os.path.basename(dataset_path)}_{timestamp}_robustness.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot robustness comparison
    if len(embedding_methods) > 0:
        plt.figure(figsize=(12, 8))
        
        # Create plot data
        for embed_method in results['robustness_results']:
            method_results = results['robustness_results'][embed_method]
            
            if 'error' in method_results:
                continue
                
            x_values = [0.0]  # Start with clean
            y_values = [method_results['clean']]
            
            # Add noisy results
            for noise_level in noise_levels:
                if noise_level == 0.0:
                    continue
                
                key = f'noise_{noise_level}'
                if key in method_results:
                    x_values.append(noise_level)
                    y_values.append(method_results[key])
            
            plt.plot(x_values, y_values, marker='o', linewidth=2, label=embed_method)
        
        plt.title(f'Robustness to Noise ({os.path.basename(dataset_path)})')
        plt.xlabel('Noise Level (Standard Deviation)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{os.path.basename(dataset_path)}_robustness.png'))
        plt.close()
    
    print(f"\nRobustness evaluation complete! Results saved to {result_file}")
    return results