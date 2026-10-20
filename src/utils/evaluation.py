"""
Evaluation and visualization utilities for time series embeddings.

This module provides functions for evaluating the quality of embeddings
and visualizing them using various techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
import umap
import time
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def evaluate_embeddings(embeddings, original_data, method='correlation'):
    """
    Evaluate how well the embeddings preserve the structure of the original data.
    
    Parameters:
    -----------
    embeddings : array-like
        Embedded representations
    original_data : array-like
        Original time series data
    method : str, default='correlation'
        Method to evaluate embeddings:
        - 'correlation': Correlation between distance matrices
        - 'reconstruction': Reconstruction error (if applicable)
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    if method == 'correlation':
        # Flatten time series if needed
        if original_data.ndim > 2:
            orig_data_flat = original_data.reshape(original_data.shape[0], -1)
        else:
            orig_data_flat = original_data
            
        # Compute distance matrices
        orig_dist = squareform(pdist(orig_data_flat))
        embed_dist = squareform(pdist(embeddings))
        
        # Compute correlation
        r, p_value = pearsonr(orig_dist.flatten(), embed_dist.flatten())
        
        metrics['correlation'] = r
        metrics['p_value'] = p_value
        
    elif method == 'reconstruction':
        # This assumes there's a way to reconstruct the original data from embeddings
        # Implementation would depend on the specific embedding method
        pass
    
    return metrics


def visualize_embeddings(embeddings, colors, labels=None, method='tsne', figsize=(10, 8), 
                        title='Embedding Visualization', filename=None):
    """
    Visualize embeddings in 2D.
    
    Parameters:
    -----------
    embeddings : array-like
        Embedded representations
    labels : array-like, optional
        Labels for color-coding points
    method : str, default='tsne'
        Visualization method: 'tsne', 'umap', or 'pca'
    figsize : tuple, default=(10, 8)
        Figure size
    title : str, default='Embedding Visualization'
        Plot title
    filename : str, optional
        If provided, save the figure to this file
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Reduce dimensions to 2D for visualization if needed
    if embeddings.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embed_2d = reducer.fit_transform(embeddings)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embed_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embed_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
    else:
        embed_2d = embeddings
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with labels if provided
    if labels is not None:
        unique_labels = np.unique(labels)
        #colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1], 
                      color=colors[label], label=f'Class {label}',
                      alpha=0.7, edgecolors='w', linewidths=0.5)
        
        ax.legend()
    else:
        ax.scatter(embed_2d[:, 0], embed_2d[:, 1], alpha=0.7, 
                  edgecolors='w', linewidths=0.5)
    
    ax.set_title(title)
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def compare_classifiers(results_dict, metric='accuracy', figsize=(12, 8), 
                      title='Classifier Comparison', filename=None):
    """
    Compare classifier performance across different embedding methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with structure:
        {
            'embedding_method1': {
                'classifier1': {'accuracy': 0.8, 'precision': 0.7, ...},
                'classifier2': {...}
            },
            'embedding_method2': {...}
        }
    metric : str, default='accuracy'
        Metric to use for comparison
    figsize : tuple, default=(12, 8)
        Figure size
    title : str, default='Classifier Comparison'
        Plot title
    filename : str, optional
        If provided, save the figure to this file
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Extract data for plotting
    embedding_methods = list(results_dict.keys())
    classifiers = set()
    
    for embed_results in results_dict.values():
        classifiers.update(embed_results.keys())
    
    classifiers = list(classifiers)
    
    # Create DataFrame
    data = []
    for embed_method in embedding_methods:
        for classifier in classifiers:
            if classifier in results_dict[embed_method]:
                value = results_dict[embed_method][classifier].get(metric, np.nan)
                data.append({
                    'Embedding Method': embed_method,
                    'Classifier': classifier,
                    metric.capitalize(): value
                })
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot as grouped bar chart
    sns.barplot(x='Embedding Method', y=metric.capitalize(), hue='Classifier', data=df, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Embedding Method')
    ax.set_ylabel(metric.capitalize())
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Classifier')
    
    # Rotate x-axis labels if there are many embedding methods
    if len(embedding_methods) > 5:
        plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def generate_classification_report(y_true, y_pred, labels=None):
    """
    Generate a comprehensive classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : array-like, optional
        List of labels to include in the report
        
    Returns:
    --------
    dict
        Dictionary with classification metrics
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 score
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create report dictionary
    report = {
        'accuracy': accuracy,
        'class_metrics': {}
    }
    
    # Add per-class metrics
    unique_labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    for i, label in enumerate(unique_labels):
        report['class_metrics'][label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    # Add confusion matrix
    report['confusion_matrix'] = cm
    
    return report


def measure_embedding_time(embedding_func, data, n_runs=5, **kwargs):
    """
    Measure the time it takes to compute embeddings.
    
    Parameters:
    -----------
    embedding_func : callable
        Function that computes embeddings
    data : array-like
        Data to embed
    n_runs : int, default=5
        Number of runs to average over
    **kwargs : dict
        Additional arguments to pass to the embedding function
        
    Returns:
    --------
    dict
        Dictionary with timing statistics
    """
    times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        embedding_func(data, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'times': times
    }