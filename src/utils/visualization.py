"""
Visualization utilities for time series embeddings.

This module provides specialized functions for visualizing 
embeddings and model outputs.
"""

import seaborn as sns
import umap
import matplotlib.pyplot as plt
import numpy as np
import time
from ..embeddings import umap_embedding

# Plotting with seaborn (if you prefer more sophisticated plots)
cp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#eb5b42"]
event_colors_dict = {'conflict': 'tomato', 'cyclone': 'cadetblue', 'earthquake': 'mediumseagreen', 'flood': 'dodgerblue'}
#%%
def umap_plt(train_emb, val_emb, test_emb, ny_train, ny_val, ny_test, name):
    """
    Create and save a UMAP projection plot of embeddings.
    
    Parameters:
    -----------
    train_emb : array-like
        Training embeddings
    val_emb : array-like
        Validation embeddings
    test_emb : array-like
        Test embeddings
    ny_train : array-like
        Training labels
    ny_val : array-like
        Validation labels
    ny_test : array-like
        Test labels
    name : str
        Name for the plot title and file
        
    Returns:
    --------
    None
        Saves and displays the plot
    """
    # To get the umap embedding
    train_um, val_um, test_um = umap_embedding(train_emb, val_emb, test_emb)
    embedding = test_um
    
    # Ensure all classes are in the legend by specifying hue_order
    hue_order = sorted(np.unique(ny_test.values))
    plt.figure(figsize=(12, 8))
    
    # Adjusting the text size for the plot
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=ny_test.values.flatten(), 
        palette=event_colors_dict,
        s=50,
        hue_order=hue_order
    )
    
    title_name = 'UMAP Projection of' + ' ' + name
    
    # Set the title with a larger font size
    plt.title(title_name, fontsize=18)
    
    # Adjust the legend text size and place it at the bottom
    plt.legend(
        title='Classes', 
        title_fontsize='13', 
        fontsize='12', 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=10
    )
    
    # Save the figure
    plt.savefig(title_name + ".png", format="png", dpi=600, bbox_inches="tight")
    
    # Display the plot
    plt.show()

def plot_embedding_comparison(embeddings_dict, labels, method='tsne', figsize=(15, 10)):
    """
    Create a grid of embedding visualizations for comparison.
    
    Parameters:
    -----------
    embeddings_dict : dict
        Dictionary with embedding method names as keys and embeddings as values
    labels : array-like
        Labels for color-coding
    method : str, default='tsne'
        Dimensionality reduction method: 'tsne', 'umap', or 'pca'
    figsize : tuple, default=(15, 10)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    n_methods = len(embeddings_dict)
    fig, axes = plt.subplots(
        nrows=(n_methods + 1) // 2, 
        ncols=min(2, n_methods), 
        figsize=figsize
    )
    
    # Flatten axes array for easy indexing
    if n_methods > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Apply dimensionality reduction and plot each embedding
    for i, (name, embedding) in enumerate(embeddings_dict.items()):
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embed_2d = reducer.fit_transform(embedding)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embed_2d = reducer.fit_transform(embedding)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embed_2d = reducer.fit_transform(embedding)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        # Ensure labels are in the right format
        if hasattr(labels, 'values'):
            label_values = labels.values.flatten()
        else:
            label_values = np.array(labels).flatten()
        
        # Plot on the corresponding axis
        sns.scatterplot(
            x=embed_2d[:, 0], 
            y=embed_2d[:, 1], 
            hue=label_values,
            palette=cp,
            s=50,
            ax=axes[i]
        )
        
        axes[i].set_title(f"{name} Embedding", fontsize=14)
        axes[i].set_xlabel(f"{method.upper()} Dimension 1")
        axes[i].set_ylabel(f"{method.upper()} Dimension 2")
        axes[i].legend(title='Classes', title_fontsize='10', fontsize='8')
    
    # Remove any unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    return fig

def plot_training_history(history, metric='accuracy', figsize=(10, 6)):
    """
    Plot training history for a model.
    
    Parameters:
    -----------
    history : dict or History object
        Training history 
    metric : str, default='accuracy'
        Metric to plot
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert from keras History object if needed
    if hasattr(history, 'history'):
        history = history.history
    
    # Determine available metrics
    metrics = []
    val_metrics = []
    
    for key in history.keys():
        if key.endswith(metric) and not key.startswith('val_'):
            metrics.append(key)
        elif key.startswith('val_') and key.endswith(metric):
            val_metrics.append(key)
    
    # Plot each metric
    for m in metrics:
        ax.plot(history[m], label=m)
    
    for vm in val_metrics:
        ax.plot(history[vm], label=vm, linestyle='--')
    
    ax.set_title(f'Training {metric.capitalize()} History', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names=None, figsize=(10, 8), cmap='Blues', normalize=True):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Names of the classes
    figsize : tuple, default=(10, 8)
        Figure size
    cmap : str, default='Blues'
        Colormap
    normalize : bool, default=True
        Whether to normalize the confusion matrix
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd', 
        cmap=cmap,
        square=True,
        cbar=True,
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto',
        ax=ax
    )
    
    # Customize plot
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=16)
    
    plt.tight_layout()
    return fig