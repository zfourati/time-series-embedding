"""
Time Series Embedding Methods.

This package provides various embedding techniques for time series data,
including statistical, transformation-based, topological, and neural methods.
"""

from .time_series_embeddings import (
    pca_embedding,
    lle_embedding,
    umap_embedding,
    wavelet_embedding,
    fft_embedding,
    AE_embedding,
    graph_embedding,
    tda_embedding
)

from .nnclr import (
    nnclr_cnn_embedding,
    nnclr_lstm_embedding
)

from .foundation_models import (mantis_embedding , chronos_embedding
                                ,tivit_embedding,concat_embeddings, ts2vec_embedding)

__all__ = [
    'pca_embedding',
    'lle_embedding',
    'umap_embedding',
    'wavelet_embedding',
    'fft_embedding',
    'AE_embedding',
    'graph_embedding',
    'tda_embedding',
    'nnclr_cnn_embedding',
    'nnclr_lstm_embedding',
    'mantis_embedding',
    'chronos_embedding',
    'tivit_embedding',
    'concat_embeddings',
    'ts2vec_embedding'
]