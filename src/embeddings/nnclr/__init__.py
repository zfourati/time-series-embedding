"""
Nearest-Neighbor Contrastive Learning of Representations (NNCLR) implementations.

This module provides implementations of NNCLR for time series embedding,
including CNN and LSTM based architectures.
"""

"""
Nearest-Neighbor Contrastive Learning of Representations (NNCLR) implementations.

This module provides implementations of NNCLR for time series embedding,
including CNN, LSTM and Transformer based architectures.
"""

from .nnclr_pytorch_embedding import (
    NNCLR_Embedding as NNCLR,
    nnclr_cnn_embedding_with_timing as nnclr_cnn_embedding,
    nnclr_lstm_embedding_with_timing as nnclr_lstm_embedding,
    nnclr_transformer_embedding_with_timing as nnclr_transformer_embedding,
)

__all__ = [
    'NNCLR',
    'nnclr_cnn_embedding',
    'nnclr_lstm_embedding',
    'nnclr_transformer_embedding',
]