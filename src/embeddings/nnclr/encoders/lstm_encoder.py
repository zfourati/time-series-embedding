"""
LSTM Encoder for NNCLR Time Series Embedding
"""


import torch

import torch._dynamo  # force torch's heavy lazy imports to load before TF/grpc/protobuf
import torch.nn as nn


class LSTM_Encoder(nn.Module):
    """
    An LSTM-based encoder for time series embedding.

    Args:
        input_size (int): The number of features or channels in the input time series.
        embedding_dim (int): The desired dimension of the output embedding.
        hidden_size (int): The number of features in the LSTM's hidden state.
                           Defaults to 128.
    """
    def __init__(self, 
                 input_size: int, 
                 embedding_dim: int, 
                 hidden_size: int = 128):
        
        super(LSTM_Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # --- Best-practice Architectural Choices ---
        self.num_layers = 2
        self.bidirectional = True
        dropout_rate = 0.3

        # --- LSTM Layer ---
        # A 2-layer bidirectional LSTM is a robust choice.
        # - 2 layers help learn hierarchical temporal features.
        # - Bidirectional captures context from both past and future.
        # - `dropout_rate` is applied between LSTM layers for regularization.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,  # Input shape: (batch, seq_len, features)
            dropout=dropout_rate if self.num_layers > 1 else 0
        )

        # --- Projection Head ---
        # This projects the LSTM's summary of the sequence to the final embedding space.
        # The input size is `hidden_size * 2` because the LSTM is bidirectional.
        rnn_output_size = hidden_size * 2
        
        self.projection_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(rnn_output_size, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): The input batch of time series segments.
                              Shape: (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: The embedding vector for each time series in the batch.
                          Shape: (batch_size, embedding_dim)
        """
        # x shape: (batch_size, sequence_length, input_size)
        
        # The LSTM returns:
        # - outputs: hidden states for each time step.
        # - (h_n, c_n): final hidden and cell states for each layer.
        _, (h_n, _) = self.lstm(x)
        # `h_n` shape: (num_layers * 2, batch_size, hidden_size)

        # We concatenate the final hidden states from the last layer:
        # - The forward direction's final state is h_n[-2, :, :]
        # - The backward direction's final state is h_n[-1, :, :]
        last_layer_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # `last_layer_hidden_state` shape: (batch_size, hidden_size * 2)

        # Project the concatenated hidden states to the final embedding dimension
        embedding = self.projection_head(last_layer_hidden_state)
        # `embedding` shape: (batch_size, embedding_dim)
        
        return embedding


# Example usage and testing
if __name__ == "__main__":
    # Test parameters - UniMiB-SHAR dataset properties
    BATCH_SIZE = 32
    SEQ_LENGTH = 151
    NUM_CHANNELS = 3
    EMBEDDING_DIM = 256

    # Instantiate the LSTM encoder
    encoder = LSTM_Encoder(
        input_size=NUM_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=128  # This is a good default to start with
    )

    print("LSTM Encoder Architecture:")
    print(encoder)

    # Create a dummy input tensor
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_CHANNELS)

    # Get the embeddings
    embeddings = encoder(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output embedding shape: {embeddings.shape}")
    
    # Verify output shape
    assert embeddings.shape == (BATCH_SIZE, EMBEDDING_DIM), f"Expected shape {(BATCH_SIZE, EMBEDDING_DIM)}, got {embeddings.shape}"
    print("✅ LSTM Encoder test passed!")