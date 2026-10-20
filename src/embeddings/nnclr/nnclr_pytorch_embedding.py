"""
PyTorch NNCLR Embedding Module

A simplified PyTorch implementation of Nearest Neighbor Contrastive Learning of Representations (NNCLR)
for generating high-quality embeddings from time series data.

Supports three encoder types: CNN, LSTM, and Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Import the modular encoders
from .encoders.cnn_encoder import CNN_Encoder
from .encoders.lstm_encoder import LSTM_Encoder
from .encoders.transformer_encoder import Transformer_Encoder


class Augmenter(nn.Module):
    """Simple augmentation module for time series data."""

    def __init__(self, noise_factor=0.1, dropout_rate=0.1):
        super(Augmenter, self).__init__()
        self.noise_factor = noise_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, training=True):
        if training:
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_factor
            x_aug = x + noise

            # Apply dropout (note: nn.Dropout is only active when the module is in train mode)
            x_aug = self.dropout(x_aug)

            return x_aug
        return x


class NNCLR_Embedding(nn.Module):
    """
    NNCLR model for generating embeddings from time series data.
    """

    def __init__(
        self,
        input_shape,
        embedding_dim=128,
        encoder_type='cnn',
        temperature=0.1,
        queue_size=5000,
        n_classes=None
    ):
        super(NNCLR_Embedding, self).__init__()

        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.n_classes = n_classes

        sequence_length, n_features = input_shape

        # Create encoder based on type
        if encoder_type == 'cnn':
            self.encoder = CNN_Encoder(
                input_size=n_features,
                embedding_dim=embedding_dim,
                sequence_length=sequence_length
            )
        elif encoder_type == 'lstm':
            self.encoder = LSTM_Encoder(
                input_size=n_features,
                embedding_dim=embedding_dim,
                hidden_size=128
            )
        elif encoder_type == 'transformer':
            self.encoder = Transformer_Encoder(
                input_size=n_features,
                embedding_dim=embedding_dim,
                sequence_length=sequence_length,
                patch_size=8,
                num_heads=8,
                num_layers=6,
                dim_feedforward=256,
                dropout=0.1
            )
        else:
            raise ValueError("encoder_type must be 'cnn', 'lstm', or 'transformer'")

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Linear probe for classification (if needed)
        if n_classes is not None:
            self.linear_probe = nn.Linear(embedding_dim, n_classes)

        # Augmenter
        self.augmenter = Augmenter()

        # Feature queue for nearest neighbor search
        self.register_buffer(
            'feature_queue',
            nn.functional.normalize(torch.randn(queue_size, embedding_dim).float(), dim=1)
        )

    def nearest_neighbor(self, projections):
        """Find nearest neighbors in the feature queue."""
        # Normalize projections
        projections = nn.functional.normalize(projections, dim=1)

        # Compute similarities with queue
        similarities = torch.matmul(projections, self.feature_queue.t())

        # Get nearest neighbors
        _, nn_indices = similarities.max(dim=1)
        nn_projections = self.feature_queue[nn_indices]

        # Return projections + stop_gradient(nn_projections - projections)
        return projections + (nn_projections - projections).detach()

    def contrastive_loss(self, z1, z2):
        """Compute contrastive loss with nearest neighbors."""
        batch_size = z1.size(0)

        # Normalize projections
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Get nearest neighbors
        nn_z1 = self.nearest_neighbor(z1)
        nn_z2 = self.nearest_neighbor(z2)

        # Compute similarities
        sim_1_2_1 = torch.matmul(nn_z1, z2.t()) / self.temperature
        sim_1_2_2 = torch.matmul(z2, nn_z1.t()) / self.temperature
        sim_2_1_1 = torch.matmul(nn_z2, z1.t()) / self.temperature
        sim_2_1_2 = torch.matmul(z1, nn_z2.t()) / self.temperature

        # Create labels
        labels = torch.arange(batch_size).to(z1.device)

        # Compute cross-entropy loss
        loss = (
            nn.functional.cross_entropy(sim_1_2_1, labels) +
            nn.functional.cross_entropy(sim_1_2_2, labels) +
            nn.functional.cross_entropy(sim_2_1_1, labels) +
            nn.functional.cross_entropy(sim_2_1_2, labels)
        ) / 4

        # Update feature queue
        self._update_queue(z1)

        return loss

    @torch.no_grad()
    def _update_queue(self, features):
        """Update the feature queue with new features."""
        batch_size = features.size(0)
        queue_size = self.feature_queue.size(0)

        # Normalize features
        features = nn.functional.normalize(features.detach(), dim=1)

        # Keep only the newest `queue_size` features if the batch is larger than the queue
        if batch_size >= queue_size:
            self.feature_queue = features[:queue_size].clone()
            return

        # Update queue
        self.feature_queue = torch.cat([
            features,
            self.feature_queue[:-batch_size]
        ], dim=0)

    def get_embeddings(self, x):
        """Get embeddings from the encoder."""
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 2:
                x = x.unsqueeze(-1)

            # All encoders expect (batch, seq_len, features)
            embeddings = self.encoder(x)
            return embeddings

    def forward(self, x1, x2=None):
        """Forward pass for training or inference."""
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(-1)

        if self.training and x2 is not None:
            # Training mode - contrastive learning
            if len(x2.shape) == 2:
                x2 = x2.unsqueeze(-1)

            # Get embeddings
            h1 = self.encoder(x1)
            h2 = self.encoder(x2)

            # Get projections
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

            # Compute contrastive loss
            loss = self.contrastive_loss(z1, z2)

            return loss, h1, h2
        else:
            # Inference mode - just return embeddings
            return self.get_embeddings(x1)


def _embed_in_batches(model, x_tensor, batch_size):
    """Run the model over a tensor in batches and return a single CPU tensor."""
    outputs = []
    for i in range(0, len(x_tensor), batch_size):
        outputs.append(model(x_tensor[i:i + batch_size]).cpu())
    return torch.cat(outputs, dim=0)


def nnclr_embedding_base(
    x_train, x_val, x_test, y_train, y_val, y_test,
    embedding_dim=128, encoder_type='cnn', n_classes=None,
    batch_size=128, pretrain_epochs=20, learning_rate=1e-3,
    device=None, return_timing=False
):
    """
    Base function to generate embeddings using NNCLR.

    NNCLR is self-supervised: the model is pretrained on x_train only (labels are
    not used), then train/val/test are embedded with the trained encoder.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to numpy if pandas DataFrame / Series
    if hasattr(x_train, 'values'):
        x_train = x_train.values
    if hasattr(x_val, 'values'):
        x_val = x_val.values
    if hasattr(x_test, 'values'):
        x_test = x_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_val, 'values'):
        y_val = y_val.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)

    # Reshape if needed
    y_train = y_train.reshape(-1) if len(y_train.shape) > 1 else y_train
    y_val = y_val.reshape(-1) if len(y_val.shape) > 1 else y_val
    y_test = y_test.reshape(-1) if len(y_test.shape) > 1 else y_test

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    for name, arr in [('x_train', x_train), ('x_val', x_val), ('x_test', x_test)]:
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaNs; impute or drop them first")

    # Determine input shape -> (sequence_length, n_features)
    input_shape = tuple(x_train.shape[1:])
    if len(input_shape) == 1:
        input_shape = input_shape + (1,)

    # Build model
    model = NNCLR_Embedding(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        encoder_type=encoder_type,
        n_classes=n_classes
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---- Pretraining (self-supervised) ----
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(x_train) > batch_size  # avoid tiny last batches (bad for contrastive loss / BN)
    )

    start_train = time.time()
    model.train()
    for epoch in range(pretrain_epochs):
        total_loss, n_batches = 0.0, 0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{pretrain_epochs}", leave=False):
            if len(batch) < 2:
                continue
            batch = batch.to(device)

            # Two augmented views of the same batch
            x1 = model.augmenter(batch)
            x2 = model.augmenter(batch)

            loss, _, _ = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch + 1}/{pretrain_epochs} - loss: {total_loss / max(n_batches, 1):.4f}")
    train_time = time.time() - start_train

    # ---- Embedding extraction ----
    start_inference = time.time()
    model.eval()
    print("Generating embeddings...")

    with torch.no_grad():
        train_embeddings = _embed_in_batches(model, torch.from_numpy(x_train).to(device), batch_size)
        val_embeddings = _embed_in_batches(model, torch.from_numpy(x_val).to(device), batch_size)
        test_embeddings = _embed_in_batches(model, torch.from_numpy(x_test).to(device), batch_size)
    inference_time = time.time() - start_inference

    # Convert to pandas DataFrames
    train_emb_df = pd.DataFrame(train_embeddings.numpy())
    val_emb_df = pd.DataFrame(val_embeddings.numpy())
    test_emb_df = pd.DataFrame(test_embeddings.numpy())

    if return_timing:
        print(f"NNCLR {encoder_type.upper()} Training time: {train_time:.4f} seconds")
        print(f"NNCLR {encoder_type.upper()} Inference time: {inference_time:.4f} seconds")

        return train_emb_df, val_emb_df, test_emb_df, train_time, inference_time

    return train_emb_df, val_emb_df, test_emb_df


# The three specific functions you requested
def nnclr_cnn_embedding_with_timing(x_train_df, x_val_df, x_test_df,
                                    y_train_df, y_val_df, y_test_df,
                                    width, n_classes):
    """CNN-based NNCLR embedding with timing - matches TensorFlow API."""
    return nnclr_embedding_base(
        x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df,
        embedding_dim=width, encoder_type='cnn', n_classes=n_classes,
        return_timing=True
    )


def nnclr_lstm_embedding_with_timing(x_train_df, x_val_df, x_test_df,
                                     y_train_df, y_val_df, y_test_df,
                                     width, n_classes):
    """LSTM-based NNCLR embedding with timing - matches TensorFlow API."""
    return nnclr_embedding_base(
        x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df,
        embedding_dim=width, encoder_type='lstm', n_classes=n_classes,
        return_timing=True
    )


def nnclr_transformer_embedding_with_timing(x_train_df, x_val_df, x_test_df,
                                            y_train_df, y_val_df, y_test_df,
                                            width, n_classes):
    """Transformer-based NNCLR embedding with timing."""
    return nnclr_embedding_base(
        x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df,
        embedding_dim=width, encoder_type='transformer', n_classes=n_classes,
        return_timing=True
    )