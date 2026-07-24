"""Transformer-based spectral encoder."""

import math

import torch
import torch.nn as nn

from tellurics.configs.model import FusionMethod, ModelConfig
from tellurics.models.base import BaseSpectralModel
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for spectral sequences."""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. Shape: (B, N, D) -> (B, N, D)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SpectralPatchEmbedding(nn.Module):
    """Embed spectral data into patches for transformer processing."""

    def __init__(self, patch_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert spectrum to patch embeddings.

        Args:
            x: Input spectrum. Shape: (B, N).

        Returns:
            Patch embeddings. Shape: (B, N // patch_size, hidden_dim).
        """
        batch_size, seq_len = x.shape
        # Pad if necessary
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = nn.functional.pad(x, (0, pad_len))

        # Reshape into patches
        x = x.view(batch_size, -1, self.patch_size)  # (B, num_patches, patch_size)
        return self.projection(x)  # (B, num_patches, hidden_dim)


@ModelRegistry.register("transformer")
class TransformerEncoder(BaseSpectralModel):
    """Transformer encoder for telluric prediction.

    Uses patch embedding to reduce sequence length, followed by
    transformer encoder layers with atmospheric parameter fusion.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.patch_size = 16  # Reduces 4096 -> 256 tokens
        hidden_dim = config.hidden_dim

        # Patch embedding
        self.patch_embed = SpectralPatchEmbedding(self.patch_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=config.num_wavelength_bins // self.patch_size + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection: from patch space back to wavelength space
        num_patches = config.num_wavelength_bins // self.patch_size
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.patch_size),
        )
        self.output_norm = nn.LayerNorm(num_patches * self.patch_size)
        self.output_activation = nn.Sigmoid()

    def forward(
        self,
        spectrum: torch.Tensor,
        atm_params: torch.Tensor,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            spectrum: Normalized spectrum R = X/S. Shape: (B, N).
            atm_params: Atmospheric parameters. Shape: (B, N_params).

        Returns:
            ModelOutput with telluric prediction.
        """
        batch_size = spectrum.size(0)
        n_wavelength = spectrum.size(1)

        # Patch embedding
        x = self.patch_embed(spectrum)  # (B, num_patches, hidden_dim)
        x = self.pos_encoding(x)

        # Embed atmospheric parameters
        atm_embed = self.atm_embedding(atm_params)  # (B, hidden_dim)

        # Fuse atmospheric information
        if self.config.fusion_method == FusionMethod.CROSS_ATTENTION:
            x = self.fusion(x, atm_embed)
        elif self.config.fusion_method == FusionMethod.FILM:
            x = self.fusion(x, atm_embed)
        else:
            x = self.fusion(x, atm_embed)

        # Transformer encoding
        x = self.transformer(x)  # (B, num_patches, hidden_dim)

        # Project back to wavelength space
        x = self.output_proj(x)  # (B, num_patches, patch_size)
        x = x.reshape(batch_size, -1)  # (B, num_patches * patch_size)

        # Trim to original wavelength size if padded
        x = x[:, :n_wavelength]
        x = self.output_norm(x)
        telluric = self.output_activation(x)  # (B, N)

        return ModelOutput(telluric=telluric)
