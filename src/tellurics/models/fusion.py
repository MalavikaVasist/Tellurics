"""Fusion modules for combining spectral features with atmospheric parameters."""

import torch
import torch.nn as nn


class AtmosphericEmbedding(nn.Module):
    """MLP embedding for atmospheric parameters."""

    def __init__(self, num_params: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_params, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, atm_params: torch.Tensor) -> torch.Tensor:
        """Embed atmospheric parameters.

        Args:
            atm_params: Atmospheric parameter tensor. Shape: (B, num_params).

        Returns:
            Embedded representation. Shape: (B, embed_dim).
        """
        return self.mlp(atm_params)


class ConcatenationFusion(nn.Module):
    """Fuse by concatenating atmospheric embedding to each spectral position."""

    def __init__(self, spectral_dim: int, atm_embed_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(spectral_dim + atm_embed_dim, output_dim)

    def forward(self, spectral: torch.Tensor, atm_embed: torch.Tensor) -> torch.Tensor:
        """Fuse spectral features with atmospheric embedding.

        Args:
            spectral: Spectral features. Shape: (B, N, C_spectral).
            atm_embed: Atmospheric embedding. Shape: (B, C_atm).

        Returns:
            Fused features. Shape: (B, N, output_dim).
        """
        # Expand atmospheric embedding across spectral dimension
        atm_expanded = atm_embed.unsqueeze(1).expand(-1, spectral.size(1), -1)
        fused = torch.cat([spectral, atm_expanded], dim=-1)
        return self.projection(fused)


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning.

    Modulates spectral features using learned scale and shift from atmospheric parameters.
    """

    def __init__(self, atm_embed_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.scale_net = nn.Linear(atm_embed_dim, feature_dim)
        self.shift_net = nn.Linear(atm_embed_dim, feature_dim)

    def forward(self, spectral: torch.Tensor, atm_embed: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            spectral: Spectral features. Shape: (B, N, C) or (B, C, N).
            atm_embed: Atmospheric embedding. Shape: (B, C_atm).

        Returns:
            Modulated features, same shape as spectral.
        """
        gamma = self.scale_net(atm_embed)  # (B, C)
        beta = self.shift_net(atm_embed)  # (B, C)

        if spectral.dim() == 3:
            # Determine if channel-last (B, N, C) or channel-first (B, C, N)
            if spectral.size(-1) == gamma.size(-1):
                # (B, N, C) format
                gamma = gamma.unsqueeze(1)  # (B, 1, C)
                beta = beta.unsqueeze(1)  # (B, 1, C)
            else:
                # (B, C, N) format
                gamma = gamma.unsqueeze(-1)  # (B, C, 1)
                beta = beta.unsqueeze(-1)  # (B, C, 1)

        return gamma * spectral + beta


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between spectral features and atmospheric embedding."""

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, spectral: torch.Tensor, atm_embed: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention fusion.

        Args:
            spectral: Spectral features. Shape: (B, N, C).
            atm_embed: Atmospheric embedding. Shape: (B, C). Will be unsqueezed to (B, 1, C).

        Returns:
            Attended features. Shape: (B, N, C).
        """
        # Treat atmospheric embedding as a single key-value token
        kv = atm_embed.unsqueeze(1)  # (B, 1, C)
        attended, _ = self.cross_attn(query=spectral, key=kv, value=kv)
        return self.norm(spectral + attended)
