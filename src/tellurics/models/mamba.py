"""Temporal State Space Model (Mamba-inspired) for spectral processing.

This is a simplified S4-style implementation that does not require the mamba-ssm package.
If mamba-ssm is available, it can be extended to use the full Mamba block.
"""

import torch
import torch.nn as nn

from tellurics.configs.model import FusionMethod, ModelConfig
from tellurics.models.base import BaseSpectralModel
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


class SimpleS4Block(nn.Module):
    """Simplified state-space block using 1D depthwise convolutions.

    This approximates S4 behavior using causal convolutions with
    gating, providing long-range sequence modeling without attention.
    """

    def __init__(self, dim: int, state_dim: int, expand_factor: int = 2) -> None:
        super().__init__()
        inner_dim = dim * expand_factor

        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=4, padding=3, groups=inner_dim
        )
        self.out_proj = nn.Linear(inner_dim, dim)
        self.norm = nn.LayerNorm(dim)

        # State-space inspired: learnable decay and mixing
        self.dt_proj = nn.Linear(inner_dim, inner_dim)
        self.A = nn.Parameter(torch.randn(inner_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.ones(inner_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features. Shape: (B, N, D).

        Returns:
            Processed features. Shape: (B, N, D).
        """
        residual = x
        x = self.norm(x)

        # Split into gate and value paths
        xz = self.in_proj(x)  # (B, N, 2*inner_dim)
        x_val, z = xz.chunk(2, dim=-1)  # Each (B, N, inner_dim)

        # Causal convolution on value path
        x_val = x_val.permute(0, 2, 1)  # (B, inner_dim, N)
        x_val = self.conv(x_val)[:, :, : x.size(1)]  # Causal trim
        x_val = x_val.permute(0, 2, 1)  # (B, N, inner_dim)

        # Gating with SiLU
        x_val = x_val * torch.sigmoid(x_val)  # SiLU activation
        z = torch.sigmoid(z)  # Gate

        # Skip connection with D parameter
        y = x_val * self.D + x_val
        y = y * z

        # Output projection
        y = self.out_proj(y)
        return y + residual


@ModelRegistry.register("mamba")
class MambaSpectralModel(BaseSpectralModel):
    """State-space model for spectral telluric prediction.

    Uses simplified S4-style blocks for efficient long-range
    sequence modeling of spectral data.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        # Input projection
        self.input_proj = nn.Linear(1, config.hidden_dim)

        # Stack of S4 blocks
        self.blocks = nn.ModuleList([
            SimpleS4Block(
                dim=config.hidden_dim,
                state_dim=config.state_dim,
                expand_factor=config.expand_factor,
            )
            for _ in range(config.num_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

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
        # (B, N) -> (B, N, 1) -> (B, N, hidden_dim)
        x = self.input_proj(spectrum.unsqueeze(-1))

        # Embed atmospheric parameters
        atm_embed = self.atm_embedding(atm_params)  # (B, hidden_dim)

        # Fuse atmospheric information
        if self.config.fusion_method == FusionMethod.CROSS_ATTENTION:
            x = self.fusion(x, atm_embed)
        else:
            x = self.fusion(x, atm_embed)

        # Process through S4 blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        telluric = self.output_head(x).squeeze(-1)  # (B, N)

        return ModelOutput(telluric=telluric)
