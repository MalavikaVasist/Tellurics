"""CNN-based spectral regressor."""

import torch
import torch.nn as nn

from tellurics.configs.model import FusionMethod, ModelConfig
from tellurics.models.base import BaseSpectralModel
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


class ResidualBlock(nn.Module):
    """1D residual convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Shape: (B, C, N) -> (B, C_out, N)."""
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + residual)


@ModelRegistry.register("cnn")
class CNNRegressor(BaseSpectralModel):
    """1D CNN regressor for telluric prediction.

    Architecture:
        1. Input projection from 1 channel to first hidden channel
        2. Stack of residual convolutional blocks
        3. FiLM/concat/cross-attention fusion with atmospheric embedding
        4. Output projection to telluric spectrum with sigmoid activation
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        channels = config.num_channels
        kernel_size = config.kernel_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )

        # Encoder blocks
        encoder_blocks = []
        for i in range(len(channels) - 1):
            encoder_blocks.append(ResidualBlock(channels[i], channels[i + 1], kernel_size))
        self.encoder = nn.Sequential(*encoder_blocks)

        # Project to hidden_dim for fusion
        last_channel = channels[-1]
        self.pre_fusion_proj = nn.Conv1d(last_channel, config.hidden_dim, 1)

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim // 2, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(config.hidden_dim // 2, 1, 1),
            nn.Sigmoid(),  # Physical constraint: T_tell in [0, 1]
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
        # Add channel dimension: (B, N) -> (B, 1, N)
        x = spectrum.unsqueeze(1)

        # Encode spectral features
        x = self.input_proj(x)  # (B, C0, N)
        x = self.encoder(x)  # (B, C_last, N)
        x = self.pre_fusion_proj(x)  # (B, hidden_dim, N)

        # Embed atmospheric parameters
        atm_embed = self.atm_embedding(atm_params)  # (B, hidden_dim)

        # Fuse atmospheric information
        if self.config.fusion_method == FusionMethod.CROSS_ATTENTION:
            # Cross-attention expects (B, N, C)
            x = x.permute(0, 2, 1)  # (B, N, hidden_dim)
            x = self.fusion(x, atm_embed)  # (B, N, hidden_dim)
            x = x.permute(0, 2, 1)  # (B, hidden_dim, N)
        else:
            # FiLM and concatenation work with (B, C, N)
            x = self.fusion(x, atm_embed)  # (B, hidden_dim, N)

        # Output projection
        telluric = self.output_head(x).squeeze(1)  # (B, N)

        return ModelOutput(telluric=telluric)
