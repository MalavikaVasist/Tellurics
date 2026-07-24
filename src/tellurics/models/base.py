"""Base class for spectral models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from tellurics.configs.model import FusionMethod, ModelConfig
from tellurics.models.fusion import (
    AtmosphericEmbedding,
    ConcatenationFusion,
    CrossAttentionFusion,
    FiLMFusion,
)
from tellurics.models.output import ModelOutput


class BaseSpectralModel(nn.Module, ABC):
    """Abstract base class for telluric prediction models.

    All model architectures must inherit from this class and implement
    the forward method returning a ModelOutput object.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.atm_embedding = AtmosphericEmbedding(
            num_params=config.num_atm_parameters,
            embed_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self._build_fusion(config)

    def _build_fusion(self, config: ModelConfig) -> None:
        """Build the fusion module based on config."""
        match config.fusion_method:
            case FusionMethod.CONCATENATION:
                self.fusion = ConcatenationFusion(
                    spectral_dim=config.hidden_dim,
                    atm_embed_dim=config.hidden_dim,
                    output_dim=config.hidden_dim,
                )
            case FusionMethod.FILM:
                self.fusion = FiLMFusion(
                    atm_embed_dim=config.hidden_dim,
                    feature_dim=config.hidden_dim,
                )
            case FusionMethod.CROSS_ATTENTION:
                self.fusion = CrossAttentionFusion(
                    feature_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )

    @abstractmethod
    def forward(
        self,
        spectrum: torch.Tensor,
        atm_params: torch.Tensor,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            spectrum: Normalized spectrum R = X/S. Shape: (B, N_wavelength).
            atm_params: Atmospheric parameters. Shape: (B, N_params).

        Returns:
            ModelOutput with at minimum the telluric prediction.
        """
        ...
