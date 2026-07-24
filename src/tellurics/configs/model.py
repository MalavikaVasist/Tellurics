"""Model configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    """Available model architectures."""

    CNN = "cnn"
    TRANSFORMER = "transformer"
    MAMBA = "mamba"


class FusionMethod(str, Enum):
    """Methods for fusing atmospheric parameters with spectral features."""

    CONCATENATION = "concatenation"
    FILM = "film"
    CROSS_ATTENTION = "cross_attention"


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    architecture: ModelArchitecture = ModelArchitecture.CNN
    num_wavelength_bins: int = Field(default=4096, gt=0)
    num_atm_parameters: int = Field(default=6, gt=0)
    fusion_method: FusionMethod = FusionMethod.FILM
    hidden_dim: int = Field(default=256, gt=0)
    num_layers: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_residual: bool = True

    # Transformer-specific
    num_heads: int = Field(default=8, gt=0)
    ff_dim: int = Field(default=512, gt=0)

    # CNN-specific
    kernel_size: int = Field(default=7, gt=0)
    num_channels: list[int] = Field(default_factory=lambda: [64, 128, 256, 128, 64])

    # Mamba-specific
    state_dim: int = Field(default=16, gt=0)
    expand_factor: int = Field(default=2, gt=0)
