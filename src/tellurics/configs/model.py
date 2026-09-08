"""Model configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    """Available model architectures."""

    CNN = "cnn"
    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    NIGHT_PERCEIVER = "perceiver_night"
    TEMPORAL_CONV = "temporal_conv"


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

    # Night-level Perceiver-specific (see models/night.py)
    n_frames_per_series: int = Field(default=73, gt=0, description="Exposures per night (T).")
    num_queries: int = Field(default=16, gt=0, description="Number of learned Perceiver latents (Q).")
    spectral_latent_dim: int = Field(
        default=64, gt=0, description="Dim of the per-exposure spectral code (d)."
    )
    spectral_encoder_dims: list[int] = Field(
        default_factory=lambda: [2048, 512, 128],
        description="Hidden widths of the shared per-exposure spectral MLP (input -> ... -> latent).",
    )
    metadata_dim: int = Field(
        default=20, ge=0, description="Length of the per-night metadata vector fed to the fusion."
    )
    use_time_embed: bool = Field(
        default=True,
        description="Add a learned temporal (exposure-index) embedding to the per-frame codes.",
    )
    predict_spectral_dim: int = Field(
        default=0, ge=0,
        description="Output spectral dim of the telluric decoder. 0 -> use num_wavelength_bins.",
    )

    # Whole-night CNN auto-encoder with temporal Perceiver (see models/temporal_conv.py).
    # These fields are only consumed when architecture == "temporal_conv"; they
    # are ignored by every other architecture.
    x_encoder_channels: list[int] = Field(
        default_factory=lambda: [16, 32, 64, 96, 128],
        description="Channels of the shared strided 1D-CNN applied to each exposure of X.",
    )
    x_encoder_kernel: int = Field(default=7, gt=0)
    x_encoder_stride: int = Field(default=2, gt=0)
    s_encoder_channels: list[int] = Field(
        default_factory=lambda: [16, 32, 64],
        description="Channels of the independent stellar (S) 1D-CNN.",
    )
    s_encoder_kernel: int = Field(default=7, gt=0)
    s_encoder_stride: int = Field(default=2, gt=0)
    encoder_pool_bins: int = Field(
        default=16, gt=0,
        description="Adaptive spectral bins kept per channel before the code read-out.",
    )
    fusion_hidden: int = Field(
        default=0, ge=0,
        description="Fusion MLP width. 0 -> num_queries * spectral_latent_dim.",
    )
    fusion_layers: int = Field(default=2, gt=0)
    decoder_channels: list[int] = Field(
        default_factory=lambda: [64, 48, 32, 24, 16, 8, 4, 1],
        description="Channel schedule of the shared spectral decoder (must end in 1).",
    )
    decoder_kernel: int = Field(default=3, gt=0)
    output_activation: str = Field(
        default="sigmoid",
        description="Decoder output activation: 'sigmoid' (T in [0,1]) or 'none'.",
    )
    metadata_enc_dim: int = Field(
        default=16, gt=0,
        description="Width of the per-exposure metadata code (C), concatenated "
                    "with the spectral code before the temporal Perceiver "
                    "(temporal_conv only).",
    )
    metadata_enc_hidden: int | None = Field(
        default=None, gt=0,
        description="Hidden width of the per-exposure metadata encoder MLP "
                    "(None -> auto; temporal_conv only).",
    )
    param_dim: int = Field(
        default=20, gt=0,
        description="Number of telluric/atmospheric parameters predicted per "
                    "exposure by the param-decoder head (temporal_conv only).",
    )
    param_decoder_hidden: int | None = Field(
        default=None, gt=0,
        description="Hidden width of the per-exposure param-decoder MLP "
                    "(None -> auto; temporal_conv only).",
    )
