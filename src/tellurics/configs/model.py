"""Model configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    """Available model architectures (``ModelRegistry`` ids)."""

    NEURAL_TELLURIC_PREDICTOR = "neural_telluric_predictor"


class ModelConfig(BaseModel):
    """Configuration for the whole-night telluric parameter estimator.

    Every field is consumed by :mod:`tellurics.models.night` (or by the
    spectral head in ``models/decoders/telluric.py``); see the ``night`` module
    docstring for the shape-annotated data flow.
    """

    architecture: ModelArchitecture = ModelArchitecture.NEURAL_TELLURIC_PREDICTOR

    # -- shared dimensions ------------------------------------------------ #
    num_wavelength_bins: int = Field(default=4096, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    num_heads: int = Field(
        default=8, gt=0, description="Heads of the temporal cross-attention."
    )

    # -- night-level estimator (see models/night.py) ---------------------- #
    n_frames_per_series: int = Field(default=73, gt=0, description="Exposures per night (T).")
    num_queries: int = Field(default=16, gt=0, description="Number of learned Perceiver latents (Q).")
    spectral_latent_dim: int = Field(
        default=64, gt=0, description="Dim of the per-exposure spectral code (d)."
    )
    metadata_dim: int = Field(
        default=20, ge=0,
        description="Per-exposure metadata width P (data contract: 3 -> "
                    "pressure, temperature, humidity).",
    )

    # -- shared strided 1D-CNN over each exposure of X -------------------- #
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
    metadata_enc_dim: int = Field(
        default=16, gt=0,
        description="Width of the per-exposure metadata code (C), concatenated "
                    "with the spectral code before the temporal Perceiver "
                    "(neural_telluric_predictor only).",
    )
    metadata_enc_hidden: int | None = Field(
        default=None, gt=0,
        description="Hidden width of the per-exposure metadata encoder MLP "
                    "(None -> auto; neural_telluric_predictor only).",
    )
    time_enc_dim: int = Field(
        default=16, gt=0,
        description="Width (k) of the per-exposure time code produced by the "
                    "TimeEncoder from continuous exposure times (time_hours), "
                    "concatenated with the spectral + metadata codes before "
                    "the temporal Perceiver (neural_telluric_predictor only).",
    )
    param_dim: int = Field(
        default=20, gt=0,
        description="Number of telluric/atmospheric parameters predicted per "
                    "exposure by the param-decoder head (neural_telluric_predictor only).",
    )
    param_decoder_hidden: int | None = Field(
        default=None, gt=0,
        description="Hidden width of the per-exposure param-decoder MLP "
                    "(None -> auto; neural_telluric_predictor only).",
    )
