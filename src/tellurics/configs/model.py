"""Model configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    """Available model architectures (``ModelRegistry`` ids)."""

    NEURAL_TELLURIC_PREDICTOR = "neural_telluric_predictor"


class ModelConfig(BaseModel):
    """Configuration for the whole-night telluric parameter estimator.

    No field carries a default: every value is a design or data decision and
    must be stated by the manifest (see ``experiments/``).  All fields are
    consumed by :mod:`tellurics.models.night`; see that module's docstring for
    the shape-annotated data flow.
    """

    architecture: ModelArchitecture

    # -- problem dimensions (must match the DataModule contract) ---------- #
    num_wavelength_bins: int = Field(gt=0)
    n_frames_per_series: int = Field(gt=0, description="Exposures per night (T).")
    metadata_dim: int = Field(
        gt=0,
        description="Per-exposure metadata width P (data contract: 3 -> "
                    "pressure, temperature, humidity).",
    )
    param_dim: int = Field(
        gt=0,
        description="Number of telluric/atmospheric parameters per exposure "
                    "(data contract: 16 = len(TARGET_COLUMNS)).",
    )

    # -- architecture geometry -------------------------------------------- #
    num_queries: int = Field(gt=0, description="Number of learned Perceiver latents (Q).")
    spectral_latent_dim: int = Field(
        gt=0, description="Dim of the per-exposure spectral code (d)."
    )
    num_heads: int = Field(
        gt=0, description="Heads of the temporal cross-attention."
    )
    dropout: float = Field(ge=0.0, lt=1.0)

    # -- shared strided 1D-CNN over each exposure of X -------------------- #
    x_encoder_channels: list[int] = Field(
        description="Channels of the shared strided 1D-CNN applied to each exposure of X.",
    )
    x_encoder_kernel: int = Field(gt=0)
    x_encoder_stride: int = Field(gt=0)
    s_encoder_channels: list[int] = Field(
        description="Channels of the independent stellar (S) 1D-CNN.",
    )
    s_encoder_kernel: int = Field(gt=0)
    s_encoder_stride: int = Field(gt=0)
    encoder_pool_bins: int = Field(
        gt=0,
        description="Adaptive spectral bins kept per channel before the code read-out.",
    )
    fusion_hidden: int = Field(
        ge=0,
        description="Fusion MLP width. 0 -> num_queries * spectral_latent_dim.",
    )
    fusion_layers: int = Field(gt=0)
    metadata_enc_dim: int = Field(
        gt=0,
        description="Width of the per-exposure metadata code (C), concatenated "
                    "with the spectral code before the temporal Perceiver.",
    )
    metadata_enc_hidden: int | None = Field(
        gt=0,
        description="Hidden width of the per-exposure metadata encoder MLP "
                    "(None -> auto).",
    )
    time_enc_dim: int = Field(
        gt=0,
        description="Width (k) of the per-exposure time code produced by the "
                    "TimeEncoder from continuous exposure times (time_hours), "
                    "concatenated with the spectral + metadata codes before "
                    "the temporal Perceiver.",
    )
    param_decoder_hidden: int | None = Field(
        gt=0,
        description="Hidden width of the per-exposure param-decoder MLP "
                    "(None -> auto).",
    )
