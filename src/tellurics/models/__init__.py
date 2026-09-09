"""Neural network architectures for telluric prediction."""

from tellurics.models.base import BaseSpectralModel
from tellurics.models.cnn import CNNRegressor
from tellurics.models.decoders import ParamDecoder, TelluricDecoder
from tellurics.models.encoders import (
    MetadataEncoder,
    SpectralEncoder,
    StellarEncoder,
    TimeEncoder,
)
from tellurics.models.fusion import (
    AtmosphericEmbedding,
    ConcatenationFusion,
    CrossAttentionFusion,
    FiLMFusion,
    FusionMLP,
)
from tellurics.models.mamba import MambaSpectralModel
from tellurics.models.night import TelluricEstimator, TelluricEstimatorConfig
from tellurics.models.output import ModelOutput
from tellurics.models.temporal import (
    MultiHeadCrossAttention,
    TemporalDecoder,
    TemporalPerceiver,
)
from tellurics.models.transformer import TransformerEncoder

__all__ = [
    # Per-exposure (single-spectrum) models + shared base.
    "BaseSpectralModel",
    "CNNRegressor",
    "TransformerEncoder",
    "MambaSpectralModel",
    "ModelOutput",
    # Whole-night telluric parameter estimator.
    "TelluricEstimatorConfig",
    "TelluricEstimator",
    # Per-exposure encoders (spectrum / stellar / metadata / time).
    "SpectralEncoder",
    "StellarEncoder",
    "MetadataEncoder",
    "TimeEncoder",
    # Temporal (exposure-axis) attention building blocks.
    "MultiHeadCrossAttention",
    "TemporalPerceiver",
    "TemporalDecoder",
    # Per-exposure fusion / decoders.
    "FusionMLP",
    "ParamDecoder",
    "TelluricDecoder",
    # Per-exposure conditioning fusion (per-exposure models).
    "AtmosphericEmbedding",
    "ConcatenationFusion",
    "FiLMFusion",
    "CrossAttentionFusion",
]
