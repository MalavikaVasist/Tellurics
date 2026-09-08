"""Neural network architectures for telluric prediction."""

from tellurics.models.base import BaseSpectralModel
from tellurics.models.cnn import CNNRegressor
from tellurics.models.mamba import MambaSpectralModel
from tellurics.models.night import PerceiverNightModel
from tellurics.models.output import ModelOutput
from tellurics.models.temporal_conv import (
    FusionMLP,
    MetadataEncoding,
    ParamDecoder,
    SpectralDecoder,
    SpectralEncoder,
    StellarEncoder,
    TelluricModel,
    TemporalConvConfig,
    TemporalDecoder,
    TemporalPerceiver,
)
from tellurics.models.transformer import TransformerEncoder

__all__ = [
    "BaseSpectralModel",
    "CNNRegressor",
    "MambaSpectralModel",
    "PerceiverNightModel",
    "ModelOutput",
    "TransformerEncoder",
    # Whole-night CNN auto-encoder with temporal Perceiver compression.
    "TemporalConvConfig",
    "SpectralEncoder",
    "MetadataEncoding",
    "TemporalPerceiver",
    "StellarEncoder",
    "FusionMLP",
    "TemporalDecoder",
    "SpectralDecoder",
    "ParamDecoder",
    "TelluricModel",
]
