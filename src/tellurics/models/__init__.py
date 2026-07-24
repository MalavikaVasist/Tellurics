"""Neural network architectures for telluric prediction."""

from tellurics.models.base import BaseSpectralModel
from tellurics.models.cnn import CNNRegressor
from tellurics.models.output import ModelOutput
from tellurics.models.transformer import TransformerEncoder

__all__ = [
    "BaseSpectralModel",
    "CNNRegressor",
    "ModelOutput",
    "TransformerEncoder",
]
