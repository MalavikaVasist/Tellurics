"""Configuration models using Pydantic v2."""

from tellurics.configs.atmospheric import AtmosphericParameters
from tellurics.configs.data import DatasetConfig
from tellurics.configs.inference import InferenceConfig
from tellurics.configs.loss import LossConfig
from tellurics.configs.model import ModelConfig
from tellurics.configs.training import OptimizerConfig, SchedulerConfig, TrainingConfig

__all__ = [
    "AtmosphericParameters",
    "DatasetConfig",
    "InferenceConfig",
    "LossConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
]
