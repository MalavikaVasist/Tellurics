"""Tests for configuration validation."""

import pytest

from tellurics.configs.model import ModelArchitecture, ModelConfig
from tellurics.configs.training import TrainingConfig


class TestModelConfig:
    def test_default_config(self) -> None:
        config = ModelConfig()
        assert (
            config.architecture is ModelArchitecture.NEURAL_TELLURIC_PREDICTOR
        )
        assert config.n_frames_per_series == 73
        assert config.param_dim == 20
        assert config.num_heads == 8

    def test_rejects_unknown_architecture(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(architecture="cnn")


class TestTrainingConfig:
    def test_default_config(self) -> None:
        config = TrainingConfig()
        assert config.max_epochs == 100
        assert config.optimizer.learning_rate == 1e-4
