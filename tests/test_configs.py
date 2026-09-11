"""Tests for configuration validation."""

from tellurics.configs.model import ModelConfig
from tellurics.configs.training import TrainingConfig


class TestModelConfig:
    def test_default_config(self) -> None:
        config = ModelConfig()
        assert config.architecture.value == "cnn"
        assert config.hidden_dim == 256
        assert config.fusion_method.value == "film"

    def test_transformer_config(self) -> None:
        config = ModelConfig(architecture="transformer", num_heads=8)
        assert config.num_heads == 8


class TestTrainingConfig:
    def test_default_config(self) -> None:
        config = TrainingConfig()
        assert config.max_epochs == 100
        assert config.optimizer.learning_rate == 1e-4
