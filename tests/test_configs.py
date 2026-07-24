"""Tests for configuration validation."""

import pytest
from pydantic import ValidationError

from tellurics.configs.atmospheric import AtmosphericParameters
from tellurics.configs.data import DatasetConfig
from tellurics.configs.loss import LossConfig
from tellurics.configs.model import ModelConfig
from tellurics.configs.training import TrainingConfig


class TestAtmosphericParameters:
    def test_valid_parameters(self) -> None:
        params = AtmosphericParameters(
            airmass=1.5, humidity=50.0, pressure=1013.0, temperature=280.0, pwv=5.0, seeing=1.0
        )
        assert params.airmass == 1.5
        assert params.num_parameters == 6

    def test_optional_parameters(self) -> None:
        params = AtmosphericParameters(airmass=1.2)
        assert params.humidity is None
        assert params.to_tensor_list() == [1.2, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_invalid_airmass(self) -> None:
        with pytest.raises(ValidationError):
            AtmosphericParameters(airmass=0.5)  # Below minimum of 1.0

    def test_to_tensor_list(self) -> None:
        params = AtmosphericParameters(
            airmass=1.5, humidity=60.0, pressure=900.0, temperature=270.0, pwv=3.0, seeing=0.8
        )
        result = params.to_tensor_list()
        assert len(result) == 6
        assert result[0] == 1.5


class TestDatasetConfig:
    def test_valid_config(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        config = DatasetConfig(
            data_dir=tmp_path,
            dataset_type="simulated",
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
        )
        assert config.batch_size == 32

    def test_invalid_fractions(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValidationError):
            DatasetConfig(
                data_dir=tmp_path,
                dataset_type="simulated",
                train_fraction=0.5,
                val_fraction=0.5,
                test_fraction=0.5,
            )


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


class TestLossConfig:
    def test_default_config(self) -> None:
        config = LossConfig()
        assert config.lambda_telluric == 1.0
        assert config.smoothness_order == 2
