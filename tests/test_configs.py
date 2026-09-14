"""Tests for configuration validation."""

import pytest

from tellurics.configs.model import ModelArchitecture, ModelConfig
from tellurics.configs.training import TrainingConfig

# Every ModelConfig field, with the problem dims kept tiny.
MODEL_KWARGS = {
    "architecture": "neural_telluric_predictor",
    "num_wavelength_bins": 1024,
    "n_frames_per_series": 12,
    "metadata_dim": 3,
    "param_dim": 16,
    "num_queries": 8,
    "spectral_latent_dim": 32,
    "num_heads": 4,
    "dropout": 0.1,
    "x_encoder_channels": [8, 16, 32],
    "x_encoder_kernel": 7,
    "x_encoder_stride": 2,
    "s_encoder_channels": [8, 16, 32],
    "s_encoder_kernel": 7,
    "s_encoder_stride": 2,
    "encoder_pool_bins": 16,
    "fusion_hidden": 0,
    "fusion_layers": 2,
    "metadata_enc_dim": 16,
    "metadata_enc_hidden": None,
    "time_enc_dim": 16,
    "param_decoder_hidden": None,
}


class TestModelConfig:
    def test_explicit_config(self) -> None:
        config = ModelConfig(**MODEL_KWARGS)
        assert (
            config.architecture is ModelArchitecture.NEURAL_TELLURIC_PREDICTOR
        )
        assert config.n_frames_per_series == 12
        assert config.param_dim == 16

    @pytest.mark.parametrize("field", sorted(MODEL_KWARGS))
    def test_no_field_has_a_default(self, field: str) -> None:
        """Omitting any field is an error: nothing is assumed implicitly."""
        with pytest.raises(ValueError):
            ModelConfig(**{k: v for k, v in MODEL_KWARGS.items() if k != field})

    def test_rejects_unknown_architecture(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(**{**MODEL_KWARGS, "architecture": "cnn"})

    def test_metadata_dim_must_be_positive(self) -> None:
        """metadata_dim=0 would leave the metadata encoder unbuildable."""
        with pytest.raises(ValueError):
            ModelConfig(**{**MODEL_KWARGS, "metadata_dim": 0})

    def test_data_contract_dimensions(self) -> None:
        """The three dims that must agree with the DataModule."""
        config = ModelConfig(**MODEL_KWARGS)
        assert config.metadata_dim == 3   # pressure, temperature, humidity
        assert config.param_dim == 16     # TARGET_COLUMNS
        assert config.num_wavelength_bins == 1024


class TestTrainingConfig:
    def test_default_config(self) -> None:
        config = TrainingConfig()
        assert config.max_epochs == 100
        assert config.optimizer.learning_rate == 1e-4
