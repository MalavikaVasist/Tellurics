"""Tests for model architectures."""

import torch

from tellurics.configs.model import FusionMethod, ModelArchitecture, ModelConfig
from tellurics.models.cnn import CNNRegressor
from tellurics.models.mamba import MambaSpectralModel
from tellurics.models.output import ModelOutput
from tellurics.models.transformer import TransformerEncoder
from tellurics.utils.registry import ModelRegistry


class TestModelOutput:
    def test_minimal_output(self) -> None:
        telluric = torch.randn(4, 1024)
        output = ModelOutput(telluric=telluric)
        assert output.telluric.shape == (4, 1024)
        assert output.planet is None
        assert output.uncertainty is None
        assert output.latent is None

    def test_full_output(self) -> None:
        output = ModelOutput(
            telluric=torch.randn(4, 1024),
            planet=torch.randn(4, 1024),
            uncertainty=torch.randn(4, 1024),
            latent=torch.randn(4, 128),
        )
        assert output.planet is not None
        assert output.latent is not None


class TestCNNRegressor:
    def test_forward_pass(self) -> None:
        config = ModelConfig(
            architecture=ModelArchitecture.CNN,
            num_wavelength_bins=1024,
            hidden_dim=64,
            num_channels=[32, 64, 32],
        )
        model = CNNRegressor(config)
        spectrum = torch.randn(2, 1024)
        atm_params = torch.randn(2, 6)

        output = model(spectrum, atm_params)

        assert isinstance(output, ModelOutput)
        assert output.telluric.shape == (2, 1024)
        assert output.telluric.min() >= 0.0
        assert output.telluric.max() <= 1.0

    def test_film_fusion(self) -> None:
        config = ModelConfig(
            fusion_method=FusionMethod.FILM,
            num_wavelength_bins=512,
            hidden_dim=64,
            num_channels=[32, 64, 32],
        )
        model = CNNRegressor(config)
        output = model(torch.randn(2, 512), torch.randn(2, 6))
        assert output.telluric.shape == (2, 512)

    def test_cross_attention_fusion(self) -> None:
        config = ModelConfig(
            fusion_method=FusionMethod.CROSS_ATTENTION,
            num_wavelength_bins=512,
            hidden_dim=64,
            num_heads=4,
            num_channels=[32, 64, 32],
        )
        model = CNNRegressor(config)
        output = model(torch.randn(2, 512), torch.randn(2, 6))
        assert output.telluric.shape == (2, 512)


class TestTransformerEncoder:
    def test_forward_pass(self) -> None:
        config = ModelConfig(
            architecture=ModelArchitecture.TRANSFORMER,
            num_wavelength_bins=1024,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
        )
        model = TransformerEncoder(config)
        spectrum = torch.randn(2, 1024)
        atm_params = torch.randn(2, 6)

        output = model(spectrum, atm_params)

        assert isinstance(output, ModelOutput)
        assert output.telluric.shape == (2, 1024)
        assert output.telluric.min() >= 0.0
        assert output.telluric.max() <= 1.0


class TestMambaModel:
    def test_forward_pass(self) -> None:
        config = ModelConfig(
            architecture=ModelArchitecture.MAMBA,
            num_wavelength_bins=512,
            hidden_dim=64,
            num_layers=2,
            state_dim=8,
            expand_factor=2,
        )
        model = MambaSpectralModel(config)
        spectrum = torch.randn(2, 512)
        atm_params = torch.randn(2, 6)

        output = model(spectrum, atm_params)

        assert isinstance(output, ModelOutput)
        assert output.telluric.shape == (2, 512)
        assert output.telluric.min() >= 0.0
        assert output.telluric.max() <= 1.0


class TestModelRegistry:
    def test_registered_models(self) -> None:
        models = ModelRegistry.list_models()
        assert "cnn" in models
        assert "transformer" in models
        assert "mamba" in models

    def test_get_model(self) -> None:
        cls = ModelRegistry.get("cnn")
        assert cls == CNNRegressor

    def test_get_unknown_model(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model")
