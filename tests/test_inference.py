"""Tests for inference pipeline components."""

import torch

from tellurics.configs.model import ModelConfig
from tellurics.models.cnn import CNNRegressor
from tellurics.models.output import ModelOutput


class TestInferenceOutputStructure:
    """Test that model outputs have the correct structure for inference."""

    def test_output_has_correct_fields(self) -> None:
        config = ModelConfig(
            num_wavelength_bins=512,
            hidden_dim=64,
            num_channels=[32, 64, 32],
        )
        model = CNNRegressor(config)
        model.eval()

        with torch.no_grad():
            output = model(torch.randn(1, 512), torch.randn(1, 6))

        assert isinstance(output, ModelOutput)
        assert output.telluric is not None
        assert output.telluric.shape == (1, 512)

    def test_planet_recovery_computation(self) -> None:
        """Test that planet recovery P_hat = R / T_hat works correctly."""
        spectrum = torch.rand(1, 100) + 0.5  # R = X/S
        telluric_pred = torch.rand(1, 100) * 0.5 + 0.3  # T_hat in [0.3, 0.8]

        # Recover planet
        epsilon = 1e-8
        planet_hat = spectrum / (telluric_pred + epsilon)

        assert planet_hat.shape == spectrum.shape
        assert torch.all(torch.isfinite(planet_hat))

    def test_mc_dropout_shapes(self) -> None:
        """Test MC dropout produces correct shapes."""
        config = ModelConfig(
            num_wavelength_bins=256,
            hidden_dim=32,
            dropout=0.2,
            num_channels=[16, 32, 16],
        )
        model = CNNRegressor(config)
        model.train()  # Enable dropout

        spectrum = torch.randn(2, 256)
        atm_params = torch.randn(2, 6)

        predictions = []
        for _ in range(5):
            output = model(spectrum, atm_params)
            predictions.append(output.telluric)

        stacked = torch.stack(predictions, dim=0)
        assert stacked.shape == (5, 2, 256)

        uncertainty = stacked.std(dim=0)
        assert uncertainty.shape == (2, 256)
