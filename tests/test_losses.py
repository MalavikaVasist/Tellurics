"""Tests for loss functions."""

import torch

from tellurics.configs.loss import LossConfig
from tellurics.losses.composite import CompositeLoss
from tellurics.losses.physics import PhysicalConstraintLoss, SmoothnessLoss
from tellurics.losses.supervised import PlanetReconstructionLoss, TelluricRegressionLoss
from tellurics.models.output import ModelOutput


class TestTelluricRegressionLoss:
    def test_zero_loss(self) -> None:
        loss_fn = TelluricRegressionLoss()
        output = ModelOutput(telluric=torch.ones(4, 100))
        batch = {"telluric_true": torch.ones(4, 100)}
        loss = loss_fn(output, batch)
        assert loss.item() == 0.0

    def test_nonzero_loss(self) -> None:
        loss_fn = TelluricRegressionLoss()
        output = ModelOutput(telluric=torch.ones(4, 100))
        batch = {"telluric_true": torch.zeros(4, 100)}
        loss = loss_fn(output, batch)
        assert loss.item() > 0.0


class TestPlanetReconstructionLoss:
    def test_basic(self) -> None:
        loss_fn = PlanetReconstructionLoss()
        # If telluric is 1, then planet_hat = spectrum
        output = ModelOutput(telluric=torch.ones(4, 100))
        batch = {
            "spectrum": torch.ones(4, 100) * 0.5,
            "planet_true": torch.ones(4, 100) * 0.5,
        }
        loss = loss_fn(output, batch)
        assert loss.item() < 1e-6


class TestSmoothnessLoss:
    def test_smooth_signal(self) -> None:
        loss_fn = SmoothnessLoss(order=2)
        # Linear signal should have near-zero second derivative
        telluric = torch.linspace(0, 1, 100).unsqueeze(0).expand(4, -1)
        output = ModelOutput(telluric=telluric)
        loss = loss_fn(output, {})
        assert loss.item() < 1e-6

    def test_noisy_signal(self) -> None:
        loss_fn = SmoothnessLoss(order=2)
        telluric = torch.randn(4, 100)
        output = ModelOutput(telluric=telluric)
        loss = loss_fn(output, {})
        assert loss.item() > 0.0


class TestPhysicalConstraintLoss:
    def test_valid_range(self) -> None:
        loss_fn = PhysicalConstraintLoss(margin=0.0)
        telluric = torch.rand(4, 100)  # Already in [0, 1]
        output = ModelOutput(telluric=telluric)
        loss = loss_fn(output, {})
        assert loss.item() == 0.0

    def test_out_of_range(self) -> None:
        loss_fn = PhysicalConstraintLoss(margin=0.0)
        telluric = torch.tensor([[1.5, -0.5, 0.5, 2.0]])
        output = ModelOutput(telluric=telluric)
        loss = loss_fn(output, {})
        assert loss.item() > 0.0


class TestCompositeLoss:
    def test_simulated_batch(self) -> None:
        config = LossConfig()
        criterion = CompositeLoss(config)

        output = ModelOutput(telluric=torch.sigmoid(torch.randn(4, 100)))
        batch = {
            "spectrum": torch.rand(4, 100) + 0.1,
            "atm_params": torch.randn(4, 6),
            "telluric_true": torch.rand(4, 100),
            "planet_true": torch.rand(4, 100),
            "observed": torch.rand(4, 100),
            "stellar": torch.rand(4, 100) + 0.1,
            "is_simulated": torch.ones(4),
        }

        losses = criterion(output, batch)
        assert "total" in losses
        assert "telluric" in losses
        assert losses["total"].item() > 0.0

    def test_mixed_batch(self) -> None:
        config = LossConfig()
        criterion = CompositeLoss(config)

        output = ModelOutput(telluric=torch.sigmoid(torch.randn(4, 100)))
        batch = {
            "spectrum": torch.rand(4, 100) + 0.1,
            "atm_params": torch.randn(4, 6),
            "telluric_true": torch.rand(4, 100),
            "planet_true": torch.rand(4, 100),
            "observed": torch.rand(4, 100),
            "stellar": torch.rand(4, 100) + 0.1,
            "is_simulated": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        }

        losses = criterion(output, batch)
        assert "total" in losses
        assert losses["total"].item() > 0.0
