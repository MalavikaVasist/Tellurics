"""Supervised loss functions for simulated data."""

import torch
import torch.nn as nn

from tellurics.models.output import ModelOutput


class TelluricRegressionLoss(nn.Module):
    """MSE loss between predicted and true telluric spectrum.

    L_tell = MSE(output.telluric, T_true)
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output: ModelOutput, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute telluric regression loss.

        Args:
            output: Model output containing telluric prediction.
            batch: Data batch with 'telluric_true' key.

        Returns:
            Scalar loss value.
        """
        return self.mse(output.telluric, batch["telluric_true"])


class PlanetReconstructionLoss(nn.Module):
    """Loss on the reconstructed planet spectrum.

    P_hat = (X/S) / T_hat = spectrum / output.telluric
    L_planet = MSE(P_hat, P_true)
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.epsilon = epsilon

    def forward(self, output: ModelOutput, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute planet reconstruction loss.

        Args:
            output: Model output containing telluric prediction.
            batch: Data batch with 'spectrum' (R = X/S) and 'planet_true' keys.

        Returns:
            Scalar loss value.
        """
        # Reconstruct planet: P_hat = R / T_hat
        planet_hat = batch["spectrum"] / (output.telluric + self.epsilon)
        return self.mse(planet_hat, batch["planet_true"])
