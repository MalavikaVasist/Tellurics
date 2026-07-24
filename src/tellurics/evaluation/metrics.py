"""Evaluation metrics for telluric correction and planet recovery."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class MetricResult:
    """Container for evaluation metric results."""

    rmse: float
    mae: float
    r_squared: float
    correlation: float | None = None
    snr_improvement: float | None = None


class TelluricMetrics:
    """Compute evaluation metrics for telluric and planet predictions."""

    @staticmethod
    def compute_telluric_metrics(
        predicted: npt.NDArray[np.float64] | torch.Tensor,
        true: npt.NDArray[np.float64] | torch.Tensor,
    ) -> MetricResult:
        """Compute metrics for telluric prediction.

        Args:
            predicted: Predicted telluric spectrum.
            true: Ground truth telluric spectrum.

        Returns:
            MetricResult with RMSE, MAE, R².
        """
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        if isinstance(true, torch.Tensor):
            true = true.detach().cpu().numpy()

        residuals = predicted - true
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r_squared = float(1.0 - ss_res / (ss_tot + 1e-10))

        return MetricResult(rmse=rmse, mae=mae, r_squared=r_squared)

    @staticmethod
    def compute_planet_metrics(
        recovered_planet: npt.NDArray[np.float64] | torch.Tensor,
        true_planet: npt.NDArray[np.float64] | torch.Tensor,
        observed_spectrum: npt.NDArray[np.float64] | torch.Tensor | None = None,
    ) -> MetricResult:
        """Compute metrics for planet signal recovery.

        Args:
            recovered_planet: Recovered planet spectrum.
            true_planet: Ground truth planet spectrum.
            observed_spectrum: Original observed spectrum (for SNR computation).

        Returns:
            MetricResult with RMSE, MAE, R², correlation, and optionally SNR improvement.
        """
        if isinstance(recovered_planet, torch.Tensor):
            recovered_planet = recovered_planet.detach().cpu().numpy()
        if isinstance(true_planet, torch.Tensor):
            true_planet = true_planet.detach().cpu().numpy()

        residuals = recovered_planet - true_planet
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((true_planet - np.mean(true_planet)) ** 2)
        r_squared = float(1.0 - ss_res / (ss_tot + 1e-10))

        # Pearson correlation coefficient
        correlation = float(np.corrcoef(recovered_planet.flatten(), true_planet.flatten())[0, 1])

        # SNR improvement
        snr_improvement = None
        if observed_spectrum is not None:
            if isinstance(observed_spectrum, torch.Tensor):
                observed_spectrum = observed_spectrum.detach().cpu().numpy()
            noise_before = np.std(observed_spectrum - true_planet)
            noise_after = np.std(recovered_planet - true_planet)
            if noise_after > 0:
                snr_improvement = float(noise_before / noise_after)

        return MetricResult(
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            correlation=correlation,
            snr_improvement=snr_improvement,
        )
