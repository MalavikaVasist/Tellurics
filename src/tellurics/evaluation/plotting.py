"""Visualization utilities for evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_predictions(
    wavelength: npt.NDArray[np.float64],
    predicted_telluric: npt.NDArray[np.float64],
    true_telluric: npt.NDArray[np.float64] | None = None,
    recovered_planet: npt.NDArray[np.float64] | None = None,
    true_planet: npt.NDArray[np.float64] | None = None,
    save_path: Path | None = None,
    title: str = "Telluric Correction Results",
) -> plt.Figure:
    """Generate diagnostic plots for telluric correction.

    Args:
        wavelength: Wavelength grid.
        predicted_telluric: Predicted telluric transmission.
        true_telluric: Ground truth telluric (if available).
        recovered_planet: Recovered planet spectrum (if available).
        true_planet: Ground truth planet spectrum (if available).
        save_path: Path to save figure (None to show).
        title: Figure title.

    Returns:
        Matplotlib figure object.
    """
    num_panels = 2
    if recovered_planet is not None:
        num_panels += 1

    fig, axes = plt.subplots(num_panels, 1, figsize=(12, 4 * num_panels), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Panel 1: Telluric spectrum
    ax = axes[0]
    ax.plot(wavelength, predicted_telluric, "b-", alpha=0.8, label="Predicted", linewidth=0.8)
    if true_telluric is not None:
        ax.plot(wavelength, true_telluric, "r--", alpha=0.6, label="True", linewidth=0.8)
    ax.set_ylabel("Telluric Transmission")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.set_title("Telluric Spectrum")

    # Panel 2: Residuals
    ax = axes[1]
    if true_telluric is not None:
        residuals = predicted_telluric - true_telluric
        ax.plot(wavelength, residuals, "k-", alpha=0.7, linewidth=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Residual (Pred - True)")
        rms = np.sqrt(np.mean(residuals**2))
        ax.set_title(f"Telluric Residuals (RMS = {rms:.6f})")
    else:
        ax.text(0.5, 0.5, "No ground truth available", transform=ax.transAxes, ha="center")

    # Panel 3: Planet recovery
    if recovered_planet is not None:
        ax = axes[2]
        ax.plot(wavelength, recovered_planet, "g-", alpha=0.8, label="Recovered", linewidth=0.8)
        if true_planet is not None:
            ax.plot(wavelength, true_planet, "r--", alpha=0.6, label="True", linewidth=0.8)
        ax.set_ylabel("Planet Spectrum")
        ax.legend(loc="upper right")
        ax.set_title("Planet Signal Recovery")

    axes[-1].set_xlabel("Wavelength")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
