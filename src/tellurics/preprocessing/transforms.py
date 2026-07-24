"""Individual preprocessing transforms for spectral data."""

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class WavelengthAlignment:
    """Align spectra to a common wavelength grid via interpolation."""

    def __init__(self, target_grid: npt.NDArray[np.float64]) -> None:
        """Initialize with target wavelength grid.

        Args:
            target_grid: Target wavelength array. Shape: (N_wavelength,).
        """
        self.target_grid = target_grid

    def __call__(
        self,
        spectrum: npt.NDArray[np.float64],
        wavelength: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Interpolate spectrum onto target grid.

        Args:
            spectrum: Input spectrum values. Shape: (N,) or (B, N).
            wavelength: Input wavelength array. Shape: (N,).

        Returns:
            Aligned spectrum on target grid.
        """
        if spectrum.ndim == 1:
            interpolator = interp1d(
                wavelength, spectrum, kind="cubic", bounds_error=False, fill_value=0.0
            )
            return interpolator(self.target_grid)

        # Batched
        results = np.zeros((spectrum.shape[0], len(self.target_grid)))
        for i in range(spectrum.shape[0]):
            interpolator = interp1d(
                wavelength, spectrum[i], kind="cubic", bounds_error=False, fill_value=0.0
            )
            results[i] = interpolator(self.target_grid)
        return results


class BadPixelMask:
    """Identify and mask bad pixels in spectra."""

    def __init__(
        self,
        sigma_clip: float = 5.0,
        min_value: float = 0.0,
        max_value: float | None = None,
    ) -> None:
        """Initialize bad pixel masking.

        Args:
            sigma_clip: Number of standard deviations for outlier rejection.
            min_value: Minimum valid pixel value.
            max_value: Maximum valid pixel value (None for no upper limit).
        """
        self.sigma_clip = sigma_clip
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Generate boolean mask for good pixels.

        Args:
            spectrum: Input spectrum. Shape: (N,) or (B, N).

        Returns:
            Boolean mask where True = good pixel.
        """
        mask = spectrum > self.min_value
        if self.max_value is not None:
            mask &= spectrum < self.max_value

        # Sigma clipping
        if spectrum.ndim == 1:
            median = np.median(spectrum)
            std = np.std(spectrum)
            mask &= np.abs(spectrum - median) < self.sigma_clip * std
        else:
            median = np.median(spectrum, axis=-1, keepdims=True)
            std = np.std(spectrum, axis=-1, keepdims=True)
            mask &= np.abs(spectrum - median) < self.sigma_clip * std

        return mask


class StellarDivision:
    """Divide observed spectrum by stellar template: R = X / S."""

    def __init__(self, epsilon: float = 1e-10) -> None:
        """Initialize with numerical stability epsilon.

        Args:
            epsilon: Small value to prevent division by zero.
        """
        self.epsilon = epsilon

    def __call__(
        self,
        observed: npt.NDArray[np.float64],
        stellar: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute R = X / S.

        Args:
            observed: Observed spectrum X. Shape: (N,) or (B, N).
            stellar: Stellar template S. Shape: (N,) or (B, N).

        Returns:
            Normalized spectrum R.
        """
        return observed / (stellar + self.epsilon)


class ContinuumNormalization:
    """Normalize spectrum by fitting and dividing by continuum."""

    def __init__(
        self,
        window_length: int = 201,
        polyorder: int = 3,
    ) -> None:
        """Initialize continuum normalization.

        Args:
            window_length: Savitzky-Golay filter window length.
            polyorder: Polynomial order for the filter.
        """
        self.window_length = window_length
        self.polyorder = polyorder

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize by estimated continuum.

        Args:
            spectrum: Input spectrum. Shape: (N,) or (B, N).

        Returns:
            Continuum-normalized spectrum.
        """
        if spectrum.ndim == 1:
            continuum = savgol_filter(spectrum, self.window_length, self.polyorder)
            return spectrum / (continuum + 1e-10)

        # Batched
        results = np.zeros_like(spectrum)
        for i in range(spectrum.shape[0]):
            continuum = savgol_filter(spectrum[i], self.window_length, self.polyorder)
            results[i] = spectrum[i] / (continuum + 1e-10)
        return results
