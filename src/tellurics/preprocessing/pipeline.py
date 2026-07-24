"""Preprocessing pipeline combining multiple transforms."""

import numpy as np
import numpy.typing as npt

from tellurics.preprocessing.transforms import (
    BadPixelMask,
    ContinuumNormalization,
    StellarDivision,
    WavelengthAlignment,
)
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Configurable preprocessing pipeline for spectral data.

    Applies transforms in sequence:
        1. Wavelength alignment (optional)
        2. Bad pixel masking
        3. Stellar division (R = X/S)
        4. Continuum normalization (optional)
    """

    def __init__(
        self,
        target_wavelength_grid: npt.NDArray[np.float64] | None = None,
        apply_continuum_norm: bool = False,
        sigma_clip: float = 5.0,
        epsilon: float = 1e-10,
    ) -> None:
        """Initialize the preprocessing pipeline.

        Args:
            target_wavelength_grid: Common wavelength grid for alignment.
            apply_continuum_norm: Whether to apply continuum normalization.
            sigma_clip: Sigma clipping threshold for bad pixel masking.
            epsilon: Numerical stability for division.
        """
        self.wavelength_alignment = (
            WavelengthAlignment(target_wavelength_grid)
            if target_wavelength_grid is not None
            else None
        )
        self.bad_pixel_mask = BadPixelMask(sigma_clip=sigma_clip)
        self.stellar_division = StellarDivision(epsilon=epsilon)
        self.continuum_norm = ContinuumNormalization() if apply_continuum_norm else None

    def __call__(
        self,
        observed: npt.NDArray[np.float64],
        stellar: npt.NDArray[np.float64],
        wavelength: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Run the full preprocessing pipeline.

        Args:
            observed: Observed spectrum X. Shape: (N,) or (B, N).
            stellar: Stellar template S. Shape: (N,) or (B, N).
            wavelength: Wavelength array (required if alignment is enabled).

        Returns:
            Tuple of (preprocessed spectrum R, good pixel mask).
        """
        # Wavelength alignment
        if self.wavelength_alignment is not None:
            if wavelength is None:
                msg = "Wavelength array required for alignment"
                raise ValueError(msg)
            observed = self.wavelength_alignment(observed, wavelength)
            stellar = self.wavelength_alignment(stellar, wavelength)

        # Bad pixel masking
        mask = self.bad_pixel_mask(observed)

        # Stellar division: R = X / S
        r_spectrum = self.stellar_division(observed, stellar)

        # Apply mask (set bad pixels to 1.0 = no telluric absorption)
        if r_spectrum.ndim == 1:
            r_spectrum[~mask] = 1.0
        else:
            r_spectrum[~mask] = 1.0

        # Continuum normalization
        if self.continuum_norm is not None:
            r_spectrum = self.continuum_norm(r_spectrum)

        return r_spectrum, mask
