"""Tests for preprocessing transforms."""

import numpy as np

from tellurics.preprocessing.transforms import (
    BadPixelMask,
    ContinuumNormalization,
    StellarDivision,
    WavelengthAlignment,
)
from tellurics.preprocessing.pipeline import PreprocessingPipeline


class TestStellarDivision:
    def test_basic_division(self) -> None:
        division = StellarDivision()
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        stellar = np.array([2.0, 2.0, 2.0, 2.0])
        result = division(observed, stellar)
        np.testing.assert_allclose(result, [0.5, 1.0, 1.5, 2.0])

    def test_no_divide_by_zero(self) -> None:
        division = StellarDivision()
        observed = np.array([1.0, 2.0, 3.0])
        stellar = np.array([0.0, 0.0, 0.0])
        result = division(observed, stellar)
        assert np.all(np.isfinite(result))

    def test_batched(self) -> None:
        division = StellarDivision()
        observed = np.random.rand(10, 100)
        stellar = np.random.rand(10, 100) + 0.1
        result = division(observed, stellar)
        assert result.shape == (10, 100)


class TestBadPixelMask:
    def test_basic_masking(self) -> None:
        masker = BadPixelMask(sigma_clip=3.0, min_value=0.0)
        # Use a larger array so std is meaningful, with a clear outlier
        spectrum = np.ones(50)
        spectrum[-1] = 100.0  # Clear outlier
        mask = masker(spectrum)
        assert not mask[-1]  # Outlier should be masked

    def test_negative_values(self) -> None:
        masker = BadPixelMask(min_value=0.0)
        spectrum = np.array([-1.0, 1.0, 1.0, 1.0])
        mask = masker(spectrum)
        assert mask[0] is np.False_


class TestWavelengthAlignment:
    def test_interpolation(self) -> None:
        target = np.linspace(1.0, 2.0, 100)
        aligner = WavelengthAlignment(target)

        wavelength = np.linspace(0.8, 2.2, 50)
        spectrum = np.sin(wavelength)

        result = aligner(spectrum, wavelength)
        assert result.shape == (100,)
        # Check that interpolated values are reasonable
        expected = np.sin(target)
        np.testing.assert_allclose(result, expected, atol=0.1)


class TestContinuumNormalization:
    def test_flat_continuum(self) -> None:
        norm = ContinuumNormalization(window_length=51, polyorder=3)
        # A flat spectrum should stay roughly flat
        spectrum = np.ones(200) + 0.01 * np.random.randn(200)
        result = norm(spectrum)
        np.testing.assert_allclose(result, np.ones(200), atol=0.05)


class TestPreprocessingPipeline:
    def test_full_pipeline(self) -> None:
        pipeline = PreprocessingPipeline(sigma_clip=5.0)
        observed = np.random.rand(100) + 0.5
        stellar = np.ones(100) * 0.8

        result, mask = pipeline(observed, stellar)
        assert result.shape == (100,)
        assert mask.shape == (100,)
        assert mask.dtype == np.bool_
