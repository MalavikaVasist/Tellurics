"""Dataset implementations for simulated and real observations."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tellurics.configs.data import DataFormat
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class SimulatedDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for simulated observations with full labels.

    Each sample contains:
        - spectrum: Normalized spectrum R = X/S
        - atm_params: Atmospheric parameters
        - telluric_true: Ground truth telluric transmission
        - planet_true: Ground truth planet spectrum
        - observed: Original observed spectrum X
        - stellar: Stellar template S
        - is_simulated: Boolean flag (always True)
    """

    def __init__(
        self,
        data_dir: Path,
        data_format: DataFormat = DataFormat.HDF5,
        indices: list[int] | None = None,
    ) -> None:
        """Initialize simulated dataset.

        Args:
            data_dir: Path to the data directory.
            data_format: Format of the data files.
            indices: Optional subset of indices to use.
        """
        self.data_dir = data_dir
        self.data_format = data_format
        self._load_data(indices)

    def _load_data(self, indices: list[int] | None) -> None:
        """Load data from disk."""
        match self.data_format:
            case DataFormat.HDF5:
                self._load_hdf5(indices)
            case DataFormat.NUMPY:
                self._load_numpy(indices)
            case DataFormat.FITS:
                self._load_fits(indices)

    def _load_hdf5(self, indices: list[int] | None) -> None:
        """Load data from HDF5 file."""
        filepath = self.data_dir / "simulated.h5"
        if not filepath.exists():
            # Try alternative naming
            filepath = self.data_dir / "simulated.hdf5"

        with h5py.File(filepath, "r") as f:
            sl = np.s_[:] if indices is None else np.s_[indices]
            self.spectra = torch.from_numpy(f["spectrum"][sl].astype(np.float32))
            self.atm_params = torch.from_numpy(f["atm_params"][sl].astype(np.float32))
            self.telluric_true = torch.from_numpy(f["telluric"][sl].astype(np.float32))
            self.planet_true = torch.from_numpy(f["planet"][sl].astype(np.float32))
            self.observed = torch.from_numpy(f["observed"][sl].astype(np.float32))
            self.stellar = torch.from_numpy(f["stellar"][sl].astype(np.float32))

        logger.info(f"Loaded {len(self.spectra)} simulated samples from {filepath}")

    def _load_numpy(self, indices: list[int] | None) -> None:
        """Load data from numpy files."""
        sl = np.s_[:] if indices is None else np.s_[indices]
        data_dir = self.data_dir

        self.spectra = torch.from_numpy(
            np.load(data_dir / "spectrum.npy")[sl].astype(np.float32)
        )
        self.atm_params = torch.from_numpy(
            np.load(data_dir / "atm_params.npy")[sl].astype(np.float32)
        )
        self.telluric_true = torch.from_numpy(
            np.load(data_dir / "telluric.npy")[sl].astype(np.float32)
        )
        self.planet_true = torch.from_numpy(
            np.load(data_dir / "planet.npy")[sl].astype(np.float32)
        )
        self.observed = torch.from_numpy(
            np.load(data_dir / "observed.npy")[sl].astype(np.float32)
        )
        self.stellar = torch.from_numpy(
            np.load(data_dir / "stellar.npy")[sl].astype(np.float32)
        )

        logger.info(f"Loaded {len(self.spectra)} simulated samples from numpy")

    def _load_fits(self, indices: list[int] | None) -> None:
        """Load data from FITS files."""
        from astropy.io import fits

        filepath = self.data_dir / "simulated.fits"
        with fits.open(filepath) as hdul:
            sl = slice(None) if indices is None else indices
            self.spectra = torch.from_numpy(hdul["SPECTRUM"].data[sl].astype(np.float32))
            self.atm_params = torch.from_numpy(hdul["ATM_PARAMS"].data[sl].astype(np.float32))
            self.telluric_true = torch.from_numpy(hdul["TELLURIC"].data[sl].astype(np.float32))
            self.planet_true = torch.from_numpy(hdul["PLANET"].data[sl].astype(np.float32))
            self.observed = torch.from_numpy(hdul["OBSERVED"].data[sl].astype(np.float32))
            self.stellar = torch.from_numpy(hdul["STELLAR"].data[sl].astype(np.float32))

        logger.info(f"Loaded {len(self.spectra)} simulated samples from FITS")

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "spectrum": self.spectra[idx],
            "atm_params": self.atm_params[idx],
            "telluric_true": self.telluric_true[idx],
            "planet_true": self.planet_true[idx],
            "observed": self.observed[idx],
            "stellar": self.stellar[idx],
            "is_simulated": torch.tensor(True),
        }


class RealObservationDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for real observations without labels.

    Each sample contains:
        - spectrum: Normalized spectrum R = X/S
        - atm_params: Atmospheric parameters
        - observed: Original observed spectrum X
        - stellar: Stellar template S
        - is_simulated: Boolean flag (always False)
    """

    def __init__(
        self,
        data_dir: Path,
        data_format: DataFormat = DataFormat.HDF5,
        indices: list[int] | None = None,
    ) -> None:
        """Initialize real observation dataset.

        Args:
            data_dir: Path to the data directory.
            data_format: Format of the data files.
            indices: Optional subset of indices to use.
        """
        self.data_dir = data_dir
        self.data_format = data_format
        self._load_data(indices)

    def _load_data(self, indices: list[int] | None) -> None:
        """Load data from disk."""
        match self.data_format:
            case DataFormat.HDF5:
                self._load_hdf5(indices)
            case DataFormat.NUMPY:
                self._load_numpy(indices)
            case DataFormat.FITS:
                self._load_fits(indices)

    def _load_hdf5(self, indices: list[int] | None) -> None:
        """Load from HDF5."""
        filepath = self.data_dir / "real.h5"
        if not filepath.exists():
            filepath = self.data_dir / "real.hdf5"

        with h5py.File(filepath, "r") as f:
            sl = np.s_[:] if indices is None else np.s_[indices]
            self.spectra = torch.from_numpy(f["spectrum"][sl].astype(np.float32))
            self.atm_params = torch.from_numpy(f["atm_params"][sl].astype(np.float32))
            self.observed = torch.from_numpy(f["observed"][sl].astype(np.float32))
            self.stellar = torch.from_numpy(f["stellar"][sl].astype(np.float32))

        logger.info(f"Loaded {len(self.spectra)} real samples from {filepath}")

    def _load_numpy(self, indices: list[int] | None) -> None:
        """Load from numpy files."""
        sl = np.s_[:] if indices is None else np.s_[indices]
        data_dir = self.data_dir

        self.spectra = torch.from_numpy(
            np.load(data_dir / "spectrum_real.npy")[sl].astype(np.float32)
        )
        self.atm_params = torch.from_numpy(
            np.load(data_dir / "atm_params_real.npy")[sl].astype(np.float32)
        )
        self.observed = torch.from_numpy(
            np.load(data_dir / "observed_real.npy")[sl].astype(np.float32)
        )
        self.stellar = torch.from_numpy(
            np.load(data_dir / "stellar_real.npy")[sl].astype(np.float32)
        )

        logger.info(f"Loaded {len(self.spectra)} real samples from numpy")

    def _load_fits(self, indices: list[int] | None) -> None:
        """Load from FITS."""
        from astropy.io import fits

        filepath = self.data_dir / "real.fits"
        with fits.open(filepath) as hdul:
            sl = slice(None) if indices is None else indices
            self.spectra = torch.from_numpy(hdul["SPECTRUM"].data[sl].astype(np.float32))
            self.atm_params = torch.from_numpy(hdul["ATM_PARAMS"].data[sl].astype(np.float32))
            self.observed = torch.from_numpy(hdul["OBSERVED"].data[sl].astype(np.float32))
            self.stellar = torch.from_numpy(hdul["STELLAR"].data[sl].astype(np.float32))

        logger.info(f"Loaded {len(self.spectra)} real samples from FITS")

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "spectrum": self.spectra[idx],
            "atm_params": self.atm_params[idx],
            "observed": self.observed[idx],
            "stellar": self.stellar[idx],
            "is_simulated": torch.tensor(False),
        }
