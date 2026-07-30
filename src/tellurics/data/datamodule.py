"""Lightning DataModule for telluric data."""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from tellurics.configs.data import DataFormat, DatasetConfig
from tellurics.data.datasets import RealObservationDataset, SimulatedDataset
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class TelluricDataModule(pl.LightningDataModule):
    """Lightning DataModule supporting mixed simulated and real data.

    Handles data loading, splitting, and mixed batch creation.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the DataModule.

        Args:
            config: Dataset configuration.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training, validation, or testing.

        Args:
            stage: One of 'fit', 'validate', 'test', 'predict'.
        """
        data_dir = self.config.data_dir
        data_format = self.config.data_format

        # Determine number of samples for splitting
        n_samples = self._count_samples(data_dir, data_format)
        indices = np.random.default_rng(seed=42).permutation(n_samples)

        n_train = int(n_samples * self.config.train_fraction)
        n_val = int(n_samples * self.config.val_fraction)

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train : n_train + n_val].tolist()
        test_idx = indices[n_train + n_val :].tolist()

        if stage in ("fit", None):
            self.train_sim = SimulatedDataset(data_dir, data_format, train_idx)
            self.val_sim = SimulatedDataset(data_dir, data_format, val_idx)

            # Load real data if available
            self.train_real = self._try_load_real(data_dir, data_format)

        if stage in ("test", None):
            self.test_dataset = SimulatedDataset(data_dir, data_format, test_idx)

        if stage == "predict":
            # For prediction, load all available data
            self.predict_dataset = SimulatedDataset(data_dir, data_format)

    def _count_samples(self, data_dir: Path, data_format: DataFormat) -> int:
        """Count total number of simulated samples."""
        import h5py

        match data_format:
            case DataFormat.HDF5:
                filepath = data_dir / "simulated.h5"
                if not filepath.exists():
                    filepath = data_dir / "simulated.hdf5"
                with h5py.File(filepath, "r") as f:
                    return f["spectrum"].shape[0]
            case DataFormat.NUMPY:
                return np.load(data_dir / "spectrum.npy", mmap_mode="r").shape[0]
            case DataFormat.FITS:
                from astropy.io import fits

                with fits.open(data_dir / "simulated.fits") as hdul:
                    return hdul["SPECTRUM"].data.shape[0]

    def _try_load_real(
        self, data_dir: Path, data_format: DataFormat
    ) -> RealObservationDataset | None:
        """Attempt to load real observation data."""
        try:
            return RealObservationDataset(data_dir, data_format)
        except (FileNotFoundError, OSError):
            logger.info("No real observation data found, training with simulated only")
            return None

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Create training DataLoader with mixed batches."""
        datasets = [self.train_sim]
        if self.train_real is not None:
            datasets.append(self.train_real)

        combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

        return DataLoader(
            combined,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Create validation DataLoader."""
        return DataLoader(
            self.val_sim,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Create prediction DataLoader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
