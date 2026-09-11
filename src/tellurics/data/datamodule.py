"""Lightning DataModule: orchestration for whole-night telluric training.

Composes the pieces of :mod:`tellurics.data`:

    wavegrid.Wavegrid                           the shared wavelength grid
    stellar.StellarPool                         the Phoenix spectrum pool
    splits.split_night_indices                  night-level train/val/test split
    datasets.night.TelluricTimeseriesDataset    one sample = one whole night

Each night is multiplied by a single, fixed, seeded stellar spectrum:
``observed = transmission * S``. The night -> star mapping (with its split) is
written to ``<run_dir>/stellar_assignment.csv`` for auditing.

``setup()`` is idempotent: the first call loads the metadata, makes the splits,
assigns the stars and builds the datasets; later calls (e.g. a subsequent
``stage='test'``) return immediately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from tellurics.configs.data import DataConfig
from tellurics.data.datasets.night import TelluricTimeseriesDataset
from tellurics.data.splits import split_night_indices
from tellurics.data.stellar import (
    StellarPool,
    assign_stellar_per_night,
    save_stellar_assignment,
)
from tellurics.data.wavegrid import Wavegrid
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class TelluricDataModule(pl.LightningDataModule):
    """Whole-night DataModule for training the :class:`TelluricEstimator`.

    Args:
        config: Input paths and night-level split settings (the manifest's
            ``data:`` section).
        run_dir: This run's artefact directory; the night -> star audit CSV is
            written to ``<run_dir>/stellar_assignment.csv``.
        batch_size: Whole nights per batch (the manifest's ``training:``
            section).
        num_workers: DataLoader workers (the manifest's ``training:`` section).
    """

    def __init__(
        self,
        config: DataConfig,
        run_dir: str | Path,
        batch_size: int = 4,
        num_workers: int = 2,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.run_dir = Path(run_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    # -- attributes set by setup() ------------------------------------------ #
    wavegrid: Wavegrid | None = None
    stellar_pool: StellarPool | None = None
    stellar_assignment: np.ndarray | None = None      # (n_nights,) star indices
    n_nights: int = 0
    n_frames: int = 0
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    train_ds: TelluricTimeseriesDataset | None = None
    val_ds: TelluricTimeseriesDataset | None = None
    test_ds: TelluricTimeseriesDataset | None = None

    # -- setup -------------------------------------------------------------- #
    def setup(self, stage: str | None = None) -> None:
        """Prepare metadata, splits, star assignment and datasets (once)."""
        if self.train_ds is not None:
            return

        self._load_metadata()  # shapes + wavelength grid
        self._make_splits()    # night-level train/val/test
        self._assign_stars()   # fixed stellar spectrum per night + audit CSV
        self._make_datasets()  # one dataset per split

    def _load_metadata(self) -> None:
        """Read the shapes and the wavelength grid."""
        night_h5 = Path(self.config.night_h5)
        if not night_h5.exists():
            raise FileNotFoundError(
                f"night-major HDF5 not found at {night_h5}. "
                "Create it once with tests/testing_dataset_reshape.ipynb "
                "(it reshapes the flat data/telluric_templates.h5)."
            )

        import h5py

        with h5py.File(night_h5, "r") as f:
            self.n_nights = int(f["transmission"].shape[0])
            self.n_frames = int(f["transmission"].shape[1])
            wl = np.asarray(f["wavelength"][:], np.float64)
            unit = str(f["wavelength"].attrs.get("unit", "nm"))

        self.wavegrid = Wavegrid(wavelength=wl, unit=unit)
        logger.info(
            f"{self.n_nights} nights x {self.n_frames} frames from {night_h5}"
        )

    def _make_splits(self) -> None:
        """Night-level train/val/test split (no exposure crosses splits)."""
        self.train_ids, self.val_ids, self.test_ids = split_night_indices(
            self.n_nights,
            self.config.train_fraction,
            self.config.val_fraction,
            self.config.seed,
        )

    def _assign_stars(self) -> None:
        """Assign a fixed stellar spectrum per night and write the audit CSV."""
        self.stellar_pool = StellarPool(self.config.phoenix_dir)
        self.stellar_assignment = assign_stellar_per_night(
            self.n_nights, self.stellar_pool, seed=self.config.seed
        )
        audit_csv = self.run_dir / "stellar_assignment.csv"
        save_stellar_assignment(
            self.stellar_assignment,
            self.stellar_pool,
            {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids},
            audit_csv,
        )
        logger.info(
            f"train {len(self.train_ids)} / val {len(self.val_ids)} / "
            f"test {len(self.test_ids)}; stellar audit -> {audit_csv}"
        )

    def _make_datasets(self) -> None:
        """Wrap each split in a :class:`TelluricTimeseriesDataset`."""
        common = dict(
            h5_path=self.config.night_h5,
            pool=self.stellar_pool,
            star_assignment=self.stellar_assignment,
        )
        self.train_ds = TelluricTimeseriesDataset(night_ids=self.train_ids, **common)
        self.val_ds = TelluricTimeseriesDataset(night_ids=self.val_ids, **common)
        self.test_ds = TelluricTimeseriesDataset(night_ids=self.test_ids, **common)

    # -- loaders ------------------------------------------------------------ #
    def _loader(self, ds: TelluricTimeseriesDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,  # drop partial training batch only
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
