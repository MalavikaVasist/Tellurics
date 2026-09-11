"""Whole-night telluric data: the dataset, its DataModule, and its pieces.

    datamodule.py          Lightning orchestration (TelluricDataModule)
    datasets/night.py      one sample = one whole night
    stellar.py             StellarPool + per-night star assignment
    splits.py              night-level train/val/test split
    wavegrid.py            Wavegrid
"""

from tellurics.data.datamodule import TelluricDataModule
from tellurics.data.datasets.night import (
    METADATA_COLUMNS,
    PHYSICAL_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
    TelluricTimeseriesDataset,
)
from tellurics.data.splits import split_night_indices
from tellurics.data.stellar import (
    StellarPool,
    assign_stellar_per_night,
    save_stellar_assignment,
)
from tellurics.data.wavegrid import Wavegrid

__all__ = [
    # DataModule.
    "TelluricDataModule",
    # Dataset + the label-column schema it expects.
    "TelluricTimeseriesDataset",
    "TIME_COLUMN",
    "METADATA_COLUMNS",
    "PHYSICAL_COLUMNS",
    "TARGET_COLUMNS",
    # Supporting pieces.
    "Wavegrid",
    "StellarPool",
    "assign_stellar_per_night",
    "save_stellar_assignment",
    "split_night_indices",
]
