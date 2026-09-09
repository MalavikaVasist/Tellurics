"""Dataset and DataModule implementations."""

from tellurics.data.datamodule import (
    StellarPool,
    TelluricDataModule,
    TelluricTimeseriesDataset,
    Wavegrid,
    assign_stellar_per_night,
    build_timeseries_h5,
    save_stellar_assignment,
)
from tellurics.data.datasets import RealObservationDataset, SimulatedDataset
from tellurics.data.night import (
    NightDataModule,
    NightTelluricDataset,
    build_night_file,
    split_night_indices,
)

__all__ = [
    "RealObservationDataset",
    "SimulatedDataset",
    "TelluricDataModule",
    "NightDataModule",
    "NightTelluricDataset",
    "build_night_file",
    "split_night_indices",
    # Whole-night telluric timeseries helpers.
    "Wavegrid",
    "build_timeseries_h5",
    "StellarPool",
    "assign_stellar_per_night",
    "save_stellar_assignment",
    "TelluricTimeseriesDataset",
]

