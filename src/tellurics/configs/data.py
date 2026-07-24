"""Dataset configuration."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class DataFormat(str, Enum):
    """Supported input data formats."""

    HDF5 = "hdf5"
    FITS = "fits"
    NUMPY = "numpy"


class DatasetType(str, Enum):
    """Type of dataset."""

    SIMULATED = "simulated"
    REAL = "real"


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and splitting."""

    data_dir: Path
    dataset_type: DatasetType
    data_format: DataFormat = DataFormat.HDF5
    num_wavelength_bins: int = Field(default=4096, gt=0)
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    simulated_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of simulated data in mixed batches",
    )

    @model_validator(mode="after")
    def validate_fractions(self) -> "DatasetConfig":
        """Ensure train/val/test fractions sum to 1.0."""
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            msg = f"Train/val/test fractions must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self
