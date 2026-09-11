"""Data configuration: input paths and the night-level split.

This is the ``data:`` section of an experiment manifest. It feeds
:class:`~tellurics.data.datamodule.TelluricDataModule`, which additionally
takes ``batch_size`` / ``num_workers`` from ``TrainingConfig``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Input paths and night-level split settings for :class:`TelluricDataModule`.

    Relative paths are interpreted against the current working directory (i.e.
    the repo root when running ``python -m tellurics.scripts.train``).
    """

    # -- inputs ---------------------------------------------------------- #
    night_h5: Path = Field(
        default=Path("data/telluric_timeseries/telluric_templates.h5"),
        description="Night-major HDF5; built once by "
                    "tests/testing_dataset_reshape.ipynb.",
    )
    phoenix_dir: Path = Field(
        default=Path("data/phoenix/convolved"),
        description="Directory holding the stellar *.fits pool.",
    )

    # -- DataLoader ------------------------------------------------------ #
    pin_memory: bool = True

    # -- night-level split ----------------------------------------------- #
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = Field(
        default=42,
        description="Seeds the night split and the per-night stellar assignment.",
    )

    @model_validator(mode="after")
    def _validate_fractions(self) -> DataConfig:
        """Ensure train + val leave room for a non-empty test split."""
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction must be < 1.0 (the remainder is "
                f"the test split); got {self.train_fraction} + "
                f"{self.val_fraction}"
            )
        return self
