"""Data configuration: input paths, label scaling and the night-level split.

This is the ``data:`` section of an experiment manifest. It feeds
:class:`~tellurics.data.datamodule.TelluricDataModule`, which additionally
takes ``batch_size`` / ``num_workers`` from ``TrainingConfig``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from tellurics.configs.bounds import ParameterBounds


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
    stellar_scale: bool = Field(
        default=True,
        description="Normalize each stellar spectrum to unit mean flux "
                    "before use (``StellarPool(scale=...)``).",
    )

    # -- [0, 1] label scaling --------------------------------------------- #
    scale_metadata: bool = Field(
        default=True,
        description="Map the per-exposure metadata columns (pressure, "
                    "temperature, humidity) onto [0, 1] with the "
                    "``param_bounds`` envelope, before the MetadataEncoder.",
    )
    scale_params: bool = Field(
        default=True,
        description="Map the regression targets (``TARGET_COLUMNS``) onto "
                    "[0, 1] with the ``param_bounds`` envelope, so the MSE is "
                    "computed on one scale. Pair it with "
                    "``model.param_activation: sigmoid`` (the head must be "
                    "bounded too) and invert predictions with "
                    "ParameterScaler.inverse_params.",
    )
    scale_time: bool = Field(
        default=True,
        description="Map ``time_hours`` onto [0, 1] with its ``param_bounds`` "
                    "envelope (default [0, 16], i.e. time_hours / 16).",
    )
    param_bounds: ParameterBounds = Field(
        default_factory=ParameterBounds,
        description="Authoritative physical [min, max] bounds of every "
                    "parameter. They drive the [0, 1] scaling, the validation "
                    "of the stored labels, and the inverse transform that "
                    "returns predictions to physical units. Declaring them "
                    "(instead of reading statistics off the data) keeps the "
                    "mapping deterministic and independent of the "
                    "train/val/test split.",
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
    max_nights: int | None = Field(
        default=None,
        gt=0,
        description="Use only the first N nights (e.g. 100) for a quick "
                    "smoke/debug run, instead of the whole file. The split and "
                    "the stellar assignment then only cover those nights. "
                    "None (default) uses every night in ``night_h5``.",
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
