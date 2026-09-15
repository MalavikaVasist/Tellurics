"""Authoritative physical bounds for every parameter in the label table.

This module is the **single source of truth** for three things that must never
disagree:

1. the **schema** -- the field order of :class:`ParameterBounds` defines
   ``PHYSICAL_COLUMNS`` and therefore ``TARGET_COLUMNS``, i.e. the meaning of
   every index of the target vector ``theta["params"]``;
2. the **[0, 1] scaling** applied by :mod:`tellurics.data.scaling`;
3. the **validation** of raw label values and the **inverse transform** that
   maps a normalized prediction back to physical units.

Because the bounds are *declared* here instead of being derived from the data,
the normalization is deterministic and independent of the train/validation/test
split, of ``data.max_nights`` and of the batch composition.

Provenance of the defaults (``ai-tfm-tellurics`` generators that produced the
grid):

* ``pressure`` / ``temperature`` / ``humidity`` -- ``configs/paranal_generator.toml``
  (``physical_min`` / ``physical_max``; temperature converted from degC to K);
* ``angle`` (zenith angle, ``< 90`` deg) and ``airmass`` (``>= 1``; the airmass
  pool keeps the altitude ``>= 30`` deg, i.e. ``sec z <= 2``) --
  ``scripts/15b_build_telfit_condition_series_variable_molecules.py`` and
  ``scripts/14_build_paranal_airmass_pool.py``;
* molecular abundances in ppmv -- ``configs/telfit_conditions_production_variable_molecules.toml``
  (the night-to-night abundance envelope).

Two consequences worth knowing before reading labels through these bounds:

* the generator perturbs each nightly abundance by a truncated Gaussian of at
  most +/-2 %, so individual frames can sit slightly *above* the envelope of the
  nightly base abundances (e.g. ``o3`` reaches 0.0428 against a declared 0.042).
  Such values are clipped to ``[0, 1]`` on the way in and counted by
  :meth:`tellurics.data.scaling.MinMax.count_out_of_bounds`, which the
  DataModule reports once per setup -- widen the bound if that distortion is
  not acceptable;
* ``no`` is sampled **log-uniformly**, so its linear ``[min, max]`` width is
  enormous and every realistic value collapses to ~0 after linear min-max
  scaling. A log transform would be the right scaling for that column.

Degenerate bounds (``min == max``) are intentionally *allowed*: several
abundances are constant in the production grid (``co``, ``o2``, ``no``, ``so2``,
``no2``, ``nh3``, ``hno3``). They scale to a constant ``0.0`` and invert back to
``min`` -- never a division by zero.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ValidationInfo, field_validator

__all__ = [
    "ParameterBounds",
    "PHYSICAL_COLUMNS",
    "TARGET_COLUMNS",
    "METADATA_COLUMNS",
    "TIME_COLUMN",
]

# The column consumed by the TimeEncoder (never a regression target).
TIME_COLUMN = "time_hours"

# The columns consumed by the MetadataEncoder.
METADATA_COLUMNS = ("pressure", "temperature", "humidity")


class ParameterBounds(BaseModel):
    """Physical ``[min, max]`` bounds of every column the model touches.

    The field order is the canonical (file) column order, so it defines
    ``PHYSICAL_COLUMNS`` and the index layout of the target vector. Units:
    hPa, K, %, deg, dimensionless and ppmv respectively.

    A value of ``min == max`` is legal and means "this parameter is constant
    over the grid" (see the module docstring).
    """

    # -- atmosphere / geometry --------------------------------------------- #
    time_hours: tuple[float, float] = (0.0, 16.0)         # hours
    pressure: tuple[float, float] = (600.0, 850.0)        # hPa
    temperature: tuple[float, float] = (233.15, 318.15)   # K (-40 .. +45 degC)
    humidity: tuple[float, float] = (0.0, 100.0)          # %
    angle: tuple[float, float] = (0.0, 90.0)              # zenith angle, deg
    airmass: tuple[float, float] = (1.0, 2.0)             # sec(z), alt >= 30 deg
    # -- molecular abundances (ppmv) --------------------------------------- #
    co2: tuple[float, float] = (420.0, 435.0)
    # Upper edge is the fixed-production value (0.0429), which also covers the
    # +/-2 % intra-night perturbation of the variable envelope (0.042 * 1.02 =
    # 0.04284); the pure variable envelope would clip ~12 % of the o3 frames.
    o3: tuple[float, float] = (0.018, 0.0429)
    n2o: tuple[float, float] = (0.335, 0.345)
    co: tuple[float, float] = (0.088, 0.169)
    ch4: tuple[float, float] = (1.90, 1.98)
    o2: tuple[float, float] = (205800.0, 214200.0)
    # ``no`` is sampled log-uniformly, so its linear width is huge on purpose:
    # realistic values collapse onto ~0 after linear min-max scaling. The lower
    # edge also brackets the fixed production value (1.1e-19), which the
    # variable-molecule envelope (min 1.5e-19) would otherwise exclude.
    no: tuple[float, float] = (1.0e-19, 1.2e-5)
    so2: tuple[float, float] = (9.8e-5, 1.02e-4)
    # The upper edges bracket the fixed production values (no2 1.0e-4,
    # hno3 5.6e-4) that the variable-molecule envelope alone would exclude.
    no2: tuple[float, float] = (2.57e-5, 1.0e-4)
    nh3: tuple[float, float] = (9.8e-5, 1.02e-4)
    hno3: tuple[float, float] = (1.74e-5, 5.6e-4)

    @field_validator("*", mode="after")
    @classmethod
    def _check_ordered_pair(
        cls, value: tuple[float, float], info: ValidationInfo
    ) -> tuple[float, float]:
        """Reject non-finite or inverted bounds (``min == max`` stays legal)."""
        low, high = value
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(
                f"{info.field_name} bounds must be finite, got {value}"
            )
        if low > high:
            raise ValueError(
                f"{info.field_name} bounds must satisfy min <= max, got {value}"
            )
        return value

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        """Column names in canonical (definition) order."""
        return tuple(cls.model_fields)

    def to_mapping(self) -> dict[str, tuple[float, float]]:
        """The bounds as a plain ``{column: (min, max)}`` mapping."""
        return {name: tuple(getattr(self, name)) for name in self.column_names()}


# The schema derived from the bound declarations (canonical order preserved).
PHYSICAL_COLUMNS: tuple[str, ...] = ParameterBounds.column_names()
TARGET_COLUMNS: tuple[str, ...] = tuple(
    c for c in PHYSICAL_COLUMNS if c != TIME_COLUMN
)
