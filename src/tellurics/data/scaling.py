"""Deterministic [0, 1] min-max scaling of the per-exposure label table.

For every exposure the night-major HDF5 carries three groups of interest:

============  =================================================================
group         columns
============  =================================================================
``metadata``  ``METADATA_COLUMNS`` (pressure, temperature, humidity) -> the
              ``MetadataEncoder`` conditioning input
``params``    ``TARGET_COLUMNS`` (everything but ``time_hours``) -> the MSE
              targets regressed by the model
``time``      ``time_hours`` -> the ``TimeEncoder`` input
============  =================================================================

Each group is mapped onto ``[0, 1]`` with the explicit, config-declared
physical bounds (:class:`tellurics.configs.bounds.ParameterBounds`)::

    scaled  = (x - min) / (max - min)          # [min, max] -> [0, 1]
    x       = min + scaled * (max - min)       # inverse, back to physical units

so ``loss = MSE(param_pred, params)`` compares two tensors on the *same* scale
and a bounded (``sigmoid``) head can actually reach the targets.

Why explicit bounds instead of statistics read off the data?

* the mapping is **deterministic and split-independent** -- identical for
  train / validation / test, for a 100-night smoke run and for the full file,
  so a checkpoint's predictions can be inverted by rebuilding the scaler from
  the manifest alone;
* it does not drift when the dataset is regenerated: a min/max read off a
  finite sample is an order statistic and moves with the sample, a *physical*
  bound does not;
* it keeps the transform auditable -- the manifest states the range.

Degenerate bounds (``min == max``) are supported: such a column maps to a
constant ``0.0`` and inverts back to ``min``, never a division by zero. Values
outside the declared bounds are clipped to ``[0, 1]`` by :meth:`MinMax.transform`
(``clip=True``); :meth:`MinMax.count_out_of_bounds` reports how many there were
(the DataModule logs that once per setup).

This module deliberately knows nothing about the file layout or the config
models: it is driven by explicit column names and a ``{column: (min, max)}``
mapping, which keeps it importable from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = ["MinMax", "ParameterScaler"]


def _as_array(values: Sequence[float]) -> np.ndarray:
    """Bounds as a float64 array (float64 keeps the ratios exact)."""
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class MinMax:
    """Per-column ``[min, max]`` min-max scaling over the last tensor axis.

    ``columns`` names the columns; ``minimum`` / ``maximum`` are aligned to
    them. Both 1-D ``(T,)`` and 2-D ``(T, P)`` inputs are supported because
    every operation broadcasts against the last axis.

    Degenerate columns (``min == max``) scale to ``0.0`` and invert to ``min``.
    """

    columns: tuple[str, ...]
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape agreement and the ``min <= max`` ordering."""
        if len(self.columns) != self.minimum.size:
            raise ValueError(
                f"{len(self.columns)} columns but {self.minimum.size} minima"
            )
        if self.minimum.shape != self.maximum.shape:
            raise ValueError(
                f"min shape {self.minimum.shape} != max shape "
                f"{self.maximum.shape}"
            )
        if np.any(self.minimum > self.maximum):
            raise ValueError(
                f"inverted bounds for "
                f"{[c for c, lo, hi in zip(self.columns, self.minimum, self.maximum) if lo > hi]}"
            )

    @classmethod
    def from_bounds(
        cls, columns: Sequence[str], bounds: Mapping[str, tuple[float, float]]
    ) -> MinMax:
        """Build a group from a ``{column: (min, max)}`` mapping.

        Raises:
            KeyError: If a column of ``columns`` has no declared bound.
        """
        columns = tuple(columns)
        missing = [c for c in columns if c not in bounds]
        if missing:
            raise KeyError(f"no bounds declared for {missing}")
        return cls(
            columns=columns,
            minimum=_as_array([bounds[c][0] for c in columns]),
            maximum=_as_array([bounds[c][1] for c in columns]),
        )

    # -- derived quantities ------------------------------------------------- #
    @property
    def span(self) -> np.ndarray:
        """``max - min``, with a safe ``1.0`` wherever the span is zero."""
        return np.where(
            self.maximum > self.minimum, self.maximum - self.minimum, 1.0
        )

    @property
    def degenerate(self) -> np.ndarray:
        """Boolean mask of columns with ``min == max`` (constant over the grid)."""
        return self.maximum <= self.minimum

    # -- transforms --------------------------------------------------------- #
    def _check_axis(self, values: np.ndarray) -> np.ndarray:
        """Validate that the array is explicitly shaped over :attr:`columns`.

        Multi-column groups require at least two dimensions, i.e. ``(..., P)``
        with ``P == len(columns)``. A bare ``(P,)`` array is refused even when
        the length happens to match: it is indistinguishable from ``T``
        samples, so a frame axis would be silently compared column-by-column.
        A single-column group (e.g. the time group) accepts any shape, since
        its one bound broadcasts correctly.
        """
        values = np.asarray(values)
        n = len(self.columns)
        if n > 1 and (values.ndim < 2 or values.shape[-1] != n):
            raise ValueError(
                f"expected a last axis of {n} ({', '.join(self.columns)}) with "
                f"at least two dimensions, got shape {values.shape}"
            )
        return values

    def transform(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        """Map physical values to ``[0, 1]``.

        Args:
            values: Array whose last axis is aligned with :attr:`columns`.
            clip: Clamp the result to ``[0, 1]`` (default). Values outside the
                declared bounds would otherwise land outside ``[0, 1]``, where
                a ``sigmoid`` head can never reach them. Degenerate columns are
                always exactly ``0.0``.

        Returns:
            The scaled values, in the *input* dtype (so float32 labels stay
            float32 and match the rest of the model).
        """
        values = self._check_axis(values)
        scaled = (values.astype(np.float64, copy=False) - self.minimum) / self.span
        # np.where (not boolean indexing) so that a (1,)-wide group also works
        # for 1-D (T,) inputs.
        scaled = np.where(self.degenerate, 0.0, scaled)
        if clip:
            scaled = np.clip(scaled, 0.0, 1.0)
        return scaled.astype(values.dtype, copy=False)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Map ``[0, 1]`` values back to physical units.

        The exact inverse of :meth:`transform` for in-range, non-degenerate
        columns; degenerate columns return the constant ``min``.
        """
        values = self._check_axis(values)
        physical = values.astype(np.float64, copy=False) * self.span + self.minimum
        physical = np.where(self.degenerate, self.minimum, physical)
        return physical.astype(values.dtype, copy=False)

    def out_of_bounds_per_column(
        self, values: np.ndarray, rtol: float = 1e-6
    ) -> np.ndarray:
        """Per-column number of entries outside the declared ``[min, max]``.

        The count is aligned with :attr:`columns`, which makes the report
        actionable (it names the column whose bound needs revisiting). ``rtol``
        is a small *relative* slack that absorbs storage rounding: the labels
        are float32, so a value stored exactly at a bound can come back a hair
        above the float64 bound (e.g. a declared ``5.6e-4``). Such a value is
        not out of range and must not be reported or distorted.
        """
        values = self._check_axis(values)
        slack = rtol * np.maximum(
            self.maximum - self.minimum, np.abs(self.maximum)
        )
        outside = (values < self.minimum - slack) | (values > self.maximum + slack)
        return np.asarray(outside.sum(axis=0)).astype(int)

    def count_out_of_bounds(self, values: np.ndarray, rtol: float = 1e-6) -> int:
        """Total number of entries outside the declared ``[min, max]`` range."""
        return int(self.out_of_bounds_per_column(values, rtol).sum())

    def summary(self) -> dict[str, list[float]]:
        """``{column: [min, max]}`` for logging / auditing."""
        return {
            c: [float(lo), float(hi)]
            for c, lo, hi in zip(self.columns, self.minimum, self.maximum)
        }


@dataclass(frozen=True, eq=False)
class ParameterScaler:
    """The three optional :class:`MinMax` groups of one dataset.

    Each group is ``None`` when the corresponding ``data.scale_*`` flag is off,
    in which case its transform is the identity:

    ====================  ====================  ==============================
    field                 config flag           transforms
    ====================  ====================  ==============================
    ``metadata``          ``data.scale_metadata``  ``transform_metadata``
    ``params``            ``data.scale_params``    ``transform_params`` /
                                                   ``inverse_params``
    ``time``              ``data.scale_time``      ``transform_time``
    ====================  ====================  ==============================
    """

    metadata: MinMax | None = field(default=None)
    params: MinMax | None = field(default=None)
    time: MinMax | None = field(default=None)

    @classmethod
    def from_bounds(
        cls,
        bounds: Mapping[str, tuple[float, float]],
        *,
        metadata_columns: Sequence[str] = (),
        target_columns: Sequence[str] = (),
        time_column: str | None = None,
        scale_metadata: bool = True,
        scale_params: bool = True,
        scale_time: bool = True,
    ) -> ParameterScaler:
        """Assemble the scaler described by a ``data:`` section.

        Args:
            bounds: ``{column: (min, max)}`` declared physical bounds.
            metadata_columns: Columns fed to the ``MetadataEncoder``.
            target_columns: Columns regressed by the model (MSE targets).
            time_column: Column fed to the ``TimeEncoder`` (e.g. time_hours).
            scale_metadata: Enable the metadata group.
            scale_params: Enable the target-parameter group.
            scale_time: Enable the time group.

        Returns:
            A :class:`ParameterScaler`; groups that are disabled (or have no
            columns) are ``None``.
        """
        return cls(
            metadata=(
                MinMax.from_bounds(metadata_columns, bounds)
                if scale_metadata and len(metadata_columns)
                else None
            ),
            params=(
                MinMax.from_bounds(target_columns, bounds)
                if scale_params and len(target_columns)
                else None
            ),
            time=(
                MinMax.from_bounds((time_column,), bounds)
                if scale_time and time_column
                else None
            ),
        )

    # -- flags -------------------------------------------------------------- #
    @property
    def scale_metadata(self) -> bool:
        """Whether metadata is scaled."""
        return self.metadata is not None

    @property
    def scale_params(self) -> bool:
        """Whether the target parameters are scaled."""
        return self.params is not None

    @property
    def scale_time(self) -> bool:
        """Whether ``time_hours`` is scaled."""
        return self.time is not None

    # -- forward transforms ------------------------------------------------- #
    def transform_metadata(self, values: np.ndarray) -> np.ndarray:
        """Scale metadata (``(T, P)`` or ``(T,)``); identity when disabled."""
        return values if self.metadata is None else self.metadata.transform(values)

    def transform_params(self, values: np.ndarray) -> np.ndarray:
        """Scale the targets (``(T, P)``); identity when disabled."""
        return values if self.params is None else self.params.transform(values)

    def transform_time(self, values: np.ndarray) -> np.ndarray:
        """Scale ``time_hours`` (``(T,)``); identity when disabled."""
        return values if self.time is None else self.time.transform(values)

    # -- inverse (predictions back to physical units) ----------------------- #
    def inverse_params(self, values: np.ndarray) -> np.ndarray:
        """Map normalized predictions back to physical units (numpy).

        With ``data.scale_params: false`` this is the identity.

        Example:
            >>> physical = scaler.inverse_params(pred.detach().cpu().numpy())
        """
        return values if self.params is None else self.params.inverse(values)

    def inverse_params_torch(self, values: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        """Map normalized predictions back to physical units (torch).

        Keeps the tensor on its device/dtype so it can be used directly in a
        Lightning ``predict_step`` or a downstream pipeline. With
        ``data.scale_params: false`` this is the identity.
        """
        if self.params is None:
            return values

        import torch  # local import: keep the module torch-free for numpy-only use

        low = torch.as_tensor(
            self.params.minimum, dtype=values.dtype, device=values.device
        )
        high = torch.as_tensor(
            self.params.maximum, dtype=values.dtype, device=values.device
        )
        span = torch.where(high > low, high - low, torch.ones_like(low))
        physical = values * span + low
        # Constant columns (min == max) invert to their constant value.
        return torch.where(low < high, physical, low.expand_as(physical))

    def summary(self) -> dict[str, dict[str, list[float]]]:
        """``{group: {column: [min, max]}}`` for logging / auditing."""
        groups = {"metadata": self.metadata, "params": self.params, "time": self.time}
        return {
            name: group.summary()
            for name, group in groups.items()
            if group is not None
        }
