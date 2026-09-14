"""Whole-night dataset: one sample = one night of exposures.

Reads the night-major HDF5 built once by ``tests/testing_dataset_reshape.ipynb``
(``transmission`` / ``labels`` / ``wavelength``) and turns each night into the
inputs the :class:`~tellurics.models.night.NeuralTelluricPredictor` consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tellurics.data.stellar import StellarPool

# --------------------------------------------------------------------------- #
# Column book-keeping (from the labels 'columns' attr of the night-major HDF5)
# --------------------------------------------------------------------------- #
TIME_COLUMN = "time_hours"
METADATA_COLUMNS = ("pressure", "temperature", "humidity")

# Full physical parameter set in canonical (file) order.
PHYSICAL_COLUMNS = (
    "time_hours", "pressure", "temperature", "humidity", "angle", "airmass",
    "co2", "o3", "n2o", "co", "ch4", "o2", "no", "so2", "no2", "nh3", "hno3",
)

# Columns regressed by the estimator (everything except the time that is fed
# to the TimeEncoder and the flat/series/frame indices).
TARGET_COLUMNS = tuple(c for c in PHYSICAL_COLUMNS if c != TIME_COLUMN)


def _decode_columns(columns) -> list[str]:
    """Decode an h5py 'columns' attribute into a list of strings."""
    out: list[str] = []
    for c in columns:
        if isinstance(c, bytes):
            out.append(c.decode("utf-8"))
        else:
            out.append(str(c))
    return out


class TelluricTimeseriesDataset(Dataset[dict[str, object]]):
    """Whole-night samples from the night-major HDF5.

    Each sample is one night ``(T, N)``. The observed spectrum is
    ``observed = transmission * S`` with ``S`` the stellar spectrum assigned
    to that night (fixed), which is also returned so the estimator encodes it.

    Returns a dict with:
        observed:   (T, N)   X = T_tell * S
        transmission: (T, N) ground-truth T_tell
        stellar:    (N,)     the S used to build observed
        theta: {
            "time":     (T,)
            "metadata": (T, 3)   pressure, temperature, humidity
            "params":   (T, P)   the P target params
        }
        night_id:   int      index in [0, n_nights)
        star_index: int      index into the stellar pool
    """

    metadata_columns = METADATA_COLUMNS
    time_column = TIME_COLUMN
    target_columns = TARGET_COLUMNS

    def __init__(
        self,
        night_ids: list[int],
        h5_path: str | Path,
        pool: StellarPool,
        star_assignment: np.ndarray,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.night_ids = list(night_ids)
        self.pool = pool
        self.star_assignment = np.asarray(star_assignment)

        # Resolve the column layout from the file's stored 'columns' attr.
        # Single source of truth, so the indices can never disagree with the
        # data they index into.
        self._label_columns = self._read_label_columns()
        for expected in (*self.metadata_columns, self.time_column,
                         *self.target_columns):
            if expected not in self._label_columns:
                raise KeyError(
                    f"expected label column '{expected}' not in file columns "
                    f"{self._label_columns}"
                )
        idx = {c: self._label_columns.index(c) for c in self._label_columns}
        self._time_idx = idx[self.time_column]
        self._metadata_idx = [idx[c] for c in self.metadata_columns]
        self._target_idx = [idx[c] for c in self.target_columns]

    def _open(self):
        import h5py

        return h5py.File(self.h5_path, "r")

    def _read_label_columns(self) -> list[str]:
        with self._open() as f:
            return _decode_columns(f["labels"].attrs.get("columns", []))

    def __len__(self) -> int:
        return len(self.night_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        night = self.night_ids[idx]
        star_idx = int(self.star_assignment[night])
        s, _ = self.pool[star_idx]                    # (N,) float32

        with self._open() as f:
            trans = np.asarray(f["transmission"][night], np.float32)   # (T, N)
            lab = np.asarray(f["labels"][night], np.float32)           # (T, P)

        s_t = s[None, :]                              # (1, N)
        observed = trans * s_t                        # (T, N)

        time = lab[:, self._time_idx]                                   # (T,)
        metadata = lab[:, self._metadata_idx]                           # (T,3)
        params = lab[:, self._target_idx]                               # (T,P)

        return {
            "observed": torch.from_numpy(observed),
            "transmission": torch.from_numpy(trans),
            "stellar": torch.from_numpy(s),
            "theta": {
                "time": torch.from_numpy(time),
                "metadata": torch.from_numpy(metadata),
                "params": torch.from_numpy(params),
            },
            "night_id": night,
            "star_index": star_idx,
        }
