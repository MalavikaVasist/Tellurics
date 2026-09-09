"""Whole-night telluric timeseries DataModule for the :class:`TelluricEstimator`.

Data layout (source ``data/telluric_templates.h5``)
---------------------------------------------------
The raw file is *flat*:

    transmission     (328500, 51556)  float32   T_tell(l)  (328500 == 4500 * 73)
    labels           (328500, 20)     float32   per-row parameters
    wavelength       (51556,)         float64   nm grid
    attrs: n_frames_per_series = 73, lowfreq_cm1 / highfreq_cm1, ...

Rows are ordered ``series-major``: the first 73 rows are night 0, the next 73
night 1, ... ``build_timeseries_h5`` reshapes this once into a *night-major*
HDF5 (``data/telluric_timeseries/telluric_templates.h5``):

    transmission     (4500, 73, 51556)   float32
    labels           (4500, 73, 20)      float32   + attr ``columns``
    wavelength       (51556,)            float64

What the loaders return
-----------------------
One sample = one whole night. The estimator is fed ``observed = T_tell * S``,
where a **single** stellar spectrum ``S`` (sampled, fixed per night, from the
``data/phoenix/convolved/*.fits`` pool) is used across the whole night.  The
same ``S`` is returned so the estimator can encode it and fuse it. ``theta`` is
a *dictionary* mirroring how ``TelluricEstimator`` consumes its inputs::

    theta = {
        "time":     (T,)          time_hours          ->  TimeEncoder
        "metadata": (T, 3)        pressure, temperature, humidity
                                                        ->  MetadataEncoder
        "params":   (T, 16)       the 16 target params ->  MSE target vs. pred
    }

``x`` (the observed night ``T*S``) and ``theta`` are grouped so that a batch
yields ``observed (B, T, N)``, ``stellar (B, N)`` and the ``theta`` dict with
``(B, T, ...)`` tensors -- exactly what the estimator + a param MSE loss need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from tellurics.data.night import split_night_indices
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Column book-keeping (from labels attrs 'columns' in telluric_templates.h5)
# --------------------------------------------------------------------------- #
INDEX_COLUMNS = ("flat_index", "series_index", "frame_index")
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


# --------------------------------------------------------------------------- #
# Wavelengths as a first-class object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Wavegrid:
    """The wavelength grid the transmission / stellar spectra live on.

    Attributes:
        wavelength: (N,) float64 grid in ``unit``.
        unit: physical unit of the grid (default 'nm').
    """

    wavelength: np.ndarray
    unit: str = "nm"

    def __post_init__(self) -> None:
        if self.wavelength.ndim != 1:
            raise ValueError("wavelength must be a 1-D array")
        # Sort-on-init guarantees monotonic interpolation afterwards.
        wl = np.asarray(self.wavelength, np.float64)
        if np.any(np.diff(wl) <= 0):
            order = np.argsort(wl)
            object.__setattr__(self, "wavelength", wl[order])
        else:
            object.__setattr__(self, "wavelength", wl)

    @property
    def n_wavelength(self) -> int:
        return self.wavelength.size

    @property
    def wavelength_min(self) -> float:
        return float(self.wavelength.min())

    @property
    def wavelength_max(self) -> float:
        return float(self.wavelength.max())

    def __len__(self) -> int:
        return self.n_wavelength

    def __getitem__(self, idx):
        return self.wavelength[idx]

    def as_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.wavelength)


# --------------------------------------------------------------------------- #
# Reshape the flat HDF5 into the night-major HDF5 (run once)
# --------------------------------------------------------------------------- #
def build_timeseries_h5(
    flat_h5: str | Path,
    out_h5: str | Path,
    n_frames: int | None = None,
) -> Wavegrid:
    """Reshape the flat transmission file into a night-major HDF5.

    Args:
        flat_h5: Raw file with top-level datasets 'transmission' (n_flat, N),
            'labels' (n_flat, P) with attr 'columns', and 'wavelength' (N,).
        out_h5: Output night-major file:
            'transmission' (n_series, T, N), 'labels' (n_series, T, P)
            with attr 'columns', 'wavelength' (N,).
        n_frames: exposures per night (auto-read from file attrs if None).

    Returns:
        The :class:`Wavegrid` read from the source file.
    """
    import h5py

    flat_h5 = Path(flat_h5)
    out_h5 = Path(out_h5)

    with h5py.File(flat_h5, "r") as f:
        trans = f["transmission"]
        n_flat, n_wave = trans.shape
        if n_frames is None:
            n_frames = int(f.attrs.get("n_frames_per_series", 73))
        n_series = n_flat // n_frames
        if n_flat % n_frames != 0:
            raise ValueError(
                f"n_flat={n_flat} not divisible by n_frames={n_frames}"
            )
        wl = np.asarray(f["wavelength"][:], np.float64)
        n_labels = f["labels"].shape[1]
        cols = _decode_columns(f["labels"].attrs.get("columns", []))
        unit = str(f["wavelength"].attrs.get("unit", "nm"))

    logger.info(
        f"Reshaping {flat_h5}: {n_series} nights x {n_frames} frames "
        f"x {n_wave} samples -> {out_h5}"
    )
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as o:
        o.create_dataset(
            "transmission", shape=(n_series, n_frames, n_wave),
            dtype=np.float32, chunks=(1, n_frames, min(256, n_wave)),
            compression="gzip", compression_opts=4,
        )
        o.create_dataset(
            "labels", shape=(n_series, n_frames, n_labels),
            dtype=np.float32, chunks=(1, n_frames, n_labels),
        )
        o.create_dataset("wavelength", data=wl)
        o["wavelength"].attrs["unit"] = unit
        if cols:
            o["labels"].attrs["columns"] = cols
        with h5py.File(flat_h5, "r") as src:
            for s in range(n_series):
                sl = slice(s * n_frames, (s + 1) * n_frames)
                o["transmission"][s] = src["transmission"][sl]
                o["labels"][s] = src["labels"][sl]
                if s % 500 == 0:
                    logger.info(f"  series {s}/{n_series}")
    logger.info(f"Done -> {out_h5}")
    return Wavegrid(wavelength=wl, unit=unit)


def _decode_columns(columns) -> list[str]:
    """Decode an h5py 'columns' attribute into a list of strings."""
    out: list[str] = []
    for c in columns:
        if isinstance(c, bytes):
            out.append(c.decode("utf-8"))
        else:
            out.append(str(c))
    return out


# --------------------------------------------------------------------------- #
# Stellar pool (the ~20 Phoenix spectra, interpolated onto the wavegrid)
# --------------------------------------------------------------------------- #
class StellarPool:
    """All stellar spectra from ``data/phoenix/convolved/*.fits``.

    Each spectrum is interpolated once onto the transmission wavelength grid so
    that ``observed = T_tell * S`` is grid-consistent. A single spectrum is
    used per whole night.
    """

    _cache: tuple[tuple[str, ...], np.ndarray] | None = None
    _cache_key: tuple[object, ...] | None = None

    def __init__(
        self,
        directory: str | Path,
        wavegrid: Wavegrid,
        star_files: list[str] | None = None,
    ) -> None:
        directory = Path(directory)
        cache_key = (str(directory.resolve()), wavegrid.wavelength.tobytes())
        if StellarPool._cache is not None and StellarPool._cache_key == cache_key:
            names, spectra = StellarPool._cache
        else:
            files = sorted(directory.glob("*.fits"))
            if not files:
                raise FileNotFoundError(
                    f"No *.fits stellar spectra in {directory}"
                )
            if star_files is not None:
                files = [directory / s for s in star_files]
            names, spectra = self._load(files, wavegrid.wavelength)
            StellarPool._cache_key = cache_key
            StellarPool._cache = (names, spectra)

        self.directory = directory
        self.wavegrid = wavegrid
        self.names = names
        self.spectra = spectra                      # (n_stars, N) float32

    @staticmethod
    def _load(files: list[Path], wl_target: np.ndarray):
        """Load each FITS (WAVE + FLUX) and interpolate onto ``wl_target``."""
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover
            raise ImportError("astropy is required to read Phoenix FITS") from exc

        names: list[str] = []
        spectra: list[np.ndarray] = []
        for path in files:
            with fits.open(path) as hdul:
                data = hdul["SPECTRUM"].data
                if "WAVE_MICRON" in data.names:
                    wl_s = np.asarray(data["WAVE_MICRON"], np.float64) * 1000.0
                elif "WAVE_NANO" in data.names:
                    wl_s = np.asarray(data["WAVE_NANO"], np.float64)
                else:
                    raise KeyError(f"No wavelength column in {data.names}")
                flux = np.asarray(data["FLUX"], np.float64)
            order = np.argsort(wl_s)
            wl_s, flux = wl_s[order], flux[order]
            # Normalise so the *mean* stellar level is 1 (keeps T*S well scaled).
            flux = flux / np.interp(wl_target, wl_s, flux).mean()
            spectra.append(
                np.interp(wl_target, wl_s, flux).astype(np.float32)
            )
            names.append(path.name)
        return tuple(names), np.stack(spectra)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        """Return ``(spectrum (N,) float32, name)`` for stellar index ``idx``."""
        return self.spectra[idx], self.names[idx]


def assign_stellar_per_night(
    n_nights: int,
    pool: StellarPool,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic, fixed-per-night star assignment.

    Returns an ``(n_nights,)`` int array of stellar-pool indices. The same
    star is used for every exposure of a night and stays the same across
    epochs (no augmentation), so the mapping is reproducible and auditable.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(pool), size=n_nights)


def save_stellar_assignment(
    assignment: np.ndarray,
    pool: StellarPool,
    splits: dict[str, list[int]] | None,
    out_csv: str | Path,
) -> None:
    """Write ``night_id, split, star_index, star_file`` so the used stars are
    auditable per night and per split."""
    import csv

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_of = {}
    if splits is not None:
        for split, ids in splits.items():
            for nid in ids:
                split_of[int(nid)] = split
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["night_id", "split", "star_index", "star_file"])
        for night in range(assignment.size):
            sidx = int(assignment[night])
            writer.writerow(
                [night, split_of.get(int(night), ""), sidx, pool.names[sidx]]
            )


# --------------------------------------------------------------------------- #
# Dataset: one sample = one whole night
# --------------------------------------------------------------------------- #
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
        h5_path: str | Path,
        night_ids: list[int],
        pool: StellarPool,
        star_assignment: np.ndarray,
        label_columns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.night_ids = list(night_ids)
        self.pool = pool
        self.star_assignment = np.asarray(star_assignment)

        # Resolve column indices from the stored 'columns' attr.
        cols = self._read_label_columns()
        self._label_columns = cols if label_columns is None else list(label_columns)
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

        # Metadata sanity: guard against duplicate assignment lengths.
        if self.star_assignment.size < (max(self.night_ids, default=-1) + 1):
            raise ValueError(
                "star_assignment too short for the requested night ids"
            )

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


# --------------------------------------------------------------------------- #
# Lightning DataModule
# --------------------------------------------------------------------------- #
class TelluricDataModule(pl.LightningDataModule):
    """Whole-night DataModule for training the :class:`TelluricEstimator`.

    Expects the night-major HDF5 produced by :func:`build_timeseries_h5` (the
    raw flat ``telluric_templates.h5`` is reshaped separately). Splits are made
    at the *night* level so no exposure of a night leaks across splits. Each
    night is assigned a fixed stellar spectrum (seeded, reproducible), and the
    night -> star mapping (with its split) is saved to ``stellar_assignment.csv``
    for auditing what spectrum was used where.
    """

    def __init__(
        self,
        night_h5: str | Path = "data/telluric_timeseries/telluric_templates.h5",
        phoenix_dir: str | Path = "data/phoenix/convolved",
        batch_size: int = 4,
        num_workers: int = 2,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        assignment_csv: str | Path | None = None,
    ) -> None:
        """Initialize.

        Args:
            night_h5: Night-major HDF5 (see :func:`build_timeseries_h5`).
            phoenix_dir: Directory holding the stellar ``*.fits`` pool.
            batch_size: Batches of whole nights.
            num_workers: DataLoader workers.
            train_fraction, val_fraction: night-level split fractions
                (test = 1 - train - val).
            seed: RNG seed for both the night split and the stellar assignment.
            pin_memory: DataLoader pin_memory flag.
            assignment_csv: Where to write the night -> star audit file
                (default: next to ``night_h5`` as ``stellar_assignment.csv``).
        """
        super().__init__()
        self.save_hyperparameters()
        self.night_h5 = Path(night_h5)
        self.phoenix_dir = Path(phoenix_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        self.pin_memory = pin_memory
        self.assignment_csv = (
            Path(assignment_csv)
            if assignment_csv is not None
            else self.night_h5.parent / "stellar_assignment.csv"
        )

    # -- attributes set during setup ---------------------------------------- #
    wavegrid: Wavegrid | None = None
    stellar_pool: StellarPool | None = None
    stellar_assignment: np.ndarray | None = None      # (n_nights,) star indices

    def setup(self, stage: str | None = None) -> None:
        """Build splits, the stellar pool, and per-night star assignment."""
        if not self.night_h5.exists():
            raise FileNotFoundError(
                f"night-major HDF5 not found at {self.night_h5}. "
                "Build it once from the flat file with "
                "tellurics.data.datamodule.build_timeseries_h5(...)."
            )

        import h5py

        with h5py.File(self.night_h5, "r") as f:
            n_nights = f["transmission"].shape[0]
            n_frames = f["transmission"].shape[1]
            wl = np.asarray(f["wavelength"][:], np.float64)
            unit = str(f["wavelength"].attrs.get("unit", "nm"))
            cols = _decode_columns(f["labels"].attrs.get("columns", []))

        self.wavegrid = Wavegrid(wavelength=wl, unit=unit)
        self.stellar_pool = StellarPool(self.phoenix_dir, self.wavegrid)

        train_ids, val_ids, test_ids = split_night_indices(
            n_nights, self.train_fraction, self.val_fraction, self.seed
        )
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        # Fixed per-night stellar assignment across the whole file.
        self.stellar_assignment = assign_stellar_per_night(
            n_nights, self.stellar_pool, seed=self.seed
        )
        save_stellar_assignment(
            self.stellar_assignment, self.stellar_pool,
            {"train": train_ids, "val": val_ids, "test": test_ids},
            self.assignment_csv,
        )
        logger.info(
            f"{n_nights} nights (T={n_frames}) -> "
            f"train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}; "
            f"stellar audit -> {self.assignment_csv}"
        )

        kw = dict(
            h5_path=self.night_h5,
            pool=self.stellar_pool,
            star_assignment=self.stellar_assignment,
            label_columns=cols,
        )
        self.train_ds = TelluricTimeseriesDataset(train_ids, **kw)
        self.val_ds = TelluricTimeseriesDataset(val_ids, **kw)
        self.test_ds = TelluricTimeseriesDataset(test_ids, **kw)

    # -- helper the user asked for: which star was used, per night ---------- #
    def stellar_for_night(self, night_id: int) -> tuple[np.ndarray, str]:
        """Return ``(spectrum (N,), star_file)`` assigned to ``night_id``."""
        if self.stellar_assignment is None or self.stellar_pool is None:
            raise RuntimeError("call setup() first")
        return self.stellar_pool[int(self.stellar_assignment[night_id])]

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
