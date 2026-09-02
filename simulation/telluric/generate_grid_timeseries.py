#!/usr/bin/env python3
"""
Generate telluric transmissions over a full night (time series) in parallel
using dawgz + SLURM.

Adapted from Malavika Vasist's ``17_generate_telfit_timeseries.py`` in the
ai-tfm-tellurics project, ported to the Tellurics repo conventions:

- shared spectral utilities from ``simulation.utils.spectral``
- output under ``data/telluric_timeseries/``
- TelFit source resolved from the repo's vendored ``TelFit/src``

Difference from ``generate_grid.py``
------------------------------------
``generate_grid.py`` draws independent atmospheric conditions via Latin
Hypercube sampling: each sample is one unrelated telluric snapshot.

This script instead consumes a *condition series* (``telfit_condition_series.h5``)
describing how the atmosphere evolves through an observing night. Each flattened
sample corresponds to one (series, frame) pair, so the output is a set of
telluric transmissions that vary smoothly across the frames of each night.

The TelFit parameters therefore come from the condition file rather than a
sampler. Frame-wise quantities (pressure, temperature, humidity, zenith angle)
vary per frame; molecular abundances are constant per series.

Method
------
Native TelFit/LBLRTM is generated first (no wavegrid, no resolution), then
convolved to the target resolving power in log-lambda space and resampled onto
the shared constant-R output grid. This matches ``generate_grid.py``.

Usage
-----
    # Submit all condition frames in parallel on SLURM + an aggregation job:
    python simulation/telluric/generate_grid_timeseries.py --parallel \
        --conditions <path>/telfit_condition_series.h5 \
        --batch-size 200

    # Generate a single batch under SLURM:
    python simulation/telluric/generate_grid_timeseries.py \
        --conditions <path>/telfit_condition_series.h5 \
        --array-index 5 --batch-size 200

    # Aggregate existing chunks:
    python simulation/telluric/generate_grid_timeseries.py \
        --conditions <path>/telfit_condition_series.h5 --aggregate
"""

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

# Make the repo root importable so `simulation` is a package when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from simulation.utils.spectral import build_constant_R_grid, convolve_and_resample
from simulation.utils.site_parameters import (
    read_condition_layout,
    load_condition_sample,
    MOLECULES,
)
from simulation.utils.running_hpc import create_slurm_workspace


# === Configuration ===
REPO_ROOT = _REPO_ROOT

DEFAULT_CONDITIONS_H5 = (
    REPO_ROOT / "site_parameter_evolution" / "telfit_condition_series.h5"
)
DEFAULT_OUTDIR = REPO_ROOT / "data" / "telluric_timeseries"

# TelFit source (vendored software)
TELFIT_SRC = REPO_ROOT / "TelFit" / "src"

# Wavelength range (nm) and target resolution (must match Phoenix pool)
WAVESTART_NM = 800.0
WAVEEND_NM = 950.0
TARGET_R = 100_000
SAMPLES_PER_RESEL = 3.0
WAVE_PADDING_NM = 0.1  


# MOLECULES is imported from simulation.utils.site_parameters

LABEL_COLUMNS = (
    "flat_index",
    "series_index",
    "frame_index",
    "time_hours",
    "pressure",
    "temperature",
    "humidity",
    "angle",
    "airmass",
    *MOLECULES,
)


def get_output_wavegrid():
    """Build the constant-R output wavelength grid (nm), shared with Phoenix."""
    return build_constant_R_grid(
        WAVESTART_NM, WAVEEND_NM, TARGET_R, SAMPLES_PER_RESEL,
    ).astype(np.float64)


def generate_batch(
    array_index: int,
    n_samples: int,
    batch_size: int,
    conditions_h5: Path,
    outdir: Path,
    seed: int = 42,
):
    """Generate a single batch of telluric models and save to an HDF5 chunk."""
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    warnings.filterwarnings("ignore", message="PYSYN_CDBS is undefined.*")
    warnings.filterwarnings("ignore", message="Extinction files not found.*")

    sys.path.insert(0, str(TELFIT_SRC))
    from telfit import Modeler

    chunks_dir = outdir / "chunks"
    outdir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = chunks_dir / f"chunk_{array_index:06d}.h5"
    if chunk_path.exists():
        print(f"Chunk {array_index} already exists, skipping.")
        return

    workdir = create_slurm_workspace()

    n_series, n_frames, observatory = read_condition_layout(conditions_h5)
    total_available = n_series * n_frames
    if n_samples > total_available:
        raise ValueError(
            f"Requested n_samples={n_samples}, but conditions contain only "
            f"{total_available} frames ({n_series} x {n_frames})."
        )

    start = array_index * batch_size
    end = min(start + batch_size, n_samples)
    if start >= n_samples:
        print(f"Chunk {array_index}: no samples (start={start} >= n_samples={n_samples})")
        return

    flat_indices = np.arange(start, end, dtype=np.int64)
    print(f"Chunk {array_index}: samples [{start}:{end}] ({len(flat_indices)} models)")

    # Shared constant-R output grid.
    wavegrid_nm = get_output_wavegrid()

    # Native TelFit over a padded range for clean convolution edges.
    lowfreq = 1e7 / (WAVEEND_NM + WAVE_PADDING_NM)
    highfreq = 1e7 / (WAVESTART_NM - WAVE_PADDING_NM)

    modeler = Modeler(print_lblrtm_output=False, debug=False)
    transmission_list = []
    labels_list = []
    failed_indices = []

    with h5py.File(conditions_h5, "r") as conditions:
        for i, flat_index in enumerate(flat_indices):
            params, label = load_condition_sample(conditions, int(flat_index), n_frames)

            if i % 10 == 0:
                print(f"  Chunk {array_index}: {i+1}/{len(flat_indices)}")

            try:
                model = modeler.MakeModel(
                    lowfreq=lowfreq,
                    highfreq=highfreq,
                    lat=observatory["lat"],
                    alt=observatory["alt"],
                    workdir=str(workdir),
                    vac2air=False,
                    libfile="",
                    save=False,
                    **params,
                )

                _, transmission = convolve_and_resample(
                    np.asarray(model.x, dtype=np.float64),
                    np.asarray(model.y, dtype=np.float64),
                    target_R=TARGET_R,
                    target_wave=wavegrid_nm,
                )

                transmission_list.append(transmission.astype(np.float32))
                labels_list.append(label.astype(np.float32))

            except Exception as e:
                print(f"  FAILED at sample {int(flat_index)}: {type(e).__name__}: {e}")
                print(f"    params: {params}")
                failed_indices.append(int(flat_index))
                continue

    if not transmission_list:
        print(f"Chunk {array_index}: no models generated!")
        return

    transmission = np.stack(transmission_list)
    labels = np.stack(labels_list)

    with h5py.File(chunk_path, "w") as hf:
        hf.create_dataset("wavelength", data=wavegrid_nm.astype(np.float32))
        hf.create_dataset(
            "transmission",
            data=transmission,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        hf.create_dataset("labels", data=labels)
        hf.create_dataset("failed_indices", data=np.asarray(failed_indices, dtype=np.int64))

        hf["wavelength"].attrs["unit"] = "nm"
        hf["labels"].attrs["columns"] = LABEL_COLUMNS
        hf.attrs["conditions_h5"] = str(conditions_h5)
        hf.attrs["target_resolution"] = TARGET_R
        hf.attrs["samples_per_resolution_element"] = SAMPLES_PER_RESEL
        hf.attrs["n_frames_per_series"] = n_frames

    print(
        f"Chunk {array_index} done: {len(transmission_list)} models succeeded -> "
        f"{chunk_path}, {len(failed_indices)} failed"
    )


def aggregate(outdir: Path):
    """Combine all chunk HDF5 files into telluric_timeseries.h5 (streamed)."""
    chunks_dir = outdir / "chunks"
    final_path = outdir / "telluric_timeseries.h5"
    chunk_files = sorted(chunks_dir.glob("chunk_*.h5"))

    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return

    print(f"Aggregating {len(chunk_files)} chunk files...")

    wavelength = None
    label_columns = None
    total_models = 0
    total_failed = 0
    metadata = {}

    for cf in chunk_files:
        with h5py.File(cf, "r") as hf:
            wave = hf["wavelength"][:]
            if wavelength is None:
                wavelength = wave
                label_columns = tuple(hf["labels"].attrs["columns"])
                metadata = dict(hf.attrs)
            elif wave.shape != wavelength.shape or not np.allclose(
                wave, wavelength, rtol=0, atol=1e-6
            ):
                raise ValueError(f"Wavelength grid mismatch in {cf}")

            total_models += int(hf["transmission"].shape[0])
            total_failed += int(hf["failed_indices"].shape[0])

    n_wave = int(wavelength.shape[0])
    n_labels = len(label_columns)

    with h5py.File(final_path, "w") as out:
        out.create_dataset("wavelength", data=wavelength)
        out["wavelength"].attrs["unit"] = "nm"

        ds_transmission = out.create_dataset(
            "transmission",
            shape=(total_models, n_wave),
            dtype=np.float32,
            chunks=(1, n_wave),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        ds_labels = out.create_dataset(
            "labels", shape=(total_models, n_labels), dtype=np.float32,
        )
        ds_labels.attrs["columns"] = label_columns
        ds_failed = out.create_dataset(
            "failed_indices", shape=(total_failed,), dtype=np.int64,
        )

        for key, value in metadata.items():
            out.attrs[key] = value

        model_pos = 0
        fail_pos = 0
        for cf in chunk_files:
            with h5py.File(cf, "r") as hf:
                n = hf["transmission"].shape[0]
                nf = hf["failed_indices"].shape[0]

                ds_transmission[model_pos:model_pos + n] = hf["transmission"][:]
                ds_labels[model_pos:model_pos + n] = hf["labels"][:]
                if nf:
                    ds_failed[fail_pos:fail_pos + nf] = hf["failed_indices"][:]

                model_pos += n
                fail_pos += nf

    print(f"Total models: {total_models}")
    print(f"Failed models: {total_failed}")
    print(f"Wavelength points: {n_wave}")
    print(f"Parameters: {label_columns}")
    print(f"\nSaved: {final_path}")
    print(f"  wavelength:   {wavelength.shape}")
    print(f"  transmission: ({total_models}, {n_wave})")
    print(f"  labels:       ({total_models}, {n_labels})")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate telluric time-series transmissions")
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS_H5,
                        help="telfit_condition_series.h5 (per-night atmospheric conditions)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR,
                        help="Output directory for chunks and final HDF5")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Max flattened condition frames to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="TelFit calls per chunk/job")
    parser.add_argument("--seed", type=int, default=42,
                        help="Retained for CLI compatibility; conditions are deterministic")
    parser.add_argument("--array-index", type=int, default=None,
                        help="Generate a single chunk (for manual runs)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate chunk files into final HDF5")
    parser.add_argument("--parallel", action="store_true",
                        help="Submit SLURM jobs via dawgz")
    parser.add_argument("--conda-env", type=str, default="tellurics",
                        help="Conda environment for SLURM jobs")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per SLURM job")
    parser.add_argument("--ram", type=str, default="1GB", help="RAM per SLURM job")
    parser.add_argument("--aggregate-ram", type=str, default="4GB",
                        help="RAM for aggregation SLURM job")
    parser.add_argument("--time", type=str, default="10:00:00",
                        help="Wall time per SLURM job")
    return parser.parse_args()


def main():
    args = parse_args()

    conditions_h5 = args.conditions.expanduser().resolve()
    outdir = args.output_dir.expanduser().resolve()

    n_series, n_frames, _ = read_condition_layout(conditions_h5)
    total_available = n_series * n_frames
    n_samples = total_available if args.n_samples is None else int(args.n_samples)

    if n_samples < 1 or n_samples > total_available:
        raise ValueError(f"--n-samples must be in [1, {total_available}], got {n_samples}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    n_jobs = int(np.ceil(n_samples / args.batch_size))
    wavegrid = get_output_wavegrid()

    print("=" * 88)
    print("TELFIT TIME-SERIES GENERATOR")
    print("=" * 88)
    print(f"Conditions:                  {conditions_h5}")
    print(f"Series / frames:             {n_series} / {n_frames}")
    print(f"Available TelFit calls:      {total_available}")
    print(f"Calls requested:             {n_samples}")
    print(f"Batch size / jobs:           {args.batch_size} / {n_jobs}")
    print(f"Wavelength range:            {WAVESTART_NM:.1f}-{WAVEEND_NM:.1f} nm")
    print(f"Target resolving power:      {TARGET_R}")
    print(f"Output wavelength points:    {len(wavegrid)}")
    print(f"Output directory:            {outdir}")

    # === Parallel mode: submit generation + aggregation with dependency ===
    if args.parallel:
        from dawgz import job, after, schedule

        @job(array=n_jobs, cpus=args.cpus, ram=args.ram, time=args.time)
        def telluric_job(array_index: int):
            generate_batch(
                array_index=array_index,
                n_samples=n_samples,
                batch_size=args.batch_size,
                conditions_h5=conditions_h5,
                outdir=outdir,
                seed=args.seed,
            )

        @after(telluric_job)
        @job(cpus=1, ram=args.aggregate_ram, time="1:00:00")
        def aggregate_job():
            aggregate(outdir)

        schedule(
            aggregate_job,
            name="TelluricTimeseries",
            backend="slurm",
            env=["source ~/.bashrc", f"conda activate {args.conda_env}"],
        )
        print(f"Submitted {n_jobs} generation jobs + 1 aggregation job (with dependency)")
        return

    # === Non-parallel modes ===
    if args.aggregate:
        aggregate(outdir)
        return

    if args.array_index is not None:
        generate_batch(
            array_index=args.array_index,
            n_samples=n_samples,
            batch_size=args.batch_size,
            conditions_h5=conditions_h5,
            outdir=outdir,
            seed=args.seed,
        )
        return

    # Sequential: generate all then aggregate (still needs SLURM_TMPDIR).
    print(f"Running {n_jobs} batches sequentially...")
    for i in range(n_jobs):
        generate_batch(
            array_index=i,
            n_samples=n_samples,
            batch_size=args.batch_size,
            conditions_h5=conditions_h5,
            outdir=outdir,
            seed=args.seed,
        )

    print("\nAll batches done. Aggregating...")
    aggregate(outdir)


if __name__ == "__main__":
    main()
