# Simulation

Data-generation pipeline for training the tellurics NN. This is self-contained
and separate from the `tellurics` ML package (`src/tellurics/`). Shared spectral
utilities (convolution, constant-R grids) live in `simulation/utils/spectral.py`.

## Pipeline

```
1. PHOENIX stellar spectra      simulation/phoenix/download_pool.py
   download R~500k -> convolve to R=100k -> data/phoenix/convolved/

2. TelFit telluric transmissions
   a) independent snapshots (LHS)   simulation/telluric/generate_grid.py
      native LBLRTM -> convolve to R=100k -> data/telluric/telluric_templates.h5
   b) full-night time series         simulation/telluric/generate_grid_timeseries.py
      reads a condition series -> data/telluric_timeseries/telluric_timeseries.h5

3. Combine into observations     simulation/combine/make_observations.py
   stellar x telluric -> data/observations/observations.h5
```

The NN is trained on `observed` (input) to recover `telluric` (target).

## Resolution consistency

All three stages use the same constant-R grid (R=100,000, 3 samples per
resolution element, 0.8-0.95 um), built by `build_constant_R_grid` in
`simulation.utils.spectral`. Both stellar and telluric spectra are properly
convolved with a Gaussian instrumental profile (FWHM = 1/R in ln-lambda)
before being resampled onto this grid.

## Usage

```bash
# 1. Download and convolve the PHOENIX pool (20 stars)
python simulation/phoenix/download_pool.py

# 2. Generate telluric transmissions (needs TelFit + SLURM for full grid)
python simulation/telluric/generate_grid.py --array-index 0 --batch-size 200

# 3. Combine into observations
python simulation/combine/make_observations.py --normalize
```

## Directory layout

```
simulation/
├── utils/
│   ├── spectral.py            # convolution + constant-R grid helpers
│   ├── site_parameters.py     # load atmospheric condition series
│   └── running_hpc.py         # SLURM workspace management
├── site_parameters/
│   └── telfit_condition_series.h5   # per-night atmospheric conditions
├── phoenix/
│   ├── configs/
│   │   └── phoenix_pool.csv   # stellar pool definition
│   └── download_pool.py       # PHOENIX download + convolution
├── telluric/
│   ├── generate_grid.py            # independent snapshots (LHS sampling)
│   └── generate_grid_timeseries.py # full-night series from condition file
└── combine/
    └── make_observations.py   # stellar x telluric

data/                          # generated outputs (gitignored)
├── phoenix/
│   ├── downloads/             # raw PHOENIX HiRes FITS
│   └── convolved/             # R=100k convolved spectra
├── telluric/                  # independent snapshots (generate_grid.py)
│   ├── chunks/
│   └── telluric_templates.h5
├── telluric_timeseries/       # full-night series (generate_grid_timeseries.py)
│   ├── chunks/
│   └── telluric_timeseries.h5
└── observations/
    └── observations.h5        # combined training data
```
