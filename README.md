# Tellurics

Physics-informed deep learning pipeline for telluric correction and exoplanet signal recovery from high-resolution spectroscopy.

## Scientific Background

The observed spectrum decomposes as:

$$X(t, \lambda) = S(\lambda) \cdot T_{\text{tell}}(t,\lambda) \cdot P(t,\lambda) + \epsilon$$

This package trains a neural network to predict the telluric transmission:

$$f(R, \text{atmospheric parameters}) \rightarrow T_{\text{tell}}$$

where $R = X / S$, enabling recovery of the planetary signal:

$$\hat{P} = \frac{X/S}{\hat{T}_{\text{tell}}}$$

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from tellurics.inference import InferencePipeline
from tellurics.configs import InferenceConfig

config = InferenceConfig(checkpoint_path="path/to/model.ckpt")
pipeline = InferencePipeline(config)
output = pipeline.predict(observed_spectrum, stellar_template, atm_params)
print(output.telluric.shape)
```

## Project Structure

```
src/tellurics/
├── configs/         # Pydantic configuration models
├── data/            # Datasets and DataModules
├── preprocessing/   # Wavelength alignment, normalization, masking
├── models/          # Neural network architectures
├── losses/          # Modular loss functions
├── training/        # Lightning training module
├── evaluation/      # Metrics and plotting
├── inference/       # Production inference pipeline
├── utils/           # Logging, registry, helpers
└── scripts/         # CLI entry points
```

## Development

```bash
ruff check src/ tests/
mypy src/
pytest
```
