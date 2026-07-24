"""Inference configuration."""

from pathlib import Path

from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Configuration for the inference pipeline."""

    checkpoint_path: Path
    device: str = "cuda"
    batch_size: int = Field(default=64, gt=0)
    compute_planet: bool = True
    compute_uncertainty: bool = False
    mc_dropout_samples: int = Field(default=50, gt=0, description="For MC Dropout uncertainty")
    return_latent: bool = False
    num_wavelength_bins: int = Field(default=4096, gt=0)
