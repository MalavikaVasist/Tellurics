"""Top-level experiment manifest: the ``model:`` / ``data:`` / ``training:`` sections."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from tellurics.configs.data import DataConfig
from tellurics.configs.model import ModelConfig
from tellurics.configs.training import TrainingConfig
from tellurics.configs.wandb import WandbConfig


class ExperimentConfig(BaseModel):
    """A complete experiment: what to build, on what data, how to train it.

    Artefacts are written under ``run_dir``::

        runs/temporal_conv_001/
        |-- config.yaml               # resolved copy of this manifest
        |-- stellar_assignment.csv    # night -> star audit (from the DataModule)
        |-- logs/                     # logger output
        +-- checkpoints/              # ModelCheckpoint output
    """

    run_dir: Path = Field(
        default=Path("runs/default"),
        description="Where this run's artefacts are written: the resolved "
                    "config.yaml, the stellar-assignment audit, logs/ and "
                    "checkpoints/. Overridden by nothing; set it per manifest.",
    )
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: str | Path | None) -> ExperimentConfig:
    """Load and validate a YAML experiment manifest.

    Args:
        path: Path to the YAML manifest, or ``None`` for all-default config.

    Returns:
        The validated :class:`ExperimentConfig`. Sections omitted from the
        manifest fall back to the corresponding config model's defaults.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        ValueError: If the file is not a YAML mapping.
    """
    if path is None:
        return ExperimentConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"experiment manifest not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping with 'model', 'data' and/or "
            "'training' sections"
        )
    return ExperimentConfig(**raw)
