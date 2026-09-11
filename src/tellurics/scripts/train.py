"""Train the whole-night telluric estimator on the night-major telluric file.

Pipeline
--------
1. The night-major ``data/telluric_timeseries/telluric_templates.h5``
   (4500 nights x 73 frames x 51556 samples + labels) is produced once, out of
   band, by the ``tests/testing_dataset_reshape.ipynb`` notebook, which
   reshapes the flat ``data/telluric_templates.h5``.
2. :class:`TelluricDataModule` loads whole nights, multiplies each night by a
   fixed (seeded) Phoenix stellar spectrum sampled from ``data/phoenix/convolved``,
   and returns ``observed (B,T,N)`` + ``stellar (B,N)`` + ``theta`` (dict with
   time / metadata (pressure,temp,humidity) / params (16 target cols)).
3. :class:`TelluricEstimatorModule` trains/validates with
   ``loss = MSE(estimator(...).params, theta["params"])``.

Manifest layout (see ``experiments/experiment2.yaml``)::

    run_dir:    # where the artefacts go, e.g. runs/temporal_conv_001/
    wandb:      # -> WandbConfig     (project + run title)
    model:      # -> ModelConfig     (estimator architecture + dims)
    data:       # -> DataConfig      (input paths + night-level split)
        night_h5:     data/telluric_timeseries/telluric_templates.h5
        phoenix_dir:  data/phoenix/convolved
    training:   # -> TrainingConfig  (optimiser, epochs, batch size, logging)

Each run writes a self-describing directory::

    runs/temporal_conv_001/
    |-- config.yaml               # resolved copy of the manifest
    |-- stellar_assignment.csv    # night -> star audit
    |-- logs/                     # W&B local files
    |-- checkpoints/              # ModelCheckpoint output

The CLI is deliberately tiny -- everything else lives in the manifest::

    python -m tellurics.scripts.train --config experiments/experiment2.yaml

    python -m tellurics.scripts.train --config experiments/experiment2.yaml \
        --device cpu --resume runs/temporal_conv_001/checkpoints/telluric_estimator/x.ckpt
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from tellurics.configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
    load_config,
)
from tellurics.data.datamodule import TelluricDataModule
from tellurics.training.module import TelluricEstimatorModule
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the deliberately tiny CLI: the manifest plus two run overrides."""
    parser = argparse.ArgumentParser(
        description="Train the whole-night telluric parameter estimator."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML experiment manifest with 'model', 'data' and 'training' "
             "sections (see experiments/).",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Checkpoint (.ckpt) to resume training from.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'auto' (default), 'cpu', 'gpu'/'cuda', 'mps', or a GPU index "
             "such as '0'.",
    )
    return parser.parse_args()


def device_kwargs(device: str) -> dict:
    """Translate ``--device`` into ``pl.Trainer`` accelerator/devices kwargs."""
    value = device.strip().lower()
    if value == "auto":
        return {"accelerator": "auto", "devices": "auto"}
    if value == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    if value in {"gpu", "cuda"}:
        return {"accelerator": "gpu", "devices": "auto"}
    if value == "mps":
        return {"accelerator": "mps", "devices": 1}
    if value.isdigit():
        return {"accelerator": "gpu", "devices": int(value)}
    raise ValueError(
        "--device must be 'auto', 'cpu', 'gpu'/'cuda', 'mps' or an integer "
        f"GPU index, got {device!r}"
    )


def prepare_run_dir(config: ExperimentConfig) -> Path:
    """Create ``run_dir`` and drop a resolved manifest copy inside."""
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    logger.info(f"run directory: {run_dir}")
    return run_dir


def build_datamodule(
    data: DataConfig, training: TrainingConfig, run_dir: Path
) -> TelluricDataModule:
    """Build the DataModule, failing early on missing inputs."""
    if not data.night_h5.exists():
        raise FileNotFoundError(
            f"night-major HDF5 not found at {data.night_h5}. "
            "Create it once with tests/testing_dataset_reshape.ipynb "
            "(it reshapes the flat data/telluric_templates.h5)."
        )
    if not data.phoenix_dir.is_dir():
        raise FileNotFoundError(
            f"Phoenix stellar pool directory not found at {data.phoenix_dir}."
        )
    return TelluricDataModule(
        data,
        run_dir=run_dir,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
    )


def build_model(
    model_config: ModelConfig, training_config: TrainingConfig
) -> TelluricEstimatorModule:
    """Build the LightningModule (registry-resolved estimator) to train."""
    logger.info(
        f"Training {model_config.architecture.value} | "
        f"N={model_config.num_wavelength_bins}, "
        f"T={model_config.n_frames_per_series}, "
        f"metadata_dim={model_config.metadata_dim}, "
        f"param_dim={model_config.param_dim}"
    )
    return TelluricEstimatorModule(model_config, training_config)


def build_trainer(
    config: TrainingConfig,
    wandb_config: WandbConfig,
    run_dir: Path,
    device: str,
) -> pl.Trainer:
    """Build the Trainer: callbacks, W&B logger and device placement."""
    checkpoint_dir = run_dir / config.checkpoint_dir
    log_dir = run_dir / config.log_dir

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir / "telluric_estimator",
            monitor=config.early_stopping_metric,
            mode="min",
            save_top_k=1,
            filename="estimator-{epoch:03d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=config.early_stopping_metric,
                patience=config.early_stopping_patience,
                mode="min",
            )
        )

    return pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        # W&B auth: run `wandb login` once, or set WANDB_API_KEY; use
        # WANDB_MODE=offline for a purely local run.
        logger=WandbLogger(
            save_dir=str(log_dir),
            project=wandb_config.project,
            name=wandb_config.title,
        ),
        callbacks=callbacks,
        enable_progress_bar=True,
        **device_kwargs(device),
    )


def train(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    config: TrainingConfig,
    wandb_config: WandbConfig,
    run_dir: Path,
    device: str,
    resume: Path | None,
) -> None:
    """Build the Trainer and run ``fit`` (optionally resuming a checkpoint)."""
    if resume is not None and not resume.exists():
        raise FileNotFoundError(f"checkpoint to resume from not found: {resume}")

    trainer = build_trainer(config, wandb_config, run_dir, device)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=None if resume is None else str(resume),
    )


def main() -> None:
    """parse CLI -> load YAML -> build configs -> build objects -> train."""
    args = parse_args()

    config = load_config(args.config)

    run_dir = prepare_run_dir(config)
    datamodule = build_datamodule(config.data, config.training, run_dir)
    model = build_model(config.model, config.training)

    train(
        model=model,
        datamodule=datamodule,
        config=config.training,
        wandb_config=config.wandb,
        run_dir=run_dir,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
