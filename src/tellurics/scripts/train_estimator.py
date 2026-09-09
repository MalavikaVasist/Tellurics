"""Train the whole-night telluric estimator on the night-major telluric file.

Pipeline
--------
1. (optional) ``--build`` reshapes the flat ``data/telluric_templates.h5`` into
   the night-major ``data/telluric_timeseries/telluric_templates.h5``
   (4500 nights x 73 frames x 51556 samples + reshaped labels).
2. :class:`TelluricDataModule` loads whole nights, multiplies each night by a
   fixed (seeded) Phoenix stellar spectrum sampled from ``data/phoenix/convolved``,
   and returns ``observed (B,T,N)`` + ``stellar (B,N)`` + ``theta`` (dict with
   time / metadata (pressure,temp,humidity) / params (16 target cols)).
3. :class:`TelluricEstimatorModule` trains/validates with
   ``loss = MSE(estimator(...).params, theta["params"])``.

Example
-------
    # one-time reshape (streams the ~4500-night file)
    python -m tellurics.scripts.train_estimator --build

    # train (default paths)
    python -m tellurics.scripts.train_estimator --max-epochs 20 --batch-size 4
"""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from tellurics.configs.model import ModelArchitecture, ModelConfig
from tellurics.configs.training import TrainingConfig
from tellurics.data.datamodule import (
    TelluricDataModule,
    build_timeseries_h5,
)
from tellurics.training.module_estimator import TelluricEstimatorModule
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


# Default estimator hyper-parameters consistent with the DataModule contract:
# metadata_dim == 3 (pressure, temperature, humidity), param_dim == 16
# (all physical labels except time_hours and the flat/series/frame indices).
def default_model_kwargs() -> dict:
    return {
        "architecture": ModelArchitecture.TELLURIC_ESTIMATOR,
        "num_wavelength_bins": 51556,
        "n_frames_per_series": 73,
        "num_queries": 16,
        "spectral_latent_dim": 64,
        "metadata_dim": 3,
        "metadata_enc_dim": 16,
        "metadata_enc_hidden": None,
        "time_enc_dim": 16,
        "param_dim": 16,
        "param_decoder_hidden": None,
        "num_heads": 4,
        "dropout": 0.1,
        "x_encoder_channels": [16, 32, 64, 96, 128],
        "x_encoder_kernel": 7,
        "x_encoder_stride": 2,
        "s_encoder_channels": [16, 32, 64],
        "s_encoder_kernel": 7,
        "s_encoder_stride": 2,
        "encoder_pool_bins": 16,
        "fusion_hidden": 0,
        "fusion_layers": 2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the whole-night telluric parameter estimator."
    )
    parser.add_argument(
        "--flat-h5", type=Path,
        default=Path("data/telluric_templates.h5"),
        help="Flat transmission HDF5 (used only with --build).",
    )
    parser.add_argument(
        "--night-h5", type=Path,
        default=Path("data/telluric_timeseries/telluric_templates.h5"),
        help="Night-major HDF5 (built by --build, or pre-existing).",
    )
    parser.add_argument(
        "--phoenix-dir", type=Path,
        default=Path("data/phoenix/convolved"),
        help="Directory with the stellar *.fits pool.",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Reshape --flat-h5 into --night-h5 first.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON with optional 'model' and 'training' sections "
             "(keys override the estimator defaults).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.build:
        build_timeseries_h5(args.flat_h5, args.night_h5)
    elif not args.night_h5.exists():
        raise FileNotFoundError(
            f"night-major HDF5 not found at {args.night_h5}. "
            "Run with --build to create it from the flat file."
        )

    model_kwargs = default_model_kwargs()
    training_kwargs: dict = {}

    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
        model_kwargs.update(cfg.get("model", {}))
        training_kwargs.update(cfg.get("training", {}))

    model_kwargs["architecture"] = ModelArchitecture(model_kwargs["architecture"])
    model_config = ModelConfig(**model_kwargs)
    training_config = TrainingConfig(**training_kwargs)

    datamodule = TelluricDataModule(
        night_h5=args.night_h5,
        phoenix_dir=args.phoenix_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    module = TelluricEstimatorModule(model_config, training_config)

    logger.info(
        f"Training telluric_estimator | N={model_config.num_wavelength_bins}, "
        f"T={model_config.n_frames_per_series}, "
        f"metadata_dim={model_config.metadata_dim}, param_dim={model_config.param_dim}"
    )

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=training_config.checkpoint_dir / "telluric_estimator",
            monitor=training_config.early_stopping_metric,
            mode="min",
            save_top_k=1,
            filename="estimator-{epoch:03d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if training_config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=training_config.early_stopping_metric,
                patience=training_config.early_stopping_patience,
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        val_check_interval=training_config.val_check_interval,
        log_every_n_steps=training_config.log_every_n_steps,
        logger=TensorBoardLogger(
            save_dir=str(training_config.log_dir), name="telluric_estimator"
        ),
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
