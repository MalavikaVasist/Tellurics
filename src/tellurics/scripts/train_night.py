"""Train the whole-night Perceiver telluric regressor.

The model consumes one full night ``observed (B, T, N) = X = T*S`` and
regresses the whole-night telluric transmission with an MSE loss.

Example:
    # 1) build the night-major HDF5 once (streams the flat templates file):
    python -m tellurics.scripts.train_night \
        --build \
        --transmission data/telluric_templates.h5 \
        --stellar data/phoenix/convolved/phoenix_A0V_T9600_logg4.0_feh+0.0_R100000_s3.fits \
        --night-h5 data/night_dataset.h5

    # 2) train:
    python -m tellurics.scripts.train_night \
        --night-h5 data/night_dataset.h5 \
        --config configs/night_perceiver.json \
        --max-epochs 60 --batch-size 4
"""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tellurics.configs.model import ModelConfig
from tellurics.configs.training import TrainingConfig
from tellurics.data.night import NightDataModule, build_night_file
from tellurics.training.module_night import NightTelluricModule
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the whole-night Perceiver telluric regressor."
    )
    parser.add_argument(
        "--night-h5", type=Path, required=True,
        help="Night-major HDF5 built with --build (or existing).",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build the night-major HDF5 from a flat transmission file first.",
    )
    parser.add_argument(
        "--transmission", type=Path, default=None,
        help="Flat transmission HDF5 (n_flat, N) for --build.",
    )
    parser.add_argument(
        "--stellar", type=Path, default=None,
        help="Stellar spectrum FITS/.npy used by --build to form X = T*S.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON with 'model' and 'training' sections.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lambda-x", type=float, default=None,
                        help="Weight of the forward-model consistency loss "
                             "L_X = MSE(T_hat*S, X). Overrides the config.")
    parser.add_argument("--n-nights", type=int, default=None,
                        help="Limit --build to this many nights (testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_kwargs: dict = {
        "architecture": "perceiver_night",
        "num_wavelength_bins": 51556,
        "n_frames_per_series": 73,
        "num_queries": 16,
        "spectral_latent_dim": 64,
        "spectral_encoder_dims": [2048, 512, 128],
        "hidden_dim": 256,
        "num_heads": 4,
        "dropout": 0.1,
        "metadata_dim": 11,
    }
    training_kwargs: dict = {}
    loss_kwargs: dict = {}

    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
        model_kwargs.update(cfg.get("model", {}))
        training_kwargs.update(cfg.get("training", {}))
        loss_kwargs.update(cfg.get("loss", {}))

    if args.max_epochs is not None:
        training_kwargs["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        model_kwargs.setdefault("batch_size", args.batch_size)

    lambda_x = loss_kwargs.get("lambda_x", 0.0)
    if args.lambda_x is not None:
        lambda_x = args.lambda_x

    model_config = ModelConfig(**model_kwargs)
    training_config = TrainingConfig(**training_kwargs)

    # --- Build the night-major dataset if requested ---
    if args.build:
        if args.transmission is None or args.stellar is None:
            raise SystemExit("--build requires --transmission and --stellar.")
        build_night_file(
            transmission_h5=args.transmission,
            stellar_source=args.stellar,
            out_h5=args.night_h5,
            n_nights=args.n_nights,
        )

    pl.seed_everything(training_config.seed, workers=True)

    datamodule = NightDataModule(
        h5_path=args.night_h5,
        batch_size=args.batch_size or model_kwargs.get("batch_size", 4),
        train_fraction=0.8,
        val_fraction=0.1,
    )

    module = NightTelluricModule(model_config, training_config, lambda_x=lambda_x)

    callbacks = [
        ModelCheckpoint(
            dirpath=training_config.checkpoint_dir,
            filename="night-{epoch:03d}-{val_rmse:.6f}",
            monitor="val_rmse",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val_rmse", patience=training_config.early_stopping_patience,
                      mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    tb_logger = TensorBoardLogger(save_dir=training_config.log_dir, name="night_telluric")

    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=training_config.log_every_n_steps,
        val_check_interval=training_config.val_check_interval,
        deterministic=True,
    )

    logger.info(f"Architecture: {model_config.architecture.value}")
    logger.info(f"Params: {sum(p.numel() for p in module.parameters()):,}")
    logger.info(f"Night data: {args.night_h5}")
    logger.info("Starting training...")
    trainer.fit(module, datamodule=datamodule)
    logger.info("Running test evaluation...")
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
