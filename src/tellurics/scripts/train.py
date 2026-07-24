"""Training script entry point."""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tellurics.configs.data import DatasetConfig
from tellurics.configs.loss import LossConfig
from tellurics.configs.model import ModelConfig
from tellurics.configs.training import TrainingConfig
from tellurics.data.datamodule import TelluricDataModule
from tellurics.training.module import TelluricLightningModule
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run training from command line."""
    parser = argparse.ArgumentParser(description="Train telluric correction model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        raw_config = json.load(f)

    dataset_config = DatasetConfig(**raw_config["dataset"])
    model_config = ModelConfig(**raw_config["model"])
    training_config = TrainingConfig(**raw_config["training"])
    loss_config = LossConfig(**raw_config.get("loss", {}))

    logger.info(f"Architecture: {model_config.architecture.value}")
    logger.info(f"Fusion method: {model_config.fusion_method.value}")
    logger.info(f"Max epochs: {training_config.max_epochs}")

    # Set seed for reproducibility
    pl.seed_everything(training_config.seed, workers=True)

    # Initialize DataModule
    datamodule = TelluricDataModule(dataset_config)

    # Initialize model
    module = TelluricLightningModule(
        model_config=model_config,
        training_config=training_config,
        loss_config=loss_config,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=training_config.checkpoint_dir,
            filename="telluric-{epoch:03d}-{val_total:.6f}",
            monitor=training_config.early_stopping_metric,
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor=training_config.early_stopping_metric,
            patience=training_config.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=training_config.log_dir,
        name="telluric_correction",
    )

    # Trainer
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

    # Train
    logger.info("Starting training...")
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)

    # Test
    logger.info("Running test evaluation...")
    trainer.test(module, datamodule=datamodule)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
