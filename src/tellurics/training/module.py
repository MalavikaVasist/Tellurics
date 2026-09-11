"""PyTorch Lightning module for the whole-night telluric estimator.

This module trains :class:`~tellurics.models.night.TelluricEstimator` on the
night-major batches produced by
:class:`~tellurics.data.datamodule.TelluricDataModule`.

Each batch is a dict::

    {
        "observed":    (B, T, N)   X = T_tell * S
        "stellar":     (B, N)      the S used to build observed
        "theta": {
            "time":     (B, T)     time_hours           -> TimeEncoder
            "metadata": (B, T, 3)  pressure/temp/humidity -> MetadataEncoder
            "params":   (B, T, P)  ground-truth target   (P = param_dim)
        },
    }

The estimator is run with ``observed``, ``stellar``, ``theta["metadata"]`` and
``theta["time"]``; the loss is the MSE between its predicted ``param_pred`` and
the ground-truth ``theta["params"]`` (the shared training/validation loop).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from tellurics.configs.model import ModelConfig
from tellurics.configs.training import OptimizerType, SchedulerType, TrainingConfig
from tellurics.models.output import ModelOutput
from tellurics.utils.logging import get_logger
from tellurics.utils.registry import ModelRegistry

import tellurics.models  # noqa: F401  (executes the @ModelRegistry.register decorators)

logger = get_logger(__name__)


class TelluricEstimatorModule(pl.LightningModule):
    """Train the whole-night telluric parameter estimator (MSE on params).

    Handles model instantiation from the ``telluric_estimator`` registry entry,
    the ``MSE(param_pred, params)`` training/validation loop, and optimizer /
    scheduler configuration.

    The estimator config must be consistent with the DataModule contract:
    ``param_dim`` == width of ``theta["params"]`` (16) and ``metadata_dim`` == 3.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Initialize.

        Args:
            model_config: Model architecture configuration (architecture must be
                ``telluric_estimator``).
            training_config: Training hyperparameter configuration.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        model_cls = ModelRegistry.get(model_config.architecture.value)
        self.model = model_cls(model_config)
        logger.info(
            f"TelluricEstimatorModule: {model_cls.__name__} with "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} "
            "trainable parameters"
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        observed: torch.Tensor,
        stellar: torch.Tensor,
        metadata: torch.Tensor,
        time: torch.Tensor,
    ) -> ModelOutput:
        """Forward through the estimator.

        Args:
            observed: (B, T, N) observed spectrum X = T_tell * S.
            stellar: (B, N) or (N,) the stellar spectrum used to build X.
            metadata: (B, T, 3) per-exposure metadata (pressure, temp, humidity).
            time: (B, T) continuous exposure times (time_hours).

        Returns:
            ModelOutput with ``params (B, T, P)``.
        """
        return self.model(
            observed, stellar=stellar, metadata=metadata, time=time
        )

    # ------------------------------------------------------------------ #
    def _run_estimator(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        """Run the estimator on a DataModule batch."""
        return self.model(
            batch["observed"],
            stellar=batch["stellar"],
            metadata=batch["theta"]["metadata"],
            time=batch["theta"]["time"],
        )

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Shared MSE(param_pred, theta.params) step for train/val/test.

        Args:
            batch: A batch from :class:`TelluricDataModule`.
            stage: One of 'train', 'val', 'test'.

        Returns:
            The MSE loss value.
        """
        pred = self._run_estimator(batch).params                 # (B, T, P)
        target = batch["theta"]["params"]                        # (B, T, P)
        loss = F.mse_loss(pred, target)
        rmse = loss.sqrt()

        self.log(
            f"{stage}_loss", loss,
            on_step=(stage == "train"), on_epoch=True, prog_bar=True,
            batch_size=batch["observed"].size(0),
        )
        self.log(
            f"{stage}_rmse", rmse,
            on_step=False, on_epoch=True,
            batch_size=batch["observed"].size(0),
        )
        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step."""
        return self._shared_step(batch, "test")

    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> dict:
        """Configure the optimizer and LR scheduler (mirrors module.py)."""
        opt_config = self.training_config.optimizer
        sched_config = self.training_config.scheduler

        match opt_config.optimizer_type:
            case OptimizerType.ADAM:
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=opt_config.learning_rate,
                    betas=opt_config.betas,
                    weight_decay=opt_config.weight_decay,
                )
            case OptimizerType.ADAMW:
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=opt_config.learning_rate,
                    betas=opt_config.betas,
                    weight_decay=opt_config.weight_decay,
                )
            case OptimizerType.SGD:
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=opt_config.learning_rate,
                    momentum=opt_config.momentum,
                    weight_decay=opt_config.weight_decay,
                )

        match sched_config.scheduler_type:
            case SchedulerType.COSINE:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.training_config.max_epochs,
                    eta_min=sched_config.min_lr,
                )
            case SchedulerType.STEP:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sched_config.step_size,
                    gamma=sched_config.gamma,
                )
            case SchedulerType.ONE_CYCLE:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt_config.learning_rate,
                    total_steps=self.trainer.estimated_stepping_batches,
                )
            case SchedulerType.REDUCE_ON_PLATEAU:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=sched_config.patience,
                    factor=sched_config.gamma,
                    min_lr=sched_config.min_lr,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.training_config.early_stopping_metric,
                        "interval": "epoch",
                    },
                }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
