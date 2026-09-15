"""PyTorch Lightning module for the whole-night telluric estimator.

This module trains :class:`~tellurics.models.night.NeuralTelluricPredictor` on the
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

Both sides of that MSE live on the same scale. When ``data.scale_params`` is
enabled the DataModule has already mapped ``theta["params"]`` onto ``[0, 1]``
using the declared ``data.param_bounds``, and the model's parameter head is
bounded to match (``model.param_activation: sigmoid``). ``training_loss``,
``validation_loss`` and the ``*_rmse`` metrics are therefore in *normalized*
units; call ``ParameterScaler.inverse_params`` (or multiply by the per-column
``max - min``) to express a prediction in physical units again.
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


class TelluricTrainingModule(pl.LightningModule):
    """Train the whole-night telluric parameter estimator (MSE on params).

    Handles model instantiation from the ``neural_telluric_predictor`` registry entry,
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
                ``neural_telluric_predictor``).
            training_config: Training hyperparameter configuration.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        model_cls = ModelRegistry.get(model_config.architecture.value)
        self.model = model_cls(model_config)
        logger.info(
            f"TelluricTrainingModule: {model_cls.__name__} with "
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
        return self(
            batch["observed"],
            stellar=batch["stellar"],
            metadata=batch["theta"]["metadata"],
            time=batch["theta"]["time"],
        )

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Shared MSE(param_pred, theta.params) step for train/val/test.

        Every metric is logged at **epoch level** (``on_step=False``,
        ``on_epoch=True``), so the keys carry no ``_step``/``_epoch`` suffix
        and W&B receives one value per epoch under a stable name:

        * ``training_loss``   -- MSE over the training batches (epoch mean).
        * ``validation_loss`` -- MSE over the validation batches (epoch mean).
        * ``learning_rate``   -- optimizer LR used during the epoch (train only).
        * ``nans``            -- percentage of the epoch's training batches
          whose loss was NaN.
        * ``nans_val``        -- percentage of the epoch's validation batches
          whose loss was NaN.

        The ``test`` stage falls back to ``test_loss`` / ``nans_test``.

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
        is_train = stage == "train"
        batch_size = batch["observed"].size(0)

        loss_name = {"train": "training_loss", "val": "validation_loss"}.get(
            stage, f"{stage}_loss"
        )
        nan_name = {"train": "nans", "val": "nans_val"}.get(
            stage, f"nans_{stage}"
        )

        self.log(
            loss_name, loss,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_rmse", rmse,
            on_step=False, on_epoch=True,
            batch_size=batch_size,
        )
        # Percentage of the epoch's batches whose loss was NaN (0% = healthy).
        # No batch_size on purpose: Lightning then takes the plain mean over the
        # epoch's batches (default reduce_fx), so the 0/1 flags average to a %.
        self.log(
            nan_name, 100.0 * torch.isnan(loss).float(),
            on_step=False, on_epoch=True,
        )
        if is_train:
            self.log(
                "learning_rate", self._current_lr(),
                on_step=False, on_epoch=True, prog_bar=True,
            )
        return loss

    def _current_lr(self) -> float:
        """Learning rate of the first optimizer's first parameter group."""
        optimizers = self.trainer.optimizers
        if not optimizers:
            return 0.0
        return float(optimizers[0].param_groups[0]["lr"])

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

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> ModelOutput:
        """Prediction step: run the trained estimator (no loss, no logging).

        Used by ``Trainer.predict(...)`` to collect one :class:`ModelOutput`
        per batch (notably ``.params``) over a dataset without training.
        """
        return self._run_estimator(batch)

    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> dict:
        """Configure the optimizer and LR scheduler."""
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
                        "monitor": self.training_config.monitor_metric,
                        "interval": "epoch",
                    },
                }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
