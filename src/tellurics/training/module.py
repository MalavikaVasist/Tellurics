"""PyTorch Lightning training module."""

import torch
import pytorch_lightning as pl

from tellurics.configs.loss import LossConfig
from tellurics.configs.model import ModelConfig
from tellurics.configs.training import OptimizerType, SchedulerType, TrainingConfig
from tellurics.losses.composite import CompositeLoss
from tellurics.models.output import ModelOutput
from tellurics.utils.logging import get_logger
from tellurics.utils.registry import ModelRegistry

logger = get_logger(__name__)


class TelluricLightningModule(pl.LightningModule):
    """Lightning module for training telluric correction models.

    Handles:
        - Model instantiation from registry
        - Composite loss computation
        - Optimizer and scheduler configuration
        - Metric logging
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        loss_config: LossConfig,
    ) -> None:
        """Initialize the Lightning module.

        Args:
            model_config: Model architecture configuration.
            training_config: Training hyperparameter configuration.
            loss_config: Loss function configuration.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config
        self.loss_config = loss_config

        # Build model from registry
        model_cls = ModelRegistry.get(model_config.architecture.value)
        self.model = model_cls(model_config)

        # Build composite loss
        self.criterion = CompositeLoss(loss_config)

    def forward(self, spectrum: torch.Tensor, atm_params: torch.Tensor) -> ModelOutput:
        """Forward pass through the model.

        Args:
            spectrum: Normalized spectrum R. Shape: (B, N).
            atm_params: Atmospheric parameters. Shape: (B, N_params).

        Returns:
            ModelOutput with predictions.
        """
        return self.model(spectrum, atm_params)

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Shared step for training and validation.

        Args:
            batch: Data batch from DataLoader.
            stage: One of 'train', 'val', 'test'.

        Returns:
            Total loss value.
        """
        output = self.model(batch["spectrum"], batch["atm_params"])
        losses = self.criterion(output, batch)

        # Log all loss components
        for name, value in losses.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(name == "total"),
                batch_size=batch["spectrum"].size(0),
            )

        return losses["total"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Returns:
            Total training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Validation batch.
            batch_idx: Batch index.

        Returns:
            Total validation loss.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Test batch.
            batch_idx: Batch index.

        Returns:
            Total test loss.
        """
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration.
        """
        opt_config = self.training_config.optimizer
        sched_config = self.training_config.scheduler

        # Build optimizer
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

        # Build scheduler
        match sched_config.scheduler_type:
            case SchedulerType.COSINE:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.training_config.max_epochs,
                    eta_min=sched_config.min_lr,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
                }
            case SchedulerType.STEP:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sched_config.step_size,
                    gamma=sched_config.gamma,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
                }
            case SchedulerType.ONE_CYCLE:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt_config.learning_rate,
                    total_steps=self.trainer.estimated_stepping_batches,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
                }
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
