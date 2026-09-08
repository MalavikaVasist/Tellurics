"""PyTorch Lightning training module for whole-night telluric regression."""

import torch
import pytorch_lightning as pl

from tellurics.configs.model import ModelConfig
from tellurics.configs.training import TrainingConfig
from tellurics.losses.forward_model import telluric_losses
from tellurics.models.output import ModelOutput
from tellurics.utils.logging import get_logger
from tellurics.utils.registry import ModelRegistry

import tellurics.models  # noqa: F401  (executes the @ModelRegistry.register decorators)

logger = get_logger(__name__)


class NightTelluricModule(pl.LightningModule):
    """Lightning module for the whole-night telluric regressors.

    Works with any whole-night architecture registered in the model registry
    that maps ``observed (B, T, N) -> telluric (B, T, N)`` -- e.g.
    ``perceiver_night`` (models/night.py) and ``temporal_conv``
    (models/temporal_conv.py).

    Loss: supervised MSE over the whole night (all exposures and wavelength
    samples):

        observed (B, T, N)  ->  T_hat (B, T, N)
        target   telluric_true (B, T, N)
        L_T = mean( (telluric_true - T_hat)^2 )

    Optionally (lambda_x > 0) a forward-model consistency term is added
    because X = T * S:

        L_X = mean( (T_hat * S - observed)^2 ),   L = L_T + lambda_x * L_X
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        lambda_x: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_config = training_config
        self.lambda_x = float(lambda_x)

        model_cls = ModelRegistry.get(model_config.architecture.value)
        self.model = model_cls(model_config)

        self.mse = torch.nn.MSELoss()

    def forward(
        self,
        observed: torch.Tensor,
        stellar: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Forward through the night model.

        Args:
            observed: (B, T, N) observed spectrum X = T*S.
            stellar: (B, N) or (N,) stellar spectrum S.
            metadata: (B, P) per-night metadata.

        Returns:
            ModelOutput with telluric (B, T, N).
        """
        return self.model(observed, stellar=stellar, metadata=metadata, time=time)

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        observed = batch["observed"]          # (B, T, N)
        telluric_true = batch["telluric_true"]  # (B, T, N)
        stellar = batch.get("stellar")        # (N,) or (B, N)
        metadata = batch.get("metadata")      # (P,) or (B, P)

        output = self.model(observed, stellar=stellar, metadata=metadata)

        if output.telluric.shape != telluric_true.shape:
            raise ValueError(
                f"Prediction shape {tuple(output.telluric.shape)} != target "
                f"{tuple(telluric_true.shape)}. Check n_frames_per_series and "
                "num_wavelength_bins in the model config."
            )

        if self.lambda_x > 0.0:
            if stellar is None:
                raise ValueError(
                    "lambda_x > 0 requires a stellar spectrum in the batch."
                )
            loss, l_t, l_x = telluric_losses(
                output.telluric, telluric_true,
                x=observed, s=stellar, lambda_x=self.lambda_x,
            )
            self.log(f"{stage}_lx", l_x, on_step=(stage == "train"), on_epoch=True,
                     batch_size=observed.size(0))
        else:
            loss = self.mse(output.telluric, telluric_true)
            l_t = loss
        rmse = torch.sqrt(l_t)

        self.log(f"{stage}_mse", l_t, on_step=(stage == "train"), on_epoch=True,
                 prog_bar=True, batch_size=observed.size(0))
        self.log(f"{stage}_rmse", rmse, on_step=False, on_epoch=True,
                 batch_size=observed.size(0))
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict:
        """AdamW + cosine LR schedule."""
        opt_cfg = self.training_config.optimizer
        sched_cfg = self.training_config.scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.learning_rate,
            betas=opt_cfg.betas,
            weight_decay=opt_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config.max_epochs,
            eta_min=sched_cfg.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
