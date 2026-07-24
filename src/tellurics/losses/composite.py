"""Composite loss combining multiple objectives for mixed batches."""

import torch
import torch.nn as nn

from tellurics.configs.loss import LossConfig
from tellurics.losses.physics import PhysicalConstraintLoss, SmoothnessLoss
from tellurics.losses.supervised import PlanetReconstructionLoss, TelluricRegressionLoss
from tellurics.models.output import ModelOutput
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class CompositeLoss(nn.Module):
    """Composite loss function supporting mixed simulated/real batches.

    For simulated samples:
        L = λ1 * L_telluric + λ2 * L_planet + λ3 * L_smoothness + λ4 * L_physical

    For real samples:
        L = λ_real_smooth * L_smoothness + λ_real_physical * L_physical
    """

    def __init__(self, config: LossConfig) -> None:
        """Initialize composite loss.

        Args:
            config: Loss configuration with weights.
        """
        super().__init__()
        self.config = config

        # Supervised losses (simulated only)
        self.telluric_loss = TelluricRegressionLoss()
        self.planet_loss = PlanetReconstructionLoss()

        # Physics losses (both simulated and real)
        self.smoothness_loss = SmoothnessLoss(order=config.smoothness_order)
        self.physical_loss = PhysicalConstraintLoss()

    def forward(
        self, output: ModelOutput, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss with per-component breakdown.

        Args:
            output: Model output with telluric prediction.
            batch: Data batch with labels and 'is_simulated' flag.

        Returns:
            Dictionary with 'total' loss and individual components.
        """
        losses: dict[str, torch.Tensor] = {}
        is_simulated = batch["is_simulated"]  # (B,)

        # Separate simulated and real samples
        sim_mask = is_simulated.bool()
        has_simulated = sim_mask.any()
        has_real = (~sim_mask).any()

        total_loss = torch.tensor(0.0, device=output.telluric.device)

        # Supervised losses on simulated data
        if has_simulated:
            sim_output = ModelOutput(telluric=output.telluric[sim_mask])
            sim_batch = {k: v[sim_mask] for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Telluric regression
            l_telluric = self.telluric_loss(sim_output, sim_batch)
            losses["telluric"] = l_telluric
            total_loss = total_loss + self.config.lambda_telluric * l_telluric

            # Planet reconstruction
            if self.config.lambda_planet > 0 and "planet_true" in sim_batch:
                l_planet = self.planet_loss(sim_output, sim_batch)
                losses["planet"] = l_planet
                total_loss = total_loss + self.config.lambda_planet * l_planet

            # Smoothness on simulated
            if self.config.lambda_smoothness > 0:
                l_smooth = self.smoothness_loss(sim_output, sim_batch)
                losses["smoothness_sim"] = l_smooth
                total_loss = total_loss + self.config.lambda_smoothness * l_smooth

            # Physical constraints on simulated
            if self.config.lambda_physical > 0:
                l_phys = self.physical_loss(sim_output, sim_batch)
                losses["physical_sim"] = l_phys
                total_loss = total_loss + self.config.lambda_physical * l_phys

        # Physics-only losses on real data
        if has_real and self.config.use_real_regularization:
            real_output = ModelOutput(telluric=output.telluric[~sim_mask])
            real_batch = {k: v[~sim_mask] for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if self.config.lambda_real_smoothness > 0:
                l_smooth_real = self.smoothness_loss(real_output, real_batch)
                losses["smoothness_real"] = l_smooth_real
                total_loss = total_loss + self.config.lambda_real_smoothness * l_smooth_real

            if self.config.lambda_real_physical > 0:
                l_phys_real = self.physical_loss(real_output, real_batch)
                losses["physical_real"] = l_phys_real
                total_loss = total_loss + self.config.lambda_real_physical * l_phys_real

        losses["total"] = total_loss
        return losses
