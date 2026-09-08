"""Modular loss functions for telluric correction training."""

from tellurics.losses.composite import CompositeLoss
from tellurics.losses.forward_model import TelluricForwardModelLoss, telluric_losses
from tellurics.losses.physics import PhysicalConstraintLoss, SmoothnessLoss
from tellurics.losses.supervised import PlanetReconstructionLoss, TelluricRegressionLoss

__all__ = [
    "CompositeLoss",
    "PhysicalConstraintLoss",
    "PlanetReconstructionLoss",
    "SmoothnessLoss",
    "TelluricRegressionLoss",
    "TelluricForwardModelLoss",
    "telluric_losses",
]
