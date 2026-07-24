"""Physics-informed loss functions."""

import torch
import torch.nn as nn

from tellurics.models.output import ModelOutput


class SmoothnessLoss(nn.Module):
    """Penalize the second (or higher) derivative of predicted telluric spectrum.

    Encourages physically smooth predictions by penalizing rapid oscillations.
    """

    def __init__(self, order: int = 2) -> None:
        """Initialize smoothness loss.

        Args:
            order: Derivative order to penalize (1=first, 2=second, 3=third).
        """
        super().__init__()
        self.order = order

    def forward(self, output: ModelOutput, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute smoothness penalty on telluric prediction.

        Args:
            output: Model output containing telluric prediction.
            batch: Data batch (unused, kept for consistent interface).

        Returns:
            Scalar smoothness penalty.
        """
        telluric = output.telluric  # (B, N)

        # Compute finite differences up to specified order
        diff = telluric
        for _ in range(self.order):
            diff = diff[:, 1:] - diff[:, :-1]

        return torch.mean(diff**2)


class PhysicalConstraintLoss(nn.Module):
    """Penalize predictions outside physical bounds [0, 1].

    While sigmoid ensures outputs are in [0, 1], this provides a softer
    penalty that can be used with other activation functions or to
    encourage predictions away from boundaries.
    """

    def __init__(self, margin: float = 0.01) -> None:
        """Initialize physical constraint loss.

        Args:
            margin: Soft margin from bounds (penalizes values within margin of 0 or 1).
        """
        super().__init__()
        self.margin = margin

    def forward(self, output: ModelOutput, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physical constraint penalty.

        Args:
            output: Model output containing telluric prediction.
            batch: Data batch (unused, kept for consistent interface).

        Returns:
            Scalar penalty value.
        """
        telluric = output.telluric

        # Penalty for values below 0 (soft lower bound)
        lower_violation = torch.relu(-telluric + self.margin)

        # Penalty for values above 1 (soft upper bound)
        upper_violation = torch.relu(telluric - 1.0 + self.margin)

        return torch.mean(lower_violation**2 + upper_violation**2)
