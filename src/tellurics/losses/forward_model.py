"""Supervised + forward-model consistency loss for whole-night telluric models.

The model predicts the telluric transmission ``T_hat``.  Because the
observation is ``X = T * S`` we can also reconstruct the observation from the
prediction and check it against the true observation (a physical, label-free
regularizer)::

    L_T = MSE(T_hat, T)                 # supervised term
    L_X = MSE(T_hat * S, X)             # forward-model consistency term
    L   = L_T + lambda_X * L_X

``lambda_X = 0`` (the default) gives the pure supervised baseline; set it to a
non-zero value to add the forward-model regularizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tellurics.models.output import ModelOutput

__all__ = ["telluric_losses", "TelluricForwardModelLoss"]


def _stellar_as_night(s: torch.Tensor, t: int) -> torch.Tensor:
    """Broadcast stellar spectrum to night shape.

    Args:
        s: (B, N) or (N,)
        t: number of exposures T.

    Returns:
        (B, 1, N) broadcastable against (B, T, N).
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
    if s.dim() == 2:
        s = s.unsqueeze(1)  # (B, 1, N)
    if s.dim() == 3 and s.shape[1] == 1:
        return s
    if s.dim() == 3 and s.shape[1] == t:
        return s
    raise ValueError(
        f"stellar must be (B, N), (N,) or (B, {t}, N); got {tuple(s.shape)}"
    )


def telluric_losses(
    t_hat: torch.Tensor,
    t_true: torch.Tensor,
    x: torch.Tensor | None = None,
    s: torch.Tensor | None = None,
    lambda_x: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the telluric loss terms.

    Args:
        t_hat: (B, T, N) predicted transmission.
        t_true: (B, T, N) true transmission.
        x: (B, T, N) observed spectrum X = T*S (optional).
        s: (B, N), (N,) or (B, T, N) stellar spectrum (optional).
        lambda_x: weight of the forward-model consistency term.

    Returns:
        (L_total, L_T, L_X) where L_total = L_T + lambda_X * L_X.

    Raises:
        ValueError: if ``lambda_x > 0`` but ``x`` or ``s`` is missing.
    """
    l_t = F.mse_loss(t_hat, t_true)

    if lambda_x > 0:
        if x is None or s is None:
            raise ValueError(
                "lambda_x > 0 requires both the observed spectra `x` and the "
                "stellar spectrum `s`."
            )
        t = t_true.shape[1]
        s_b = _stellar_as_night(s, t)
        x_hat = t_hat * s_b                       # forward model X_hat = T_hat*S
        l_x = F.mse_loss(x_hat, x)
    else:
        l_x = torch.zeros((), device=t_hat.device)

    return l_t + lambda_x * l_x, l_t, l_x


class TelluricForwardModelLoss(nn.Module):
    """nn.Module wrapper around :func:`telluric_losses`.

    Follows the codebase convention ``forward(output, batch)`` where ``batch``
    carries the keys ``telluric_true``, ``observed`` (X) and optionally
    ``stellar`` (S).
    """

    def __init__(self, lambda_x: float = 0.0) -> None:
        super().__init__()
        self.lambda_x = float(lambda_x)

    def forward(self, output: ModelOutput, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute L = MSE(T_hat, T) + lambda_X * MSE(T_hat*S, X).

        Args:
            output: model output with ``telluric`` (B, T, N).
            batch: dict with ``telluric_true``, ``observed`` and ``stellar``.

        Returns:
            Scalar total loss.
        """
        t_hat = output.telluric
        t_true = batch["telluric_true"]
        if self.lambda_x > 0:
            x = batch["observed"]
            s = batch.get("stellar")
            if s is None:
                raise ValueError(
                    "lambda_x > 0 requires 'stellar' in the batch."
                )
        else:
            x = s = None
        total, _, _ = telluric_losses(
            t_hat, t_true, x=x, s=s, lambda_x=self.lambda_x
        )
        return total
