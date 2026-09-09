"""Per-exposure parameter decoder (shared MLP)."""

import torch
import torch.nn as nn

__all__ = ["ParamDecoder"]


class ParamDecoder(nn.Module):
    """Shared per-exposure MLP decoding an exposure code into ``P`` parameters.

    Maps each ``d``-dim per-exposure code of ``H_T`` to the ``P`` telluric /
    atmospheric parameters that parametrize that exposure's transmission:

        H_T (B, T, d)  ->  param_pred (B, T, P)

    The same MLP instance is applied to every exposure (never re-instantiated
    per frame), mirroring the spectral-encoder sharing.  The output is linear
    (raw parameter values) so the caller decides the loss, e.g.
    ``MSE(param_input, param_pred)``.

    Forward accepts any leading dimensions: ``(..., d) -> (..., P)``.
    """

    def __init__(
        self,
        latent_dim: int,
        param_dim: int,
        hidden_dim: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert latent_dim > 0 and param_dim > 0 and n_layers >= 1
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        hidden = hidden_dim or max(latent_dim, 2 * param_dim)
        layers: list[nn.Module] = []
        cur = latent_dim
        for i in range(n_layers):
            out = param_dim if i == n_layers - 1 else hidden
            layers.append(nn.Linear(cur, out))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            cur = out
        self.net = nn.Sequential(*layers)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        """Decode exposure codes. ``(..., d) -> (..., P)``."""
        lead = code.shape[:-1]
        x = code.reshape(-1, self.latent_dim)      # (M, d)
        out = self.net(x)                          # (M, P)
        return out.view(*lead, self.param_dim)
