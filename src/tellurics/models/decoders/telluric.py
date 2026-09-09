"""Shared per-exposure telluric-transmission decoder (1D CNN upsampler)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TelluricDecoder"]


class TelluricDecoder(nn.Module):
    """Shared 1D CNN decoder mapping a ``d``-dim code back to ``N`` samples.

    The code is first expanded to a small 2-D feature map (``d -> C0 x L0``,
    ``L0 ~ N / 2**stages``), which is then progressively upsampled by 2x
    nearest interpolation followed by convolutions whose channel count decays
    towards 1, mirroring the encoder hierarchy.  A final interpolation snaps
    the length exactly to ``N`` (no ``64 -> 51556`` giant linear layer).

    Forward accepts any leading dimensions: ``(..., d) -> (..., N)``.
    """

    def __init__(
        self,
        n_wavelength: int,
        latent_dim: int,
        channels: tuple[int, ...] = (64, 48, 32, 24, 16, 8, 4, 1),
        kernel: int = 3,
        dropout: float = 0.0,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        assert n_wavelength > 0 and latent_dim > 0
        assert all(c > 0 for c in channels) and channels[-1] == 1
        assert output_activation in ("sigmoid", "none")
        # Nearest 2x upsampling + same-padding conv at every stage doubles the
        # length exactly and keeps the upsampler fully convolutional.  Odd
        # kernel is required so the conv preserves the length it receives.
        assert kernel % 2 == 1, "decoder conv kernel must be odd"
        self.n_wavelength = n_wavelength
        self.latent_dim = latent_dim
        self.c0 = channels[0]
        self.n_stages = len(channels) - 1
        # base length such that base * 2**stages just covers N
        self.l0 = max(1, math.ceil(n_wavelength / (2 ** self.n_stages)))
        self.final_len = self.l0 * (2 ** self.n_stages)

        self.expand = nn.Linear(latent_dim, self.c0 * self.l0)

        blocks: list[nn.Module] = []
        cin = self.c0
        pad = (kernel - 1) // 2
        for cout in channels[1:]:
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(cin, cout, kernel, padding=pad),
            ]
            if cout > 1:  # keep final stage linear (raw logit channel)
                blocks += [nn.GroupNorm(1, cout), nn.GELU(), nn.Dropout(dropout)]
            cin = cout
        self.upsampler = nn.Sequential(*blocks)
        self.activation = (
            torch.sigmoid if output_activation == "sigmoid" else nn.Identity()
        )

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        """Decode spectrum/spectra.

        Args:
            code: (..., d).

        Returns:
            (..., N) predicted transmission in [0, 1] if sigmoid enabled.
        """
        lead = code.shape[:-1]
        x = code.reshape(-1, self.latent_dim)              # (M, d)
        x = self.expand(x)                                 # (M, C0 * L0)
        x = x.view(-1, self.c0, self.l0)                   # (M, C0, L0)
        x = self.upsampler(x)                              # (M, 1, final_len)
        x = x.squeeze(1)                                   # (M, final_len)
        if self.final_len != self.n_wavelength:
            x = F.interpolate(
                x.unsqueeze(1),
                size=self.n_wavelength,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        x = self.activation(x)                             # (M, N)
        return x.view(*lead, self.n_wavelength)
