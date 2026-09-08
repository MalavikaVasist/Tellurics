"""Whole-night telluric CNN autoencoder with temporal Perceiver compression.

Task
----
A full night of observations is a stack of ``T`` (= 73) exposures of ``N``
(= 51,556) spectral samples.  The observation satisfies::

    X(t, l) = T_tell(t, l) * S(l)

where ``S`` is the (night-constant) stellar spectrum and ``T_tell`` is the
telluric transmission.  Instead of reconstructing the full ``N``-sample
transmission, we regress the ``P`` telluric / atmospheric parameters that
parametrize each exposure (e.g. via a TelFit-style forward model):

    X: (B, T, N)  ->  param_pred: (B, T, P)   loss = MSE(param_input - param_pred)

The two natural dimensions are exploited separately:

1. spectral compression  : the shared 1D CNN maps each of the ``N`` samples of
   one exposure to a ``d``-dim embedding (one CNN, weights shared over all
   exposures),  N = 51,556 -> d = 64 per exposure;
2. temporal compression  : the Perceiver cross-attention compresses the ``T``
   exposures into ``Q`` learned latent tokens,  T = 73 -> Q = 16.

Architecture (shape-annotated)
------------------------------
    X (B, 73, 51556)              M (B, 73, 3)  per-exposure metadata
        | SpectralEncoder                 | MetadataEncoding (shared MLP)
        v (one 1D CNN per exposure)       v
    Z (B, 73, 64)                 M_code (B, 73, 16)
        |                                |
        +-- concat[ Z, M_code ] --> tokens (B, 73, 80)
                       | Linear(80 -> 64) input projection
                       v
    TemporalPerceiver (cross-attn, 16 learned latent queries)
                       v
    L (B, 16, 64)  --flatten-->  h_X (B, 1024)
        |
        +-- concat[ h_X (B,1024),  h_S (B,64) = StellarEncoder(S) ] -> (B, 1088)
        v
    FusionMLP (1088 -> 1024 -> 1024)
        | h_fused (B, 1024) --reshape (1024 == 16*64)-->
        v
    latent (B, 16, 64)
        | TemporalDecoder (cross-attn, 73 learned exposure queries)
        v
    H_T (B, 73, 64)
        | ParamDecoder (shared MLP, applied per exposure)
        v
    param_pred (B, 73, 20)      loss = MSE(param_input, param_pred)

Notes
-----
* The *same* module instance (hence the same weights) is applied to every
  exposure; it is never re-instantiated per frame.
* The X spectral encoder and the S stellar encoder are **independent**
  networks with independent weights (S is *not* encoded with the X encoder).
* The S spectrum and the metadata enter the model only as small vectors
  (``h_S`` = 64; metadata is encoded to 16 dims and concatenated with the
  spectral codes *before* the temporal Perceiver), so the fusion stays small.
* Everything downstream of the per-exposure CNN codes is tiny; the parameter
  count is dominated by the two spectral CNNs and the fusion MLP.
* No batch size is hard-coded anywhere.

First-implementation priorities
-------------------------------
Verify end-to-end shapes, gradient flow, then overfit a handful of synthetic
nights.  Only after that should CNN depth, heads, latent size, number of
temporal queries, decoder depth, dropout and loss weighting be tuned.

Example (tiny dims, shape check)
--------------------------------
    cfg = TemporalConvConfig(n_wavelength=1024, n_frames=12, n_queries=8,
                             latent_dim=32, metadata_dim=3, n_heads=4,
                             x_encoder_channels=(8, 16, 32),
                             s_encoder_channels=(8, 16, 32))
    model = TelluricModel(cfg)
    out = model(torch.randn(2, 12, 1024),            # X = T*S
                torch.randn(2, 1024),                # S (stellar)
                torch.randn(2, 12, 3))               # per-exposure metadata
    assert out.params.shape == (2, 12, 20)           # param_pred (B, T, P)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tellurics.configs.model import ModelConfig
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry

__all__ = [
    "TemporalConvConfig",
    "SpectralEncoder",
    "MetadataEncoding",
    "TemporalPerceiver",
    "StellarEncoder",
    "FusionMLP",
    "TemporalDecoder",
    "SpectralDecoder",
    "ParamDecoder",
    "TelluricModel",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TemporalConvConfig:
    """Hyper-parameters of the whole-night CNN auto-encoder.

    Defaults target the real problem: N = 51,556 samples, T = 73 exposures,
    Q = 16 latent queries, d = 64-dim per-exposure code.
    """

    # ---- problem dimensions -------------------------------------------------
    n_wavelength: int = 51556          # spectral samples N per exposure
    n_frames: int = 73                 # exposures T per night
    n_queries: int = 16                # learned latent tokens Q (temporal)
    latent_dim: int = 64               # code dim d (per exposure / per token)
    metadata_dim: int = 3              # per-exposure scalar metadata width P
    metadata_enc_dim: int = 16         # encoded metadata code width C (per exposure)
    metadata_enc_hidden: int | None = None  # metadata MLP hidden (None -> auto)
    param_dim: int = 20                # predicted per-exposure parameter width
    param_decoder_hidden: int | None = None  # param-decoder MLP hidden (None -> auto)

    # ---- attention ----------------------------------------------------------
    n_heads: int = 4                   # heads for Perceiver + temporal decoder
    dropout: float = 0.0

    # ---- shared spectral encoder applied to X ------------------------------
    x_encoder_channels: tuple[int, ...] = (16, 32, 64, 96, 128)
    x_encoder_kernel: int = 7
    x_encoder_stride: int = 2

    # ---- independent stellar encoder applied to S --------------------------
    s_encoder_channels: tuple[int, ...] = (16, 32, 64)
    s_encoder_kernel: int = 7
    s_encoder_stride: int = 2

    # ---- spectral-code read-out --------------------------------------------
    # After the strided CNN the remaining wavelength axis is adaptively pooled
    # to this many bins per channel, flattened and projected to d.  Keeping a
    # handful of spatial bins (instead of a single global average) preserves
    # local spectral structure while still yielding exactly d numbers/sample.
    encoder_pool_bins: int = 16

    # ---- fusion MLP ---------------------------------------------------------
    fusion_hidden: int | None = None   # None -> n_queries * latent_dim (1024)
    fusion_layers: int = 2             # (in -> hidden) then (hidden -> q*d)

    # ---- shared spectral decoder (2x nearest-upsample + conv upsampler) -----
    decoder_channels: tuple[int, ...] = (64, 48, 32, 24, 16, 8, 4, 1)
    decoder_kernel: int = 3
    output_activation: str = "sigmoid"  # 'sigmoid' -> T_hat in [0, 1]

    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> "TemporalConvConfig":
        """Map the repo-wide :class:`ModelConfig` onto this config object.

        Shared night-level fields (``num_wavelength_bins``,
        ``n_frames_per_series``, ``num_queries``, ``spectral_latent_dim``,
        ``metadata_dim``, ``num_heads``, ``dropout``) plus the
        ``temporal_conv``-specific fields added to ``ModelConfig`` are mapped
        one-to-one.  ``fusion_hidden == 0`` means "use n_queries * latent_dim".
        """
        return cls(
            n_wavelength=cfg.num_wavelength_bins,
            n_frames=cfg.n_frames_per_series,
            n_queries=cfg.num_queries,
            latent_dim=cfg.spectral_latent_dim,
            metadata_dim=cfg.metadata_dim,
            metadata_enc_dim=cfg.metadata_enc_dim,
            metadata_enc_hidden=cfg.metadata_enc_hidden,
            param_dim=cfg.param_dim,
            param_decoder_hidden=cfg.param_decoder_hidden,
            n_heads=cfg.num_heads,
            dropout=cfg.dropout,
            x_encoder_channels=tuple(cfg.x_encoder_channels),
            x_encoder_kernel=cfg.x_encoder_kernel,
            x_encoder_stride=cfg.x_encoder_stride,
            s_encoder_channels=tuple(cfg.s_encoder_channels),
            s_encoder_kernel=cfg.s_encoder_kernel,
            s_encoder_stride=cfg.s_encoder_stride,
            encoder_pool_bins=cfg.encoder_pool_bins,
            fusion_hidden=None if cfg.fusion_hidden == 0 else cfg.fusion_hidden,
            fusion_layers=cfg.fusion_layers,
            decoder_channels=tuple(cfg.decoder_channels),
            decoder_kernel=cfg.decoder_kernel,
            output_activation=cfg.output_activation,
        )


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _conv_stack(
    channels: tuple[int, ...],
    kernel: int,
    stride: int,
    dropout: float,
) -> nn.Sequential:
    """Strided 1D-conv stack: 1 ch -> channels[0] -> ... -> channels[-1].

    Each stage halves the spectral length (stride ``stride``) while widening
    the receptive field, then normalizes with GroupNorm(1, *) so behaviour is
    independent of batch size.
    """
    blocks: list[nn.Module] = []
    cin = 1
    padding = kernel // 2
    for cout in channels:
        blocks += [
            nn.Conv1d(cin, cout, kernel, stride=stride, padding=padding),
            nn.GroupNorm(1, cout),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        cin = cout
    return nn.Sequential(*blocks)


def _broadcast_trailing(t: torch.Tensor, batch: int, feat: int) -> torch.Tensor:
    """Allow a per-sample vector ``(feat,)`` to be given as ``(B, feat)``."""
    if t.dim() == 1:
        return t.unsqueeze(0).expand(batch, -1)
    return t


def _broadcast_metadata(
    meta: torch.Tensor, batch: int, t: int, feat: int
) -> torch.Tensor:
    """Broadcast per-exposure metadata to ``(B, T, feat)``.

    Accepts ``(B, T, feat)`` (full batch), ``(T, feat)`` (one night, shared
    over the batch) or ``(feat,)`` (one exposure, shared over time + batch)
    and validates the trailing feature width.
    """
    if meta.dim() == 3:
        out = meta
    elif meta.dim() == 2:                      # (T, feat): single night
        out = meta.unsqueeze(0).expand(batch, -1, -1)
    elif meta.dim() == 1:                      # (feat,): single exposure vector
        out = meta.unsqueeze(0).unsqueeze(0).expand(batch, t, -1)
    else:
        raise ValueError(
            f"metadata must be (B, T, {feat}), ({t}, {feat}) or ({feat},), "
            f"got {tuple(meta.shape)}"
        )
    if out.shape != (batch, t, feat):
        raise ValueError(
            f"metadata must be (B, T, {feat}) = ({batch}, {t}, {feat}), "
            f"got {tuple(meta.shape)}"
        )
    return out


class MultiHeadCrossAttention(nn.Module):
    """Scaled dot-product multi-head cross attention.

    ``query (..., Lq, d)`` attends over ``key/value (..., Lk, d)`` and returns
    ``(..., Lq, d)`` plus the raw attention map ``(B, h, Lq, Lk)``.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim={dim} must be divisible by num_heads={num_heads}"
        )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross attention.

        Args:
            query: (B, Lq, dim)
            key:   (B, Lk, dim)
            value: (B, Lk, dim)

        Returns:
            (B, Lq, dim) output and (B, h, Lq, Lk) attention.
        """
        batch = query.shape[0]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        def _heads(x: torch.Tensor) -> torch.Tensor:
            # (B, L, dim) -> (B, h, L, head_dim)
            L = x.shape[1]
            return x.view(batch, L, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _heads(q), _heads(k), _heads(v)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, Lq, Lk)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                   # (B, h, Lq, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.dim)
        out = self.out_proj(out)
        return out, attn


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP residual block.

    ``query`` attends over ``key/value`` (query and key can differ in length),
    which is the building block used both by the Perceiver compressor (Q<<T)
    and the temporal decoder (T>>Q).
    """

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, ff_mult: int = 4
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B, Lq, dim) block output and (B, h, Lq, Lk) attention."""

        q = self.norm_q(query)
        k = self.norm_kv(key)
        v = self.norm_kv(value)
        out, attn = self.attn(q, k, v)
        x = query + self.drop1(out)
        x = x + self.drop1(self.ff(self.norm_ff(x)))
        return x, attn


# --------------------------------------------------------------------------- #
# 1. Spectral encoder for X  (shared 1D CNN over the exposures)
# --------------------------------------------------------------------------- #
class SpectralEncoder(nn.Module):
    """Shared 1D CNN compressing one spectrum ``(N,)`` to a ``d``-dim code.

    The *same instance* is applied to every exposure, i.e. it is never
    re-instantiated per frame.  It progressively downsamples the wavelength
    axis with strided convolutions (local receptive fields, no giant linear
    layer on the raw spectrum), then adaptively pools the remaining axis to a
    fixed number of bins per channel (preserving local spectral structure),
    flattens and projects to exactly ``latent_dim`` numbers per spectrum.

    Forward accepts any leading dimensions: ``(..., N) -> (..., d)``.
    """

    def __init__(
        self,
        n_wavelength: int,
        latent_dim: int,
        channels: tuple[int, ...] = (16, 32, 64, 96, 128),
        kernel: int = 7,
        stride: int = 2,
        dropout: float = 0.0,
        pool_bins: int = 16,) -> None:

        super().__init__()

        assert n_wavelength > 0 and latent_dim > 0 and pool_bins >= 1
        assert channels, "need at least one encoder channel"
        self.n_wavelength = n_wavelength
        self.latent_dim = latent_dim
        self.pool_bins = pool_bins
        self.stack = _conv_stack(channels, kernel, stride, dropout)
        self.readout = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(channels[-1] * pool_bins, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress spectrum/spectra.

        Args:
            x: (..., N) raw spectrum, wavelength on the last axis.

        Returns:
            (..., d) per-sample embedding.
        """
        lead = x.shape[:-1]                                 # (B, 73, N)
        x = x.reshape(-1, self.n_wavelength).unsqueeze(1)   # (M, 1, N) where M = B*73
        h = self.stack(x)                                   # (M, C, N')
        h = F.adaptive_avg_pool1d(h, self.pool_bins)        # (M, C, pool_bins)
        z = self.readout(h)                                 # (M, d)
        return z.view(*lead, self.latent_dim)


# --------------------------------------------------------------------------- #
# 1.5 Per-exposure metadata encoder  (B, T, P) -> (B, T, C)
# --------------------------------------------------------------------------- #
class MetadataEncoding(nn.Module):
    """Encode per-exposure metadata scalars into a conditioning code.

    Maps the ``P`` scalar conditions of every exposure (e.g. airmass,
    humidity, seeing, ...) to a ``C``-dim code with a small MLP **shared**
    over the ``T`` exposures (the same instance is applied to each one):

        metadata (B, T, P)  ->  M_code (B, T, C)

    The code is meant to be concatenated with the per-exposure spectral code
    ``Z (B, T, d)``, yielding ``(B, T, d + C)`` tokens for the temporal
    Perceiver, so the attention can condition on per-exposure metadata.

    Forward accepts any leading dimensions: ``(..., P) -> (..., C)``.
    """

    def __init__(
        self,
        n_metadata: int,
        enc_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_metadata > 0 and enc_dim > 0
        self.n_metadata = n_metadata
        self.enc_dim = enc_dim
        hidden = hidden_dim or max(enc_dim, 2 * n_metadata)
        self.net = nn.Sequential(
            nn.Linear(n_metadata, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, enc_dim),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """Encode metadata scalars. ``(..., P) -> (..., C)``."""
        lead = meta.shape[:-1]
        x = meta.reshape(-1, self.n_metadata)   # (M, P)
        code = self.net(x)                      # (M, C)
        return code.view(*lead, self.enc_dim)


# --------------------------------------------------------------------------- #
# 2. Temporal Perceiver compression (73 exposures -> 16 latent tokens)
# --------------------------------------------------------------------------- #
class TemporalPerceiver(nn.Module):
    """Compress ``Z (B, T, in)`` into ``L (B, Q, d)`` with ``Q`` learned queries.

    Uses a single multi-head cross-attention block where:

    * queries are the ``Q`` learned latent vectors (broadcast over the batch),
    * keys and values are the ``T`` per-exposure codes ``Z``.

    The ``Q`` latent tokens are learned parameters (this is *not* mean
    pooling over the time axis).  ``input_dim`` lets the keys/values be wider
    than the latent ``dim`` (e.g. ``64`` spectral + ``16`` metadata = ``80``):
    a learned ``Linear(input_dim -> dim)`` projects them down right at the
    input, so the latent space and everything downstream stay at ``dim``.
    """

    def __init__(
        self,
        n_queries: int,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 4,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.dim = dim
        self.input_dim = dim if input_dim is None else input_dim
        # Project wide input tokens (spectral + metadata) down to ``dim``.
        self.input_proj = (
            None if self.input_dim == dim else nn.Linear(self.input_dim, dim)
        )
        self.queries = nn.Parameter(
            torch.randn(n_queries, dim) * (dim ** -0.5) ##* or 0.02
        )
        self.block = _CrossAttentionBlock(dim, num_heads, dropout, ff_mult)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress the temporal axis.

        Args:
            z: (B, T, input_dim) per-exposure codes (spectral + optional
                metadata), projected to ``(B, T, dim)`` when ``input_dim``
                differs from ``dim``.

        Returns:
            L (B, Q, dim) latent tokens and (B, h, Q, T) attention.
        """
        batch, t, width = z.shape
        assert width == self.input_dim, (
            f"expected input_dim={self.input_dim}, got {width}"
        )
        if self.input_proj is not None:
            z = self.input_proj(z)                         # (B, T, dim)
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)   # (B, Q, dim) queries initialization shared across each night
        l, attn = self.block(query=q, key=z, value=z)  # (B,Q,dim),(B,h,Q,T)
        return l, attn


# --------------------------------------------------------------------------- #
# 4. Stellar spectrum S encoder (separate, independent 1D CNN)
# --------------------------------------------------------------------------- #
class StellarEncoder(nn.Module):
    """Independent 1D CNN encoding the night-constant stellar spectrum ``S``.

    This encoder has its **own** weights and never shares them with the X
    spectral encoder (a fresh ``_conv_stack`` + readout is created here).
    ``S`` is a single spectrum per night: ``(B, N) -> (B, d)``.
    """

    def __init__(
        self,
        n_wavelength: int,
        latent_dim: int,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel: int = 7,
        stride: int = 2,
        dropout: float = 0.0,
        pool_bins: int = 16,
    ) -> None:
        super().__init__()
        assert n_wavelength > 0 and latent_dim > 0 and pool_bins >= 1
        assert channels, "need at least one encoder channel"
        self.n_wavelength = n_wavelength
        self.latent_dim = latent_dim
        self.pool_bins = pool_bins
        self.stack = _conv_stack(channels, kernel, stride, dropout)
        self.readout = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(channels[-1] * pool_bins, latent_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Encode stellar spectra. ``(B, N) -> (B, d)``."""
        lead = s.shape[:-1]
        x = s.reshape(-1, self.n_wavelength).unsqueeze(1)
        h = self.stack(x)
        h = F.adaptive_avg_pool1d(h, self.pool_bins)
        code = self.readout(h)
        return code.view(*lead, self.latent_dim)


# --------------------------------------------------------------------------- #
# 6. Fusion MLP
# --------------------------------------------------------------------------- #
class FusionMLP(nn.Module):
    """Concatenating fusion MLP.

    ``concat[h_X (q*d), h_S (d)]`` -> MLP -> ``h_fused (q*d)``.

    The output width equals ``q*d`` on purpose: it is reshaped to ``(Q, d)``
    later without any projection/dimension loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_layers >= 1
        layers: list[nn.Module] = []
        cur = input_dim
        for i in range(n_layers):
            out = output_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(cur, out))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            cur = out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in) -> (B, output_dim)``."""
        return self.net(x)


# --------------------------------------------------------------------------- #
# 8. Temporal decoder (16 latent tokens -> 73 exposure codes)
# --------------------------------------------------------------------------- #
class TemporalDecoder(nn.Module):
    """Expand ``latent (B, Q, d)`` back to one code per exposure ``(B, T, d)``.

    ``T`` learned temporal (exposure-index) queries cross-attend over the ``Q``
    latent tokens, i.e. the reconstruction of each exposure is a learned
    function of the whole latent set -- not a broadcast of a mean vector.
    """

    def __init__(
        self,
        n_frames: int,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.dim = dim
        # 73 learned temporal query / position embeddings (T, d)
        self.queries = nn.Parameter(
            torch.randn(n_frames, dim) * (dim ** -0.5)
        )
        self.block = _CrossAttentionBlock(dim, num_heads, dropout, ff_mult)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Expand latent tokens to per-exposure codes.

        Args:
            latent: (B, Q, d).

        Returns:
            (B, T, d) per-exposure spectral embeddings ``H_T``.
        """
        batch, q, d = latent.shape
        assert d == self.dim, f"expected dim={self.dim}, got {d}"
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)   # (B, T, d)
        h_t, _ = self.block(query=queries, key=latent, value=latent)  # (B,T,d)
        return h_t


# --------------------------------------------------------------------------- #
# 9. Spectral decoder (shared 1D CNN decoder)
# --------------------------------------------------------------------------- #
# class SpectralDecoder(nn.Module):
#     """Shared 1D CNN decoder mapping a ``d``-dim code back to ``N`` samples.

#     The code is first expanded to a small 2-D feature map (``d -> C0 x L0``,
#     ``L0 ~ N / 2**stages``), which is then progressively upsampled by 2x
#     nearest interpolation followed by convolutions whose channel count decays
#     towards 1, mirroring the encoder hierarchy.  A final interpolation snaps
#     the length exactly to ``N`` (no ``64 -> 51556`` giant linear layer).
#     """

#     def __init__(
#         self,
#         n_wavelength: int,
#         latent_dim: int,
#         channels: tuple[int, ...] = (64, 48, 32, 24, 16, 8, 4, 1),
#         kernel: int = 4,
#         dropout: float = 0.0,
#         output_activation: str = "sigmoid",
#     ) -> None:
#         super().__init__()
#         assert n_wavelength > 0 and latent_dim > 0
#         assert all(c > 0 for c in channels) and channels[-1] == 1
#         assert output_activation in ("sigmoid", "none")
#         self.n_wavelength = n_wavelength
#         self.latent_dim = latent_dim
#         self.c0 = channels[0]
#         self.n_stages = len(channels) - 1
#         # base length such that base * 2**stages just covers N
#         self.l0 = max(1, math.ceil(n_wavelength / (2 ** self.n_stages)))
#         self.final_len = self.l0 * (2 ** self.n_stages)

#         self.expand = nn.Linear(latent_dim, self.c0 * self.l0)

#         # Nearest 2x upsampling + same-padding conv at every stage doubles the
#         # length exactly and keeps the upsampler fully convolutional.  Odd
#         # kernel is required so the conv preserves the length it receives.
#         assert kernel % 2 == 1, "decoder conv kernel must be odd"
#         blocks: list[nn.Module] = []
#         cin = self.c0
#         pad = (kernel - 1) // 2
#         for cout in channels[1:]:
#             blocks += [
#                 nn.Upsample(scale_factor=2, mode="nearest"),
#                 nn.Conv1d(cin, cout, kernel, padding=pad),
#             ]
#             if cout > 1:  # keep final stage linear (raw logit channel)
#                 blocks += [nn.GroupNorm(1, cout), nn.GELU(), nn.Dropout(dropout)]
#             cin = cout
#         self.upsampler = nn.Sequential(*blocks)
#         self.activation = torch.sigmoid if output_activation == "sigmoid" else nn.Identity()

#     def forward(self, code: torch.Tensor) -> torch.Tensor:
#         """Decode spectrum/spectra.

#         Args:
#             code: (..., d).

#         Returns:
#             (..., N) predicted transmission in [0, 1] if sigmoid enabled.
#         """
#         lead = code.shape[:-1]
#         x = code.reshape(-1, self.latent_dim)              # (M, d)
#         x = self.expand(x)                                 # (M, C0 * L0)
#         x = x.view(-1, self.c0, self.l0)                   # (M, C0, L0)
#         x = self.upsampler(x)                              # (M, 1, final_len)
#         x = x.squeeze(1)                                   # (M, final_len)
#         if self.final_len != self.n_wavelength:
#             x = F.interpolate(
#                 x.unsqueeze(1),
#                 size=self.n_wavelength,
#                 mode="linear",
#                 align_corners=False,
#             ).squeeze(1)
#         x = self.activation(x)                             # (M, N)
#         return x.view(*lead, self.n_wavelength)


# --------------------------------------------------------------------------- #
# 9.5 Per-exposure parameter decoder  (B, T, d) -> (B, T, P)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 10. Top-level model
# --------------------------------------------------------------------------- #
@ModelRegistry.register("temporal_conv")
class TelluricModel(nn.Module):
    """Whole-night telluric regressor (CNN spectral AE + Perceiver bottleneck).

    Inputs:
        observed (B, T, N): observed spectra X = T * S for one night.
        stellar  (B, N) or (N,): night-constant stellar spectrum S.
        metadata (B, T, P), (T, P) or (P,): per-exposure metadata scalars,
            encoded and concatenated with the spectral codes before the
            temporal Perceiver.

    Output (ModelOutput):
        params (B, T, P): predicted per-exposure telluric parameters param_pred.
        latent (B, Q*d): flattened Perceiver night summary h_X.
        attention_weights (B, Q, T): query -> exposure attention (mean over h).
        intermediate_features: dictionary with every intermediate tensor.
    """

    def __init__(self, config: TemporalConvConfig | ModelConfig) -> None:
        # Accept the repo-wide ModelConfig (registry / Lightning path) as well
        # as the native TemporalConvConfig (standalone / scripting path).
        if isinstance(config, ModelConfig):
            config = TemporalConvConfig.from_model_config(config)
        super().__init__()
        self.config = config
        n_w = config.n_wavelength
        t = config.n_frames
        q = config.n_queries
        d = config.latent_dim
        p = config.metadata_dim
        assert n_w > 0 and t > 0 and q > 0 and d > 0
        assert d % config.n_heads == 0, (
            f"latent_dim={d} must be divisible by n_heads={config.n_heads}"
        )

        self.q = q
        self.d = d
        self.t = t
        self.n_wavelength = n_w
        self.metadata_dim = p
        self.param_dim = config.param_dim

        c = config.metadata_enc_dim                     # metadata code width C
        fusion_hidden = config.fusion_hidden or (q * d)      # default 1024
        fusion_in = q * d + d                                # 1024 + 64 (h_X + h_S)
        fusion_out = q * d                                   # 1024 == 16*64

        # 1. shared spectral CNN over the exposures of X
        self.spectral_encoder = SpectralEncoder(
            n_w, d, config.x_encoder_channels,
            config.x_encoder_kernel, config.x_encoder_stride, config.dropout,
            config.encoder_pool_bins,
        )
        # 1.5 per-exposure metadata encoder: (B, T, P) -> (B, T, C)
        self.metadata_encoding = (
            MetadataEncoding(p, c, config.metadata_enc_hidden, config.dropout)
            if p > 0 else None
        )
        # 2. temporal Perceiver: concat[Z (d), M_code (C)] = (d + C)-wide tokens
        #    projected to d at the input; 73 exposures -> 16 latent tokens
        self._token_dim = d + (c if p > 0 else 0)
        self.perceiver = TemporalPerceiver(
            q, d, config.n_heads, config.dropout, input_dim=self._token_dim
        )
        # 4. independent stellar encoder for S (own weights)
        self.stellar_encoder = StellarEncoder(
            n_w, d, config.s_encoder_channels,
            config.s_encoder_kernel, config.s_encoder_stride, config.dropout,
            config.encoder_pool_bins,
        )
        # 6. fusion MLP
        self.fusion = FusionMLP(
            fusion_in, fusion_hidden, fusion_out,
            n_layers=config.fusion_layers, dropout=config.dropout,
        )
        # 8. temporal decoder: 16 latent tokens -> 73 per-exposure codes
        self.temporal_decoder = TemporalDecoder(
            t, d, config.n_heads, config.dropout
        )
        # 9. per-exposure parameter decoder: d-dim code -> P parameters
        self.param_decoder = ParamDecoder(
            d, config.param_dim, config.param_decoder_hidden,
            dropout=config.dropout,
        )

        self._intermediate: dict[str, torch.Tensor] = {}

    # -- parameter book-keeping ---------------------------------------------- #
    def parameter_counts(self) -> dict[str, int]:
        """Per-component trainable parameter counts."""
        components: dict[str, nn.Module] = {
            "SpectralEncoder (X)": self.spectral_encoder,
            "MetadataEncoding": self.metadata_encoding,
            "TemporalPerceiver": self.perceiver,
            "StellarEncoder (S)": self.stellar_encoder,
            "FusionMLP": self.fusion,
            "TemporalDecoder": self.temporal_decoder,
            "ParamDecoder": self.param_decoder,
        }
        return {
            name: sum(p.numel() for p in m.parameters() if p.requires_grad)
            for name, m in components.items()
            if m is not None
        }

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        observed: torch.Tensor,
        stellar: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Regress the whole-night telluric transmission.

        Args:
            observed: (B, T, N) observed spectrum X = T_tell * S.
            stellar: (B, N) or (N,) night-constant stellar spectrum S.
            metadata: (B, T, P), (T, P) or (P,) per-exposure metadata scalars
                (P == metadata_dim), e.g. airmass/humidity per exposure.
            time: reserved (B, T) continuous exposure times; currently unused
                but accepted for API parity with NightTelluricModule.

        Returns:
            ModelOutput(params=(B, T, P), latent=(B, Q*d),
                        attention_weights=(B, Q, T),
                        intermediate_features={...}).
        """
        del time  # reserved for a future continuous-time temporal decoder
        batch, t, n = observed.shape
        if t != self.t:
            raise ValueError(
                f"Expected T={self.t} exposures per night, got {t}. "
                "Set TemporalConvConfig.n_frames accordingly."
            )
        if n != self.n_wavelength:
            raise ValueError(
                f"Expected N={self.n_wavelength} wavelength samples, got {n}."
            )
        device = observed.device

        # 1. shared spectral CNN over the exposures (same weights for all t)
        z = self.spectral_encoder(observed)          # (B, T, d)
        assert z.shape == (batch, self.t, self.d), z.shape

        # 1.5 per-exposure metadata -> C-dim code (zeros when absent/disabled)
        if self.metadata_dim > 0:
            if metadata is not None:
                m = _broadcast_metadata(
                    metadata, batch, self.t, self.metadata_dim
                )
            else:
                m = torch.zeros(
                    batch, self.t, self.metadata_dim, device=device
                )
            m_code = self.metadata_encoding(m)       # (B, T, C)
        else:
            m_code = None

        # 1.6 concat spectral + metadata codes -> Perceiver input tokens
        #     (B, T, d) + (B, T, C) = (B, T, 80) for the default d=64, C=16
        tokens = (
            z if m_code is None else torch.cat([z, m_code], dim=-1)
        )
        assert tokens.shape == (batch, self.t, self._token_dim), tokens.shape

        # 2. Perceiver: 80-dim tokens projected to d=64 at the input;
        #    73 exposures -> 16 learned latent tokens
        l, attn_heads = self.perceiver(tokens)       # (B, Q, d), (B, h, Q, T)
        assert l.shape == (batch, self.q, self.d), l.shape

        # 3. flatten X representation  (16 * 64 == 1024)
        h_x = l.reshape(batch, self.q * self.d)      # (B, 1024)

        # 4. stellar code (separate encoder, own weights)
        if stellar is not None:
            s = _broadcast_trailing(stellar, batch, self.n_wavelength)
            if s.shape != (batch, self.n_wavelength):
                raise ValueError(
                    f"stellar must be ({batch}, {self.n_wavelength}) or "
                    f"({self.n_wavelength},), got {tuple(stellar.shape)}"
                )
            h_s = self.stellar_encoder(s)            # (B, d)
        else:
            h_s = torch.zeros(batch, self.d, device=device)
        assert h_s.shape == (batch, self.d), h_s.shape

        # 6. fusion  concat[h_X, h_S] -> MLP -> h_fused (B, q*d)
        #    (metadata now enters earlier, concatenated with the spectral
        #    codes feeding the Perceiver, not here)
        fusion_vec = torch.cat([h_x, h_s], dim=-1)   # (B, q*d + d) = (B, 1088)
        assert fusion_vec.shape == (
            batch, self.q * self.d + self.d,
        ), fusion_vec.shape
        h_fused = self.fusion(fusion_vec)            # (B, q*d) == (B, 1024)
        assert h_fused.shape == (batch, self.q * self.d), h_fused.shape

        # 7. reshape 1024 == 16*64 -> structured latent tokens (B, Q, d)
        latent = h_fused.view(batch, self.q, self.d)
        assert latent.shape == (batch, self.q, self.d), latent.shape

        # 8. temporal decoder: 16 tokens -> 73 per-exposure codes
        h_t = self.temporal_decoder(latent)          # (B, T, d)
        assert h_t.shape == (batch, self.t, self.d), h_t.shape

        # 9. shared per-exposure param decoder: d-code -> P parameters
        param_pred = self.param_decoder(h_t)         # (B, T, P)
        assert param_pred.shape == (
            batch, self.t, self.param_dim,
        ), param_pred.shape

        attn = attn_heads.mean(dim=1)                # (B, Q, T) mean over heads

        self._intermediate = {
            "z": z,                      # (B, T, d)
            "metadata_code": m_code,     # (B, T, C) or None
            "tokens": tokens,            # (B, T, d + C)
            "l": l,                      # (B, Q, d)
            "h_x": h_x,                  # (B, q*d)
            "h_s": h_s,                  # (B, d)
            "fusion": fusion_vec,        # (B, q*d + d)
            "h_fused": h_fused,          # (B, q*d)
            "latent": latent,            # (B, Q, d)
            "temporal": h_t,             # (B, T, d)
            "param_pred": param_pred,    # (B, T, P)
        }
        return ModelOutput(
            params=param_pred,
            latent=h_x,
            attention_weights=attn,
            intermediate_features=self._intermediate,
        )
