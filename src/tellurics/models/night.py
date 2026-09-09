"""Whole-night telluric estimator (CNN spectral AE + temporal Perceiver).

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
    X (B, 73, 51556)              M (B, 73, 3)       time_hours (B, 73)
        | SpectralEncoder         | MetadataEncoder   | TimeEncoder (MLP)
        v (one 1D CNN/exposure)   v (shared MLP)      v (shared MLP)
    Z (B, 73, 64)            M_code (B, 73, 16)   z_time (B, 73, 16)
        |                        |                     |
        +-------- concat[Z, M_code, z_time] --> tokens (B, 73, 96)
                            | pre_temporal Linear(96 -> 64)
                            v
                TemporalPerceiver (cross-attn, 16 latent queries)
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
* The S spectrum, the per-exposure metadata and the exposure times enter
  the model only as small codes (``h_S`` = 64; metadata -> 16 dims;
  ``time_hours`` -> 16 dims via :class:`TimeEncoder`), concatenated with the
  spectral codes *before* the temporal Perceiver, so the fusion stays small.
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
    cfg = TelluricEstimatorConfig(n_wavelength=1024, n_frames=12, n_queries=8,
                             latent_dim=32, metadata_dim=3, n_heads=4,
                             x_encoder_channels=(8, 16, 32),
                             s_encoder_channels=(8, 16, 32))
    model = TelluricEstimator(cfg)
    out = model(torch.randn(2, 12, 1024),            # X = T*S
                stellar=torch.randn(2, 1024),        # S (stellar)
                metadata=torch.randn(2, 12, 3))      # per-exposure metadata
    assert out.params.shape == (2, 12, 20)           # param_pred (B, T, P)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from tellurics.configs.model import ModelConfig
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry

from .decoders import ParamDecoder
from .encoders import MetadataEncoder, SpectralEncoder, StellarEncoder, TimeEncoder
from .fusion import FusionMLP
from .temporal import TemporalDecoder, TemporalPerceiver

__all__ = ["TelluricEstimatorConfig", "TelluricEstimator"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TelluricEstimatorConfig:
    """Hyper-parameters of the whole-night telluric estimator.

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
    time_enc_dim: int = 16             # width k of the time code (per exposure)
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

    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> "TelluricEstimatorConfig":
        """Map the repo-wide :class:`ModelConfig` onto this config object.

        Shared night-level fields (``num_wavelength_bins``,
        ``n_frames_per_series``, ``num_queries``, ``spectral_latent_dim``,
        ``metadata_dim``, ``num_heads``, ``dropout``) plus the estimator
        specific fields added to ``ModelConfig`` are mapped one-to-one.
        ``fusion_hidden == 0`` means "use n_queries * latent_dim".
        """
        return cls(
            n_wavelength=cfg.num_wavelength_bins,
            n_frames=cfg.n_frames_per_series,
            n_queries=cfg.num_queries,
            latent_dim=cfg.spectral_latent_dim,
            metadata_dim=cfg.metadata_dim,
            metadata_enc_dim=cfg.metadata_enc_dim,
            metadata_enc_hidden=cfg.metadata_enc_hidden,
            time_enc_dim=cfg.time_enc_dim,
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
        )


# --------------------------------------------------------------------------- #
# Small broadcast helpers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
@ModelRegistry.register("telluric_estimator")
class TelluricEstimator(nn.Module):
    """Whole-night telluric parameter estimator (CNN AE + Perceiver bottleneck).

    Inputs:
        observed (B, T, N): observed spectra X = T * S for one night.
        stellar  (B, N) or (N,): night-constant stellar spectrum S.
        metadata (B, T, P), (T, P) or (P,): per-exposure metadata scalars,
            encoded and concatenated with the spectral codes before the
            temporal Perceiver.
        time     (B, T): continuous exposure times (``time_hours``) encoded by
            the :class:`TimeEncoder`; None -> a zero offset is used.

    Output (ModelOutput):
        params (B, T, P): predicted per-exposure telluric parameters param_pred.
        latent (B, Q*d): flattened Perceiver night summary h_X.
        attention_weights (B, Q, T): query -> exposure attention (mean over h).
        intermediate_features: dictionary with every intermediate tensor.
    """

    def __init__(self, config: TelluricEstimatorConfig | ModelConfig) -> None:
        # Accept the repo-wide ModelConfig (registry / Lightning path) as well
        # as the native TelluricEstimatorConfig (standalone / scripting path).
        if isinstance(config, ModelConfig):
            config = TelluricEstimatorConfig.from_model_config(config)
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
        self.time_enc_dim = config.time_enc_dim
        self.param_dim = config.param_dim

        c = config.metadata_enc_dim                     # metadata code width C
        k = config.time_enc_dim                         # time-code width k
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
        self.metadata_encoder = (
            MetadataEncoder(p, c, config.metadata_enc_hidden, config.dropout)
            if p > 0 else None
        )
        # 1.55 continuous exposure-time encoder: time_hours (B, T) -> (B, T, k)
        self.time_encoder = TimeEncoder(out_dim=k)
        # 1.6 pre-temporal projection: concat[Z (d), M_code (C), time (k)]
        #     = (d + C + k)-wide tokens -> d-wide Perceiver tokens.
        self._token_dim = d + (c if p > 0 else 0) + k
        self.pre_temporal = nn.Linear(self._token_dim, d)
        # 2. temporal Perceiver: d-dim exposure tokens -> Q latent tokens
        #    (input_dim == dim, so it performs no extra internal projection)
        self.temporal_perceiver = TemporalPerceiver(
            q, d, config.n_heads, config.dropout
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
        # 8. temporal decoder: Q latent tokens -> T per-exposure codes
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
            "MetadataEncoder": self.metadata_encoder,
            "TimeEncoder": self.time_encoder,
            "PreTemporal": self.pre_temporal,
            "TemporalPerceiver": self.temporal_perceiver,
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
        """Regress the whole-night telluric parameters.

        Args:
            observed: (B, T, N) observed spectrum X = T_tell * S.
            stellar: (B, N) or (N,) night-constant stellar spectrum S.
            metadata: (B, T, P), (T, P) or (P,) per-exposure metadata scalars
                (P == metadata_dim), e.g. airmass/humidity per exposure.
            time: (B, T) continuous exposure times (e.g. ``time_hours`` from
                the data), used to build the per-exposure time code ``z_time``.
                If None (shape-only calls), a zero offset is used.

        Returns:
            ModelOutput(params=(B, T, P), latent=(B, Q*d),
                        attention_weights=(B, Q, T),
                        intermediate_features={...}).
        """
        batch, t, n = observed.shape
        if t != self.t:
            raise ValueError(
                f"Expected T={self.t} exposures per night, got {t}. "
                "Set TelluricEstimatorConfig.n_frames accordingly."
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
            m_code = self.metadata_encoder(m)        # (B, T, C)
        else:
            m_code = None

        # 1.55 continuous exposure time (time_hours) -> k-dim time code.
        #     (B, T) -> (B, T, k); acts as a continuous positional encoding
        #     over the exposure axis.
        if time is None:
            # No exposure times supplied (e.g. pure-shape calls): use a zero
            # offset so the code path still runs. Production batches should
            # pass time = time_hours (B, T).
            time = torch.zeros(batch, self.t, device=device)
        z_time = self.time_encoder(time)             # (B, T, k)

        # 1.6 concat spectral + metadata + time codes, then project to d:
        #     (B,T,d) + (B,T,C) + (B,T,k) = (B,T,96) for d=64, C=16, k=16
        parts = [z]
        if m_code is not None:
            parts.append(m_code)
        parts.append(z_time)
        tokens_concat = torch.cat(parts, dim=-1)     # (B, T, d + C + k)
        assert tokens_concat.shape == (
            batch, self.t, self._token_dim,
        ), tokens_concat.shape

        tokens = self.pre_temporal(tokens_concat)    # (B, T, d)  e.g. 96 -> 64
        assert tokens.shape == (batch, self.t, self.d), tokens.shape

        # 2. Perceiver: d-dim exposure tokens -> Q learned latent tokens
        l, attn_heads = self.temporal_perceiver(tokens)  # (B,Q,d),(B,h,Q,T)
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
        #    (metadata enters earlier, concatenated with the spectral codes
        #    feeding the Perceiver, not here)
        fusion_vec = torch.cat([h_x, h_s], dim=-1)   # (B, q*d + d) = (B, 1088)
        assert fusion_vec.shape == (
            batch, self.q * self.d + self.d,
        ), fusion_vec.shape
        h_fused = self.fusion(fusion_vec)            # (B, q*d) == (B, 1024)
        assert h_fused.shape == (batch, self.q * self.d), h_fused.shape

        # 7. reshape 1024 == 16*64 -> structured latent tokens (B, Q, d)
        latent = h_fused.view(batch, self.q, self.d)
        assert latent.shape == (batch, self.q, self.d), latent.shape

        # 8. temporal decoder: Q tokens -> T per-exposure codes
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
            "time_code": z_time,         # (B, T, k)
            "tokens": tokens_concat,     # (B, T, d + C + k) concatenated codes
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
