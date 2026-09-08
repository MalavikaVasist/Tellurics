"""Night-level Telluric regression model (shared spectral MLP + Perceiver).

Task
----
A full night of observations has ``T`` exposures, each with ``N`` spectral
samples. The observation is ``X(t, l) = T_tell(t, l) * S(l)`` where ``S`` is an
(approximately night-constant) stellar spectrum. We regress the whole-night
telluric transmission:

    X: (B, T, N)  ->  T_hat: (B, T, N)        loss = MSE(T - T_hat)

Architecture (one night in, whole night out)
--------------------------------------------
    (B, T, N)                       observed spectra X = T*S
        |  shared spectral MLP  (one MLP, applied independently to every
        v                          exposure -- weights never re-instantiated
    (B, T, d)                       per frame)
        |  + learned temporal (exposure-index) position embedding
        v
    (B, T, d)  Z
        |  ONE Perceiver cross-attention layer with Q learned queries
        v
    (B, Q, d)  Y            (each query attends over the T exposures)
        |
        +--> flatten -> (B, Q*d)  [= the 1024-dim night summary when d=64,Q=16]
        |
        v
    context = MLP( concat[ flatten(Y), S_code, metadata_embed ] )   (B, H)
        |
        v
    T_hat(t) = shared per-frame decoder( concat[ Z(t), context ] )  (B, T, N)

Why the decoder is per-frame instead of a single dense 1024 -> (T*N) head
------------------------------------------------------------------------
A single 1024-dim vector fundamentally cannot emit ``T*N ~ 73*51556`` numbers,
so the flattened summary cannot be the *sole* input of the final head. We keep
the flattened summary (as required) but use it to *condition* a decoder that
expands each per-exposure code ``Z(t)`` back to a transmission. The decoder's
weights are shared across all ``T`` exposures (same trick as the encoder), so
the added cost is a single ``Linear(d+H, N)`` plus the context MLP.

Stellar S is only ever injected as a *compact code*
---------------------------------------------------
``S_code = shared_encoder(S)`` (a ``d``-dim vector) plus a small projection.
The raw 51,556-dim ``S`` never enters the fusion/decoder, which is what keeps
the fusion layer small and lets the model learn to remove the stellar imprint
from ``X`` so the decoder can emit a pure telluric transmission.

Regularization / caveats (only ~4500 independent nights!)
----------------------------------------------------------
* The dense ``51556 -> 2048`` spectral MLP alone is ~105M parameters (see
  ``count_parameters``). With ~4500 nights this will overfit hard and is slow.
  Strongly consider an alternative Stage-1 encoder before training at full N:
    - bin/avg-pool the spectrum to a smaller grid (e.g. 4096) before the MLP;
    - a small 1D-conv spectral stem (params independent of N);
    - a frozen/pretrained PCA or per-wavelength linear+BN layer.
  Everything downstream (Perceiver, fusion, decoder) is tiny by comparison, so
  the parameter count is dominated by Stage 1.
* Use weight decay (AdamW), Dropout, LayerNorm, night-level CV, and possibly
  label smoothing / spectral augmentation. Do NOT split exposures of the same
  night across train/val/test (see ``tellurics.data.night`` for a night-level
  split helper).

Example (tiny dims, for shape checks)
-------------------------------------
    config = ModelConfig(architecture=ModelArchitecture.NIGHT_PERCEIVER,
                         num_wavelength_bins=1024, n_frames_per_series=12,
                         spectral_latent_dim=32,
                         spectral_encoder_dims=[256, 128],
                         num_queries=8, hidden_dim=64, metadata_dim=6)
    model = PerceiverNightModel(config)
    out = model(torch.randn(2, 12, 1024),           # observed X = T*S
                torch.randn(2, 1024),               # stellar S
                torch.randn(2, 6))                  # per-night metadata
    assert out.telluric.shape == (2, 12, 1024)
"""

import torch
import torch.nn as nn

from tellurics.configs.model import ModelConfig
from tellurics.models.fusion import AtmosphericEmbedding
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


class SpectralMLPEncoder(nn.Module):
    """Shared MLP mapping one (N-dim) spectrum to a d-dim code.

    The exact same module is applied to every exposure, i.e. it is never
    re-instantiated per frame. It also encodes the night-constant stellar
    spectrum ``S`` so that ``S`` only ever enters the fusion as a compact code.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        blocks = []
        prev = input_dim
        for width in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.Linear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            prev = width
        blocks.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*blocks)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spectrum/spectra.

        Args:
            x: (..., N) raw spectrum (spectral dim is the last axis).

        Returns:
            (..., d) spectral code.
        """
        return self.net(x)


class TemporalPositionalEncoding(nn.Module):
    """Learned exposure-index embedding added to the per-frame codes.

    Includes the time/position of each of the T exposures inside the night so
    the Perceiver can use ordering information. ``time`` (optional continuous
    hours) can be supplied to modulate the learned embedding.
    """

    def __init__(self, max_frames: int, dim: int, dropout: float = 0.1,
                 enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.dropout = nn.Dropout(dropout)
        self.pos = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.time_proj = nn.Linear(1, dim) if enabled else None

    def forward(self, z: torch.Tensor,
                time: torch.Tensor | None = None) -> torch.Tensor:
        """Add position to per-frame codes.

        Args:
            z: (B, T, d).
            time: optional (B, T) continuous exposure time (hours).

        Returns:
            (B, T, d) position-modulated codes.
        """
        if not self.enabled:
            return self.dropout(z)
        z = z + self.pos[:, : z.size(1), :]
        if time is not None and self.time_proj is not None:
            z = z + self.time_proj(time.unsqueeze(-1))
        return self.dropout(z)


class PerceiverCrossAttention(nn.Module):
    """Single cross-attention layer compressing T exposures into Q latents.

    Q are *learned query vectors*, broadcast across the batch. Keys/values are
    projected from the T exposure codes. Returns attended latents ``Y`` and the
    QxT attention map (which query summarizes which part of the night).
    """

    def __init__(self, dim: int, num_queries: int, num_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        # Learned queries: (Q, d).
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * (dim ** -0.5))
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend learned queries to the exposures.

        Args:
            z: exposure codes (B, T, d).

        Returns:
            y: latent summaries (B, Q, d).
            attn: query->exposure attention weights (B, Q, T).
        """
        batch, t, d = z.shape
        q = self.to_q(self.queries).unsqueeze(0).expand(batch, -1, -1)  # (B, Q, d)
        k = self.to_k(z)  # (B, T, d)
        v = self.to_v(z)  # (B, T, d)

        scores = q @ k.transpose(-1, -2) * (d ** -0.5)  # (B, Q, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # (B, Q, d)
        y = self.norm(y)
        return y, attn


@ModelRegistry.register("perceiver_night")
class PerceiverNightModel(nn.Module):
    """Whole-night telluric regressor (shared spectral MLP + one Perceiver layer).

    Inputs:
        observed (B, T, N): observed spectra X = T * S for one night.
        stellar  (B, N) or (N,): night-constant stellar spectrum S.
        metadata (B, P) or (P,): per-night scalar metadata (optional).

    Output (ModelOutput):
        telluric (B, T, N): predicted whole-night telluric transmission in [0, 1].
        latent (B, Q*d): flattened Perceiver night summary.
        attention_weights (B, Q, T): learned query -> exposure attention.
        intermediate_features["z"]: per-exposure codes (B, T, d).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_wavelength = config.predict_spectral_dim or config.num_wavelength_bins
        self.d = config.spectral_latent_dim
        self.t = config.n_frames_per_series
        self.q = config.num_queries
        self.hidden = config.hidden_dim

        # --- Stage 1: shared per-exposure spectral MLP ---
        self.encoder = SpectralMLPEncoder(
            input_dim=self.n_wavelength,
            hidden_dims=config.spectral_encoder_dims,
            latent_dim=self.d,
            dropout=config.dropout,
        )

        # --- Temporal / positional info of the T exposures ---
        self.temporal = TemporalPositionalEncoding(
            max_frames=self.t,
            dim=self.d,
            dropout=config.dropout,
            enabled=config.use_time_embed,
        )

        # --- Stage 2: one Perceiver cross-attention layer ---
        self.perceiver = PerceiverCrossAttention(
            dim=self.d,
            num_queries=self.q,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # --- Compact stellar embedding ---
        self.stellar_proj = nn.Sequential(
            nn.Linear(self.d, self.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --- Metadata embedding (per-night scalars) ---
        if config.metadata_dim > 0:
            self.metadata_embed = AtmosphericEmbedding(
                num_params=config.metadata_dim,
                embed_dim=self.hidden,
                dropout=config.dropout,
            )
        else:
            self.metadata_embed = None

        # --- Fusion: flatten(Q*d) + stellar + metadata -> context ---
        self.context_mlp = nn.Sequential(
            nn.Linear(self.q * self.d + self.hidden + self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )

        # --- Shared per-frame decoder: (d + hidden) -> N, sigmoid in [0, 1] ---
        self.decoder = nn.Sequential(
            nn.Linear(self.d + self.hidden, self.n_wavelength),
            nn.Sigmoid(),
        )

    def _broadcast_to_batch(
        self, tensor: torch.Tensor | None, batch: int, device: torch.device
    ) -> torch.Tensor | None:
        """Allow stellar/metadata to be given as 1D (per-sample) or (B, ...)."""
        if tensor is None:
            return None
        if tensor.dim() == 1:
            return tensor.unsqueeze(0).expand(batch, -1)
        return tensor

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
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
            metadata: (B, P) or (P,) per-night metadata scalars.
            time: optional (B, T) continuous exposure times (hours).

        Returns:
            ModelOutput(telluric=(B, T, N), latent=(B, Q*d),
                        attention_weights=(B, Q, T)).
        """
        batch, t, n = observed.shape
        if t != self.t:
            raise ValueError(
                f"Expected T={self.t} exposures per night, got {t}. "
                "Set ModelConfig.n_frames_per_series accordingly."
            )
        if n != self.n_wavelength:
            raise ValueError(
                f"Expected N={self.n_wavelength} wavelength samples, got {n}."
            )
        device = observed.device

        # --- Stage 1: shared MLP over each exposure independently ---
        z = self.encoder(observed.reshape(batch * t, n))  # (B*T, d)
        z = z.reshape(batch, t, self.d)  # (B, T, d)

        # --- Temporal / positional info ---
        z = self.temporal(z, time=time)  # (B, T, d)

        # --- Stage 2: one Perceiver cross-attention layer ---
        y, attn = self.perceiver(z)  # (B, Q, d), (B, Q, T)
        night_summary = y.reshape(batch, self.q * self.d)  # (B, Q*d) = 1024 for d=64,Q=16

        # --- Compact stellar code (raw S never reaches the fusion) ---
        stellar = self._broadcast_to_batch(stellar, batch, device)
        if stellar is not None:
            s_code = self.encoder(stellar)  # (B, d) -- shares Stage-1 weights
            s_emb = self.stellar_proj(s_code)  # (B, hidden)
        else:
            s_emb = torch.zeros(batch, self.hidden, device=device)

        # --- Metadata embedding ---
        metadata = self._broadcast_to_batch(metadata, batch, device)
        if metadata is not None and self.metadata_embed is not None:
            m_emb = self.metadata_embed(metadata)  # (B, hidden)
        else:
            m_emb = torch.zeros(batch, self.hidden, device=device)

        # --- Fusion + regression head ---
        context = self.context_mlp(
            torch.cat([night_summary, s_emb, m_emb], dim=-1)
        )  # (B, hidden)

        # --- Shared per-frame decoder to the full spectral grid ---
        ctx_expanded = context.unsqueeze(1).expand(batch, t, -1)  # (B, T, hidden)
        frame_feat = torch.cat([z, ctx_expanded], dim=-1)  # (B, T, d+hidden)
        telluric = self.decoder(frame_feat.reshape(batch * t, -1))  # (B*T, N)
        telluric = telluric.reshape(batch, t, self.n_wavelength)  # (B, T, N)

        return ModelOutput(
            telluric=telluric,
            latent=night_summary,
            attention_weights=attn,
            intermediate_features={
                "z": z,
                "y": y,
                "stellar_code": s_emb,
                "metadata_code": m_emb,
                "context": context,
            },
        )
