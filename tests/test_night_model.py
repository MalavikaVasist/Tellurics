"""Tests for the whole-night telluric parameter estimator (TelluricEstimator)."""

import os

import pytest
import torch
import torch.nn.functional as F

from tellurics.configs.model import ModelArchitecture, ModelConfig
from tellurics.models.night import TelluricEstimator, TelluricEstimatorConfig
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


def _config(
    n_wave: int = 1024,
    t: int = 12,
    q: int = 8,
    d: int = 32,
    metadata_dim: int = 3,
    heads: int = 4,
) -> TelluricEstimatorConfig:
    """Small dims so the tests stay fast on CPU."""
    return TelluricEstimatorConfig(
        n_wavelength=n_wave,
        n_frames=t,
        n_queries=q,
        latent_dim=d,
        metadata_dim=metadata_dim,
        n_heads=heads,
        x_encoder_channels=(8, 16, 32),
        s_encoder_channels=(8, 16, 32),
        dropout=0.0,
    )


class TestTelluricEstimatorShapes:
    def test_shapes(self) -> None:
        cfg = _config()
        model = TelluricEstimator(cfg)

        observed = torch.randn(2, cfg.n_frames, cfg.n_wavelength)
        stellar = torch.randn(2, cfg.n_wavelength)
        metadata = torch.randn(2, cfg.n_frames, cfg.metadata_dim)
        time = torch.randn(2, cfg.n_frames)

        out = model(observed, stellar=stellar, metadata=metadata, time=time)

        assert isinstance(out, ModelOutput)
        assert out.params is not None
        assert out.params.shape == (2, cfg.n_frames, cfg.param_dim)
        assert out.latent.shape == (2, cfg.n_queries * cfg.latent_dim)
        assert out.attention_weights.shape == (
            2, cfg.n_queries, cfg.n_frames,
        )
        # the spectral (transmission) head is gone: params only
        assert out.telluric is None

        # intermediate features have exactly the documented shapes
        feats = out.intermediate_features
        assert feats["z"].shape == (2, cfg.n_frames, cfg.latent_dim)
        assert feats["metadata_code"].shape == (
            2, cfg.n_frames, cfg.metadata_enc_dim,
        )
        assert feats["time_code"].shape == (
            2, cfg.n_frames, cfg.time_enc_dim,
        )
        assert feats["tokens"].shape == (
            2, cfg.n_frames,
            cfg.latent_dim + cfg.metadata_enc_dim + cfg.time_enc_dim,
        )
        assert feats["l"].shape == (2, cfg.n_queries, cfg.latent_dim)
        assert feats["h_x"].shape == (2, cfg.n_queries * cfg.latent_dim)
        assert feats["h_s"].shape == (2, cfg.latent_dim)
        assert feats["fusion"].shape == (
            2, cfg.n_queries * cfg.latent_dim + cfg.latent_dim,
        )
        assert feats["h_fused"].shape == (2, cfg.n_queries * cfg.latent_dim)
        assert feats["latent"].shape == (2, cfg.n_queries, cfg.latent_dim)
        assert feats["temporal"].shape == (2, cfg.n_frames, cfg.latent_dim)
        assert feats["param_pred"].shape == (2, cfg.n_frames, cfg.param_dim)

    def test_time_is_optional(self) -> None:
        """Without a `time` input the model still runs (zero time offset)."""
        cfg = _config()
        model = TelluricEstimator(cfg)
        observed = torch.randn(3, cfg.n_frames, cfg.n_wavelength)
        out = model(observed, stellar=None, metadata=None)
        assert out.params.shape == (3, cfg.n_frames, cfg.param_dim)
        assert out.intermediate_features["time_code"].shape == (
            3, cfg.n_frames, cfg.time_enc_dim,
        )

    def test_stellar_and_metadata_are_optional(self) -> None:
        cfg = _config()
        model = TelluricEstimator(cfg)
        observed = torch.randn(3, 12, 1024)
        out = model(observed, stellar=None, metadata=None)
        assert out.params.shape == (3, 12, cfg.param_dim)

    def test_broadcasts_single_stellar_and_metadata(self) -> None:
        cfg = _config()
        model = TelluricEstimator(cfg)
        observed = torch.randn(4, 12, 1024)
        out = model(
            observed,
            stellar=torch.randn(1024),   # (N,)
            metadata=torch.randn(3),     # (P,) shared over exposures
        )
        assert out.params.shape == (4, 12, cfg.param_dim)

    def test_dimension_mismatch_raises(self) -> None:
        model = TelluricEstimator(_config(t=12))
        with pytest.raises(ValueError):
            model(torch.randn(2, 10, 1024))  # wrong number of exposures

    def test_arbitrary_batch_sizes(self) -> None:
        cfg = _config()
        model = TelluricEstimator(cfg)
        for batch in (1, 2, 5):
            out = model(torch.randn(batch, 12, 1024))
            assert out.params.shape == (batch, 12, cfg.param_dim)

    @pytest.mark.skipif(
        os.environ.get("TELLURIC_SLOW_TESTS") != "1",
        reason="single full-resolution (N=51556, T=73) forward is slow; "
               "set TELLURIC_SLOW_TESTS=1",
    )
    def test_full_problem_shape(self) -> None:
        """Sanity shape at the real problem dims (small batch, single pass)."""
        cfg = TelluricEstimatorConfig(
            n_wavelength=51556, n_frames=73, n_queries=16, latent_dim=64,
            metadata_dim=3, n_heads=4, dropout=0.0,
        )
        model = TelluricEstimator(cfg)
        batch = 1
        out = model(
            torch.randn(batch, 73, 51556),
            stellar=torch.randn(batch, 51556),
            metadata=torch.randn(batch, 73, 3),
        )
        assert out.params.shape == (batch, 73, cfg.param_dim)
        assert out.latent.shape == (batch, 1024)
        assert out.attention_weights.shape == (batch, 16, 73)


class TestGradientFlow:
    def test_gradient_flow_through_all_parameters(self) -> None:
        cfg = _config()
        model = TelluricEstimator(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        observed = torch.randn(2, 12, 1024)
        stellar = torch.randn(2, 1024)
        metadata = torch.randn(2, 12, cfg.metadata_dim)
        target = torch.randn(2, 12, cfg.param_dim)  # random param target

        opt.zero_grad()
        out = model(observed, stellar=stellar, metadata=metadata)
        loss = F.mse_loss(out.params, target)
        loss.backward()
        opt.step()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert len(grads) > 0
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)
        assert torch.isfinite(loss)

    def test_parameter_counts_and_components(self) -> None:
        model = TelluricEstimator(_config())
        counts = model.parameter_counts()
        assert set(counts) == {
            "SpectralEncoder (X)", "MetadataEncoder", "TimeEncoder",
            "PreTemporal", "TemporalPerceiver", "StellarEncoder (S)",
            "FusionMLP", "TemporalDecoder", "ParamDecoder",
        }
        assert all(v > 0 for v in counts.values())
        assert sum(counts.values()) == model.count_parameters()


def _synthetic_night(n_wave: int, n_frames: int, rng: torch.Generator):
    """One smooth synthetic night: X = T * S with airmass-scaled tellurics.

    Returns (observed, stellar, telluric, metadata) where ``metadata`` is the
    per-exposure airmass code ``(T, 3)`` -- the physical driver of the drift.
    """
    device = "cpu"
    w = torch.linspace(0.0, 1.0, n_wave, device=device).unsqueeze(0)  # (1, N)
    noise = torch.sin(2 * torch.pi * torch.arange(n_wave, dtype=torch.float32, device=device).unsqueeze(0) * 3.0 / n_wave)
    stellar = 1.0 + 0.4 * noise + 0.2 * torch.cos(2 * torch.pi * w * 6.0)
    stellar = stellar / stellar.mean()

    tau = 0.05 + 0.25 * (0.5 + 0.5 * torch.sin(2 * torch.pi * w * 8.0))
    tau = tau * (0.7 + 0.3 * torch.sin(2 * torch.pi * w * 37.0))
    airm = torch.linspace(1.05, 2.0, n_frames, device=device).unsqueeze(1)  # (T,1)
    tell = torch.exp(-tau * airm)                                        # (T,N)
    observed = stellar * tell                                             # (T,N)
    airm_flat = airm.squeeze(1)                                           # (T,)
    metadata = torch.stack(
        [airm_flat, airm_flat ** 2, torch.ones_like(airm_flat)], dim=-1   # (T,3)
    )
    return observed, stellar, tell, metadata


class TestOverfitTiny:
    def test_fits_a_tiny_param_regression(self) -> None:
        """The estimator must reduce MSE(param_input, param_pred) on a tiny set."""
        cfg = _config(n_wave=512, t=12, q=8, d=32, heads=4)
        model = TelluricEstimator(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)

        nights = 4
        xs, ss, metas = [], [], []
        g = torch.Generator().manual_seed(0)
        for _ in range(nights):
            x, s, _, meta = _synthetic_night(
                cfg.n_wavelength, cfg.n_frames, g
            )
            xs.append(x); ss.append(s.squeeze(0)); metas.append(meta)
        x = torch.stack(xs)               # (B, T, N)
        s = torch.stack(ss)               # (B, N)
        metadata = torch.stack(metas)     # (B, T, P) per-exposure metadata
        param_input = torch.randn(nights, cfg.n_frames, cfg.param_dim)

        losses = []
        for _ in range(120):
            opt.zero_grad()
            out = model(x, stellar=s, metadata=metadata)
            loss = F.mse_loss(out.params, param_input)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def _repo_model_config(n_wave: int = 1024, t: int = 12) -> "ModelConfig":
    """Small repo-wide ModelConfig targeting the telluric_estimator arch."""
    return ModelConfig(
        architecture=ModelArchitecture.TELLURIC_ESTIMATOR,
        num_wavelength_bins=n_wave,
        n_frames_per_series=t,
        num_queries=8,
        spectral_latent_dim=32,
        metadata_dim=3,
        num_heads=4,
        dropout=0.0,
        x_encoder_channels=[8, 16, 32],
        s_encoder_channels=[8, 16, 32],
    )


class TestRepoIntegration:
    """Wiring of TelluricEstimator into ModelConfig / ModelRegistry."""

    def test_registry(self) -> None:
        assert "telluric_estimator" in ModelRegistry.list_models()
        assert ModelRegistry.get("telluric_estimator") is TelluricEstimator

    def test_construct_from_repo_model_config(self) -> None:
        model = TelluricEstimator(_repo_model_config())  # ModelConfig path
        out = model(
            torch.randn(2, 12, 1024),
            stellar=torch.randn(2, 1024),
            metadata=torch.randn(2, 12, 3),
        )
        assert out.params.shape == (2, 12, 20)

    def test_from_model_config_maps_fields(self) -> None:
        cfg = TelluricEstimatorConfig.from_model_config(_repo_model_config())
        assert cfg.n_wavelength == 1024
        assert cfg.n_frames == 12 and cfg.n_queries == 8 and cfg.latent_dim == 32
        assert cfg.x_encoder_channels == (8, 16, 32)
        assert cfg.metadata_enc_dim == 16   # ModelConfig default
        assert cfg.time_enc_dim == 16       # ModelConfig default
        assert cfg.param_dim == 20          # ModelConfig default
        assert cfg.fusion_hidden is None    # ModelConfig.fusion_hidden == 0 sentinel


class TestNightLevelSplit:
    def test_split_is_on_nights_only(self) -> None:
        from tellurics.data.splits import split_night_indices

        train, val, test = split_night_indices(100, 0.8, 0.1, seed=7)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        # Disjoint sets of night ids.
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)
