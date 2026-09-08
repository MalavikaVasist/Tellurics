"""Tests for the night-level Perceiver telluric model."""

import pytest
import torch

from tellurics.configs.model import ModelArchitecture, ModelConfig
from tellurics.models.night import PerceiverNightModel
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


def _config(
    n_wave: int = 1024,
    t: int = 12,
    q: int = 8,
    d: int = 32,
    hidden: int = 64,
    metadata_dim: int = 6,
    hidden_dims: list[int] | None = None,
) -> ModelConfig:
    return ModelConfig(
        architecture=ModelArchitecture.NIGHT_PERCEIVER,
        num_wavelength_bins=n_wave,
        n_frames_per_series=t,
        num_queries=q,
        spectral_latent_dim=d,
        spectral_encoder_dims=hidden_dims or [256, 128],
        hidden_dim=hidden,
        metadata_dim=metadata_dim,
        dropout=0.0,
    )


class TestPerceiverNightModel:
    def test_shapes(self) -> None:
        cfg = _config()
        model = PerceiverNightModel(cfg)

        observed = torch.randn(2, cfg.n_frames_per_series, cfg.num_wavelength_bins)
        stellar = torch.randn(2, cfg.num_wavelength_bins)
        metadata = torch.randn(2, cfg.metadata_dim)

        out = model(observed, stellar=stellar, metadata=metadata)

        assert isinstance(out, ModelOutput)
        assert out.telluric.shape == (2, 12, 1024)
        assert out.latent.shape == (2, cfg.num_queries * cfg.spectral_latent_dim)
        assert out.attention_weights is not None
        assert out.attention_weights.shape == (2, cfg.num_queries, cfg.n_frames_per_series)
        assert out.telluric.min() >= 0.0
        assert out.telluric.max() <= 1.0  # sigmoid: transmission in [0, 1]

    def test_stellar_and_metadata_are_optional(self) -> None:
        model = PerceiverNightModel(_config())
        observed = torch.randn(2, 12, 1024)
        out = model(observed, stellar=None, metadata=None)
        assert out.telluric.shape == (2, 12, 1024)

    def test_broadcasts_single_stellar_and_metadata(self) -> None:
        model = PerceiverNightModel(_config())
        observed = torch.randn(4, 12, 1024)
        out = model(
            observed,
            stellar=torch.randn(1024),       # (N,)
            metadata=torch.randn(6),          # (P,)
        )
        assert out.telluric.shape == (4, 12, 1024)

    def test_dimension_mismatch_raises(self) -> None:
        model = PerceiverNightModel(_config(t=12))
        with pytest.raises(ValueError):
            model(torch.randn(2, 10, 1024))  # wrong number of exposures

    def test_registry(self) -> None:
        assert "perceiver_night" in ModelRegistry.list_models()
        assert ModelRegistry.get("perceiver_night") == PerceiverNightModel

    def test_parameter_count_is_bounded(self) -> None:
        # Hidden width is the dominant term; keep it tractable in a unit test.
        model = PerceiverNightModel(_config(n_wave=1024, hidden_dims=[256, 128]))
        n_params = model.count_parameters()
        assert n_params > 0
        # ~ (1024*256 + 256*128 + 128*32) encoder + small perceiver/head.
        assert n_params < 500_000
        # Shared encoder must only be instantiated once.
        assert sum(1 for _ in model.encoder.parameters()) > 0

    def test_gradient_flow(self) -> None:
        cfg = _config()
        model = PerceiverNightModel(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        observed = torch.randn(2, 12, 1024)
        stellar = torch.randn(2, 1024)
        metadata = torch.randn(2, cfg.metadata_dim)
        target = torch.rand(2, 12, 1024)  # transmission in [0, 1]

        opt.zero_grad()
        out = model(observed, stellar=stellar, metadata=metadata)
        loss = torch.nn.functional.mse_loss(out.telluric, target)
        loss.backward()
        opt.step()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert torch.isfinite(loss)


class TestNightLevelSplit:
    def test_split_is_on_nights_only(self) -> None:
        from tellurics.data.night import split_night_indices

        train, val, test = split_night_indices(100, 0.8, 0.1, seed=7)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        # Disjoint sets of night ids.
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)
