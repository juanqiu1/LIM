"""Tests for ``lim.eof``: masked + lat-weighted SVD on a synthetic field."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lim.eof import eof, project, reconstruct


def _grid(n_t: int = 200, n_lat: int = 12, n_lon: int = 18) -> xr.DataArray:
    time = pd.date_range("2000-01-03", periods=n_t, freq="7D")
    lat = np.linspace(-60, 60, n_lat)
    lon = np.linspace(0, 350, n_lon)
    return xr.DataArray(
        np.zeros((n_t, n_lat, n_lon)),
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
    )


def test_eof_recovers_single_pattern() -> None:
    """Field = pc(t) * pattern(lat, lon) + small noise → leading EOF == pattern (up to sign)."""
    da = _grid()
    rng = np.random.default_rng(0)
    pc_true = np.cos(2 * np.pi * np.arange(da.sizes["time"]) / 30.0)
    pattern = np.outer(np.exp(-(da.lat.values / 30.0) ** 2), np.cos(np.deg2rad(da.lon.values)))
    field = pc_true[:, None, None] * pattern[None, :, :] + 0.01 * rng.standard_normal(da.shape)
    da = da.copy(data=field)

    result = eof(da, n_modes=3, lat_weight=False)
    leading = result.eofs.isel(mode=0).values
    flat_pat = pattern / np.linalg.norm(pattern)
    flat_leading = leading / np.linalg.norm(leading)
    cos = abs(float(np.sum(flat_pat * flat_leading)))
    assert cos > 0.99, f"leading EOF cosine similarity {cos:.4f} < 0.99"


def test_eof_passes_through_masked_pixels() -> None:
    da = _grid()
    rng = np.random.default_rng(1)
    da = da + rng.standard_normal(da.shape)
    da[:, 0, 0] = np.nan

    result = eof(da, n_modes=2, lat_weight=False)
    assert bool(result.mask.isel(lat=0, lon=0).values) is False
    assert bool(np.isnan(result.eofs.isel(mode=0, lat=0, lon=0).values))


def test_eof_pcs_have_zero_time_mean() -> None:
    da = _grid()
    rng = np.random.default_rng(2)
    da = da + rng.standard_normal(da.shape)
    result = eof(da, n_modes=4, lat_weight=False)
    pc_means = result.pcs.mean("time").values
    assert np.all(np.abs(pc_means) < 1e-10)


def test_eof_reconstruction_matches_input_at_full_rank() -> None:
    da = _grid(n_t=20, n_lat=4, n_lon=5)
    rng = np.random.default_rng(3)
    da = da + rng.standard_normal(da.shape)
    result = eof(da, n_modes=20, lat_weight=False)
    rebuilt = reconstruct(result, result.pcs)
    assert np.allclose(rebuilt.transpose(*da.dims).values, da.values, atol=1e-9)


def test_project_round_trips() -> None:
    da = _grid()
    rng = np.random.default_rng(4)
    da = da + rng.standard_normal(da.shape)
    result = eof(da, n_modes=5, lat_weight=False)
    re_pcs = project(result, da)
    assert np.allclose(re_pcs.values, result.pcs.values, atol=1e-8)


def test_lat_weighting_downweights_polar_pixels() -> None:
    """A polar grid point should get a smaller weight than an equatorial one."""
    n_lat = 13  # odd → includes lat=0 exactly given linspace(-60, 60, n_lat)
    da = _grid(n_lat=n_lat, n_lon=2)
    rng = np.random.default_rng(5)
    da = da + 0.01 * rng.standard_normal(da.shape)
    result = eof(da, n_modes=1, lat_weight=True)
    w_polar = float(result.weights.sel(lat=da.lat.values[-1], lon=0))
    w_equator = float(result.weights.sel(lat=0.0, lon=0))
    assert w_polar < w_equator
    assert w_equator == pytest.approx(1.0, abs=1e-9)


def test_eof_rejects_invalid_inputs() -> None:
    da = _grid()
    with pytest.raises(ValueError, match="time dimension"):
        eof(da, n_modes=1, time_dim="foo")
    with pytest.raises(ValueError, match="n_modes"):
        eof(da, n_modes=0)
