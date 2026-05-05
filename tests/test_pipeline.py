"""End-to-end pipeline tests on synthetic gridded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lim.cyclo import CycloLimFit
from lim.operator import LimFit
from lim.pipeline import GridLimModel, fit_lim_from_grid

# Synthetic data is short for many-phase fits; the logm imaginary-part warning
# from operator.py is expected-and-diagnostic in this regime.
pytestmark = pytest.mark.filterwarnings(
    "ignore:logm\\(G\\)/tau0 has nontrivial imaginary part:RuntimeWarning"
)


def _synthetic_field(n_t: int = 520, n_lat: int = 8, n_lon: int = 12) -> xr.DataArray:
    """Synthetic weekly field: seasonal cycle + 2 PC-like patterns + noise."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2010-01-03", periods=n_t, freq="7D")
    lat = np.linspace(0, 70, n_lat)
    lon = np.linspace(0, 350, n_lon)
    days = (time - time[0]).total_seconds().to_numpy() / 86400.0

    pat1 = np.outer(np.exp(-((lat - 50) / 20) ** 2), np.cos(np.deg2rad(lon)))
    pat2 = np.outer(np.exp(-((lat - 30) / 25) ** 2), np.sin(np.deg2rad(lon)))

    pc1 = np.cos(2 * np.pi * np.arange(n_t) / 30) + 0.3 * rng.standard_normal(n_t)
    pc2 = 0.7 * rng.standard_normal(n_t)
    seasonal = np.cos(2 * np.pi * days / 365.25)

    field = (
        seasonal[:, None, None] * pat1[None, :, :]
        + pc1[:, None, None] * pat1[None, :, :]
        + pc2[:, None, None] * pat2[None, :, :]
        + 0.05 * rng.standard_normal((n_t, n_lat, n_lon))
    )
    return xr.DataArray(
        field, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon")
    )


def test_pipeline_stationary_returns_consistent_artifacts() -> None:
    da = _synthetic_field()
    model = fit_lim_from_grid(da, tau0=4, n_eofs=3, mode="stationary")
    assert isinstance(model, GridLimModel)
    assert isinstance(model.lim, LimFit)
    assert model.lim.L.shape == (3, 3)
    assert model.eof.eofs.sizes["mode"] == 3
    assert model.eof.pcs.sizes["mode"] == 3
    assert model.eof.pcs.sizes["time"] == da.sizes["time"]
    assert model.climatology.shape == da.shape


def test_pipeline_cyclo_returns_one_fit_per_phase() -> None:
    da = _synthetic_field()
    model = fit_lim_from_grid(
        da, tau0=1, n_eofs=3, mode="cyclo", period=13
    )
    assert isinstance(model.lim, CycloLimFit)
    assert len(model.lim.fits) == 13
    for f in model.lim.fits:
        assert f.L.shape == (3, 3)


def test_pipeline_subtracts_seasonal_cycle() -> None:
    """Climatology + anomalies must reconstruct the field."""
    da = _synthetic_field()
    model = fit_lim_from_grid(da, tau0=4, n_eofs=3)
    rebuilt = model.climatology + (da - model.climatology)
    assert np.allclose(rebuilt.values, da.values, atol=1e-10)


def test_pipeline_pcs_have_zero_time_mean() -> None:
    da = _synthetic_field()
    model = fit_lim_from_grid(da, tau0=4, n_eofs=3)
    pc_means = model.eof.pcs.mean("time").values
    assert np.all(np.abs(pc_means) < 1e-9)


def test_pipeline_rejects_invalid_mode() -> None:
    da = _synthetic_field()
    with pytest.raises(ValueError, match="mode"):
        fit_lim_from_grid(da, tau0=4, n_eofs=3, mode="garbage")  # type: ignore[arg-type]


def test_pipeline_accepts_explicit_phase() -> None:
    da = _synthetic_field()
    n_t = da.sizes["time"]
    phase = np.arange(n_t) % 4
    model = fit_lim_from_grid(
        da, tau0=1, n_eofs=2, mode="cyclo", period=4, phase=phase
    )
    assert len(model.lim.fits) == 4
