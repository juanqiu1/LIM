"""Tests for ``lim.seasonal.deseasonalize`` (harmonic / doy / woy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lim.seasonal import deseasonalize


def _weekly_time(n_years: int = 8, start: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_years * 52, freq="7D")


def _sin_year(t: pd.DatetimeIndex, amplitude: float = 1.0) -> np.ndarray:
    days = (t - t[0]).total_seconds().to_numpy() / 86400.0
    return amplitude * np.cos(2 * np.pi * days / 365.25)


def test_harmonic_recovers_pure_annual_signal() -> None:
    """3-harmonic fit on a pure-annual signal removes ~all of it."""
    t = _weekly_time()
    rng = np.random.default_rng(0)
    signal = _sin_year(t, amplitude=2.0) + 0.1 * rng.standard_normal(len(t))
    da = xr.DataArray(signal, coords={"time": t}, dims="time")
    anom, clim = deseasonalize(da, method="harmonic", n_harmonics=3)
    assert anom.shape == da.shape
    assert clim.shape == da.shape
    # Reconstruction is exact (up to float).
    assert np.allclose(anom + clim, da, atol=1e-10)
    # Anomaly std should be dominated by the noise (~0.1), not the signal (~1.4 std).
    assert float(anom.std()) < 0.3


def test_harmonic_handles_2d_field_with_nan_column() -> None:
    """A grid-pixel column that's all-NaN should pass through as NaN."""
    t = _weekly_time(n_years=4)
    rng = np.random.default_rng(1)
    field = np.zeros((len(t), 5))
    for j in range(5):
        field[:, j] = _sin_year(t, amplitude=j + 1.0) + 0.05 * rng.standard_normal(len(t))
    field[:, 2] = np.nan
    da = xr.DataArray(field, coords={"time": t, "x": np.arange(5)}, dims=("time", "x"))
    anom, clim = deseasonalize(da, method="harmonic", n_harmonics=2)
    assert np.all(np.isnan(anom.isel(x=2).values))
    assert np.all(np.isnan(clim.isel(x=2).values))
    assert float(anom.isel(x=0).std()) < 0.2
    assert float(anom.isel(x=4).std()) < 0.2


def test_doy_climatology_removes_annual_mean() -> None:
    t = _weekly_time(n_years=10)
    rng = np.random.default_rng(2)
    signal = _sin_year(t, amplitude=1.5) + 0.2 * rng.standard_normal(len(t))
    da = xr.DataArray(signal, coords={"time": t}, dims="time")
    anom, clim = deseasonalize(da, method="doy")
    assert np.allclose(anom + clim, da, atol=1e-10)
    assert abs(float(anom.mean())) < 0.1


def test_woy_climatology_removes_annual_mean() -> None:
    t = _weekly_time(n_years=10)
    rng = np.random.default_rng(3)
    signal = _sin_year(t, amplitude=1.5) + 0.2 * rng.standard_normal(len(t))
    da = xr.DataArray(signal, coords={"time": t}, dims="time")
    anom, clim = deseasonalize(da, method="woy")
    assert np.allclose(anom + clim, da, atol=1e-10)
    assert abs(float(anom.mean())) < 0.1


def test_deseasonalize_rejects_invalid_inputs() -> None:
    t = _weekly_time(n_years=2)
    da = xr.DataArray(np.zeros(len(t)), coords={"time": t}, dims="time")
    with pytest.raises(ValueError, match="time dimension"):
        deseasonalize(da, time_dim="foo")
    with pytest.raises(ValueError, match="method"):
        deseasonalize(da, method="garbage")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_harmonics"):
        deseasonalize(da, method="harmonic", n_harmonics=0)
