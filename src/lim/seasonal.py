"""Deseasonalization of climate time series.

Three methods:

* ``harmonic`` (default) — least-squares fit of the first ``n_harmonics``
  annual harmonics to each grid point. Smooth, leap-year-safe, standard in
  sea-ice LIM literature.
* ``doy`` — subtract the per-day-of-year mean.
* ``woy`` — subtract the per-week-of-year mean.

The returned ``climatology`` is aligned to the input time axis (same dims as
the input), so ``anomalies + climatology == input``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

Method = Literal["harmonic", "doy", "woy"]


def deseasonalize(
    da: xr.DataArray,
    *,
    method: Method = "harmonic",
    n_harmonics: int = 3,
    time_dim: str = "time",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Remove the seasonal cycle from ``da``.

    Parameters
    ----------
    da : xr.DataArray
        Input field with a time dimension.
    method : {"harmonic", "doy", "woy"}
    n_harmonics : int
        Number of annual harmonics for ``method="harmonic"`` (ignored otherwise).
    time_dim : str
        Name of the time dimension on ``da``.

    Returns
    -------
    anomalies, climatology
        Both have the same dims/shape as ``da``. ``anomalies + climatology``
        equals ``da`` to numerical precision.
    """
    if time_dim not in da.dims:
        raise ValueError(f"time dimension {time_dim!r} not in da.dims={da.dims}")

    if method == "harmonic":
        if n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1; got {n_harmonics}")
        clim = _harmonic_climatology(da, n_harmonics, time_dim)
    elif method == "doy":
        clim = _groupby_climatology(da, time_dim, "dayofyear")
    elif method == "woy":
        clim = _groupby_climatology(da, time_dim, "weekofyear")
    else:
        raise ValueError(
            f"method must be one of 'harmonic', 'doy', 'woy'; got {method!r}"
        )

    anom = da - clim
    return anom, clim


def _harmonic_basis(time: xr.DataArray, n_harmonics: int) -> np.ndarray:
    """Build a (n_times, 1 + 2*n_harmonics) design matrix at the given times."""
    t_ns = time.values.astype("datetime64[ns]").astype("int64")
    days = (t_ns - t_ns[0]) / (1e9 * 86400.0)
    omega = 2.0 * np.pi / 365.25
    cols = [np.ones_like(days)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * omega * days))
        cols.append(np.cos(k * omega * days))
    return np.stack(cols, axis=1)


def _harmonic_climatology(
    da: xr.DataArray, n_harmonics: int, time_dim: str
) -> xr.DataArray:
    """Per-pixel OLS fit of mean + n_harmonics annual harmonics; NaN columns pass through."""
    other_dims = [d for d in da.dims if d != time_dim]
    if other_dims:
        stacked = da.stack(_point=other_dims).transpose(time_dim, "_point")
        Y = stacked.values
    else:
        Y = da.values[:, None]

    B = _harmonic_basis(da[time_dim], n_harmonics)
    clim_vals = np.full_like(Y, np.nan, dtype=float)
    valid = ~np.any(np.isnan(Y), axis=0)
    if valid.any():
        beta, *_ = np.linalg.lstsq(B, Y[:, valid], rcond=None)
        clim_vals[:, valid] = B @ beta

    if other_dims:
        out = stacked.copy(data=clim_vals).unstack("_point")
        return out.transpose(*da.dims)
    return da.copy(data=clim_vals[:, 0])


def _groupby_climatology(
    da: xr.DataArray, time_dim: str, key: str
) -> xr.DataArray:
    """Subtract per-``key`` (dayofyear or weekofyear) mean."""
    if key == "weekofyear":
        idx = da[time_dim].to_index()
        woy = xr.DataArray(
            idx.isocalendar().week.to_numpy(),
            coords={time_dim: da[time_dim]},
            dims=time_dim,
            name="weekofyear",
        )
        da_keyed = da.assign_coords(weekofyear=woy)
        clim_per = da_keyed.groupby("weekofyear").mean()
        clim = clim_per.sel(weekofyear=woy)
        return clim.drop_vars("weekofyear").transpose(*da.dims)
    else:
        attr_key = f"{time_dim}.{key}"
        clim_per = da.groupby(attr_key).mean()
        sel_key = da[time_dim].dt.dayofyear if key == "dayofyear" else None
        clim = clim_per.sel({key: sel_key})
        return clim.drop_vars(key).transpose(*da.dims)
