"""EOFs of a gridded field with land-mask and latitude-weighting support.

A thin wrapper around ``np.linalg.svd``: stack non-time dims, apply
``sqrt(cos(lat))`` weights, drop columns that are entirely NaN (e.g. land in a
sea-ice mask), SVD, and reshape EOFs back to the grid with NaN at masked points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class EofResult:
    """Output of :func:`eof`.

    ``eofs[k]`` is the spatial pattern of mode ``k`` and ``pcs[k]`` its time
    series; ``eigvals[k]`` is the variance explained (squared singular value
    divided by ``n_times - 1``).
    """

    eofs: xr.DataArray
    pcs: xr.DataArray
    eigvals: np.ndarray
    mean: xr.DataArray
    mask: xr.DataArray
    weights: xr.DataArray


def eof(
    da: xr.DataArray,
    *,
    n_modes: int,
    mask: xr.DataArray | None = None,
    lat_weight: bool = True,
    time_dim: str = "time",
    lat_dim: str = "lat",
) -> EofResult:
    """Compute EOFs of ``da`` (a single ``time_dim`` plus any number of spatial dims).

    Parameters
    ----------
    da : xr.DataArray
        Input field. Must have ``time_dim`` and at least one spatial dim.
    n_modes : int
        Number of EOF modes to retain.
    mask : xr.DataArray, optional
        Boolean mask over the spatial dims; ``True`` = include in the SVD.
        If ``None``, the mask is taken to be ``~da.mean(time_dim).isnull()``.
    lat_weight : bool, default True
        If ``True`` and ``lat_dim`` is a coordinate of ``da``, multiply each
        grid point by ``sqrt(cos(lat))`` before SVD (and divide back out so
        ``eofs`` are returned in physical units).
    time_dim, lat_dim : str
    """
    if time_dim not in da.dims:
        raise ValueError(f"time dimension {time_dim!r} not in da.dims={da.dims}")
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}")

    spatial_dims = [d for d in da.dims if d != time_dim]
    if not spatial_dims:
        raise ValueError("da must have at least one non-time dimension")

    mean = da.mean(time_dim)
    centered = da - mean

    if lat_weight and lat_dim in da.coords:
        lat = da[lat_dim]
        w_lat = np.sqrt(np.cos(np.deg2rad(lat)).clip(min=0.0))
        weights = w_lat.broadcast_like(mean)
    else:
        weights = xr.ones_like(mean)

    if mask is None:
        mask = ~mean.isnull()

    centered_2d = centered.stack(_point=spatial_dims).transpose(time_dim, "_point")
    weights_1d = weights.stack(_point=spatial_dims)
    mask_1d = mask.stack(_point=spatial_dims)

    valid = mask_1d.values.astype(bool)
    Y = centered_2d.values
    if not valid.any():
        raise ValueError("mask excludes every grid point")
    Y_valid = Y[:, valid] * weights_1d.values[valid]
    if not np.all(np.isfinite(Y_valid)):
        raise ValueError("data has NaN/Inf inside the mask; tighten the mask first")

    U, s, Vt = np.linalg.svd(Y_valid, full_matrices=False)
    n_modes_eff = int(min(n_modes, len(s)))
    n_times = Y.shape[0]

    eigvals = (s[:n_modes_eff] ** 2) / (n_times - 1)

    eofs_valid = Vt[:n_modes_eff, :].T / weights_1d.values[valid][:, None]
    eofs_full = np.full((Y.shape[1], n_modes_eff), np.nan)
    eofs_full[valid, :] = eofs_valid

    pcs_arr = (U[:, :n_modes_eff] * s[:n_modes_eff]).T

    eofs_da = (
        xr.DataArray(
            eofs_full.T,
            coords={"mode": np.arange(n_modes_eff), "_point": centered_2d["_point"]},
            dims=("mode", "_point"),
        )
        .unstack("_point")
        .transpose("mode", *spatial_dims)
    )
    pcs_da = xr.DataArray(
        pcs_arr,
        coords={"mode": np.arange(n_modes_eff), time_dim: da[time_dim]},
        dims=("mode", time_dim),
    )

    return EofResult(
        eofs=eofs_da,
        pcs=pcs_da,
        eigvals=eigvals,
        mean=mean,
        mask=mask,
        weights=weights,
    )


def project(result: EofResult, da: xr.DataArray) -> xr.DataArray:
    """Project a new field onto the stored EOF basis, returning ``(mode, time)`` PCs."""
    spatial_dims = [d for d in da.dims if d in result.eofs.dims and d != "mode"]
    centered = da - result.mean
    weighted = centered * result.weights
    mask_1d = result.mask.stack(_point=spatial_dims)
    valid = mask_1d.values.astype(bool)

    weighted_2d = weighted.stack(_point=spatial_dims).transpose("time", "_point")
    eofs_2d = (result.eofs * result.weights).stack(_point=spatial_dims).transpose(
        "mode", "_point"
    )
    Y = weighted_2d.values[:, valid]
    E = eofs_2d.values[:, valid]
    norms = (E * E).sum(axis=1, keepdims=True)
    pcs = (Y @ E.T) / norms.T
    return xr.DataArray(
        pcs.T,
        coords={"mode": result.eofs["mode"], "time": da["time"]},
        dims=("mode", "time"),
    )


def reconstruct(result: EofResult, pcs: xr.DataArray) -> xr.DataArray:
    """Inverse of :func:`project`: rebuild a gridded field from PCs + EOFs + mean."""
    expanded = (result.eofs * pcs).sum("mode")
    return expanded + result.mean
