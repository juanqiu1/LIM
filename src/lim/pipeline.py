"""High-level pipeline: gridded NetCDF → deseasonalize → EOFs → LIM.

End-to-end glue around :mod:`lim.seasonal`, :mod:`lim.eof`, :mod:`lim.operator`,
and :mod:`lim.cyclo`. Most of the design decisions live in those modules; this
file just wires them together and bundles the artifacts you need to run
forecasts back on the original grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from .cyclo import CycloLimFit, fit_cyclo
from .eof import EofResult, eof
from .operator import LimFit, fit_operator
from .seasonal import deseasonalize

Mode = Literal["stationary", "cyclo"]


@dataclass(frozen=True)
class GridLimModel:
    """Bundle of fitted artifacts produced by :func:`fit_lim_from_grid`.

    Attributes
    ----------
    eof : EofResult
        EOF basis for the deseasonalized field; the LIM operates in this PC
        space.
    lim : LimFit | CycloLimFit
        Stationary or cyclostationary LIM fit on the PCs.
    climatology : xr.DataArray
        Time-aligned seasonal cycle that was subtracted from ``da``. Add this
        back to forecast anomalies to get full-field forecasts.
    mode : {"stationary", "cyclo"}
    """

    eof: EofResult
    lim: LimFit | CycloLimFit
    climatology: xr.DataArray
    mode: Mode


def fit_lim_from_grid(
    da: xr.DataArray,
    *,
    tau0: int,
    n_eofs: int,
    mode: Mode = "stationary",
    period: int = 52,
    deseasonalize_method: str = "harmonic",
    n_harmonics: int = 3,
    mask: xr.DataArray | None = None,
    phase: np.ndarray | None = None,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lat_weight: bool = True,
) -> GridLimModel:
    """Deseasonalize ``da``, project onto ``n_eofs`` EOFs, fit a LIM on the PCs.

    Parameters
    ----------
    da : xr.DataArray
        Gridded field with a regular time axis. Land or always-NaN points are
        skipped automatically; pass an explicit ``mask`` to override.
    tau0 : int
        LIM training lag in samples.
    n_eofs : int
    mode : {"stationary", "cyclo"}
        ``"cyclo"`` fits one operator per phase of the periodic cycle.
    period : int, default 52
        Cycle length, only used when ``mode="cyclo"``.
    deseasonalize_method : {"harmonic", "doy", "woy"}, default "harmonic"
    n_harmonics : int
        Used by ``deseasonalize_method="harmonic"``.
    mask : xr.DataArray, optional
        Boolean spatial mask passed through to :func:`lim.eof.eof`.
    phase : (n_times,) integer array, optional
        Per-sample phase for ``mode="cyclo"``. Defaults to
        ``np.arange(n_times) % period`` (correct when ``da`` is regularly
        sampled at the cycle resolution).
    time_dim, lat_dim, lat_weight
        Forwarded to :func:`lim.eof.eof`.
    """
    if mode not in ("stationary", "cyclo"):
        raise ValueError(f"mode must be 'stationary' or 'cyclo'; got {mode!r}")

    anom, clim = deseasonalize(
        da,
        method=deseasonalize_method,  # type: ignore[arg-type]
        n_harmonics=n_harmonics,
        time_dim=time_dim,
    )
    eof_result = eof(
        anom,
        n_modes=n_eofs,
        mask=mask,
        lat_weight=lat_weight,
        time_dim=time_dim,
        lat_dim=lat_dim,
    )
    pcs = np.asarray(eof_result.pcs.values)

    lim_fit: LimFit | CycloLimFit
    if mode == "stationary":
        lim_fit = fit_operator(pcs, tau0)
    else:
        n_times = pcs.shape[1]
        if phase is None:
            phase_arr = np.arange(n_times) % period
        else:
            phase_arr = np.asarray(phase)
            if phase_arr.shape != (n_times,):
                raise ValueError(
                    f"phase must have shape ({n_times},); got {phase_arr.shape}"
                )
        lim_fit = fit_cyclo(pcs, phase_arr, tau0=tau0, period=period)

    return GridLimModel(eof=eof_result, lim=lim_fit, climatology=clim, mode=mode)
