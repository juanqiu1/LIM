"""Deterministic LIM forecasts: ``E[x(t+tau)] = expm(L*tau) x(t)``.

Stochastic ensemble forecasts are added in a later step.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.linalg

from .operator import LimFit


def forecast_deterministic(
    fit: LimFit,
    x0: np.ndarray,
    leads: float | np.ndarray | Sequence[float],
) -> np.ndarray:
    """Noise-free forecast for one or many lead times.

    Parameters
    ----------
    fit : LimFit
    x0 : (n_modes,) or (n_modes, n_init) array
        Initial condition(s). A 1D input is treated as a single initial state.
    leads : float or array-like of float
        Lead times, in the same units as ``fit.tau0``. Scalars are accepted.

    Returns
    -------
    (n_modes, n_init, n_leads) array
        Forecast trajectory. The ``n_init`` axis is preserved even when ``x0``
        was 1D, so the output is always 3D for downstream slicing consistency.
    """
    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim == 1:
        x0_arr = x0_arr[:, None]
    elif x0_arr.ndim != 2:
        raise ValueError(f"x0 must be 1D or 2D; got shape {x0_arr.shape}")
    if x0_arr.shape[0] != fit.L.shape[0]:
        raise ValueError(
            f"x0 has {x0_arr.shape[0]} modes but L has {fit.L.shape[0]}"
        )

    leads_arr = np.atleast_1d(np.asarray(leads, dtype=float))
    if leads_arr.ndim != 1:
        raise ValueError(f"leads must be a scalar or 1D; got shape {leads_arr.shape}")

    n_modes, n_init = x0_arr.shape
    n_leads = leads_arr.size
    out = np.empty((n_modes, n_init, n_leads))
    for k, tau in enumerate(leads_arr):
        out[:, :, k] = scipy.linalg.expm(fit.L * tau) @ x0_arr
    return out
