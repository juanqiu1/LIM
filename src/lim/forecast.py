"""LIM forecasts.

Two modes:

* :func:`forecast_deterministic` — noise-free ``E[x(t+tau)] = expm(L*tau) x(t)``.
* :func:`forecast_ensemble` — stochastic initial-value problem: integrates
  ``dx = L x dt + xi`` from ``x0`` over user-chosen leads with ``n_members``
  ensemble members.
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


def forecast_ensemble(
    fit: LimFit,
    x0: np.ndarray,
    leads: float | np.ndarray | Sequence[float],
    n_members: int,
    *,
    dt: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Stochastic ensemble forecast: integrate ``dx = L x dt + xi`` from ``x0``.

    Parameters
    ----------
    fit : LimFit
    x0 : (n_modes,) or (n_modes, n_init) array
        Initial condition(s).
    leads : float or array-like of float
        Lead times to record, in the same units as ``fit.tau0``. Each must be a
        non-negative integer multiple of ``dt``.
    n_members : int
        Ensemble size.
    dt : float, default 1.0
        Integration substep, in the same units as ``leads``.
    rng : Generator, int, or None
        Random source.

    Returns
    -------
    (n_modes, n_init, n_leads, n_members) array
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
    if n_members < 1:
        raise ValueError(f"n_members must be >= 1; got {n_members}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0; got {dt}")

    leads_arr = np.atleast_1d(np.asarray(leads, dtype=float))
    if leads_arr.ndim != 1:
        raise ValueError(f"leads must be a scalar or 1D; got shape {leads_arr.shape}")
    if np.any(leads_arr < 0):
        raise ValueError(f"leads must be >= 0; got {leads_arr}")
    lead_steps = np.round(leads_arr / dt).astype(int)
    if not np.allclose(lead_steps * dt, leads_arr, atol=1e-9):
        raise ValueError(
            f"each lead must be an integer multiple of dt; got leads={leads_arr}, dt={dt}"
        )

    rng_obj = np.random.default_rng(rng)
    n_modes = fit.L.shape[0]
    n_init = x0_arr.shape[1]
    n_leads = leads_arr.size

    Q_sym = 0.5 * (fit.Q + fit.Q.T)
    eigvals, V = np.linalg.eigh(Q_sym)
    pos_mask = eigvals > 0
    if not pos_mask.any():
        raise ValueError("Q has no positive eigenvalues; cannot draw noise")
    Dp = eigvals[pos_mask]
    Vp = V[:, pos_mask]
    Dp = Dp * float(eigvals.sum()) / float(Dp.sum())
    noise_amp = Vp * np.sqrt(Dp * dt)
    coef = np.eye(n_modes) + fit.L * dt
    n_pos = Dp.size

    n_traj = n_init * n_members
    x = np.repeat(x0_arr, n_members, axis=1)

    out = np.empty((n_modes, n_init, n_leads, n_members))
    order = np.argsort(lead_steps)
    current_step = 0
    for idx in order:
        target = int(lead_steps[idx])
        while current_step < target:
            x = coef @ x + noise_amp @ rng_obj.standard_normal((n_pos, n_traj))
            current_step += 1
        out[:, :, idx, :] = x.reshape(n_modes, n_init, n_members)

    if not np.all(np.isfinite(out)):
        raise RuntimeError("ensemble forecast blew up")
    return out
