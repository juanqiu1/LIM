"""Cyclostationary LIM (CS-LIM): one operator per phase of a periodic cycle.

For seasonally varying systems like Arctic sea ice, the linear dynamics
``L`` depend on the time of year. CS-LIM (Ortiz-Bevia 1997; Shin et al. 2010;
Wang et al. 2019) handles this by fitting a separate ``L_p`` at each phase
``p`` (e.g., week-of-year for weekly data, ``period=52``).

The propagator from phase ``p0`` over ``n_steps`` lags of length ``tau0`` is
the **left-multiplied product** of the per-phase propagators::

    G(p0, n_steps) = expm(L_{p_{n-1}} * tau0) ... expm(L_{p_1} * tau0) expm(L_{p_0} * tau0)

where ``p_i = (p0 + i * tau0) % period``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .operator import LimFit, fit_operator_pair


@dataclass(frozen=True)
class CycloLimFit:
    """Fitted CS-LIM: one ``LimFit`` per phase of a periodic cycle.

    Attributes
    ----------
    fits : list of LimFit
        Length = ``period``. ``fits[p]`` propagates a state at phase ``p``
        forward by ``tau0`` samples, ending at phase ``(p + tau0) % period``.
    period : int
    tau0 : int
        Training lag in samples (shared across phases).
    """

    fits: Sequence[LimFit]
    period: int
    tau0: int


def fit_cyclo(
    X: np.ndarray,
    phase: np.ndarray,
    *,
    tau0: int,
    period: int,
) -> CycloLimFit:
    """Fit one stationary LIM per phase.

    Parameters
    ----------
    X : (n_modes, n_times) array
        State time series.
    phase : (n_times,) integer array
        Phase of each sample, values in ``[0, period)``.
    tau0 : int
        Training lag in samples.
    period : int
        Number of phases per cycle (e.g., 52 for weekly).
    """
    X = np.asarray(X)
    phase = np.asarray(phase)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_modes, n_times); got shape {X.shape}")
    if phase.shape != (X.shape[1],):
        raise ValueError(
            f"phase must have shape ({X.shape[1]},); got {phase.shape}"
        )
    if not isinstance(tau0, (int, np.integer)) or tau0 < 1:
        raise ValueError(f"tau0 must be a positive integer; got {tau0!r}")
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise ValueError(f"period must be an integer >= 2; got {period!r}")
    if phase.min() < 0 or phase.max() >= period:
        raise ValueError(
            f"phase values must lie in [0, {period}); got [{phase.min()}, {phase.max()}]"
        )

    n_times = X.shape[1]
    fits: list[LimFit] = []
    for p in range(period):
        k_now = np.where(phase == p)[0]
        k_valid = k_now[k_now + tau0 < n_times]
        if k_valid.size < 2:
            raise ValueError(
                f"phase {p} has only {k_valid.size} usable samples (need >= 2); "
                "either lengthen the input or reduce tau0/period"
            )
        X0_p = X[:, k_valid]
        Xtau_p = X[:, k_valid + tau0]
        fits.append(fit_operator_pair(X0_p, Xtau_p, tau0))
    return CycloLimFit(fits=fits, period=int(period), tau0=int(tau0))


def cyclo_propagator(fit: CycloLimFit, phase0: int, n_steps: int) -> np.ndarray:
    """Product of single-phase propagators starting at ``phase0`` for ``n_steps`` lags.

    Each step advances by ``fit.tau0`` samples. Returns the matrix that
    propagates a state at ``phase0`` to phase ``(phase0 + n_steps * tau0) % period``.

    ``n_steps == 0`` returns the identity.
    """
    if not 0 <= phase0 < fit.period:
        raise ValueError(f"phase0 must lie in [0, {fit.period}); got {phase0}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0; got {n_steps}")

    n_modes = fit.fits[0].L.shape[0]
    G = np.eye(n_modes)
    for k in range(n_steps):
        p = (phase0 + k * fit.tau0) % fit.period
        G = scipy.linalg.expm(fit.fits[p].L * fit.tau0) @ G
    return G
