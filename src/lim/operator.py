"""Stationary LIM: fit linear operator L and noise covariance Q from snapshots.

Ports ``matlab/tx_lim_operator.m`` to Python and adds a single-array entry point
plus a propagator helper. All time quantities (``tau0``, ``tau``) are in units of
the input sampling interval; ``L`` therefore has units of ``1 / sample``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg

_IMAG_REL_TOL = 1e-8


@dataclass(frozen=True)
class LimFit:
    """Fitted stationary LIM.

    Attributes
    ----------
    L : (n_modes, n_modes) array
        Linear operator. ``dx/dt = L x + xi``.
    Q : (n_modes, n_modes) array
        Noise covariance, ``Q = -(L C0 + C0 L.T)``.
    C0, Ctau, G : (n_modes, n_modes) arrays
        Lag-0 covariance, lag-``tau0`` cross-covariance, and Green function
        ``G = Ctau C0^{-1} = expm(L tau0)``.
    tau0 : int
        Training lag, in samples.
    """

    L: np.ndarray
    Q: np.ndarray
    C0: np.ndarray
    Ctau: np.ndarray
    G: np.ndarray
    tau0: int


def fit_operator(X: np.ndarray, tau0: int) -> LimFit:
    """Fit a stationary LIM from a single state array.

    Parameters
    ----------
    X : (n_modes, n_times) array
        State time series; columns are time samples (matches the MATLAB
        convention from ``matlab/tx_lim_operator.m``).
    tau0 : int
        Training lag in samples; must satisfy ``1 <= tau0 < n_times``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_modes, n_times); got shape {X.shape}")
    if not isinstance(tau0, (int, np.integer)) or tau0 < 1:
        raise ValueError(f"tau0 must be a positive integer; got {tau0!r}")
    if tau0 >= X.shape[1]:
        raise ValueError(f"tau0={tau0} must be < n_times={X.shape[1]}")
    X0 = X[:, : X.shape[1] - tau0]
    Xtau = X[:, tau0:]
    return fit_operator_pair(X0, Xtau, tau0)


def fit_operator_pair(X0: np.ndarray, Xtau: np.ndarray, tau0: int) -> LimFit:
    """Fit a stationary LIM from a pre-split snapshot pair.

    Mirrors the MATLAB ``tx_lim_operator(X0, Xtau, tau0)`` signature for porting
    analysis scripts; ``Xtau[:, k]`` is the lag-``tau0`` snapshot of ``X0[:, k]``.
    """
    X0 = np.asarray(X0)
    Xtau = np.asarray(Xtau)
    if X0.shape != Xtau.shape:
        raise ValueError(
            f"X0 and Xtau must have the same shape; got {X0.shape} vs {Xtau.shape}"
        )
    if X0.ndim != 2:
        raise ValueError(f"X0 must be 2D (n_modes, n_samples); got shape {X0.shape}")
    n_samples = X0.shape[1]
    if n_samples < 2:
        raise ValueError(f"need at least 2 paired samples; got n_samples={n_samples}")

    C0 = X0 @ X0.T / (n_samples - 1)
    Ctau = Xtau @ X0.T / (n_samples - 1)

    # G = Ctau * inv(C0), equivalent to MATLAB's right-divide ``Ctau / C0``.
    G = np.linalg.solve(C0.T, Ctau.T).T

    L_complex = scipy.linalg.logm(G) / tau0
    real_norm = float(np.linalg.norm(L_complex.real))
    imag_norm = float(np.linalg.norm(L_complex.imag))
    if real_norm > 0 and imag_norm / real_norm > _IMAG_REL_TOL:
        warnings.warn(
            f"logm(G)/tau0 has nontrivial imaginary part "
            f"(||imag||/||real|| = {imag_norm / real_norm:.3e}); "
            "LIM stability assumption may be violated.",
            RuntimeWarning,
            stacklevel=2,
        )
    L = np.ascontiguousarray(L_complex.real)
    Q = -(L @ C0 + C0 @ L.T)
    return LimFit(L=L, Q=Q, C0=C0, Ctau=Ctau, G=G, tau0=tau0)


def propagator(L: np.ndarray, tau: float) -> np.ndarray:
    """Green function ``G(tau) = expm(L * tau)``.

    ``tau`` shares units with the ``tau0`` used to fit ``L``.
    """
    return scipy.linalg.expm(L * tau)
