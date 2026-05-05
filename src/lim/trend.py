"""Least-damped mode extraction from a fitted LIM (a.k.a. the LIM trend mode).

Several studies (Frankignoul et al. 2017; Alexander et al. 2022; Di Lorenzo et
al. 2023) show that the externally forced trend in climate data is often
captured by the least-damped eigenmode of ``L``. This module extracts that mode.

The MATLAB original (``matlab/tx_lim_trend.m`` line 37) references an undefined
variable ``X`` inside the function body. This Python port fixes that bug by
making ``X`` an explicit required argument.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .operator import LimFit


@dataclass(frozen=True)
class TrendMode:
    """Least-damped eigenmode of ``L`` and its time-domain projection.

    Attributes
    ----------
    u : (n_modes,) array
        Spatial pattern (right eigenvector of ``L`` with the largest
        ``real(eigval)``).
    v : (n_modes,) array
        Adjoint eigenvector (column of ``inv(U).conj().T``).
    alpha : (n_times,) array
        Time series ``v.conj() @ X``.
    Xtr : (n_modes, n_times) array
        Rank-1 reconstruction ``outer(u, alpha)``: the trend component in
        whatever basis ``X`` is expressed (typically PC space).
    eigval : complex
        Leading eigenvalue of ``L``.
    """

    u: np.ndarray
    v: np.ndarray
    alpha: np.ndarray
    Xtr: np.ndarray
    eigval: complex


def trend_mode(fit: LimFit, X: np.ndarray) -> TrendMode:
    """Extract the least-damped mode of ``L`` and project ``X`` onto it.

    Parameters
    ----------
    fit : LimFit
        Output of :func:`lim.operator.fit_operator`.
    X : (n_modes, n_times) array
        State series in the same basis as the data used to fit ``fit``.

    Notes
    -----
    Warns when the leading eigenvalue is complex — in that regime the
    "trend mode" interpretation is no longer well-defined and the caller
    should inspect the eigenspectrum directly. (This matches the warning
    in ``matlab/tx_lim_trend.m``.)
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n_modes, n_times); got shape {X_arr.shape}")
    if X_arr.shape[0] != fit.L.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} modes but L has {fit.L.shape[0]}"
        )

    eigvals, U = np.linalg.eig(fit.L)
    order = np.argsort(-eigvals.real)  # least damped first
    eigvals = eigvals[order]
    U = U[:, order]
    V = np.linalg.inv(U).conj().T

    u = U[:, 0]
    v = V[:, 0]
    leading = complex(eigvals[0])

    if abs(leading.imag) > 1e-10 * max(abs(leading.real), 1.0):
        warnings.warn(
            f"leading eigenvalue is complex ({leading:.4g}); the LIM trend-mode "
            "interpretation assumes a real least-damped eigenvalue.",
            RuntimeWarning,
            stacklevel=2,
        )

    alpha = v.conj() @ X_arr
    Xtr = np.outer(u, alpha)

    return TrendMode(u=u, v=v, alpha=alpha, Xtr=Xtr, eigval=leading)
