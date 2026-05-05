"""Tests for ``lim.trend.trend_mode``.

Covers the bug fix from ``matlab/tx_lim_trend.m`` line 37 (X-undefined) by
asserting that ``X`` must be supplied as an explicit argument.
"""

from __future__ import annotations

import numpy as np
import pytest

from lim.operator import fit_operator
from lim.trend import TrendMode, trend_mode
from tests.test_operator_synthetic import _simulate_var1, _stable_L


@pytest.fixture(scope="module")
def fit_and_X() -> tuple[object, np.ndarray]:
    L_true = _stable_L(4, seed=0)
    Q_true = np.eye(4) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=20_000, seed=99)[:, 1_000:]
    return fit_operator(X, tau0=4), X


def test_trend_mode_returns_least_damped_eigenvalue(fit_and_X) -> None:
    """The returned eigval should have the largest real part among eig(L)."""
    fit, X = fit_and_X
    tm = trend_mode(fit, X)
    all_eigvals = np.linalg.eigvals(fit.L)
    assert np.isclose(tm.eigval.real, all_eigvals.real.max(), atol=1e-12)


def test_trend_mode_outputs_have_correct_shapes(fit_and_X) -> None:
    fit, X = fit_and_X
    tm = trend_mode(fit, X)
    n_modes, n_times = X.shape
    assert tm.u.shape == (n_modes,)
    assert tm.v.shape == (n_modes,)
    assert tm.alpha.shape == (n_times,)
    assert tm.Xtr.shape == (n_modes, n_times)
    assert isinstance(tm, TrendMode)


def test_trend_mode_reconstruction_is_rank_one(fit_and_X) -> None:
    """Xtr = u outer alpha must have rank exactly 1."""
    fit, X = fit_and_X
    tm = trend_mode(fit, X)
    rank = np.linalg.matrix_rank(tm.Xtr.real, tol=1e-10)
    assert rank == 1, f"expected rank 1, got {rank}"


def test_trend_mode_reconstruction_matches_outer_product(fit_and_X) -> None:
    fit, X = fit_and_X
    tm = trend_mode(fit, X)
    assert np.allclose(tm.Xtr, np.outer(tm.u, tm.alpha), atol=1e-12)


def test_trend_mode_alpha_matches_adjoint_projection(fit_and_X) -> None:
    fit, X = fit_and_X
    tm = trend_mode(fit, X)
    assert np.allclose(tm.alpha, tm.v.conj() @ X, atol=1e-12)


def test_trend_mode_rejects_wrong_shape_X(fit_and_X) -> None:
    fit, _ = fit_and_X
    with pytest.raises(ValueError, match="2D"):
        trend_mode(fit, np.zeros(100))
    with pytest.raises(ValueError, match="modes"):
        trend_mode(fit, np.zeros((99, 100)))


def test_trend_mode_warns_on_complex_leading_eigenvalue() -> None:
    """If the least-damped mode is oscillatory (complex eigval), we warn."""
    # Construct an L whose least-damped mode is a damped oscillator.
    # Eigenvalues -0.05 +/- 1j (slowest) and -1.0, -1.0 (faster decay).
    L_blocks = np.array(
        [
            [-0.05, -1.0, 0.0, 0.0],
            [1.0, -0.05, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    )
    # Build a synthetic LimFit-compatible object: only L is needed by trend_mode.
    from lim.operator import LimFit

    fake_fit = LimFit(
        L=L_blocks,
        Q=np.eye(4),
        C0=np.eye(4),
        Ctau=np.eye(4),
        G=np.eye(4),
        tau0=1,
    )
    X = np.random.default_rng(0).standard_normal((4, 50))
    with pytest.warns(RuntimeWarning, match="complex"):
        trend_mode(fake_fit, X)
