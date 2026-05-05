"""Tests for ``lim.operator``: structural correctness and recovery of a known L."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from lim.operator import LimFit, fit_operator, fit_operator_pair, propagator


def _stable_L(n: int, *, seed: int = 0) -> np.ndarray:
    """Random stable LIM operator (all eigenvalues with negative real part)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    sym = -(A @ A.T) / n - 0.5 * np.eye(n)
    skew = 0.1 * (A - A.T)
    return sym + skew


def _simulate_var1(
    L: np.ndarray, Q: np.ndarray, *, dt: float, n_steps: int, seed: int
) -> np.ndarray:
    """Euler-Maruyama integration of dx = L x dt + noise; returns (n, n_steps)."""
    rng = np.random.default_rng(seed)
    n = L.shape[0]
    Lc = scipy.linalg.cholesky(Q + 1e-12 * np.eye(n), lower=True)
    coef = np.eye(n) + L * dt
    X = np.empty((n, n_steps))
    x = np.zeros(n)
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        x = coef @ x + Lc @ rng.standard_normal(n) * sqrt_dt
        X[:, k] = x
    return X


@pytest.fixture(scope="module")
def lim_like_X() -> np.ndarray:
    """Short trajectory from a known stable L; cached for module-scoped tests."""
    L_true = _stable_L(4, seed=0)
    Q_true = np.eye(4) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=20_000, seed=99)
    return X[:, 1_000:]  # discard transient


def test_fit_operator_matches_hand_computation(lim_like_X: np.ndarray) -> None:
    """fit_operator output matches direct numpy computation of C0, Ctau, G, L, Q."""
    tau0 = 3
    fit = fit_operator(lim_like_X, tau0)

    n_samples = lim_like_X.shape[1] - tau0
    X0 = lim_like_X[:, :n_samples]
    Xtau = lim_like_X[:, tau0:]
    C0_expected = X0 @ X0.T / (n_samples - 1)
    Ctau_expected = Xtau @ X0.T / (n_samples - 1)
    G_expected = Ctau_expected @ np.linalg.inv(C0_expected)
    L_expected = np.real(scipy.linalg.logm(G_expected)) / tau0
    Q_expected = -(L_expected @ C0_expected + C0_expected @ L_expected.T)

    assert np.allclose(fit.C0, C0_expected, atol=1e-12)
    assert np.allclose(fit.Ctau, Ctau_expected, atol=1e-12)
    assert np.allclose(fit.G, G_expected, atol=1e-10)
    assert np.allclose(fit.L, L_expected, atol=1e-10)
    assert np.allclose(fit.Q, Q_expected, atol=1e-10)


def test_fit_operator_pair_matches_fit_operator(lim_like_X: np.ndarray) -> None:
    tau0 = 3
    a = fit_operator(lim_like_X, tau0)
    b = fit_operator_pair(
        lim_like_X[:, : lim_like_X.shape[1] - tau0], lim_like_X[:, tau0:], tau0
    )
    assert np.allclose(a.L, b.L)
    assert np.allclose(a.Q, b.Q)
    assert np.allclose(a.G, b.G)


def test_propagator_closes_logm_loop(lim_like_X: np.ndarray) -> None:
    """propagator(L, tau0) should reproduce G_fit (expm(logm(G)) == G)."""
    fit = fit_operator(lim_like_X, tau0=5)
    assert np.allclose(propagator(fit.L, fit.tau0), fit.G, atol=1e-10)


def test_fit_recovers_known_L_within_sampling_error() -> None:
    """Long simulation from L_true: fit recovers L_true to ~few percent."""
    L_true = _stable_L(3, seed=0)
    Q_true = np.eye(3) * 0.5
    X = _simulate_var1(L_true, Q_true, dt=0.01, n_steps=200_000, seed=42)
    # Discard a transient.
    X = X[:, 10_000:]
    fit = fit_operator(X, tau0=10)
    # tau0=10 steps × dt=0.01 = lag 0.1 in physical units; L is in 1/step,
    # so L_fit should approximate L_true * dt (since simulation step IS the unit).
    rel_err = np.linalg.norm(fit.L - L_true * 0.01) / np.linalg.norm(L_true * 0.01)
    assert rel_err < 0.10, f"relative error {rel_err:.3e} > 10%"


def test_fit_operator_rejects_invalid_inputs() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2D"):
        fit_operator(rng.standard_normal(100), tau0=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        fit_operator(rng.standard_normal((3, 100)), tau0=0)
    with pytest.raises(ValueError, match="positive integer"):
        fit_operator(rng.standard_normal((3, 100)), tau0=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="< n_times"):
        fit_operator(rng.standard_normal((3, 10)), tau0=10)


def test_limfit_is_immutable(lim_like_X: np.ndarray) -> None:
    """LimFit is a frozen dataclass; assignment should fail."""
    fit = fit_operator(lim_like_X, tau0=2)
    assert isinstance(fit, LimFit)
    with pytest.raises(Exception):
        fit.tau0 = 99  # type: ignore[misc]
