"""Tests for ``lim.cyclo``: cyclostationary LIM fits and propagators."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from lim.cyclo import CycloLimFit, cyclo_propagator, fit_cyclo
from lim.operator import LimFit
from tests.test_operator_synthetic import _simulate_var1, _stable_L


def _periodic_phase(n_times: int, period: int) -> np.ndarray:
    return np.arange(n_times) % period


def test_cyclo_fit_returns_one_fit_per_phase() -> None:
    L_true = _stable_L(3, seed=0)
    Q_true = np.eye(3) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=40_000, seed=99)[:, 1_000:]
    period = 13
    phase = _periodic_phase(X.shape[1], period)
    fit = fit_cyclo(X, phase, tau0=1, period=period)
    assert isinstance(fit, CycloLimFit)
    assert len(fit.fits) == period
    for f in fit.fits:
        assert isinstance(f, LimFit)
        assert f.L.shape == (3, 3)
        assert f.tau0 == 1


def test_cyclo_recovers_stationary_when_data_is_stationary() -> None:
    """Data from a single L: every phase fit should approximate that L."""
    L_true = _stable_L(3, seed=0)
    Q_true = np.eye(3) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=80_000, seed=99)[:, 2_000:]
    period = 4
    phase = _periodic_phase(X.shape[1], period)
    fit = fit_cyclo(X, phase, tau0=1, period=period)
    Ls = np.stack([f.L for f in fit.fits])
    spread = float(np.std(Ls, axis=0).max())
    mean_norm = float(np.linalg.norm(Ls.mean(axis=0)))
    # Per-phase L's should be close to each other relative to their magnitude.
    assert spread / max(mean_norm, 1e-10) < 0.3


def test_cyclo_propagator_at_zero_steps_is_identity() -> None:
    L_true = _stable_L(3, seed=0)
    Q_true = np.eye(3) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=20_000, seed=99)[:, 1_000:]
    fit = fit_cyclo(X, _periodic_phase(X.shape[1], 4), tau0=1, period=4)
    G0 = cyclo_propagator(fit, phase0=0, n_steps=0)
    assert np.allclose(G0, np.eye(3))


def test_cyclo_propagator_full_cycle_matches_period_product() -> None:
    """A full-cycle propagator equals the product of all per-phase propagators."""
    L_true = _stable_L(3, seed=0)
    Q_true = np.eye(3) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=20_000, seed=99)[:, 1_000:]
    period = 5
    fit = fit_cyclo(X, _periodic_phase(X.shape[1], period), tau0=1, period=period)
    G_full = cyclo_propagator(fit, phase0=0, n_steps=period)
    expected = np.eye(3)
    for p in range(period):
        expected = scipy.linalg.expm(fit.fits[p].L) @ expected
    assert np.allclose(G_full, expected, atol=1e-12)


def test_cyclo_propagator_phase_wrapping() -> None:
    """phase0 + n_steps wraps modulo period; check by comparing matched phase sequences."""
    L_true = _stable_L(3, seed=0)
    Q_true = np.eye(3) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=20_000, seed=99)[:, 1_000:]
    period = 4
    fit = fit_cyclo(X, _periodic_phase(X.shape[1], period), tau0=1, period=period)
    # Two cycles starting at phase 2 = 8 single-phase steps; should match
    # cycling phase 2 → 3 → 0 → 1 twice.
    G = cyclo_propagator(fit, phase0=2, n_steps=2 * period)
    expected = np.eye(3)
    for k in range(2 * period):
        p = (2 + k) % period
        expected = scipy.linalg.expm(fit.fits[p].L) @ expected
    assert np.allclose(G, expected, atol=1e-12)


def test_fit_cyclo_rejects_invalid_inputs() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 200))
    phase_ok = _periodic_phase(200, 4)
    with pytest.raises(ValueError, match="2D"):
        fit_cyclo(rng.standard_normal(200), phase_ok, tau0=1, period=4)
    with pytest.raises(ValueError, match="phase must"):
        fit_cyclo(X, phase_ok[:100], tau0=1, period=4)
    with pytest.raises(ValueError, match="phase values"):
        fit_cyclo(X, phase_ok + 100, tau0=1, period=4)
    with pytest.raises(ValueError, match="tau0"):
        fit_cyclo(X, phase_ok, tau0=0, period=4)
    with pytest.raises(ValueError, match="period"):
        fit_cyclo(X, phase_ok, tau0=1, period=1)
