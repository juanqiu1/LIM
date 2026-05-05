"""Tests for ``lim.forecast.forecast_deterministic``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from lim.forecast import forecast_deterministic, forecast_ensemble
from lim.operator import fit_operator
from tests.test_operator_synthetic import _simulate_var1, _stable_L


@pytest.fixture(scope="module")
def fit() -> object:
    L_true = _stable_L(4, seed=0)
    Q_true = np.eye(4) * 0.3
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=20_000, seed=99)[:, 1_000:]
    return fit_operator(X, tau0=4)


def test_forecast_at_zero_lead_returns_initial_state(fit) -> None:
    x0 = np.array([1.0, -0.5, 0.2, 0.8])
    out = forecast_deterministic(fit, x0, leads=0.0)
    assert out.shape == (4, 1, 1)
    assert np.allclose(out[:, 0, 0], x0, atol=1e-12)


def test_forecast_at_tau0_equals_G_times_x0(fit) -> None:
    """At lead=tau0, forecast should equal G @ x0 to numerical precision."""
    x0 = np.array([1.0, -0.5, 0.2, 0.8])
    out = forecast_deterministic(fit, x0, leads=fit.tau0)
    expected = fit.G @ x0
    assert np.allclose(out[:, 0, 0], expected, atol=1e-10)


def test_forecast_matches_propagator(fit) -> None:
    """forecast_deterministic == propagator(L, tau) @ x0 for each lead."""
    rng = np.random.default_rng(5)
    x0 = rng.standard_normal((4, 7))
    leads = [1.0, 3.5, 10.0]
    out = forecast_deterministic(fit, x0, leads)
    for k, tau in enumerate(leads):
        expected = scipy.linalg.expm(fit.L * tau) @ x0
        assert np.allclose(out[:, :, k], expected, atol=1e-10)


def test_forecast_decays_for_stable_L(fit) -> None:
    """For a stable L, forecasts at long lead converge to 0."""
    x0 = np.ones(4) * 5.0
    out = forecast_deterministic(fit, x0, leads=[0.0, 100.0, 1000.0])
    norms = np.linalg.norm(out[:, 0, :], axis=0)
    assert norms[0] > norms[1] > norms[2]
    assert norms[2] < 1e-6


def test_forecast_shape_preserves_n_init(fit) -> None:
    rng = np.random.default_rng(13)
    x0_2d = rng.standard_normal((4, 5))
    out = forecast_deterministic(fit, x0_2d, leads=[1.0, 2.0])
    assert out.shape == (4, 5, 2)

    x0_1d = rng.standard_normal(4)
    out = forecast_deterministic(fit, x0_1d, leads=[1.0, 2.0])
    assert out.shape == (4, 1, 2)


def test_forecast_rejects_invalid_inputs(fit) -> None:
    with pytest.raises(ValueError, match="modes"):
        forecast_deterministic(fit, np.zeros(99), leads=1.0)
    with pytest.raises(ValueError, match="1D or 2D"):
        forecast_deterministic(fit, np.zeros((4, 2, 2)), leads=1.0)


def test_ensemble_at_zero_lead_returns_x0(fit) -> None:
    x0 = np.array([1.0, -0.5, 0.2, 0.8])
    out = forecast_ensemble(fit, x0, leads=0.0, n_members=8, rng=0)
    assert out.shape == (4, 1, 1, 8)
    for m in range(8):
        assert np.allclose(out[:, 0, 0, m], x0, atol=1e-12)


def test_ensemble_shape_with_multiple_inits(fit) -> None:
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((4, 3))
    out = forecast_ensemble(fit, x0, leads=[1.0, 2.0, 5.0], n_members=10, rng=0)
    assert out.shape == (4, 3, 3, 10)


def test_ensemble_mean_approaches_deterministic_forecast(fit) -> None:
    """E[ensemble] should match expm(L*tau)*x0 for large n_members."""
    x0 = np.array([1.0, -0.5, 0.2, 0.8])
    leads = [1.0, 5.0]
    ens = forecast_ensemble(fit, x0, leads=leads, n_members=2000, rng=0)
    det = forecast_deterministic(fit, x0, leads=leads)
    ens_mean = ens.mean(axis=-1)
    rel_err = np.linalg.norm(ens_mean - det) / np.linalg.norm(det)
    assert rel_err < 0.10, f"ensemble-mean error {rel_err:.3e}"


def test_ensemble_long_lead_covariance_approaches_C0(fit) -> None:
    """At long lead, ensemble covariance approaches the stationary covariance C0."""
    x0 = np.zeros(4)
    out = forecast_ensemble(fit, x0, leads=200.0, n_members=2000, rng=0)
    samples = out[:, 0, 0, :]
    C_emp = samples @ samples.T / (samples.shape[1] - 1)
    rel_err = np.linalg.norm(C_emp - fit.C0) / np.linalg.norm(fit.C0)
    assert rel_err < 0.15, f"|C_emp - C0| / |C0| = {rel_err:.3e}"


def test_ensemble_is_reproducible(fit) -> None:
    x0 = np.array([1.0, 0.0, 0.0, 0.0])
    a = forecast_ensemble(fit, x0, leads=[1.0, 2.0], n_members=4, rng=42)
    b = forecast_ensemble(fit, x0, leads=[1.0, 2.0], n_members=4, rng=42)
    assert np.array_equal(a, b)


def test_ensemble_rejects_invalid_inputs(fit) -> None:
    x0 = np.zeros(4)
    with pytest.raises(ValueError, match="multiple of dt"):
        forecast_ensemble(fit, x0, leads=0.7, n_members=4, dt=1.0)
    with pytest.raises(ValueError, match="leads must be >= 0"):
        forecast_ensemble(fit, x0, leads=-1.0, n_members=4)
    with pytest.raises(ValueError, match="n_members"):
        forecast_ensemble(fit, x0, leads=1.0, n_members=0)
