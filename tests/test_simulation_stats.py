"""Statistical tests for ``lim.simulation.simulate``.

Compares simulated trajectories' first and second moments against the fitted
``C0`` / ``Ctau``, rather than testing pointwise — RNG streams are not portable
across implementations and the goal is correctness of the stochastic process.
"""

from __future__ import annotations

import numpy as np
import pytest

from lim.operator import fit_operator
from lim.simulation import simulate
from tests.test_operator_synthetic import _simulate_var1, _stable_L


@pytest.fixture(scope="module")
def fit() -> object:
    L_true = _stable_L(3, seed=0)
    Q_true = np.eye(3) * 0.5
    X = _simulate_var1(L_true, Q_true, dt=0.05, n_steps=20_000, seed=99)[:, 1_000:]
    return fit_operator(X, tau0=4)


def test_simulate_output_shape(fit) -> None:
    Xg = simulate(fit, n_steps=50, dt=1.0, n_members=4, spinup_steps=10, rng=0)
    assert Xg.shape == (3, 50, 4)


def test_simulate_is_reproducible_with_seed(fit) -> None:
    a = simulate(fit, n_steps=20, dt=1.0, n_members=2, spinup_steps=5, rng=42)
    b = simulate(fit, n_steps=20, dt=1.0, n_members=2, spinup_steps=5, rng=42)
    assert np.array_equal(a, b)


def test_simulate_recovers_stationary_covariance(fit) -> None:
    """Long simulation: empirical cov(X) ≈ fit.C0 within ~10% Frobenius."""
    Xg = simulate(
        fit, n_steps=20_000, dt=0.5, n_members=4, spinup_steps=200, rng=7
    )
    flat = Xg.reshape(Xg.shape[0], -1)
    C_emp = flat @ flat.T / (flat.shape[1] - 1)
    rel_err = np.linalg.norm(C_emp - fit.C0) / np.linalg.norm(fit.C0)
    assert rel_err < 0.10, f"empirical-vs-fit cov mismatch {rel_err:.3e}"


def test_simulate_recovers_lag_covariance(fit) -> None:
    """Empirical lag-tau0 covariance from a long sim should approximate fit.Ctau."""
    Xg = simulate(
        fit, n_steps=20_000, dt=0.5, n_members=4, spinup_steps=200, rng=11
    )
    # Per-member lag-tau0 cov, then average.
    tau0 = fit.tau0
    accumulators = []
    for m in range(Xg.shape[2]):
        Xm = Xg[:, :, m]
        X0 = Xm[:, : Xm.shape[1] - tau0]
        Xt = Xm[:, tau0:]
        accumulators.append(Xt @ X0.T / (X0.shape[1] - 1))
    Ctau_emp = np.mean(accumulators, axis=0)
    rel_err = np.linalg.norm(Ctau_emp - fit.Ctau) / np.linalg.norm(fit.Ctau)
    assert rel_err < 0.15, f"empirical-vs-fit Ctau mismatch {rel_err:.3e}"


def test_simulate_members_are_independent(fit) -> None:
    """Two members in the same call should be uncorrelated (over a long run)."""
    Xg = simulate(
        fit, n_steps=20_000, dt=0.5, n_members=2, spinup_steps=200, rng=3
    )
    a = Xg[:, :, 0].ravel()
    b = Xg[:, :, 1].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.05, f"|corr| {abs(corr):.3e} too large"


def test_simulate_rejects_invalid_inputs(fit) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        simulate(fit, n_steps=0)
    with pytest.raises(ValueError, match="n_members"):
        simulate(fit, n_steps=10, n_members=0)
    with pytest.raises(ValueError, match="spinup_steps"):
        simulate(fit, n_steps=10, spinup_steps=-1)
    with pytest.raises(ValueError, match="dt"):
        simulate(fit, n_steps=10, dt=0)
    with pytest.raises(ValueError, match="integer"):
        simulate(fit, n_steps=10, dt=0.3)
