"""Free-running stochastic simulation of a fitted LIM.

Ports ``matlab/tx_lim_simulation.m`` to Python. Two changes from the MATLAB
original:

* The hardcoded ``dt = 16/24/30`` (months) is dropped — ``dt`` and
  ``spinup_steps`` are kwargs in the user's chosen time unit.
* The output is shaped ``(n_modes, n_steps, n_members)`` directly, not the
  ``subgroup × group/subgroup`` reshuffle from ``tx_lim_simulation.m:46-53``.
"""

from __future__ import annotations

import numpy as np

from .operator import LimFit


def simulate(
    fit: LimFit,
    n_steps: int,
    *,
    dt: float = 1.0,
    n_members: int = 1,
    spinup_steps: int = 24_000,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Stochastic free-running simulation; returns ``(n_modes, n_steps, n_members)``.

    Integrates ``dx = L x dt + xi`` via Euler-Maruyama with substep ``dt`` and
    samples every ``round(1/dt)`` substeps so output samples are spaced by 1
    fit-time unit. The first ``spinup_steps`` output samples are discarded.

    The noise is shaped from the positive-eigenvalue subspace of ``Q``, with
    the eigenvalues rescaled to preserve ``trace(Q)`` (matches
    ``tx_lim_simulation.m:29``).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    if n_members < 1:
        raise ValueError(f"n_members must be >= 1; got {n_members}")
    if spinup_steps < 0:
        raise ValueError(f"spinup_steps must be >= 0; got {spinup_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0; got {dt}")

    substeps_per_output = int(round(1.0 / dt))
    if substeps_per_output < 1:
        raise ValueError(f"dt={dt} is too large; need 1/dt >= 1")
    if abs(substeps_per_output * dt - 1.0) > 1e-9:
        raise ValueError(
            f"1/dt must be (approximately) integer; got 1/dt = {1.0 / dt}"
        )

    rng_obj = np.random.default_rng(rng)
    n_modes = fit.L.shape[0]

    Q_sym = 0.5 * (fit.Q + fit.Q.T)
    eigvals, V = np.linalg.eigh(Q_sym)
    pos_mask = eigvals > 0
    if not pos_mask.any():
        raise ValueError("Q has no positive eigenvalues; cannot simulate")
    Dp = eigvals[pos_mask]
    Vp = V[:, pos_mask]
    trace_total = float(eigvals.sum())
    Dp = Dp * trace_total / float(Dp.sum())

    noise_amp = Vp * np.sqrt(Dp * dt)
    coef = np.eye(n_modes) + fit.L * dt
    n_pos = Dp.size

    x = np.zeros((n_modes, n_members))

    total_spinup_substeps = spinup_steps * substeps_per_output
    for _ in range(total_spinup_substeps):
        x = coef @ x + noise_amp @ rng_obj.standard_normal((n_pos, n_members))
    if not np.all(np.isfinite(x)):
        raise RuntimeError("simulation blew up during spinup")

    out = np.empty((n_modes, n_steps, n_members))
    for k in range(n_steps):
        for _ in range(substeps_per_output):
            x_prev = x
            x = coef @ x + noise_amp @ rng_obj.standard_normal((n_pos, n_members))
        out[:, k, :] = 0.5 * (x_prev + x)

    if not np.all(np.isfinite(out)):
        raise RuntimeError("simulation blew up")
    return out
