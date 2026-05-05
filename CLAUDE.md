# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

MATLAB implementation of the Linear Inverse Model (LIM; Penland & Sardeshmukh 1995) accompanying Xu et al. 2022. The code fits a stochastically forced linear dynamical system

    dx/dt = L x + xi

to climate-state time series `x(t)` (typically leading EOF amplitudes), and uses it for trend extraction and ensemble simulation.

## Tooling

There is no build, lint, package manager, or test suite. The three `.m` files are standalone MATLAB functions intended to be called from a user's own analysis script. Run them in MATLAB (or Octave with the `statistics` package for `normrnd`); `logm` and `eig` are built-ins.

## Architecture

The three functions form a small dependency chain — read them together to understand the data flow:

- `tx_lim_operator.m` — fits the operator. Given snapshots `X0 = X(:, 1:end-tau0)` and `Xtau = X(:, 1+tau0:end)`, computes the lag-0 and lag-`tau0` covariances, the Green function `G = Ctau / C0`, then `L = logm(G)/tau0` and the noise covariance `Q = -(L*C0 + C0*L')`. This is the core primitive; the other two functions call it.
- `tx_lim_trend.m` — eigendecomposes `L`, sorts modes by `real(eig)` descending, and returns the **least damped** mode (largest `real(lambda)`) as the trend pattern `u`, its time series `alpha = v'*x`, and the rank-1 reconstruction `Xtr = u*alpha`. Adjoint eigenvectors come from `V = inv(U)'`. The function header warns: only valid when `DD(1)` is real (stationary least-damped mode).
- `tx_lim_simulation.m` — stochastic integration. Eigendecomposes `Q`, **drops negative eigenvalues and rescales the positive ones to preserve the trace**, then integrates `x_{t+dt} = (I + L*dt) x_t + V_p sqrt(D_p*dt) * randn` forward. A 2000-year spin-up is discarded before sampling.

Conventions used throughout:
- `X` is `[n_modes x n_times]` — columns are time, rows are state components (one column per month).
- `tau0` is in **months**, matching the monthly cadence of the input data and the `dt = 16/24/30` (months) integration step inside `tx_lim_simulation`.
- The simulation organizes output as `Xg(:,:,k)` of shape `[n_modes x samplelen x group]`. `group` **must be a multiple of `subgroup = 20`**, and `samplelen*group/subgroup` is the per-trajectory step count, so choose lengths accordingly.

## Known caveat

`tx_lim_trend.m:37` references `X` (`alpha = v'*X`), but the function signature only declares `X0` and `Xtau` — the variable `X` is undefined inside the function as written. Treat this as a real bug, not a stylistic issue; downstream callers either pass the full series in via a workspace variable named `X` or need to patch the function (e.g., reconstruct the full series from `[X0, Xtau(:,end)]` or change the signature to take `X` directly). Confirm with the user before "fixing" it, since changing the API affects callers.

## Branch policy

This worktree is on `claude/add-claude-documentation-kOvjw`. Develop, commit, and push to that branch unless the user says otherwise.
