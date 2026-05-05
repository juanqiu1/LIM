# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

MATLAB implementation of the Linear Inverse Model (LIM; Penland & Sardeshmukh 1995) accompanying Xu et al. 2022. The code fits a stochastically forced linear dynamical system

    dx/dt = L x + xi

to climate-state time series `x(t)` (typically leading EOF amplitudes). A Python port (in progress) extends this to weekly Arctic sea-ice prediction with cyclostationary fitting, full xarray preprocessing, and deterministic + ensemble forecasting.

## Current state

Three standalone MATLAB functions at the repo root (`tx_lim_operator.m`, `tx_lim_trend.m`, `tx_lim_simulation.m`). No build, lint, or tests yet. Run them in MATLAB or Octave (with the `statistics` package for `normrnd`).

## Architecture (MATLAB)

The three functions form a small dependency chain — read them together:

- `tx_lim_operator.m` — fits the operator. Given snapshots `X0 = X(:, 1:end-tau0)` and `Xtau = X(:, 1+tau0:end)`, computes the lag-0 and lag-`tau0` covariances, the Green function `G = Ctau / C0`, then `L = logm(G)/tau0` and the noise covariance `Q = -(L*C0 + C0*L')`. Core primitive; the other two call it.
- `tx_lim_trend.m` — eigendecomposes `L`, sorts modes by `real(eig)` descending, returns the **least damped** mode (largest `real(lambda)`) as trend pattern `u`, time series `alpha = v'*x`, and rank-1 reconstruction `Xtr = u*alpha`. Adjoint eigenvectors come from `V = inv(U)'`. Header warns: only valid when `DD(1)` is real.
- `tx_lim_simulation.m` — stochastic integration. Eigendecomposes `Q`, **drops negative eigenvalues and rescales the positive ones to preserve the trace**, then integrates `x_{t+dt} = (I + L*dt) x_t + V_p sqrt(D_p*dt) * randn` forward. A 2000-year spinup is discarded before sampling.

Conventions:
- `X` is `[n_modes x n_times]` — columns are time. Monthly cadence assumed.
- `tau0` is in months; `dt = 16/24/30` (months) is hardcoded in `tx_lim_simulation.m`.
- Simulation output `Xg(:,:,k)` has shape `[n_modes x samplelen x group]`. `group` **must be a multiple of `subgroup = 20`**, and `samplelen*group/subgroup` is the per-trajectory step count.

## Known caveat (MATLAB)

`tx_lim_trend.m:37` references `X` (`alpha = v'*X`), but the function signature only declares `X0` and `Xtau`. Treat this as a real bug; downstream callers either pass the full series via a workspace variable named `X` or need to patch the function. The Python port (below) fixes this by making `X` an explicit argument; the `.m` file will be retained verbatim in `matlab/` for archival.

## Planned Python port

The repo will gain a Python implementation alongside the MATLAB code. Layout (after migration):

```
matlab/                  # original .m files moved here verbatim
src/lim/                 # Python package (PEP 621, Python ≥3.10)
  operator.py            # fit_operator, propagator (expm)
  cyclo.py               # cyclostationary LIM (52 weekly phases)
  forecast.py            # deterministic + ensemble forecasts
  simulation.py          # free-running stochastic simulation
  trend.py               # least-damped mode (X-bug fixed)
  eof.py                 # masked, area-weighted SVD
  seasonal.py            # deseasonalization (default: 3 annual harmonics)
  pipeline.py            # gridded NetCDF → PCs → LIM → grid glue
tests/
scripts/generate_matlab_fixture.m   # Octave-runnable; produces parity .mat
examples/arctic_sic_weekly.ipynb
pyproject.toml
```

Commands (post-port): `pip install -e .[dev]`, `pytest`, `ruff check src tests`, `mypy src/`.

### Design decisions locked in

- **Unit-agnostic time**: callers pass `dt`, `tau0`, `lead` in whatever shared unit they choose; `L` is `1/[that unit]`. The MATLAB hardcoded `dt = 16/24/30` is dropped.
- **Cyclostationary LIM is first-class in v1** (not a v2 extension), because sea ice has a strong seasonal cycle. Pipeline takes `mode={"stationary", "cyclo"}`; `cyclo` fits 52 weekly operators and the `cyclo_propagator` is the product of single-phase propagators around the cycle.
- **Three forecast/sim modes** with separate functions: `forecast_deterministic` (`expm(L*tau) x0`), `forecast_ensemble` (initial-value problem with stochastic forcing), `simulate` (free-running long control runs — the closest analog to MATLAB `tx_lim_simulation.m`).
- **Default deseasonalization**: 3 annual harmonics (smooth, leap-year-safe, standard in sea-ice LIM literature). DOY/WOY climatologies are alternatives.
- **EOF**: roll our own thin wrapper (masked, `cos(lat)`-weighted SVD) — `xeofs` deferred (heavy dep, partial-NaN handling conflicts with sea-ice masks).
- **Parity testing**: an Octave step in CI runs `scripts/generate_matlab_fixture.m` to produce `L`, `Q`, `Xtr` from a fixed random `X`; Python asserts `np.allclose` at `atol=1e-10` (Octave's `logm` differs from MATLAB at the lowest decimals — don't tighten further).
- **Trend bug**: fixed in `lim.trend.trend_mode` by making `X` an explicit required argument; the MATLAB `tx_lim_trend.m` is retained verbatim in `matlab/`.
- **API divergence**: simulation output is `(n_modes, n_steps, n_members)`, not the MATLAB `subgroup × group/subgroup` reshuffle (`tx_lim_simulation.m:46-53`). Spinup is a kwarg (`spinup_steps=24000`), not hardcoded `12*2000`.

### Source-of-truth note

When MATLAB and Python disagree, the Python port is authoritative for active development; MATLAB is preserved for reference and parity testing only.

## Branch policy

This worktree is on `claude/add-claude-documentation-kOvjw`. Develop, commit, and push to that branch unless the user says otherwise.
