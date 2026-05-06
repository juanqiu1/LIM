# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Arctic sea-ice LIM (weekly) — end-to-end example
#
# This notebook walks through the full pipeline:
#
# 1. Load Arctic sea-ice concentration (HadISST, monthly — the same code path
#    works on weekly NSIDC or OISST data; just change the `xr.open_dataset`
#    call).
# 2. Deseasonalize (3 annual harmonics).
# 3. EOF on a masked, latitude-weighted polar field.
# 4. Fit both a stationary LIM and a cyclostationary LIM (CS-LIM, one operator
#    per phase of the cycle).
# 5. Deterministic and ensemble forecasts; free-running simulation.
#
# **Open in Jupyter**: this file is a [jupytext](https://jupytext.readthedocs.io/)
# percent-format script. To get a `.ipynb`:
# ```bash
# jupytext --to notebook examples/arctic_sic_weekly.py
# ```
# Or open the `.py` directly in VS Code (renders as cells) or in JupyterLab
# (with the jupytext extension).

# %%
import io
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

import lim

# %% [markdown]
# ## 1. Load Arctic sea-ice concentration
#
# **Primary path**: HadISST sea-ice concentration from the Met Office
# (monthly, 1° gridded, 1870-present, single ~50 MB file).
#
# **Fallback**: if the download fails (corporate firewall, etc.), generate a
# synthetic Arctic-like field so the rest of the notebook still runs. Replace
# this whole cell with your own loader for weekly NSIDC/OISST data when you're
# ready.

# %%
HADISST_URL = "https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_ice.nc.gz"


def _load_hadisst() -> xr.DataArray:
    import gzip
    req = urllib.request.Request(HADISST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        gz_bytes = resp.read()
    nc_bytes = gzip.decompress(gz_bytes)
    ds = xr.open_dataset(io.BytesIO(nc_bytes))
    return ds["sic"].rename({"latitude": "lat", "longitude": "lon"})


def _synthetic_arctic(n_years: int = 40) -> xr.DataArray:
    """Pure-synthetic Arctic-like SIC field; only used when the download fails."""
    rng = np.random.default_rng(0)
    time = pd.date_range("1980-01-15", periods=12 * n_years, freq="MS")
    lat = np.arange(60.5, 89.6, 1.0)
    lon = np.arange(0.5, 360.0, 2.0)
    days = (time - time[0]).total_seconds().to_numpy() / 86400.0

    seasonal = 0.5 + 0.45 * np.cos(2 * np.pi * days / 365.25 + np.pi)
    lat_envelope = np.clip((lat - 60) / 30, 0, 1)
    base = seasonal[:, None, None] * lat_envelope[None, :, None] * np.ones_like(lon)[None, None, :]

    pat = np.outer(np.exp(-((lat - 75) / 10) ** 2), np.cos(np.deg2rad(lon)))
    pc = 0.05 * np.cos(2 * np.pi * np.arange(time.size) / 60) + 0.02 * rng.standard_normal(time.size)
    field = base + pc[:, None, None] * pat[None, :, :] + 0.005 * rng.standard_normal(base.shape)

    field = np.clip(field, 0, 1)
    return xr.DataArray(
        field, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon")
    )


try:
    sic = _load_hadisst()
    print(f"Loaded HadISST: {sic.sizes}, time {sic.time.values[0]} → {sic.time.values[-1]}")
except Exception as exc:  # pragma: no cover - depends on network
    print(f"HadISST download failed ({exc!r}); using synthetic Arctic field.")
    sic = _synthetic_arctic()

# %% [markdown]
# ## 2. Restrict to the Arctic and to a recent window
#
# HadISST goes back to 1870 but the early decades are observation-sparse.
# Take 1980-onwards and keep latitudes ≥ 50°N.

# %%
sic = sic.sel(lat=slice(89, 50)) if sic.lat[0] > sic.lat[-1] else sic.sel(lat=slice(50, 89))
sic = sic.sel(time=slice("1980-01-01", None))
sic = sic.where((sic >= 0) & (sic <= 1))  # mask invalid / land sentinel values
print(sic.sizes)

# %% [markdown]
# ## 3. Run the full pipeline (stationary)
#
# `fit_lim_from_grid` deseasonalizes (3-harmonic default), takes EOFs with
# `cos(lat)` weighting and the implicit land mask, and fits a stationary LIM
# on the leading PCs.

# %%
model = lim.fit_lim_from_grid(
    sic,
    tau0=1,            # 1-month lag for the fit
    n_eofs=5,
    mode="stationary",
)
print("L shape:", model.lim.L.shape)
print("Variance explained by leading EOFs:", model.eof.eigvals)

# %% [markdown]
# ## 4. Deterministic and ensemble forecasts at lead 1–6
#
# Forecast the most recent state forward, in PC space.

# %%
x_now = model.eof.pcs.isel(time=-1).values
det = lim.forecast_deterministic(model.lim, x_now, leads=[1, 3, 6])
ens = lim.forecast_ensemble(model.lim, x_now, leads=[1, 3, 6], n_members=200, rng=0)
print("deterministic shape:", det.shape, " | ensemble shape:", ens.shape)

ens_mean = ens.mean(axis=-1)
ens_std = ens.std(axis=-1)
for k, lead in enumerate([1, 3, 6]):
    print(
        f"lead={lead}  ‖det-ens_mean‖={np.linalg.norm(det[:, 0, k] - ens_mean[:, 0, k]):.3f}  "
        f"mean_std={ens_std[:, 0, k].mean():.3f}"
    )

# %% [markdown]
# ## 5. Cyclostationary LIM
#
# Refit with one operator per phase of the year. `period=12` for monthly
# data; switch to `period=52` (and use weekly input) for weekly forecasting.

# %%
cs_model = lim.fit_lim_from_grid(
    sic,
    tau0=1,
    n_eofs=5,
    mode="cyclo",
    period=12,
)
print("CS-LIM has", len(cs_model.lim.fits), "phase operators")

# Eigenvalues of L_p across phases give a sense of how the dynamics rotate
# with the seasonal cycle.
import numpy as np
phase_real_min = np.array([np.linalg.eigvals(f.L).real.max() for f in cs_model.lim.fits])
print("Least-damped real eigenvalue per phase:", np.round(phase_real_min, 3))

# %% [markdown]
# ## 6. Free-running simulation
#
# Generate a 50-year Monte-Carlo simulation in PC space; the long-run
# covariance should match `fit.C0` and is useful as a null hypothesis
# for variability metrics.

# %%
try:
    sim = lim.simulate(
        model.lim,
        n_steps=12 * 50,
        dt=0.5,
        n_members=4,
        spinup_steps=2_000,
        rng=42,
    )
    print("simulation shape:", sim.shape)
    flat = sim.reshape(sim.shape[0], -1)
    emp_cov = flat @ flat.T / (flat.shape[1] - 1)
    rel_err = np.linalg.norm(emp_cov - model.lim.C0) / np.linalg.norm(model.lim.C0)
    print(f"long-run cov vs fit.C0: relative error = {rel_err:.3f}")
except RuntimeError as exc:
    print(f"simulate() did not converge ({exc!r}).")
    print("This typically means the fitted L has a non-physical positive-real eigenvalue,")
    print("which can happen with short / synthetic data; with full HadISST it should be stable.")
    eigvals = np.linalg.eigvals(model.lim.L)
    print("eig(L) real parts:", np.round(eigvals.real, 3))

# %% [markdown]
# ## Adapting to weekly Arctic SIC
#
# To run this on weekly NSIDC G02202 or OISST V2.1 sea-ice concentration:
#
# 1. Replace the `_load_hadisst()` call with your own loader (e.g.
#    `xr.open_dataset(<NSIDC URL or local path>)`).
# 2. Resample to weekly if your source is daily: `sic = sic.resample(time="7D").mean()`.
# 3. Keep `tau0=1` (1 week) and switch the cyclo `period` to `52`.
# 4. Use `lim.simulate(..., n_steps=52 * 50, dt=1.0)` for a 50-year weekly run.
#
# The rest of this notebook is unchanged.
