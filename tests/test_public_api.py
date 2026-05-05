"""Smoke test: every public symbol is importable from the top-level ``lim`` package."""

from __future__ import annotations

import lim


def test_public_api_is_importable() -> None:
    expected = {
        "CycloLimFit",
        "EofResult",
        "GridLimModel",
        "LimFit",
        "TrendMode",
        "cyclo_propagator",
        "deseasonalize",
        "eof",
        "fit_cyclo",
        "fit_lim_from_grid",
        "fit_operator",
        "fit_operator_pair",
        "forecast_deterministic",
        "forecast_ensemble",
        "project",
        "propagator",
        "reconstruct",
        "simulate",
        "trend_mode",
    }
    assert set(lim.__all__) == expected
    for name in expected:
        assert hasattr(lim, name), f"lim.{name} missing"
