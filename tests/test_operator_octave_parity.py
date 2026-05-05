"""Parity test against an Octave-generated MATLAB-format fixture.

To regenerate the fixture::

    octave --no-gui --eval "addpath('matlab','scripts'); generate_matlab_fixture"

The test is skipped automatically when the fixture is absent so the suite
remains runnable in environments without Octave.

Tolerance is ``atol=1e-10``: Octave's ``logm`` differs from MATLAB at the
lowest decimals, and our right-divide via ``np.linalg.solve`` is not bit-
identical to MATLAB's ``Ctau / C0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lim.operator import fit_operator

FIXTURE = Path(__file__).parent / "data" / "matlab_reference" / "lim_fixture.mat"


@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=(
        "no Octave/MATLAB parity fixture; "
        "run scripts/generate_matlab_fixture.m to generate it"
    ),
)
def test_operator_matches_octave_fixture() -> None:
    from scipy.io import loadmat

    data = loadmat(str(FIXTURE))
    X = np.asarray(data["X"])
    tau0 = int(np.atleast_1d(data["tau0"]).ravel()[0])
    L_oct = np.asarray(data["L"])
    Q_oct = np.asarray(data["Q"])

    fit = fit_operator(X, tau0)
    assert np.allclose(fit.L, L_oct, atol=1e-10), (
        f"max |L_py - L_oct| = {np.max(np.abs(fit.L - L_oct)):.3e}"
    )
    assert np.allclose(fit.Q, Q_oct, atol=1e-10), (
        f"max |Q_py - Q_oct| = {np.max(np.abs(fit.Q - Q_oct)):.3e}"
    )
