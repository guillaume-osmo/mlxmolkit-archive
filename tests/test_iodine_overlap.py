"""Regression guard for the n=5 (iodine) two-center Slater-overlap fix.

The vendored PYSEQM overlap table mis-transcribes the qn>=5 reduced-overlap coefficients,
producing local overlaps > 1 (s-d ~ 27) and a catastrophic SCF collapse (CH2I2 Hf = -13937
kcal). slater_overlap_ref recomputes the 14 reduced overlaps exactly for qn>=5. These tests
fail loudly if that regresses.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.rm1 import nddo_energy
from mlxmolkit.rm1.methods import get_params
from mlxmolkit.rm1.overlap_d import overlap_d_molecular_frame

P = get_params("PM6_D")


@pytest.mark.parametrize("zA,zB,R", [(53, 53, 2.66), (53, 35, 2.50), (53, 6, 2.10), (53, 1, 1.60)])
def test_iodine_overlaps_are_physical(zA, zB, R):
    """Every AO overlap magnitude must be <= 1 (was up to 27 for I-I)."""
    for axis in (np.array([0.0, 0.0, R]), np.array([R * 0.6, R * 0.5, R * 0.62])):
        S = np.asarray(overlap_d_molecular_frame(P[zA], P[zB], np.zeros(3), axis / np.linalg.norm(axis) * R))
        assert np.abs(S).max() <= 1.0 + 1e-6, f"Z={zA},{zB}: max|S|={np.abs(S).max():.3f}"


@pytest.mark.parametrize("atoms,coords", [
    # I2 at 2.66 A (was -21482 kcal)
    ([53, 53], [(0.0, 0.0, 0.0), (0.0, 0.0, 2.66)]),
    # CH2I2-like: two iodines on one carbon (was -13937 kcal)
    ([6, 53, 53, 1, 1], [(0, 0, 0), (0, 1.5, 1.5), (0, -1.5, 1.5), (0.9, 0, -0.6), (-0.9, 0, -0.6)]),
])
def test_multi_iodine_does_not_collapse(atoms, coords):
    r = nddo_energy(atoms, np.array(coords, float), method="PM6_D", max_iter=300)
    assert r["converged"]
    # a sane PM6 heat of formation is well within +-500 kcal/mol; the bug gave < -10000
    assert -500.0 < r["heat_of_formation_kcal"] < 500.0, r["heat_of_formation_kcal"]
