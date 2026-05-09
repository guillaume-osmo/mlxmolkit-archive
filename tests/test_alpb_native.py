"""Tests for the pure-numpy ALPB(water) implementation."""
import numpy as np
import pytest

from mlxmolkit.xtb.solvation_alpb_native import (
    alpb_water_correction_native,
    compute_bornr,
)


def test_h2o_born_radii_sensible():
    """Born radii for water O/H should be in the typical range
    (~2-4 Bohr) and respect the heavy-atom > hydrogen ordering."""
    atoms = [8, 1, 1]
    coords = np.array([
        [0.0, 0.0, 0.117790],
        [0.0, 0.755, -0.471],
        [0.0, -0.755, -0.471],
    ])
    q_at = np.array([-0.563, 0.282, 0.282])  # vacuum-like Mulliken
    nat = alpb_water_correction_native(atoms, coords, q_at)
    brad = nat["brad_bohr"]
    # All in the 2-4 Bohr range
    assert all(2.0 < r < 4.5 for r in brad), f"unrealistic Born radii: {brad}"
    # O > H (heavier atom has larger Born radius for water-like
    # geometry under OBCII)
    assert brad[0] > brad[1] > 0
    assert abs(brad[1] - brad[2]) < 1e-6   # symmetric H atoms


def test_charged_ion_alpb_negative_and_large():
    """For a small charged ion (NH4+), ALPB(water) should give a
    strongly negative solvation energy (>50 kcal/mol stabilization)
    dominated by the Born term."""
    atoms = [7, 1, 1, 1, 1]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.6, 0.6, 0.6],
        [-0.6, -0.6, 0.6],
        [-0.6, 0.6, -0.6],
        [0.6, -0.6, -0.6],
    ])
    # Approx GFN2 charges for NH4+: N delocalises slight negative,
    # H atoms each carry ~+0.31
    q_at = np.array([-0.24, 0.31, 0.31, 0.31, 0.31])
    nat = alpb_water_correction_native(atoms, coords, q_at)
    e_kcal = nat["e_total_hartree"] * 627.5095
    assert e_kcal < -50, f"expected strongly stabilizing, got {e_kcal:.1f} kcal/mol"


def test_neutral_nonpolar_alpb_small():
    """For a neutral non-polar molecule (CH4), ALPB(water) should be
    near zero (small dipole, small charges → small Born coupling)."""
    atoms = [6, 1, 1, 1, 1]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.65, 0.65, 0.65],
        [-0.65, -0.65, 0.65],
        [-0.65, 0.65, -0.65],
        [0.65, -0.65, -0.65],
    ])
    q_at = np.array([-0.45, 0.11, 0.11, 0.11, 0.11])  # ~GFN2 vacuum
    nat = alpb_water_correction_native(atoms, coords, q_at)
    e_kcal = nat["e_total_hartree"] * 627.5095
    assert abs(e_kcal) < 5.0, f"|E| too large for non-polar: {e_kcal:.2f} kcal/mol"
