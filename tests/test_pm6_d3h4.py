"""Smoke tests for PM6-D3H4 corrections (D3 dispersion + H4 H-bond + HH-rep).

The MOPAC-port behavior:
- D3 is a geometric pairwise sum; should give negative energies in kcal/mol.
- H4 returns negative values for proper N-H/O-H...N/O hydrogen bonds.
- HH-rep is always non-negative; small for non-bonded H pairs.

These tests check signs, magnitudes, and a handful of known reference values.
For bit-exact validation against MOPAC, run the same geometry through the
``mopac`` binary with ``PM6-D3H4`` keyword and compare the dispersion +
H-bond + HH-rep block (it's printed by MOPAC with the DERIV keyword).
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from mlxmolkit.rm1.pm6_d3h4 import (
    d3_energy, h4_energy, hh_repulsion, pm6_d3h4_correction,
    _poly_hh, _pauling_coordination, _load_c6ab, _load_r0ab,
    PM6_D3H4_DISP, BOHR, RCOV,
)


def test_tables_load():
    """c6ab and r0ab npz files load with expected shapes."""
    c6ab, mxc = _load_c6ab()
    assert c6ab.shape == (94, 94, 5, 5, 3)
    assert mxc.shape == (94,)
    # H has 2 CN slots, C has 5
    assert mxc[0] == 2
    assert mxc[5] == 5
    r0ab = _load_r0ab()
    assert r0ab.shape == (94, 94)
    assert r0ab[0, 0] > 0
    # Symmetric
    np.testing.assert_allclose(r0ab, r0ab.T, atol=0)


def test_coordination_number():
    """Pauling CN: H in H2 should be ~1; C in CH4 should be ~4."""
    # H2
    h2_atoms = [1, 1]
    h2_coords = np.array([[0., 0., 0.], [0.74, 0., 0.]])
    rcov_bohr = 4.0 / 3.0 * RCOV / BOHR
    cn = _pauling_coordination(h2_atoms, h2_coords / BOHR, rcov_bohr)
    # CN counts via a smooth step that saturates near 1 for covalent partners.
    # H2 (r ≈ 0.74 Å) gives CN ≈ 0.92 because Pauling rcov + 4/3 scaling for
    # H is on the soft end of the damping function.
    assert cn[0] == pytest.approx(0.92, abs=0.05)
    assert cn[1] == pytest.approx(0.92, abs=0.05)

    # CH4 (tetrahedral)
    ch4_atoms = [6, 1, 1, 1, 1]
    a = 0.629
    ch4_coords = np.array([
        [0, 0, 0],
        [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a],
    ])
    cn = _pauling_coordination(ch4_atoms, ch4_coords / BOHR, rcov_bohr)
    assert cn[0] == pytest.approx(4.0, abs=0.2)
    assert cn[1] == pytest.approx(1.0, abs=0.1)


def test_d3_dispersion_sign_and_scale():
    """D3 energy is negative; scales with system size."""
    # methane (small, mostly atomic dispersion)
    ch4_atoms = [6, 1, 1, 1, 1]
    a = 0.629
    ch4_coords = np.array([
        [0, 0, 0], [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a],
    ])
    r = d3_energy(ch4_atoms, ch4_coords)
    assert r['e_disp'] < 0  # attractive
    assert r['e_disp'] > -50.0  # not absurd
    # PM6-D3H4 zero-damping has s8 = 0 → e8 contribution is exactly zero
    assert r['e8'] == 0.0


def test_d3_benzene_vs_naphthalene():
    """Naphthalene should have more attractive D3 than benzene."""
    # Approximate planar coords (Å)
    def aromatic_ring(n_carbons, radius=1.4):
        atoms = [6] * n_carbons + [1] * n_carbons
        coords = []
        for k in range(n_carbons):
            theta = 2 * math.pi * k / n_carbons
            coords.append([radius * math.cos(theta), radius * math.sin(theta), 0])
        for k in range(n_carbons):
            theta = 2 * math.pi * k / n_carbons
            coords.append([(radius + 1.08) * math.cos(theta),
                           (radius + 1.08) * math.sin(theta), 0])
        return atoms, np.array(coords)
    b_atoms, b_coords = aromatic_ring(6)
    e_b = d3_energy(b_atoms, b_coords)['e_disp']
    # naphthalene (fused) — approximate
    n_atoms = b_atoms + [6, 6] + [1, 1]
    n_coords = np.vstack([
        b_coords,
        np.array([[2.4, 0.7, 0], [2.4, -0.7, 0]]),
        np.array([[3.5, 1.4, 0], [3.5, -1.4, 0]]),
    ])
    e_n = d3_energy(n_atoms, n_coords)['e_disp']
    assert e_n < e_b  # more atoms → more attraction


def test_h4_water_dimer():
    """H4 should fire (negative e_hb) for a proper water dimer."""
    atoms = [8, 1, 1, 8, 1, 1]
    coords = np.array([
        [0., 0., 0.],                 # O1
        [-0.586, 0.756, 0.],          # spectator H on O1
        [0.957, 0., 0.],              # donor H pointing at O2
        [2.91, 0., 0.],               # O2 (acceptor)
        [3.28, 0.756, 0.],
        [3.28, -0.756, 0.],
    ])
    e_hb = h4_energy(atoms, coords)
    assert e_hb < 0  # stabilizing
    # With water-scaling 0.42 applied, the bare H4 of ~-2.3 → ~-0.97
    assert -2.0 < e_hb < -0.3


def test_h4_zero_for_no_hbonders():
    """Methane has no N/O atoms; H4 should be 0."""
    atoms = [6, 1, 1, 1, 1]
    a = 0.629
    coords = np.array([
        [0, 0, 0], [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a],
    ])
    assert h4_energy(atoms, coords) == 0.0


def test_hh_repulsion_polynomial():
    """Spot-check poly(r) values."""
    assert _poly_hh(0.5) == pytest.approx(25.4629, abs=1e-3)  # capped at r ≤ 1
    # At r = 1.5 (transition), polynomial gives:
    # Cross-check approximation; should be smooth.
    p_15 = _poly_hh(1.5)
    p_exp_15 = 118.7326 * math.exp(-1.53965 * (1.5 ** 1.72905))
    assert p_15 == pytest.approx(p_exp_15, rel=1e-3)


def test_hh_zero_for_no_hydrogens():
    """CCl4: no H atoms, no HH-rep."""
    atoms = [6, 17, 17, 17, 17]
    coords = np.array([
        [0, 0, 0], [1.02, 1.02, 1.02], [-1.02, -1.02, 1.02],
        [-1.02, 1.02, -1.02], [1.02, -1.02, -1.02],
    ])
    assert hh_repulsion(atoms, coords) == 0.0


def test_full_correction_returns_all_components():
    """pm6_d3h4_correction packs all three terms + total."""
    atoms = [6, 1, 1, 1, 1]
    a = 0.629
    coords = np.array([
        [0, 0, 0], [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a],
    ])
    r = pm6_d3h4_correction(atoms, coords)
    assert set(r) == {'e_disp', 'e_hb', 'e_hh', 'e_total'}
    assert r['e_total'] == pytest.approx(r['e_disp'] + r['e_hb'] + r['e_hh'])
