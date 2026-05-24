"""Vendored NumPy port of selected PYSEQM modules (BSD-3 Clause, LANL).

Source: https://github.com/lanl/PYSEQM

Bit-exact to PYSEQM (~1e-15) — verified by tests/test_pyseqm_port.py.

Exports
-------
diatom_overlap_matrixD(ni, nj, xij, rij, zeta_a, zeta_b, qn_int, qnD_int)
    Diatomic overlap matrix in molecular frame. Covers qn=1..6 (H..I).
    Returns (npairs, 9, 9) for PM6 (with d-orbitals).

two_elec_two_center_int(const, idxi, idxj, ni, nj, xij, rij, Z, ...)
    Per-pair two-electron two-center integrals. Returns the rotated
    (npairs, 45, 45) w tensor plus core-electron e1b/e2a (4x4 sp blocks)
    and electron-electron rho terms.

qn_int, qnD_int : ndarray
    Period of the periodic table for valence sp and d shells, indexed by Z.
"""
from .diat_overlapD_np import diatom_overlap_matrixD
from .two_elec_two_center_int_np import two_elec_two_center_int
from .constants_np import qn_int, qnD_int, Constants

__all__ = [
    "diatom_overlap_matrixD",
    "two_elec_two_center_int",
    "qn_int",
    "qnD_int",
    "Constants",
]
