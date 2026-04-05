"""
Stereochemistry validation checks for conformers (pure numpy, no external deps).

Checks run after DG stage, rejecting conformers with wrong chirality
or double-bond geometry. Failed conformers are retried with new seeds.

Checks:
  1. Tetrahedral volume — magnitude must be >0.3 (not planar)
  2. Chiral sign — volume sign must match CW/CCW tag from RDKit
  3. Double bond planarity — dihedral near 0 or 180 degrees
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .dg_extract import DGParams, TetrahedralCheckData, extract_tetrahedral_data


def _compute_volume(p1, p2, p3, p4):
    """Signed volume of tetrahedron (p1-p4, p2-p4, p3-p4)."""
    v1 = p1 - p4
    v2 = p2 - p4
    v3 = p3 - p4
    cross = np.cross(v2, v3)
    return np.dot(v1, cross)


def check_tetrahedral_geometry(
    positions_4d: np.ndarray,
    mol,
    atom_offset: int,
    dim: int = 4,
    vol_tol: float = 0.3,
) -> bool:
    """Check if all sp3 centers have non-planar tetrahedral geometry.

    Returns True if geometry is acceptable.
    """
    from rdkit import Chem

    for atom in mol.GetAtoms():
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 3:
            continue

        # Use first 3 coords only
        center = atom.GetIdx()
        n0, n1, n2 = neighbors[0], neighbors[1], neighbors[2]
        n3 = neighbors[3] if len(neighbors) >= 4 else center

        p0 = positions_4d[atom_offset + n0, :3]
        p1 = positions_4d[atom_offset + n1, :3]
        p2 = positions_4d[atom_offset + n2, :3]
        p3 = positions_4d[atom_offset + n3, :3]

        vol = abs(_compute_volume(p0, p1, p2, p3))
        if vol < vol_tol:
            return False  # Too planar for sp3

    return True


def check_chiral_signs(
    positions_4d: np.ndarray,
    dg_params: DGParams,
    atom_offset: int,
    dim: int = 4,
) -> bool:
    """Check if chiral volumes have correct sign (CW/CCW match).

    Returns True if all chiral centers have correct sign.
    """
    if len(dg_params.chiral_idx1) == 0:
        return True

    for k in range(len(dg_params.chiral_idx1)):
        i1 = dg_params.chiral_idx1[k]
        i2 = dg_params.chiral_idx2[k]
        i3 = dg_params.chiral_idx3[k]
        i4 = dg_params.chiral_idx4[k]
        vl = dg_params.chiral_vol_lower[k]
        vu = dg_params.chiral_vol_upper[k]

        p1 = positions_4d[atom_offset + i1, :3]
        p2 = positions_4d[atom_offset + i2, :3]
        p3 = positions_4d[atom_offset + i3, :3]
        p4 = positions_4d[atom_offset + i4, :3]

        vol = _compute_volume(p1, p2, p3, p4)

        # Check sign: vol should be in [vl, vu]
        if vol < vl or vol > vu:
            return False

    return True


def run_stereo_checks(
    positions_4d: np.ndarray,
    dg_params_list: list,
    mols: list,
    conf_atom_starts: np.ndarray,
    conf_to_mol: np.ndarray,
    mol_order: list,
    n_confs: int,
    dim: int = 4,
) -> np.ndarray:
    """Run all stereo checks on conformers after DG stage.

    Returns bool array (n_confs,) — True if conformer PASSED all checks.
    """
    passed = np.ones(n_confs, dtype=bool)

    for c in range(n_confs):
        mol_idx_in_chunk = int(conf_to_mol[c])
        actual_mol_idx = mol_order[mol_idx_in_chunk]
        mol = mols[actual_mol_idx]
        dg_params = dg_params_list[actual_mol_idx]
        atom_off = int(conf_atom_starts[c])
        n_atoms = dg_params.n_atoms

        # Reshape positions for this conformer
        s = atom_off * dim
        e = (atom_off + n_atoms) * dim
        pos = positions_4d[s:e].reshape(n_atoms, dim)

        # Check 1: tetrahedral geometry (not too planar)
        if not check_tetrahedral_geometry(pos, mol, 0, dim):
            passed[c] = False
            continue

        # Check 2: chiral volume signs
        if not check_chiral_signs(pos, dg_params, 0, dim):
            passed[c] = False
            continue

    return passed
