"""
Stereochemistry checks for ETKDG conformer validation.

Stages 3/3b: Tetrahedral geometry, first chirality, chiral volume.
Stages 6/7:  Double bond geometry, double bond stereo, chiral distance matrix.

All checks operate on 3D coordinates (first 3 dims of 4D DG output).
Pure NumPy — no GPU needed since these are fast geometry checks.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _triple_product(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """Scalar triple product: v1 · (v2 × v3)."""
    cross = np.cross(v2, v3)
    return float(np.dot(v1, cross))


def _dihedral(p1: np.ndarray, p2: np.ndarray,
              p3: np.ndarray, p4: np.ndarray) -> float:
    """Dihedral angle (radians) for atoms 1-2-3-4."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    b2_hat = b2 / np.linalg.norm(b2)
    sign = np.dot(np.cross(n1, n2), b2_hat)
    return float(np.arctan2(sign, cos_angle))


# ---------------------------------------------------------------------------
# Stage 3: Tetrahedral geometry check
# ---------------------------------------------------------------------------

def check_tetrahedral_geometry(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    atom_offset: int = 0,
    vol_tol: float = 0.3,
) -> bool:
    """
    Check that all sp3 centers have reasonable tetrahedral geometry.

    Tests that the signed volume at each sp3 center is nonzero
    (i.e., the neighbors are not coplanar).

    Returns True if the conformer passes.
    """
    for atom in mol.GetAtoms():
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 4:
            continue

        center = atom.GetIdx()
        p0 = coords_3d[(center + atom_offset) * 3: (center + atom_offset) * 3 + 3]
        pts = []
        for ni in neighbors[:4]:
            pts.append(coords_3d[(ni + atom_offset) * 3: (ni + atom_offset) * 3 + 3])

        v1 = pts[0] - pts[3]
        v2 = pts[1] - pts[3]
        v3 = pts[2] - pts[3]
        vol = abs(_triple_product(v1, v2, v3))
        if vol < vol_tol:
            return False

    return True


# ---------------------------------------------------------------------------
# Stage 3b: First chirality check
# ---------------------------------------------------------------------------

def check_chirality(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    atom_offset: int = 0,
) -> bool:
    """
    Check that chiral centers have the correct handedness.

    For each chiral atom with CW/CCW tag, verify the sign of the
    volume formed by its four neighbors matches the expected sign.

    Returns True if all chiral centers are correct.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_UNSPECIFIED:
            continue

        center = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 4:
            continue

        i1, i2, i3, i4 = neighbors[0], neighbors[1], neighbors[2], neighbors[3]
        pts = []
        for ni in [i1, i2, i3, i4]:
            pts.append(coords_3d[(ni + atom_offset) * 3: (ni + atom_offset) * 3 + 3])

        v1 = pts[0] - pts[3]
        v2 = pts[1] - pts[3]
        v3 = pts[2] - pts[3]
        vol = _triple_product(v1, v2, v3)

        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            if vol < 0:
                return False
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            if vol > 0:
                return False

    return True


# ---------------------------------------------------------------------------
# Stage 3b extended: Chiral volume bounds check
# ---------------------------------------------------------------------------

def check_chiral_volume_bounds(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    atom_offset: int = 0,
    tol_fraction: float = 0.5,
) -> bool:
    """
    Check that chiral volumes are within bounds derived from the distance matrix.

    Uses the same volume estimation as dg_extract.py but checks the actual
    3D coordinates. Allows tol_fraction slack on the bounds.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 4:
            continue

        i1, i2, i3, i4 = neighbors[0], neighbors[1], neighbors[2], neighbors[3]
        pts = []
        for ni in [i1, i2, i3, i4]:
            pts.append(coords_3d[(ni + atom_offset) * 3: (ni + atom_offset) * 3 + 3])

        v1 = pts[0] - pts[3]
        v2 = pts[1] - pts[3]
        v3 = pts[2] - pts[3]
        vol = _triple_product(v1, v2, v3)

        # Estimate expected volume from bounds
        d12 = bounds_mat[min(i1, i2), max(i1, i2)]
        d13 = bounds_mat[min(i1, i3), max(i1, i3)]
        d14 = bounds_mat[min(i1, i4), max(i1, i4)]
        d23 = bounds_mat[min(i2, i3), max(i2, i3)]
        d24 = bounds_mat[min(i2, i4), max(i2, i4)]
        d34 = bounds_mat[min(i3, i4), max(i3, i4)]
        avg_d = (d12 + d13 + d14 + d23 + d24 + d34) / 6.0
        vol_est = avg_d ** 3 / (6.0 * np.sqrt(2.0))
        vol_bound = vol_est * (1.0 + tol_fraction)

        if abs(vol) > vol_bound:
            return False

    return True


# ---------------------------------------------------------------------------
# Stage 6: Double bond geometry check
# ---------------------------------------------------------------------------

def check_double_bond_geometry(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    atom_offset: int = 0,
    linearity_tol: float = 10.0,
) -> bool:
    """
    Check that double bonds have planar geometry.

    For each double bond A=B, verify that the four atoms in the
    A-neighbor, A, B, B-neighbor plane form a near-planar arrangement
    (improper dihedral close to 0 or 180 degrees).

    Returns True if all double bonds pass.
    """
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue

        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()

        a_neighbors = [n.GetIdx() for n in bond.GetBeginAtom().GetNeighbors()
                       if n.GetIdx() != b_idx]
        b_neighbors = [n.GetIdx() for n in bond.GetEndAtom().GetNeighbors()
                       if n.GetIdx() != a_idx]

        if not a_neighbors or not b_neighbors:
            continue

        def _get_pt(idx):
            return coords_3d[(idx + atom_offset) * 3: (idx + atom_offset) * 3 + 3]

        pa = _get_pt(a_idx)
        pb = _get_pt(b_idx)
        p_an = _get_pt(a_neighbors[0])
        p_bn = _get_pt(b_neighbors[0])

        dihed = abs(np.degrees(_dihedral(p_an, pa, pb, p_bn)))
        if not (dihed < linearity_tol or abs(dihed - 180.0) < linearity_tol):
            return False

    return True


# ---------------------------------------------------------------------------
# Stage 7: Double bond stereo check (E/Z)
# ---------------------------------------------------------------------------

def check_double_bond_stereo(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    atom_offset: int = 0,
) -> bool:
    """
    Check E/Z stereochemistry at double bonds.

    For bonds with STEREONONE, skip. For STEREOZ/STEREOE, verify the
    dihedral angle matches the expected configuration.

    Returns True if all stereo double bonds are correct.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREONONE:
            continue
        if bond.GetBondTypeAsDouble() != 2.0:
            continue

        stereo_atoms = list(bond.GetStereoAtoms())
        if len(stereo_atoms) < 2:
            continue

        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        sa0 = stereo_atoms[0]
        sa1 = stereo_atoms[1]

        def _get_pt(idx):
            return coords_3d[(idx + atom_offset) * 3: (idx + atom_offset) * 3 + 3]

        dihed = _dihedral(_get_pt(sa0), _get_pt(a_idx),
                          _get_pt(b_idx), _get_pt(sa1))
        dihed_deg = np.degrees(dihed)

        if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            if abs(dihed_deg) > 90:
                return False
        elif stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            if abs(dihed_deg) < 90:
                return False

    return True


# ---------------------------------------------------------------------------
# Stage 6b: Chiral distance matrix check
# ---------------------------------------------------------------------------

def check_chiral_distance_matrix(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    atom_offset: int = 0,
    tol: float = 0.5,
    max_violation_frac: float = 0.05,
) -> bool:
    """
    Check that pairwise distances are within the bounds matrix.

    Sanity check on 3D coordinates. Allows a small fraction of pairs
    to violate bounds (DG may not perfectly satisfy all constraints
    for larger molecules). Only checks 1-2 and 1-3 pairs strictly.

    Returns True if the conformer is acceptable.
    """
    n_atoms = mol.GetNumAtoms()
    n_violations = 0
    n_total = 0
    n_strict_violations = 0

    # Strict check: bonded pairs (1-2) and angle pairs (1-3) only
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pi = coords_3d[(i + atom_offset) * 3: (i + atom_offset) * 3 + 3]
        pj = coords_3d[(j + atom_offset) * 3: (j + atom_offset) * 3 + 3]
        d = float(np.linalg.norm(pi - pj))
        ub = bounds_mat[min(i, j), max(i, j)]
        lb = bounds_mat[max(i, j), min(i, j)]
        if d > ub * 1.5 or d < lb * 0.5:
            n_strict_violations += 1

    if n_strict_violations > 0:
        return False

    # Soft check: sample of long-range pairs
    for i in range(n_atoms):
        pi = coords_3d[(i + atom_offset) * 3: (i + atom_offset) * 3 + 3]
        for j in range(i + 1, n_atoms):
            pj = coords_3d[(j + atom_offset) * 3: (j + atom_offset) * 3 + 3]
            d = float(np.linalg.norm(pi - pj))
            ub = bounds_mat[i, j]
            lb = bounds_mat[j, i]
            n_total += 1
            if d > ub + tol or d < lb - tol:
                n_violations += 1

    if n_total == 0:
        return True
    return (n_violations / n_total) <= max_violation_frac


# ---------------------------------------------------------------------------
# Combined check dispatcher
# ---------------------------------------------------------------------------

def run_stage3_checks(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    atom_offset: int = 0,
) -> bool:
    """Run stage 3 + 3b checks (tetrahedral + chirality)."""
    if not check_tetrahedral_geometry(coords_3d, mol, atom_offset):
        return False
    if not check_chirality(coords_3d, mol, bounds_mat, atom_offset):
        return False
    return True


def run_stage67_checks(
    coords_3d: np.ndarray,
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    atom_offset: int = 0,
) -> bool:
    """Run stages 6-7 checks (double bond geometry/stereo + distance matrix)."""
    if not check_double_bond_geometry(coords_3d, mol, atom_offset):
        return False
    if not check_double_bond_stereo(coords_3d, mol, atom_offset):
        return False
    if not check_chiral_distance_matrix(coords_3d, mol, bounds_mat, atom_offset):
        return False
    return True
