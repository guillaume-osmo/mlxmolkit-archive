"""
Extract distance geometry parameters from RDKit molecules for Metal ETKDG.

Converts RDKit bounds matrices into flat CSR-style arrays suitable for
batched Metal kernel dispatch. Handles distance bounds, chiral volume
constraints, and fourth-dimension penalty terms.

Reference: nvMolKit dist_geom_kernels_device.cuh
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom


@dataclass
class DGParams:
    """Distance geometry parameters for a single molecule."""
    n_atoms: int

    # Distance violation terms: E = w*(d²/ub² - 1)² or w*(2lb²/(lb²+d²) - 1)²
    dist_idx1: np.ndarray    # (n_dist,) int32 — first atom
    dist_idx2: np.ndarray    # (n_dist,) int32 — second atom
    dist_lb2: np.ndarray     # (n_dist,) float32 — squared lower bound
    dist_ub2: np.ndarray     # (n_dist,) float32 — squared upper bound
    dist_weight: np.ndarray  # (n_dist,) float32 — per-pair weight

    # Chiral volume terms: E = w*(vol - bound)² if outside [lower, upper]
    chiral_idx1: np.ndarray     # (n_chiral,) int32
    chiral_idx2: np.ndarray     # (n_chiral,) int32
    chiral_idx3: np.ndarray     # (n_chiral,) int32
    chiral_idx4: np.ndarray     # (n_chiral,) int32
    chiral_vol_lower: np.ndarray  # (n_chiral,) float32
    chiral_vol_upper: np.ndarray  # (n_chiral,) float32

    # Fourth dimension atoms (all atoms get 4th-dim penalty)
    fourth_idx: np.ndarray  # (n_atoms,) int32


@dataclass
class BatchedDGSystem:
    """Batched DG parameters for N molecules, ready for Metal kernel."""
    n_mols: int
    dim: int
    n_atoms_total: int

    atom_starts: np.ndarray  # (n_mols+1,) int32 — CSR atom offsets

    # Distance terms (global indices)
    dist_idx1: np.ndarray
    dist_idx2: np.ndarray
    dist_lb2: np.ndarray
    dist_ub2: np.ndarray
    dist_weight: np.ndarray
    dist_term_starts: np.ndarray   # (n_mols+1,) int32
    dist_mol_indices: np.ndarray   # (n_dist_total,) int32

    # Chiral terms (global indices)
    chiral_idx1: np.ndarray
    chiral_idx2: np.ndarray
    chiral_idx3: np.ndarray
    chiral_idx4: np.ndarray
    chiral_vol_lower: np.ndarray
    chiral_vol_upper: np.ndarray
    chiral_term_starts: np.ndarray  # (n_mols+1,) int32
    chiral_mol_indices: np.ndarray

    # Fourth dimension (global indices)
    fourth_idx: np.ndarray
    fourth_term_starts: np.ndarray  # (n_mols+1,) int32
    fourth_mol_indices: np.ndarray


@dataclass
class TetrahedralCheckData:
    """Data for tetrahedral geometry checks."""
    idx0: np.ndarray  # center atom (global)
    idx1: np.ndarray
    idx2: np.ndarray
    idx3: np.ndarray
    idx4: np.ndarray  # 4th neighbor or idx0 if only 3 neighbors
    in_fused_small_ring: np.ndarray  # bool
    mol_indices: np.ndarray  # int32


def get_bounds_matrix(mol: Chem.Mol) -> np.ndarray:
    """Get distance bounds matrix from RDKit."""
    bmat = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    return bmat


def extract_dg_params(
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    dim: int = 4,
    basin_size_tol: float = 1e8,
) -> DGParams:
    """Extract DG force field parameters from a molecule and its bounds matrix.

    Converts the upper-triangular bounds matrix into flat arrays of
    (idx1, idx2, lb², ub², weight) for the distance violation energy,
    plus chiral volume terms and fourth-dimension penalty indices.
    """
    n_atoms = mol.GetNumAtoms()

    # --- Distance terms from bounds matrix ---
    # bounds_mat[i,j] (i < j) = upper bound, bounds_mat[j,i] (j > i) = lower bound
    idx1_list, idx2_list = [], []
    lb2_list, ub2_list = [], []
    weight_list = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            ub = float(bounds_mat[i, j])
            lb = float(bounds_mat[j, i])
            if ub <= 0 or lb <= 0:
                continue
            # Weight based on basin size tolerance (nvMolKit pattern)
            ub2 = ub * ub
            lb2 = lb * lb
            if ub2 - lb2 > basin_size_tol:
                continue
            w = 1.0 / max(ub2 - lb2, 1e-8)
            idx1_list.append(i)
            idx2_list.append(j)
            lb2_list.append(lb2)
            ub2_list.append(ub2)
            weight_list.append(w)

    n_dist = len(idx1_list)
    dist_idx1 = np.array(idx1_list, dtype=np.int32) if n_dist > 0 else np.zeros(0, dtype=np.int32)
    dist_idx2 = np.array(idx2_list, dtype=np.int32) if n_dist > 0 else np.zeros(0, dtype=np.int32)
    dist_lb2 = np.array(lb2_list, dtype=np.float32) if n_dist > 0 else np.zeros(0, dtype=np.float32)
    dist_ub2 = np.array(ub2_list, dtype=np.float32) if n_dist > 0 else np.zeros(0, dtype=np.float32)
    dist_weight = np.array(weight_list, dtype=np.float32) if n_dist > 0 else np.zeros(0, dtype=np.float32)

    # --- Chiral volume terms ---
    chiral_idx1, chiral_idx2, chiral_idx3, chiral_idx4 = [], [], [], []
    chiral_vol_lower, chiral_vol_upper = [], []

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_UNSPECIFIED:
            continue

        center = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 4:
            continue

        # Compute volume bounds from distance bounds
        # V = (p1-p4) . ((p2-p4) x (p3-p4))
        i1, i2, i3, i4 = neighbors[0], neighbors[1], neighbors[2], neighbors[3]

        # Estimate volume from bounds (nvMolKit approach)
        d12 = bounds_mat[min(i1, i2), max(i1, i2)]
        d13 = bounds_mat[min(i1, i3), max(i1, i3)]
        d14 = bounds_mat[min(i1, i4), max(i1, i4)]
        d23 = bounds_mat[min(i2, i3), max(i2, i3)]
        d24 = bounds_mat[min(i2, i4), max(i2, i4)]
        d34 = bounds_mat[min(i3, i4), max(i3, i4)]

        avg_d = (d12 + d13 + d14 + d23 + d24 + d34) / 6.0
        vol_est = avg_d ** 3 / (6.0 * np.sqrt(2.0))

        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            vol_lower = 0.0
            vol_upper = vol_est
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            vol_lower = -vol_est
            vol_upper = 0.0
        else:
            vol_lower = -vol_est
            vol_upper = vol_est

        chiral_idx1.append(i1)
        chiral_idx2.append(i2)
        chiral_idx3.append(i3)
        chiral_idx4.append(i4)
        chiral_vol_lower.append(vol_lower)
        chiral_vol_upper.append(vol_upper)

    n_chiral = len(chiral_idx1)

    # --- Fourth dimension indices (all atoms) ---
    fourth_idx = np.arange(n_atoms, dtype=np.int32)

    return DGParams(
        n_atoms=n_atoms,
        dist_idx1=dist_idx1, dist_idx2=dist_idx2,
        dist_lb2=dist_lb2, dist_ub2=dist_ub2, dist_weight=dist_weight,
        chiral_idx1=np.array(chiral_idx1, dtype=np.int32) if n_chiral else np.zeros(0, dtype=np.int32),
        chiral_idx2=np.array(chiral_idx2, dtype=np.int32) if n_chiral else np.zeros(0, dtype=np.int32),
        chiral_idx3=np.array(chiral_idx3, dtype=np.int32) if n_chiral else np.zeros(0, dtype=np.int32),
        chiral_idx4=np.array(chiral_idx4, dtype=np.int32) if n_chiral else np.zeros(0, dtype=np.int32),
        chiral_vol_lower=np.array(chiral_vol_lower, dtype=np.float32) if n_chiral else np.zeros(0, dtype=np.float32),
        chiral_vol_upper=np.array(chiral_vol_upper, dtype=np.float32) if n_chiral else np.zeros(0, dtype=np.float32),
        fourth_idx=fourth_idx,
    )


def extract_tetrahedral_data(mol: Chem.Mol) -> tuple[list, list, list, list, list, list]:
    """Extract tetrahedral check atoms for a molecule.

    Returns lists of (center, n1, n2, n3, n4, in_fused_small_ring) per chiral center.
    n4 = center if only 3 neighbors.
    """
    centers, n1s, n2s, n3s, n4s, fused = [], [], [], [], [], []
    ring_info = mol.GetRingInfo()

    for atom in mol.GetAtoms():
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 3:
            continue

        idx = atom.GetIdx()
        in_fused = False
        if ring_info.NumAtomRings(idx) > 1:
            for size in [3, 4, 5]:
                if ring_info.IsAtomInRingOfSize(idx, size):
                    in_fused = True
                    break

        centers.append(idx)
        n1s.append(neighbors[0])
        n2s.append(neighbors[1])
        n3s.append(neighbors[2])
        n4s.append(neighbors[3] if len(neighbors) >= 4 else idx)
        fused.append(in_fused)

    return centers, n1s, n2s, n3s, n4s, fused


def batch_dg_params(
    params_list: list[DGParams],
    dim: int = 4,
) -> BatchedDGSystem:
    """Batch per-molecule DG params into CSR arrays with global atom indices."""
    n_mols = len(params_list)

    # Atom starts
    atom_starts = np.zeros(n_mols + 1, dtype=np.int32)
    for i, p in enumerate(params_list):
        atom_starts[i + 1] = atom_starts[i] + p.n_atoms
    n_atoms_total = int(atom_starts[-1])

    # Distance terms
    dist_parts_idx1, dist_parts_idx2 = [], []
    dist_parts_lb2, dist_parts_ub2, dist_parts_w = [], [], []
    dist_term_starts = np.zeros(n_mols + 1, dtype=np.int32)
    dist_mol_parts = []

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.dist_idx1)
        dist_term_starts[i + 1] = dist_term_starts[i] + n
        if n > 0:
            dist_parts_idx1.append(p.dist_idx1 + offset)
            dist_parts_idx2.append(p.dist_idx2 + offset)
            dist_parts_lb2.append(p.dist_lb2)
            dist_parts_ub2.append(p.dist_ub2)
            dist_parts_w.append(p.dist_weight)
            dist_mol_parts.append(np.full(n, i, dtype=np.int32))

    def _concat_or_empty(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    # Chiral terms
    ch_parts_1, ch_parts_2, ch_parts_3, ch_parts_4 = [], [], [], []
    ch_parts_lo, ch_parts_hi = [], []
    chiral_term_starts = np.zeros(n_mols + 1, dtype=np.int32)
    ch_mol_parts = []

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.chiral_idx1)
        chiral_term_starts[i + 1] = chiral_term_starts[i] + n
        if n > 0:
            ch_parts_1.append(p.chiral_idx1 + offset)
            ch_parts_2.append(p.chiral_idx2 + offset)
            ch_parts_3.append(p.chiral_idx3 + offset)
            ch_parts_4.append(p.chiral_idx4 + offset)
            ch_parts_lo.append(p.chiral_vol_lower)
            ch_parts_hi.append(p.chiral_vol_upper)
            ch_mol_parts.append(np.full(n, i, dtype=np.int32))

    # Fourth dim terms
    f_parts = []
    fourth_term_starts = np.zeros(n_mols + 1, dtype=np.int32)
    f_mol_parts = []

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.fourth_idx)
        fourth_term_starts[i + 1] = fourth_term_starts[i] + n
        if n > 0:
            f_parts.append(p.fourth_idx + offset)
            f_mol_parts.append(np.full(n, i, dtype=np.int32))

    return BatchedDGSystem(
        n_mols=n_mols,
        dim=dim,
        n_atoms_total=n_atoms_total,
        atom_starts=atom_starts,
        dist_idx1=_concat_or_empty(dist_parts_idx1, np.int32),
        dist_idx2=_concat_or_empty(dist_parts_idx2, np.int32),
        dist_lb2=_concat_or_empty(dist_parts_lb2, np.float32),
        dist_ub2=_concat_or_empty(dist_parts_ub2, np.float32),
        dist_weight=_concat_or_empty(dist_parts_w, np.float32),
        dist_term_starts=dist_term_starts,
        dist_mol_indices=_concat_or_empty(dist_mol_parts, np.int32),
        chiral_idx1=_concat_or_empty(ch_parts_1, np.int32),
        chiral_idx2=_concat_or_empty(ch_parts_2, np.int32),
        chiral_idx3=_concat_or_empty(ch_parts_3, np.int32),
        chiral_idx4=_concat_or_empty(ch_parts_4, np.int32),
        chiral_vol_lower=_concat_or_empty(ch_parts_lo, np.float32),
        chiral_vol_upper=_concat_or_empty(ch_parts_hi, np.float32),
        chiral_term_starts=chiral_term_starts,
        chiral_mol_indices=_concat_or_empty(ch_mol_parts, np.int32),
        fourth_idx=_concat_or_empty(f_parts, np.int32),
        fourth_term_starts=fourth_term_starts,
        fourth_mol_indices=_concat_or_empty(f_mol_parts, np.int32),
    )


def batch_tetrahedral_data(
    mols: list[Chem.Mol],
    atom_starts: np.ndarray,
) -> TetrahedralCheckData | None:
    """Batch tetrahedral check data for multiple molecules."""
    all_c, all_1, all_2, all_3, all_4, all_f, all_m = [], [], [], [], [], [], []

    for i, mol in enumerate(mols):
        centers, n1s, n2s, n3s, n4s, fused = extract_tetrahedral_data(mol)
        offset = int(atom_starts[i])
        for j in range(len(centers)):
            all_c.append(centers[j] + offset)
            all_1.append(n1s[j] + offset)
            all_2.append(n2s[j] + offset)
            all_3.append(n3s[j] + offset)
            all_4.append(n4s[j] + offset)
            all_f.append(fused[j])
            all_m.append(i)

    if not all_c:
        return None

    return TetrahedralCheckData(
        idx0=np.array(all_c, dtype=np.int32),
        idx1=np.array(all_1, dtype=np.int32),
        idx2=np.array(all_2, dtype=np.int32),
        idx3=np.array(all_3, dtype=np.int32),
        idx4=np.array(all_4, dtype=np.int32),
        in_fused_small_ring=np.array(all_f, dtype=bool),
        mol_indices=np.array(all_m, dtype=np.int32),
    )
