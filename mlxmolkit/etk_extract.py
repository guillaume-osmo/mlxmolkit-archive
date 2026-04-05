"""
Extract ETKDG torsion parameters from RDKit molecules.

Extracts:
  - CSD experimental torsion preferences (6-term Fourier)
  - Improper torsion terms (planarity at sp2 centers)
  - 1-4 distance constraints (from bounds matrix)

These parameters are used in stage 5 of the ETKDG pipeline, where 3D
coordinates are refined after 4D→3D collapse to match torsional
preferences from the Cambridge Structural Database (CSD).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom


@dataclass
class ETKParams:
    """ETK torsion parameters for a single molecule."""
    n_atoms: int

    # CSD torsion terms: E = Σ V_k * (1 + sign_k * cos(k * φ)) / 2
    torsion_idx: np.ndarray      # (n_torsions, 4) int32 — i,j,k,l atoms
    torsion_V: np.ndarray        # (n_torsions, 6) float32 — Fourier coefficients
    torsion_signs: np.ndarray    # (n_torsions, 6) int32 — sign multipliers

    # Improper torsions (planarity at sp2): E = w * (1 - cos(2ω))
    improper_idx: np.ndarray     # (n_improper, 4) int32 — center,n1,n2,n3
    improper_weight: np.ndarray  # (n_improper,) float32

    # 1-2 distance constraints (bonds): flat-bottom harmonic
    dist12_idx1: np.ndarray      # (n_dist12,) int32
    dist12_idx2: np.ndarray      # (n_dist12,) int32
    dist12_lb: np.ndarray        # (n_dist12,) float32
    dist12_ub: np.ndarray        # (n_dist12,) float32
    dist12_weight: np.ndarray    # (n_dist12,) float32

    # 1-3 distance constraints (angles): flat-bottom harmonic
    dist13_idx1: np.ndarray      # (n_dist13,) int32
    dist13_idx2: np.ndarray      # (n_dist13,) int32
    dist13_lb: np.ndarray        # (n_dist13,) float32
    dist13_ub: np.ndarray        # (n_dist13,) float32
    dist13_weight: np.ndarray    # (n_dist13,) float32

    # 1-4 distance constraints: E = w * (d - target)² if violated
    dist14_idx1: np.ndarray      # (n_dist14,) int32
    dist14_idx2: np.ndarray      # (n_dist14,) int32
    dist14_lb: np.ndarray        # (n_dist14,) float32 — lower bound distance
    dist14_ub: np.ndarray        # (n_dist14,) float32 — upper bound distance
    dist14_weight: np.ndarray    # (n_dist14,) float32


@dataclass
class BatchedETKSystem:
    """Batched ETK parameters for N molecules, ready for Metal kernel."""
    n_mols: int
    n_atoms_total: int

    atom_starts: np.ndarray  # (n_mols+1,) int32

    # CSD torsion terms (global atom indices)
    torsion_idx: np.ndarray      # (n_torsions_total, 4) int32
    torsion_V: np.ndarray        # (n_torsions_total, 6) float32
    torsion_signs: np.ndarray    # (n_torsions_total, 6) int32
    torsion_term_starts: np.ndarray  # (n_mols+1,) int32

    # Improper torsion terms (global atom indices)
    improper_idx: np.ndarray         # (n_improper_total, 4) int32
    improper_weight: np.ndarray      # (n_improper_total,) float32
    improper_term_starts: np.ndarray # (n_mols+1,) int32

    # 1-4 distance constraints (global atom indices)
    dist14_idx1: np.ndarray
    dist14_idx2: np.ndarray
    dist14_lb: np.ndarray
    dist14_ub: np.ndarray
    dist14_weight: np.ndarray
    dist14_term_starts: np.ndarray   # (n_mols+1,) int32


def extract_etk_params(
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    improper_weight: float = 10.0,
    dist14_weight: float = 1.0,
    *,
    use_exp_torsion: bool = True,
    use_basic_knowledge: bool = True,
    use_small_ring_torsions: bool = False,
    use_macrocycle_torsions: bool = True,
    et_version: int = 2,
    variant: str | None = None,
) -> ETKParams:
    """
    Extract ETKDG parameters from an RDKit molecule.

    Supports all ETKDG variants via the ``variant`` shortcut or individual flags:

    ========== ============ ================ ========== ==========
    variant    exp_torsion  basic_knowledge  small_ring et_version
    ========== ============ ================ ========== ==========
    DG         False        False            False      —
    KDG        False        True             False      2
    ETDG       True         False            False      2
    ETKDG      True         True             False      1
    ETKDGv2    True         True             False      2
    ETKDGv3    True         True             False      3
    srETKDGv3  True         True             True       3
    ========== ============ ================ ========== ==========
    """
    # Variant shortcut
    _VARIANTS = {
        "DG":        (False, False, False, 2),
        "KDG":       (False, True,  False, 2),
        "ETDG":      (True,  False, False, 2),
        "ETKDG":     (True,  True,  False, 1),
        "ETKDGv2":   (True,  True,  False, 2),
        "ETKDGv3":   (True,  True,  False, 3),
        "srETKDGv3": (True,  True,  True,  3),
    }
    if variant is not None:
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
        use_exp_torsion, use_basic_knowledge, use_small_ring_torsions, et_version = _VARIANTS[variant]

    n_atoms = mol.GetNumAtoms()

    # --- CSD experimental torsion preferences ---
    torsion_idx_list = []
    torsion_V_list = []
    torsion_signs_list = []

    if use_exp_torsion or use_basic_knowledge:
        try:
            kwargs = {}
            # Check if GetExperimentalTorsions supports these params
            try:
                exp_torsions = rdDistGeom.GetExperimentalTorsions(
                    mol,
                    useExpTorsionAnglePrefs=use_exp_torsion,
                    useSmallRingTorsions=use_small_ring_torsions,
                    useMacrocycleTorsions=use_macrocycle_torsions,
                    useBasicKnowledge=use_basic_knowledge,
                    ETversion=et_version,
                )
            except TypeError:
                # Older RDKit: no variant params
                exp_torsions = rdDistGeom.GetExperimentalTorsions(mol)

            for t in exp_torsions:
                atoms = list(t["atomIndices"])
                V = list(t["V"])
                signs = list(t["signs"])
                if len(atoms) == 4 and len(V) == 6 and len(signs) == 6:
                    torsion_idx_list.append(atoms)
                    torsion_V_list.append(V)
                    torsion_signs_list.append(signs)
        except Exception:
            pass
    n_torsions = len(torsion_idx_list)
    if n_torsions > 0:
        torsion_idx = np.array(torsion_idx_list, dtype=np.int32)
        torsion_V = np.array(torsion_V_list, dtype=np.float32)
        torsion_signs = np.array(torsion_signs_list, dtype=np.int32)
    else:
        torsion_idx = np.zeros((0, 4), dtype=np.int32)
        torsion_V = np.zeros((0, 6), dtype=np.float32)
        torsion_signs = np.zeros((0, 6), dtype=np.int32)

    # --- Improper torsions (planarity at sp2 centers) ---
    # Only included when use_basic_knowledge is True (ETKDG/KDG, not ETDG/DG)
    improper_idx_list = []
    improper_w_list = []

    if use_basic_knowledge:
        for atom in mol.GetAtoms():
            hyb = atom.GetHybridization()
            if hyb != Chem.HybridizationType.SP2:
                continue
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) < 3:
                continue

            center = atom.GetIdx()
            improper_idx_list.append([center, neighbors[0], neighbors[1], neighbors[2]])
            improper_w_list.append(improper_weight)

    n_improper = len(improper_idx_list)
    if n_improper > 0:
        imp_idx = np.array(improper_idx_list, dtype=np.int32)
        imp_w = np.array(improper_w_list, dtype=np.float32)
    else:
        imp_idx = np.zeros((0, 4), dtype=np.int32)
        imp_w = np.zeros(0, dtype=np.float32)

    # --- 1-2 distance constraints (bonds) ---
    # Enforce correct bond lengths in 3D. Uses bounds matrix midpoint ± tolerance.
    BOND_TOL = 0.01  # A
    BOND_FC = 100.0
    d12_i1, d12_i2, d12_lb, d12_ub, d12_w = [], [], [], [], []
    if use_basic_knowledge:
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            lo, hi = min(a, b), max(a, b)
            lb_val = bounds_mat[hi, lo]
            ub_val = bounds_mat[lo, hi]
            if lb_val > 0 and ub_val > 0:
                mid = (lb_val + ub_val) / 2.0
                d12_i1.append(a)
                d12_i2.append(b)
                d12_lb.append(mid - BOND_TOL)
                d12_ub.append(mid + BOND_TOL)
                d12_w.append(BOND_FC)

    n_d12 = len(d12_i1)

    # --- 1-3 distance constraints (angles) ---
    # Enforce correct angles via 1-3 distance bounds. Critical for geometry quality.
    ANGLE_FC = 100.0
    d13_i1, d13_i2, d13_lb, d13_ub, d13_w = [], [], [], [], []
    if use_basic_knowledge:
        for atom in mol.GetAtoms():
            center = atom.GetIdx()
            neighbors = sorted([n.GetIdx() for n in atom.GetNeighbors()])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, b = neighbors[i], neighbors[j]
                    lo, hi = min(a, b), max(a, b)
                    lb_val = bounds_mat[hi, lo]
                    ub_val = bounds_mat[lo, hi]
                    if lb_val > 0 and ub_val > 0:
                        mid = (lb_val + ub_val) / 2.0
                        d13_i1.append(a)
                        d13_i2.append(b)
                        d13_lb.append(mid - BOND_TOL)
                        d13_ub.append(mid + BOND_TOL)
                        d13_w.append(ANGLE_FC)

    n_d13 = len(d13_i1)

    # --- 1-4 distance constraints ---
    d14_i1, d14_i2, d14_lb, d14_ub, d14_w = [], [], [], [], []

    # Collect 1-4 pairs (atoms separated by exactly 3 bonds)
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        # From atom a, find neighbors of b that aren't a (1-3 from a)
        for n_b in mol.GetAtomWithIdx(b).GetNeighbors():
            c = n_b.GetIdx()
            if c == a:
                continue
            # From c, find neighbors that aren't b (1-4 from a)
            for n_c in mol.GetAtomWithIdx(c).GetNeighbors():
                d = n_c.GetIdx()
                if d == b or d == a:
                    continue
                if a < d:
                    ub = bounds_mat[a, d]
                    lb = bounds_mat[d, a]
                    if ub > 0 and lb > 0:
                        d14_i1.append(a)
                        d14_i2.append(d)
                        d14_lb.append(lb)
                        d14_ub.append(ub)
                        d14_w.append(dist14_weight)

    # Deduplicate
    seen = set()
    unique_i1, unique_i2, unique_lb, unique_ub, unique_w = [], [], [], [], []
    for i in range(len(d14_i1)):
        key = (d14_i1[i], d14_i2[i])
        if key not in seen:
            seen.add(key)
            unique_i1.append(d14_i1[i])
            unique_i2.append(d14_i2[i])
            unique_lb.append(d14_lb[i])
            unique_ub.append(d14_ub[i])
            unique_w.append(d14_w[i])

    n_d14 = len(unique_i1)

    def _a(lst, dt=np.int32):
        return np.array(lst, dtype=dt) if lst else np.zeros(0, dtype=dt)

    return ETKParams(
        n_atoms=n_atoms,
        torsion_idx=torsion_idx,
        torsion_V=torsion_V,
        torsion_signs=torsion_signs,
        improper_idx=imp_idx,
        improper_weight=imp_w,
        dist12_idx1=_a(d12_i1), dist12_idx2=_a(d12_i2),
        dist12_lb=_a(d12_lb, np.float32), dist12_ub=_a(d12_ub, np.float32),
        dist12_weight=_a(d12_w, np.float32),
        dist13_idx1=_a(d13_i1), dist13_idx2=_a(d13_i2),
        dist13_lb=_a(d13_lb, np.float32), dist13_ub=_a(d13_ub, np.float32),
        dist13_weight=_a(d13_w, np.float32),
        dist14_idx1=_a(unique_i1), dist14_idx2=_a(unique_i2),
        dist14_lb=_a(unique_lb, np.float32), dist14_ub=_a(unique_ub, np.float32),
        dist14_weight=_a(unique_w, np.float32),
    )


def batch_etk_params(
    params_list: list[ETKParams],
    atom_starts: np.ndarray,
) -> BatchedETKSystem:
    """Batch per-molecule ETK params into CSR arrays with global atom indices."""
    n_mols = len(params_list)
    n_atoms_total = int(atom_starts[-1])

    def _concat_or_empty(parts, dtype, shape_suffix=None):
        if parts:
            return np.concatenate(parts).astype(dtype)
        if shape_suffix:
            return np.zeros((0,) + shape_suffix, dtype=dtype)
        return np.zeros(0, dtype=dtype)

    # --- CSD torsion terms ---
    tor_idx_parts, tor_V_parts, tor_signs_parts = [], [], []
    torsion_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.torsion_idx)
        torsion_term_starts[i + 1] = torsion_term_starts[i] + n
        if n > 0:
            idx_shifted = p.torsion_idx.copy()
            idx_shifted += offset
            tor_idx_parts.append(idx_shifted)
            tor_V_parts.append(p.torsion_V)
            tor_signs_parts.append(p.torsion_signs)

    # --- Improper torsion terms ---
    imp_idx_parts, imp_w_parts = [], []
    improper_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.improper_idx)
        improper_term_starts[i + 1] = improper_term_starts[i] + n
        if n > 0:
            idx_shifted = p.improper_idx.copy()
            idx_shifted += offset
            imp_idx_parts.append(idx_shifted)
            imp_w_parts.append(p.improper_weight)

    # --- 1-4 distance constraints ---
    d14_i1_parts, d14_i2_parts = [], []
    d14_lb_parts, d14_ub_parts, d14_w_parts = [], [], []
    dist14_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.dist14_idx1)
        dist14_term_starts[i + 1] = dist14_term_starts[i] + n
        if n > 0:
            d14_i1_parts.append(p.dist14_idx1 + offset)
            d14_i2_parts.append(p.dist14_idx2 + offset)
            d14_lb_parts.append(p.dist14_lb)
            d14_ub_parts.append(p.dist14_ub)
            d14_w_parts.append(p.dist14_weight)

    return BatchedETKSystem(
        n_mols=n_mols,
        n_atoms_total=n_atoms_total,
        atom_starts=atom_starts,
        torsion_idx=_concat_or_empty(tor_idx_parts, np.int32, (4,)),
        torsion_V=_concat_or_empty(tor_V_parts, np.float32, (6,)),
        torsion_signs=_concat_or_empty(tor_signs_parts, np.int32, (6,)),
        torsion_term_starts=torsion_term_starts,
        improper_idx=_concat_or_empty(imp_idx_parts, np.int32, (4,)),
        improper_weight=_concat_or_empty(imp_w_parts, np.float32),
        improper_term_starts=improper_term_starts,
        dist14_idx1=_concat_or_empty(d14_i1_parts, np.int32),
        dist14_idx2=_concat_or_empty(d14_i2_parts, np.int32),
        dist14_lb=_concat_or_empty(d14_lb_parts, np.float32),
        dist14_ub=_concat_or_empty(d14_ub_parts, np.float32),
        dist14_weight=_concat_or_empty(d14_w_parts, np.float32),
        dist14_term_starts=dist14_term_starts,
    )
