"""
Extract MMFF94 force field parameters from RDKit into numpy arrays.

This is the Python equivalent of nvMolKit's mmff_flattened_builder.cpp.
Parameters are extracted ONCE and stored as flat numpy arrays that can be
sent to Metal kernels. No RDKit callback is needed during optimization.

MMFF94 energy terms (Halgren, J. Comput. Chem. 1996, 17, 490-519):
  1. Bond stretch    2. Angle bend    3. Stretch-bend coupling
  4. Out-of-plane    5. Torsion       6. Van der Waals
  7. Electrostatic

Reference: https://www.charmm-gui.org/charmmdoc/mmff.html
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers


RELATION_1_2 = 0
RELATION_1_3 = 1
RELATION_1_4 = 2
RELATION_1_X = 3


@dataclass
class MMFFParams:
    """All MMFF94 parameters for a molecule, stored as flat numpy arrays."""

    n_atoms: int

    bond_idx1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    bond_idx2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    bond_kb: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    bond_r0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    angle_idx1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    angle_idx2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    angle_idx3: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    angle_ka: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    angle_theta0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    angle_is_linear: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    strbend_idx1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    strbend_idx2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    strbend_idx3: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    strbend_kba_ijk: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    strbend_kba_kji: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    strbend_r0_ij: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    strbend_r0_kj: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    strbend_theta0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    oop_idx1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    oop_idx2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    oop_idx3: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    oop_idx4: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    oop_koop: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    torsion_idx1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    torsion_idx2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    torsion_idx3: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    torsion_idx4: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    torsion_V1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    torsion_V2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    torsion_V3: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    vdw_idx1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    vdw_idx2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    vdw_R_star: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    vdw_eps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    ele_idx1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    ele_idx2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    ele_charge_term: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    ele_diel_model: int = 1
    ele_is_1_4: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


def _build_neighbor_matrix(mol: Chem.Mol) -> np.ndarray:
    """Build neighbor relationship matrix using RDKit's shortest path distances."""
    n = mol.GetNumAtoms()
    dm = Chem.GetDistanceMatrix(mol)
    rel = np.full((n, n), RELATION_1_X, dtype=np.int8)
    for i in range(n):
        for j in range(n):
            d = int(dm[i, j])
            if d == 1:
                rel[i, j] = RELATION_1_2
            elif d == 2:
                rel[i, j] = RELATION_1_3
            elif d == 3:
                rel[i, j] = RELATION_1_4
    return rel


def extract_mmff_params(
    mol: Chem.Mol,
    conf_id: int = 0,
    non_bonded_thresh: float = 100.0,
    mmff_variant: str = "MMFF94",
) -> MMFFParams:
    """
    Extract MMFF94 or MMFF94s parameters from an RDKit molecule.

    This mirrors nvMolKit's constructForcefieldContribs(). Parameters are
    extracted ONCE and stored as flat arrays for GPU computation.

    Args:
        mol: RDKit molecule with at least one conformer.
        conf_id: conformer ID for distance-based VdW/ele filtering.
        non_bonded_thresh: distance threshold for non-bonded terms.
        mmff_variant: 'MMFF94' (default) or 'MMFF94s' (softer torsion
            barriers for conjugated/aromatic systems).

    Returns:
        MMFFParams with all force field parameters.
    """
    mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
    if mp is None:
        raise ValueError("Could not compute MMFF properties for molecule")

    n_atoms = mol.GetNumAtoms()
    params = MMFFParams(n_atoms=n_atoms)

    # --- 1. Bond stretch ---
    b_idx1, b_idx2, b_kb, b_r0 = [], [], [], []
    bond_r0_map = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        res = mp.GetMMFFBondStretchParams(mol, i, j)
        if res:
            _, kb, r0 = res
            b_idx1.append(i)
            b_idx2.append(j)
            b_kb.append(kb)
            b_r0.append(r0)
            bond_r0_map[(i, j)] = r0
            bond_r0_map[(j, i)] = r0
    params.bond_idx1 = np.array(b_idx1, dtype=np.int32)
    params.bond_idx2 = np.array(b_idx2, dtype=np.int32)
    params.bond_kb = np.array(b_kb, dtype=np.float32)
    params.bond_r0 = np.array(b_r0, dtype=np.float32)

    # --- 2. Angle bend ---
    a_idx1, a_idx2, a_idx3, a_ka, a_theta0, a_linear = [], [], [], [], [], []
    for center in range(n_atoms):
        atom = mol.GetAtomWithIdx(center)
        if atom.GetDegree() < 2:
            continue
        nbrs = [x.GetIdx() for x in atom.GetNeighbors()]
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                res = mp.GetMMFFAngleBendParams(mol, nbrs[ii], center, nbrs[jj])
                if res:
                    _, ka, theta0 = res
                    a_idx1.append(nbrs[ii])
                    a_idx2.append(center)
                    a_idx3.append(nbrs[jj])
                    a_ka.append(ka)
                    a_theta0.append(theta0)
                    at_type = mp.GetMMFFAtomType(center)
                    is_lin = 1 if atom.GetHybridization() == Chem.HybridizationType.SP else 0
                    a_linear.append(is_lin)
    params.angle_idx1 = np.array(a_idx1, dtype=np.int32)
    params.angle_idx2 = np.array(a_idx2, dtype=np.int32)
    params.angle_idx3 = np.array(a_idx3, dtype=np.int32)
    params.angle_ka = np.array(a_ka, dtype=np.float32)
    params.angle_theta0 = np.array(a_theta0, dtype=np.float32)
    params.angle_is_linear = np.array(a_linear, dtype=np.int32)

    # --- 3. Stretch-bend ---
    sb_idx1, sb_idx2, sb_idx3 = [], [], []
    sb_kba_ijk, sb_kba_kji, sb_r0_ij, sb_r0_kj, sb_theta0 = [], [], [], [], []
    for center in range(n_atoms):
        atom = mol.GetAtomWithIdx(center)
        if atom.GetDegree() < 2:
            continue
        if atom.GetHybridization() == Chem.HybridizationType.SP:
            continue
        nbrs = [x.GetIdx() for x in atom.GetNeighbors()]
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                i, j = nbrs[ii], nbrs[jj]
                res_sb = mp.GetMMFFStretchBendParams(mol, i, center, j)
                res_ang = mp.GetMMFFAngleBendParams(mol, i, center, j)
                if res_sb and res_ang:
                    _, kba_ijk, kba_kji = res_sb
                    _, _, theta0 = res_ang
                    r0_ij = bond_r0_map.get((i, center), 1.5)
                    r0_kj = bond_r0_map.get((j, center), 1.5)
                    sb_idx1.append(i)
                    sb_idx2.append(center)
                    sb_idx3.append(j)
                    sb_kba_ijk.append(kba_ijk)
                    sb_kba_kji.append(kba_kji)
                    sb_r0_ij.append(r0_ij)
                    sb_r0_kj.append(r0_kj)
                    sb_theta0.append(theta0)
    params.strbend_idx1 = np.array(sb_idx1, dtype=np.int32)
    params.strbend_idx2 = np.array(sb_idx2, dtype=np.int32)
    params.strbend_idx3 = np.array(sb_idx3, dtype=np.int32)
    params.strbend_kba_ijk = np.array(sb_kba_ijk, dtype=np.float32)
    params.strbend_kba_kji = np.array(sb_kba_kji, dtype=np.float32)
    params.strbend_r0_ij = np.array(sb_r0_ij, dtype=np.float32)
    params.strbend_r0_kj = np.array(sb_r0_kj, dtype=np.float32)
    params.strbend_theta0 = np.array(sb_theta0, dtype=np.float32)

    # --- 4. Out-of-plane bending ---
    o_idx1, o_idx2, o_idx3, o_idx4, o_koop = [], [], [], [], []
    for center in range(n_atoms):
        atom = mol.GetAtomWithIdx(center)
        if atom.GetDegree() != 3:
            continue
        nbrs = [x.GetIdx() for x in atom.GetNeighbors()]
        res = mp.GetMMFFOopBendParams(mol, nbrs[0], center, nbrs[1], nbrs[2])
        if res is None:
            continue
        koop = res
        permutations = [
            (nbrs[0], center, nbrs[1], nbrs[2]),
            (nbrs[0], center, nbrs[2], nbrs[1]),
            (nbrs[1], center, nbrs[2], nbrs[0]),
        ]
        for i1, i2, i3, i4 in permutations:
            o_idx1.append(i1)
            o_idx2.append(i2)
            o_idx3.append(i3)
            o_idx4.append(i4)
            o_koop.append(koop)
    params.oop_idx1 = np.array(o_idx1, dtype=np.int32)
    params.oop_idx2 = np.array(o_idx2, dtype=np.int32)
    params.oop_idx3 = np.array(o_idx3, dtype=np.int32)
    params.oop_idx4 = np.array(o_idx4, dtype=np.int32)
    params.oop_koop = np.array(o_koop, dtype=np.float32)

    # --- 5. Torsion ---
    t_idx1, t_idx2, t_idx3, t_idx4, t_V1, t_V2, t_V3 = [], [], [], [], [], [], []
    for bond in mol.GetBonds():
        i2, i3 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a2 = mol.GetAtomWithIdx(i2)
        a3 = mol.GetAtomWithIdx(i3)
        for n1 in a2.GetNeighbors():
            i1 = n1.GetIdx()
            if i1 == i3:
                continue
            for n4 in a3.GetNeighbors():
                i4 = n4.GetIdx()
                if i4 == i2 or i4 == i1:
                    continue
                res = mp.GetMMFFTorsionParams(mol, i1, i2, i3, i4)
                if res:
                    _, V1, V2, V3 = res
                    t_idx1.append(i1)
                    t_idx2.append(i2)
                    t_idx3.append(i3)
                    t_idx4.append(i4)
                    t_V1.append(V1)
                    t_V2.append(V2)
                    t_V3.append(V3)
    params.torsion_idx1 = np.array(t_idx1, dtype=np.int32)
    params.torsion_idx2 = np.array(t_idx2, dtype=np.int32)
    params.torsion_idx3 = np.array(t_idx3, dtype=np.int32)
    params.torsion_idx4 = np.array(t_idx4, dtype=np.int32)
    params.torsion_V1 = np.array(t_V1, dtype=np.float32)
    params.torsion_V2 = np.array(t_V2, dtype=np.float32)
    params.torsion_V3 = np.array(t_V3, dtype=np.float32)

    # --- 6 & 7. Non-bonded (VdW + Electrostatic) ---
    rel = _build_neighbor_matrix(mol)
    conf = mol.GetConformer(conf_id)

    v_idx1, v_idx2, v_R_star, v_eps = [], [], [], []
    e_idx1, e_idx2, e_charge, e_is14 = [], [], [], []

    charges = np.array(
        [mp.GetMMFFPartialCharge(i) for i in range(n_atoms)],
        dtype=np.float64,
    )

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if rel[i, j] < RELATION_1_4:
                continue
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            dist = pi.Distance(pj)
            if dist > non_bonded_thresh:
                continue

            vdw = mp.GetMMFFVdWParams(i, j)
            if vdw:
                v_idx1.append(i)
                v_idx2.append(j)
                v_R_star.append(vdw[0])
                v_eps.append(vdw[1])

            qi, qj = charges[i], charges[j]
            if abs(qi) > 1e-10 and abs(qj) > 1e-10:
                charge_term = float(qi * qj)
                e_idx1.append(i)
                e_idx2.append(j)
                e_charge.append(charge_term)
                e_is14.append(1 if rel[i, j] == RELATION_1_4 else 0)

    params.vdw_idx1 = np.array(v_idx1, dtype=np.int32)
    params.vdw_idx2 = np.array(v_idx2, dtype=np.int32)
    params.vdw_R_star = np.array(v_R_star, dtype=np.float32)
    params.vdw_eps = np.array(v_eps, dtype=np.float32)

    params.ele_idx1 = np.array(e_idx1, dtype=np.int32)
    params.ele_idx2 = np.array(e_idx2, dtype=np.int32)
    params.ele_charge_term = np.array(e_charge, dtype=np.float32)
    params.ele_is_1_4 = np.array(e_is14, dtype=np.int32)
    params.ele_diel_model = 0

    return params
