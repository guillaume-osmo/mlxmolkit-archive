"""
Batch processing for RM1 SCF: N molecules simultaneously on Metal GPU.

Pads all molecules to uniform max_atoms × max_basis dimensions
so a single Metal kernel dispatch handles all Fock matrices.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from .params import RM1_PARAMS, ElementParams, ANG_TO_BOHR, EV_TO_KCAL
from .overlap import overlap_molecular_frame
from .rotation import rotate_integrals_to_molecular_frame
from .integrals import compute_nuclear_repulsion


@dataclass
class RM1Batch:
    """Padded batch of N molecules for GPU SCF."""
    n_mols: int
    max_atoms: int
    max_basis: int

    # Per-mol sizes
    n_atoms_arr: np.ndarray    # (N,) int32
    n_basis_arr: np.ndarray    # (N,) int32
    n_occ_arr: np.ndarray      # (N,) int32
    atoms_list: list            # list of atom-number lists

    # Padded matrices (N, MB, MB) — MB = max_basis
    H_core: np.ndarray         # (N, MB, MB) core Hamiltonian
    # Two-center w tensor (N, MA, MA, 4, 4, 4, 4) flattened last 4 dims
    w: np.ndarray              # (N, MA, MA, 256)

    # Per-atom parameters for Fock kernel
    atom_params: np.ndarray    # (N, MA, 5) [gss,gsp,gpp,gp2,hsp]
    atom_map: np.ndarray       # (N, MB) int32: basis→atom
    type_map: np.ndarray       # (N, MB) int32: basis→orbital type
    atom_starts: np.ndarray    # (N, MA+1) int32: CSR offsets

    # Pre-computed nuclear repulsion energies
    E_nuc: np.ndarray          # (N,) float64

    # Coords for reference
    coords_list: list          # list of coords arrays


def prepare_batch(
    molecules: list[tuple[list[int], np.ndarray]],
) -> RM1Batch:
    """Pre-compute all integrals for a batch of molecules.

    Args:
        molecules: list of (atoms, coords) tuples
            atoms: list of atomic numbers
            coords: (n_atoms, 3) array in Angstrom

    Returns:
        RM1Batch with all pre-computed data
    """
    N = len(molecules)

    # Determine max sizes
    max_atoms = max(len(atoms) for atoms, _ in molecules)
    max_basis = 0
    for atoms, _ in molecules:
        nb = sum(RM1_PARAMS[z].n_basis for z in atoms)
        max_basis = max(max_basis, nb)

    MB = max_basis
    MA = max_atoms

    # Allocate padded arrays
    n_atoms_arr = np.zeros(N, dtype=np.int32)
    n_basis_arr = np.zeros(N, dtype=np.int32)
    n_occ_arr = np.zeros(N, dtype=np.int32)

    H_core_all = np.zeros((N, MB, MB), dtype=np.float64)
    w_all = np.zeros((N, MA, MA, 256), dtype=np.float64)
    atom_params_all = np.zeros((N, MA, 5), dtype=np.float64)
    atom_map_all = np.zeros((N, MB), dtype=np.int32)
    type_map_all = np.zeros((N, MB), dtype=np.int32)
    atom_starts_all = np.zeros((N, MA + 1), dtype=np.int32)
    E_nuc_arr = np.zeros(N, dtype=np.float64)

    atoms_list = []
    coords_list = []

    for mol_idx, (atoms, coords) in enumerate(molecules):
        coords = np.array(coords, dtype=np.float64)
        n_at = len(atoms)
        params = [RM1_PARAMS[z] for z in atoms]
        n_bas = sum(p.n_basis for p in params)
        n_elec = sum(p.n_valence for p in params)
        n_occ = n_elec // 2

        n_atoms_arr[mol_idx] = n_at
        n_basis_arr[mol_idx] = n_bas
        n_occ_arr[mol_idx] = n_occ
        atoms_list.append(atoms)
        coords_list.append(coords)

        # Build basis info
        basis_to_atom = []
        basis_type = []
        atom_basis_start = []
        for i, p in enumerate(params):
            atom_basis_start.append(len(basis_to_atom))
            for k in range(p.n_basis):
                basis_to_atom.append(i)
                basis_type.append(k)
        atom_basis_start.append(n_bas)

        b2a = np.array(basis_to_atom, dtype=np.int32)
        btype = np.array(basis_type, dtype=np.int32)

        atom_map_all[mol_idx, :n_bas] = b2a
        type_map_all[mol_idx, :n_bas] = btype
        for i in range(n_at + 1):
            atom_starts_all[mol_idx, i] = atom_basis_start[i]

        # Atom params
        for i, p in enumerate(params):
            atom_params_all[mol_idx, i] = [p.gss, p.gsp, p.gpp, p.gp2, p.hsp]

        # === Build H_core ===
        H = np.zeros((n_bas, n_bas), dtype=np.float64)

        # Diagonal: Uss/Upp
        for mu in range(n_bas):
            i = b2a[mu]
            p = params[i]
            H[mu, mu] = p.Uss if btype[mu] == 0 else p.Upp

        starts = atom_basis_start

        # Off-diagonal resonance + nuclear attraction
        for i in range(n_at):
            for j in range(i + 1, n_at):
                pA, pB = params[i], params[j]

                # Overlap for resonance
                S_ij = overlap_molecular_frame(pA, pB, coords[i], coords[j])
                for mu_off in range(pA.n_basis):
                    mu = starts[i] + mu_off
                    beta_mu = pA.beta_s if btype[mu] == 0 else pA.beta_p
                    for nu_off in range(pB.n_basis):
                        nu = starts[j] + nu_off
                        beta_nu = pB.beta_s if btype[nu] == 0 else pB.beta_p
                        H[mu, nu] = 0.5 * (beta_mu + beta_nu) * S_ij[mu_off, nu_off]
                        H[nu, mu] = H[mu, nu]

        # Nuclear attraction + w tensor
        for i in range(n_at):
            for j in range(n_at):
                if i == j:
                    continue
                w_ij, e1b_ij, e2a_ij = rotate_integrals_to_molecular_frame(
                    params[i], params[j], coords[i], coords[j],
                )
                # Nuclear attraction on atom i from nucleus j
                nA = params[i].n_basis
                for mu_a in range(nA):
                    for nu_a in range(nA):
                        H[starts[i] + mu_a, starts[i] + nu_a] += e1b_ij[mu_a, nu_a]

                # Store w tensor (only upper triangle i<j)
                if i < j:
                    w_all[mol_idx, i, j] = w_ij.flatten()
                    # Also store transpose for j>i access
                    w_t = np.transpose(w_ij, (2, 3, 0, 1))
                    w_all[mol_idx, j, i] = w_t.flatten()

        H_core_all[mol_idx, :n_bas, :n_bas] = H

        # Nuclear repulsion
        E_nuc_arr[mol_idx] = compute_nuclear_repulsion(atoms, coords)

    return RM1Batch(
        n_mols=N,
        max_atoms=MA,
        max_basis=MB,
        n_atoms_arr=n_atoms_arr,
        n_basis_arr=n_basis_arr,
        n_occ_arr=n_occ_arr,
        atoms_list=atoms_list,
        H_core=H_core_all,
        w=w_all,
        atom_params=atom_params_all,
        atom_map=atom_map_all,
        type_map=type_map_all,
        atom_starts=atom_starts_all,
        E_nuc=E_nuc_arr,
        coords_list=coords_list,
    )
