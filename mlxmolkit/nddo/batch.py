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
from .two_center_d import two_center_w_10x10
from .scf import _pair_resonance_block, _pair_core_attraction
from .integrals import compute_nuclear_repulsion, nuclear_repulsion_for_method


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
    molecular_charges: np.ndarray  # (N,) float64

    # Padded matrices (N, MB, MB) — MB = max_basis
    H_core: np.ndarray         # (N, MB, MB) core Hamiltonian
    # Two-center w tensor (N, MA, MA, MO, MO, MO, MO) flattened last 4 dims,
    # where MO is the largest per-atom orbital count in the batch: 4 for an
    # sp basis, 9 once any atom carries d orbitals. Blocks for atoms with
    # fewer orbitals are zero-padded up to MO, so the stride is uniform.
    w: np.ndarray              # (N, MA, MA, MO**4)
    max_orb: int               # MO

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
    param_dict: dict = None,
    molecular_charges: list[float] | None = None,
    method: str = 'RM1',
) -> RM1Batch:
    """Pre-compute all integrals for a batch of molecules.

    Args:
        molecules: list of (atoms, coords) tuples
            atoms: list of atomic numbers
            coords: (n_atoms, 3) array in Angstrom
        param_dict: parameter dictionary (default: RM1_PARAMS)
        method: NDDO method name — selects the core-core repulsion form. Must
            match the method `param_dict` came from, or PM6/PM6_D energies
            come out several eV wrong.

    Returns:
        RM1Batch with all pre-computed data
    """
    if param_dict is None:
        param_dict = RM1_PARAMS
    normalized_molecules = []
    tuple_charges = []
    for molecule in molecules:
        if len(molecule) == 2:
            atoms, coords = molecule
            charge = 0.0
        elif len(molecule) == 3:
            atoms, coords, charge = molecule
        else:
            raise ValueError("molecules must contain (atoms, coords) or (atoms, coords, charge) tuples")
        normalized_molecules.append((atoms, coords))
        tuple_charges.append(float(charge))

    molecules = normalized_molecules
    if molecular_charges is None:
        molecular_charges_arr = np.asarray(tuple_charges, dtype=np.float64)
    else:
        if len(molecular_charges) != len(molecules):
            raise ValueError("molecular_charges must match the number of molecules")
        molecular_charges_arr = np.asarray(molecular_charges, dtype=np.float64)

    N = len(molecules)

    # Determine max sizes
    max_atoms = max(len(atoms) for atoms, _ in molecules)
    max_basis = 0
    for atoms, _ in molecules:
        nb = sum(param_dict[z].n_basis for z in atoms)
        max_basis = max(max_basis, nb)

    MB = max_basis
    MA = max_atoms
    # Widest single-atom basis in the batch. Hydrogen contributes 1, an sp
    # atom 4, an atom with d orbitals 9. The w tensor is sized MO**4 so a
    # 9-orbital atom fits; for an sp-only batch MO is 4 and the layout is
    # byte-for-byte what it has always been.
    MO = max(param_dict[z].n_basis for atoms, _ in molecules for z in atoms)

    # Allocate padded arrays
    n_atoms_arr = np.zeros(N, dtype=np.int32)
    n_basis_arr = np.zeros(N, dtype=np.int32)
    n_occ_arr = np.zeros(N, dtype=np.int32)

    H_core_all = np.zeros((N, MB, MB), dtype=np.float64)
    w_all = np.zeros((N, MA, MA, MO ** 4), dtype=np.float64)
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
        params = [param_dict[z] for z in atoms]
        n_bas = sum(p.n_basis for p in params)
        n_elec_float = float(sum(p.n_valence for p in params)) - float(molecular_charges_arr[mol_idx])
        n_elec = int(round(n_elec_float))
        if not np.isclose(n_elec_float, n_elec, atol=1.0e-6):
            raise ValueError(f"molecule {mol_idx} has non-integer electron count: {n_elec_float}")
        if n_elec < 0:
            raise ValueError(f"molecule {mol_idx} has negative electron count")
        if n_elec % 2 != 0:
            raise ValueError(
                f"molecule {mol_idx} is open shell ({n_elec} electrons); "
                "only closed-shell NDDO batches are currently supported"
            )
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

        # Diagonal: Uss/Upp/Udd
        for mu in range(n_bas):
            i = b2a[mu]
            p = params[i]
            if btype[mu] == 0:
                H[mu, mu] = p.Uss
            elif btype[mu] <= 3:
                H[mu, mu] = p.Upp
            else:
                H[mu, mu] = p.Udd

        starts = atom_basis_start

        # Off-diagonal resonance and electron-nuclear attraction. These reuse
        # the same per-pair functions the sequential solver uses, rather than
        # the sp-only copies that lived here: those assumed 4 orbitals per atom
        # and overran their own arrays on sulfur or a halogen.
        for i in range(n_at):
            for j in range(i + 1, n_at):
                block = _pair_resonance_block(params[i], params[j],
                                              coords[i], coords[j])
                si, sj = starts[i], starts[j]
                nA, nB = params[i].n_basis, params[j].n_basis
                H[si:si + nA, sj:sj + nB] = block
                H[sj:sj + nB, si:si + nA] = block.T

        for i in range(n_at):
            si, nA = starts[i], params[i].n_basis
            for j in range(n_at):
                if i == j:
                    continue
                H[si:si + nA, si:si + nA] += _pair_core_attraction(
                    params[i], params[j], coords[i], coords[j])

        # w tensor
        for i in range(n_at):
            for j in range(n_at):
                if i == j:
                    continue
                if params[i].n_basis == 9 or params[j].n_basis == 9:
                    w_ij = two_center_w_10x10(params[i], params[j],
                                              coords[i], coords[j])
                else:
                    w_ij, _, _ = rotate_integrals_to_molecular_frame(
                        params[i], params[j], coords[i], coords[j],
                    )

                # Store w tensor (only upper triangle i<j)
                if i < j:
                    # w_ij is (nA, nA, nB, nB); pad it into a uniform
                    # (MO, MO, MO, MO) block so every pair has the same stride
                    # regardless of how many orbitals each atom carries.
                    # rotate_integrals_to_molecular_frame already pads each
                    # centre to its shell width, so take the extents from the
                    # returned block rather than from n_basis — hydrogen comes
                    # back 4-wide, not 1-wide.
                    wa, wb = w_ij.shape[0], w_ij.shape[2]
                    blk = np.zeros((MO, MO, MO, MO), dtype=np.float64)
                    blk[:wa, :wa, :wb, :wb] = w_ij
                    w_all[mol_idx, i, j] = blk.ravel()
                    # Also store transpose for j>i access
                    blk_t = np.zeros((MO, MO, MO, MO), dtype=np.float64)
                    blk_t[:wb, :wb, :wa, :wa] = np.transpose(w_ij, (2, 3, 0, 1))
                    w_all[mol_idx, j, i] = blk_t.ravel()

        H_core_all[mol_idx, :n_bas, :n_bas] = H

        # Nuclear repulsion
        E_nuc_arr[mol_idx] = nuclear_repulsion_for_method(
            atoms, coords, param_dict, method)

    return RM1Batch(
        n_mols=N,
        max_atoms=MA,
        max_basis=MB,
        max_orb=MO,
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
        molecular_charges=molecular_charges_arr,
    )
