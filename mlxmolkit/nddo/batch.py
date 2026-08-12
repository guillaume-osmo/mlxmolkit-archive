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
from .packing import pack, packed_size, unpack
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
    # Two-centre integrals, lower-triangle packed per atom pair. `w` is a flat
    # buffer; `pair_offset[m, i, j]` is where the (i, j) block starts in it, or
    # -1 if there is none. A block is (packed(nA), packed(nB)) row-major, where
    # packed(n) = n(n+1)/2 — 1 for hydrogen, 10 for an sp atom, 45 with d.
    # Sizing each pair by its own two atoms rather than padding to the widest
    # atom in the batch is what keeps a PM6 batch containing sulfur in tens of
    # megabytes instead of gigabytes.
    w: np.ndarray              # (N, max_pair_storage) float64
    atom_w: np.ndarray         # (N, MA, 243) one-centre d integrals
    pair_offset: np.ndarray    # (N, MA, MA) int32
    atom_norb: np.ndarray      # (N, MA) int32 — orbitals per atom
    max_orb: int               # widest per-atom basis in the batch

    # Per-atom parameters for Fock kernel
    atom_params: np.ndarray    # (N, MA, 5) [gss,gsp,gpp,gp2,hsp]
    atom_map: np.ndarray       # (N, MB) int32: basis→atom
    type_map: np.ndarray       # (N, MB) int32: basis→orbital type
    atom_starts: np.ndarray    # (N, MA+1) int32: CSR offsets

    # Pre-computed nuclear repulsion energies
    E_nuc: np.ndarray          # (N,) float64

    # Coords for reference
    coords_list: list          # list of coords arrays




def _one_centre_w(p) -> np.ndarray:
    """The 243 one-centre two-electron integrals for a d-bearing atom.

    A function of the atom's Slater exponents alone, so it is constant across
    the SCF and across geometries — computed once per atom when the batch is
    prepared. Mirrors what the sequential solver does per iteration.
    """
    from .tetci_multipole_pyseqm import PM6_TAIL_EXPONENTS
    from .w_integrals import compute_w_integrals
    from .params import principal_qn

    qn = principal_qn(p.Z)
    if p.Z in PM6_TAIL_EXPONENTS:
        zs_t, zp_t, zd_t = PM6_TAIL_EXPONENTS[p.Z]
    else:
        zs_t, zp_t, zd_t = p.zeta_s, p.zeta_p, p.zeta_d
    return compute_w_integrals(zs_t, zp_t, zd_t, qn, qn,
                               getattr(p, 'F0SD', 0.0), getattr(p, 'G2SD', 0.0))


def _two_centre_packed(pA, pB, rA, rB, dense=None) -> np.ndarray:
    """Packed two-electron block for one atom pair, (packed(nA), packed(nB)).

    Two sources, one output convention:

    * pairs where either atom carries d orbitals come from the vendored TETCI
      port, which already computes in the packed convention — the dense (9,9,9,9)
      expansion that ``d_two_center`` builds is undone immediately afterwards, so
      the packed tensor is taken directly and the round-trip is skipped;
    * everything else comes from the sp rotation, which returns dense, and is
      packed here.

    The two packings nest — for orbitals below 4 the 9-basis index equals the
    4-basis one — so both land in the same convention with no branch downstream.
    """
    nA, nB = pA.n_basis, pB.n_basis

    if nA == 9 or nB == 9:
        from .d_two_center import _tetci_pair_w
        w, first_is_A = _tetci_pair_w(pA, pB, rA, rB)
        if w is not None:
            pa, pb = packed_size(nA), packed_size(nB)
            # TETCI indexes [second centre pair, first centre pair].
            return (w.T[:pa, :pb] if first_is_A else w[:pa, :pb]).copy()

    if dense is None:
        dense, _, _ = rotate_integrals_to_molecular_frame(pA, pB, rA, rB)
    # The rotation pads every centre to its shell width — hydrogen comes back
    # 4-wide — so trim to the orbitals that actually exist before packing.
    return pack(dense[:nA, :nA, :nB, :nB], nA, nB)


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
    # Packed two-centre storage: a flat buffer per molecule plus an offset
    # table. -1 marks a pair with no block (an atom with itself, or padding).
    w_packed_rows = [[] for _ in range(N)]
    pair_cursor = [0] * N
    pair_offset_all = np.full((N, MA, MA), -1, dtype=np.int32)
    atom_norb_all = np.zeros((N, MA), dtype=np.int32)
    # One-centre d integrals. These depend only on the atom's own
    # parameters, never on geometry, so they are computed once here
    # rather than per SCF iteration. Zero for atoms without d.
    atom_w_all = np.zeros((N, MA, 243), dtype=np.float64)
    atom_params_all = np.zeros((N, MA, 5), dtype=np.float64)
    atom_map_all = np.zeros((N, MB), dtype=np.int32)
    type_map_all = np.zeros((N, MB), dtype=np.int32)
    atom_starts_all = np.zeros((N, MA + 1), dtype=np.int32)
    E_nuc_arr = np.zeros(N, dtype=np.float64)

    atoms_list = []
    coords_list = []

    # Every sp pair in the whole batch is rotated in one vectorised call
    # before the per-molecule loop below. Doing it inside the loop meant one
    # scalar rotation per pair per molecule, which was ~69% of prepare_batch.
    from .rotation_batch import rotate_pairs
    _pw_keys, _pw_params, _pw_coords = [], [], []
    for mol_idx, (atoms, coords) in enumerate(molecules):
        coords_arr = np.asarray(coords, dtype=np.float64)
        mol_params = [param_dict[z] for z in atoms]
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if mol_params[i].n_basis <= 4 and mol_params[j].n_basis <= 4:
                    _pw_keys.append((mol_idx, i, j))
                    _pw_params.append((mol_params[i], mol_params[j]))
                    _pw_coords.append((coords_arr[i], coords_arr[j]))
    if _pw_keys:
        _rotated = rotate_pairs(_pw_params, _pw_coords)
        pair_w = {key: _rotated[k] for k, key in enumerate(_pw_keys)}
    else:
        pair_w = {}

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
            atom_norb_all[mol_idx, i] = p.n_basis
            if p.n_basis == 9:
                atom_w_all[mol_idx, i] = _one_centre_w(p)

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

        # Electron-nuclear attraction. One rotation of the pair (i, j) yields the
        # attraction on i from j *and* on j from i, so the unordered loop below
        # does half the work the ordered one did. d pairs keep the 9x9 Wigner-D
        # path, which is not expressible through the sp rotation.
        for i in range(n_at):
            si, nA = starts[i], params[i].n_basis
            for j in range(i + 1, n_at):
                sj, nB = starts[j], params[j].n_basis
                pA, pB = params[i], params[j]
                if pA.n_basis == 9 or pB.n_basis == 9:
                    H[si:si + nA, si:si + nA] += _pair_core_attraction(
                        pA, pB, coords[i], coords[j])
                    H[sj:sj + nB, sj:sj + nB] += _pair_core_attraction(
                        pB, pA, coords[j], coords[i])
                else:
                    w_ij = pair_w[(mol_idx, i, j)]
                    # e1b and e2a are slices of the rotated tensor.
                    H[si:si + nA, si:si + nA] += (
                        -float(pB.n_valence) * w_ij[:nA, :nA, 0, 0])
                    H[sj:sj + nB, sj:sj + nB] += (
                        -float(pA.n_valence) * w_ij[0, 0, :nB, :nB])

        # === Two-centre integrals, packed ===
        # Stored as lower-triangle packed pair blocks with a per-pair offset,
        # not as a dense 4-index block padded to the widest atom in the batch.
        # Padding cost a 100-molecule PM6 batch containing one sulfur ~5 GB
        # because every C-H pair was inflated to 9**4; packed per-pair it is
        # ~24 MB, and sp-only methods travel the same path as the small case.
        for i in range(n_at):
            for j in range(i + 1, n_at):
                block = _two_centre_packed(params[i], params[j],
                                           coords[i], coords[j],
                                           dense=pair_w.get((mol_idx, i, j)))
                off_ij = pair_cursor[mol_idx]
                w_packed_rows[mol_idx].append(block.ravel())
                pair_offset_all[mol_idx, i, j] = off_ij
                pair_cursor[mol_idx] += block.size

                # The (j, i) ordering is the transpose in packed space too.
                off_ji = pair_cursor[mol_idx]
                w_packed_rows[mol_idx].append(block.T.ravel())
                pair_offset_all[mol_idx, j, i] = off_ji
                pair_cursor[mol_idx] += block.size

        H_core_all[mol_idx, :n_bas, :n_bas] = H

        # Nuclear repulsion
        E_nuc_arr[mol_idx] = nuclear_repulsion_for_method(
            atoms, coords, param_dict, method)

    # Pad the ragged per-molecule buffers into one (N, max_storage) array so
    # the GPU sees a single contiguous upload. Molecules needing less are
    # zero-filled; the offset table means the padding is never read.
    max_storage = max(pair_cursor) if N else 0
    w_all = np.zeros((N, max_storage), dtype=np.float64)
    for m in range(N):
        if w_packed_rows[m]:
            flat = np.concatenate(w_packed_rows[m])
            w_all[m, :flat.size] = flat

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
        pair_offset=pair_offset_all,
        atom_norb=atom_norb_all,
        atom_w=atom_w_all,
        atom_params=atom_params_all,
        atom_map=atom_map_all,
        type_map=type_map_all,
        atom_starts=atom_starts_all,
        E_nuc=E_nuc_arr,
        coords_list=coords_list,
        molecular_charges=molecular_charges_arr,
    )
