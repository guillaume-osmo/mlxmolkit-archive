"""
Batch processing for RM1 SCF: N molecules simultaneously on Metal GPU.

Pads all molecules to uniform max_atoms × max_basis dimensions
so a single Metal kernel dispatch handles all Fock matrices.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from dataclasses import dataclass, field
from .params import RM1_PARAMS, ElementParams, ANG_TO_BOHR, EV_TO_KCAL
from .overlap import overlap_molecular_frame
from .rotation import rotate_integrals_to_molecular_frame
from .packing import pack, packed_size, unpack
from .scf import (_pair_resonance_block, _pair_core_attraction,
                  _beta_for_orbital)
from .integrals import (compute_nuclear_repulsion, nuclear_repulsion_for_method,
                        PM6_CORE_CORE_METHODS, normalize_method)


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




@lru_cache(maxsize=None)
def _one_centre_w_cached(Z, zeta_s, zeta_p, zeta_d, F0SD, G2SD) -> np.ndarray:
    from .tetci_multipole_pyseqm import PM6_TAIL_EXPONENTS
    from .w_integrals import compute_w_integrals
    from .params import principal_qn

    qn = principal_qn(Z)
    if Z in PM6_TAIL_EXPONENTS:
        zs_t, zp_t, zd_t = PM6_TAIL_EXPONENTS[Z]
    else:
        zs_t, zp_t, zd_t = zeta_s, zeta_p, zeta_d
    out = compute_w_integrals(zs_t, zp_t, zd_t, qn, qn, F0SD, G2SD)
    out.flags.writeable = False
    return out


def _one_centre_w(p) -> np.ndarray:
    """The 243 one-centre two-electron integrals for a d-bearing atom.

    A function of the atom's Slater exponents alone — no geometry, no density —
    so it is computed once per element rather than once per atom per molecule.
    An 800-molecule batch has 760 d atoms and two distinct d elements; without
    the cache that is 760 runs of the Slater-Condon machinery (126560 calls to
    _binom alone) for two distinct answers.

    The cached array is returned read-only so a caller cannot mutate the shared
    copy.
    """
    return _one_centre_w_cached(
        p.Z, p.zeta_s, p.zeta_p, p.zeta_d,
        getattr(p, 'F0SD', 0.0), getattr(p, 'G2SD', 0.0))


def _one_centre_w_uncached(p) -> np.ndarray:
    """Kept as the reference the cache is checked against."""
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


def _two_centre_packed(pA, pB, rA, rB, dense=None, tetci=None) -> np.ndarray:
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
        if tetci is not None:
            w, first_is_A = tetci
        else:
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
        param_dict: parameter dictionary. Defaults to `method`'s own
            parameters; pass it only to override them.
        method: NDDO method name — selects the core-core repulsion form and,
            by default, the parameters. Must match the method `param_dict`
            came from, or PM6/PM6_D energies come out several eV wrong.

    Returns:
        RM1Batch with all pre-computed data
    """
    if param_dict is None:
        # `method`'s parameters, not RM1's. The old default ignored `method`
        # entirely, so `prepare_batch(mols, method='PM6')` built PM6 core-core
        # repulsion on top of RM1 parameters — and RM1 gives P, S, Cl and Br
        # four basis functions where PM6 gives them nine, so a batch
        # containing sulfur silently came back sp-only with no error anywhere.
        # `nddo_energy_batch` always passed param_dict explicitly and was
        # never affected; a direct caller was.
        from .methods import get_params
        param_dict = get_params(method)
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

    # ---- one pair enumeration for the whole batch -------------------------
    # prepare_batch used to walk every molecule's atom pairs seven times: three
    # here to build the spec lists for the bulk precomputes, and four more in
    # the per-molecule loop below. The enumeration is identical every time and
    # only the predicate differs, so it is built once and the groupings are
    # masks over it. np.triu_indices emits the same (0,1),(0,2),...,(1,2),...
    # order the nested loops did, which the packed-pair cursor depends on.
    #
    # Every pair also gets a dense id. The precomputed results are indexed by
    # it instead of by a (mol_idx, i, j) tuple: keying them on tuples cost
    # 730838 dict lookups per batch, and hashing a 3-tuple is not free.
    params_by_mol = [[param_dict[z] for z in atoms] for atoms, _c in molecules]
    coords_by_mol = [np.asarray(c, dtype=np.float64) for _a, c in molecules]

    _mol, _i, _j, _nA, _nB = [], [], [], [], []
    mol_pair_start = np.zeros(N + 1, dtype=np.int64)
    for m, ps in enumerate(params_by_mol):
        i_m, j_m = np.triu_indices(len(ps), k=1)
        norb = np.array([p.n_basis for p in ps], dtype=np.int32)
        _mol.append(np.full(i_m.size, m, dtype=np.int32))
        _i.append(i_m.astype(np.int32))
        _j.append(j_m.astype(np.int32))
        _nA.append(norb[i_m])
        _nB.append(norb[j_m])
        mol_pair_start[m + 1] = mol_pair_start[m] + i_m.size

    atom_flat_start = np.zeros(N + 1, dtype=np.int64)
    for m, ps in enumerate(params_by_mol):
        atom_flat_start[m + 1] = atom_flat_start[m] + len(ps)

    # Where each atom's basis functions start inside its own molecule, in the
    # flat atom order the pair table uses. Lets a pair's H_core rows be
    # computed from its dense id alone, with no per-molecule bookkeeping.
    if N:
        norb_flat = np.concatenate(
            [np.array([p.n_basis for p in ps], dtype=np.int64)
             for ps in params_by_mol])
        basis_start_flat = np.zeros(norb_flat.size, dtype=np.int64)
        for m in range(N):
            a0, a1 = atom_flat_start[m], atom_flat_start[m + 1]
            if a1 > a0:
                basis_start_flat[a0:a1] = np.cumsum(norb_flat[a0:a1]) - norb_flat[a0:a1]
    else:
        norb_flat = basis_start_flat = np.zeros(0, dtype=np.int64)

    if N:
        pair_mol = np.concatenate(_mol)
        pair_i = np.concatenate(_i)
        pair_j = np.concatenate(_j)
        pair_nA = np.concatenate(_nA)
        pair_nB = np.concatenate(_nB)
    else:
        pair_mol = pair_i = pair_j = pair_nA = pair_nB = np.zeros(0, dtype=np.int32)
    n_pairs = pair_mol.size

    # An sp pair is one both of whose atoms fit in 4 orbitals; everything else
    # needs the d path. One mask, two index sets, two dense slot maps.
    sp_mask = (pair_nA <= 4) & (pair_nB <= 4)
    sp_ids = np.flatnonzero(sp_mask)
    d_ids = np.flatnonzero(~sp_mask)
    sp_slot = np.full(n_pairs, -1, dtype=np.int64)
    sp_slot[sp_ids] = np.arange(sp_ids.size)
    d_slot = np.full(n_pairs, -1, dtype=np.int64)
    d_slot[d_ids] = np.arange(d_ids.size)

    def _params_of(ids):
        return [(params_by_mol[pair_mol[k]][pair_i[k]],
                 params_by_mol[pair_mol[k]][pair_j[k]]) for k in ids]

    def _coords_of(ids):
        return [(coords_by_mol[pair_mol[k]][pair_i[k]],
                 coords_by_mol[pair_mol[k]][pair_j[k]]) for k in ids]

    sp_params, sp_coords = _params_of(sp_ids), _coords_of(sp_ids)
    d_params, d_coords = _params_of(d_ids), _coords_of(d_ids)
    d_specs = [(pA, pB, rA, rB) for (pA, pB), (rA, rB) in zip(d_params, d_coords)]

    # Every sp pair in the whole batch is rotated in one vectorised call.
    # Doing it inside the per-molecule loop meant one scalar rotation per pair,
    # which was ~69% of prepare_batch.
    from .rotation_batch import rotate_pairs
    pair_w = rotate_pairs(sp_params, sp_coords) if sp_ids.size else None

    # d-bearing pairs the same way. two_elec_two_center_int is vectorised over
    # a pair axis, so asking it for one pair at a time cost 2.58 ms each and
    # made a batch with 10% d-containing molecules twice as expensive per
    # molecule as an sp-only one.
    from .d_two_center import _tetci_pairs_w
    pair_tetci = _tetci_pairs_w(d_specs) if d_ids.size else None

    # Resonance overlaps. diatom_overlap_matrixD is written for a batch and was
    # being handed one pair at a time at ~1.4 ms each; overlap_pairs routes
    # anything its table does not cover — d orbitals, or qn > 3 like Br and I —
    # back to the scalar routine itself.
    from .overlap_d import overlap_d_batch
    from .overlap_batch import overlap_pairs
    overlap_d = overlap_d_batch(d_specs) if d_ids.size else None
    overlap_sp = overlap_pairs(
        [(pA, pB, rA, rB) for (pA, pB), (rA, rB) in zip(sp_params, sp_coords)]
    ) if sp_ids.size else None

    # Electron-nuclear attraction on the d atoms, and on the sp atom of a mixed
    # pair. Both directions of every d pair land in one of the two lists.
    from .tetci_yh import yh_e1b_batch
    _yh_at, _yh_pp, _yh_pc = [], [], []
    _sp_at, _sp_att_params, _sp_att_coords = [], [], []
    for pos, k in enumerate(d_ids):
        pA, pB, rA, rB = d_specs[pos]
        for slot, (pX, pY, rX, rY) in enumerate(((pA, pB, rA, rB), (pB, pA, rB, rA))):
            if pX.n_basis == 9:
                _yh_at.append((pos, slot))
                _yh_pp.append((pX, pY))
                _yh_pc.append((rX, rY))
            else:
                _sp_at.append((pos, slot))
                _sp_att_params.append((pX, pY))
                _sp_att_coords.append((rX, rY))
    pair_attraction: list = [[None, None] for _ in range(d_ids.size)]
    if _yh_at:
        for (pos, slot), blk in zip(_yh_at, yh_e1b_batch(_yh_pp, _yh_pc)):
            pair_attraction[pos][slot] = blk
    if _sp_at:
        _w_att = rotate_pairs(_sp_att_params, _sp_att_coords)
        for n, (pos, slot) in enumerate(_sp_at):
            nX = _sp_att_params[n][0].n_basis
            pair_attraction[pos][slot] = (
                -float(_sp_att_params[n][1].n_valence) * _w_att[n][:nX, :nX, 0, 0])

    # Resonance blocks in bulk. With the overlap already computed this is only
    # the Wolfsberg-Helmholz weighting 0.5*(beta_mu + beta_nu)*S, which is an
    # outer sum over orbital indices — 65400 per-pair calls became one
    # expression per element pair.
    pair_resonance: list = [None] * sp_ids.size
    _res_shapes: dict[tuple, list] = {}
    for pos, (pA, pB) in enumerate(sp_params):
        _res_shapes.setdefault((pA.Z, pB.Z), []).append(pos)
    for (zA, zB), positions in _res_shapes.items():
        pA, pB = param_dict[zA], param_dict[zB]
        betaA = np.array([_beta_for_orbital(pA, k) for k in range(pA.n_basis)])
        betaB = np.array([_beta_for_orbital(pB, k) for k in range(pB.n_basis)])
        weight = 0.5 * (betaA[:, None] + betaB[None, :])
        stack = np.stack([overlap_sp[pos] for pos in positions]) * weight
        for n, pos in enumerate(positions):
            pair_resonance[pos] = stack[n]

    # Pack every sp pair's two-electron block in bulk, grouped by orbital
    # widths. pack() called per pair was 132 ms across 64640 calls on an
    # 800-molecule batch; the index map is the same for every pair of a given
    # shape, so each shape is one scatter.
    from .packing import pack_batch
    pair_packed: list = [None] * sp_ids.size
    by_shape: dict[tuple, list] = {}
    for pos, (pA, pB) in enumerate(sp_params):
        by_shape.setdefault((pA.n_basis, pB.n_basis), []).append(pos)
    for (nA, nB), positions in by_shape.items():
        stack = np.stack([pair_w[pos][:nA, :nA, :nB, :nB] for pos in positions])
        packed = pack_batch(stack, nA, nB)
        for n, pos in enumerate(positions):
            pair_packed[pos] = packed[n]

    # === H_core off-diagonal and electron-nuclear attraction, in bulk ===
    #
    # The per-pair loop below used to do four numpy slice operations per pair
    # for this — resonance into two off-diagonal blocks, attraction onto two
    # diagonal ones — which is ~240k tiny calls for a 200-molecule batch and
    # most of `prepare_batch`'s own time.
    #
    # Grouped by orbital shape it is four scatters per shape. The off-diagonal
    # blocks are written (each belongs to exactly one pair, so assignment is
    # safe); the diagonal ones are accumulated through bincount, because every
    # pair touching an atom lands on that atom's block.
    #
    # d pairs keep the loop: their resonance comes from a different overlap
    # routine and their attraction from the Wigner-D path.
    _H_flat = H_core_all.reshape(-1)
    _stride = MB * MB
    if sp_ids.size:
        _sp_by_shape: dict[tuple, list] = {}
        for pos, (pA, pB) in enumerate(sp_params):
            _sp_by_shape.setdefault((pA.n_basis, pB.n_basis), []).append(pos)

        _acc_idx, _acc_val = [], []
        for (nA, nB), positions in _sp_by_shape.items():
            sel = np.asarray(positions)
            pids = sp_ids[sel]
            mols = pair_mol[pids].astype(np.int64)
            base = atom_flat_start[mols]
            sA = basis_start_flat[base + pair_i[pids]]
            sB = basis_start_flat[base + pair_j[pids]]
            rows_a = sA[:, None] + np.arange(nA)
            rows_b = sB[:, None] + np.arange(nB)
            off = mols * _stride

            blocks = np.stack([pair_resonance[pos] for pos in positions])
            _H_flat[(off[:, None, None] + rows_a[:, :, None] * MB
                     + rows_b[:, None, :]).ravel()] = blocks.ravel()
            _H_flat[(off[:, None, None] + rows_b[:, :, None] * MB
                     + rows_a[:, None, :]).ravel()] = np.swapaxes(
                         blocks, 1, 2).ravel()

            w_grp = np.stack([pair_w[pos] for pos in positions])
            val_b = np.array([sp_params[pos][1].n_valence for pos in positions],
                             dtype=np.float64)
            val_a = np.array([sp_params[pos][0].n_valence for pos in positions],
                             dtype=np.float64)
            att_a = -val_b[:, None, None] * w_grp[:, :nA, :nA, 0, 0]
            att_b = -val_a[:, None, None] * w_grp[:, 0, 0, :nB, :nB]
            _acc_idx.append((off[:, None, None] + rows_a[:, :, None] * MB
                             + rows_a[:, None, :]).ravel())
            _acc_val.append(att_a.ravel())
            _acc_idx.append((off[:, None, None] + rows_b[:, :, None] * MB
                             + rows_b[:, None, :]).ravel())
            _acc_val.append(att_b.ravel())

        _H_flat += np.bincount(np.concatenate(_acc_idx),
                               weights=np.concatenate(_acc_val),
                               minlength=_H_flat.size)

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

        # === H_core diagonal: Uss/Upp/Udd ===
        # The off-diagonal resonance and the electron-nuclear attraction were
        # scattered in bulk above; this adds the one-centre diagonal on top,
        # so it accumulates rather than assigns.
        H = H_core_all[mol_idx, :n_bas, :n_bas]
        for mu in range(n_bas):
            i = b2a[mu]
            p = params[i]
            if btype[mu] == 0:
                H[mu, mu] += p.Uss
            elif btype[mu] <= 3:
                H[mu, mu] += p.Upp
            else:
                H[mu, mu] += p.Udd

        starts = atom_basis_start

        # One walk over this molecule's slice of the pair table, doing what
        # four separate (i, j) walks used to: resonance, electron-nuclear
        # attraction, packing, and the core-core term list. Each precomputed
        # result is reached by the pair's dense id rather than by hashing an
        # (mol_idx, i, j) tuple.
        starts = atom_basis_start
        pm6_core = normalize_method(method) in PM6_CORE_CORE_METHODS

        for pid in range(mol_pair_start[mol_idx], mol_pair_start[mol_idx + 1]):
            i, j = int(pair_i[pid]), int(pair_j[pid])
            pA, pB = params[i], params[j]
            nA, nB = int(pair_nA[pid]), int(pair_nB[pid])
            si, sj = starts[i], starts[j]
            sp_pos = int(sp_slot[pid])

            # Resonance. The batched block covers every sp pair; the d path
            # keeps the per-pair routine, which reuses the precomputed overlap.
            if sp_pos < 0:
                block = _pair_resonance_block(
                    pA, pB, coords[i], coords[j],
                    overlap=overlap_d[int(d_slot[pid])])
                H[si:si + nA, sj:sj + nB] = block
                H[sj:sj + nB, si:si + nA] = block.T

            # Electron-nuclear attraction. One rotation of the pair yields the
            # attraction on i from j *and* on j from i, so this does half the
            # work an ordered loop would.
            if sp_pos < 0:
                blk_a, blk_b = pair_attraction[int(d_slot[pid])]
                H[si:si + nA, si:si + nA] += blk_a
                H[sj:sj + nB, sj:sj + nB] += blk_b

            # Two-centre integrals, stored as lower-triangle packed pair blocks
            # with a per-pair offset rather than a dense 4-index block padded to
            # the widest atom in the batch. Padding cost a 100-molecule PM6
            # batch containing one sulfur ~5 GB because every C-H pair was
            # inflated to 9**4; packed per-pair it is ~24 MB.
            if sp_pos >= 0:
                block = pair_packed[sp_pos]
            else:
                block = _two_centre_packed(
                    pA, pB, coords[i], coords[j],
                    dense=None, tetci=pair_tetci[int(d_slot[pid])])
            rows = w_packed_rows[mol_idx]
            pair_offset_all[mol_idx, i, j] = pair_cursor[mol_idx]
            rows.append(block.ravel())
            pair_cursor[mol_idx] += block.size
            # The (j, i) ordering is the transpose in packed space too.
            pair_offset_all[mol_idx, j, i] = pair_cursor[mol_idx]
            rows.append(block.T.ravel())
            pair_cursor[mol_idx] += block.size

        # The AM1-style core-core form stays per molecule; PM6's is summed from
        # the batched per-pair terms after the loop.
        if not pm6_core:
            E_nuc_arr[mol_idx] = nuclear_repulsion_for_method(
                atoms, coords, param_dict, method)

    # PM6 core-core: every pair in the batch in one call, then summed back per
    # molecule. The AM1-style form stays per molecule above. Both endpoints
    # come straight off the pair table by fancy indexing — collecting them in
    # the loop meant four list appends per pair and two more comprehensions to
    # turn them back into arrays.
    if normalize_method(method) in PM6_CORE_CORE_METHODS and n_pairs:
        from .pwcct import pm6_pair_repulsion_batch
        flat_z = np.concatenate([np.asarray(a, dtype=np.int64)
                                 for a, _c in molecules])
        flat_c = np.concatenate(coords_by_mol)
        base = atom_flat_start[pair_mol]
        ia, ja = base + pair_i, base + pair_j
        terms = pm6_pair_repulsion_batch(
            flat_z[ia], flat_z[ja], None, flat_c[ia], flat_c[ja],
            param_dict=param_dict)
        np.add.at(E_nuc_arr, pair_mol, terms)

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
