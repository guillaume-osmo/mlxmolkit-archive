"""
Analytical gradient for NDDO methods.

Frozen-density (Hellmann-Feynman) gradient: one converged SCF, then the energy
is re-evaluated at 6N displaced geometries holding the density fixed. Exact for
a variational density — it reproduces a central-difference gradient to 1e-4
eV/A.

The cost used to be 6N *full* rebuilds of H and F. Displacing one atom only
changes the pairs that touch it, so all but O(N) of that work was thrown away:
at 31 atoms the full rebuild does 465 pair integrals where 30 suffice, and
H_core alone was 69% of a gradient evaluation. This module now rebuilds only
the dirty pairs and patches the reference matrices.
"""
from __future__ import annotations

import numpy as np
from .scf import nddo_energy
from .methods import get_params
from .integrals import (compute_nuclear_repulsion, nuclear_repulsion_for_method,
                        pair_repulsion_for_method)
from .integrals import PM6_CORE_CORE_METHODS
from .pwcct import normalize_method


def _pair_terms(params, coords, i, j, starts, P, n_basis, w=None):
    """Everything about the pair (i, j) that moves when either atom moves.

    Returns (dH, dT): the pair's additive contribution to the core Hamiltonian
    and to the two-centre part of the Fock matrix, both full-size so they can
    simply be added to or subtracted from the reference matrices.
    """
    from .scf import (_pair_resonance_block, _pair_core_attraction,
                      _pair_fock_twocentre)
    from .rotation import rotate_integrals_to_molecular_frame

    pA, pB = params[i], params[j]
    sA, sB = starts[i], starts[j]
    nA, nB = pA.n_basis, pB.n_basis
    rA, rB = coords[i], coords[j]

    dH = np.zeros((n_basis, n_basis))
    block = _pair_resonance_block(pA, pB, rA, rB)
    dH[sA:sA + nA, sB:sB + nB] += block
    dH[sB:sB + nB, sA:sA + nA] += block.T

    if nA == 9 or nB == 9:
        # d pairs keep the 9x9 Wigner-D attraction and the d two-centre path.
        dH[sA:sA + nA, sA:sA + nA] += _pair_core_attraction(pA, pB, rA, rB)
        dH[sB:sB + nB, sB:sB + nB] += _pair_core_attraction(pB, pA, rB, rA)
        dT = _pair_fock_twocentre(np.zeros((n_basis, n_basis)), P,
                                  pA, pB, sA, sB, rA, rB)
        return dH, dT

    # One rotation supplies all three sp contributions. Attraction is not
    # symmetric in the pair — (i,j) lands on i's diagonal block and (j,i) on
    # j's — but e1b and e2a are exactly those two orderings, so asking for the
    # rotation three times was paying three times for one answer.
    if w is None:
        w, e1b, e2a = rotate_integrals_to_molecular_frame(pA, pB, rA, rB)
    else:
        # e1b and e2a are slices of w, so a batched caller need only supply w.
        e1b = -float(pB.n_valence) * w[:, :, 0, 0]
        e2a = -float(pA.n_valence) * w[0, 0, :, :]
    dH[sA:sA + nA, sA:sA + nA] += e1b[:nA, :nA]
    dH[sB:sB + nB, sB:sB + nB] += e2a[:nB, :nB]
    dT = _pair_fock_twocentre(np.zeros((n_basis, n_basis)), P,
                              pA, pB, sA, sB, rA, rB, w=w)
    return dH, dT



def _pair_terms_many(params, coords, pairs, starts, P, n_basis):
    """`_pair_terms` for a list of pairs, rotating them in one vectorised call.

    The scalar rotation costs ~75 us per pair and was the single largest line
    in a gradient profile. Grouping the pairs and rotating them together turns
    that into a handful of array operations; the assembly below stays per pair,
    but it is only numpy slicing.

    d-bearing pairs used to fall back to the scalar path on the grounds that
    "there are few of them". That is true of the pair count and false of the
    cost: one d pair costs 2.58 ms in `_tetci_pair_w` against ~70 us for an sp
    pair, so thioanisole (16 atoms, one sulfur) took 1440 ms per gradient while
    menthol (31 atoms, no d) took 389 ms. They are batched here through the
    same TETCI call `prepare_batch` uses.
    """
    from .rotation_batch import rotate_pairs

    sp, dd = [], []
    for i, j in pairs:
        (sp if params[i].n_basis <= 4 and params[j].n_basis <= 4 else dd).append((i, j))

    from .d_two_center import _ROT_CACHE, _pair_key

    out = {}
    if sp:
        if _ROT_CACHE is not None:
            ws = [_ROT_CACHE.get(_pair_key(params[i], params[j],
                                           coords[i], coords[j])) for i, j in sp]
            if any(w is None for w in ws):        # a geometry the cache never saw
                ws = rotate_pairs([(params[i], params[j]) for i, j in sp],
                                  [(coords[i], coords[j]) for i, j in sp])
        else:
            ws = rotate_pairs([(params[i], params[j]) for i, j in sp],
                              [(coords[i], coords[j]) for i, j in sp])
        for k, (i, j) in enumerate(sp):
            out[(i, j)] = _pair_terms(params, coords, i, j, starts, P, n_basis,
                                      w=ws[k])
    for i, j in dd:
        out[(i, j)] = _pair_terms(params, coords, i, j, starts, P, n_basis)
    return out


def _pair_energy_many(params, coords, pairs, starts, P, n_basis, shift=None):
    """Each pair's scalar contribution to `sum(P * (2H + T))`.

    A pair's dH and dT are exactly zero outside its four blocks — [sA,sA],
    [sB,sB], [sA,sB], [sB,sA] — for the d path as well as sp, so its whole
    effect on the electronic energy collapses to one number. Returning that
    number instead of two (n_basis, n_basis) matrices means a displaced
    geometry costs a few float subtractions rather than a full-matrix copy
    and 2(N-1) full-matrix adds carrying 4x4 changes.

    The sp path builds the blocks directly and never allocates a full matrix.
    d pairs go through :func:`_pair_terms` and are reduced afterwards; they are
    rare, and the point here is to stop paying n_basis**2 for every sp pair.

    `shift` displaces every pair's *second* atom by the same vector. A pair's
    energy depends only on the relative vector between its two atoms, so one
    global shift gives the displaced energy of every pair at once — which is
    what lets the gradient take six batched passes over all pairs instead of
    6N passes over the N-1 pairs touching each atom in turn.
    """
    from .scf import (_pair_resonance_block, _pair_fock_twocentre,
                      _pair_core_attraction)
    from .rotation_batch import rotate_pairs
    from .d_two_center import _ROT_CACHE, _pair_key

    if shift is None:
        coords_b = coords
    else:
        coords_b = coords + np.asarray(shift, dtype=np.float64)

    sp, dd = [], []
    for i, j in pairs:
        (sp if params[i].n_basis <= 4 and params[j].n_basis <= 4
         else dd).append((i, j))

    out: dict[tuple[int, int], float] = {}

    if sp:
        ws = None
        if _ROT_CACHE is not None and shift is None:
            ws = [_ROT_CACHE.get(_pair_key(params[i], params[j],
                                           coords[i], coords[j])) for i, j in sp]
            if any(w is None for w in ws):
                ws = None
        if ws is None:
            ws = rotate_pairs([(params[i], params[j]) for i, j in sp],
                              [(coords[i], coords_b[j]) for i, j in sp])

        # Grouped by orbital shape so the contractions run once per shape
        # rather than once per pair. A displaced geometry touches N-1 pairs
        # and each was doing three 4x4x4x4 einsums plus four reductions —
        # ~97k tiny numpy calls per gradient on cholesterol, where the arrays
        # are far too small to cover the per-call overhead.
        shapes: dict[tuple[int, int], list[int]] = {}
        for k, (i, j) in enumerate(sp):
            shapes.setdefault((params[i].n_basis, params[j].n_basis),
                              []).append(k)

        for (nA, nB), ks in shapes.items():
            sel = np.asarray(ks)
            ia = np.asarray([starts[sp[k][0]] for k in ks])
            ib = np.asarray([starts[sp[k][1]] for k in ks])
            rows_a = ia[:, None] + np.arange(nA)
            rows_b = ib[:, None] + np.arange(nB)

            W = np.stack([ws[k] for k in ks])[:, :nA, :nA, :nB, :nB]
            P_AA = P[rows_a[:, :, None], rows_a[:, None, :]]
            P_BB = P[rows_b[:, :, None], rows_b[:, None, :]]
            P_AB = P[rows_a[:, :, None], rows_b[:, None, :]]
            P_BA = P[rows_b[:, :, None], rows_a[:, None, :]]

            h_ab = np.stack([
                _pair_resonance_block(params[sp[k][0]], params[sp[k][1]],
                                      coords[sp[k][0]], coords_b[sp[k][1]])
                for k in ks])
            val_b = np.asarray([float(params[sp[k][1]].n_valence) for k in ks])
            val_a = np.asarray([float(params[sp[k][0]].n_valence) for k in ks])
            h_aa = -val_b[:, None, None] * W[:, :, :, 0, 0]
            h_bb = -val_a[:, None, None] * W[:, 0, 0, :, :]

            t_aa = np.einsum('gabcd,gcd->gab', W, P_BB)
            t_bb = np.einsum('gabcd,gab->gcd', W, P_AA)
            t_ab = -0.5 * np.einsum('gabcd,gbd->gac', W, P_AB)

            totals = (
                np.sum(P_AA * (2.0 * h_aa + t_aa), axis=(1, 2))
                + np.sum(P_BB * (2.0 * h_bb + t_bb), axis=(1, 2))
                + np.sum(P_AB * (2.0 * h_ab + t_ab), axis=(1, 2))
                + np.sum(P_BA * (2.0 * np.swapaxes(h_ab, 1, 2)
                                 + np.swapaxes(t_ab, 1, 2)), axis=(1, 2))
            )
            for pos, k in enumerate(ks):
                out[sp[k]] = float(totals[pos])

    for i, j in dd:
        if shift is None:
            dH, dT = _pair_terms(params, coords, i, j, starts, P, n_basis)
        else:
            shifted = coords.copy()
            shifted[j] = coords_b[j]
            dH, dT = _pair_terms(params, shifted, i, j, starts, P, n_basis)
        sA, nA = starts[i], params[i].n_basis
        sB, nB = starts[j], params[j].n_basis
        total = 0.0
        for rows, cols in (
            (slice(sA, sA + nA), slice(sA, sA + nA)),
            (slice(sB, sB + nB), slice(sB, sB + nB)),
            (slice(sA, sA + nA), slice(sB, sB + nB)),
            (slice(sB, sB + nB), slice(sA, sA + nA)),
        ):
            total += float(np.sum(P[rows, cols]
                                  * (2.0 * dH[rows, cols] + dT[rows, cols])))
        out[(i, j)] = total

    return out


def analytical_gradient(
    atoms: list[int],
    coords: np.ndarray,
    method: str = 'RM1',
    step: float = 1e-5,
    molecular_charge: float = 0.0,
    scf_result: dict | None = None,
) -> tuple[dict, np.ndarray]:
    """Compute energy and gradient.

    Costs 1 SCF plus 6N displacements, each touching only the N-1 pairs that
    moved rather than all N(N-1)/2.

    Returns:
        result: SCF result dict
        gradient: (n_atoms, 3) in eV/Angstrom
    """
    from .scf import _build_basis_info, _build_core_hamiltonian, _build_fock
    from .d_two_center import pair_cache

    PARAMS = get_params(method)
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)
    info = _build_basis_info(atoms, PARAMS, molecular_charge=molecular_charge)
    params = info['params']
    starts = info['atom_basis_start']

    # The 6N displaced geometries, built once and reused verbatim below. The
    # TETCI cache is keyed on coordinate bytes, and a geometry rebuilt as
    # `coords.copy(); c[a, d] += step` need not come out bit-identical, so the
    # arrays a displacement is evaluated at must be the same objects the cache
    # was keyed on.
    # Six rigid shifts, not 6N displaced geometries. The total energy is a
    # constant plus a sum of per-pair terms (the one-centre part of H and
    # G_one do not move), and each pair term depends only on the vector
    # between its two atoms — so shifting every pair's second atom by the same
    # delta gives every pair's displaced energy in one pass. MOPAC's dcart is
    # organised the same way, per pair rather than per atom.
    shifts = []
    for d in range(3):
        for sign in (1.0, -1.0):
            delta = np.zeros(3)
            delta[d] = sign * step
            shifts.append((d, sign, delta))

    # Every pair this gradient will ask for, at every geometry it will ask at:
    # the reference geometry — which the SCF, H_ref, F_ref and pair_ref all
    # use — and each displacement's own N-1 dirty pairs. `_pair_terms_many`
    # already batched the sp rotation, but only within one displacement: 86
    # calls of ~15 pairs for benzaldehyde, where all 1092 in one call is 2.4 ms
    # against 23.6. Batching across all 6N geometries is the whole win.
    pair_specs = [(params[i], params[j], coords[i], coords[j])
                  for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    for _d, _sign, delta in shifts:
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pair_specs.append((params[i], params[j],
                                   coords[i], coords[j] + delta))

    with pair_cache(pair_specs):
        return _gradient_body(atoms, coords, method, step, molecular_charge,
                              scf_result, PARAMS, info, params, starts,
                              n_atoms, shifts)


def _gradient_body(atoms, coords, method, step, molecular_charge, scf_result,
                   PARAMS, info, params, starts, n_atoms, shifts):
    """The gradient itself, run inside the TETCI cache installed above."""
    from .scf import _build_core_hamiltonian, _build_fock

    # `scf_result` lets a batched caller solve every molecule's SCF in one
    # dispatch and hand the converged density in, rather than each gradient
    # re-solving its own.
    result = scf_result if scf_result is not None else nddo_energy(
        atoms, coords, method=method, max_iter=200, conv_tol=1e-8,
        molecular_charge=molecular_charge,
    )
    P = result['density']
    n_basis = info['n_basis']

    H_ref = _build_core_hamiltonian(atoms, coords, info)
    F_ref = _build_fock(H_ref, P, info, atoms, coords)

    # The one-centre Fock block is whatever F is not explained by H and the
    # two-centre pairs. It depends only on P and the atom parameters, so it is
    # the same at every displaced geometry and never needs rebuilding.
    all_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    pair_ref = _pair_terms_many(params, coords, all_pairs, starts, P, n_basis)
    T_ref = np.zeros((n_basis, n_basis))
    for _dH, dT in pair_ref.values():
        T_ref += dT
    G_one = F_ref - H_ref - T_ref

    # Core-core repulsion is a plain sum over pairs, so it gets the same
    # treatment as H and F: keep the reference total and the per-pair terms,
    # and patch only the pairs that move. Rebuilding it in full at each of the
    # 6N displacements was 46% of a menthol gradient.
    E_nuc_ref = nuclear_repulsion_for_method(atoms, coords, PARAMS, method)
    nuc_ref = {(i, j): pair_repulsion_for_method(atoms, coords, i, j, PARAMS, method)
               for i in range(n_atoms) for j in range(i + 1, n_atoms)}

    # E_elec = 0.5 * sum(P * (H + F)) with F = H + G_one + T, so it is
    # 0.5 * sum(P * (2H + G_one + T)). G_one and the one-centre part of H do
    # not move, so a displaced geometry differs from the reference only by the
    # pairs that changed — and each pair's whole contribution is one scalar.
    E_elec_ref = 0.5 * float(np.sum(P * (H_ref + F_ref)))
    pair_energy_ref = _pair_energy_many(params, coords, all_pairs, starts, P,
                                        n_basis)

    # Six batched passes over every pair, rather than 6N passes over the N-1
    # pairs touching one atom. Each pass shifts every pair's second atom by
    # the same delta, which is legitimate because a pair's energy depends only
    # on the vector between its two atoms. 6 * N(N-1)/2 = 3N^2 pair
    # evaluations against the 6N * (N-1) = 6N^2 the per-atom scheme needed,
    # and each pass is one shape-grouped contraction over all 2701 pairs
    # instead of 444 contractions over 73.
    pair_index = {key: k for k, key in enumerate(all_pairs)}
    ia = np.fromiter((i for i, _ in all_pairs), dtype=int, count=len(all_pairs))
    ib = np.fromiter((j for _, j in all_pairs), dtype=int, count=len(all_pairs))

    # Core-core repulsion for every pair in one call rather than one per pair.
    # The scalar dispatcher takes a coords array and two indices, so feeding it
    # a shifted second atom meant copying the whole geometry per pair — 6 *
    # N(N-1)/2 copies. The batch form takes coordinate arrays directly.
    zi = np.asarray([atoms[i] for i, _ in all_pairs])
    zj = np.asarray([atoms[j] for _, j in all_pairs])
    _pm6_core_core = normalize_method(method) in PM6_CORE_CORE_METHODS
    if _pm6_core_core:
        from .pwcct import pm6_pair_repulsion_batch

    def pair_repulsions(coords_i, coords_j):
        if _pm6_core_core:
            return np.asarray(pm6_pair_repulsion_batch(
                zi, zj, None, coords_i, coords_j, param_dict=PARAMS))
        # The two-element atom list matters: passing the full `atoms` with
        # indices 0 and 1 silently gives every pair the parameters of the
        # first two atoms, which leaves the gradient wrong for every method
        # that is not on the PM6 core-core path.
        return np.asarray([
            pair_repulsion_for_method(
                [atoms[i], atoms[j]],
                np.stack([coords_i[k], coords_j[k]]), 0, 1, PARAMS, method)
            for k, (i, j) in enumerate(all_pairs)])

    def pair_energies(delta):
        """Every pair's total energy with its second atom shifted by delta."""
        elec = _pair_energy_many(params, coords, all_pairs, starts, P, n_basis,
                                 shift=delta)
        elec_arr = np.fromiter((elec[key] for key in all_pairs), dtype=float,
                               count=len(all_pairs))
        return 0.5 * elec_arr + pair_repulsions(coords[ia], coords[ib] + delta)

    plus = {}
    minus = {}
    for d, sign, delta in shifts:
        (plus if sign > 0 else minus)[d] = pair_energies(delta)

    gradient = np.zeros((n_atoms, 3))
    for d in range(3):
        # d(E_pair)/d(r_j) by central difference; the pair's dependence on
        # r_i is exactly the negative of it, which is why one difference
        # feeds both atoms with opposite sign.
        deriv = (plus[d] - minus[d]) / (2.0 * step)
        np.add.at(gradient[:, d], ib, deriv)
        np.add.at(gradient[:, d], ia, -deriv)

    return result, gradient


def _energy_frozen_density(atoms, coords, P, PARAMS, method='RM1', molecular_charge: float = 0.0):
    """Total energy with frozen density P at a new geometry, rebuilt in full.

    Kept as the reference the incremental path above is checked against.
    """
    from .scf import _build_basis_info, _build_core_hamiltonian, _build_fock

    info = _build_basis_info(atoms, PARAMS, molecular_charge=molecular_charge)
    H = _build_core_hamiltonian(atoms, coords, info)
    F = _build_fock(H, P, info, atoms, coords)

    E_elec = 0.5 * np.sum(P * (H + F))

    # Nuclear repulsion — PM6 variants use the PWCCT core-core (must match scf.py so the
    # frozen-density gradient is consistent with the energy it differentiates).
    E_nuc = nuclear_repulsion_for_method(atoms, coords, PARAMS, method)

    return E_elec + E_nuc
