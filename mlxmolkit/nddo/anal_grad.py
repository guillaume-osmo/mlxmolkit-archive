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



def _pair_energy_many(params, coords, pairs, starts, P, n_basis, shift=None,
                      ws_all=None, S_all=None):
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

    `ws_all` and `S_all` are the rotations and overlaps for exactly `pairs`, in
    order. A caller that computed them itself passes them here rather than
    letting this look each one up: the cache is keyed on coordinate bytes, so
    serving 2701 pairs from it costs 2701 key constructions and 2701 dict gets
    per pass, which was the largest cost in this function — larger than the
    contractions it exists to perform. `ws_all` also arrives as one stacked
    array, so a shape group slices it instead of restacking 2701 rows.
    """
    from .scf import (_pair_resonance_block, _pair_resonance_blocks,
                      _pair_fock_twocentre, _pair_core_attraction)
    from .rotation_batch import rotate_pairs
    from .d_two_center import _OVERLAP_CACHE, _ROT_CACHE, _pair_key

    if shift is None:
        coords_b = coords
    else:
        coords_b = coords + np.asarray(shift, dtype=np.float64)

    sp, dd, sp_pos = [], [], []
    for k, (i, j) in enumerate(pairs):
        if params[i].n_basis <= 4 and params[j].n_basis <= 4:
            sp.append((i, j))
            sp_pos.append(k)
        else:
            dd.append((i, j))

    out: dict[tuple[int, int], float] = {}

    if sp:
        ws = None
        if ws_all is not None:
            ws = ws_all[np.asarray(sp_pos)] if dd else ws_all
        elif _ROT_CACHE is not None:
            ws = [_ROT_CACHE.get(_pair_key(params[i], params[j],
                                           coords[i], coords_b[j])) for i, j in sp]
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

            W = (ws[sel] if isinstance(ws, np.ndarray)
                 else np.stack([ws[k] for k in ks]))[:, :nA, :nA, :nB, :nB]
            P_AA = P[rows_a[:, :, None], rows_a[:, None, :]]
            P_BB = P[rows_b[:, :, None], rows_b[:, None, :]]
            P_AB = P[rows_a[:, :, None], rows_b[:, None, :]]
            P_BA = P[rows_b[:, :, None], rows_a[:, None, :]]

            # The cache holds this group's overlaps whenever the caller
            # declared the geometry, which is the gradient's own path. Fall
            # back to the scalar block otherwise — the cache is an
            # optimisation, never a precondition.
            pA = [params[sp[k][0]] for k in ks]
            pB = [params[sp[k][1]] for k in ks]
            S = None
            if S_all is not None:
                fetched = [S_all[sp_pos[k]] for k in ks]
                if all(s is not None for s in fetched):
                    S = np.stack(fetched)
            elif _OVERLAP_CACHE is not None:
                fetched = [_OVERLAP_CACHE.get(
                    _pair_key(params[sp[k][0]], params[sp[k][1]],
                              coords[sp[k][0]], coords_b[sp[k][1]])) for k in ks]
                if all(s is not None for s in fetched):
                    S = np.stack(fetched)
            if S is not None:
                h_ab = _pair_resonance_blocks(pA, pB, S)
            else:
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
    P_init: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """Compute energy and gradient.

    Costs 1 SCF plus 6N displacements, each touching only the N-1 pairs that
    moved rather than all N(N-1)/2.

    Returns:
        result: SCF result dict
        gradient: (n_atoms, 3) in eV/Angstrom
    """
    from .scf import _build_basis_info
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

    all_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    ref_specs = [(params[i], params[j], coords[i], coords[j])
                 for i, j in all_pairs]

    # An all-sp molecule takes the ordered path: the six displaced geometries
    # are computed here in one batch each and handed to `_pair_energy_many`
    # positionally, so they never enter the cache. The cache is keyed on
    # coordinate bytes, and for 6 * N(N-1)/2 pairs that key is most of the
    # cost — building the dict, then a key and a dict get per pair per pass,
    # to serve a consumer that already knows exactly which row it wants. Only
    # the reference geometry still needs to be looked up by key, because the
    # SCF reaches it through routines that take one pair at a time.
    #
    # A d-bearing molecule keeps declaring everything: its displaced pairs are
    # consumed by `_pair_terms`, which reaches the TETCI, attraction and
    # overlap caches by key from several call sites.
    if any(p.n_basis == 9 for p in params):
        pair_specs = list(ref_specs)
        for _d, _sign, delta in shifts:
            pair_specs.extend((params[i], params[j], coords[i], coords[j] + delta)
                              for i, j in all_pairs)
        with pair_cache(pair_specs):
            return _gradient_body(atoms, coords, method, step, molecular_charge,
                                  scf_result, PARAMS, info, params, starts,
                                  n_atoms, shifts, all_pairs, None, None,
                                  P_init)

    from .overlap_batch import overlap_pairs
    from .rotation_batch import rotate_pairs

    shifted_specs = []
    for _d, _sign, delta in shifts:
        shifted_specs.extend((params[i], params[j], coords[i], coords[j] + delta)
                             for i, j in all_pairs)
    ws_shift = rotate_pairs([(a, b) for a, b, _c, _d in shifted_specs],
                            [(c, d) for _a, _b, c, d in shifted_specs])
    S_shift = overlap_pairs(shifted_specs)

    with pair_cache(ref_specs):
        return _gradient_body(atoms, coords, method, step, molecular_charge,
                              scf_result, PARAMS, info, params, starts,
                              n_atoms, shifts, all_pairs, ws_shift, S_shift,
                              P_init)


def _gradient_body(atoms, coords, method, step, molecular_charge, scf_result,
                   PARAMS, info, params, starts, n_atoms, shifts, all_pairs,
                   ws_shift, S_shift, P_init=None):
    """The gradient itself, run inside the TETCI cache installed above."""
    # `scf_result` lets a batched caller solve every molecule's SCF in one
    # dispatch and hand the converged density in, rather than each gradient
    # re-solving its own.
    result = scf_result if scf_result is not None else nddo_energy(
        atoms, coords, method=method, max_iter=200, conv_tol=1e-8,
        molecular_charge=molecular_charge, P_init=P_init,
    )
    P = result['density']
    n_basis = info['n_basis']

    # Nothing at the reference geometry is needed. A central difference is
    # (E(+d) - E(-d)) / 2d: the reference value cancels. The earlier scheme
    # patched a displaced energy onto a reference total, so it had to build
    # H_ref, F_ref, the one-centre Fock block and a per-pair core-core dict
    # first; the rigid-shift scheme differences the pair terms directly and
    # never refers to any of them. Building them anyway cost 36% of a
    # cholesterol gradient, most of it the 2 * N(N-1)/2 full (n_basis,
    # n_basis) matrices `_pair_terms_many` allocates to be summed into a
    # T_ref that then fed only a dead variable.

    # Six batched passes over every pair, rather than 6N passes over the N-1
    # pairs touching one atom. Each pass shifts every pair's second atom by
    # the same delta, which is legitimate because a pair's energy depends only
    # on the vector between its two atoms. 6 * N(N-1)/2 = 3N^2 pair
    # evaluations against the 6N * (N-1) = 6N^2 the per-atom scheme needed,
    # and each pass is one shape-grouped contraction over all 2701 pairs
    # instead of 444 contractions over 73.
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

    n_pairs = len(all_pairs)

    def pair_energies(g, delta):
        """Every pair's total energy with its second atom shifted by delta.

        `g` indexes the shift, which is also the block of `ws_shift`/`S_shift`
        this geometry occupies — they were built shift-major over `all_pairs`.
        """
        block = slice(g * n_pairs, (g + 1) * n_pairs)
        elec = _pair_energy_many(
            params, coords, all_pairs, starts, P, n_basis, shift=delta,
            ws_all=None if ws_shift is None else ws_shift[block],
            S_all=None if S_shift is None else S_shift[block])
        elec_arr = np.fromiter((elec[key] for key in all_pairs), dtype=float,
                               count=n_pairs)
        return 0.5 * elec_arr + pair_repulsions(coords[ia], coords[ib] + delta)

    plus = {}
    minus = {}
    for g, (d, sign, delta) in enumerate(shifts):
        (plus if sign > 0 else minus)[d] = pair_energies(g, delta)

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
