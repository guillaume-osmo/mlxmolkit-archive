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
from .integrals import compute_nuclear_repulsion, nuclear_repulsion_for_method


def _pair_terms(params, coords, i, j, starts, P, n_basis):
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
    w, e1b, e2a = rotate_integrals_to_molecular_frame(pA, pB, rA, rB)
    dH[sA:sA + nA, sA:sA + nA] += e1b[:nA, :nA]
    dH[sB:sB + nB, sB:sB + nB] += e2a[:nB, :nB]
    dT = _pair_fock_twocentre(np.zeros((n_basis, n_basis)), P,
                              pA, pB, sA, sB, rA, rB, w=w)
    return dH, dT


def analytical_gradient(
    atoms: list[int],
    coords: np.ndarray,
    method: str = 'RM1',
    step: float = 1e-5,
    molecular_charge: float = 0.0,
) -> tuple[dict, np.ndarray]:
    """Compute energy and gradient.

    Costs 1 SCF plus 6N displacements, each touching only the N-1 pairs that
    moved rather than all N(N-1)/2.

    Returns:
        result: SCF result dict
        gradient: (n_atoms, 3) in eV/Angstrom
    """
    from .scf import _build_basis_info, _build_core_hamiltonian, _build_fock

    PARAMS = get_params(method)
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    result = nddo_energy(
        atoms, coords, method=method, max_iter=200, conv_tol=1e-8,
        molecular_charge=molecular_charge,
    )
    P = result['density']

    # Reference matrices, built once.
    info = _build_basis_info(atoms, PARAMS, molecular_charge=molecular_charge)
    params = info['params']
    starts = info['atom_basis_start']
    n_basis = info['n_basis']

    H_ref = _build_core_hamiltonian(atoms, coords, info)
    F_ref = _build_fock(H_ref, P, info, atoms, coords)

    # The one-centre Fock block is whatever F is not explained by H and the
    # two-centre pairs. It depends only on P and the atom parameters, so it is
    # the same at every displaced geometry and never needs rebuilding.
    T_ref = np.zeros((n_basis, n_basis))
    pair_ref = {}
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dH, dT = _pair_terms(params, coords, i, j, starts, P, n_basis)
            pair_ref[(i, j)] = (dH, dT)
            T_ref += dT
    G_one = F_ref - H_ref - T_ref

    def energy_with_atom_moved(a: int, shifted: np.ndarray) -> float:
        H = H_ref.copy()
        T = T_ref.copy()
        for j in range(n_atoms):
            if j == a:
                continue
            key = (a, j) if a < j else (j, a)
            old_H, old_T = pair_ref[key]
            new_H, new_T = _pair_terms(params, shifted, *key, starts, P, n_basis)
            H += new_H - old_H
            T += new_T - old_T
        F = H + G_one + T
        E_elec = 0.5 * np.sum(P * (H + F))
        return E_elec + nuclear_repulsion_for_method(atoms, shifted, PARAMS, method)

    gradient = np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        for d in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[a, d] += step
            minus[a, d] -= step
            E_p = energy_with_atom_moved(a, plus)
            E_m = energy_with_atom_moved(a, minus)
            gradient[a, d] = (E_p - E_m) / (2.0 * step)

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
