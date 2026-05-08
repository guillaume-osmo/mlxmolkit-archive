# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN0-xTB single-molecule energy orchestrator.

Wires together the per-phase primitives:

1. :func:`mlxmolkit.xtb.basis.build_basis` + ``overlap_matrix`` — STO-3G
   AO basis and overlap ``S``.
2. :func:`mlxmolkit.xtb.cn.coordination_number_erf` — erf-CN per atom.
3. :func:`mlxmolkit.xtb.eeq.eeq_charges` — EEQ atomic charges ``q``.
4. :func:`mlxmolkit.xtb.hcore.build_hcore` — distance-dependent
   tight-binding core Hamiltonian ``H``.
5. :func:`mlx_addons.linalg.gen_eigh` — generalized symmetric
   eigenproblem ``H C = S C diag(ε)``.
6. Density build, band energy, repulsion, total.

Phase A0 deferred (TODOs flagged in the result dict):
- D4 dispersion (``E_disp``).
- SRB short-range bond correction (``E_SRB``).
- Halogen-bond correction (only for halogen-containing systems).
- Atomic reference energies (``eatoms``) for atomization-energy reporting.

Without those terms, ``E_total`` will deviate from xtb's GFN0 by the sum
of the missing components — typically a few kcal/mol for E_SRB on bonded
systems, a few kcal/mol for D4. Heat-of-formation requires ``eatoms``
which is not yet vendored; we report ``heat_of_formation_eV = None``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .basis import build_basis, basis_summary, overlap_matrix
from .cn import coordination_number_erf
from .eeq import eeq_charges_and_energy
from .hcore import build_hcore
from .params_gfn0 import GFN0_PARAMS
from .repulsion import compute_repulsion


def _canonical_eigh(H: np.ndarray, S: np.ndarray, eig_tol: float = 1e-2):
    """Generalized eigh ``H C = S C diag(w)`` via canonical
    orthogonalization. Diagonalizes ``S`` first, drops basis directions
    with eigenvalue below ``eig_tol`` (which would otherwise produce
    spurious huge eigenvalues), forms ``X = U Λ^{-1/2}`` on the kept
    block, transforms ``H' = Xᵀ H X`` to the orthonormal subspace,
    diagonalizes there, and back-transforms ``C = X C'``.

    This is the standard treatment for AO bases with redundancy /
    near-degeneracy (xtb does the same in its SCF solver).
    """
    s_eig, U = np.linalg.eigh(S)
    keep = s_eig > eig_tol
    s_kept = s_eig[keep]
    U_kept = U[:, keep]
    X = U_kept * (1.0 / np.sqrt(s_kept))[None, :]   # (n, m) with m <= n
    H_prime = X.T @ H @ X
    w, Cprime = np.linalg.eigh(H_prime)
    C = X @ Cprime                                  # (n, m); m occ-able
    return w, C, X.shape[1]


_EV_PER_HARTREE = 27.211386245988
_KCAL_PER_HARTREE = 627.5094740631
_KCAL_PER_EV = _KCAL_PER_HARTREE / _EV_PER_HARTREE


def gfn0_energy(atoms: list[int], coords_ang: np.ndarray, *, charge: int = 0) -> dict:
    """Compute the GFN0-xTB single-point energy for one molecule.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.
        charge: integer net charge.

    Returns:
        Result dict matching the ``rm1_energy`` shape (with ``method``
        set to ``"GFN0"`` and ``converged`` always ``True`` since GFN0
        is non-self-consistent). ``heat_of_formation_*`` are ``None`` for
        Phase A0 because atomic reference energies are not yet vendored.
    """
    atoms_list = [int(a) for a in atoms]
    coords = np.asarray(coords_ang, dtype=np.float64)
    n_atoms = len(atoms_list)

    # 1. AO basis + overlap matrix (numpy single-mol).
    basis = build_basis(atoms_list, coords)
    S = overlap_matrix(basis)
    n_basis = S.shape[0]

    # 2. CN (erf-CN, k=7.5).
    cn_mx = coordination_number_erf(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms_list, dtype=np.int32)),
    )
    mx.eval(cn_mx)
    cn = np.asarray(cn_mx).astype(np.float64)

    # 3. EEQ charges + EEQ Lagrangian energy (xtb's `ees` term).
    q_mx, E_eeq_mx = eeq_charges_and_energy(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms_list, dtype=np.int32)),
        total_charge=float(charge),
    )
    mx.eval(q_mx, E_eeq_mx)
    q = np.asarray(q_mx).astype(np.float64)
    E_eeq_hartree = float(E_eeq_mx)

    # 4. Core Hamiltonian (numpy → mx.array boundary cast).
    H_np = build_hcore(atoms_list, coords, basis, S, cn, q)

    # 5. Generalized eigh via canonical orthogonalization (handles the
    # near-singular S that arises with auxiliary shells included).
    eigvals, C, n_active = _canonical_eigh(H_np, S, eig_tol=1e-2)

    # 6. Closed-shell density: P = 2 * C_occ @ C_occ.T
    n_elec = int(round(sum(GFN0_PARAMS[Z].shells.__len__() == 1 and GFN0_PARAMS[Z].Z or 0 for Z in atoms_list)))
    # Better: count valence electrons directly from per-element data. For
    # GFN0, the valence electrons per element are derivable from the shells
    # but simplest is to use a vendored table. For Phase A0 we use
    # standard valence electron counts.
    n_elec = sum(_VAL_ELEC[Z] for Z in atoms_list) - charge
    n_occ = n_elec // 2
    if n_elec % 2 != 0:
        # Open-shell not supported in this Phase A0 closed-shell path.
        raise NotImplementedError(
            f"Open-shell systems not supported (n_elec={n_elec})"
        )
    C_occ = C[:, :n_occ]
    P = 2.0 * (C_occ @ C_occ.T)

    # 7. Energy components.
    # E_band: closed-shell sum of 2*ε for occupied levels (Hartree).
    E_band_hartree = 2.0 * float(np.sum(eigvals[:n_occ]))

    # Repulsion (Hartree).
    E_rep_hartree = compute_repulsion(atoms_list, coords)

    # Total: matches xtb's etot = eel + ees + ep + (esrb + ed deferred).
    E_total_hartree = E_band_hartree + E_eeq_hartree + E_rep_hartree
    E_total_eV = E_total_hartree * _EV_PER_HARTREE

    return {
        "energy_eV": E_total_eV,
        "energy_kcal": E_total_eV * _KCAL_PER_EV,
        "energy_hartree": E_total_hartree,
        "electronic_eV": E_band_hartree * _EV_PER_HARTREE,
        "eeq_eV": E_eeq_hartree * _EV_PER_HARTREE,
        "repulsion_eV": E_rep_hartree * _EV_PER_HARTREE,
        "dispersion_eV": None,           # TODO Phase A5 (D4)
        "srb_eV": None,                  # TODO Phase A5 (SRB)
        "heat_of_formation_eV": None,    # TODO atomic reference energies
        "heat_of_formation_kcal": None,
        "converged": True,               # GFN0 is non-SCC
        "n_iter": 0,
        "eigenvalues": eigvals,
        "density": P,
        "charges": q,
        "coordination_number": cn,
        "n_basis": n_basis,
        "n_elec": n_elec,
        "n_occ": n_occ,
        "method": "GFN0",
    }


# Standard valence-electron counts for GFN0's elements (covers what's
# accessible without the per-element shell-electron table). For elements
# beyond H..Ar this would need refinement — for Phase A0 (CHNO + a few
# more) it's correct.
_VAL_ELEC: dict[int, int] = {
    1: 1, 2: 2,
    3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
    35: 7,    # Br
    53: 7,    # I
}
