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
from .srb import compute_srb


def _per_atom_lowdin(
    S: np.ndarray, basis: list, atoms: list[int]
) -> np.ndarray:
    """Per-atom Löwdin orthogonalization: build a transformation matrix
    that orthogonalizes each atom's own basis-function block against
    itself, leaving inter-atomic couplings untouched.

    For atoms where multiple shells share the same l (H 2s aux on H
    1s), this brings the same-atom block from singular toward identity
    without touching the inter-atom physics. The result is a (n, n)
    matrix ``T`` such that ``S' = Tᵀ S T`` has same-atom block ≈ I.
    """
    n = S.shape[0]
    T = np.eye(n, dtype=np.float64)
    # Group basis indices by atom.
    atom_groups: dict[int, list[int]] = {}
    for i, bf in enumerate(basis):
        atom_groups.setdefault(bf.atom_idx, []).append(i)
    for at_idx, idxs in atom_groups.items():
        if len(idxs) == 1:
            continue
        block = S[np.ix_(idxs, idxs)]
        # Symmetric Lowdin: T_block = block^(-1/2)
        w, U = np.linalg.eigh(block)
        # Tiny eigenvalue safety (avoids 1/sqrt(0) if perfectly redundant).
        w_safe = np.maximum(w, 1e-12)
        T_block = U @ np.diag(1.0 / np.sqrt(w_safe)) @ U.T
        for ii, gi in enumerate(idxs):
            for jj, gj in enumerate(idxs):
                T[gi, gj] = T_block[ii, jj]
    return T


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


# Reference shell occupations indexed by l (0..2) and atomic number.
# Vendored verbatim from xtb/src/xtb/gfn2.f90:336-365 (the same table is
# used for GFN0/1/2 via setGFN2ReferenceOcc). Only entries we need for
# Phase A0 (CHNOFSCl + a few neighbors) are populated; others default
# to zero. Populating more is mechanical when needed.
_REF_OCC_PER_L: dict[int, tuple[float, float, float]] = {
    1:  (1.0, 0.0,  0.0),  # H
    2:  (2.0, 0.0,  0.0),  # He
    3:  (1.0, 0.0,  0.0),  # Li
    4:  (2.0, 0.0,  0.0),  # Be
    5:  (2.0, 1.0,  0.0),  # B
    6:  (1.0, 3.0,  0.0),  # C
    7:  (1.5, 3.5,  0.0),  # N
    8:  (2.0, 4.0,  0.0),  # O
    9:  (2.0, 5.0,  0.0),  # F
    10: (2.0, 6.0,  0.0),  # Ne
    11: (1.0, 0.0,  0.0),  # Na
    12: (2.0, 0.0,  0.0),  # Mg
    13: (2.0, 1.0,  0.0),  # Al
    14: (1.5, 2.5,  0.0),  # Si
    15: (1.5, 3.5,  0.0),  # P
    16: (2.0, 4.0,  0.0),  # S
    17: (2.0, 5.0,  0.0),  # Cl
    18: (2.0, 6.0,  0.0),  # Ar
    35: (2.0, 5.0,  0.0),  # Br
    53: (2.0, 5.0,  0.0),  # I
}


def _ncore(Z: int) -> int:
    """Number of core (non-valence) electrons. From xtb's
    ``type/molecule.f90:571-595``."""
    if Z <= 2:   return 0
    if Z <= 10:  return 2
    if Z <= 18:  return 10
    if Z <= 29:  return 18
    if Z <= 36:  return 28
    if Z <= 47:  return 36
    if Z <= 54:  return 46
    if Z <= 71:  return 54
    if Z <= 79:  return 68
    if Z <= 86:  return 78
    return 86


def _atomic_reference_energy_eV(atoms_list: list[int]) -> float:
    """Sum over isolated free atoms of (selfEnergy_shell · refOcc_shell).

    Mirrors xtb's ``setzshell`` (scc_core.f90:1509-1542): per atom,
    accumulate the reference occupation from each shell up to the
    atom's valence electron count ``z = at − ncore(at)``; the BARE
    h-level (no CN, q correction) is used since CN=0, q=0 for an
    isolated atom. Returns the total in eV.
    """
    e_eV = 0.0
    for Z in atoms_list:
        Z = int(Z)
        ref_per_l = _REF_OCC_PER_L.get(Z)
        if ref_per_l is None:
            raise KeyError(
                f"No GFN0 reference occupation for Z={Z} — populate "
                f"_REF_OCC_PER_L from xtb/src/xtb/gfn2.f90:336-365."
            )
        z_val = float(Z - _ncore(Z))
        ntot = -1e-6
        for shell in GFN0_PARAMS[Z].shells:
            occ = ref_per_l[shell.l] if shell.l < len(ref_per_l) else 0.0
            ntot += occ
            if ntot > z_val:
                occ = 0.0
            e_eV += shell.h * occ
    return e_eV


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

    # 5. Generalized eigh on the (H, S) pair. With aux shells skipped
    # (see basis.py) S is well-conditioned and standard eigh works.
    from scipy.linalg import eigh as _scipy_eigh
    eigvals, C = _scipy_eigh(H_np, S)
    n_active = len(eigvals)

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

    # Short-range bond correction (Hartree). Only acts on hetero pairs
    # in {B, C, N, O, F}; H-containing systems with no C/N/O/F-C/N/O/F
    # heteros (e.g. H2O alone) yield 0.
    E_srb_hartree = compute_srb(atoms_list, coords, cn)

    # Total: matches xtb's etot = eel + ees + ep + esrb + (ed deferred).
    E_total_hartree = E_band_hartree + E_eeq_hartree + E_rep_hartree + E_srb_hartree
    E_total_eV = E_total_hartree * _EV_PER_HARTREE

    # Atomization energy: E_isolated_atoms − E_total. Closely matches
    # xtb's printed "atomisation" line (peeq_module.f90:612). Reported
    # in the result dict under ``heat_of_formation_*`` for parity with
    # the RM1 result shape — for GFN0 this is *atomization energy*,
    # not the experimental heat of formation (which would need
    # vendored elemental ΔH_f).
    E_atoms_eV = _atomic_reference_energy_eV(atoms_list)
    E_atoms_hartree = E_atoms_eV / _EV_PER_HARTREE
    atomization_hartree = E_atoms_hartree - E_total_hartree
    atomization_eV = atomization_hartree * _EV_PER_HARTREE

    return {
        "energy_eV": E_total_eV,
        "energy_kcal": E_total_eV * _KCAL_PER_EV,
        "energy_hartree": E_total_hartree,
        "electronic_eV": E_band_hartree * _EV_PER_HARTREE,
        "eeq_eV": E_eeq_hartree * _EV_PER_HARTREE,
        "repulsion_eV": E_rep_hartree * _EV_PER_HARTREE,
        "dispersion_eV": None,           # TODO Phase A5 (D4)
        "srb_eV": E_srb_hartree * _EV_PER_HARTREE,
        "heat_of_formation_eV": atomization_eV,
        "heat_of_formation_kcal": atomization_eV * _KCAL_PER_EV,
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


def gfn0_energy_batch(
    molecules: list[tuple[list[int], np.ndarray]],
    charges: list[int] | None = None,
) -> list[dict]:
    """Compute GFN0-xTB energies for ``len(molecules)`` molecules.

    The per-molecule setup (AO basis, overlap, core Hamiltonian, EEQ
    charges, repulsion, SRB) is mol-size-dependent and runs in numpy.
    The generalized eigenproblem ``H C = S C diag(ε)`` is the only
    truly batchable operation in GFN0 (since there's no SCF) and it
    runs as a single batched call to
    :func:`mlx_addons.linalg.gen_eigh`. Padded rows/cols of ``S`` are
    set to identity so the Cholesky stays PD.

    Args:
        molecules: list of ``(atoms, coords_ang)`` tuples. ``atoms`` is
            a list/array of atomic numbers, ``coords_ang`` is the
            ``(n_atoms, 3)`` coordinate array in Angstrom.
        charges: optional list of integer net charges, one per
            molecule. Defaults to all-neutral.

    Returns:
        ``list[dict]`` — one result dict per molecule, same keys as
        :func:`gfn0_energy`.
    """
    import mlx.core as mx
    from mlx_addons.linalg import gen_eigh

    N = len(molecules)
    if N == 0:
        return []
    if charges is None:
        charges = [0] * N

    # ---- Phase 1: per-molecule numpy setup. -----------------------
    per_mol = []  # list of dicts with all the numpy intermediates
    max_basis = 0
    for (atoms, coords_ang), chg in zip(molecules, charges):
        atoms_list = [int(a) for a in atoms]
        coords = np.asarray(coords_ang, dtype=np.float64)

        basis = build_basis(atoms_list, coords)
        S = overlap_matrix(basis)
        n_basis = S.shape[0]
        max_basis = max(max_basis, n_basis)

        cn_mx = coordination_number_erf(
            mx.array(coords.astype(np.float32)),
            mx.array(np.asarray(atoms_list, dtype=np.int32)),
        )
        mx.eval(cn_mx)
        cn = np.asarray(cn_mx).astype(np.float64)

        q_mx, E_eeq_mx = eeq_charges_and_energy(
            mx.array(coords.astype(np.float32)),
            mx.array(np.asarray(atoms_list, dtype=np.int32)),
            total_charge=float(chg),
        )
        mx.eval(q_mx, E_eeq_mx)
        q = np.asarray(q_mx).astype(np.float64)
        E_eeq_hartree = float(E_eeq_mx)

        H = build_hcore(atoms_list, coords, basis, S, cn, q)
        E_rep = compute_repulsion(atoms_list, coords)
        E_srb = compute_srb(atoms_list, coords, cn)
        n_elec = sum(_VAL_ELEC[Z] for Z in atoms_list) - chg
        if n_elec % 2 != 0:
            raise NotImplementedError(
                f"Open-shell systems not supported (n_elec={n_elec})"
            )

        per_mol.append({
            "atoms": atoms_list,
            "S": S,
            "H": H,
            "n_basis": n_basis,
            "cn": cn,
            "q": q,
            "E_eeq": E_eeq_hartree,
            "E_rep": E_rep,
            "E_srb": E_srb,
            "n_elec": n_elec,
            "n_occ": n_elec // 2,
            "charge": chg,
        })

    # ---- Phase 2: pad and run batched generalized eigh ------------
    H_padded = np.zeros((N, max_basis, max_basis), dtype=np.float32)
    S_padded = np.tile(np.eye(max_basis, dtype=np.float32), (N, 1, 1))
    for i, m in enumerate(per_mol):
        nb = m["n_basis"]
        H_padded[i, :nb, :nb] = m["H"].astype(np.float32)
        S_padded[i, :nb, :nb] = m["S"].astype(np.float32)
    H_mx = mx.array(H_padded)
    S_mx = mx.array(S_padded)
    eigvals_all, C_all = gen_eigh(H_mx, S_mx)
    mx.eval(eigvals_all, C_all)
    eigvals_np = np.asarray(eigvals_all)
    C_np = np.asarray(C_all)

    # ---- Phase 3: per-molecule energy assembly --------------------
    results = []
    for i, m in enumerate(per_mol):
        nb = m["n_basis"]
        n_occ = m["n_occ"]
        eigvals = eigvals_np[i, :nb].astype(np.float64)
        C = C_np[i, :nb, :nb].astype(np.float64)
        C_occ = C[:, :n_occ]
        P = 2.0 * (C_occ @ C_occ.T)

        E_band = 2.0 * float(np.sum(eigvals[:n_occ]))
        E_total_h = E_band + m["E_eeq"] + m["E_rep"] + m["E_srb"]
        E_total_eV = E_total_h * _EV_PER_HARTREE

        E_atoms_eV = _atomic_reference_energy_eV(m["atoms"])
        atomization_h = E_atoms_eV / _EV_PER_HARTREE - E_total_h
        atomization_eV = atomization_h * _EV_PER_HARTREE

        results.append({
            "energy_eV": E_total_eV,
            "energy_kcal": E_total_eV * _KCAL_PER_EV,
            "energy_hartree": E_total_h,
            "electronic_eV": E_band * _EV_PER_HARTREE,
            "eeq_eV": m["E_eeq"] * _EV_PER_HARTREE,
            "repulsion_eV": m["E_rep"] * _EV_PER_HARTREE,
            "dispersion_eV": None,
            "srb_eV": m["E_srb"] * _EV_PER_HARTREE,
            "heat_of_formation_eV": atomization_eV,
            "heat_of_formation_kcal": atomization_eV * _KCAL_PER_EV,
            "converged": True,
            "n_iter": 0,
            "eigenvalues": eigvals,
            "density": P,
            "charges": m["q"],
            "coordination_number": m["cn"],
            "n_basis": nb,
            "n_elec": m["n_elec"],
            "n_occ": n_occ,
            "method": "GFN0",
        })
    return results


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
