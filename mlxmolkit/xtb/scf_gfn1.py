# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN1-xTB self-consistent charge SCF + total energy.

Closely mirrors xtb's implementation:
    H0     — hcore_gfn1.build_hcore_gfn1
    Coulomb (γ matrix) — Klopman-Ohno with harmonic-average hardness,
                         exponent gExp=2 (gfn1.f90:617 alphaj=2.0).
    Third order — atomic-resolved Hubbard derivative (thirdOrder
                  shell-resolved is *false* for GFN1).

Iterative loop (mirrors scf_module.F90):
    1. Initial qsh, qat: from EEQ atomic charges split by reference
       shell occupations.
    2. shellShift[ish] = Σ_jsh jmat[ish,jsh] · qsh[jsh].
       atomicShift[iat] += qat[iat]² · Γ³_iat.
    3. F_μν (eV) = H0_μν − S_μν · ½ · (V_ish + V_jsh) · autoev,
       with V_ish = shellShift[ish_μ] + atomicShift[iat_μ].
    4. eigh(F, S) → C, ε.
    5. P = 2 · C_occ · C_occᵀ.
    6. Mulliken: qsh[ish] = z_ref[ish] − Σ_μ∈ish (P S)_μμ.
    7. Mix qsh ← α·qsh_new + (1−α)·qsh_old; check convergence.

Skipped in this MVP: D3 dispersion, halogen-bond correction, halogen
shifts. Repulsion uses GFN0's classical pairwise form with GFN1
parameters — exactly what xtb does (gfn1.f90:584-591 wraps the same
TRepulsionData; the only difference vs GFN0 is the per-element
``rep_alpha`` / ``rep_zeff`` values).

Phase B0 scope: numpy single-molecule. Mirror the GFN0 pattern
(orchestrator builds everything once per call, no batched API yet).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .basis import (
    build_basis,
    overlap_matrix,
    sao_basis_metadata,
    BasisFunction,
)
from .cn import coordination_number_erf
from .dispersion_d3 import d3bj_dispersion_gfn1
from .eeq import eeq_charges_and_energy
from .halogen_bond import halogen_bond_energy
from .hcore_gfn1 import build_hcore_gfn1
from .params_gfn1 import GFN1_GLOBALS, GFN1_PARAMS, GFN1Shell
from .sto_ng import gfn1_n_gauss


_HARTREE_PER_EV = 1.0 / 27.211386245988
_EV_PER_HARTREE = 27.211386245988
_KCAL_PER_HARTREE = 627.5094740631
_KCAL_PER_EV = _KCAL_PER_HARTREE / _EV_PER_HARTREE
_ANG_TO_BOHR = 1.8897259886


# Reference occupations per (l, Z), from gfn1.f90:272-302. Same table
# layout as gfn2 (referenceOcc(0:2, Z)).
from .params_gfn1 import _GFN1_REFERENCEOCC


def _ncore(Z: int) -> int:
    """Mirror of energy._ncore — number of core electrons."""
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


def _build_shell_layout(
    atoms: list[int],
    basis: list[BasisFunction],
):
    """Build the atom→shell→BF index tables we need.

    Returns:
        bf_shells:   list of GFN1Shell, one per BF (length n_basis).
        bf_to_shell: int array shape (n_basis,) mapping BF index to a
                     unique shell index in [0, n_shell).
        shell_atom:  int array shape (n_shell,) — atom index per shell.
        shell_l:     int array shape (n_shell,) — l per shell.
        shell_zref:  float array shape (n_shell,) — reference occupation
                     per shell (with the cumulative-z cap from
                     scc_core.f90:setzshell).
        shell_hardness: float array shape (n_shell,) — per-shell
                        hardness η for the Coulomb matrix.
    """
    bf_shells: list[GFN1Shell] = []
    bf_to_shell = np.zeros(len(basis), dtype=np.int64)
    shell_atom: list[int] = []
    shell_l: list[int] = []
    shell_zref_per_shell: list[float] = []
    shell_hard: list[float] = []

    cursor = 0
    sh_idx = 0
    for at_idx, Z in enumerate(atoms):
        p = GFN1_PARAMS[int(Z)]
        z_val = float(int(Z) - _ncore(int(Z)))
        ntot = -1.0e-6
        for shell in p.shells:
            if shell.l > 2:
                # f shells deferred — nothing in GFN1/GFN2 needs them
                continue
            # n_components in *the basis we're indexing* (SAO):
            #   l=0 → 1 (s)
            #   l=1 → 3 (px, py, pz)
            #   l=2 → 5 (pure d: x²-y², z², xy, xz, yz — after dtrf2)
            if shell.l == 0:
                n_components = 1
            elif shell.l == 1:
                n_components = 3
            else:
                n_components = 5
            occ = float(_GFN1_REFERENCEOCC[int(Z)][shell.l])
            ntot += occ
            if ntot > z_val:
                occ = 0.0
            shell_atom.append(at_idx)
            shell_l.append(shell.l)
            shell_zref_per_shell.append(occ)
            sh_hard = p.chemical_hardness * (1.0 + shell.shell_hardness)
            shell_hard.append(sh_hard)
            for _ in range(n_components):
                bf_shells.append(shell)
                bf_to_shell[cursor] = sh_idx
                cursor += 1
            sh_idx += 1
    return (
        bf_shells,
        bf_to_shell,
        np.asarray(shell_atom, dtype=np.int64),
        np.asarray(shell_l, dtype=np.int64),
        np.asarray(shell_zref_per_shell, dtype=np.float64),
        np.asarray(shell_hard, dtype=np.float64),
    )


def _coulomb_matrix(
    coords_bohr: np.ndarray,
    shell_atom: np.ndarray,
    shell_hardness: np.ndarray,
    g_exp: float = 2.0,
) -> np.ndarray:
    """Klopman-Ohno γ(R, η) Coulomb matrix with harmonic-average hardness.

    For inter-atomic (i, j): jmat[i,j] = 1/(R^k + (1/γ_ij)^k)^(1/k)
    where γ_ij = harmonic_avg(η_i, η_j) and k = ``g_exp``.

    For same-atom inter-shell: jmat[i,j] = γ_ij (no R dependence).

    For same-atom diagonal: jmat[i,i] = η_i.
    """
    n_sh = len(shell_atom)
    jmat = np.zeros((n_sh, n_sh), dtype=np.float64)
    for i in range(n_sh):
        ai = shell_atom[i]
        gi = shell_hardness[i]
        jmat[i, i] = gi
        for j in range(i):
            aj = shell_atom[j]
            gj = shell_hardness[j]
            gij = 2.0 / (1.0 / gi + 1.0 / gj)   # harmonic average
            if ai == aj:
                jmat[i, j] = gij
            else:
                R = float(np.linalg.norm(coords_bohr[ai] - coords_bohr[aj]))
                rterm = 1.0 / (R ** g_exp + gij ** (-g_exp)) ** (1.0 / g_exp)
                jmat[i, j] = rterm
            jmat[j, i] = jmat[i, j]
    return jmat


def _cao_bf_shells(
    atoms: list[int],
    cao_basis: list[BasisFunction],
) -> list[GFN1Shell]:
    """Return the per-CAO-BF :class:`GFN1Shell` tag list.

    ``build_hcore_gfn1`` was originally written for an SAO-shaped basis;
    we still need the per-BF shell tag for the CAO basis so the diagonal
    self-energy and the off-diagonal K_AB (h0scal) lookups are correct
    in the CAO matrix elements. The transform to SAO is applied
    afterwards.
    """
    out: list[GFN1Shell] = []
    cursor = 0
    for at_idx, Z in enumerate(atoms):
        p = GFN1_PARAMS[int(Z)]
        for shell in p.shells:
            if shell.l > 2:
                continue
            n_comp = 1 if shell.l == 0 else (3 if shell.l == 1 else 6)
            for _ in range(n_comp):
                # Sanity check: this BF's shell really points at this atom.
                assert cao_basis[cursor].atom_idx == at_idx
                out.append(shell)
                cursor += 1
    assert cursor == len(cao_basis), (
        f"_cao_bf_shells exhausted {cursor} BFs, expected {len(cao_basis)}"
    )
    return out


def _pulay_diis_numpy(
    F_hist: list[np.ndarray],
    e_hist: list[np.ndarray],
) -> np.ndarray:
    """Numpy mirror of :func:`mlx_addons.solvers.pulay_diis` for a single
    Fock matrix at a time.

    Solves the augmented Pulay system

        [B  -1] [c]   [0 ]
        [-1  0] [λ] = [-1]

    with ``B[i,j] = <e_i, e_j>_F`` (Frobenius inner product), and
    returns ``F_extrap = Σ c_i F_i``. The same shape and convention as
    the MLX version on the feature/scf-primitives branch — kept in
    numpy here because the GFN1 SCF loop currently runs single-mol on
    the CPU (no batched GPU hot path yet).
    """
    nd = len(F_hist)
    if nd < 2:
        return F_hist[-1]
    B = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        for j in range(i, nd):
            B[i, j] = float(np.sum(e_hist[i] * e_hist[j]))
            B[j, i] = B[i, j]
    A = np.zeros((nd + 1, nd + 1), dtype=np.float64)
    A[:nd, :nd] = B
    A[:nd, nd] = -1.0
    A[nd, :nd] = -1.0
    rhs = np.zeros(nd + 1, dtype=np.float64)
    rhs[nd] = -1.0
    try:
        coeffs = np.linalg.solve(A, rhs)[:nd]
    except np.linalg.LinAlgError:
        # singular — fall back to the latest Fock unmixed
        return F_hist[-1]
    F_extrap = np.zeros_like(F_hist[-1])
    for i, c in enumerate(coeffs):
        F_extrap += c * F_hist[i]
    return F_extrap


def _mulliken_shell_charges(
    P: np.ndarray, S: np.ndarray, bf_to_shell: np.ndarray, n_shell: int,
    z_ref: np.ndarray,
) -> np.ndarray:
    """qsh[ish] = z_ref[ish] − Σ_μ∈ish (P · S)_μμ. xtb sign convention:
    positive qsh means electron deficiency on the shell.
    """
    PS = P @ S
    pop = np.zeros(n_shell, dtype=np.float64)
    for mu in range(P.shape[0]):
        pop[bf_to_shell[mu]] += PS[mu, mu]
    return z_ref - pop


def gfn1_energy(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    mix: float = 0.4,   # kept for API stability — DIIS supersedes
    verbose: bool = False,
) -> dict:
    """Compute the GFN1-xTB single-point energy for one molecule.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.
        charge: integer net charge.
        max_iter: max SCF iterations.
        conv_tol: max abs change in shell charges between iterations.
        mix: linear-mixing fraction for the new charges (0<mix≤1).
        verbose: print per-iteration convergence info.

    Returns:
        Result dict matching the GFN0 shape, with ``method='GFN1'``.
        Halogen-bond and D3 contributions are ``None`` for now (TODO).
    """
    atoms_list = [int(a) for a in atoms]
    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * _ANG_TO_BOHR
    n_atoms = len(atoms_list)

    # 1. Basis (Cartesian, with 6 components per d-shell). Build the
    # 6×6 d-blocks in CAO and apply xtb's dtrf2 (CAO→SAO) reduction
    # *after* building S/H0, so SCF runs in the 5-pure-d SAO basis.
    cao_basis = build_basis(
        atoms_list, coords, params_dict=GFN1_PARAMS, n_gauss_fn=gfn1_n_gauss,
    )
    S_cao = overlap_matrix(cao_basis)

    # 2. CAO→SAO transform (drops the spurious xx+yy+zz s-component
    # from each d-shell). For systems with no d-shells, T = I.
    sao_basis, T = sao_basis_metadata(cao_basis)
    n_basis = T.shape[0]

    # 3. Shell layout (in SAO terms — 1/3/5 BFs per s/p/d shell).
    (bf_shells, bf_to_shell, shell_atom, shell_l, z_ref, shell_hard) = (
        _build_shell_layout(atoms_list, sao_basis)
    )
    n_shell = len(shell_atom)

    # 4. CN (same erf-CN as GFN0 — k=7.5).
    cn_mx = coordination_number_erf(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms_list, dtype=np.int32)),
    )
    mx.eval(cn_mx)
    cn = np.asarray(cn_mx).astype(np.float64)

    # 5. H0 build — xtb-faithful (hamiltonian.F90:307-369):
    #    • SAO diagonal H0[μ, μ] = selfE_sao[μ] is set DIRECTLY, NOT
    #      via the T·H0_cao·T^T transform (otherwise the d-shell
    #      diagonal gets the wrong scaling because S_cao d-block has
    #      off-diag 1/3 between xx/yy/zz that bleeds into the SAO
    #      diagonal).
    #    • Off-diagonal H0[μ, ν] for μ on atom A ≠ B for ν is built in
    #      CAO and projected via T (= dtrf2 in xtb).
    cao_bf_shells = _cao_bf_shells(atoms_list, cao_basis)
    H0_cao_offdiag, selfE_eV_cao = build_hcore_gfn1(
        atoms_list, coords, cao_basis, S_cao, cn, cao_bf_shells,
    )
    # Strip the CAO diagonal — we only want inter-atomic off-diagonal
    # contributions to be T-projected.
    np.fill_diagonal(H0_cao_offdiag, 0.0)

    S = T @ S_cao @ T.T
    H0 = T @ H0_cao_offdiag @ T.T
    # Build per-SAO-BF self-energy by mapping back through the SAO
    # basis: each SAO BF inherits its underlying CAO shell's selfE.
    # (For s/p, T is identity, so selfE_sao[μ] = selfE_cao[μ]. For d,
    # all 5 SAO BFs of one d-shell share the same shell selfE — we
    # pick any one of the 6 CAO BFs in that shell.)
    selfE_eV = np.zeros(n_basis, dtype=np.float64)
    for mu_sao, b in enumerate(sao_basis):
        # Find any CAO BF with matching shell_id (or atom + same l_total
        # for the s/p case where shell_id matches).
        for mu_cao, bc in enumerate(cao_basis):
            if bc.shell_id == b.shell_id:
                selfE_eV[mu_sao] = selfE_eV_cao[mu_cao]
                break
    # Add the diagonal (selfE in Hartree) on the SAO side.
    H0 = H0 + np.diag(selfE_eV * _HARTREE_PER_EV)

    # 5. Coulomb matrix γ(R, η).
    jmat = _coulomb_matrix(coords_bohr, shell_atom, shell_hard, g_exp=GFN1_GLOBALS.alphaj)

    # 6. Initial qsh from EEQ — split atom charge q_A into shells using
    # reference occupations as weights.
    q_mx, _ = eeq_charges_and_energy(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms_list, dtype=np.int32)),
        total_charge=float(charge),
    )
    mx.eval(q_mx)
    q_at_init = np.asarray(q_mx).astype(np.float64)
    qsh = np.zeros(n_shell, dtype=np.float64)
    # split atom charge across its shells weighted by reference occupations
    # so that Σ_ish qsh = q_at (consistent with z_ref Mulliken convention).
    for ish in range(n_shell):
        a = shell_atom[ish]
        z_atom_ref = sum(z_ref[k] for k in range(n_shell) if shell_atom[k] == a)
        if z_atom_ref > 1e-9:
            qsh[ish] = q_at_init[a] * z_ref[ish] / z_atom_ref

    # 7. SCF iteration — Pulay DIIS on the Fock matrix (replaces the
    #    earlier linear-mixing on qsh). Falls back to linear mixing on
    #    the first iteration when no history exists.
    P = np.zeros((n_basis, n_basis), dtype=np.float64)
    converged = False
    n_iter = 0
    last_qsh = qsh.copy()
    n_elec = sum(int(z_ref[ish]) for ish in range(n_shell)) - int(charge)
    if n_elec % 2 != 0:
        raise NotImplementedError(
            f"Open-shell GFN1 not supported (n_elec={n_elec})"
        )
    n_occ = n_elec // 2

    # SCF iteration with Pulay DIIS on the Fock matrix.
    #
    # Strategy:
    #   - iters 0..diis_warmup-1: linear mixing on qsh (build history)
    #   - iters >= diis_warmup:   DIIS extrapolation of F using the
    #                              generalized commutator e = F P S − S P F
    #
    # The early linear-mixing phase ensures we have ≥ 2 *non-trivial*
    # error vectors before DIIS kicks in (e at iter 0/1 is degenerate
    # because P is built from a not-yet-self-consistent F).
    F_hist: list[np.ndarray] = []
    e_hist: list[np.ndarray] = []
    diis_max = 6
    diis_warmup = 3

    for it in range(max_iter):
        shell_shift = jmat @ qsh
        q_at = np.zeros(n_atoms)
        for ish in range(n_shell):
            q_at[shell_atom[ish]] += qsh[ish]
        atom_shift = np.zeros(n_atoms)
        for a in range(n_atoms):
            atom_shift[a] = q_at[a] ** 2 * GFN1_PARAMS[atoms_list[a]].third_order
        V_sh = shell_shift + atom_shift[shell_atom]
        V_bf = V_sh[bf_to_shell]
        F = H0 - 0.5 * (V_bf[:, None] + V_bf[None, :]) * S

        if it >= diis_warmup:
            e = F @ P @ S - S @ P @ F
            F_hist.append(F)
            e_hist.append(e)
            if len(F_hist) > diis_max:
                F_hist.pop(0); e_hist.pop(0)
            if len(F_hist) >= 2:
                F_use = _pulay_diis_numpy(F_hist, e_hist)
            else:
                F_use = F
        else:
            F_use = F

        from scipy.linalg import eigh as _scipy_eigh
        try:
            eigvals_h, C = _scipy_eigh(F_use, S)
        except Exception:
            s_eig, U = np.linalg.eigh(S)
            keep = s_eig > 1e-3
            X = U[:, keep] * (1.0 / np.sqrt(s_eig[keep]))[None, :]
            F_p = X.T @ F_use @ X
            w_p, Cp = np.linalg.eigh(F_p)
            eigvals_h = w_p
            C = X @ Cp

        nb_occ = min(n_occ, C.shape[1])
        C_occ = C[:, :nb_occ]
        P = 2.0 * (C_occ @ C_occ.T)

        qsh_new = _mulliken_shell_charges(P, S, bf_to_shell, n_shell, z_ref)
        dq = float(np.max(np.abs(qsh_new - last_qsh)))
        if verbose:
            tag = f"DIIS hist={len(F_hist)}" if it >= diis_warmup else "linear"
            print(f"  iter {it+1:3d}: dq={dq:.2e}  ({tag})")
        if dq < conv_tol:
            qsh = qsh_new
            converged = True
            n_iter = it + 1
            break
        # During linear-mixing warmup: damp the new charges; once DIIS
        # is on, just use the un-mixed Mulliken result.
        if it < diis_warmup:
            qsh = mix * qsh_new + (1.0 - mix) * qsh
        else:
            qsh = qsh_new
        last_qsh = qsh.copy()

    if not converged:
        n_iter = max_iter

    # Final F build for energy
    shell_shift = jmat @ qsh
    q_at = np.zeros(n_atoms)
    for ish in range(n_shell):
        q_at[shell_atom[ish]] += qsh[ish]
    atom_shift = np.zeros(n_atoms)
    for a in range(n_atoms):
        atom_shift[a] = q_at[a] ** 2 * GFN1_PARAMS[atoms_list[a]].third_order
    V_sh = shell_shift + atom_shift[shell_atom]
    V_bf = V_sh[bf_to_shell]
    F = H0 - 0.5 * (V_bf[:, None] + V_bf[None, :]) * S
    from scipy.linalg import eigh as _scipy_eigh
    eigvals_h, C = _scipy_eigh(F, S)
    C_occ = C[:, :n_occ]
    P = 2.0 * (C_occ @ C_occ.T)

    # Energy decomposition (all Hartree).
    # CORRECT formula (verified empirically against xtb on H2O):
    #     E_elec = Σ_uv P_uv · H0_uv  +  ½ qsh·J·qsh  +  ⅓ Σ qat³·Γ
    # which is the Pulay-style decomposition of the SCF energy. The
    # 2·Σε form (E_band) equals Σ P·F = Σ P·H0 + Σ P·Δ where Δ is the
    # SCF correction; the relationship between Σ P·Δ and the Coulomb
    # double-counting is not a simple −E_es as one might guess from
    # standard HF (because the Mulliken split into shell charges has
    # an extra z·V offset). Direct trace(P·H0) avoids that subtlety.
    PH0_h = float(np.sum(P * H0))
    E_es_h = 0.5 * float(qsh @ (jmat @ qsh))
    E_3rd_h = 0.0
    for a in range(n_atoms):
        E_3rd_h += q_at[a] ** 3 * GFN1_PARAMS[atoms_list[a]].third_order / 3.0
    E_band_h = 2.0 * float(np.sum(eigvals_h[:n_occ]))   # diagnostic only
    # Repulsion (re-uses GFN0's repulsion module with GFN1 params).
    E_rep_h = _gfn1_repulsion(atoms_list, coords)

    # Atomic reference energy (sum over atoms of selfEnergy · refOcc on
    # isolated atoms — mirrors scc_core.f90:setzshell).
    E_atoms_eV = 0.0
    for Z in atoms_list:
        z_val = float(int(Z) - _ncore(int(Z)))
        ntot = -1e-6
        for shell in GFN1_PARAMS[Z].shells:
            occ = float(_GFN1_REFERENCEOCC[int(Z)][shell.l])
            ntot += occ
            if ntot > z_val:
                occ = 0.0
            E_atoms_eV += shell.h * occ
    E_atoms_h = E_atoms_eV * _HARTREE_PER_EV

    # Total: H0 contribution from band + Coulomb double-count subtraction.
    # In xtb: E_elec = E_band - 0.5 qsh·shellShift - eThird ...
    # but their sign convention: E_total = E_band - E_es + E_3rd (3rd is added
    # because it's the q³·Γ/3 = atomicShift_energy).
    # Matches scf_module.F90:891 form via algebraic equivalence.
    # eel ≡ Σ P·H0 + ½ qsh·J·qsh + ⅓ qat³·Γ. xtb prints H0 + ees + e3 here.
    # D3(BJ) dispersion (uses the same erf-CN we built above)
    E_d3_h = d3bj_dispersion_gfn1(atoms_list, coords, cn=cn)
    # GFN1 halogen-bond correction (zero unless Br/I/At present with N/O/P/S)
    E_xb_h = halogen_bond_energy(atoms_list, coords)
    E_total_h = PH0_h + E_es_h + E_3rd_h + E_rep_h + E_d3_h + E_xb_h
    atomization_h = E_atoms_h - E_total_h

    return {
        "energy_hartree": E_total_h,
        "energy_eV": E_total_h * _EV_PER_HARTREE,
        "energy_kcal": E_total_h * _KCAL_PER_HARTREE,
        "electronic_eV": (PH0_h + E_es_h + E_3rd_h) * _EV_PER_HARTREE,
        "eeq_eV": None,                    # GFN1 absorbs ES into SCF
        "repulsion_eV": E_rep_h * _EV_PER_HARTREE,
        "dispersion_eV": E_d3_h * _EV_PER_HARTREE,
        "halogen_bond_eV": E_xb_h * _EV_PER_HARTREE,
        "third_order_eV": E_3rd_h * _EV_PER_HARTREE,
        "heat_of_formation_eV": atomization_h * _EV_PER_HARTREE,
        "heat_of_formation_kcal": atomization_h * _KCAL_PER_HARTREE,
        "converged": converged,
        "n_iter": n_iter,
        "eigenvalues": eigvals_h,
        "density": P,
        "shell_charges": qsh,
        "atom_charges": q_at,
        "coordination_number": cn,
        "n_basis": n_basis,
        "n_elec": n_elec,
        "n_occ": n_occ,
        "method": "GFN1",
    }


def _gfn1_repulsion(atoms: list[int], coords_ang: np.ndarray) -> float:
    """GFN1 classical pairwise repulsion. Same functional form as GFN0
    but with GFN1's per-element ``rep_alpha`` / ``rep_zeff`` and the
    GFN1 globals (``kexp = 1.5`` same as GFN0; no light/heavy split).
    """
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n = len(atoms)
    if n < 2:
        return 0.0
    kexp = 1.5
    e = 0.0
    for i in range(n - 1):
        Zi = atoms[i]
        pi = GFN1_PARAMS[Zi]
        for j in range(i + 1, n):
            Zj = atoms[j]
            pj = GFN1_PARAMS[Zj]
            R = float(np.linalg.norm(coords[i] - coords[j]))
            alpha = float(np.sqrt(pi.rep_alpha * pj.rep_alpha))
            zeff = pi.rep_zeff * pj.rep_zeff
            e += zeff / R * float(np.exp(-alpha * R ** kexp))
    return float(e)
