# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2-xTB self-consistent charge SCF + total energy (Phase C0).

This is the **monopole-only** GFN2 — it has all the GFN2 parameter
differences vs GFN1 wired in (wExp=0.5, ksd/kpd, sign-flipped enshell,
shell-resolved third-order via gam3shell, GFN2 repulsion light/heavy
split), but does NOT yet include the AES (anisotropic electrostatics)
multipole machinery (dipole and quadrupole AO integrals). Full GFN2
parity vs tblite requires AES — that lives in :mod:`scf_gfn2_aes`
(Phase C1, TODO).

Pieces relative to scf_gfn1.gfn1_energy:
    H0        ← :mod:`hcore_gfn2`
    Coulomb γ ← Klopman-Ohno harmonic-average hardness, gExp=2 (same form)
    3rd-order ← shell-resolved: ``Γ_l(Z) = thirdOrderAtom[Z] · gam3shell[l]``
    Repulsion ← GFN2 form: kExpLight=1.0 if min(Z_i, Z_j) ≤ 2 else
                kExp=1.5 (light/heavy split); same ``Z_i Z_j / R · exp``
                form as GFN0/1.
    Dispersion ← D4 (Caldeweyher 2019), see :mod:`dispersion_d4`. Uses
                 charge-dependent C6 derived from EEQ charges.
    Halogen   ← *not* part of GFN2 (xtb only enables xbpot for GFN1).
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
from .eeq import eeq_charges_and_energy
from .hcore_gfn2 import build_hcore_gfn2
from .params_gfn2 import (
    GFN2_GLOBALS,
    GFN2_PARAMS,
    GFN2Shell,
    _GFN2_GAM3SHELL,
    _GFN2_REFERENCEOCC,
)
from .sto_ng import gfn2_n_gauss

_HARTREE_PER_EV = 1.0 / 27.211386245988
_EV_PER_HARTREE = 27.211386245988
_KCAL_PER_HARTREE = 627.5094740631
_ANG_TO_BOHR = 1.8897259886


def _ncore(Z: int) -> int:
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


def _build_shell_layout(atoms, basis):
    """Per-shell index tables (atom, l, z_ref, hardness, third-order).

    GFN2 shell hardness uses the same multiplicative form as GFN1
    (``η_l(Z) = chemicalHardness · (1 + shellHardness[l])``).
    GFN2 third-order is **shell-resolved**:
    ``Γ_l(Z) = thirdOrderAtom[Z] · gam3shell[kind, l]``.
    """
    bf_shells: list[GFN2Shell] = []
    bf_to_shell = np.zeros(len(basis), dtype=np.int64)
    shell_atom: list[int] = []
    shell_l: list[int] = []
    shell_zref_per_shell: list[float] = []
    shell_hard: list[float] = []
    shell_3rd: list[float] = []

    cursor = 0
    sh_idx = 0
    for at_idx, Z in enumerate(atoms):
        p = GFN2_PARAMS[int(Z)]
        for shell in p.shells:
            if shell.l > 2:
                continue
            n_components = (1, 3, 5)[shell.l]
            # GFN2: refOcc is per-l (already fractional and pre-clipped
            # to a sensible occupation in the param file — no z_val
            # cap needed here).
            occ = float(_GFN2_REFERENCEOCC[int(Z)][shell.l])
            shell_atom.append(at_idx)
            shell_l.append(shell.l)
            shell_zref_per_shell.append(occ)
            sh_hard = p.chemical_hardness * (1.0 + shell.shell_hardness)
            shell_hard.append(sh_hard)
            # Shell-resolved 3rd order (gam3shell currently identical
            # for kind=1 and kind=2; we read the row by kind anyway for
            # future-proofing if xtb upstream diverges them).
            gam3 = _GFN2_GAM3SHELL[p.kind - 1][shell.l]
            shell_3rd.append(p.third_order * gam3)
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
        np.asarray(shell_3rd, dtype=np.float64),
    )


def _coulomb_matrix(coords_bohr, shell_atom, shell_hardness, g_exp=2.0):
    """Klopman-Ohno γ(R, η) Coulomb matrix — same form as GFN1."""
    n_sh = len(shell_atom)
    jmat = np.zeros((n_sh, n_sh), dtype=np.float64)
    for i in range(n_sh):
        ai = shell_atom[i]
        gi = shell_hardness[i]
        jmat[i, i] = gi
        for j in range(i):
            aj = shell_atom[j]
            gj = shell_hardness[j]
            gij = 2.0 / (1.0 / gi + 1.0 / gj)
            if ai == aj:
                jmat[i, j] = gij
            else:
                R = float(np.linalg.norm(coords_bohr[ai] - coords_bohr[aj]))
                rterm = 1.0 / (R ** g_exp + gij ** (-g_exp)) ** (1.0 / g_exp)
                jmat[i, j] = rterm
            jmat[j, i] = jmat[i, j]
    return jmat


def _mulliken_shell_charges(P, S, bf_to_shell, n_shell, z_ref):
    PS = P @ S
    pop = np.zeros(n_shell, dtype=np.float64)
    for mu in range(P.shape[0]):
        pop[bf_to_shell[mu]] += PS[mu, mu]
    return z_ref - pop


def _pulay_diis_numpy(F_hist, e_hist):
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
        return F_hist[-1]
    F_extrap = np.zeros_like(F_hist[-1])
    for i, c in enumerate(coeffs):
        F_extrap += c * F_hist[i]
    return F_extrap


def _cao_bf_shells(atoms, cao_basis):
    out: list[GFN2Shell] = []
    cursor = 0
    for at_idx, Z in enumerate(atoms):
        p = GFN2_PARAMS[int(Z)]
        for shell in p.shells:
            if shell.l > 2:
                continue
            n_comp = 1 if shell.l == 0 else (3 if shell.l == 1 else 6)
            for _ in range(n_comp):
                assert cao_basis[cursor].atom_idx == at_idx
                out.append(shell)
                cursor += 1
    assert cursor == len(cao_basis)
    return out


def _gfn2_repulsion(atoms, coords_ang) -> float:
    """GFN2 classical pairwise repulsion (gfn2.f90:88-117 + repulsion.f90).

    Form (same as GFN0/GFN1 family):
        E_rep = Σ_{i<j} (Z*_i Z*_j / R_ij^rExp)
                       · exp(-sqrt(α_i α_j) · R_ij^kExp_eff)

    with ``kExp_eff = kExp = 1.5`` if both atoms have Z > 2, else
    ``kExpLight = 1.0`` (gfn2.f90:88-92 — the light-element override).
    rExp = 1.0 (just R, no extra power).
    """
    g = GFN2_GLOBALS
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n = len(atoms)
    if n < 2:
        return 0.0
    e = 0.0
    for i in range(n - 1):
        Zi = atoms[i]
        pi = GFN2_PARAMS[Zi]
        for j in range(i + 1, n):
            Zj = atoms[j]
            pj = GFN2_PARAMS[Zj]
            R = float(np.linalg.norm(coords[i] - coords[j]))
            kexp = g.kexp_light if min(Zi, Zj) <= 2 else g.kexp
            alpha = float(np.sqrt(pi.rep_alpha * pj.rep_alpha))
            zeff = pi.rep_zeff * pj.rep_zeff
            e += zeff / (R ** g.rexp) * float(np.exp(-alpha * R ** kexp))
    return float(e)


def gfn2_energy(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    mix: float = 0.4,
    verbose: bool = False,
    use_d4: bool = True,
) -> dict:
    """GFN2-xTB single-point energy (monopole-only, no AES yet).

    Args:
        atoms, coords_ang, charge, max_iter, conv_tol, mix, verbose:
            same shape as :func:`scf_gfn1.gfn1_energy`.
        use_d4: include D4 dispersion (default True). D4 needs EEQ
            charges, which we already compute as the SCF initial
            guess.

    Returns:
        Dict matching the GFN1 result shape, with ``method='GFN2'``.
        ``aes_eV`` is None (placeholder until Phase C1).
    """
    atoms_list = [int(a) for a in atoms]
    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * _ANG_TO_BOHR
    n_atoms = len(atoms_list)

    # 1. Basis (CAO) + S; SAO transform.
    cao_basis = build_basis(
        atoms_list, coords, params_dict=GFN2_PARAMS, n_gauss_fn=gfn2_n_gauss,
    )
    S_cao = overlap_matrix(cao_basis)
    sao_basis, T = sao_basis_metadata(cao_basis)
    n_basis = T.shape[0]

    # 2. Shell layout.
    (bf_shells, bf_to_shell, shell_atom, shell_l, z_ref,
     shell_hard, shell_third) = _build_shell_layout(atoms_list, sao_basis)
    n_shell = len(shell_atom)

    # 3. CN.
    cn_mx = coordination_number_erf(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms_list, dtype=np.int32)),
    )
    mx.eval(cn_mx)
    cn = np.asarray(cn_mx).astype(np.float64)

    # 4. H0 (CAO basis), then SAO via T.
    cao_bf_shells = _cao_bf_shells(atoms_list, cao_basis)
    H0_cao_offdiag, selfE_eV_cao = build_hcore_gfn2(
        atoms_list, coords, cao_basis, S_cao, cn, cao_bf_shells,
    )
    np.fill_diagonal(H0_cao_offdiag, 0.0)
    S = T @ S_cao @ T.T
    H0 = T @ H0_cao_offdiag @ T.T
    selfE_eV = np.zeros(n_basis, dtype=np.float64)
    for mu_sao, b in enumerate(sao_basis):
        for mu_cao, bc in enumerate(cao_basis):
            if bc.shell_id == b.shell_id:
                selfE_eV[mu_sao] = selfE_eV_cao[mu_cao]
                break
    H0 = H0 + np.diag(selfE_eV * _HARTREE_PER_EV)

    # 5. Coulomb matrix.
    jmat = _coulomb_matrix(coords_bohr, shell_atom, shell_hard,
                            g_exp=GFN2_GLOBALS.alphaj)

    # 6. Initial qsh from EEQ.
    q_mx, _ = eeq_charges_and_energy(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms_list, dtype=np.int32)),
        total_charge=float(charge),
    )
    mx.eval(q_mx)
    q_at_init = np.asarray(q_mx).astype(np.float64)
    qsh = np.zeros(n_shell, dtype=np.float64)
    for ish in range(n_shell):
        a = shell_atom[ish]
        z_atom_ref = sum(z_ref[k] for k in range(n_shell) if shell_atom[k] == a)
        if z_atom_ref > 1e-9:
            qsh[ish] = q_at_init[a] * z_ref[ish] / z_atom_ref

    # 7. SCF.
    P = np.zeros((n_basis, n_basis), dtype=np.float64)
    converged = False
    n_iter = 0
    last_qsh = qsh.copy()
    # GFN2 uses fractional reference occupations (N is 1.5s + 3.5p,
    # for instance). Sum as floats, round to nearest int.
    n_elec_f = float(np.sum(z_ref)) - float(charge)
    n_elec = int(round(n_elec_f))
    if abs(n_elec - n_elec_f) > 1e-6:
        raise NotImplementedError(
            f"Open-shell GFN2 not supported "
            f"(non-integer n_elec={n_elec_f:.6f})"
        )
    if n_elec % 2 != 0:
        raise NotImplementedError(
            f"Open-shell GFN2 not supported (n_elec={n_elec})"
        )
    n_occ = n_elec // 2

    F_hist: list[np.ndarray] = []
    e_hist: list[np.ndarray] = []
    diis_max = 6
    diis_warmup = 3

    for it in range(max_iter):
        # Shell-resolved Coulomb shift.
        shell_shift = jmat @ qsh
        # GFN2 third-order is *shell-resolved*: each shell's energy
        # gets q_sh^2 · Γ_l. The Fock potential adds 2·q_sh·Γ_l (since
        # ∂/∂q_sh of q_sh^3·Γ/3 = q_sh²·Γ, and the Fock potential is
        # ∂E/∂P which carries an extra q_sh of population — net effect
        # in the V_sh is q_sh^2 · Γ_l).
        # Reference: scc_core.f90 third-order shell loop.
        shell_third_shift = qsh ** 2 * shell_third
        V_sh = shell_shift + shell_third_shift

        V_bf = V_sh[bf_to_shell]
        F = H0 - 0.5 * (V_bf[:, None] + V_bf[None, :]) * S

        if it >= diis_warmup:
            e_diis = F @ P @ S - S @ P @ F
            F_hist.append(F)
            e_hist.append(e_diis)
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
        if it < diis_warmup:
            qsh = mix * qsh_new + (1.0 - mix) * qsh
        else:
            qsh = qsh_new
        last_qsh = qsh.copy()

    if not converged:
        n_iter = max_iter

    # Final F build for energy decomposition.
    shell_shift = jmat @ qsh
    shell_third_shift = qsh ** 2 * shell_third
    V_sh = shell_shift + shell_third_shift
    V_bf = V_sh[bf_to_shell]
    F = H0 - 0.5 * (V_bf[:, None] + V_bf[None, :]) * S
    from scipy.linalg import eigh as _scipy_eigh
    eigvals_h, C = _scipy_eigh(F, S)
    C_occ = C[:, :n_occ]
    P = 2.0 * (C_occ @ C_occ.T)

    # Atom charges for D4 / output.
    q_at = np.zeros(n_atoms)
    for ish in range(n_shell):
        q_at[shell_atom[ish]] += qsh[ish]

    # Energy decomposition.
    PH0_h = float(np.sum(P * H0))
    E_es_h = 0.5 * float(qsh @ (jmat @ qsh))
    # Shell-resolved third-order:  E_3rd = Σ_sh q_sh^3 · Γ_l(Z) / 3.
    E_3rd_h = float(np.sum(qsh ** 3 * shell_third) / 3.0)

    E_rep_h = _gfn2_repulsion(atoms_list, coords)

    # D4 dispersion (charge-dependent C6).
    if use_d4:
        from .dispersion_d4 import d4_dispersion_gfn2
        E_d4_h = d4_dispersion_gfn2(atoms_list, coords, cn=cn, q=q_at)
    else:
        E_d4_h = 0.0

    # Atomic reference energy (sum over isolated atoms of selfE · refOcc).
    E_atoms_eV = 0.0
    for Z in atoms_list:
        for shell in GFN2_PARAMS[Z].shells:
            occ = float(_GFN2_REFERENCEOCC[int(Z)][shell.l])
            E_atoms_eV += shell.h * occ
    E_atoms_h = E_atoms_eV * _HARTREE_PER_EV

    E_total_h = PH0_h + E_es_h + E_3rd_h + E_rep_h + E_d4_h
    atomization_h = E_atoms_h - E_total_h

    return {
        "energy_hartree": E_total_h,
        "energy_eV": E_total_h * _EV_PER_HARTREE,
        "energy_kcal": E_total_h * _KCAL_PER_HARTREE,
        "electronic_eV": (PH0_h + E_es_h + E_3rd_h) * _EV_PER_HARTREE,
        "eeq_eV": None,
        "repulsion_eV": E_rep_h * _EV_PER_HARTREE,
        "dispersion_eV": E_d4_h * _EV_PER_HARTREE,
        "halogen_bond_eV": None,            # GFN2 doesn't use halogen-bond
        "third_order_eV": E_3rd_h * _EV_PER_HARTREE,
        "aes_eV": None,                     # TODO Phase C1
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
        "method": "GFN2",
    }
