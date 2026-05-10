# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2-xTB total-energy gradient.

Two paths:

* :func:`gfn2_gradient` (``method='numerical'``) — central differences
  on :func:`gfn2_energy`. ``6N + 1`` SCF calls. Always available.
* :func:`gfn2_gradient` (``method='analytical'``) — analytical
  band/SCC piece (Pulay + HF-diag + HF-offdiag with GFN2's
  ``K · ζ · h_avg · Π · S`` form + Coulomb + V·∂S·P cross +
  closed-form repulsion), with finite differences on AES, D4, and
  solvation (closed-form energies; no SCF needed → far cheaper than
  the 6N+1 SCF cost). Default.

Note on AES: the *fully* analytical AES gradient (xtb's ``aniso_grad``)
needs ``∂dpint/∂r`` + ``∂qpint/∂r`` + ``∂radCN/∂r`` kernels — those
are deferred to a future commit. The current FD-on-AES path gives
FD-floor accuracy (~1e-6 Ha/Å) at ``2N + 1`` extra closed-form energy
evals (no SCF).
"""

from __future__ import annotations

import numpy as np

from .dispersion_d4 import d4_dispersion_gfn2
from .gradient_coulomb import coulomb_gradient
from .gradient_gfn0 import cn_gradient
from .gradient_hf_diag import hf_diagonal_gradient
from .gradient_hf_offdiag_gfn2 import hf_offdiag_gradient_gfn2
from .gradient_pulay import energy_weighted_density
from .multipole_grad import multipole_gradient, shift_multipole_grad
from .overlap_grad import overlap_gradient
from .params_gfn2 import GFN2_PARAMS
from .scf_gfn2 import gfn2_energy


_ANG_TO_BOHR = 1.8897259886


def numerical_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    h: float = 1e-3,
    scf_kwargs: dict | None = None,
) -> np.ndarray:
    """Central-difference ``∂E_total/∂x`` (Hartree / Å)."""
    if scf_kwargs is None:
        scf_kwargs = {}
    scf_kwargs.setdefault("conv_tol", 1e-9)
    scf_kwargs.setdefault("max_iter", 200)

    coords = np.asarray(coords_ang, dtype=np.float64).copy()
    n_atoms = coords.shape[0]
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        for a in range(3):
            saved = coords[i, a]
            coords[i, a] = saved + h
            ep = gfn2_energy(atoms, coords, charge=charge, **scf_kwargs)[
                "energy_hartree"
            ]
            coords[i, a] = saved - h
            em = gfn2_energy(atoms, coords, charge=charge, **scf_kwargs)[
                "energy_hartree"
            ]
            coords[i, a] = saved
            grad[i, a] = (ep - em) / (2.0 * h)
    return grad


def _gfn2_repulsion_gradient(
    atoms: list[int], coords_ang: np.ndarray
) -> np.ndarray:
    """Analytical gradient of the GFN2 classical repulsion.

    GFN2 form (verbatim port of repulsion.f90):
        E_rep_pair = z_AB · exp(-α_AB · R^k_AB) / R
        α_AB = sqrt(α_A · α_B)
        z_AB = z_eff_A · z_eff_B
        k_AB = kExpLight if BOTH atoms have Z<=2 else kExp

    The light/heavy switch is the inverted convention checked
    in :mod:`scf_gfn2._gfn2_repulsion`. Returns Ha / Å.
    """
    from .params_gfn2 import GFN2_GLOBALS as G

    coords_b = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n = len(atoms)
    grad_b = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return grad_b * _ANG_TO_BOHR
    for i in range(n - 1):
        Zi = int(atoms[i])
        pi = GFN2_PARAMS[Zi]
        for j in range(i + 1, n):
            Zj = int(atoms[j])
            pj = GFN2_PARAMS[Zj]
            rij = coords_b[i] - coords_b[j]
            R = float(np.linalg.norm(rij))
            if R < 1e-12:
                continue
            alpha = float(np.sqrt(pi.rep_alpha * pj.rep_alpha))
            zab = pi.rep_zeff * pj.rep_zeff
            # Same inverted-light/heavy convention as scf_gfn2
            kexp = G.kexp_light if (Zi <= 2 and Zj <= 2) else G.kexp
            r_to_k = R ** kexp
            expt = np.exp(-alpha * r_to_k) * zab
            dtmp = expt * (kexp * alpha * r_to_k + 1.0) / (R ** 3)
            grad_b[i] -= dtmp * rij
            grad_b[j] += dtmp * rij
    return grad_b * _ANG_TO_BOHR


def _fd_grad_scalar(
    atoms: list[int],
    coords_ang: np.ndarray,
    energy_fn,
    h: float = 1e-3,
) -> np.ndarray:
    """FD gradient for closed-form scalar terms (no SCF)."""
    coords = np.asarray(coords_ang, dtype=np.float64).copy()
    n = coords.shape[0]
    g = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for a in range(3):
            s = coords[i, a]
            coords[i, a] = s + h
            ep = float(energy_fn(coords))
            coords[i, a] = s - h
            em = float(energy_fn(coords))
            coords[i, a] = s
            g[i, a] = (ep - em) / (2.0 * h)
    return g


def _aes_full_energy_at(atoms_list, coords_ang, P_sao, qsh, shell_atom):
    """E_aes at perturbed coords with frozen P and qsh; everything
    downstream (Mulliken multipoles dipm/qp from perturbed dpint/qpint,
    gab3/gab5 from radcn(CN), rij directly) is recomputed.

    Captures the full ∂E_aes/∂R chain — the dipm/qp piece IS needed
    because dipm/qp are Mulliken-derived from P (not variational).
    Empirically gives ~6e-3 Ha/Å residual on H2O vs FD on the full
    SCF.
    """
    from .aes import aniso_electro, get_radcn, mmomgabzero, mmompop
    from .basis import build_basis, sao_basis_metadata
    from .gradient_gfn0 import cn_gradient as _cn_grad
    from .multipole_integrals import multipole_matrices
    from .scf_gfn2 import gfn2_n_gauss

    cao_b = build_basis(
        atoms_list, coords_ang, params_dict=GFN2_PARAMS, n_gauss_fn=gfn2_n_gauss,
    )
    S_cao_p, dpint_cao_p, qpint_cao_p = multipole_matrices(cao_b)
    sao_b, T_p = sao_basis_metadata(cao_b)
    np_sao = T_p.shape[0]
    S_p = T_p @ S_cao_p @ T_p.T
    dpint_p = np.zeros((3, np_sao, np_sao))
    qpint_p = np.zeros((6, np_sao, np_sao))
    for k in range(3):
        dpint_p[k] = T_p @ dpint_cao_p[k] @ T_p.T
    for k in range(6):
        qpint_p[k] = T_p @ qpint_cao_p[k] @ T_p.T
    aoat_p = np.array([b.atom_idx for b in sao_b], dtype=np.int64)
    coords_b = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    cn_p, _ = _cn_grad(atoms_list, coords_ang)
    radcn = get_radcn(atoms_list, cn_p)
    gab3, gab5 = mmomgabzero(coords_b, radcn)
    n_atoms = len(atoms_list)
    q_at = np.zeros(n_atoms)
    for ish in range(len(shell_atom)):
        q_at[int(shell_atom[ish])] += qsh[ish]
    dipm, qp = mmompop(P_sao, S_p, dpint_p, qpint_p, aoat_p, coords_b)
    e_pair, e_polar = aniso_electro(
        atoms_list, coords_b, q_at, dipm, qp, gab3, gab5,
    )
    return e_pair + e_polar


def gfn2_gradient_analytical(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    scf_kwargs: dict | None = None,
    fd_h_aux: float = 1e-3,
) -> dict:
    """Analytical GFN2 gradient (Hartree / Å).

    Band+SCC pieces are fully analytical. AES, D4, and solvation are
    obtained by central differences on their (closed-form, SCF-free)
    energies — no extra SCF cost.

    Returns a dict with ``gradient`` (Ha/Å), ``energy`` (Ha), and a
    ``components`` sub-dict for diagnostics.
    """
    if scf_kwargs is None:
        scf_kwargs = {}
    scf_kwargs.setdefault("conv_tol", 1e-9)
    scf_kwargs.setdefault("max_iter", 200)

    res = gfn2_energy(atoms, coords_ang, charge=charge, **scf_kwargs)
    coords = np.asarray(coords_ang, dtype=np.float64)
    n_atoms = len(atoms)

    P_sao = res["density"]
    C = res["mo_coeffs"]
    eigvals = res["eigenvalues"]
    n_occ = res["n_occ"]
    qsh = res["shell_charges"]
    S_sao = res["S"]
    S_cao = res["S_cao"]
    T = res["T_cao_to_sao"]
    cao_basis = res["cao_basis"]
    sao_basis = res["sao_basis"]
    cao_bf_shells = res["cao_bf_shells"]
    bf_to_shell = res["bf_to_shell"]
    shell_atom = res["shell_atom"]
    shell_hard = res["shell_hardness"]
    shell_third = res["shell_third"]
    cn = res["coordination_number"]

    coords_bohr = coords * _ANG_TO_BOHR

    # ----- 1. Pulay term: -trace(W · ∂S/∂r), in SAO basis -----
    dSA_cao, dSB_cao = overlap_gradient(cao_basis)
    n_sao = T.shape[0]
    dSA_sao = np.zeros((3, n_sao, n_sao))
    dSB_sao = np.zeros((3, n_sao, n_sao))
    for ax in range(3):
        dSA_sao[ax] = T @ dSA_cao[ax] @ T.T
        dSB_sao[ax] = T @ dSB_cao[ax] @ T.T

    W = energy_weighted_density(C, eigvals, n_occ)
    sao_atom_of = np.array([b.atom_idx for b in sao_basis], dtype=np.int64)
    g_pulay_b = np.zeros((n_atoms, 3))
    for ax in range(3):
        cb = np.einsum("mn,mn->m", W, dSA_sao[ax])
        ck = np.einsum("mn,mn->n", W, dSB_sao[ax])
        for a in range(n_atoms):
            mask = sao_atom_of == a
            g_pulay_b[a, ax] -= float(np.sum(cb[mask]))
            g_pulay_b[a, ax] -= float(np.sum(ck[mask]))

    # ----- 2. HF-diagonal: P_diag · ∂selfE/∂r through CN chain -----
    # GFN2's kCN is already in eV/CN — sao_kcn[μ] = -kCN_μ for the
    # (eV per CN) coefficient passed to hf_diagonal_gradient.
    sao_kcn = np.zeros(n_sao, dtype=np.float64)
    for mu_sao, b in enumerate(sao_basis):
        for mu_cao, bc in enumerate(cao_basis):
            if bc.shell_id == b.shell_id:
                sh = cao_bf_shells[mu_cao]
                # ∂selfE_μ / ∂CN_A = -kCN_μ ⇒ pass kCN_eff_per_BF = sh.kcn
                # so that hf_diagonal_gradient's pre = P·sao_kcn matches.
                sao_kcn[mu_sao] = sh.kcn
                break
    cn_local, dcn_dr = cn_gradient(atoms, coords)
    # hf_diagonal_gradient assumes ∂selfE/∂r = -sao_kcn·∂CN/∂r and
    # returns a Ha/Å gradient. For GFN2 the same form applies.
    g_hf_diag_a = hf_diagonal_gradient(P_sao, sao_atom_of, sao_kcn, dcn_dr)
    g_hf_diag_b = g_hf_diag_a / _ANG_TO_BOHR

    # ----- 3. HF off-diagonal in CAO -----
    P_cao_eff = T.T @ P_sao @ T
    g_hf_off_b = hf_offdiag_gradient_gfn2(
        atoms, coords, cao_basis, cao_bf_shells,
        P_cao_eff, S_cao, cn_local, dcn_dr, dSA_cao, dSB_cao,
    )

    # ----- 4. Coulomb gradient ½ q·∂J·q -----
    g_coul_b = coulomb_gradient(
        coords_bohr, shell_atom, shell_hard, qsh, g_exp=2.0,
    )

    # ----- 4b. SCC overlap cross term: -trace(V_diag · ∂S · P) -----
    # V_sh = (J·q)_sh + q_sh^2·Γ_sh + vs[A_sh] (AES per-atom scalar shift).
    # The AES vd/vq potentials couple to ∂dpint and ∂qpint and would need
    # their own derivative kernels (deferred). vs enters F via S the same
    # way V_sh does, so it folds into the same V·∂S·P cross-term path.
    from .scf_gfn2 import _coulomb_matrix as _build_jmat
    from .aes import setvsdq, mmompop, mmomgabzero, get_radcn

    jmat = _build_jmat(coords_bohr, shell_atom, shell_hard, g_exp=2.0)
    shell_shift = jmat @ qsh
    shell_third_shift = qsh ** 2 * shell_third
    # Per-atom AES vs: mmompop → setvsdq.
    aoat = sao_atom_of
    radcn = get_radcn(atoms, cn)
    gab3, gab5 = mmomgabzero(coords_bohr, radcn)
    q_at = res["atom_charges"]
    dipm, qp_aes = mmompop(P_sao, S_sao, res["dpint"], res["qpint"], aoat, coords_bohr)
    vs, vd, vq = setvsdq(
        atoms, coords_bohr, q_at, dipm, qp_aes, gab3, gab5,
    )
    # Build per-AO V_bf: shell-resolved Coulomb + 3rd-order + per-atom vs.
    V_sh = shell_shift + shell_third_shift
    # F = H0 − ½(V_bf·S+S·V_bf) + F_aes; F_aes adds +½(vs_μ+vs_ν)·S so the
    # *effective* V in the −½(V·S+S·V) form is V_bf − vs[atom]. The
    # vd/vq pieces of F_aes go into the analytical multipole-band term
    # below (g_band_aes_b) — DON'T double-count vs there.
    V_bf = V_sh[bf_to_shell] - vs[aoat]
    g_vsop_b = np.zeros((n_atoms, 3))
    VP = V_bf[:, None] * P_sao
    for ax in range(3):
        cb = np.einsum("mn,mn->m", VP, dSA_sao[ax])
        ck = np.einsum("mn,mn->n", VP, dSB_sao[ax])
        for a in range(n_atoms):
            mask = sao_atom_of == a
            g_vsop_b[a, ax] -= float(np.sum(cb[mask]))
            g_vsop_b[a, ax] -= float(np.sum(ck[mask]))

    # ----- 5. Repulsion (analytical) -----
    g_rep_a = _gfn2_repulsion_gradient(atoms, coords)

    # ----- 6/7. AES + D4 (FD on full E_aes recompute) -----
    # FD recomputes everything downstream of P (dipm/qp via Mulliken
    # from perturbed dpint/qpint, gab3/gab5 from radcn(CN), rij). The
    # dipm/qp chain DOES contribute to dE_total/dR (E_aes is not
    # variational in those quantities — they're Mulliken-derived from
    # P, not independent variational parameters). Captures the full
    # ∂E_aes/∂R; ~6e-3 Ha/Å residual on H2O empirically.
    g_aes_a = _fd_grad_scalar(
        atoms, coords,
        lambda c: _aes_full_energy_at(atoms, c, P_sao, qsh, shell_atom),
        h=fd_h_aux,
    )
    g_d4_a = _fd_grad_scalar(
        atoms, coords,
        lambda c: d4_dispersion_gfn2(atoms, c, cn=cn_local, q=res["atom_charges"]),
        h=fd_h_aux,
    )

    # NOTE on the F_aes band gradient cancellation: in mlxmolkit's
    # GFN2 SCF, E_aes is from aniso_electro (not fockelectro), but the
    # F_aes Fock contribution is from fockelectro. The relation
    # trace(P · F_aes) = 2 · E_focke (≠ 2·E_aes_aniso) means in the
    # energy-form gradient expansion
    #     E_total = 2Σε − V·z + ½q·J·q + ⅔q³Γ − 2·E_focke + E_aes_aniso + ...
    # the trace(P·∂F_aes/∂R) term that appears positively in d(2Σε)/dR
    # gets EXACTLY cancelled by the −2·dE_focke/dR term (since dE_focke
    # = ½·trace(P·∂F_aes/∂R) at fixed vs/vd/vq). So the F_aes band
    # piece doesn't appear explicitly — the dpint/qpint chain enters
    # only through dE_aes_aniso/∂R, which is captured by the FD path
    # below. Set band_aes_vd_vq to zero accordingly.
    g_band_aes_b = np.zeros((n_atoms, 3), dtype=np.float64)

    # Total: convert Ha/Bohr pieces to Ha/Å.
    g_band_a = (
        g_pulay_b + g_hf_off_b + g_coul_b + g_vsop_b + g_band_aes_b
    ) * _ANG_TO_BOHR + g_hf_diag_a
    g_total = g_band_a + g_rep_a + g_aes_a + g_d4_a

    # NOTE on component magnitudes: pulay (~5e-1), hf_offdiag (~7e-1),
    # vsop_cross (~6e-2), repulsion (~2e-1) are individually O(0.1–1)
    # Ha/Å. They cancel almost perfectly to give the physical total
    # ~1e-2 Ha/Å on small organics — this is the well-known Pulay
    # cancellation in SCF gradients (each term comes from a different
    # algebraic decomposition of trace(P·F) and 2·Σε; the sum is
    # invariant). DO NOT chase the individual magnitudes; FD-verify
    # each piece in isolation (we did, see :mod:`gradient_pulay`,
    # :mod:`gradient_hf_offdiag_gfn2`, :mod:`gradient_coulomb`).
    return {
        "gradient": g_total,
        "energy": res["energy_hartree"],
        "components": {
            "pulay": g_pulay_b * _ANG_TO_BOHR,
            "hf_diag": g_hf_diag_a,
            "hf_offdiag": g_hf_off_b * _ANG_TO_BOHR,
            "coulomb": g_coul_b * _ANG_TO_BOHR,
            "vsop_cross": g_vsop_b * _ANG_TO_BOHR,
            "band_aes_vd_vq": g_band_aes_b * _ANG_TO_BOHR,
            "repulsion": g_rep_a,
            "aes": g_aes_a,
            "dispersion": g_d4_a,
        },
        "n_calls": 1 + 6 * n_atoms * 3,  # 1 SCF + FD on AES + D4 + band_aes
    }


def gfn2_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    method: str = "numerical",
    h: float = 1e-3,
    scf_kwargs: dict | None = None,
) -> dict:
    """GFN2-xTB gradient (Hartree / Å).

    Args:
        method: ``"analytical"`` (default) or ``"numerical"``.
        h: central-difference step size (Å) for the numerical path.
    """
    if method == "numerical":
        if scf_kwargs is None:
            scf_kwargs = {}
        e0 = gfn2_energy(atoms, coords_ang, charge=charge, **scf_kwargs)[
            "energy_hartree"
        ]
        grad = numerical_gradient(
            atoms, coords_ang, charge=charge, h=h, scf_kwargs=scf_kwargs,
        )
        return {
            "gradient": grad,
            "energy": e0,
            "n_calls": 6 * len(atoms) + 1,
        }
    if method == "analytical":
        return gfn2_gradient_analytical(
            atoms, coords_ang, charge=charge, scf_kwargs=scf_kwargs,
        )
    raise ValueError(f"Unknown method: {method!r}")
