# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN1-xTB total-energy gradient.

Two paths are provided:

* :func:`gfn1_gradient` (``method='numerical'``) — central differences
  on :func:`gfn1_energy`. ``6N + 1`` SCF calls. Always available, used
  as the FD reference and as a fallback for any term whose analytical
  port hasn't landed.

* :func:`gfn1_gradient_analytical` — pure-analytical Pulay +
  Hellmann-Feynman + SCC-Coulomb assembly for the band/SCC piece, plus
  closed-form repulsion gradient. D3 dispersion and halogen-bond
  gradients are still finite-difference (closed-form, no SCF needed)
  until their analytical kernels land.

The analytical decomposition (xtb's standard form, scc_core.f90):

    ∂E_total/∂R = trace(P · ∂H0_diag/∂R)        # gradient_hf_diag
                + trace(P · ∂H0_off /∂R)        # gradient_hf_offdiag
                + ½ q · ∂J/∂R · q               # gradient_coulomb
                - trace(W · ∂S/∂R)              # gradient_pulay
                + ∂E_rep/∂R                     # repulsion_gradient
                + ∂E_d3/∂R                      # FD on E_d3 (cheap)
                + ∂E_xb/∂R                      # FD on E_xb (cheap)

with ``W`` the energy-weighted density and the third-order term
``E_3rd = ⅓ Σ q_at³ · Γ`` having zero R-derivative at SCF convergence
(no explicit R-dependence; q held fixed by stationarity).
"""

from __future__ import annotations

import numpy as np

from .dispersion_d3 import d3bj_dispersion_gfn1
from .gradient_coulomb import coulomb_gradient
from .gradient_hf_diag import hf_diagonal_gradient
from .gradient_hf_offdiag import hf_offdiag_gradient
from .gradient_gfn0 import cn_gradient
from .gradient_pulay import energy_weighted_density, pulay_gradient
from .halogen_bond import halogen_bond_energy
from .overlap_grad import overlap_gradient
from .params_gfn1 import GFN1_PARAMS
from .scf_gfn1 import gfn1_energy


_ANG_TO_BOHR = 1.8897259886


def numerical_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    h: float = 1e-3,
    scf_kwargs: dict | None = None,
) -> np.ndarray:
    """Three-point central-difference gradient ``∂E_total/∂x`` (Hartree / Å).

    ``6 · n_atoms`` SCF calls. Used as the FD reference and as a
    fallback when an analytical piece isn't available.
    """
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
            ep = gfn1_energy(atoms, coords, charge=charge, **scf_kwargs)[
                "energy_hartree"
            ]
            coords[i, a] = saved - h
            em = gfn1_energy(atoms, coords, charge=charge, **scf_kwargs)[
                "energy_hartree"
            ]
            coords[i, a] = saved
            grad[i, a] = (ep - em) / (2.0 * h)
    return grad


def _gfn1_repulsion_gradient(
    atoms: list[int], coords_ang: np.ndarray
) -> np.ndarray:
    """Analytical gradient of the GFN1 classical repulsion.

    GFN1 form (no enscale modulation, unlike GFN0):

        E_rep_pair = z_AB · exp(-α_AB R^k) / R         (k = 1.5)
        α_AB = sqrt(α_A · α_B)
        z_AB = z_eff_A · z_eff_B

    Per-pair gradient on atom A (Hartree / Bohr):

        dtmp = z_AB · exp(-α R^k) · (k·α·R^k + 1) / R³
        ∇_A E_pair = -dtmp · (R_A − R_B)
        ∇_B E_pair =  dtmp · (R_A − R_B)

    Returns (n_atoms, 3) in **Hartree / Å**.
    """
    coords_b = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n = len(atoms)
    grad_b = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return grad_b * _ANG_TO_BOHR
    kexp = 1.5
    for i in range(n - 1):
        Zi = atoms[i]
        pi = GFN1_PARAMS[int(Zi)]
        for j in range(i + 1, n):
            Zj = atoms[j]
            pj = GFN1_PARAMS[int(Zj)]
            rij = coords_b[i] - coords_b[j]
            R = float(np.linalg.norm(rij))
            if R < 1e-12:
                continue
            alpha = float(np.sqrt(pi.rep_alpha * pj.rep_alpha))
            zab = pi.rep_zeff * pj.rep_zeff
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
    """Central-difference gradient for closed-form scalar terms (no SCF).

    Used for the D3 and halogen-bond pieces until their analytical
    derivatives are vendored. Returns (n_atoms, 3) in Hartree / Å.
    """
    coords = np.asarray(coords_ang, dtype=np.float64).copy()
    n = coords.shape[0]
    g = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for a in range(3):
            s = coords[i, a]
            coords[i, a] = s + h
            ep = float(energy_fn(atoms, coords))
            coords[i, a] = s - h
            em = float(energy_fn(atoms, coords))
            coords[i, a] = s
            g[i, a] = (ep - em) / (2.0 * h)
    return g


def gfn1_gradient_analytical(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    scf_kwargs: dict | None = None,
    fd_h_aux: float = 1e-3,
) -> dict:
    """Analytical GFN1 gradient (Hartree / Å).

    The band + SCC piece is fully analytical (Pulay + HF-diag +
    HF-offdiag + Coulomb). Repulsion is closed-form analytical. D3 and
    halogen-bond gradients are obtained by central differences on
    their (closed-form, SCF-free) energies — far cheaper than the
    ``6N+1`` SCF cost of the fully-numerical path.

    Args:
        atoms: atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Å.
        charge: integer net charge.
        scf_kwargs: extra kwargs for the underlying :func:`gfn1_energy`.
            Defaults to ``conv_tol=1e-9`` / ``max_iter=200``.
        fd_h_aux: step (Å) for the D3 + halogen-bond FD pieces.

    Returns:
        Dict with keys ``gradient`` (Ha/Å), ``energy`` (Ha), and a
        ``components`` sub-dict for diagnostics.
    """
    if scf_kwargs is None:
        scf_kwargs = {}
    scf_kwargs.setdefault("conv_tol", 1e-9)
    scf_kwargs.setdefault("max_iter", 200)

    res = gfn1_energy(atoms, coords_ang, charge=charge, **scf_kwargs)
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
    shell_atom = res["shell_atom"]
    shell_hard = res["shell_hardness"]
    cn = res["coordination_number"]

    coords_bohr = coords * _ANG_TO_BOHR

    # -------- 1. Pulay term: -trace(W · ∂S/∂r) on SAO basis -----------
    # (∂S_sao/∂r is computed by transforming ∂S_cao/∂r through T.)
    dSA_cao, dSB_cao = overlap_gradient(cao_basis)  # (3, n_cao, n_cao) /Bohr
    # Transform each axis-slice through T: ∂S_sao = T · ∂S_cao · T^T.
    n_sao = T.shape[0]
    dSA_sao = np.zeros((3, n_sao, n_sao))
    dSB_sao = np.zeros((3, n_sao, n_sao))
    for ax in range(3):
        dSA_sao[ax] = T @ dSA_cao[ax] @ T.T
        dSB_sao[ax] = T @ dSB_cao[ax] @ T.T

    W = energy_weighted_density(C, eigvals, n_occ)
    # pulay_gradient takes (basis, W, n_atoms) but uses overlap_gradient
    # internally. We instead inline the SAO-basis Pulay assembly here so
    # we don't recompute ∂S/∂r.
    sao_atom_of = np.array([b.atom_idx for b in sao_basis], dtype=np.int64)
    g_pulay_b = np.zeros((n_atoms, 3))  # Ha/Bohr
    for ax in range(3):
        contrib_bra = np.einsum("mn,mn->m", W, dSA_sao[ax])
        contrib_ket = np.einsum("mn,mn->n", W, dSB_sao[ax])
        for a in range(n_atoms):
            mask = sao_atom_of == a
            g_pulay_b[a, ax] -= float(np.sum(contrib_bra[mask]))
            g_pulay_b[a, ax] -= float(np.sum(contrib_ket[mask]))

    # -------- 2. HF-diagonal: P_sao_diag · ∂selfE/∂r ------------------
    # selfE_sao[μ] inherits the underlying CAO shell h, kCN; build the
    # per-SAO-BF kCN table from the CAO map.
    from .hcore_gfn1 import _set_gfn1_kcn

    sao_atom_of = np.array([b.atom_idx for b in sao_basis], dtype=np.int64)
    sao_kcn = np.zeros(n_sao, dtype=np.float64)
    for mu_sao, b in enumerate(sao_basis):
        # Find the matching CAO BF (any one — all CAOs in the same
        # shell share the same kCN and h).
        for mu_cao, bc in enumerate(cao_basis):
            if bc.shell_id == b.shell_id:
                sh = cao_bf_shells[mu_cao]
                Z = atoms[bc.atom_idx]
                # kCN(eV per CN) = -h · cnshell · 0.01 (matches build_hcore_gfn1)
                sao_kcn[mu_sao] = -sh.h * _set_gfn1_kcn(Z, sh.l) * 0.01
                break
    cn_local, dcn_dr = cn_gradient(atoms, coords)  # Å^-1
    g_hf_diag_a = hf_diagonal_gradient(P_sao, sao_atom_of, sao_kcn, dcn_dr)
    # hf_diagonal returns Ha/Å; convert to Ha/Bohr to mix with Pulay.
    g_hf_diag_b = g_hf_diag_a / _ANG_TO_BOHR

    # -------- 3. HF off-diagonal in CAO ------------------------------
    P_cao_eff = T.T @ P_sao @ T
    g_hf_off_b = hf_offdiag_gradient(
        atoms, coords, cao_basis, cao_bf_shells,
        P_cao_eff, S_cao, cn_local, dcn_dr, dSA_cao, dSB_cao,
    )

    # -------- 4. Coulomb gradient ½ q·∂J·q ---------------------------
    g_coul_b = coulomb_gradient(
        coords_bohr, shell_atom, shell_hard, qsh, g_exp=2.0,
    )

    # -------- 4b. SCC overlap cross term: -trace(V_diag · ∂S · P) ----
    # Comes from differentiating trace(P · F) under SCF-fixed q, since
    # F = H0 - ½(V·S + S·V). xtb's stmp = ... - Pij·(V_ish + V_jsh)
    # encodes exactly this piece (peeq_module.f90:1390-1410 / scf_module
    # .F90's build_dSDQH0 path). V_sh = (J·q)_sh + atom_shift_sh.
    bf_to_shell = res["bf_to_shell"]
    from .params_gfn1 import GFN1_PARAMS as _PARAMS
    q_at = res["atom_charges"]
    V_sh = jmat_at_R = None  # placeholder vars below
    # Recompute V_sh in Hartree (matches xtb's evtoau-converted ves).
    from .scf_gfn1 import _coulomb_matrix as _build_jmat
    jmat = _build_jmat(coords_bohr, shell_atom, shell_hard, g_exp=2.0)
    shell_shift = jmat @ qsh
    atom_shift = np.zeros(n_atoms)
    for a in range(n_atoms):
        atom_shift[a] = q_at[a] ** 2 * _PARAMS[atoms[a]].third_order
    V_sh = shell_shift + atom_shift[shell_atom]  # Hartree
    V_bf = V_sh[bf_to_shell]  # SAO-BF resolved

    # trace(V_diag·∂S/∂r_a·P) = Σ_{α,β} V_α · ∂S_αβ/∂r_a · P_αβ.
    # Splitting by which side hosts atom a:
    #   bra (α ∈ a, β anywhere):  V_α·dSA[α,β]·P_αβ — V from α (= a-side)
    #   ket (β ∈ a, α anywhere):  V_α·dSB[α,β]·P_αβ — V from α (= other-side)
    # The same V_μ-on-α convention is used in both loops; using V_ν in
    # the ket loop would conflate the two sides and double-count.
    g_vsop_b = np.zeros((n_atoms, 3))
    VP = V_bf[:, None] * P_sao  # VP[μ, ν] = V_μ · P_μν
    for ax in range(3):
        contrib_bra = np.einsum("mn,mn->m", VP, dSA_sao[ax])
        contrib_ket = np.einsum("mn,mn->n", VP, dSB_sao[ax])
        for a in range(n_atoms):
            mask = sao_atom_of == a
            g_vsop_b[a, ax] -= float(np.sum(contrib_bra[mask]))
            g_vsop_b[a, ax] -= float(np.sum(contrib_ket[mask]))

    # -------- 5. Repulsion (analytical) ------------------------------
    g_rep_a = _gfn1_repulsion_gradient(atoms, coords)  # Ha/Å

    # -------- 6/7. D3 + halogen-bond (FD on closed-form energies) -----
    g_d3_a = _fd_grad_scalar(
        atoms, coords,
        lambda a, c: d3bj_dispersion_gfn1(a, c),
        h=fd_h_aux,
    )
    g_xb_a = _fd_grad_scalar(
        atoms, coords,
        lambda a, c: halogen_bond_energy(a, c),
        h=fd_h_aux,
    )

    # -------- Total ---------------------------------------------------
    # Convert all Ha/Bohr pieces to Ha/Å.
    g_band_a = (
        g_pulay_b + g_hf_off_b + g_coul_b + g_vsop_b
    ) * _ANG_TO_BOHR + g_hf_diag_a
    g_total = g_band_a + g_rep_a + g_d3_a + g_xb_a

    return {
        "gradient": g_total,
        "energy": res["energy_hartree"],
        "components": {
            "pulay": g_pulay_b * _ANG_TO_BOHR,
            "hf_diag": g_hf_diag_a,
            "hf_offdiag": g_hf_off_b * _ANG_TO_BOHR,
            "coulomb": g_coul_b * _ANG_TO_BOHR,
            "vsop_cross": g_vsop_b * _ANG_TO_BOHR,
            "repulsion": g_rep_a,
            "dispersion": g_d3_a,
            "halogen_bond": g_xb_a,
        },
        "n_calls": 1 + 6 * n_atoms * 2,  # 1 SCF + FD on D3 + FD on XB
    }


def gfn1_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    method: str = "analytical",
    h: float = 1e-3,
    scf_kwargs: dict | None = None,
) -> dict:
    """GFN1-xTB gradient (Hartree / Å).

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.
        charge: integer net charge.
        method: ``"analytical"`` (default — uses the band+SCC chain
            rule + closed-form repulsion + FD-on-cheap-energies for D3
            and halogen-bond) or ``"numerical"`` (full 6N+1 SCF FD).
        h: central-difference step size (Å) for the numerical path.
        scf_kwargs: extra keyword args forwarded to :func:`gfn1_energy`.

    Returns:
        Dict with ``gradient`` (Ha/Å), ``energy`` (Ha), and ``n_calls``.
    """
    if method == "numerical":
        if scf_kwargs is None:
            scf_kwargs = {}
        e0 = gfn1_energy(atoms, coords_ang, charge=charge, **scf_kwargs)[
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
        return gfn1_gradient_analytical(
            atoms, coords_ang, charge=charge, scf_kwargs=scf_kwargs,
        )
    raise ValueError(f"Unknown method: {method!r}")
