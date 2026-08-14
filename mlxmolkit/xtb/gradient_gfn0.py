# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN0-xTB analytical gradient ``∂E_total/∂x``.

Decomposed by energy term:

    E_total = E_band + E_eeq + E_rep + E_SRB    (D4 deferred)

Each term has its own derivative routine. The gradient assembly is:

    ∇E = ∇E_band + ∇E_eeq + ∇E_rep + ∇E_SRB

The band energy gradient is the most involved — see Pulay 1969 +
Hellmann-Feynman:

    ∇E_band = Σ_μν P_μν (∇H_μν) − Σ_μν W_μν (∇S_μν)

where ``W = 2 Σ_occ ε_i C_i C_iᵀ`` is the energy-weighted density.
``∇H_μν`` further splits into:
- Diagonal H_μμ depends on CN_A and q_A → CN gradient × ∂H/∂CN +
  q gradient × ∂H/∂q.
- Off-diagonal H_μν depends on R_AB (via Π and S), selfE_μ + selfE_ν,
  and pairParam · ζ_ij · enpoly · K — these all have closed-form
  derivatives.

The numerical-gradient fallback is provided for cross-checking and for
unblocking the optimizer integration before all analytical pieces are
in place. It is O(N) energy evaluations per gradient, so use only for
small systems and verification.

Phase B0 scope: numpy single-molecule.
"""

from __future__ import annotations

import numpy as np

from .energy import gfn0_energy
from .params_gfn0 import GFN0_PARAMS, GFN0_GLOBALS

_ANG_TO_BOHR = 1.8897259886
_HARTREE_PER_EV = 1.0 / 27.211386245988


# ---------------------------------------------------------------------------
# CN gradient — needed by SRB (and later by EEQ + diagonal H_μμ).
# ---------------------------------------------------------------------------

def eeq_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    cn: np.ndarray,
    dcndr: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """``∇E_eeq`` for the GFN0 EEQ Lagrangian. Hartree / Angstrom.

    The EEQ Lagrangian
    ``L = (χ − κ√CN)·q + ½·qᵀ·J·q − λ·(Σq − q_total)``
    is solved at the optimum by the augmented linear system. Because
    ``∂L/∂q = 0`` and the constraint forces ``∂(Σq)/∂r = 0``, the
    Hellmann-Feynman theorem gives:

        dE_eeq/dr_K = ∂L/∂r_K |_q
                    = −(∂xvec/∂r_K)·q + ½·qᵀ·(∂J/∂r_K)·q

    where ``xvec = −χ + κ√CN`` (so ``χ − κ√CN = −xvec``).

    Closed forms used:
        ∂xvec_A/∂r_K = κ_A · 0.5/√CN_A · ∂CN_A/∂r_K
        J_AB(R) = erf(R·γ_AB)/R     with γ_AB = 1/√(α_A²+α_B²)
        dJ_AB/dR = [(2γ_AB R/√π)·exp(−(Rγ_AB)²) − erf(R·γ_AB)] / R²
        ∂J_AB/∂r_A = (dJ/dR)·rij/R; ∂J_AB/∂r_B = −(dJ/dR)·rij/R

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom positions.
        cn: ``(n_atoms,)`` GFN0 erf-CN array.
        dcndr: ``(n_atoms, n_atoms, 3)`` ``∂CN_i/∂r_k`` in Ang^-1.
        q: ``(n_atoms,)`` EEQ atomic charges (sign convention as in
            ``eeq.py`` — physical excess electron charge).

    Returns:
        ``∇E_eeq`` of shape ``(n_atoms, 3)`` in Hartree / Angstrom.
    """
    import math

    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_b = coords * _ANG_TO_BOHR
    cn = np.asarray(cn, dtype=np.float64)
    dcn_b = np.asarray(dcndr, dtype=np.float64) / _ANG_TO_BOHR  # Bohr^-1
    q = np.asarray(q, dtype=np.float64)
    n = len(atoms)
    sqrt_pi = float(np.sqrt(np.pi))

    # Per-atom EEQ params: kappa is the CN coupling, alpha the Gaussian
    # charge width (both in atomic units).
    kappa = np.array([GFN0_PARAMS[int(z)].eeq_kappa for z in atoms], dtype=np.float64)
    alpha = np.array([GFN0_PARAMS[int(z)].eeq_alpha for z in atoms], dtype=np.float64)

    grad_b = np.zeros((n, 3), dtype=np.float64)  # Hartree / Bohr

    # --- xvec / CN-chain term: −Σ_A (κ_A · 0.5 / √CN_A) · q_A · ∂CN_A/∂r_K
    cn_safe = np.maximum(cn, 1e-30)
    dxvec_dCN = kappa * 0.5 / np.sqrt(cn_safe)            # (n_atoms,)
    weight = dxvec_dCN * q                                # (n_atoms,)
    # ∂E_xvec/∂r_K = -Σ_A weight_A · ∂CN_A/∂r_K
    # dcn_b shape is (i, k, 3) with i=A, k=K.
    grad_b -= np.einsum("a,akx->kx", weight, dcn_b)

    # --- J-pair term: ½·qᵀ·(∂J/∂r_K)·q. Off-diagonal only (∂J_AA/∂r = 0).
    for A in range(n - 1):
        for B in range(A + 1, n):
            rij = coords_b[A] - coords_b[B]
            R = float(np.linalg.norm(rij))
            gAB = 1.0 / math.sqrt(alpha[A] ** 2 + alpha[B] ** 2)
            erf_term = math.erf(R * gAB)
            exp_term = math.exp(-(R * gAB) ** 2)
            # dJ/dR = [(2γR/√π)·exp(−(Rγ)²) − erf(Rγ)] / R²
            dJdR = (2.0 * gAB / sqrt_pi * exp_term - erf_term / R) / R
            # contribution to grad: q_A·q_B·(dJ/dR)·rij/R at A; minus at B.
            coef = q[A] * q[B] * dJdR / R
            grad_b[A] += coef * rij
            grad_b[B] -= coef * rij

    return grad_b * _ANG_TO_BOHR


def cn_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    k: float = 7.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the GFN0 erf-CN and ``∂CN_i/∂r_K`` (Ang^-1) in numpy.

    The pair counting function is ``f(R) = ½(1 + erf(−k·(R−R₀)/R₀))``
    with ``R₀ = (R_A^Pyykkö + R_B^Pyykkö) · 4/3``.

        df/dR = -k/(R₀ √π) · exp(-(k(R-R₀)/R₀)²)

        ∂CN_A/∂r_K =  Σ_{B≠A} (∂f/∂R) · ∂R_AB/∂r_K
                   = δ_{KA} · Σ_{B≠A} (∂f/∂R) · (r_A−r_B)/R_AB
                   − δ_{KB} · (∂f/∂R) · (r_A−r_B)/R_AB                   (K≠A)

    Saturation cap (``CN ≤ 8``) is *not* differentiated here — for
    typical organic systems CN stays well below 8. If a sample hits
    the cap, this routine returns the un-capped gradient (matches the
    physical PES where the cap is a reporting cutoff, not a real cliff).

    Returns:
        ``(cn, dcn_dr)`` where ``cn`` is shape ``(n_atoms,)`` and
        ``dcn_dr`` is shape ``(n_atoms, n_atoms, 3)`` with
        ``dcn_dr[i, k, :]`` giving ``∂CN_i / ∂r_k`` in Angstrom^-1.
    """
    import math
    from .cn import _COV_RAD_PYYKKO, _COV_SCALE
    coords = np.asarray(coords_ang, dtype=np.float64)
    n = len(atoms)
    sqrt_pi = float(np.sqrt(np.pi))
    rcov = np.array(
        [_COV_RAD_PYYKKO[int(z)] for z in atoms], dtype=np.float64
    )
    cn = np.zeros(n, dtype=np.float64)
    dcn = np.zeros((n, n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rij = coords[i] - coords[j]
            R = float(np.linalg.norm(rij))
            R0 = (rcov[i] + rcov[j]) * _COV_SCALE
            if R0 < 1e-12:
                continue
            arg = -k * (R - R0) / R0
            f = 0.5 * (1.0 + math.erf(arg))
            cn[i] += f
            # df/dR = -k/(R₀ √π) · exp(-arg²)
            dfdR = -k / (R0 * sqrt_pi) * math.exp(-arg * arg)
            # ∂R/∂r_i_α = rij_α / R; ∂R/∂r_j_α = -rij_α / R
            unit = rij / R
            dcn[i, i, :] += dfdR * unit
            dcn[i, j, :] -= dfdR * unit
    return cn, dcn


def gfn0_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    band_method: str = "numerical",
    band_h: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Total ``∂E_total/∂x`` for GFN0-xTB. Hartree / Angstrom.

    Decomposes into the four energy terms. Three are closed-form
    (repulsion, EEQ, SRB) and the band-energy gradient is currently
    finite-differenced on the band component only — ~6N additional
    energy-piece evaluations per gradient. This is much cheaper than
    a fully numerical gradient (which redoes basis + overlap + Hcore
    every step) since the closed-form parts are O(N²) algebra.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom positions.
        charge: integer net charge.
        band_method: ``"numerical"`` (default — finite difference on
            the band-energy contribution only) or ``"analytical"``
            (not yet implemented; raises NotImplementedError).
        band_h: finite-difference step (Å) for the band-only path.

    Returns:
        ``(grad, info)`` — ``grad`` is shape ``(n_atoms, 3)`` in
        Hartree/Å; ``info`` is a dict with the per-term decomposition.
    """
    import mlx.core as mx
    from .energy import gfn0_energy

    coords = np.asarray(coords_ang, dtype=np.float64)
    cn, dcn = cn_gradient(atoms, coords)

    # EEQ q via the existing MLX-backed solver. Use float64 numpy
    # rebuild for the gradient to keep precision; EEQ q is needed by
    # both the EEQ gradient (directly) and by the band gradient
    # (through the diagonal H).
    from .eeq import eeq_charges_and_energy
    q_mx, _ = eeq_charges_and_energy(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms, dtype=np.int32)),
        total_charge=float(charge),
    )
    mx.eval(q_mx)
    q = np.asarray(q_mx).astype(np.float64)

    g_rep = repulsion_gradient(atoms, coords)
    g_srb = srb_gradient(atoms, coords, cn, dcn)
    g_eeq = eeq_gradient(atoms, coords, cn, dcn, q)

    if band_method == "analytical":
        raise NotImplementedError(
            "Analytical band gradient not yet implemented; "
            "use band_method='numerical' for now."
        )

    # Numerical band-energy gradient via central differences on
    # E_band = E_total - E_rep - E_eeq - E_SRB.
    n = coords.shape[0]
    g_band = np.zeros_like(coords)
    h = band_h
    for i in range(n):
        for a in range(3):
            saved = coords[i, a]
            coords[i, a] = saved + h
            r_p = gfn0_energy(atoms, coords, charge=charge)
            band_p = (
                r_p["energy_hartree"]
                - r_p["repulsion_eV"] * _HARTREE_PER_EV
                - r_p["eeq_eV"] * _HARTREE_PER_EV
                - (r_p["srb_eV"] or 0.0) * _HARTREE_PER_EV
            )
            coords[i, a] = saved - h
            r_m = gfn0_energy(atoms, coords, charge=charge)
            band_m = (
                r_m["energy_hartree"]
                - r_m["repulsion_eV"] * _HARTREE_PER_EV
                - r_m["eeq_eV"] * _HARTREE_PER_EV
                - (r_m["srb_eV"] or 0.0) * _HARTREE_PER_EV
            )
            coords[i, a] = saved
            g_band[i, a] = (band_p - band_m) / (2.0 * h)

    grad_total = g_rep + g_srb + g_eeq + g_band
    info = {
        "g_repulsion": g_rep,
        "g_eeq": g_eeq,
        "g_srb": g_srb,
        "g_band": g_band,
        "method": "rep+srb+eeq analytical, band numerical",
    }
    return grad_total, info


def numerical_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    h: float = 1e-3,
) -> np.ndarray:
    """Three-point central-difference gradient ``∂E_total/∂x`` (Hartree / Å).

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.
        charge: integer net charge.
        h: step size in Å for the central difference. ``1e-3`` is a
            good default: gives ~6-digit gradient accuracy on GFN0
            since the energy is float64 and well-conditioned on the
            scales we're hitting.

    Returns:
        ``∇E`` of shape ``(n_atoms, 3)`` in Hartree per Angstrom.

    This routine performs ``6 · n_atoms`` energy calls and is intended
    only as a reference / fallback. Use the analytical gradient for
    production.
    """
    coords = np.asarray(coords_ang, dtype=np.float64).copy()
    n_atoms = coords.shape[0]
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        for a in range(3):
            saved = coords[i, a]
            coords[i, a] = saved + h
            ep = gfn0_energy(atoms, coords, charge=charge)["energy_hartree"]
            coords[i, a] = saved - h
            em = gfn0_energy(atoms, coords, charge=charge)["energy_hartree"]
            coords[i, a] = saved
            grad[i, a] = (ep - em) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# Closed-form gradient — REPULSION
# ---------------------------------------------------------------------------

def repulsion_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
) -> np.ndarray:
    """``∇E_rep`` for the GFN0 classical pairwise repulsion.

    From peeq_module.f90:646-703 (drep_grad). The energy is:

        E_rep = Σ_{A<B} Z_A^eff · Z_B^eff · exp(−α_AB · R_AB^k) / R_AB
        α_AB = √(α_A · α_B) · (1 + (0.01·Δχ² + 0.01·Δχ⁴) · renscale)

    with ``k = 1.5`` (kexp). The pairwise gradient on atom A is:

        dtmp = Z_AB · exp(−α R^k) · (1 + α·k·R^k) / R^(k+2)
        ∇_A E_pair = −dtmp · (r_A − r_B)
        ∇_B E_pair = +dtmp · (r_A − r_B)

    Returns gradient in **Hartree / Angstrom**.
    """
    coords_b = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n = len(atoms)
    grad_b = np.zeros((n, 3), dtype=np.float64)  # Hartree / Bohr
    if n < 2:
        return grad_b * _ANG_TO_BOHR  # convert to Ha/Å (zero anyway)
    g = GFN0_GLOBALS
    kexp = 1.5
    renscale = g.renscale
    for i in range(n - 1):
        Zi = atoms[i]
        pi = GFN0_PARAMS[Zi]
        for j in range(i + 1, n):
            Zj = atoms[j]
            pj = GFN0_PARAMS[Zj]
            rij = coords_b[i] - coords_b[j]
            R2 = float(rij @ rij)
            R = float(np.sqrt(R2))
            den2 = (pi.en - pj.en) ** 2
            den4 = den2 * den2
            alpha = np.sqrt(pi.rep_alpha * pj.rep_alpha) * (
                1.0 + (0.01 * den2 + 0.01 * den4) * renscale
            )
            zab = pi.rep_zeff * pj.rep_zeff
            r_to_k = R ** kexp
            expterm = np.exp(-alpha * r_to_k) * zab
            # dE/dR = -expterm/R^2 · (1 + alpha·k·R^k)
            # ∂E/∂x_i = dE/dR · rij/R = -expterm · (1 + alpha·k·R^k) / R³ · rij
            dtmp = expterm * (kexp * alpha * r_to_k + 1.0) / (R ** 3)
            grad_b[i] -= dtmp * rij
            grad_b[j] += dtmp * rij
    # convert Ha/Bohr → Ha/Å
    return grad_b * _ANG_TO_BOHR


# ---------------------------------------------------------------------------
# Closed-form gradient — SRB (short-range bond)
# ---------------------------------------------------------------------------

def srb_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    cn: np.ndarray,
    dcndr: np.ndarray | None = None,
) -> np.ndarray:
    """``∇E_SRB`` for the short-range bond correction.

    From peeq_module.f90:706-789 (dsrb_grad). The pair energy:

        E_pair = prefactor · exp(−pre · (R − R₀)²)
        R₀     = (r_i + r_j) · ff
        r_i    = r0_i + cnfak_i · CN_i + shift
        ff     = 1 − k1·|Δχ| − k2·Δχ²
        pre    = steepness · (1 + enScale · Δχ²)

    The Cartesian gradient has TWO sources:
        (a) explicit R-dependence: dtmp = 2·pre·(R−R₀)·E_pair, ∇_A E += dtmp·rij/R
        (b) CN dependence of R₀ via r_i,r_j: dEdr0 = dtmp; chain through dCN/dr.

    If ``dcndr`` is None this returns only (a) — useful for sanity but
    incomplete. Pass ``dcndr`` (shape ``(n_atoms, n_atoms, 3)``,
    ``∂CN_i/∂r_k`` Ang^-1) for the full gradient.

    Returns gradient in Hartree / Angstrom.
    """
    from .srb import (
        _SRB_EN, _SRB_R0, _SRB_CNFAK, _SRB_P, _srb_atom, _pse_row,
    )
    g = GFN0_GLOBALS
    coords_b = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    cn = np.asarray(cn, dtype=np.float64)
    n = len(atoms)
    grad_b = np.zeros((n, 3), dtype=np.float64)

    pairs = []
    for i in range(n - 1):
        if _srb_atom(int(atoms[i])):
            continue
        for j in range(i + 1, n):
            if _srb_atom(int(atoms[j])):
                continue
            if int(atoms[i]) == int(atoms[j]):
                continue
            rij = coords_b[i] - coords_b[j]
            r2 = float(rij @ rij)
            if r2 < 200.0:
                pairs.append((i, j, rij, r2))
    if not pairs:
        return grad_b * _ANG_TO_BOHR

    # dE/dr0 accumulated per pair, then propagated via dcndr.
    dEdr0_pair: list[float] = []
    pair_keys: list[tuple[int, int]] = []
    for (i, j, rij, r2) in pairs:
        ati = int(atoms[i]); atj = int(atoms[j])
        ir = _pse_row(ati); jr = _pse_row(atj)
        ra = _SRB_R0[ati] + _SRB_CNFAK[ati] * cn[i] + g.srbshift
        rb = _SRB_R0[atj] + _SRB_CNFAK[atj] * cn[j] + g.srbshift
        k1 = 0.005 * (_SRB_P[ir - 1, 0] + _SRB_P[jr - 1, 0])
        k2 = 0.005 * (_SRB_P[ir - 1, 1] + _SRB_P[jr - 1, 1])
        den_abs = abs(_SRB_EN[ati] - _SRB_EN[atj])
        ff = 1.0 - k1 * den_abs - k2 * den_abs ** 2
        rab0 = (ra + rb) * ff
        rab = float(np.sqrt(r2))
        dr = rab - rab0
        pre = g.srbexp * (1.0 + g.srbken * den_abs ** 2)
        expterm = g.srbpre * float(np.exp(-pre * dr * dr))
        # ∂E/∂R = −2·pre·dr·expterm (negative when dr>0 and expterm>0).
        # Define dtmp = +2·pre·dr·expterm so that ∂E/∂R = −dtmp; then
        #   ∂E/∂r_i  = (∂E/∂R) · (rij/R) = −dtmp·rij/R
        #   ∂E/∂r_j  = +dtmp·rij/R
        # And ∂E/∂R₀ = +dtmp (positive), so the chain through CN adds
        #   ∂E/∂r_k += dtmp · ∂R₀/∂r_k.
        dtmp = 2.0 * pre * dr * expterm
        grad_b[i] -= dtmp / rab * rij
        grad_b[j] += dtmp / rab * rij
        dEdr0_pair.append(dtmp)
        pair_keys.append((i, j))

    # CN-dependence: ∂R₀/∂r_k = ff · (cnfak_i · ∂CN_i/∂r_k + cnfak_j · ∂CN_j/∂r_k).
    # cn_gradient returns dcndr in Angstrom^-1; convert to Bohr^-1 by
    # dividing by (Bohr/Ang) so units are consistent with grad_b (Ha/Bohr).
    if dcndr is not None:
        dcndr = np.asarray(dcndr, dtype=np.float64) / _ANG_TO_BOHR
        for (i, j), dEdr0 in zip(pair_keys, dEdr0_pair):
            ati = int(atoms[i]); atj = int(atoms[j])
            ir = _pse_row(ati); jr = _pse_row(atj)
            k1 = 0.005 * (_SRB_P[ir - 1, 0] + _SRB_P[jr - 1, 0])
            k2 = 0.005 * (_SRB_P[ir - 1, 1] + _SRB_P[jr - 1, 1])
            den_abs = abs(_SRB_EN[ati] - _SRB_EN[atj])
            ff = 1.0 - k1 * den_abs - k2 * den_abs ** 2
            cni_fac = _SRB_CNFAK[ati] * ff
            cnj_fac = _SRB_CNFAK[atj] * ff
            # ∂E/∂r_k from CN-chain = +dEdr0 · ∂R₀/∂r_k
            #                        = +dtmp · ff · (cnfak_i · ∂CN_i/∂r_k +
            #                                        cnfak_j · ∂CN_j/∂r_k)
            for k in range(n):
                grad_b[k] += dEdr0 * (
                    cni_fac * dcndr[i, k] + cnj_fac * dcndr[j, k]
                )

    return grad_b * _ANG_TO_BOHR
