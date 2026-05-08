# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN0-xTB tight-binding core Hamiltonian H_μν.

Diagonal (xtb/src/xtb/hamiltonian.F90:47-141):

    H_μμ = h_{l_μ}^A − k_{cn,l_μ}^A · CN_A
                    − k_{q1,l_μ}^A · q_A − k_{q2}^A · q_A²

Off-diagonal (μ on A, ν on B, A ≠ B; xtb hamiltonian.F90:240-251 +
scc_core.f90:644-754):

    H_μν = ½ · K_AB^{l_μ, l_ν} · (h_μ + h_ν) · Π(R_AB) · S_μν

with
    K_AB^{l, l'} = ½ (k_l + k_{l'}) · enpoly · pairParam(A, B),
    enpoly      = 1 + enScale[l, l'] · Δχ² · (1 + enScale4 · Δχ²),
    Π(R_AB)     = (1 + 0.01 k_A^{l_μ} √(R/(r_A+r_B)))
                · (1 + 0.01 k_B^{l_ν} √(R/(r_A+r_B))),

per-atom radii ``r_A`` from D3 covalent radii (from cn._COV_RAD_PYYKKO,
NOT scaled by 4/3 here — the 4/3 factor is specific to the CN; the
shell-poly Π uses raw Pyykkö radii). Slater-exponent ratio ζ_μν = 1
(GFN0 wExp = 0; matches GFN1 convention).

Phase A scope: numpy single-molecule. Boundary cast to ``mx.array``
happens in the orchestrator (energy.py).
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction
from .params_gfn0 import GFN0_GLOBALS, GFN0_PARAMS, GFN0Shell

_HARTREE_PER_EV = 1.0 / 27.211386245988
_ANG_TO_BOHR = 1.8897259886

# Atomic radii used by xtb's shellPoly Π(R) factor — Mantina/Truhlar 2010
# tabulation (xtb/src/param/atomicrad.f90), in Angstrom. xtb scales by
# `aatoau` (the Angstrom→Bohr conversion) at module load. These differ
# from the Pyykkö covalent radii used in CN: e.g. He 0.37 vs 0.46, Li
# 1.30 vs 1.20. Indexed 1..118 (entry 0 is a sentinel).
_ATOMIC_RAD_MANTINA_ANG = (
    0.00,
    0.32, 0.37,
    1.30, 0.99, 0.84, 0.75, 0.71, 0.64, 0.60, 0.62,
    1.60, 1.40, 1.24, 1.14, 1.09, 1.04, 1.00, 1.01,
    2.00, 1.74, 1.59, 1.48, 1.44, 1.30, 1.29, 1.24, 1.18, 1.17, 1.22, 1.20,
    1.23, 1.20, 1.20, 1.18, 1.17, 1.16,
    2.15, 1.90, 1.76, 1.64, 1.56, 1.46, 1.38, 1.36, 1.34, 1.30, 1.36, 1.40,
    1.42, 1.40, 1.40, 1.37, 1.36, 1.36,
    2.38, 2.06,
    1.94, 1.84, 1.90, 1.88, 1.86, 1.85, 1.83, 1.82, 1.81, 1.80, 1.79, 1.77,
    1.77, 1.78, 1.74, 1.64, 1.58, 1.50, 1.41, 1.36, 1.32, 1.30, 1.30, 1.32,
    1.44, 1.45, 1.50, 1.42, 1.48, 1.46,
    2.42, 2.11,
    2.01, 1.90, 1.84, 1.83, 1.80, 1.80, 1.73, 1.68, 1.68, 1.68, 1.65, 1.67,
    1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28, 1.21, 1.22,
    1.36, 1.43, 1.62, 1.75, 1.65, 1.57,
)
assert len(_ATOMIC_RAD_MANTINA_ANG) == 119


def _shell_kscale(l: int) -> float:
    """k_l global per shell type (file $globpar)."""
    g = GFN0_GLOBALS
    return [g.ks, g.kp, g.kd][l]


def _shell_enscale(l: int) -> float:
    """en-l shell global (the 'ens'/'enp'/'end' globals from $globpar)."""
    g = GFN0_GLOBALS
    return [g.ens, g.enp, g.end][l]


def _find_shell(Z: int, l_total: int, n_principal_hint: int | None = None) -> GFN0Shell:
    """Locate the GFN0 shell on element Z with the given l (and matching
    n if specified). Falls back to the first matching shell if hint absent.
    """
    p = GFN0_PARAMS[int(Z)]
    candidates = [s for s in p.shells if s.l == l_total]
    if not candidates:
        raise ValueError(f"Element Z={Z} has no shell with l={l_total}")
    if n_principal_hint is not None:
        for s in candidates:
            if s.n == n_principal_hint:
                return s
    return candidates[0]


def build_hcore(
    atoms: list[int],
    coords_ang: np.ndarray,
    basis: list[BasisFunction],
    S: np.ndarray,
    cn: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Single-molecule core Hamiltonian H, in Hartree.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom positions.
        basis: AO basis from :func:`mlxmolkit.xtb.basis.build_basis`.
        S: ``(n_basis, n_basis)`` overlap matrix (numpy).
        cn: ``(n_atoms,)`` erf-CN array.
        q: ``(n_atoms,)`` EEQ atomic charges.

    Returns:
        ``H`` of shape ``(n_basis, n_basis)``, symmetric, in Hartree.
    """
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    cn = np.asarray(cn, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n_atoms = len(atoms)
    n_basis = len(basis)

    # Per-basis shell lookup (h, k_cn, k_q1, k_poly, k_q2_atom).
    # Each basis function maps to: (Z, l, n, atom_idx).
    # We stash the matching GFN0Shell on each BF for fast access.
    bf_shells: list[GFN0Shell] = []
    bf_kq2: list[float] = []
    for bf in basis:
        Z = atoms[bf.atom_idx]
        # Use the principal-quantum-number hint if multiple shells share l.
        # For Phase A0 (CHNO sp valence) every l only appears once per atom
        # except for H's auxiliary 2s. Distinguish by n via the alpha range.
        # Simpler: pick the FIRST shell of this l on this atom; auxiliary
        # 2s on H is a special case — we handle by ordering the basis to
        # match GFN0_PARAMS' shell order.
        # ...
        # For correctness, identify which shell of this (Z, l) the BF
        # corresponds to. We rely on basis.build_basis preserving the
        # order shells are listed in GFN0_PARAMS (1s, then 2s for H, etc).
        pass

    # Walk the basis in atom-major / shell-major order and tag each BF
    # with its source GFN0Shell. The atom_offset+shell_idx tracking is
    # essentially counting how many of this atom's BFs have been seen.
    bf_shells = [None] * n_basis
    bf_kq2 = [0.0] * n_basis
    cursor = 0
    for at_idx, Z in enumerate(atoms):
        p = GFN0_PARAMS[int(Z)]
        for shell in p.shells:
            if shell.l > 1:
                continue
            n_components = 1 if shell.l == 0 else 3
            for _ in range(n_components):
                bf_shells[cursor] = shell
                bf_kq2[cursor] = p.k_q2
                cursor += 1
    assert cursor == n_basis, f"basis tagging mismatch ({cursor} vs {n_basis})"

    # ---- Per-BF self-energy (CN- and q-corrected diagonal level), eV.
    # xtb uses this same quantity in BOTH the diagonal H_μμ and the
    # off-diagonal h_avg = ½(selfE_μ + selfE_ν) — see scc_core.f90:80-95.
    selfE_eV = np.zeros(n_basis, dtype=np.float64)
    for mu in range(n_basis):
        s_mu = bf_shells[mu]
        A = basis[mu].atom_idx
        selfE_eV[mu] = (
            s_mu.h
            - s_mu.k_cn * cn[A]
            - s_mu.k_q1 * q[A]
            - bf_kq2[mu] * q[A] ** 2
        )

    H = np.zeros((n_basis, n_basis), dtype=np.float64)
    for mu in range(n_basis):
        H[mu, mu] = selfE_eV[mu] * _HARTREE_PER_EV

    # ---- Off-diagonal: H_μν = ½ K_AB^{l,l'} · (h_μ + h_ν) · Π · S_μν ----
    # Atomic radii for the shell-poly Π (Mantina/Truhlar 2010, in Angstrom);
    # xtb's atomicrad.f90 uses these (NOT Pyykkö covalent radii).
    r_A_ang = np.array([_ATOMIC_RAD_MANTINA_ANG[int(Z)] for Z in atoms], dtype=np.float64)
    r_A_bohr = r_A_ang * _ANG_TO_BOHR

    g = GFN0_GLOBALS
    en_atoms = np.array([GFN0_PARAMS[int(Z)].en for Z in atoms], dtype=np.float64)

    for mu in range(n_basis):
        bm = basis[mu]
        s_mu = bf_shells[mu]
        A = bm.atom_idx
        val_mu = bm.is_valence
        for nu in range(mu + 1, n_basis):
            bn = basis[nu]
            s_nu = bf_shells[nu]
            B = bn.atom_idx
            val_nu = bn.is_valence
            if A == B:
                continue                       # same-atom: H_μν = 0 here
            l_mu = s_mu.l
            l_nu = s_nu.l
            # GFN0 off-diagonal scaling per peeq_module.f90:921-973.
            # Note: this is DIFFERENT from scc_core.f90's h0scal (which
            # is GFN1/2). For GFN0:
            #   km = kScale[il,jl] · pairParam · ζij · enpoly
            #   if both shells aux:  km = 0
            #   if exactly one aux:  km *= kDiff
            #   if both valence:     unchanged
            d_chi = en_atoms[A] - en_atoms[B]
            d_chi2 = d_chi * d_chi
            sum_en = _shell_enscale(l_mu) + _shell_enscale(l_nu)
            # peeq's enpoly = 1 + enScale·den² + enScale4·enScale·den⁴,
            # which is algebraically 1 + enScale·den²·(1 + enScale4·den²).
            enpoly = 1.0 + 0.005 * sum_en * d_chi2 * (1.0 + g.enscale4 * d_chi2)
            kscale = 0.5 * (_shell_kscale(l_mu) + _shell_kscale(l_nu))
            zi, zj = s_mu.zeta, s_nu.zeta
            zetaij = 2.0 * np.sqrt(zi * zj) / (zi + zj)
            K_AB = kscale * enpoly * zetaij    # pairParam = 1 (main group)
            if (not val_mu) and (not val_nu):
                K_AB = 0.0                     # aux-aux pair: zero coupling
            elif val_mu != val_nu:
                K_AB *= g.kdiff                # exactly one aux: × kDiff
            # Distance-dependent shell-poly Π
            R_AB_bohr = float(np.linalg.norm(coords[A] - coords[B]))
            r_sum_bohr = r_A_bohr[A] + r_A_bohr[B]
            sqrt_term = float(np.sqrt(R_AB_bohr / max(r_sum_bohr, 1e-12)))
            pi_A = 1.0 + 0.01 * s_mu.k_poly * sqrt_term
            pi_B = 1.0 + 0.01 * s_nu.k_poly * sqrt_term
            Pi = pi_A * pi_B
            # Average level energy (Hartree) using the CN/q-corrected
            # selfEnergy (matches xtb's hav in peeq_module.f90:969).
            h_avg = 0.5 * (selfE_eV[mu] + selfE_eV[nu]) * _HARTREE_PER_EV
            H_munu = K_AB * h_avg * Pi * float(S[mu, nu])
            H[mu, nu] = H_munu
            H[nu, mu] = H_munu

    return H
