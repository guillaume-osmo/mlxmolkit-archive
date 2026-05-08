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
from .cn import _COV_RAD_PYYKKO
from .params_gfn0 import GFN0_GLOBALS, GFN0_PARAMS, GFN0Shell

_HARTREE_PER_EV = 1.0 / 27.211386245988
_ANG_TO_BOHR = 1.8897259886


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
        seen_l: set[int] = set()
        for shell in p.shells:
            if shell.l > 1:
                continue
            if shell.l in seen_l:
                continue                          # aux shell skipped (matches basis.py)
            seen_l.add(shell.l)
            n_components = 1 if shell.l == 0 else 3
            for _ in range(n_components):
                bf_shells[cursor] = shell
                bf_kq2[cursor] = p.k_q2
                cursor += 1
    assert cursor == n_basis, f"basis tagging mismatch ({cursor} vs {n_basis})"

    # ---- Diagonal: H_μμ = h - k_cn·CN - k_q1·q - k_q2·q² ----
    # Convert level energies from eV to Hartree.
    H = np.zeros((n_basis, n_basis), dtype=np.float64)
    for mu in range(n_basis):
        bm = basis[mu]
        s_mu = bf_shells[mu]
        A = bm.atom_idx
        h_mu_eV = s_mu.h
        diag_eV = (
            h_mu_eV
            - s_mu.k_cn * cn[A]
            - s_mu.k_q1 * q[A]
            - bf_kq2[mu] * q[A] ** 2
        )
        H[mu, mu] = diag_eV * _HARTREE_PER_EV

    # ---- Off-diagonal: H_μν = ½ K_AB^{l,l'} · (h_μ + h_ν) · Π · S_μν ----
    # Atomic radii for the shell-poly Π (raw Pyykkö values, in Angstrom).
    r_A_ang = np.array([_COV_RAD_PYYKKO[int(Z)] for Z in atoms], dtype=np.float64)
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
            # h0scal — xtb scc.f90:644-680. Three branches:
            if val_mu and val_nu:
                # valence-valence: full EN-modulated product
                d_chi = en_atoms[A] - en_atoms[B]
                d_chi2 = d_chi * d_chi
                sum_en = _shell_enscale(l_mu) + _shell_enscale(l_nu)
                enpoly = 1.0 + 0.005 * sum_en * d_chi2 * (1.0 + g.enscale4 * d_chi2)
                kscale = 0.5 * (_shell_kscale(l_mu) + _shell_kscale(l_nu))
                K_AB = kscale * enpoly  # pairParam = 1 (main group)
            elif (not val_mu) and (not val_nu):
                # aux-aux: km = kDiff (the "DZ" scale)
                K_AB = g.kdiff
            elif val_mu and not val_nu:
                # val-aux: km = ½(kScale[lν,lν] + kDiff)
                K_AB = 0.5 * (_shell_kscale(l_nu) + g.kdiff)
            else:
                # aux-val: km = ½(kScale[lμ,lμ] + kDiff)
                K_AB = 0.5 * (_shell_kscale(l_mu) + g.kdiff)
            # Distance-dependent shell-poly Π
            R_AB_bohr = float(np.linalg.norm(coords[A] - coords[B]))
            r_sum_bohr = r_A_bohr[A] + r_A_bohr[B]
            sqrt_term = float(np.sqrt(R_AB_bohr / max(r_sum_bohr, 1e-12)))
            pi_A = 1.0 + 0.01 * s_mu.k_poly * sqrt_term
            pi_B = 1.0 + 0.01 * s_nu.k_poly * sqrt_term
            Pi = pi_A * pi_B
            # Average level energy in Hartree
            h_avg = 0.5 * (s_mu.h + s_nu.h) * _HARTREE_PER_EV
            H_munu = K_AB * h_avg * Pi * float(S[mu, nu])
            H[mu, nu] = H_munu
            H[nu, mu] = H_munu

    return H
