# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Diagonal-H0 Hellmann-Feynman gradient piece for GFN1/GFN2.

The SAO H0 diagonal in our SCF is

    H0[μ, μ] = selfE_μ · (Hartree/eV)
    selfE_μ  = h_μ - kCN_μ · CN_{A_μ}                      (eV)

so

    ∂H0[μ, μ] / ∂R_a = -kCN_μ · (∂CN_{A_μ} / ∂R_a) · (Hartree/eV)

The Hellmann-Feynman contribution to the band energy from the diagonal
is therefore

    g_HF_diag[a] = Σ_μ P[μ, μ] · ∂H0[μ, μ] / ∂R_a
                = -(Hartree/eV) · Σ_μ P[μ, μ] · kCN_μ · ∂CN_{A_μ} / ∂R_a

This is the *cheap* piece of the analytical gradient — it depends on
the converged density diagonal (a (n_basis,) vector) and the CN
gradient (already vendored in :mod:`gradient_gfn0`).

The full ``∂E_band/∂r`` also needs:
- the off-diagonal H0 piece (chain rule on K·h_avg·Π·S),
- the Pulay term ``-trace(W · ∂S/∂r)`` (in :mod:`gradient_pulay`),
- the SCC Coulomb gradient ``½ q · ∂J · q``.

Together those replace the 6N+1 numerical fallback.
"""

from __future__ import annotations

import numpy as np


_HARTREE_PER_EV = 1.0 / 27.211386245988


def hf_diagonal_gradient(
    P_sao: np.ndarray,
    sao_atom_of: np.ndarray,
    sao_kcn: np.ndarray,
    dcn_dr: np.ndarray,
) -> np.ndarray:
    """Diagonal-H0 HF gradient ``Σ_μ P_diag · ∂selfE_μ/∂r``.

    Args:
        P_sao: ``(n_basis, n_basis)`` density matrix (SAO basis).
        sao_atom_of: ``(n_basis,)`` atom index of each SAO BF.
        sao_kcn: ``(n_basis,)`` per-BF kCN coefficient (eV / CN unit;
            same value across all BFs of the same shell).
        dcn_dr: ``(n_atoms, n_atoms, 3)`` ``∂CN_i / ∂r_k`` in 1/Å.

    Returns:
        ``g[atom, axis]`` shape ``(n_atoms, 3)`` in **Hartree per
        Angstrom** (matching ``dcn_dr``'s units).
    """
    n_basis = P_sao.shape[0]
    n_atoms = dcn_dr.shape[0]
    P_diag = np.diag(P_sao)

    # Aggregate P_diag · kCN per atom: pre_a = Σ_{μ on a} P_diag[μ] · kCN_μ
    pre = np.zeros(n_atoms, dtype=np.float64)
    for mu in range(n_basis):
        a = int(sao_atom_of[mu])
        pre[a] += P_diag[mu] * sao_kcn[mu]

    # ∂E/∂r_k = -(Ha/eV) · Σ_a pre[a] · ∂CN_a/∂r_k
    g = np.zeros((n_atoms, 3), dtype=np.float64)
    for k in range(n_atoms):
        for a in range(n_atoms):
            g[k] -= _HARTREE_PER_EV * pre[a] * dcn_dr[a, k, :]
    return g
