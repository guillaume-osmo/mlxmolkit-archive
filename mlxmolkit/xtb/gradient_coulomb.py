# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""SCC Coulomb gradient ``½ q · ∂J · q`` for GFN1 (Klopman-Ohno γ).

The shell-resolved Coulomb energy at SCF convergence is

    E_es = ½ Σ_{ij} q_i · J[i, j] · q_j

with J the Klopman-Ohno γ matrix (cf. ``_coulomb_matrix`` in
:mod:`scf_gfn1`):

    J[i, j] = (R^k + γ_ij^{-k})^{-1/k}     (i ≠ j, on different atoms)
    J[i, j] = γ_ij                         (i ≠ j, same atom — geometry-free)
    J[i, i] = η_i                          (geometry-free)

Only the inter-atomic off-diagonal block depends on geometry, so the
gradient pulls only from those entries:

    dJ/dR = -J^{k+1} · R^{k-1}             (chain rule on the
                                            Klopman-Ohno form)

    ∂E_es/∂R_a = Σ_{i<j, A_i≠A_j} q_i · q_j · dJ/dR
                  · (R_{A_i} − R_{A_j}) / R · (δ_{a, A_i} − δ_{a, A_j})

The result is in **Hartree per Bohr** (since coords inside J are Bohr
and we don't apply any unit conversion).

This is the SCC piece — the Hellmann-Feynman + Pulay band-energy pieces
are in their own modules. Together they replace the 6N+1 numerical
fallback in :mod:`gradient_gfn1`.
"""

from __future__ import annotations

import numpy as np


def coulomb_gradient(
    coords_bohr: np.ndarray,
    shell_atom: np.ndarray,
    shell_hardness: np.ndarray,
    qsh: np.ndarray,
    g_exp: float = 2.0,
) -> np.ndarray:
    """``∂(½ q·J·q) / ∂R`` for the Klopman-Ohno shell-Coulomb matrix.

    Args:
        coords_bohr: ``(n_atoms, 3)`` Bohr positions.
        shell_atom: ``(n_shell,)`` atom index of each shell.
        shell_hardness: ``(n_shell,)`` per-shell hardness (η_i).
        qsh: ``(n_shell,)`` Mulliken shell charges (typically taken
            from the converged SCF; pass ``last_qsh``).
        g_exp: ``k`` exponent in the Klopman-Ohno form (xtb's
            ``alphaj``; defaults to 2.0 = canonical GFN1).

    Returns:
        ``g[atom, axis]`` shape ``(n_atoms, 3)`` in **Hartree per
        Bohr** (matches :mod:`gradient_pulay` / :mod:`gradient_hf_offdiag`).
    """
    n_atoms = coords_bohr.shape[0]
    n_sh = len(shell_atom)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    for i in range(n_sh):
        ai = int(shell_atom[i])
        gi = float(shell_hardness[i])
        qi = float(qsh[i])
        for j in range(i):
            aj = int(shell_atom[j])
            if ai == aj:
                # Same-atom intra-pair: geometry-free → no grad piece.
                continue
            gj = float(shell_hardness[j])
            qj = float(qsh[j])
            gij = 2.0 / (1.0 / gi + 1.0 / gj)  # harmonic average
            R_vec = coords_bohr[ai] - coords_bohr[aj]
            R = float(np.linalg.norm(R_vec))
            if R < 1e-12:
                continue
            # J = (R^k + γ^{-k})^{-1/k}
            inv_gk = gij ** (-g_exp)
            denom = R ** g_exp + inv_gk
            J_ij = denom ** (-1.0 / g_exp)
            # dJ/dR = -J^{k+1} · R^{k-1}
            dJ_dR = -(J_ij ** (g_exp + 1.0)) * (R ** (g_exp - 1.0))
            unit = R_vec / R
            # Both (i, j) and (j, i) contribute to ½·q·J·q (J symmetric);
            # the inner-loop guard j < i double-counts via the factor 2 baked
            # into the qi·qj prefactor below.
            pref = qi * qj * dJ_dR  # Ha/Bohr (sign already in dJ_dR)
            grad[ai] += pref * unit
            grad[aj] -= pref * unit
    return grad
