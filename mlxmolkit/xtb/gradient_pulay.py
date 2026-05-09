# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Pulay term of the analytical xTB gradient: ``-trace(W · ∂S/∂r)``.

This is the pure-overlap-derivative piece of the energy gradient, the
one term that's identical across GFN0/GFN1/GFN2 (and the cleanest to
isolate). It captures the "moving basis" contribution — the change in
the SCF energy that comes purely from rotating/displacing the AO
basis functions, holding the density matrix fixed.

The energy-weighted density matrix ``W`` for a closed-shell SCF is

    W[μ, ν] = 2 · Σ_{i ∈ occ} ε_i · C[μ, i] · C[ν, i]
            = C · diag(2 · n_i · ε_i) · C^T

where ``n_i = 1`` for doubly occupied orbitals, 0 otherwise.

The Pulay gradient on atom ``a``, axis ``α``, is

    ∂E_pulay / ∂R_a_α = - Σ_{μ on a, ν} W[μ, ν] · dS_dA[α, μ, ν]
                       - Σ_{μ, ν on a} W[μ, ν] · dS_dB[α, μ, ν]

The HF (Hellmann-Feynman) term ``+trace(P · ∂H0/∂r)`` and the SCC
coupling terms (``½ q_sh · ∂jmat/∂r · q_sh``, etc.) are *not* included
here — they're separate pieces of the full xTB analytical gradient,
to be layered on top as their respective ports land.

For now this module exists primarily to (a) verify our ∂S/∂r kernel
against numerical differentiation and (b) provide a clean Pulay
building block that can be combined with the existing closed-form
gradients (rep, CN, SRB, EEQ from gradient_gfn0).
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction
from .overlap_grad import overlap_gradient


def energy_weighted_density(
    C: np.ndarray,
    eigvals: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    """Closed-shell energy-weighted density matrix.

    Args:
        C: ``(n_basis, n_orb)`` MO coefficients (columns = MOs).
        eigvals: ``(n_orb,)`` orbital energies in Hartree.
        n_occ: number of doubly occupied MOs.

    Returns:
        ``W[μ, ν] = 2 · Σ_{i < n_occ} ε_i · C[μ, i] · C[ν, i]``.
    """
    weights = np.zeros(C.shape[1], dtype=np.float64)
    weights[:n_occ] = 2.0 * eigvals[:n_occ]
    return C @ np.diag(weights) @ C.T


def pulay_gradient(
    basis: list[BasisFunction],
    W: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Pulay gradient ``-trace(W · ∂S/∂r)`` per atom (Hartree / Bohr).

    Args:
        basis: AO basis used for the SCF.
        W: ``(n_basis, n_basis)`` energy-weighted density.
        n_atoms: number of atoms.

    Returns:
        ``g[atom, axis]`` with shape ``(n_atoms, 3)`` in **Hartree per
        Bohr** (matching ∂S/∂r's units, which are 1/Bohr since the OS
        derivative is computed in atomic units).
    """
    dSA, dSB = overlap_gradient(basis)   # both (3, n, n) in 1/Bohr
    n = len(basis)
    g = np.zeros((n_atoms, 3), dtype=np.float64)
    # Per-BF atom index — needed to localize ∂/∂R_a contributions.
    atom_of = np.array([b.atom_idx for b in basis], dtype=np.int64)
    # ∂/∂R_a = (sum over μ on a of dSA contribution) + (sum over ν on a of dSB)
    # - sign because Pulay is -trace(W · ∂S/∂r)
    for ax in range(3):
        # contribution from bra side: μ ∈ atom a
        # g_a -= Σ_{μ ∈ a, ν} W[μ, ν] · dSA[ax, μ, ν]
        contrib_bra = np.einsum("mn,mn->m", W, dSA[ax])     # (n,)
        contrib_ket = np.einsum("mn,mn->n", W, dSB[ax])     # (n,)
        for a in range(n_atoms):
            mask = atom_of == a
            g[a, ax] -= float(np.sum(contrib_bra[mask]))
            g[a, ax] -= float(np.sum(contrib_ket[mask]))
    return g
