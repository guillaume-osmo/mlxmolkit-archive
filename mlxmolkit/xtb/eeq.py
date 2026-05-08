# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN0-xTB electronegativity-equilibration (EEQ) charges.

Solves the constrained quadratic charge-equilibration system per
Caldeweyher et al., *J. Chem. Phys.* 150, 154122 (2019), specialized to
GFN0's refit χ/η/κ/α parameters (distinct from the standalone EEQ-2019
set used by D4 — see params_eeq2019 in Phase A5).

For each molecule with ``n`` atoms, the energy is

    E^EEQ(q) = Σ_A χ̃_A q_A + ½ q^T J q  s.t.  Σ_A q_A = q_total

with  χ̃_A = −χ_A + κ_A · sqrt(CN_A),
       J_AA = η_A + sqrt(2/π) / α_A,
       J_AB = erf(R_AB · γ_AB) / R_AB,  γ_AB = 1 / sqrt(α_A² + α_B²),

stationarity of which gives the augmented Lagrangian system

    [ J  1 ] [ q ]     [ -χ̃     ]
    [ 1^T 0 ] [ λ ]  =  [ q_total ]

solved here via :func:`mlx_addons.linalg.solve_lu` (batched general LU,
Metal-accelerated, ``k <= 128`` per molecule).

Reference Fortran: ``awvwgk/multicharge/src/multicharge/model/eeq.f90``
(Apache-2.0, lines 127-242 for the kernel form). GFN0's per-element
parameters live in ``mlxmolkit.xtb.params_gfn0.GFN0ElementParams``.

Padded-batch convention
-----------------------
Molecules of different sizes are stacked into ``(B, max_atoms, 3)``
coords + ``(B, max_atoms)`` atomic numbers (``Z = 0`` for padding).
``n_atoms_arr`` carries the real atom count per molecule. The augmented
matrix gets identity diagonals on padded rows/cols (so ``q_padded = 0``
is the deterministic solution there) and the constraint row's ones are
restricted to the real-atom columns.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from mlx_addons.linalg import solve_lu

from .cn import KCN_ERF, coordination_number_erf
from .params_gfn0 import GFN0_PARAMS


# ---------------------------------------------------------------------------
# Per-element EEQ parameter table for GFN0, indexed 0..118.
# Layout: [chi, eta, kappa, alpha]; index 0 is a 0.0-sentinel for padded
# atoms (Z=0). Built once at module load from GFN0_PARAMS.
# ---------------------------------------------------------------------------
def _build_eeq_param_table() -> mx.array:
    table = np.zeros((119, 4), dtype=np.float32)
    for Z, p in GFN0_PARAMS.items():
        table[Z, 0] = p.eeq_chi
        table[Z, 1] = p.eeq_eta
        table[Z, 2] = p.eeq_kappa
        table[Z, 3] = p.eeq_alpha
    return mx.array(table)


_EEQ_PARAMS_TABLE = _build_eeq_param_table()
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
# EEQ parameters live in atomic units (Caldeweyher 2019), so the Coulomb
# formula needs R in Bohr. The public API accepts Angstrom (matches
# RDKit / mlxmolkit conventions); we convert inside the function.
_ANG_TO_BOHR = 1.8897259886


def eeq_charges(
    coords: mx.array,
    atoms: mx.array,
    n_atoms_arr: mx.array | None = None,
    *,
    total_charge: float = 0.0,
    k_cn: float = KCN_ERF,
) -> mx.array:
    """GFN0-xTB EEQ atomic charges, batched.

    Args:
        coords: ``(B, max_atoms, 3)`` Angstrom positions, or ``(N, 3)``
            for single-mol; auto-promoted.
        atoms: ``(B, max_atoms)`` int atomic numbers (``Z = 0`` for
            padding), or ``(N,)`` for single-mol.
        n_atoms_arr: optional ``(B,)`` int per-mol atom count. If
            ``None``, all atoms are assumed real (no padding).
        total_charge: integer net charge for the molecule (or scalar
            broadcast across the batch).
        k_cn: counting-function steepness for the CN computation.
            Default ``7.5`` (GFN0 erf-CN).

    Returns:
        Atomic charges as ``(B, max_atoms)`` (or ``(N,)`` for single-mol
        input). Sums to ``total_charge`` per molecule (over real atoms);
        padded slots return zero.
    """
    was_2d = coords.ndim == 2
    if was_2d:
        coords = coords[None]
        atoms = atoms[None]
        if n_atoms_arr is None:
            n_atoms_arr = mx.array([atoms.shape[1]], dtype=mx.int32)
        elif n_atoms_arr.ndim == 0:
            n_atoms_arr = n_atoms_arr[None]
    B, n_max, _ = coords.shape

    # Per-atom EEQ params via table lookup; padded slots get zeros.
    atom_eeq = mx.take(_EEQ_PARAMS_TABLE, atoms, axis=0)
    chi = atom_eeq[..., 0].astype(coords.dtype)
    eta = atom_eeq[..., 1].astype(coords.dtype)
    kappa = atom_eeq[..., 2].astype(coords.dtype)
    alpha = atom_eeq[..., 3].astype(coords.dtype)

    # Build validity mask once (used for masking the linear system).
    if n_atoms_arr is not None:
        arange_n = mx.arange(n_max, dtype=mx.int32)
        valid = (arange_n[None, :] < n_atoms_arr[:, None]).astype(coords.dtype)
    else:
        valid = mx.ones((B, n_max), dtype=coords.dtype)

    # CN feeds the EEQ RHS.
    cn = coordination_number_erf(coords, atoms, n_atoms_arr, k=k_cn)
    cn_safe = mx.maximum(cn, mx.zeros_like(cn))
    eps = mx.array(1e-30, dtype=coords.dtype)
    chi_tilde = -chi + kappa * mx.sqrt(cn_safe + eps)

    # Pairwise distances IN BOHR (EEQ params are in atomic units;
    # public-API coords are Angstrom). Padded entries get masked below.
    coords_bohr = coords * mx.array(_ANG_TO_BOHR, dtype=coords.dtype)
    diff = coords_bohr[:, :, None, :] - coords_bohr[:, None, :, :]
    R = mx.sqrt(mx.sum(diff * diff, axis=-1) + eps)

    # Coulomb screening: γ_AB = 1 / sqrt(α_A² + α_B²)
    alpha_sq = alpha * alpha
    gamma = 1.0 / mx.sqrt(alpha_sq[:, :, None] + alpha_sq[:, None, :] + eps)

    # J off-diagonal: erf(R · γ) / R; diagonal: η + sqrt(2/π) / α
    eye = mx.eye(n_max, dtype=coords.dtype)
    J_off = mx.erf(R * gamma) / mx.maximum(R, eps)
    J_off = J_off * (1.0 - eye[None, :, :])
    diag_vals = eta + mx.array(_SQRT_2_OVER_PI, dtype=coords.dtype) / mx.maximum(alpha, eps)
    J = J_off + diag_vals[:, :, None] * eye[None, :, :]

    # Mask: keep J for real-real pairs, set padded rows/cols to identity
    # (so q_padded = 0 deterministically without coupling into the
    # constraint).
    pair_valid = valid[:, :, None] * valid[:, None, :]
    invalid_diag = (1.0 - valid)[:, :, None] * eye[None, :, :]
    J = J * pair_valid + invalid_diag

    # Augment with the trace-constraint row/col and Lagrange-multiplier corner.
    cons_col = (valid[:, :, None])                          # (B, n_max, 1)
    cons_row = (valid[:, None, :])                          # (B, 1, n_max)
    corner = mx.zeros((B, 1, 1), dtype=coords.dtype)
    top = mx.concatenate([J, cons_col], axis=-1)            # (B, n_max, n_max+1)
    bot = mx.concatenate([cons_row, corner], axis=-1)       # (B, 1, n_max+1)
    A_aug = mx.concatenate([top, bot], axis=-2)             # (B, n_max+1, n_max+1)

    # RHS: per multicharge eeq.f90:127-150, xvec = -χ + κ · sqrt(CN);
    # the augmented system  [J 1; 1ᵀ 0] [q; λ] = [xvec; q_total]  comes
    # from stationarity of  E = Σ_A (χ_A − κ_A √CN_A) q_A + ½ qᵀ J q
    # subject to Σ q = q_total.
    rhs_chi = (chi_tilde * valid)[:, :, None]               # (B, n_max, 1)
    q_tot_b = mx.full((B, 1, 1), float(total_charge), dtype=coords.dtype)
    rhs = mx.concatenate([rhs_chi, q_tot_b], axis=-2)       # (B, n_max+1, 1)

    sol = solve_lu(A_aug, rhs)                              # (B, n_max+1, 1)
    q = sol[:, :n_max, 0]                                   # drop Lagrange multiplier
    # Ensure padded slots are exactly zero (any solver noise → suppressed).
    q = q * valid

    if was_2d:
        q = q[0]
    return q
