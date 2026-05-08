# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN0-xTB classical pairwise repulsion energy.

Functional form (xtb/src/peeq_module.f90:684-702):

    E_rep = Σ_{A<B} Z_A^eff · Z_B^eff / R_AB · exp(−α_AB · R_AB^kexp)
    α_AB  = sqrt(α_A · α_B) · (1 + (0.01·Δχ² + 0.01·Δχ⁴) · renscale)

GFN0 globals (from $globpar): ``kexp = 1.5``, ``renscale = -0.09``.
Per-element: ``α_A`` (rep_alpha), ``Z_A^eff`` (rep_zeff), ``en_A``.
Atomic units throughout (Bohr / Hartree).
"""

from __future__ import annotations

import numpy as np

from .params_gfn0 import GFN0_GLOBALS, GFN0_PARAMS

_ANG_TO_BOHR = 1.8897259886


def compute_repulsion(atoms: list[int], coords_ang: np.ndarray) -> float:
    """Single-molecule repulsion energy in Hartree.

    Args:
        atoms: list/array of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom positions.

    Returns:
        Scalar ``E_rep`` in Hartree.
    """
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n_atoms = len(atoms)
    if n_atoms < 2:
        return 0.0

    g = GFN0_GLOBALS
    kexp = g.kexp
    renscale = g.renscale

    alpha = np.array([GFN0_PARAMS[int(z)].rep_alpha for z in atoms], dtype=np.float64)
    zeff = np.array([GFN0_PARAMS[int(z)].rep_zeff for z in atoms], dtype=np.float64)
    en = np.array([GFN0_PARAMS[int(z)].en for z in atoms], dtype=np.float64)

    E = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            R = float(np.linalg.norm(coords[i] - coords[j]))
            d_chi = en[i] - en[j]
            d_chi2 = d_chi * d_chi
            enmod = 1.0 + (0.01 * d_chi2 + 0.01 * d_chi2 * d_chi2) * renscale
            alpha_AB = float(np.sqrt(alpha[i] * alpha[j])) * enmod
            E += zeff[i] * zeff[j] / R * float(np.exp(-alpha_AB * R**kexp))
    return E
