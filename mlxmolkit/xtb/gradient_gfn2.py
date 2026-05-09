# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2-xTB total-energy gradient.

Numerical 6N+1 central-difference gradient that drives ANCopt today.
Mirrors :mod:`gradient_gfn1` exactly — same defaults, same accuracy
target. The full analytical Pulay+Hellmann-Feynman path (with the
extra multipole-integral derivative pieces ``∂dpint/∂r`` and
``∂qpint/∂r`` from xtb's ``build_dSDQH0``) is the next step; the
existing GFN0 closed-form pieces (rep, CN, SRB, EEQ) cover most of
the non-SCC ones already.
"""

from __future__ import annotations

import numpy as np

from .scf_gfn2 import gfn2_energy


def numerical_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    h: float = 1e-3,
    scf_kwargs: dict | None = None,
) -> np.ndarray:
    """Central-difference ``∂E_total/∂x`` (Hartree / Å).

    Performs 6N SCF calls with conv_tol=1e-9 so gradient noise stays
    well below ANCopt's gtol = 1e-3 Ha/Bohr.
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
        method: ``"numerical"`` (only choice for now). Analytical path
            is future work — needs ∂(P·H0)/∂r, ∂S/∂r, ∂dpint/∂r,
            ∂qpint/∂r, plus the implicit ∂q/∂r through V_es and AES.

    Returns:
        Dict with ``gradient`` (n_atoms, 3) in Ha/Å, ``energy``,
        and ``n_calls``.
    """
    if method != "numerical":
        raise NotImplementedError(
            f"method={method!r}: only 'numerical' for GFN2 today."
        )
    if scf_kwargs is None:
        scf_kwargs = {}
    e0 = gfn2_energy(atoms, coords_ang, charge=charge, **scf_kwargs)["energy_hartree"]
    grad = numerical_gradient(
        atoms, coords_ang, charge=charge, h=h, scf_kwargs=scf_kwargs,
    )
    return {
        "gradient": grad,
        "energy": e0,
        "n_calls": 6 * len(atoms) + 1,
    }
