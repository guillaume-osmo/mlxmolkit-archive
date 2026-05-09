# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN1-xTB total-energy gradient.

This module provides a working gradient that can drive ANCopt and
finite-difference verification today. It is structured as a numerical
central-difference fallback for the SCC/electronic part, with hooks
prepared for swapping in analytical pieces (Pulay + Hellmann-Feynman
on ∂(P·H0)/∂r, plus the implicit ∂q/∂r through the Coulomb potential)
when those land.

Pieces with analytical contributions already vendored elsewhere:
    - GFN1 classical repulsion → see :mod:`gradient_gfn0.repulsion_gradient`
      (GFN0/GFN1 share the form, only the per-element parameters differ)
    - GFN1 D3-BJ dispersion → its analytical gradient is in xtb's
      ``disp_gradient_neigh`` (dftd3.f90); not yet ported, currently
      caught by the numerical fallback.
    - GFN1 halogen-bond gradient → xtb's ``xbpot`` second loop
      (halogen.f90:90-157); not yet ported.

Use :func:`gfn1_gradient` as the public entry point.
"""

from __future__ import annotations

import numpy as np

from .scf_gfn1 import gfn1_energy


def numerical_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    h: float = 1e-3,
    scf_kwargs: dict | None = None,
) -> np.ndarray:
    """Three-point central-difference gradient ``∂E_total/∂x`` (Hartree / Å).

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.
        charge: integer net charge.
        h: step size in Å for the central difference. ``1e-3`` is a
            good default: gives ~6-digit gradient accuracy on GFN1
            since the energy is float64 and the SCF is tightly
            converged at conv_tol=1e-7.
        scf_kwargs: extra keyword args forwarded to :func:`gfn1_energy`
            on every probe call (e.g. ``{"conv_tol": 1e-9}``).

    Returns:
        ``∇E`` of shape ``(n_atoms, 3)`` in Hartree per Angstrom.

    This routine performs ``6 · n_atoms`` SCF calls and is intended
    primarily as a working gradient until analytical pieces land.
    """
    if scf_kwargs is None:
        scf_kwargs = {}
    # Use a tighter SCF tolerance than the default 1e-6 so that the
    # central-difference round-off doesn't bury the analytical
    # gradient (with conv_tol=1e-6 the SCF is itself only good to ~1e-6
    # Ha; FD with h=1e-3 then gives gradient noise ~1e-3 Ha/Å).
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


def gfn1_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    method: str = "numerical",
    h: float = 1e-3,
    scf_kwargs: dict | None = None,
) -> dict:
    """GFN1-xTB gradient (Hartree / Å).

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.
        charge: integer net charge.
        method: ``"numerical"`` (only choice for now). The hybrid
            analytical-numerical and full-analytical paths are TODO.
        h: central-difference step size (Å) when ``method='numerical'``.
        scf_kwargs: extra keyword args for :func:`gfn1_energy`.

    Returns:
        Dict with keys::

            gradient   :: (n_atoms, 3)  ∂E_total/∂r in Ha/Å
            energy     :: float         total energy at the input geometry
            n_calls    :: int           number of SCF calls used

    Notes:
        Calls :func:`gfn1_energy` once at the input geometry to compute
        the total energy, then 6N more times for the central differences.
        The central-difference is performed with a tightened SCF
        tolerance (``conv_tol=1e-9``) so gradient noise is ~1e-7 Ha/Å.
    """
    if method != "numerical":
        raise NotImplementedError(
            f"method={method!r}: only 'numerical' is implemented for GFN1 today. "
            "Analytical Pulay+Hellmann-Feynman is the next step (review item #8)."
        )
    if scf_kwargs is None:
        scf_kwargs = {}
    e0 = gfn1_energy(atoms, coords_ang, charge=charge, **scf_kwargs)["energy_hartree"]
    grad = numerical_gradient(
        atoms, coords_ang, charge=charge, h=h, scf_kwargs=scf_kwargs,
    )
    n_atoms = len(atoms)
    return {
        "gradient": grad,
        "energy": e0,
        "n_calls": 6 * n_atoms + 1,
    }
