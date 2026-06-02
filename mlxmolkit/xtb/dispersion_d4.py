# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2-xTB D4 dispersion (Caldeweyher et al. 2019, JCP 150:154122).

Two backends:

1. ``backend='simple-dftd4'`` (default) — calls the
   :mod:`dftd4-python` library (installed via conda or pip). This is
   the *reference* path that lets us run GFN2 end-to-end today; the
   C6 reference data and ATM three-body machinery are inside the
   ``libdftd4`` shared library.

2. ``backend='mlx-native'`` — pure-numpy/MLX implementation. **Not yet
   implemented.** This will require parsing
   ``xtb/include/param_ref.fh`` (~6300 lines of D4 reference C6 / α(iω)
   tables for 118 elements with up to 7 reference systems each) and
   re-emitting it as a numpy ``.npz``. Tracked under the strategic
   roadmap as the GFN2 “make hot path 100% MLX” task.

GFN2 D4 damping parameters (gfn2.f90:65-66):

    s6 = 1.0   s8 = 2.7   a1 = 0.52   a2 = 5.0   s9 = 5.0

The s9 coefficient enables the Axilrod-Teller-Muto three-body term.
GFN2 uses the SCF-converged Mulliken charges (passed via ``q``) to
drive the charge-dependent C6 interpolation. If ``q`` is None the
backend uses its own EEQ-based charges (small difference vs. the
SCF-converged result on most molecules).
"""

from __future__ import annotations

import numpy as np

GFN2_D4_PARAMS = {"s6": 1.0, "s8": 2.7, "a1": 0.52, "a2": 5.0, "s9": 5.0}
_ANG_TO_BOHR = 1.8897259886


def d4_dispersion_gfn2(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    cn: np.ndarray | None = None,
    q: np.ndarray | None = None,
    *,
    backend: str = "simple-dftd4",
) -> float:
    """GFN2 D4 dispersion energy in Hartree.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.
        cn, q: ignored by the simple-dftd4 backend (it computes its
            own internal CN and EEQ charges). Reserved for the future
            mlx-native path that will accept SCF-converged inputs.
        backend: ``'simple-dftd4'`` (default) or ``'mlx-native'`` (NotImplemented).

    Returns:
        Total dispersion energy in Hartree (negative = attractive).
    """
    if backend == "simple-dftd4":
        return _d4_simple_dftd4(atoms, coords_ang)
    elif backend == "mlx-native":
        raise NotImplementedError(
            "mlx-native D4 backend not yet implemented. Vendoring the "
            "param_ref.fh reference data (~6300 lines) is the next step."
        )
    else:
        raise ValueError(f"unknown backend: {backend!r}")


def _d4_simple_dftd4(atoms, coords_ang) -> float:
    """Backend dispatch via the ``dftd4`` Python package."""
    try:
        from dftd4.interface import DispersionModel, DampingParam
    except ImportError as e:
        raise ImportError(
            "simple-dftd4 backend requires the 'dftd4' Python package. "
            "Install via 'conda install -c conda-forge dftd4-python' "
            "or 'pip install dftd4'."
        ) from e
    nums = np.asarray(atoms, dtype=np.int32)
    pos_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    m = DispersionModel(nums, pos_bohr)
    p = DampingParam(**GFN2_D4_PARAMS)
    res = m.get_dispersion(p, grad=False)
    return float(res["energy"])


def d4_dispersion_gradient_gfn2(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    backend: str = "simple-dftd4",
) -> np.ndarray:
    """GFN2 D4 dispersion gradient in Hartree / Angstrom."""

    if backend != "simple-dftd4":
        raise ValueError(f"gradient backend not available: {backend!r}")
    try:
        from dftd4.interface import DispersionModel, DampingParam
    except ImportError as e:
        raise ImportError(
            "simple-dftd4 backend requires the 'dftd4' Python package. "
            "Install via 'conda install -c conda-forge dftd4-python' "
            "or 'pip install dftd4'."
        ) from e

    nums = np.asarray(atoms, dtype=np.int32)
    pos_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    m = DispersionModel(nums, pos_bohr)
    p = DampingParam(**GFN2_D4_PARAMS)
    res = m.get_dispersion(p, grad=True)
    return np.asarray(res["gradient"], dtype=np.float64) * _ANG_TO_BOHR
