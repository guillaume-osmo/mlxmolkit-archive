"""
PM6 nuclear repulsion with PWCCT (Pairwise Core-Core Terms).

Exact port of PYSEQM energy.py pair_nuclear_energy for PM6.

E_nuc = Σ_{A<B} [
    unpolcore
    + Z_A * Z_B * gam * (1 + 2*chi_{AB}*exp(-alpha_{AB}*f(R)))
    + Z_A * Z_B / R_ang * (gauss_A + gauss_B)
]

where gam = EV / sqrt(R_bohr² + (rho0_A + rho0_B)²)
      unpolcore = 1e-8 * ((Z_A^(1/3) + Z_B^(1/3)) / R_ang)^12
      f(R) = R_ang + 0.0003*R_ang^6  (general)
      f(R) = R_ang^2                  (C-H, N-H, O-H special)

Special cases:
  C-C: extra 9.28 * exp(-5.98 * R_ang) * Z_A*Z_B*gam term
"""
from __future__ import annotations

import os
import numpy as np
import math
from typing import Dict, Tuple
from .params import ANG_TO_BOHR

EV = 27.21

# PWCCT parameters cache
_PWCCT_CACHE: Dict[Tuple[int, int], Tuple[float, float]] = {}
_PWCCT_LOADED = False


def _load_pwcct(filepath: str = None):
    global _PWCCT_CACHE, _PWCCT_LOADED
    if _PWCCT_LOADED:
        return
    if filepath is None:
        # Bundled CSV in mlxmolkit/nddo/data/ (extracted from PYSEQM
        # seqm/params/, BSD-3-Clause)
        filepath = os.path.join(
            os.path.dirname(__file__), 'data', 'PWCCT_PM6_MOPAC.csv'
        )
    try:
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    z1, z2 = int(parts[0]), int(parts[1])
                    # PYSEQM convention: column 2 → alp, column 3 → chi (SWAPPED!)
                    alp = float(parts[2])   # stored as chi in CSV but used as alp
                    chi = float(parts[3])   # stored as alpha in CSV but used as chi
                    _PWCCT_CACHE[(z1, z2)] = (chi, alp)
                    if z1 != z2:
                        _PWCCT_CACHE[(z2, z1)] = (chi, alp)
        _PWCCT_LOADED = True
    except FileNotFoundError:
        _PWCCT_LOADED = True


def get_pwcct(z1: int, z2: int) -> Tuple[float, float]:
    _load_pwcct()
    return _PWCCT_CACHE.get((z1, z2), (0.0, 0.0))


def pm6_pair_repulsion(zi: int, zj: int, pA, pB,
                       coordI: np.ndarray, coordJ: np.ndarray) -> float:
    """PM6 core-core repulsion for ONE atom pair, in eV.

    Split out of :func:`pm6_nuclear_repulsion` because the total is a plain sum
    over pairs: displacing one atom changes only the pairs that touch it, so a
    gradient can update N-1 terms rather than rebuilding all N(N-1)/2. On
    menthol the full rebuild was 46% of a gradient evaluation.
    """
    R_ang = float(np.linalg.norm(coordJ - coordI))
    R_bohr = R_ang * ANG_TO_BOHR

    ZA = float(pA.n_valence)
    ZB = float(pB.n_valence)

    rho0A = 0.5 * EV / pA.gss if pA.gss > 0 else 0.0
    rho0B = 0.5 * EV / pB.gss if pB.gss > 0 else 0.0
    gam = EV / np.sqrt(R_bohr ** 2 + (rho0A + rho0B) ** 2)

    unpolcore = 1e-8 * ((float(zi) ** (1.0/3) + float(zj) ** (1.0/3)) / R_ang) ** 12
    chi, alp = get_pwcct(zi, zj)

    is_XH = ((zi in (6, 7, 8)) and zj == 1) or ((zj in (6, 7, 8)) and zi == 1)
    if is_XH:
        expo2 = unpolcore + ZA * ZB * gam * (
            1.0 + 2.0 * chi * math.exp(-alp * R_ang ** 2))
    else:
        expo2 = unpolcore + ZA * ZB * gam * (
            1.0 + 2.0 * chi * math.exp(-alp * (R_ang + 0.0003 * R_ang ** 6)))

    if zi == 6 and zj == 6:
        expo2 += ZA * ZB * gam * 9.28 * math.exp(-5.98 * R_ang)

    t4 = ZA * ZB / R_ang
    t5 = sum(pA.gauss_K[k] * math.exp(-pA.gauss_L[k] * (R_ang - pA.gauss_M[k]) ** 2)
             for k in range(4) if pA.gauss_K[k] != 0)
    t6 = sum(pB.gauss_K[k] * math.exp(-pB.gauss_L[k] * (R_ang - pB.gauss_M[k]) ** 2)
             for k in range(4) if pB.gauss_K[k] != 0)
    return expo2 + t4 * (t5 + t6)


def pm6_nuclear_repulsion(
    atoms: list[int],
    coords: np.ndarray,
    param_dict: dict,
) -> float:
    """PM6 nuclear repulsion energy. Exact PYSEQM formula."""
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)
    return float(sum(
        pm6_pair_repulsion(atoms[i], atoms[j],
                           param_dict[atoms[i]], param_dict[atoms[j]],
                           coords[i], coords[j])
        for i in range(n_atoms) for j in range(i + 1, n_atoms)))
