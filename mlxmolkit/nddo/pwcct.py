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
  C-C: extra par1 * exp(-par2 * R_ang) * Z_A*Z_B*gam term, where
       par1 = v_par6(1) = 9.278465 and par2 = v_par6(2) = 5.983752.
       MOPAC src/models/parameters_for_PM6_C.F90 lines 1449-1450 label these
       "scalar/exponent correction of C-C triple bonds"; they are applied in
       src/integrals/ccrep.F90 as `scale = scale + par1*exp(-par2*r)`, i.e.
       inside the factor multiplying Z_A*Z_B*gab, which is the form used here.
"""
from __future__ import annotations

import os
import numpy as np
import math
from typing import Dict, Tuple
from .params import ANG_TO_BOHR

EV = 27.21

# MOPAC's v_par6(1) and v_par6(2), from src/models/parameters_for_PM6_C.F90,
# where they are commented "Used in ccrep for scalar/exponent correction of C-C
# triple bonds". This code previously rounded them to 9.28 and 5.98, which is
# worth ~0.01 eV on a C-C pair at triple-bond range.
_CC_PAR1 = 9.278465
_CC_PAR2 = 5.983752

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
        expo2 += ZA * ZB * gam * _CC_PAR1 * math.exp(-_CC_PAR2 * R_ang)

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


def pm6_pair_repulsion_batch(zi, zj, pair_params, coordI, coordJ, param_dict=None):
    """PM6 core-core repulsion for many pairs at once, in eV.

    Same arithmetic as :func:`pm6_pair_repulsion`; the two element-dependent
    branches — the C/N/O-H special case and the C-C extra term — become masks.

    Element parameters are gathered into Z-indexed tables once and then read by
    fancy indexing. A batch has thousands of pairs but only a handful of
    distinct elements, so pulling attributes off ElementParams per pair costs
    more than the arithmetic it feeds.

    Args:
        zi, zj: (P,) atomic numbers.
        pair_params: sequence of (pA, pB) ElementParams, aligned with zi/zj.
            Ignored when `param_dict` is given.
        param_dict: {Z: ElementParams}. Supplying it skips the per-pair walk
            that would otherwise rediscover which elements the batch holds.
        coordI, coordJ: (P, 3) coordinates in Angstrom.
    """
    zi = np.asarray(zi, dtype=np.int64)
    zj = np.asarray(zj, dtype=np.int64)
    R_ang = np.linalg.norm(np.asarray(coordJ) - np.asarray(coordI), axis=1)
    R_bohr = R_ang * ANG_TO_BOHR

    if param_dict is not None:
        # The caller already has the parameters keyed by element, so there is
        # no reason to walk every pair to rediscover which elements occur.
        by_z = {int(z): param_dict[int(z)]
                for z in np.unique(np.concatenate([zi, zj]))}
    else:
        by_z = {}
        for (pA, pB), a, b in zip(pair_params, zi, zj):
            by_z.setdefault(int(a), pA)
            by_z.setdefault(int(b), pB)
    top = max(by_z) + 1
    n_val = np.zeros(top)
    gss = np.zeros(top)
    gK = np.zeros((top, 4))
    gL = np.zeros((top, 4))
    gM = np.zeros((top, 4))
    for z, p in by_z.items():
        n_val[z] = float(p.n_valence)
        gss[z] = p.gss
        gK[z], gL[z], gM[z] = p.gauss_K[:4], p.gauss_L[:4], p.gauss_M[:4]

    ZA, ZB = n_val[zi], n_val[zj]
    gA, gB = gss[zi], gss[zj]
    rho0A = np.where(gA > 0, 0.5 * EV / np.where(gA > 0, gA, 1.0), 0.0)
    rho0B = np.where(gB > 0, 0.5 * EV / np.where(gB > 0, gB, 1.0), 0.0)
    gam = EV / np.sqrt(R_bohr ** 2 + (rho0A + rho0B) ** 2)

    unpolcore = 1e-8 * ((zi.astype(float) ** (1.0 / 3)
                         + zj.astype(float) ** (1.0 / 3)) / R_ang) ** 12

    # PWCCT is a function of the element pair, so it tabulates the same way
    # n_val and gss do above: a handful of entries read by fancy indexing,
    # rather than a dict walk and two list comprehensions over every pair.
    chi_tab = np.zeros((top, top))
    alp_tab = np.zeros((top, top))
    for a in by_z:
        for b in by_z:
            chi_tab[a, b], alp_tab[a, b] = get_pwcct(a, b)
    chi, alp = chi_tab[zi, zj], alp_tab[zi, zj]

    is_XH = (np.isin(zi, (6, 7, 8)) & (zj == 1)) | (np.isin(zj, (6, 7, 8)) & (zi == 1))
    damp = np.where(is_XH,
                    np.exp(-alp * R_ang ** 2),
                    np.exp(-alp * (R_ang + 0.0003 * R_ang ** 6)))
    expo2 = unpolcore + ZA * ZB * gam * (1.0 + 2.0 * chi * damp)
    expo2 = np.where((zi == 6) & (zj == 6),
                     expo2 + ZA * ZB * gam * _CC_PAR1 * np.exp(-_CC_PAR2 * R_ang), expo2)

    KA, LA, MA = gK[zi], gL[zi], gM[zi]
    KB, LB, MB = gK[zj], gL[zj], gM[zj]
    r = R_ang[:, None]
    t5 = np.where(KA != 0, KA * np.exp(-LA * (r - MA) ** 2), 0.0).sum(axis=1)
    t6 = np.where(KB != 0, KB * np.exp(-LB * (r - MB) ** 2), 0.0).sum(axis=1)
    return expo2 + (ZA * ZB / R_ang) * (t5 + t6)
