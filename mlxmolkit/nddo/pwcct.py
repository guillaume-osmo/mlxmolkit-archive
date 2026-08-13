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


# MOPAC applies a flat +12 kcal/mol per C≡C to the heat of formation, and it is
# NOT part of the SCF energy — which is why an mlxmolkit total energy can agree
# with MOPAC's own ENPART ETOT to 0.003 eV while the heat of formation is 12
# kcal/mol out. See issue #33.
C_TRIPLE_BOND_KCAL = 12.0
_CTB_RMIN = 1.21
_CTB_RMAX = 1.33
_CTB_PARAM1 = -5.0
_CTB_PARAM2 = 25.0


def c_triple_bond_correction(atoms, coords) -> float:
    """MOPAC's `C_triple_bond_C`, in kcal/mol.

    Port of openMOPAC v23 ``src/corrections/set_up_dentate.F90``. Counts C-C
    bonds short enough to be acetylenic — full weight below 1.21 A, tapering to
    zero at 1.33 A through the quintic-sextic switch MOPAC uses — and multiplies
    by 12, a value its own comment calls empirical. ``src/compfg.F90`` adds the
    result to ``atheat``, the atomic-heat term of the heat of formation.

    Because it is a *switch* rather than a decaying function, no smooth term in
    R can imitate it: it is essentially a step between 1.21 and 1.33 A. That is
    why an alkyne at 1.20 A takes the full 12 kcal/mol and an alkene at 1.34 A
    takes none, and why fitting an exponential or an R^-12 to the two distances
    is hopeless.

    MOPAC walks its own bond list; any C-C pair inside 1.33 A is bonded, so
    iterating over close pairs is equivalent and needs no connectivity.

    Args:
        atoms: atomic numbers.
        coords: (n, 3) in Angstrom.

    Returns:
        The correction in kcal/mol — zero for anything without a short C-C bond.
    """
    carbons = [i for i, z in enumerate(atoms) if int(z) == 6]
    if len(carbons) < 2:
        return 0.0
    xyz = np.asarray(coords, dtype=np.float64)
    total = 0.0
    for a in range(len(carbons)):
        for b in range(a):
            r = float(np.linalg.norm(xyz[carbons[a]] - xyz[carbons[b]]))
            if r < _CTB_RMIN:
                total += 1.0
            elif r < _CTB_RMAX:
                t = (r - _CTB_RMIN) / (_CTB_RMAX - _CTB_RMIN)
                total += (1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
                          + (_CTB_PARAM1 + t * _CTB_PARAM2)
                          * (t ** 3 - 3.0 * t ** 4 + 3.0 * t ** 5 - t ** 6))
    return total * C_TRIPLE_BOND_KCAL


# The remaining two molecular-mechanics corrections MOPAC adds to `atheat` for
# PM6, from src/compfg.F90:
#
#     if (method_pm6 .and. N_3_present) atheat = atheat + nsp2_correction()
#     atheat = atheat + sum_dihed          ! htype*sin(angle)**2 over O=C-N-H
#
# Neither exists in PYSEQM, so the vendored port could never have carried them.
HTYPE_PM6 = 2.5000   # moldat.F90: `if (method_pm6) htype = 2.5000D0`


def _bonded(atoms, coords, scale: float = 1.25):
    """Neighbour lists from covalent radii.

    MOPAC carries its own `nbonds`/`ibonds`; this reconstructs the same
    connectivity from geometry, which is what the corrections below need in
    order to ask "does this nitrogen have exactly three ligands".
    """
    from .pm6_d3h4 import RCOV

    xyz = np.asarray(coords, dtype=np.float64)
    n = len(atoms)
    radii = np.array([RCOV[int(z)] if int(z) < len(RCOV) else 1.5 for z in atoms])
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(xyz[i] - xyz[j]) < scale * (radii[i] + radii[j]):
                neighbours[i].append(j)
                neighbours[j].append(i)
    return neighbours


def nsp2_correction(atoms, coords) -> float:
    """MOPAC's `nsp2_correction`, in kcal/mol.

    Port of openMOPAC v23 ``src/corrections/set_up_dentate.F90``. Every nitrogen
    with exactly three ligands, fewer than two of them hydrogen, is penalised by
    ``-0.5 * exp(-10 * tot)`` where ``tot`` is how far the three bond angles fall
    short of summing to 2*pi — i.e. how far from planar the centre is.

    So a planar three-coordinate nitrogen — pyrrole, indole, an amide, a nitro
    group, aniline — contributes the full -0.5 kcal/mol, and a pyramidal one
    contributes almost nothing. That is the signature seen in the parity set:
    after the C≡C fix the nine worst molecules were all planar-nitrogen species.

    Args:
        atoms: atomic numbers.
        coords: (n, 3) in Angstrom.

    Returns:
        The correction in kcal/mol, zero when no qualifying nitrogen is present.
    """
    xyz = np.asarray(coords, dtype=np.float64)
    neighbours = _bonded(atoms, xyz)
    total = 0.0
    for i, z in enumerate(atoms):
        if int(z) != 7 or len(neighbours[i]) != 3:
            continue
        ligands = neighbours[i]
        if sum(1 for j in ligands if int(atoms[j]) == 1) >= 2:
            continue
        angles = 0.0
        for a in range(3):
            for b in range(a):
                u = xyz[ligands[a]] - xyz[i]
                v = xyz[ligands[b]] - xyz[i]
                cosine = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
                angles += math.acos(max(-1.0, min(1.0, cosine)))
        total += -0.5 * math.exp(-10.0 * (2.0 * math.pi - angles))
    return total


def nhco_dihedral_correction(atoms, coords, htype: float = HTYPE_PM6) -> float:
    """MOPAC's `sum_dihed` over O=C-N-H linkages, in kcal/mol.

    Port of ``setup_nhco`` in openMOPAC v23 ``src/moldat.F90`` plus the sum in
    ``src/compfg.F90``: identify O=C-N-H systems by distance (C-O <= 1.3,
    C-N <= 1.6, N-H <= 1.3, N-X <= 1.7), then add ``htype * sin(angle)**2`` for
    both the O=C-N-X and O=C-N-H dihedrals. ``htype`` is 2.5 for PM6.

    It is a *planarity* penalty on the amide: a planar linkage has both
    dihedrals at 0 or 180 degrees, where sin**2 is zero, so a well-behaved amide
    pays nothing. Tertiary amides have no N-H and are skipped entirely — for
    those, :func:`nsp2_correction` is the whole story.
    """
    xyz = np.asarray(coords, dtype=np.float64)
    z = [int(a) for a in atoms]
    n = len(z)
    dist = lambda a, b: float(np.linalg.norm(xyz[a] - xyz[b]))

    quads, claimed = [], set()
    for j in range(n):                                   # carbon
        if z[j] != 6:
            continue
        found = False
        for i in range(n):                               # oxygen
            if z[i] != 8 or dist(i, j) > 1.3:
                continue
            for k in range(n):                           # nitrogen
                if z[k] != 7 or dist(k, j) > 1.6:
                    continue
                for l in range(n):                       # hydrogen on N
                    if z[l] != 1 or dist(k, l) > 1.3:
                        continue
                    for m in range(n):                   # the other substituent
                        if m in (k, l, j) or dist(m, k) > 1.7:
                            continue
                        if k in claimed:                 # one entry per nitrogen
                            continue
                        claimed.add(k)
                        quads.append((i, j, k, m))
                        quads.append((i, j, k, l))
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break

    total = 0.0
    for i, j, k, l in quads:
        b1 = xyz[j] - xyz[i]
        b2 = xyz[k] - xyz[j]
        b3 = xyz[l] - xyz[k]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        angle = math.atan2(float(m1 @ n2), float(n1 @ n2))
        total += htype * math.sin(angle) ** 2
    return total


# Post-SCF corrections that distinguish the PM6 dispersion variants. Every one
# of these is a function of geometry alone — none touches the density, the
# Mulliken charges or a COSMO sigma profile — so the variants share PM6's
# parameters and SCF exactly and differ only in the heat of formation.
DISPERSION_METHODS = frozenset({"PM6_D3", "PM6_D3H4", "PM6_D3H4X"})


def normalize_method(method: str) -> str:
    """Canonical method name — the same spelling `get_params` accepts.

    `get_params` normalises internally, so `nddo_energy(method="PM6-D3")` picks
    up the right parameters, but the raw string is what reaches the core-core
    and correction checks. Without this, "PM6-D3" missed both frozensets, fell
    through to the AM1-style core-core, and came out 185 kcal/mol wrong while
    "PM6_D3" was right.
    """
    return str(method).upper().replace("-", "_").replace("*", "_STAR")


def dispersion_correction(atoms, coords, method) -> float:
    """Dispersion / H-bond / halogen correction for a PM6 variant, kcal/mol.

    Returns 0.0 for plain PM6 and for any non-PM6 method, so callers can apply
    it unconditionally.

        PM6-D3      D3 with Becke-Johnson damping
        PM6-D3H4    D3 + the H4 hydrogen-bond term + H-H repulsion
        PM6-D3H4X   the above plus the halogen-bond term
    """
    method = normalize_method(method)
    if method not in DISPERSION_METHODS:
        return 0.0
    from .pm6_d3h4 import (PM6_D3_DISP, d3_energy, pm6_d3h4_correction,
                           pm6_d3h4x_correction)
    if method == "PM6_D3":
        # MOPAC's PM6-D3 is zero-damping with its own hard-wired parameters,
        # not Becke-Johnson — see PM6_D3_DISP.
        return float(d3_energy(atoms, coords, params=PM6_D3_DISP)["e_disp"])
    if method == "PM6_D3H4":
        return float(pm6_d3h4_correction(atoms, coords)["e_total"])
    return float(pm6_d3h4x_correction(atoms, coords)["e_total"])
