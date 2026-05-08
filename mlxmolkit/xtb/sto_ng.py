# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT
#
# Verbatim port of xtb's slaterToGauss tables (LGPL-3.0,
# `grimme-lab/xtb/src/slater.f90`). All STO-NG primitive expansions are
# from Robert F. Stewart, J. Chem. Phys. 52, 431-438 (1970), but the
# specific numerical values and the (n, l) → ityp index mapping match
# xtb's tables exactly. This is what GFN0/GFN1/GFN2 actually use.

"""xtb-faithful STO-NG primitive Gaussian expansions of Slater orbitals.

Verbatim port of ``xtb/src/slater.f90`` ``pAlphaN`` / ``pCoeffN``
tables for ng = 2..6. The xtb (n, l) → ityp mapping is

    ityp:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
       n:  1  2  3  4  5  2  3  4  5  3  4  5  4  5  5
       l:  0  0  0  0  0  1  1  1  1  2  2  2  3  3  4

so ityp = n for s (l=0), 4 + n for p (l=1), 7 + n for d (l=2).

The expansion convention: ``α_i = pAlpha * ζ²``, with optional Cartesian
Gaussian normalization applied to the coefficients via

    c_i ← c_i · (2α/π)^(3/4) · sqrt(4α)^l / sqrt((2l+1)!!_table)

where ``(2l+1)!!_table = [1, 1, 3, 15, 105, 945, ...]`` (xtb's
``dfactorial`` array, indexed from 1). For Cartesian s and p these
factors collapse to 1 in the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Cartesian "double-factorial" table (xtb's dfactorial, 1-indexed in
# Fortran; we use 0-indexed Python).
_DFACTORIAL = np.array(
    [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]
)


# (n, l) → ityp index (1..15) per the table in xtb/src/slater.f90.
def _ityp(n: int, l: int) -> int:
    if l == 0:
        return n
    if l == 1:
        return 4 + n
    if l == 2:
        return 7 + n
    if l == 3:
        return 9 + n
    raise NotImplementedError(f"l={l} not supported (only s, p, d, f)")


# ---------------------------------------------------------------------------
# pAlpha2 / pCoeff2 — STO-2G fits, unit ζ. Layout: tables[ityp_index] =
# (alphas, coeffs); ityp_index is 0-based here (= xtb's ityp - 1).
# Verbatim from slater.f90:63-98.
# ---------------------------------------------------------------------------

_PALPHA2 = (
    (8.518186635e-1, 1.516232927e-1),   # 1s
    (1.292278611e-1, 4.908584205e-2),   # 2s
    (6.694095822e-1, 5.837135094e-2),   # 3s
    (2.441785453e-1, 4.051097664e-2),   # 4s
    (1.213425654e-1, 3.133152144e-2),   # 5s
    (4.323908358e-1, 1.069439065e-1),   # 2p
    (1.458620964e-1, 5.664210742e-2),   # 3p
    (6.190052680e-2, 2.648418407e-2),   # 4p
    (2.691294191e-1, 3.980805011e-2),   # 5p
    (2.777427345e-1, 8.336507714e-2),   # 3d
    (1.330958892e-1, 5.272119659e-2),   # 4d
    (6.906014388e-2, 3.399457777e-2),   # 5d
    (2.006693538e-1, 6.865384900e-2),   # 4f
    (1.156094555e-1, 4.778940916e-2),   # 5f
    (1.554531559e-1, 5.854079811e-2),   # 5g
)

_PCOEFF2 = (
    (4.301284983e-1, 6.789135305e-1),   # 1s
    (7.470867124e-1, 2.855980556e-1),   # 2s
    (-1.529645716e-1, 1.051370110e+0),  # 3s
    (-3.046656896e-1, 1.146877294e+0),  # 4s
    (-5.114756049e-1, 1.307377277e+0),  # 5s
    (4.522627513e-1, 6.713122642e-1),   # 2p
    (5.349653144e-1, 5.299607212e-1),   # 3p
    (8.743116767e-1, 1.513640107e-1),   # 4p
    (-1.034227010e-1, 1.033376378e+0),  # 5p
    (4.666137923e-1, 6.644706516e-1),   # 3d
    (4.932764167e-1, 5.918727866e-1),   # 4d
    (6.539405185e-1, 3.948945302e-1),   # 5d
    (4.769346276e-1, 6.587383976e-1),   # 4f
    (4.856637346e-1, 6.125980914e-1),   # 5f
    (4.848298074e-1, 6.539381621e-1),   # 5g
)

# ---------------------------------------------------------------------------
# pAlpha3 / pCoeff3 — STO-3G fits. Verbatim from slater.f90:102-137.
# ---------------------------------------------------------------------------

_PALPHA3 = (
    (2.227660584e+0, 4.057711562e-1, 1.098175104e-1),    # 1s
    (2.581578398e+0, 1.567622104e-1, 6.018332272e-2),    # 2s
    (5.641487709e-1, 6.924421391e-2, 3.269529097e-2),    # 3s
    (2.267938753e-1, 4.448178019e-2, 2.195294664e-2),    # 4s
    (1.080198458e-1, 4.408119382e-2, 2.610811810e-2),    # 5s
    (9.192379002e-1, 2.359194503e-1, 8.009805746e-2),    # 2p
    (2.692880368e+0, 1.489359592e-1, 5.739585040e-2),    # 3p
    (4.859692220e-1, 7.430216918e-2, 3.653340923e-2),    # 4p
    (2.127482317e-1, 4.729648620e-2, 2.604865324e-2),    # 5p
    (5.229112225e-1, 1.639595876e-1, 6.386630021e-2),    # 3d
    (1.777717219e-1, 8.040647350e-2, 3.949855551e-2),    # 4d
    (4.913352950e-1, 7.329090601e-2, 3.594209290e-2),    # 5d
    (3.483826963e-1, 1.249380537e-1, 5.349995725e-2),    # 4f
    (1.649233885e-1, 7.487066646e-2, 3.735787219e-2),    # 5f
    (2.545432122e-1, 1.006544376e-1, 4.624463922e-2),    # 5g
)

_PCOEFF3 = (
    (1.543289673e-1, 5.353281423e-1, 4.446345422e-1),    # 1s
    (-5.994474934e-2, 5.960385398e-1, 4.581786291e-1),   # 2s
    (-1.782577972e-1, 8.612761663e-1, 2.261841969e-1),   # 3s
    (-3.349048323e-1, 1.056744667e+0, 1.256661680e-1),   # 4s
    (-6.617401158e-1, 7.467595004e-1, 7.146490945e-1),   # 5s
    (1.623948553e-1, 5.661708862e-1, 4.223071752e-1),    # 2p
    (-1.061945788e-2, 5.218564264e-1, 5.450015143e-1),   # 3p
    (-6.147823411e-2, 6.604172234e-1, 3.932639495e-1),   # 4p
    (-1.389529695e-1, 8.076691064e-1, 2.726029342e-1),   # 5p
    (1.686596060e-1, 5.847984817e-1, 4.056779523e-1),    # 3d
    (2.308552718e-1, 6.042409177e-1, 2.595768926e-1),    # 4d
    (-2.010175008e-2, 5.899370608e-1, 4.658445960e-1),   # 5d
    (1.737856685e-1, 5.973380628e-1, 3.929395614e-1),    # 4f
    (1.909729355e-1, 6.146060459e-1, 3.059611271e-1),    # 5f
    (1.780980905e-1, 6.063757846e-1, 3.828552923e-1),    # 5g
)

# ---------------------------------------------------------------------------
# pAlpha4 / pCoeff4 — STO-4G. Verbatim from slater.f90:141-206.
# ---------------------------------------------------------------------------

_PALPHA4 = (
    (5.216844534e+0, 9.546182760e-1, 2.652034102e-1, 8.801862774e-2),    # 1s
    (1.161525551e+1, 2.000243111e+0, 1.607280687e-1, 6.125744532e-2),    # 2s
    (1.513265591e+0, 4.262497508e-1, 7.643320863e-2, 3.760545063e-2),    # 3s
    (3.242212833e-1, 1.663217177e-1, 5.081097451e-2, 2.829066600e-2),    # 4s
    (8.602284252e-1, 1.189050200e-1, 3.446076176e-2, 1.974798796e-2),    # 5s
    (1.798260992e+0, 4.662622228e-1, 1.643718620e-1, 6.543927065e-2),    # 2p
    (1.853180239e+0, 1.915075719e-1, 8.655487938e-2, 4.184253862e-2),    # 3p
    (1.492607880e+0, 4.327619272e-1, 7.553156064e-2, 3.706272183e-2),    # 4p
    (3.962838833e-1, 1.838858552e-1, 4.943555157e-2, 2.750222273e-2),    # 5p
)

_PCOEFF4 = (
    (5.675242080e-2, 2.601413550e-1, 5.328461143e-1, 2.916254405e-1),    # 1s
    (-1.198411747e-2, -5.472052539e-2, 5.805587176e-1, 4.770079976e-1),  # 2s
    (-3.295496352e-2, -1.724516959e-1, 7.518511194e-1, 3.589627317e-1),  # 3s
    (-1.120682822e-1, -2.845426863e-1, 8.909873788e-1, 3.517811205e-1),  # 4s
    (1.103657561e-2, -5.606519023e-1, 1.179429987e+0, 1.734974376e-1),   # 5s
    (5.713170255e-2, 2.857455515e-1, 5.517873105e-1, 2.632314924e-1),    # 2p
    (-1.434249391e-2, 2.755177589e-1, 5.846750879e-1, 2.144986514e-1),   # 3p
    (-6.035216774e-3, -6.013310874e-2, 6.451518200e-1, 4.117923820e-1),  # 4p
    (-1.801459207e-2, -1.360777372e-1, 7.533973719e-1, 3.409304859e-1),  # 5p
)


@dataclass(frozen=True)
class STONGShell:
    n: int
    l: int
    n_gauss: int
    alphas: tuple[float, ...]
    coeffs: tuple[float, ...]


def _table_lookup(tables_alpha, tables_coeff, n: int, l: int) -> tuple[tuple, tuple]:
    """Index into a per-ityp xtb table; returns (alphas, coeffs) tuple."""
    ityp = _ityp(n, l)            # 1-based xtb index
    idx = ityp - 1                # 0-based Python index
    if idx >= len(tables_alpha):
        raise KeyError(f"No expansion at this N for (n={n}, l={l}); ityp={ityp}")
    return tables_alpha[idx], tables_coeff[idx]


def get_sto_ng(n: int, l: int, n_gauss: int) -> STONGShell:
    """Look up xtb's STO-NG primitive expansion for ``(n, l)`` with N
    Gaussians. Coefficients and exponents are unit-ζ (multiply alphas by
    ζ² at use time and apply primitive normalization to coefficients).

    Raises:
        NotImplementedError: if ``n_gauss`` is outside the {2, 3, 4} we
            currently vendor (STO-6G is in xtb but mostly for n=6).
        KeyError: if ``(n, l)`` is outside the supported table region.
    """
    if n_gauss == 2:
        a, c = _table_lookup(_PALPHA2, _PCOEFF2, n, l)
    elif n_gauss == 3:
        a, c = _table_lookup(_PALPHA3, _PCOEFF3, n, l)
    elif n_gauss == 4:
        a, c = _table_lookup(_PALPHA4, _PCOEFF4, n, l)
    else:
        raise NotImplementedError(
            f"STO-{n_gauss}G not yet vendored (port from slater.f90 if needed)"
        )
    return STONGShell(n=n, l=l, n_gauss=n_gauss, alphas=a, coeffs=c)


# Backward-compat alias (basis.py uses get_sto3g historically).
def get_sto3g(n: int, l: int) -> STONGShell:
    return get_sto_ng(n, l, 3)


# Cartesian-Gaussian primitive normalizations matching xtb/slater.f90:
# coeff_used = coeff_table * (2α/π)^(3/4) * sqrt(4α)^l / sqrt((2l+1)!!_table)
# For Cartesian s and p, (2l+1)!!_table = 1 so the denominator collapses.
def primitive_norm_s(alpha) -> np.ndarray:
    """Cartesian s-Gaussian normalization: ``(2α/π)^(3/4)``."""
    return (2.0 * np.asarray(alpha) / np.pi) ** 0.75


def primitive_norm_p(alpha) -> np.ndarray:
    """Cartesian p-Gaussian normalization: ``(2α/π)^(3/4) · sqrt(4α)``.

    The angular factor of ``1/sqrt(dfactorial(l+1))`` collapses to 1
    for Cartesian p in xtb's dfactorial table (``dfactorial(2) = 1``).
    """
    a = np.asarray(alpha)
    return (2.0 * a / np.pi) ** 0.75 * np.sqrt(4.0 * a)


def gfn0_n_gauss(Z: int, l: int, n_principal: int, is_valence: bool) -> int:
    """xtb's setGFN0NumberOfPrimitives rule (gfn0.f90:847-877).

    H, He: valence s = 3, aux s = 2; p = 3.
    Z >= 3 and n <= 5: s = 4; p = 3; d = 4.
    Z >= 3 and n > 5:  s = 6; p = 6; d = 4.
    """
    if Z <= 2:
        if l == 0:
            return 3 if is_valence else 2
        return 3
    if n_principal > 5:
        if l == 2:
            return 4
        return 6
    # n <= 5
    if l == 0:
        return 4
    if l == 1:
        return 3
    if l == 2:
        return 4
    return 4   # f, g — fallback
