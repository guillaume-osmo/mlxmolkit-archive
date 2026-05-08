# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT
#
# Standard STO-NG primitive expansions (Stewart 1969, J. Chem. Phys.
# 52, 431). Coefficients tabulated for unit-Slater-exponent (ζ=1)
# orbitals; scale exponents by ζ² and re-normalize at use time.

"""STO-3G primitive Gaussian expansion of Slater-type orbitals.

The STO-NG approach fits each Slater orbital ``r^(n−1) exp(−ζ r) Y_lm``
by a contraction of N spherical Gaussian primitives:

    χ_nl(r) = Σ_i c_i · g_i(r)
    g_i(r)  = norm * r^l * exp(−α_i r²) Y_lm

with ``α_i = α_i^unit · ζ²``. The unit-ζ tables below are from
Stewart 1969 (Table I-III). For Phase A we ship STO-3G for the
shells GFN0-xTB needs in CHNO chemistry: 1s, 2s, 2p, 3s, 3p (3d
deferred — only matters for heavy elements).

For each shell we pre-multiply Stewart's coefficients by the
primitive Gaussian normalization constant so the resulting contracted
function is normalized for ζ=1 in Cartesian-Gaussian convention. The
Slater scaling factor ζ^(n+1/2) needed for the contracted orbital
norm at general ζ is applied at overlap-build time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Stewart 1969 Table I (s shells) and Table III (p shells), STO-3G,
# unit-Slater-exponent (ζ=1) primitive expansions.
# Each entry: (alpha_i, coeff_i) for the 3 primitives.
# Source: J. Chem. Phys. 52, 431 (1970), Tables I-III.

_STO_2G_TABLES = {
    # Stewart 1969 STO-2G fits, unit ζ.
    (1, 0): [   # 1s
        (0.151623, 0.678914),
        (0.851819, 0.430129),
    ],
    (2, 0): [   # 2s
        (0.0974545, 0.963782),
        (0.384244, 0.0494718),
    ],
    (2, 1): [   # 2p
        (0.0974545, 0.612820),
        (0.384244, 0.448070),
    ],
    (3, 0): [   # 3s
        (0.0407755, 0.794530),
        (0.190771, 0.213907),
    ],
    (3, 1): [   # 3p
        (0.0407755, 0.413000),
        (0.190771, 0.624658),
    ],
}


_STO_3G_TABLES = {
    # (n, l): [(alpha, coeff), ...]
    (1, 0): [   # 1s
        (0.109818, 0.444635),
        (0.405771, 0.535328),
        (2.227660, 0.154329),
    ],
    (2, 0): [   # 2s
        (0.0751386, 0.700115),
        (0.231031,  0.399513),
        (0.994203, -0.0999672),
    ],
    (2, 1): [   # 2p
        (0.0751386, 0.391957),
        (0.231031,  0.607684),
        (0.994203,  0.155916),
    ],
    (3, 0): [   # 3s
        (0.0383884, 0.701176),
        (0.114681,  0.400142),
        (0.448441, -0.0999535),
    ],
    (3, 1): [   # 3p
        (0.0383884, 0.388442),
        (0.114681,  0.609776),
        (0.448441,  0.156876),
    ],
    (3, 2): [   # 3d
        (0.0383884, 0.219094),
        (0.114681,  0.625290),
        (0.448441,  0.273727),
    ],
    (4, 0): [   # 4s — STO-3G fit
        (0.0297127, 0.703230),
        (0.0859716, 0.401456),
        (0.299646, -0.0999670),
    ],
    (4, 1): [   # 4p
        (0.0297127, 0.385648),
        (0.0859716, 0.610837),
        (0.299646,  0.157209),
    ],
    (5, 0): [   # 5s
        (0.0269379, 0.704187),
        (0.0769557, 0.402155),
        (0.244144, -0.0999672),
    ],
    (5, 1): [   # 5p
        (0.0269379, 0.384156),
        (0.0769557, 0.611395),
        (0.244144,  0.157387),
    ],
}


@dataclass(frozen=True)
class STO3GShell:
    """A 3-primitive STO-3G expansion at a unit Slater exponent."""
    n: int
    l: int
    alphas: tuple[float, float, float]      # primitive Gaussian exponents (ζ=1)
    coeffs: tuple[float, float, float]      # contraction coefficients


def get_sto3g(n: int, l: int) -> STO3GShell:
    """Look up the STO-3G expansion for principal quantum n and angular l."""
    key = (n, l)
    if key not in _STO_3G_TABLES:
        raise KeyError(f"No STO-3G expansion for (n={n}, l={l})")
    rows = _STO_3G_TABLES[key]
    return STO3GShell(
        n=n, l=l,
        alphas=tuple(r[0] for r in rows),
        coeffs=tuple(r[1] for r in rows),
    )


def get_sto_ng(n: int, l: int, n_gauss: int) -> STO3GShell:
    """Look up the STO-NG expansion for ``(n, l)`` with ``n_gauss``
    primitives. Currently supports ``n_gauss in {2, 3}``.
    """
    if n_gauss == 2:
        table = _STO_2G_TABLES
    elif n_gauss == 3:
        table = _STO_3G_TABLES
    else:
        # STO-4G / -6G could be added; for Phase A0 (CHNO + heavy aux 2s)
        # we only need STO-2G (H aux) and STO-3G (everything else).
        raise NotImplementedError(f"STO-{n_gauss}G not yet vendored")
    key = (n, l)
    if key not in table:
        raise KeyError(f"No STO-{n_gauss}G expansion for (n={n}, l={l})")
    rows = table[key]
    return STO3GShell(
        n=n, l=l,
        alphas=tuple(r[0] for r in rows),
        coeffs=tuple(r[1] for r in rows),
    )


def gfn0_n_gauss(Z: int, l: int, n_principal: int, is_valence: bool) -> int:
    """xtb's setGFN0NumberOfPrimitives rule (gfn0.f90:847-877).

    H, He: valence s = 3, aux s = 2; p = 3.
    Z >= 3: s = 4 (n<=5) / 6 (n>5); p = 3 (n<=5) / 6 (n>5); d = 4.

    For Phase A0 we cap at STO-3G (CHNO valence is N=3 always; only
    H/He aux 2s drops to N=2). Heavier elements would need STO-4G+
    which is not yet vendored — they currently fall back to STO-3G.
    """
    if Z <= 2:
        if l == 0:
            return 3 if is_valence else 2
        return 3
    # Z >= 3: per Stewart 1969 + xtb the canonical N is >= 3 for valence
    # s/p. For Phase A0 we use STO-3G; heavier-element parity awaits
    # STO-4G/-6G vendoring.
    return 3


# ---------------------------------------------------------------------------
# Cartesian-Gaussian normalization constants for primitive Gaussians.
# norm(α, l) = (2α/π)^(3/4) · (4α)^(l/2) · 1/sqrt((2l-1)!!)
# For l = 0:  norm = (2α/π)^(3/4)
# For l = 1:  norm = (2α/π)^(3/4) · sqrt(4α)
# (Cartesian p_x, p_y, p_z share this prefactor; the angular factor is
# the bare Cartesian coordinate r_α.)
# ---------------------------------------------------------------------------


def primitive_norm_s(alpha: float | np.ndarray) -> np.ndarray:
    """Normalization constant for a Cartesian s-type Gaussian primitive."""
    return (2.0 * np.asarray(alpha) / np.pi) ** 0.75


def primitive_norm_p(alpha: float | np.ndarray) -> np.ndarray:
    """Normalization constant for a Cartesian p-type Gaussian primitive."""
    a = np.asarray(alpha)
    return (2.0 * a / np.pi) ** 0.75 * np.sqrt(4.0 * a)
