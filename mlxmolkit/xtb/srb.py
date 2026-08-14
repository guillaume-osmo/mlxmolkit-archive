# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN0-xTB short-range bond (SRB) correction.

Verbatim port of xtb/src/peeq_module.f90:706-789 (``dsrb_grad``) and
xtb/src/approxrab.f90:215-264 (``approx_rab``). The SRB term is a
Gaussian-localized correction over hetero element pairs in {B, C, N,
O, F} (atomic numbers 5..9). Hydrogen and elements outside that range
are excluded; same-element pairs are excluded. The functional form is

    E_SRB = Σ_{(i,j) ∈ srblist} prefactor · exp(−pre_ij · (R_ij − R_ij^0)²)

with

    pre_ij    = steepness · (1 + enScale · (en_i − en_j)²)
    R_ij^0    = (r_i + r_j) · (1 − k1·|Δχ| − k2·Δχ²)
    r_i       = r0(at_i) + cnfak(at_i)·CN_i + shift
    k1, k2    = 0.005 · (p(row_i, 1..2) + p(row_j, 1..2))

The four ``srb`` globals come from the GFN0 ``$globpar`` block:
    ``shift=srbshift, prefactor=srbpre, steepness=srbexp, enScale=srbken``.

The ``en``, ``r0``, ``cnfak`` and ``p`` tables are SRB-specific and
*differ* from the GFN0 electronegativity / repulsion radii — vendored
verbatim from approxrab.f90.

Phase A0: numpy single-molecule. Returns Hartree.
"""

from __future__ import annotations

import numpy as np

from .params_gfn0 import GFN0_GLOBALS

_ANG_TO_BOHR = 1.8897259886


# Element-keyed Pauling-style electronegativity used INSIDE SRB only
# (different from the gfn0 electronegativity table). approxrab.f90:30-55.
_SRB_EN = (
    0.0,
    2.30085633, 2.78445145, 1.52956084, 1.51714704, 2.20568300,
    2.49640820, 2.81007174, 4.51078438, 4.67476223, 3.29383610,
    2.84505365, 2.20047950, 2.31739628, 2.03636974, 1.97558064,
    2.13446570, 2.91638164, 1.54098156, 2.91656301, 2.26312147,
    2.25621439, 1.32628677, 2.27050569, 1.86790977, 2.44759456,
    2.49480042, 2.91545568, 3.25897750, 2.68723778, 1.86132251,
    2.01200832, 1.97030722, 1.95495427, 2.68920990, 2.84503857,
    2.61591858, 2.64188286, 2.28442252, 1.33011187, 1.19809388,
    1.89181390, 2.40186898, 1.89282464, 3.09963488, 2.50677823,
    2.61196704, 2.09943450, 2.66930105, 1.78349472, 2.09634533,
    2.00028974, 1.99869908, 2.59072029, 2.54497829, 2.52387890,
    2.30204667, 1.60119300, 2.00000000, 2.00000000, 2.00000000,
    2.00000000, 2.00000000, 2.00000000, 2.00000000, 2.00000000,
    2.00000000, 2.00000000, 2.00000000, 2.00000000, 2.00000000,
    2.00000000, 2.30089349, 1.75039077, 1.51785130, 2.62972945,
    2.75372921, 2.62540906, 2.55860939, 3.32492356, 2.65140898,
    1.52014458, 2.54984804, 1.72021963, 2.69303422, 1.81031095,
    2.34224386,
    2.52387890,
    2.30204667, 1.60119300, 2.00000000, 2.00000000, 2.00000000,
    2.00000000, 2.00000000, 2.00000000, 2.00000000, 2.00000000,
    2.00000000, 2.00000000, 2.00000000, 2.00000000, 2.00000000,
    2.00000000,
)
assert len(_SRB_EN) == 104  # 0 sentinel + 103 elements

_SRB_R0 = (
    0.0,
    0.55682207, 0.80966997, 2.49092101, 1.91705642, 1.35974851,
    0.98310699, 0.98423007, 0.76716063, 1.06139799, 1.17736822,
    2.85570926, 2.56149012, 2.31673425, 2.03181740, 1.82568535,
    1.73685958, 1.97498207, 2.00136196, 3.58772537, 2.68096221,
    2.23355957, 2.33135502, 2.15870365, 2.10522128, 2.16376162,
    2.10804037, 1.96460045, 2.00476257, 2.22628712, 2.43846700,
    2.39408483, 2.24245792, 2.05751204, 2.15427677, 2.27191920,
    2.19722638, 3.80910350, 3.26020971, 2.99716916, 2.71707818,
    2.34950167, 2.11644818, 2.47180659, 2.32198800, 2.32809515,
    2.15244869, 2.55958313, 2.59141300, 2.62030465, 2.39935278,
    2.56912355, 2.54374096, 2.56914830, 2.53680807, 4.24537037,
    3.66542289, 3.22480000, 3.21280000, 3.10550000, 3.10200000,
    3.10840000, 3.14030000, 3.06390000, 3.10730000, 3.10000000,
    3.11910000, 3.10760000, 3.13740000, 3.09740000, 2.92860000,
    3.05880000, 2.34880037, 2.37597108, 2.49067697, 2.14100577,
    2.33473532, 2.19498900, 2.12678348, 2.34895048, 2.33422774,
    2.86560827, 2.62488837, 2.88376127, 2.75174124, 2.83054552,
    2.63264944,
    4.24537037,
    3.66542289, 4.20000000, 4.20000000, 4.20000000, 4.20000000,
    4.20000000, 4.20000000, 4.20000000, 4.20000000, 4.20000000,
    4.20000000, 4.20000000, 4.20000000, 4.20000000, 4.20000000,
    4.20000000,
)
assert len(_SRB_R0) == 104

_SRB_CNFAK = (
    0.0,
    0.17957827, 0.25584045, -0.02485871, 0.00374217, 0.05646607,
    0.10514203, 0.09753494, 0.30470380, 0.23261783, 0.36752208,
    0.00131819, -0.00368122, -0.01364510, 0.04265789, 0.07583916,
    0.08973207, -0.00589677, 0.13689929, -0.01861307, 0.11061699,
    0.10201137, 0.05426229, 0.06014681, 0.05667719, 0.02992924,
    0.03764312, 0.06140790, 0.08563465, 0.03707679, 0.03053526,
    -0.00843454, 0.01887497, 0.06876354, 0.01370795, -0.01129196,
    0.07226529, 0.01005367, 0.01541506, 0.05301365, 0.07066571,
    0.07637611, 0.07873977, 0.02997732, 0.04745400, 0.04582912,
    0.10557321, 0.02167468, 0.05463616, 0.05370913, 0.05985441,
    0.02793994, 0.02922983, 0.02220438, 0.03340460, -0.04110969,
    -0.01987240, 0.07260201, 0.07700000, 0.07700000, 0.07700000,
    0.07700000, 0.07700000, 0.07700000, 0.07700000, 0.07700000,
    0.07700000, 0.07700000, 0.07700000, 0.07700000, 0.07700000,
    0.07700000, 0.08379100, 0.07314553, 0.05318438, 0.06799334,
    0.04671159, 0.06758819, 0.09488437, 0.07556405, 0.13384502,
    0.03203572, 0.04235009, 0.03153769, -0.00152488, 0.02714675,
    0.04800662,
    0.04582912,
    0.10557321, 0.02167468, 0.05463616, 0.05370913, 0.05985441,
    0.02793994, 0.02922983, 0.02220438, 0.03340460, -0.04110969,
    -0.01987240, 0.07260201, 0.07700000, 0.07700000, 0.07700000,
    0.07700000,
)
assert len(_SRB_CNFAK) == 104

# Polynomial coefficients in (4, 2): rows = PSE row (1..4), cols = (1, 2).
# approxrab.f90:108-110.
_SRB_P = np.array([
    [29.84522887, -8.87843763],
    [-1.70549806,  2.10878369],
    [ 6.54013762,  0.08009374],
    [ 6.39169003, -0.85808076],
], dtype=np.float64)


def _srb_atom(Z: int) -> bool:
    """True if atom Z is *excluded* from SRB. Matches xtb's ``srbatom``
    helper: SRB acts only on elements with 5 ≤ Z ≤ 9 (B, C, N, O, F)."""
    return Z < 5 or Z > 9


def _pse_row(Z: int) -> int:
    """1-based PSE row used to index the SRB polynomial ``p``. Maps
    1..2→1, 3..10→2, 11..18→3, 19+→4. Matches approxrab.f90:266-279."""
    if 1 <= Z <= 2:
        return 1
    if 3 <= Z <= 10:
        return 2
    if 11 <= Z <= 18:
        return 3
    return 4


def compute_srb(
    atoms: list[int],
    coords_ang: np.ndarray,
    cn: np.ndarray,
) -> float:
    """Single-molecule SRB short-range bond correction (Hartree).

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom positions.
        cn: ``(n_atoms,)`` GFN0 erf-coordination numbers.

    Returns:
        Scalar ``E_SRB`` in Hartree. Returns 0.0 when there are no
        hetero-pair (B,C,N,O,F)×(B,C,N,O,F) bonds within the cutoff.
    """
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    cn = np.asarray(cn, dtype=np.float64)
    n_atoms = len(atoms)

    g = GFN0_GLOBALS
    shift = g.srbshift
    prefactor = g.srbpre
    steepness = g.srbexp
    en_scale = g.srbken
    cutoff_sq = 200.0

    # Build srblist: hetero pairs in 5..9, within cutoff
    pairs = []
    for i in range(n_atoms - 1):
        if _srb_atom(int(atoms[i])):
            continue
        for j in range(i + 1, n_atoms):
            if _srb_atom(int(atoms[j])):
                continue
            if int(atoms[i]) == int(atoms[j]):
                continue
            r2 = float(np.sum((coords[i] - coords[j]) ** 2))
            if r2 < cutoff_sq:
                pairs.append((i, j, r2))
    if not pairs:
        return 0.0

    e_srb = 0.0
    for i, j, r2 in pairs:
        ati = int(atoms[i])
        atj = int(atoms[j])
        ir = _pse_row(ati)
        jr = _pse_row(atj)
        ra = _SRB_R0[ati] + _SRB_CNFAK[ati] * cn[i] + shift
        rb = _SRB_R0[atj] + _SRB_CNFAK[atj] * cn[j] + shift
        # k1, k2 with row indexing 1..4 → 0..3 for numpy
        k1 = 0.005 * (_SRB_P[ir - 1, 0] + _SRB_P[jr - 1, 0])
        k2 = 0.005 * (_SRB_P[ir - 1, 1] + _SRB_P[jr - 1, 1])
        # In approx_rab the EN difference is *unsigned* (`abs`), so the
        # linear k1·den term takes the magnitude of Δχ, while the
        # quadratic k2·den² is Δχ² regardless.
        den_abs = abs(_SRB_EN[ati] - _SRB_EN[atj])
        ff = 1.0 - k1 * den_abs - k2 * den_abs * den_abs
        rab0 = (ra + rb) * ff

        # Energy contribution
        rab = float(np.sqrt(r2))
        dr = rab - rab0
        # Note: for the pre-exponent the EN difference is signed-squared
        # (just (en_i − en_j)² — same as |Δχ|²), so we can reuse den_abs².
        pre = steepness * (1.0 + en_scale * den_abs * den_abs)
        e_srb += prefactor * float(np.exp(-pre * dr * dr))

    return float(e_srb)
