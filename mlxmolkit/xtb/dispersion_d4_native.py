# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2-xTB D4 dispersion — pure-numpy port (FOUNDATIONS, partial).

**STATUS (2026-05-09)**: scaffold complete, energies are systematically
~5× too weak vs simple-dftd4 (verified on H2O, CH4, ethanol — see
``benchmarks/d4_native_vs_wrapper.py``). The cause is the
*secondary-reference subtraction* xtb applies to the raw polarizability
tables (dftd4.F90:181):

    dispm%alpha[w, j, Z] = max(ascale[j, Z] · (alphaiw[w, j, Z]
                              − hcount[j, Z] · sec_al[w, j, Z]), 0)

with ``sec_al = sscale[is] · secaiw[w, is] · zeta(g_a, gam[is]·g_c,
secq[is]+Zeff, tmp_hq[j, Z]+Zeff)`` and ``is = refsys[j, Z]``.

What's vendored today:
    refn[Z], refcn[Z, ref], refq_gfn2[Z, ref],
    alphaiw[Z, ref, w] (raw), omega_w/omega_weights, r4r2[Z]

What's still needed for full parity with simple-dftd4:
    refsys[Z, ref] : reference system index (1..17)
    hcount[Z, ref] : H count of each reference (for subtraction)
    ascale[Z, ref] : per-reference scaling factor
    secq[is], sscale[is], secaiw[w, is] : secondary references (~17)

All of these live in ``xtb/include/param_ref.fh`` (~6300 lines). The
parser ``tools/parse_d4_refs.py`` already extracts refn/refcn/refq;
extending it for the secondary-reference table is the obvious next
step. Until then, **production code should use the simple-dftd4
backend in :mod:`dispersion_d4`** — this module exists to anchor the
algorithm and the data layout.

Reference for the algorithm (xtb/src/disp/{dftd4,ncoord}.f90,
Caldeweyher 2019, JCP 150:154122):

1. **D4 coordination number** (electronegativity-modified erf-CN):
   ``CN_i = Σ_{j≠i} k4 · exp(-(|EN_i − EN_j| + k5)² / k6) · erf-count(R, rco)``
   with ``k4=4.10451``, ``k5=19.08857``, ``k6=2·11.28174²`` and
   the same Pyykkö covalent radii × 4/3 scaling as D3.

2. **Reference weighting**:
   ``w[k, i] = cngw(wf, CN_i, refcn[k, Z]) · zeta(g_a, gam[Z]·g_c, refq[k, Z]+Zeff, q_atom+Zeff)``
   where ``cngw`` is the Gaussian over CN, ``zeta`` the
   exponential charge-correction, and ``wf=6``, ``g_a=3``, ``g_c=2``
   for GFN2.

3. **Atomic polarizability**:
   ``α_atom(iω_w) = Σ_k w[k] · alphaiw[k, Z, w]``.

4. **C6 via Casimir-Polder**:
   ``C6_ij = (3/π) · Σ_w omega_weights[w] · α_i(iω_w) · α_j(iω_w)``.

5. **BJ damping** (s6, s8, a1, a2 from GFN2 D4 globals):
   ``E = -Σ_pair C6_ij · (s6/(R⁶+R0⁶) + s8 r4r2_ij/(R⁸+R0⁸))``
   with ``R0 = a1·sqrt(3·r4r2_i·r4r2_j) + a2``.

The s9 ATM three-body term is not yet implemented (small contribution
on small molecules; can be added when needed).

Reference data is vendored in ``params/d4_data.npz`` (parsed by
``tools/parse_d4_refs.py`` from dftd4-python's references.json + xtb's
include/param_ref.fh — both are MIT/LGPL-licensed).
"""

from __future__ import annotations

import os
import numpy as np

_DATA_PATH = os.path.join(os.path.dirname(__file__), "params", "d4_data.npz")
_D4 = np.load(_DATA_PATH)
_REFN = _D4["refn"]                  # (118,)
_REFCN = _D4["refcn"]                # (118, 7)
_REFQ_GFN2 = _D4["refq_gfn2"]        # (118, 7)
_ALPHAIW = _D4["alphaiw"]            # (118, 7, 23) — RAW dftd4 polarizabilities
_HCOUNT = _D4["hcount"] if "hcount" in _D4.files else None  # (118, 7)
_ASCALE = _D4["ascale"] if "ascale" in _D4.files else None  # (118, 7)
_REFSYS = _D4["refsys"] if "refsys" in _D4.files else None  # (118, 7) int
_SECQ = _D4["secq"] if "secq" in _D4.files else None        # (17,)
_SSCALE = _D4["sscale"] if "sscale" in _D4.files else None  # (17,)
_SECAIW = _D4["secaiw"] if "secaiw" in _D4.files else None  # (17, 23)
_OMEGA_W = _D4["omega_w"]            # (23,) — frequency grid
_OMEGA_WEIGHTS = _D4["omega_weights"] # (23,) — trapezoidal weights
_R4R2 = _D4["r4r2"]                  # (118,) — sqrt(0.5·r4_over_r2·sqrt(Z))


_ANG_TO_BOHR = 1.8897259886

# D4 globals for GFN2-xTB (read_gfn_param.f90:177-179)
GFN2_D4 = {
    "g_a": 3.0,
    "g_c": 2.0,
    "wf":  6.0,
    "s6":  1.0,
    "s8":  2.7,
    "a1":  0.52,
    "a2":  5.0,
}

# D4 CN constants (ncoord.f90:55-59)
_K4 = 4.10451
_K5 = 19.08857
_K6 = 2.0 * 11.28174 ** 2
_KN = 7.50

# Pauling electronegativity (ncoord.f90:90-110, same table as
# _GFN2_PAULING_EN; we reuse from params_gfn2 for consistency).
from .params_gfn2 import _GFN2_PAULING_EN as _EN

# zeff per element (dftd4_parameters.f90:41-51 — first 36 are nuclear charges)
_ZEFF = np.array([
    0,                                                            # 0 sentinel
    1, 2,                                                          # H He
    3, 4, 5, 6, 7, 8, 9, 10,                                       # Li-Ne
    11, 12, 13, 14, 15, 16, 17, 18,                                # Na-Ar
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36,                                        # K-Kr
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26,                                        # Rb-Xe
    9, 10, 11, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,    # Hf-Rn
], dtype=np.float64)


# Pyykkö covalent radii (Å) for D3-style erf CN counting — reuse the
# table already shipped in mlxmolkit.xtb.cn.
from .cn import _COV_RAD_PYYKKO, _COV_SCALE


def d4_coordination_number(
    atoms: list[int],
    coords_ang: np.ndarray,
) -> np.ndarray:
    """D4 covalent CN with electronegativity scaling (ncoord_d4)."""
    coords = np.asarray(coords_ang, dtype=np.float64)
    n = len(atoms)
    rcov = np.array([_COV_RAD_PYYKKO[int(z)] for z in atoms], dtype=np.float64)
    cn = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rij = coords[i] - coords[j]
            R = float(np.linalg.norm(rij))
            rco = (rcov[i] + rcov[j]) * _COV_SCALE
            den = _K4 * np.exp(-(abs(_EN[atoms[i]] - _EN[atoms[j]]) + _K5) ** 2 / _K6)
            erf_count = 0.5 * (1.0 + _erf(_KN * (rco - R) / rco))
            cn[i] += den * erf_count
    return cn


def _erf(x):
    from math import erf
    if isinstance(x, np.ndarray):
        return np.vectorize(erf)(x)
    return erf(x)


def _zeta(a: float, c: float, qref: float, qmod: float) -> float:
    """xtb's ``zeta(a, c, qref, qmod)`` (dftd4.F90:348-362).

    For ``qmod < 0``: returns ``exp(a)``. Otherwise:
    ``exp(a · (1 − exp(c · (1 − qref/qmod))))``.
    """
    if qmod < 0.0:
        return float(np.exp(a))
    return float(np.exp(a * (1.0 - np.exp(c * (1.0 - qref / qmod)))))


def _cngw(wf: float, cn: float, cnref: float) -> float:
    """xtb's ``cngw(wf, cn, cnref)`` (dftd4.F90:451-465)."""
    val = -wf * (cn - cnref) ** 2
    if val < -200.0:
        return 0.0
    return float(np.exp(val))


def _build_d4_alpha_table(g_a: float, g_c: float, all_hardness: np.ndarray) -> np.ndarray:
    """Build the D4 reference α table.

    **NOTE**: xtb's full ``newD4Model`` applies a secondary-reference
    subtraction (dftd4.F90:181):

        alpha[w, j, Z] = max(ascale[j, Z] ·
                             (alphaiw_raw[Z, j, w] −
                              hcount[j, Z] · sec_al[w, j, Z]), 0)

    with ``sec_al = sscale[is] · secaiw[is, w] · zeta(...)``
    and ``is = refsys[j, Z]``. The ``zeta`` argument ``tmp_hq[j, Z]``
    is xtb's ``solh`` array under refq=gfn2-xtb, which we don't
    vendor yet (only ``refq`` is available). Without solh the
    subtraction overshoots and produces α values that are too small.

    Until ``solh`` is vendored from xtb's param_ref.fh (~120 more
    Fortran data lines), we ship the *raw* alphaiw — this gives D4
    energies that are ~5× too weak vs simple-dftd4 but correct in
    structure.
    """
    return _ALPHAIW.copy()


def _atomic_alpha_iw(
    atoms: np.ndarray,
    cn: np.ndarray,
    q: np.ndarray,
    *,
    g_a: float,
    g_c: float,
    wf: float,
    chemical_hardness: np.ndarray,
) -> np.ndarray:
    """Per-atom polarizability α(iω_w) array of shape (n_atoms, 23).

    Sums over reference systems with CN-Gaussian weights and
    charge-correction zeta function. Matches xtb's ``mdisp`` /
    ``d4`` driver loop.
    """
    n = len(atoms)
    aw = np.zeros((n, 23), dtype=np.float64)
    # Build the subtracted reference α table once (it depends on g_a,
    # g_c via the zeta function).
    alpha_table = _build_d4_alpha_table(g_a, g_c, chemical_hardness)

    for i, Z in enumerate(atoms):
        nref = int(_REFN[Z - 1])
        if nref == 0:
            continue
        # CN-Gaussian weighting normalization (xtb retries with
        # twf = iii*wf for iii ∈ {1, 2, 3} until norm > 0).
        gw = np.zeros(nref, dtype=np.float64)
        for iii in range(1, 4):
            twf = iii * wf
            tmp = np.zeros(nref, dtype=np.float64)
            for k in range(nref):
                tmp[k] = _cngw(twf, float(cn[i]), float(_REFCN[Z - 1, k]))
            norm = float(np.sum(tmp))
            if norm > 1e-80:
                gw[:] = tmp / norm
                break
        # Apply charge-correction zeta on top of the CN weight.
        zeff = float(_ZEFF[Z])
        gam_z = float(chemical_hardness[i])
        for k in range(nref):
            qref = float(_REFQ_GFN2[Z - 1, k])
            zfac = _zeta(g_a, gam_z * g_c, qref + zeff, float(q[i]) + zeff)
            aw[i] += gw[k] * zfac * alpha_table[Z - 1, k, :]
    return aw


def _trapzd(pol: np.ndarray) -> float:
    """Casimir-Polder trapezoidal integral over 23 frequency points.
    ``pol`` shape (..., 23); the last axis is summed.
    """
    return float(np.sum(pol * _OMEGA_WEIGHTS))


def d4_dispersion_native(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    q: np.ndarray | None = None,
    *,
    s6: float | None = None,
    s8: float | None = None,
    a1: float | None = None,
    a2: float | None = None,
    g_a: float | None = None,
    g_c: float | None = None,
    wf: float | None = None,
) -> float:
    """D4 dispersion energy in Hartree (BJ damping, no ATM yet).

    Args:
        atoms: list of atomic numbers.
        coords_ang: (n_atoms, 3) Angstrom.
        q: optional atomic charges. If None, uses zero charges (the
            EEQ-based "minimal" D4). For GFN2 parity, pass the SCF-
            converged Mulliken charges.
        s6, s8, a1, a2, g_a, g_c, wf: damping parameters. Defaults are
            the GFN2-xTB values.

    Returns:
        Total D4 two-body BJ energy (Hartree, negative = attractive).
    """
    if s6 is None: s6 = GFN2_D4["s6"]
    if s8 is None: s8 = GFN2_D4["s8"]
    if a1 is None: a1 = GFN2_D4["a1"]
    if a2 is None: a2 = GFN2_D4["a2"]
    if g_a is None: g_a = GFN2_D4["g_a"]
    if g_c is None: g_c = GFN2_D4["g_c"]
    if wf is None:  wf  = GFN2_D4["wf"]

    atoms_arr = np.asarray(atoms, dtype=np.int64)
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n = len(atoms_arr)
    if n < 2:
        return 0.0

    if q is None:
        q = np.zeros(n, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # D4 CN
    cn = d4_coordination_number(atoms_arr.tolist(), coords_ang)

    # Per-atom hardness for the zeta function. Also build a 0-indexed
    # element-resolved table for the secondary-reference loop.
    from .params_gfn2 import GFN2_PARAMS
    chemical_hardness = np.array(
        [GFN2_PARAMS[int(z)].chemical_hardness for z in atoms_arr],
        dtype=np.float64,
    )
    all_hardness = np.zeros(87, dtype=np.float64)
    for zz in range(1, 87):
        all_hardness[zz] = GFN2_PARAMS[zz].chemical_hardness

    # Atomic polarizability α(iω_w) for each atom
    aw = _atomic_alpha_iw(
        atoms_arr, cn, q,
        g_a=g_a, g_c=g_c, wf=wf,
        chemical_hardness=all_hardness,
    )

    # C6_ij via Casimir-Polder integral
    THOPI = 3.0 / np.pi
    c6 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            c6[i, j] = THOPI * _trapzd(aw[i] * aw[j])
            c6[j, i] = c6[i, j]

    # r4r2 per atom
    r4r2_per = _R4R2[atoms_arr - 1]

    # BJ damping pair sum
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r4r2ij = 3.0 * r4r2_per[i] * r4r2_per[j]
            r0 = a1 * np.sqrt(r4r2ij) + a2
            R = np.linalg.norm(coords[i] - coords[j])
            R2 = R * R
            t6 = 1.0 / (R2 ** 3 + r0 ** 6)
            t8 = 1.0 / (R2 ** 4 + r0 ** 8)
            disp = s6 * t6 + s8 * r4r2ij * t8
            e += -float(c6[i, j]) * disp
    return e


def d4_dispersion_gfn2_native(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    q: np.ndarray | None = None,
) -> float:
    """Drop-in replacement for :func:`mlxmolkit.xtb.dispersion_d4.d4_dispersion_gfn2`
    using the pure-numpy backend instead of simple-dftd4.
    """
    return d4_dispersion_native(atoms, coords_ang, q=q)
