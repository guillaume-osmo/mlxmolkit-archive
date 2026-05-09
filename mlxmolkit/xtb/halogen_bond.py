# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN1's halogen-bond correction (verbatim port of xbpot in
``xtb/src/xtb/halogen.f90``).

GFN1-only term — adds an attractive correction for X···Y interactions
where X ∈ {Cl=17, Br=35, I=53, At=85} and Y ∈ {N=7, O=8, P=15, S=16}.
The neighbour B is the atom closest to X by raw distance² (xtb's
"closest-neighbor" rule, scf_module.F90:387-391); the term has the
form

    exb = aterm · cc · (t14 − dampingPar · t13^lj2) / (1 + t14)

with
    aterm = (½ − ¼ · cos(B-X-A))^α      (α = 6)
    t13   = r0_AX / r_AX
    t14   = t13^a                         (a = ljexp = 12)
    lj2   = a/2 = 6
    r0_AX = radScale · (atomicRad[X] + atomicRad[A])

where A = halogen-bond acceptor (Y), X = halogen, B = halogen's covalent
neighbor. cc = halogenBond[X] (only nonzero for Br=0.0381, I=0.0322 and
At=0.0220 in GFN1 — Cl is included structurally but its parameter is
zero, see scf_module.F90 comment at line 932).

GFN1 globals (gfn1.f90:533-557 + scf_module.F90:197):
    radScale = 1.3
    dampingPar = 0.44
    ljexp (a) = 12
    halogenBond × 0.1 (already applied in the params table)
"""

from __future__ import annotations

import numpy as np

_ANG_TO_BOHR = 1.8897259886

# atomicRad in Bohr (param/atomicrad.f90:42-57). xtb uses these for
# both shell-poly and halogen-bond.
_ATOMIC_RAD_BOHR = (
    np.array([
        0.32, 0.37, 1.30, 0.99, 0.84, 0.75, 0.71, 0.64,
        0.60, 0.62, 1.60, 1.40, 1.24, 1.14, 1.09, 1.04,
        1.00, 1.01, 2.00, 1.74, 1.59, 1.48, 1.44, 1.30,
        1.29, 1.24, 1.18, 1.17, 1.22, 1.20, 1.23, 1.20,
        1.20, 1.18, 1.17, 1.16, 2.15, 1.90, 1.76, 1.64,
        1.56, 1.46, 1.38, 1.36, 1.34, 1.30, 1.36, 1.40,
        1.42, 1.40, 1.40, 1.37, 1.36, 1.36, 2.38, 2.06,
        1.94, 1.84, 1.90, 1.88, 1.86, 1.85, 1.83, 1.82,
        1.81, 1.80, 1.79, 1.77, 1.77, 1.78, 1.74, 1.64,
        1.58, 1.50, 1.41, 1.36, 1.32, 1.30, 1.30, 1.32,
        1.44, 1.45, 1.50, 1.42, 1.48, 1.46,
    ], dtype=np.float64)
    * _ANG_TO_BOHR
)
# 0-indexed: _ATOMIC_RAD_BOHR[Z-1].

# halogenBond per element (gfn1.f90:539-557, already × 0.1).
_HALOGENBOND = np.zeros(86, dtype=np.float64)
_HALOGENBOND[35 - 1] = 0.0381742
_HALOGENBOND[53 - 1] = 0.0321944
_HALOGENBOND[85 - 1] = 0.0220000

_HALOGEN_X = (17, 35, 53, 85)         # Cl, Br, I, At
_HALOGEN_Y = (7, 8, 15, 16)           # N, O, P, S

_RAD_SCALE = 1.3
_DAMPING = 0.44
_LJ_EXP = 12.0
_LJ2 = 0.5 * _LJ_EXP
_ALPHA = 6.0


def _xbond_pair(zi: int, zj: int) -> bool:
    """Mirror of xtb's ``xbond(ati, atj)`` (scf_module.F90:932-953).

    True iff one is a halogen X ∈ {Cl, Br, I, At} and the other is in
    {N, O, P, S}.
    """
    lx1 = zi in _HALOGEN_X
    lx2 = zj in _HALOGEN_X
    ly1 = zi in _HALOGEN_Y
    ly2 = zj in _HALOGEN_Y
    return (lx1 and ly2) or (lx2 and ly1)


def _build_xb_list(atoms: list[int], coords_bohr: np.ndarray) -> list[tuple[int, int, int]]:
    """Construct the (X, A, B) halogen-bond list per scf_module.F90:357-407.

    For every (i, j) pair satisfying ``xbond(i, j)`` and ``r²_ij < 400``
    (Bohr²), find the closest neighbour B of the halogen atom by raw r²
    over all atoms ≠ X. Returns a list of 3-tuples
    ``(X_index, A_index, B_index)``.
    """
    n = len(atoms)
    out: list[tuple[int, int, int]] = []
    sqrab = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            d = coords_bohr[i] - coords_bohr[j]
            r2 = float(np.dot(d, d))
            sqrab[i, j] = sqrab[j, i] = r2
    for i in range(n):
        ati = atoms[i]
        for j in range(i):
            atj = atoms[j]
            if not _xbond_pair(ati, atj):
                continue
            if sqrab[i, j] >= 400.0:
                continue
            # Halogen is the X-set member; the other is acceptor A.
            if ati in _HALOGEN_X:
                X, A = i, j
            elif atj in _HALOGEN_X:
                X, A = j, i
            else:
                continue
            # Closest neighbour B of X (raw distance², over all atoms ≠ X).
            best_m = -1
            best_r2 = 1e42
            for m in range(n):
                if m == X:
                    continue
                if sqrab[m, X] < best_r2:
                    best_r2 = sqrab[m, X]
                    best_m = m
            if best_m < 0:
                continue
            out.append((X, A, best_m))
    return out


def halogen_bond_energy(atoms: list[int], coords_ang: np.ndarray) -> float:
    """GFN1 halogen-bond correction in Hartree.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom coordinates.

    Returns:
        Total halogen-bond correction energy in Hartree (positive = repulsive,
        negative = attractive — both branches are possible depending on
        geometry, but for typical X···Y geometries with B-X-A near 180°
        the correction is attractive).
    """
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    triples = _build_xb_list(atoms, coords_bohr)
    if not triples:
        return 0.0

    e = 0.0
    for X, A, B in triples:
        zX = atoms[X]
        zA = atoms[A]
        cc = float(_HALOGENBOND[zX - 1])
        if cc == 0.0:
            continue
        r0ax = _RAD_SCALE * (_ATOMIC_RAD_BOHR[zX - 1] + _ATOMIC_RAD_BOHR[zA - 1])
        dxa = coords_bohr[A] - coords_bohr[X]
        dxb = coords_bohr[B] - coords_bohr[X]
        dba = coords_bohr[A] - coords_bohr[B]
        d2ax = float(np.dot(dxa, dxa))
        d2bx = float(np.dot(dxb, dxb))
        d2ab = float(np.dot(dba, dba))
        rax = float(np.sqrt(d2ax))
        if rax < 1e-12:
            continue
        # cos(angle B-X-A) via law of cosines (xtb form is dimensionless).
        XY = float(np.sqrt(d2bx * d2ax))
        term = (d2bx + d2ax - d2ab) / XY
        aterm = (0.5 - 0.25 * term) ** _ALPHA
        t13 = r0ax / rax
        t14 = t13 ** _LJ_EXP
        e += aterm * cc * (t14 - _DAMPING * t13 ** _LJ2) / (1.0 + t14)
    return float(e)
