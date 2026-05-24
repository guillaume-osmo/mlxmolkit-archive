"""Batched sp-only two-center integral assembly (TETCI step 2).

Port of PYSEQM ``w_withquaternion`` (BSD-3-Clause,
github.com/lanl/PYSEQM, seqm/seqm_functions/two_elec_two_center_int.py
lines 1384-1574).

Given a batch of atom pairs, their bond vectors, and the local-frame
two-electron integrals, returns:
    - ``w``:    (n_pairs, 100)        rotated two-electron integrals
                                       (10×10 packed orbital pairs on
                                       each of A and B in molecular frame)
    - ``wXH``:  (n_pairs_XH, 10)      same for X-H pairs (B has only s)
    - ``e1b``:  (n_pairs, 4, 4)       -Z_B (μν_A | s_B s_B), electrons on
                                       A attracted by B's nucleus
    - ``e2a``:  (n_pairs, 4, 4)       counterpart for B's electrons

This is the sp-only path (orbitals s, p_x, p_y, p_z = 4 per heavy
atom). The d-orbital extension is in :func:`rotate` (TETCI step 3,
1022 LOC, still pending).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tetci_quaternion import rotate_with_quaternion


def w_withquaternion(
    ni: NDArray[np.int64],
    nj: NDArray[np.int64],
    xij: NDArray[np.float64],
    tore: NDArray[np.float64],
    riXH: NDArray[np.float64],
    ri: NDArray[np.float64],
    wHH: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],  # e1b: (n_pairs, 4, 4)
    NDArray[np.float64],  # e2a: (n_pairs, 4, 4)
    NDArray[np.float64],  # wXH: (n_pairs_XH, 10)
    NDArray[np.float64],  # w:   (n_pairs, 100)
]:
    """Assemble batched two-center two-electron integrals in molecular frame.

    Parameters
    ----------
    ni, nj : (n_pairs,) int
        Atomic numbers of the two atoms in each pair (A, B).
    xij : (n_pairs, 3) float
        Bond vector from A to B (cartesian, Angstrom or Bohr — must
        match ``ri``; standard NDDO local frame uses Bohr).
    tore : (n_elem+1,) float
        Valence electron count indexed by atomic number, used to scale
        electron-nuclear attraction.
    riXH : (n_pairs_XH, 4) float
        Local-frame integrals for X-H pairs (B has only s):
        ``[ss|ss], [ps|ss], [pp_sigma|ss], [pp_pi|ss]``.
    ri : (n_pairs_XX, 22) float
        Local-frame integrals for X-X pairs (full 22 NDDO integrals).
    wHH : (n_pairs_HH,) float
        ``(ss|ss)`` integral for H-H pairs.

    Returns
    -------
    e1b, e2a, wXH, w
        - ``w`` shape is ``(n_XX, 100)`` (XX-only), matching PYSEQM.
        - ``wXH`` shape is ``(n_XH, 10)``.
        - ``e1b``, ``e2a`` are ``(n_pairs, 4, 4)`` and indexed by the
          pair masks ``HH``, ``XH``, ``XX``.
    """
    n_pairs = xij.shape[0]
    dtype = xij.dtype

    # Classification masks
    HH = (ni == 1) & (nj == 1)
    XH = (ni > 1) & (nj == 1)
    XX = (ni > 1) & (nj > 1)
    n_HH = int(HH.sum())
    n_XH = int(XH.sum())
    n_XX = int(XX.sum())

    # PYSEQM convention: v = -xij (rotate B onto A's x-axis)
    v = -xij
    rot = rotate_with_quaternion(v, calculate_gradient=False)
    rotXH = rot[XH]
    rotXX = rot[XX]

    # w matches PYSEQM shape: (n_XX, 100) — XX-only
    w = np.zeros((n_XX, 100), dtype=dtype)
    wXH = np.zeros((n_XH, 10), dtype=dtype)

    # Row slices of rotation matrix (B, 3)
    r0, r1, r2 = rotXX[:, 0], rotXX[:, 1], rotXX[:, 2]
    rx0, rx1, rx2 = rotXH[:, 0], rotXH[:, 1], rotXH[:, 2]

    # PYSEQM calls w_withquaternion with ri already restricted to XX pairs.
    # We follow the same convention: ri shape is (n_XX, 22).
    ri_xx = ri  # (n_XX, 22), caller's responsibility to pre-slice

    # Combos: 100 (kk, ll, mm, nn) tuples, in fixed order
    combos = [
        (kk, ll, mm, nn)
        for kk in range(4)
        for ll in range(kk + 1)
        for mm in range(4)
        for nn in range(mm + 1)
    ]

    # Accumulator for XX-pair w tensor
    w_xx = w  # alias (we write into the returned (n_XX, 100) directly)

    idx = 0
    idxXH = 0
    for kk, ll, mm, nn in combos:
        k = kk - 1
        l = ll - 1
        m = mm - 1
        n = nn - 1

        if kk == 0:
            # ss | ··
            if mm == 0:
                # (ss|ss)
                w_xx[:, idx] = ri_xx[:, 0]
                wXH[:, idxXH] = riXH[:, 0]
                idxXH += 1
            elif nn == 0:
                # (ss|ps)
                w_xx[:, idx] = ri_xx[:, 4] * r0[:, m]
            else:
                # (ss|pp)
                w_xx[:, idx] = (
                    ri_xx[:, 10] * (r0[:, m] * r0[:, n])
                    + ri_xx[:, 11] * (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n])
                )

        elif ll == 0:
            # ps | ··
            if mm == 0:
                # (ps|ss)
                w_xx[:, idx] = ri_xx[:, 1] * r0[:, k]
                wXH[:, idxXH] = riXH[:, 1] * rx0[:, k]
                idxXH += 1
            elif nn == 0:
                # (ps|ps)
                w_xx[:, idx] = (
                    ri_xx[:, 5] * (r0[:, k] * r0[:, m])
                    + ri_xx[:, 6] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m])
                )
            else:
                # (ps|pp)
                t0 = r0[:, k] * r0[:, m] * r0[:, n]
                t1 = (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * r0[:, k]
                mix = r1[:, k] * (r1[:, n] * r0[:, m] + r1[:, m] * r0[:, n]) + r2[
                    :, k
                ] * (r2[:, m] * r0[:, n] + r2[:, n] * r0[:, m])
                w_xx[:, idx] = ri_xx[:, 12] * t0 + ri_xx[:, 13] * t1 + ri_xx[:, 14] * mix

        else:
            # pp | ··
            if mm == 0:
                # (pp|ss)
                t0 = r0[:, k] * r0[:, l]
                t1 = r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]
                w_xx[:, idx] = ri_xx[:, 2] * t0 + ri_xx[:, 3] * t1

                x0 = rx0[:, k] * rx0[:, l]
                x1 = rx1[:, k] * rx1[:, l] + rx2[:, k] * rx2[:, l]
                wXH[:, idxXH] = riXH[:, 2] * x0 + riXH[:, 3] * x1
                idxXH += 1
            elif nn == 0:
                # (pp|ps)
                t0 = r0[:, k] * r0[:, l] * r0[:, m]
                t1 = (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m]
                t2 = r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m]
                w_xx[:, idx] = (
                    ri_xx[:, 7] * t0
                    + ri_xx[:, 8] * t1
                    + ri_xx[:, 9]
                    * (
                        r0[:, k] * t2
                        + r0[:, l] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m])
                    )
                )
            else:
                # (pp|pp) — 7 terms
                t0 = r0[:, k] * r0[:, l] * r0[:, m] * r0[:, n]
                w_xx[:, idx] = ri_xx[:, 15] * t0

                t1 = (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m] * r0[:, n]
                w_xx[:, idx] += ri_xx[:, 16] * t1

                t2 = (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * (
                    r0[:, k] * r0[:, l]
                )
                w_xx[:, idx] += ri_xx[:, 17] * t2

                quad = (
                    r1[:, k] * r1[:, l] * r1[:, m] * r1[:, n]
                    + r2[:, k] * r2[:, l] * r2[:, m] * r2[:, n]
                )
                w_xx[:, idx] += ri_xx[:, 18] * quad

                mix1 = r0[:, m] * (r1[:, l] * r1[:, n] + r2[:, l] * r2[:, n])
                mix2 = r0[:, n] * (r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m])
                val5 = r0[:, k] * (mix1 + mix2) + r0[:, l] * (
                    r0[:, m] * (r1[:, k] * r1[:, n] + r2[:, k] * r2[:, n])
                    + r0[:, n] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m])
                )
                w_xx[:, idx] += ri_xx[:, 19] * val5

                mix3 = (
                    r1[:, k] * r1[:, l] * r2[:, m] * r2[:, n]
                    + r2[:, k] * r2[:, l] * r1[:, m] * r1[:, n]
                )
                w_xx[:, idx] += ri_xx[:, 20] * mix3

                cross = (r1[:, k] * r2[:, l] + r2[:, k] * r1[:, l]) * (
                    r1[:, m] * r2[:, n] + r2[:, m] * r1[:, n]
                )
                w_xx[:, idx] += ri_xx[:, 21] * cross

        idx += 1

    # w (= w_xx) already populated for XX pairs.

    # ── Build core-electron attraction matrices e1b, e2a ──────────────
    e1b = np.zeros((n_pairs, 4, 4), dtype=dtype)
    e2a = np.zeros((n_pairs, 4, 4), dtype=dtype)
    # (w is XX-only; reshape only when n_XX > 0 — handled below.)

    # HH pairs: just (ss|ss)
    if n_HH:
        e1b[HH, 0, 0] = -tore[1] * wHH
        e2a[HH, 0, 0] = -tore[1] * wHH

    # XH pairs: B has only s
    if n_XH:
        nj_XH = nj[XH]
        ni_XH = ni[XH]
        e1b[XH, 0, 0] = -tore[nj_XH] * wXH[:, 0]
        e2a[XH, 0, 0] = -tore[ni_XH] * wXH[:, 0]
        e1b[XH, 0, 1] = -tore[nj_XH] * wXH[:, 1]
        e1b[XH, 1, 1] = -tore[nj_XH] * wXH[:, 2]
        e1b[XH, 0, 2] = -tore[nj_XH] * wXH[:, 3]
        e1b[XH, 1, 2] = -tore[nj_XH] * wXH[:, 4]
        e1b[XH, 2, 2] = -tore[nj_XH] * wXH[:, 5]
        e1b[XH, 0, 3] = -tore[nj_XH] * wXH[:, 6]
        e1b[XH, 1, 3] = -tore[nj_XH] * wXH[:, 7]
        e1b[XH, 2, 3] = -tore[nj_XH] * wXH[:, 8]
        e1b[XH, 3, 3] = -tore[nj_XH] * wXH[:, 9]

    # XX pairs: B has full sp; e1b from column 0, e2a from row 0 of w
    if n_XX:
        ni_XX = ni[XX]
        nj_XX = nj[XX]
        w_xx_view = w.reshape(n_XX, 10, 10)
        e1b[XX, 0, 0] = -tore[nj_XX] * w_xx_view[:, 0, 0]
        e2a[XX, 0, 0] = -tore[ni_XX] * w_xx_view[:, 0, 0]
        e1b[XX, 0, 1] = -tore[nj_XX] * w_xx_view[:, 1, 0]
        e1b[XX, 1, 1] = -tore[nj_XX] * w_xx_view[:, 2, 0]
        e1b[XX, 0, 2] = -tore[nj_XX] * w_xx_view[:, 3, 0]
        e1b[XX, 1, 2] = -tore[nj_XX] * w_xx_view[:, 4, 0]
        e1b[XX, 2, 2] = -tore[nj_XX] * w_xx_view[:, 5, 0]
        e1b[XX, 0, 3] = -tore[nj_XX] * w_xx_view[:, 6, 0]
        e1b[XX, 1, 3] = -tore[nj_XX] * w_xx_view[:, 7, 0]
        e1b[XX, 2, 3] = -tore[nj_XX] * w_xx_view[:, 8, 0]
        e1b[XX, 3, 3] = -tore[nj_XX] * w_xx_view[:, 9, 0]
        e2a[XX, 0, 1] = -tore[ni_XX] * w_xx_view[:, 0, 1]
        e2a[XX, 1, 1] = -tore[ni_XX] * w_xx_view[:, 0, 2]
        e2a[XX, 0, 2] = -tore[ni_XX] * w_xx_view[:, 0, 3]
        e2a[XX, 1, 2] = -tore[ni_XX] * w_xx_view[:, 0, 4]
        e2a[XX, 2, 2] = -tore[ni_XX] * w_xx_view[:, 0, 5]
        e2a[XX, 0, 3] = -tore[ni_XX] * w_xx_view[:, 0, 6]
        e2a[XX, 1, 3] = -tore[ni_XX] * w_xx_view[:, 0, 7]
        e2a[XX, 2, 3] = -tore[ni_XX] * w_xx_view[:, 0, 8]
        e2a[XX, 3, 3] = -tore[ni_XX] * w_xx_view[:, 0, 9]

    return e1b, e2a, wXH, w
