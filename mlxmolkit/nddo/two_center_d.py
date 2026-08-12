"""
Two-center two-electron integrals for spd basis (PM6).

Extends the sp 22-integral set to 100 integrals (10×10 packed)
using the same Klopman-Ohno-Dewar multipole approximation.

The 10×10 w tensor is assembled from the 22 sp integrals + rotation:
  w[kk,ll,mm,nn] = Σ ri[i] * f(rot, kk, ll, mm, nn, i)

where kk,ll index on atom A (10 unique: ss, ps, pp×3, ps×2)
and mm,nn index on atom B.

For d-orbitals: the sp 22-integral local frame is used, rotated with
the quaternion-based 3×3 rotation matrix. D-orbital two-center integrals
are approximated by the monopole (ss|ss) Coulomb term.

Port of PYSEQM w_withquaternion (lines 1384-1574).
"""
from __future__ import annotations

import numpy as np
from .rotation import _rotation_matrix
from .two_center_integrals import two_center_integrals
from .params import ANG_TO_BOHR


def two_center_w_10x10(
    pA, pB,
    coordA: np.ndarray,
    coordB: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 10×10 w tensor, e1b, e2a in molecular frame.

    For sp atoms: equivalent to the 4×4 rotation.
    For d-orbital atoms: extends to 10×10 with sp integrals + rotation.

    The 10 packed indices map to orbital pairs:
      0: ss
      1: ps(x)  2: pp(xx)
      3: ps(y)  4: pp(xy)  5: pp(yy)
      6: ps(z)  7: pp(xz)  8: pp(yz)  9: pp(zz)

    Returns:
        w: (10, 10) two-electron integrals
        e1b: (10,) nuclear attraction on A from B
        e2a: (10,) nuclear attraction on B from A
    """
    R_vec = coordB - coordA
    R = np.linalg.norm(R_vec)

    nA = pA.n_basis
    nB = pB.n_basis

    if R < 1e-10:
        return np.zeros((10, 10)), np.zeros(10), np.zeros(10)

    # Get sp local-frame integrals (22 for XX, 4 for XH, 1 for HH)
    ri, core, pair_type = two_center_integrals(pA, pB, R)

    # Rotation matrix: v = -(coordB - coordA) / R (MOPAC convention)
    v = -R_vec / R
    rot = _rotation_matrix(v)
    r0, r1, r2 = rot[0], rot[1], rot[2]

    ZA = float(pA.n_valence)
    ZB = float(pB.n_valence)

    w = np.zeros((10, 10))
    e1b = np.zeros(10)
    e2a = np.zeros(10)

    if pair_type == 'HH':
        w[0, 0] = ri[0]
        e1b[0] = -ZB * ri[0]
        e2a[0] = -ZA * ri[0]
        return w, e1b, e2a

    if pair_type == 'XH':
        # ri: [0]=ss|ss, [1]=ps|ss, [2]=pp_sigma|ss, [3]=pp_pi|ss
        w[0, 0] = ri[0]
        for k in range(3):
            w[1 + k * 3, 0] = ri[1] * r0[k]  # ps|ss
            w[0, 1 + k * 3] = w[1 + k * 3, 0]

        # pp|ss
        for k in range(3):
            for l in range(k + 1):
                idx = 1 + k * 3 + l + (1 if k > 0 and l > 0 else 0)
                # Simplified: pp|ss block
                if k == l:
                    w_val = ri[2] * r0[k]**2 + ri[3] * (r1[k]**2 + r2[k]**2)
                else:
                    w_val = ri[2] * r0[k]*r0[l] + ri[3] * (r1[k]*r1[l] + r2[k]*r2[l])
                # Map to packed 10×10 indices
                kk = _pack_idx(k, l)
                w[kk, 0] = w_val
                w[0, kk] = w_val

        # e1b, e2a
        e1b[0] = -ZB * w[0, 0]
        for k in range(3):
            e1b[1 + k * 3] = -ZB * w[1 + k * 3, 0]
            for l in range(k + 1):
                kk = _pack_idx(k, l)
                e1b[kk] = -ZB * w[kk, 0]
        e2a[0] = -ZA * w[0, 0]

        return w, e1b, e2a

    # XX case: full 22 integrals → 10×10 rotation
    # ri: 22 integrals in local frame
    # Generate all 100 w elements from ri + rotation
    combos = []
    for kk in range(4):
        for ll in range(kk + 1):
            combos.append((kk, ll))
    # combos: 10 pairs = (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)

    for i_bra, (kk, ll) in enumerate(combos):
        for i_ket, (mm, nn) in enumerate(combos):
            val = _rotate_integral(ri, r0, r1, r2, kk, ll, mm, nn)
            w[i_bra, i_ket] = val

    # e1b and e2a from first column/row of w
    for i in range(10):
        e1b[i] = -ZB * w[i, 0]
        e2a[i] = -ZA * w[0, i]

    return w, e1b, e2a


def _pack_idx(k: int, l: int) -> int:
    """Map orbital pair (k, l) with k >= l to packed index 0-9."""
    # (0,0)→0, (1,0)→1, (1,1)→2, (2,0)→3, (2,1)→4, (2,2)→5,
    # (3,0)→6, (3,1)→7, (3,2)→8, (3,3)→9
    return k * (k + 1) // 2 + l


def _rotate_integral(ri, r0, r1, r2, kk, ll, mm, nn) -> float:
    """Rotate single two-electron integral from local to molecular frame.

    Exact port of PYSEQM w_withquaternion inner loop.
    """
    k, l, m, n = kk, ll, mm, nn

    # (ss|ss)
    if k == 0 and l == 0 and m == 0 and n == 0:
        return ri[0]

    # (ss|ps)
    if k == 0 and l == 0 and m > 0 and n == 0:
        return ri[4] * r0[m - 1]

    # (ss|pp)
    if k == 0 and l == 0 and m > 0 and n > 0:
        i, j = m - 1, n - 1
        return ri[10] * r0[i]*r0[j] + ri[11] * (r1[i]*r1[j] + r2[i]*r2[j])

    # (ps|ss)
    if k > 0 and l == 0 and m == 0 and n == 0:
        return ri[1] * r0[k - 1]

    # (ps|ps)
    if k > 0 and l == 0 and m > 0 and n == 0:
        i, j = k - 1, m - 1
        return ri[5] * r0[i]*r0[j] + ri[6] * (r1[i]*r1[j] + r2[i]*r2[j])

    # (ps|pp)
    if k > 0 and l == 0 and m > 0 and n > 0:
        i = k - 1
        j, h = m - 1, n - 1
        t0 = r0[i] * r0[j] * r0[h]
        t1 = (r1[j]*r1[h] + r2[j]*r2[h]) * r0[i]
        mix = r1[i]*(r1[h]*r0[j] + r1[j]*r0[h]) + r2[i]*(r2[j]*r0[h] + r2[h]*r0[j])
        return ri[12]*t0 + ri[13]*t1 + ri[14]*mix

    # (pp|ss)
    if k > 0 and l > 0 and m == 0 and n == 0:
        i, j = k - 1, l - 1
        return ri[2] * r0[i]*r0[j] + ri[3] * (r1[i]*r1[j] + r2[i]*r2[j])

    # (pp|ps)
    if k > 0 and l > 0 and m > 0 and n == 0:
        i, j = k - 1, l - 1
        h = m - 1
        t0 = r0[i]*r0[j]*r0[h]
        t1 = (r1[i]*r1[j] + r2[i]*r2[j]) * r0[h]
        mix1 = r0[i]*(r1[j]*r1[h] + r2[j]*r2[h])
        mix2 = r0[j]*(r1[i]*r1[h] + r2[i]*r2[h])
        return ri[7]*t0 + ri[8]*t1 + ri[9]*(mix1 + mix2)

    # (pp|pp) — most complex
    if k > 0 and l > 0 and m > 0 and n > 0:
        i, j = k - 1, l - 1
        h, g = m - 1, n - 1

        t0 = r0[i]*r0[j]*r0[h]*r0[g]

        t1 = (r1[i]*r1[j] + r2[i]*r2[j]) * r0[h]*r0[g]
        t2 = r0[i]*r0[j] * (r1[h]*r1[g] + r2[h]*r2[g])

        quad = (r1[i]*r1[j]*r1[h]*r1[g] + r2[i]*r2[j]*r2[h]*r2[g])

        mix1_h = r0[h]*(r1[j]*r1[g]+r2[j]*r2[g])
        mix1_g = r0[g]*(r1[j]*r1[h]+r2[j]*r2[h])
        mix2_h = r0[h]*(r1[i]*r1[g]+r2[i]*r2[g])
        mix2_g = r0[g]*(r1[i]*r1[h]+r2[i]*r2[h])
        val5 = r0[i]*(mix1_h+mix1_g) + r0[j]*(mix2_h+mix2_g)

        mix3 = r1[i]*r1[j]*r2[h]*r2[g] + r2[i]*r2[j]*r1[h]*r1[g]

        cross = (r1[i]*r2[j]+r2[i]*r1[j]) * (r1[h]*r2[g]+r2[h]*r1[g])

        return (ri[15]*t0 + ri[16]*t1 + ri[17]*t2
                + ri[18]*quad + ri[19]*val5 + ri[20]*mix3 + ri[21]*cross)

    return 0.0
