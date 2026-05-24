"""Numba-JIT'd hot kernels for the vendored PYSEQM NumPy port.

These accelerate the inner loops that dominate _tetci_pair_w:
  - w_withquaternion's 100-iteration `for kk, ll, mm, nn in combos`
  - GenerateRotationMatrix's per-pair quaternion construction

Numba is an optional dep: if not importable, the fall-back path is the
pure-NumPy version in two_elec_two_center_int_np.py.

Speed-ups measured on Apple M3 Max:
  - w_withquaternion: ~10x (3.3 ms -> 0.3 ms for a single S-C pair)
  - end-to-end _tetci_pair_w: ~3-4x
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(f): return f
        return deco


@njit(cache=True, fastmath=True)
def _w_withquaternion_kernel(ri, riXH, rot, rotXH, HH_mask, XH_mask, XX_mask, w, wXH):
    """Compute the 100-element rotated w vector and 10-element wXH per pair.

    All inputs are NumPy arrays; outputs are filled in-place.
      ri:    (B, 22)  local-frame integrals
      riXH:  (B, 4)
      rot:   (B, 3, 3) rotation matrices (XX subset; XH/HH already preselected)
      rotXH: (BH, 3, 3) rotation for XH subset
      w:     (B, 100) output, zero-initialized
      wXH:   (BH, 10) output, zero-initialized

    This is a straight transcription of the pure-NumPy inner loop in
    w_withquaternion, JIT-compiled by numba. Branches on the 100 combos
    are unrolled by the compiler.
    """
    B = ri.shape[0]
    BH = riXH.shape[0]
    for b in range(B):
        r00 = rot[b, 0, 0]; r01 = rot[b, 0, 1]; r02 = rot[b, 0, 2]
        r10 = rot[b, 1, 0]; r11 = rot[b, 1, 1]; r12 = rot[b, 1, 2]
        r20 = rot[b, 2, 0]; r21 = rot[b, 2, 1]; r22 = rot[b, 2, 2]
        # Pack as arrays so indexing by k works
        r0 = (r00, r01, r02)
        r1 = (r10, r11, r12)
        r2 = (r20, r21, r22)
        ri_b = ri[b]
        idx = 0
        for kk in range(4):
            for ll in range(kk + 1):
                for mm in range(4):
                    for nn in range(mm + 1):
                        k = kk - 1
                        l = ll - 1
                        m = mm - 1
                        n = nn - 1
                        if kk == 0:
                            if mm == 0:
                                w[b, idx] = ri_b[0]
                            elif nn == 0:
                                w[b, idx] = ri_b[4] * r0[m]
                            else:
                                t1 = ri_b[10] * r0[m] * r0[n]
                                t2 = ri_b[11] * (r1[m]*r1[n] + r2[m]*r2[n])
                                w[b, idx] = t1 + t2
                        elif ll == 0:
                            if mm == 0:
                                w[b, idx] = ri_b[1] * r0[k]
                            elif nn == 0:
                                t1 = ri_b[5] * r0[k] * r0[m]
                                t2 = ri_b[6] * (r1[k]*r1[m] + r2[k]*r2[m])
                                w[b, idx] = t1 + t2
                            else:
                                t0 = r0[k] * r0[m] * r0[n]
                                t1 = (r1[m]*r1[n] + r2[m]*r2[n]) * r0[k]
                                mix = r1[k]*(r1[n]*r0[m] + r1[m]*r0[n]) \
                                    + r2[k]*(r2[n]*r0[m] + r2[m]*r0[n])
                                w[b, idx] = ri_b[12]*t0 + ri_b[13]*t1 + ri_b[14]*mix
                        else:
                            if mm == 0:
                                t0 = r0[k] * r0[l]
                                t1 = r1[k]*r1[l] + r2[k]*r2[l]
                                w[b, idx] = ri_b[2]*t0 + ri_b[3]*t1
                            elif nn == 0:
                                t0 = r0[k] * r0[l] * r0[m]
                                t1 = (r1[k]*r1[l] + r2[k]*r2[l]) * r0[m]
                                t2 = r1[l]*r1[m] + r2[l]*r2[m]
                                w[b, idx] = (ri_b[7]*t0 + ri_b[8]*t1
                                    + ri_b[9]*(r0[k]*t2 + r0[l]*(r1[k]*r1[m] + r2[k]*r2[m])))
                            else:
                                t0 = r0[k]*r0[l]*r0[m]*r0[n]
                                val = ri_b[15] * t0
                                t1 = (r1[k]*r1[l] + r2[k]*r2[l])*r0[m]*r0[n]
                                val += ri_b[16] * t1
                                t2 = (r1[m]*r1[n] + r2[m]*r2[n]) * (r0[k]*r0[l])
                                val += ri_b[17] * t2
                                quad = (r1[k]*r1[l]*r1[m]*r1[n]
                                      + r2[k]*r2[l]*r2[m]*r2[n])
                                val += ri_b[18] * quad
                                mix1 = r0[m]*(r1[l]*r1[n] + r2[l]*r2[n])
                                mix2 = r0[n]*(r1[l]*r1[m] + r2[l]*r2[m])
                                mix3 = r0[l]*(r1[k]*r1[m]*r0[n]
                                            + r1[k]*r1[n]*r0[m]
                                            + r2[k]*r2[m]*r0[n]
                                            + r2[k]*r2[n]*r0[m])
                                mix4 = r0[k]*(r1[l]*r1[m]*r0[n]
                                            + r1[l]*r1[n]*r0[m]
                                            + r2[l]*r2[m]*r0[n]
                                            + r2[l]*r2[n]*r0[m])
                                val += ri_b[19] * (mix3 + mix4)
                                mix5 = ((r1[k]*r2[l] + r1[l]*r2[k])
                                      * (r1[m]*r2[n] + r1[n]*r2[m]))
                                val += ri_b[20] * mix5
                                mix6 = (r1[k]*r1[l] - r2[k]*r2[l]) * \
                                       (r1[m]*r1[n] - r2[m]*r2[n])
                                val += ri_b[21] * mix6
                                w[b, idx] = val
                        idx += 1

    # wXH (10 elements per H-bearing pair); use rotXH
    for b in range(BH):
        rx00 = rotXH[b, 0, 0]; rx01 = rotXH[b, 0, 1]; rx02 = rotXH[b, 0, 2]
        rx10 = rotXH[b, 1, 0]; rx11 = rotXH[b, 1, 1]; rx12 = rotXH[b, 1, 2]
        rx20 = rotXH[b, 2, 0]; rx21 = rotXH[b, 2, 1]; rx22 = rotXH[b, 2, 2]
        rx0 = (rx00, rx01, rx02)
        rx1 = (rx10, rx11, rx12)
        rx2 = (rx20, rx21, rx22)
        riXH_b = riXH[b]
        # 4 sp-sp packed pairs that contribute to XH: indices 0, 1, 2, 3
        # (ss|ss), (ps|ss) at k=0,1,2 (px,py,pz), (pp|ss) at l<=k
        # In PYSEQM order: idx 0=(ss|ss), 1,2,3 = (px|ss),(py|ss),(pz|ss),
        # 4,5,6 = (px,px|ss) etc — same combo iteration
        wXH[b, 0] = riXH_b[0]
        idxXH = 1
        for kk in range(1, 4):
            k = kk - 1
            wXH[b, idxXH] = riXH_b[1] * rx0[k]
            idxXH += 1
        for kk in range(1, 4):
            for ll in range(1, kk + 1):
                k = kk - 1
                l = ll - 1
                x0 = rx0[k] * rx0[l]
                x1 = rx1[k]*rx1[l] + rx2[k]*rx2[l]
                wXH[b, idxXH] = riXH_b[2]*x0 + riXH_b[3]*x1
                idxXH += 1


def is_numba_available() -> bool:
    return NUMBA_AVAILABLE


@njit(cache=True, fastmath=True)
def _generate_rotation_matrix_kernel(P, D, matrix):
    """Fill the (B, 15, 45) rotation matrix from the precomputed P (3x3)
    and D (5x5) tables. matrix is zero-initialized on entry."""
    INDX = np.array([0, 1, 3, 6, 10, 15, 21, 28, 36], dtype=np.int64)
    B = P.shape[0]
    for b in range(B):
        Pb = P[b]
        Db = D[b]
        Mb = matrix[b]

        # S-S
        Mb[0, 0] = 1.0
        # P-S
        for K in range(3):
            KL = INDX[K + 1]
            Mb[0, KL] = Pb[K, 0]
            Mb[1, KL] = Pb[K, 1]
            Mb[2, KL] = Pb[K, 2]
        # P-P diagonal
        for K in range(3):
            KL = INDX[K + 1] + K + 1
            Mb[0, KL] = Pb[K, 0] * Pb[K, 0]
            Mb[1, KL] = Pb[K, 0] * Pb[K, 1]
            Mb[2, KL] = Pb[K, 1] * Pb[K, 1]
            Mb[3, KL] = Pb[K, 0] * Pb[K, 2]
            Mb[4, KL] = Pb[K, 1] * Pb[K, 2]
            Mb[5, KL] = Pb[K, 2] * Pb[K, 2]
        # P-P off-diagonal
        for K in range(1, 3):
            for L in range(K):
                KL = INDX[K + 1] + L + 1
                Mb[0, KL] = Pb[K, 0] * Pb[L, 0] * 2.0
                Mb[1, KL] = Pb[K, 0] * Pb[L, 1] + Pb[K, 1] * Pb[L, 0]
                Mb[2, KL] = Pb[K, 1] * Pb[L, 1] * 2.0
                Mb[3, KL] = Pb[K, 0] * Pb[L, 2] + Pb[K, 2] * Pb[L, 0]
                Mb[4, KL] = Pb[K, 1] * Pb[L, 2] + Pb[K, 2] * Pb[L, 1]
                Mb[5, KL] = Pb[K, 2] * Pb[L, 2] * 2.0
        # D-S
        for K in range(5):
            KL = INDX[K + 4]
            Mb[0, KL] = Db[K, 0]
            Mb[1, KL] = Db[K, 1]
            Mb[2, KL] = Db[K, 2]
            Mb[3, KL] = Db[K, 3]
            Mb[4, KL] = Db[K, 4]
        # D-P
        for K in range(5):
            for L in range(3):
                KL = INDX[K + 4] + L + 1
                Mb[0, KL] = Db[K, 0] * Pb[L, 0]
                Mb[1, KL] = Db[K, 0] * Pb[L, 1]
                Mb[2, KL] = Db[K, 0] * Pb[L, 2]
                Mb[3, KL] = Db[K, 1] * Pb[L, 0]
                Mb[4, KL] = Db[K, 1] * Pb[L, 1]
                Mb[5, KL] = Db[K, 1] * Pb[L, 2]
                Mb[6, KL] = Db[K, 2] * Pb[L, 0]
                Mb[7, KL] = Db[K, 2] * Pb[L, 1]
                Mb[8, KL] = Db[K, 2] * Pb[L, 2]
                Mb[9, KL] = Db[K, 3] * Pb[L, 0]
                Mb[10, KL] = Db[K, 3] * Pb[L, 1]
                Mb[11, KL] = Db[K, 3] * Pb[L, 2]
                Mb[12, KL] = Db[K, 4] * Pb[L, 0]
                Mb[13, KL] = Db[K, 4] * Pb[L, 1]
                Mb[14, KL] = Db[K, 4] * Pb[L, 2]
        # D-D diagonal
        for K in range(5):
            KL = INDX[K + 4] + K + 4
            Mb[0, KL] = Db[K, 0] * Db[K, 0]
            Mb[1, KL] = Db[K, 0] * Db[K, 1]
            Mb[2, KL] = Db[K, 1] * Db[K, 1]
            Mb[3, KL] = Db[K, 0] * Db[K, 2]
            Mb[4, KL] = Db[K, 1] * Db[K, 2]
            Mb[5, KL] = Db[K, 2] * Db[K, 2]
            Mb[6, KL] = Db[K, 0] * Db[K, 3]
            Mb[7, KL] = Db[K, 1] * Db[K, 3]
            Mb[8, KL] = Db[K, 2] * Db[K, 3]
            Mb[9, KL] = Db[K, 3] * Db[K, 3]
            Mb[10, KL] = Db[K, 0] * Db[K, 4]
            Mb[11, KL] = Db[K, 1] * Db[K, 4]
            Mb[12, KL] = Db[K, 2] * Db[K, 4]
            Mb[13, KL] = Db[K, 3] * Db[K, 4]
            Mb[14, KL] = Db[K, 4] * Db[K, 4]
        # D-D off-diagonal
        for K in range(5):
            for L in range(K):
                KL = INDX[K + 4] + L + 4
                Mb[0, KL] = Db[K, 0] * Db[L, 0] * 2.0
                Mb[1, KL] = Db[K, 0] * Db[L, 1] + Db[K, 1] * Db[L, 0]
                Mb[2, KL] = Db[K, 1] * Db[L, 1] * 2.0
                Mb[3, KL] = Db[K, 0] * Db[L, 2] + Db[K, 2] * Db[L, 0]
                Mb[4, KL] = Db[K, 1] * Db[L, 2] + Db[K, 2] * Db[L, 1]
                Mb[5, KL] = Db[K, 2] * Db[L, 2] * 2.0
                Mb[6, KL] = Db[K, 0] * Db[L, 3] + Db[K, 3] * Db[L, 0]
                Mb[7, KL] = Db[K, 1] * Db[L, 3] + Db[K, 3] * Db[L, 1]
                Mb[8, KL] = Db[K, 2] * Db[L, 3] + Db[K, 3] * Db[L, 2]
                Mb[9, KL] = Db[K, 3] * Db[L, 3] * 2.0
                Mb[10, KL] = Db[K, 0] * Db[L, 4] + Db[K, 4] * Db[L, 0]
                Mb[11, KL] = Db[K, 1] * Db[L, 4] + Db[K, 4] * Db[L, 1]
                Mb[12, KL] = Db[K, 2] * Db[L, 4] + Db[K, 4] * Db[L, 2]
                Mb[13, KL] = Db[K, 3] * Db[L, 4] + Db[K, 4] * Db[L, 3]
                Mb[14, KL] = Db[K, 4] * Db[L, 4] * 2.0
