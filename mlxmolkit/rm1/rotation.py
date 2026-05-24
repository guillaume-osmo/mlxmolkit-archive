"""
Rotation of NDDO two-center integrals from local frame to molecular frame.

The local frame has the bond axis along x (MOPAC convention: atom i → j).
The rotation matrix R transforms local-frame integrals to the molecular frame:

  w(μν|λσ)_mol = Σ R_μa R_νb R_λc R_σd * ri(ab|cd)_local

Uses quaternion-based rotation for numerical stability (no singularity
when bond is along z-axis).

Reference: rotate.f in MOPAC, w_withquaternion in PYSEQM.
"""
from __future__ import annotations

import numpy as np
from .two_center_integrals import two_center_integrals
from .params import RM1_PARAMS, ElementParams


def _rotation_matrix(v: np.ndarray) -> np.ndarray:
    """Build 3x3 rotation matrix that rotates unit vector v to x-axis.

    Uses quaternion q = (0, vz, -vy, 1+vx) normalized.
    """
    vx, vy, vz = v[0], v[1], v[2]
    w = 1.0 + vx

    if abs(w) < 1e-7:
        # Antipodal case: v ≈ (-1,0,0), use 180° flip about z
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

    # Quaternion (0, vz, -vy, 1+vx)
    qy = vz
    qz = -vy
    qw = w
    norm = np.sqrt(qy * qy + qz * qz + qw * qw)
    qy /= norm
    qz /= norm
    qw /= norm

    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = -2 * qz * qw
    R[0, 2] = 2 * qy * qw
    R[1, 0] = 2 * qz * qw
    R[1, 1] = 1 - 2 * qz * qz
    R[1, 2] = 2 * qy * qz
    R[2, 0] = -2 * qy * qw
    R[2, 1] = 2 * qy * qz
    R[2, 2] = 1 - 2 * qy * qy
    return R


def rotate_integrals_to_molecular_frame(
    pA: ElementParams,
    pB: ElementParams,
    coordA: np.ndarray,
    coordB: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rotated two-center integrals in the molecular frame.

    Returns:
        w: (nA*nB, nA*nB) two-electron integral matrix in molecular frame.
           Indexed as w[(μ_A,ν_A)][(λ_B,σ_B)] where μ,ν on atom A, λ,σ on atom B.
           Stored as flattened (10,10) for sp basis (4+4 → 10 unique pairs).
        e1b: (4,4) electron on A attracted to nucleus B
        e2a: (4,4) electron on B attracted to nucleus A
    """
    R_vec = coordB - coordA
    R = np.linalg.norm(R_vec)
    if R < 1e-10:
        nA = min(pA.n_basis, 4)
        nB = min(pB.n_basis, 4)
        return np.zeros((4, 4, 4, 4)), np.zeros((4, 4)), np.zeros((4, 4))

    # Unit vector from A to B (MOPAC convention: negative for rotation)
    v = -R_vec / R

    # Rotation matrix
    rot = _rotation_matrix(v)
    r0 = rot[0]  # sigma direction (along bond)
    r1 = rot[1]  # pi_x direction
    r2 = rot[2]  # pi_y direction

    # Local-frame integrals
    ri, core, pair_type = two_center_integrals(pA, pB, R)

    # Clamp to sp block (d-orbitals handled separately in SCF)
    nA = min(pA.n_basis, 4)
    nB = min(pB.n_basis, 4)

    # Rotate to molecular frame
    # w is stored as (kk,ll | mm,nn) with kk>=ll, mm>=nn, indexed linearly
    # For sp basis: 10 unique pairs on each atom → 10×10 = 100 elements
    # But we need the 4×4 blocks for the Fock matrix

    # Build the full w matrix as 4x4 x 4x4 = (nA, nA, nB, nB)
    w = np.zeros((4, 4, 4, 4))

    # e1b[μ,ν] = electron(μ,ν on A) attracted to nucleus B
    # e2a[μ,ν] = electron(μ,ν on B) attracted to nucleus A
    e1b = np.zeros((4, 4))
    e2a = np.zeros((4, 4))

    ZA = float(pA.n_valence)
    ZB = float(pB.n_valence)

    if pair_type == 'HH':
        w[0, 0, 0, 0] = ri[0]
        e1b[0, 0] = -ZB * ri[0]
        e2a[0, 0] = -ZA * ri[0]
        return w, e1b, e2a

    if pair_type == 'XH':
        # A=heavy(sp), B=H(s only)
        # ri: (ss|ss), (sσ|ss), (σσ|ss), (ππ|ss)
        # Rotate: w(kl|00) where k,l are on A

        # (ss|ss)
        w[0, 0, 0, 0] = ri[0]

        # (ps|ss) — rotated
        for k in range(3):
            w[k + 1, 0, 0, 0] = ri[1] * r0[k]
            w[0, k + 1, 0, 0] = w[k + 1, 0, 0, 0]

        # (pp|ss) — rotated
        for k in range(3):
            for l in range(3):
                w[k + 1, l + 1, 0, 0] = ri[2] * r0[k] * r0[l] + ri[3] * (r1[k] * r1[l] + r2[k] * r2[l])

        # Core attractions
        e1b[0, 0] = -ZB * ri[0]
        for k in range(3):
            e1b[k + 1, 0] = -ZB * ri[1] * r0[k]
            e1b[0, k + 1] = e1b[k + 1, 0]
        for k in range(3):
            e1b[k + 1, k + 1] = -ZB * (ri[2] * r0[k] ** 2 + ri[3] * (r1[k] ** 2 + r2[k] ** 2))
            for l in range(k + 1, 3):
                e1b[k + 1, l + 1] = -ZB * (ri[2] * r0[k] * r0[l] + ri[3] * (r1[k] * r1[l] + r2[k] * r2[l]))
                e1b[l + 1, k + 1] = e1b[k + 1, l + 1]

        e2a[0, 0] = -ZA * ri[0]

        return w, e1b, e2a

    if pair_type == 'HX':
        # A=H(s), B=heavy(sp)
        # Swap and transpose
        w_swap, e1b_swap, e2a_swap = rotate_integrals_to_molecular_frame(pB, pA, coordB, coordA)
        # Transpose: w(kl|mn) → w(mn|kl)
        w_t = np.transpose(w_swap, (2, 3, 0, 1))
        return w_t, e2a_swap, e1b_swap

    # XX: Full sp-sp rotation
    # Apply the rotation formula from PYSEQM's w_withquaternion
    # w(kk,ll | mm,nn) = Σ_local ri[idx] * rotation_products

    # All (kk,ll,mm,nn) combos where kk>=ll, mm>=nn
    # Mapped to linear index in the 10×10 w matrix
    idx = 0
    for kk in range(4):
        for ll in range(kk + 1):
            for mm in range(4):
                for nn in range(mm + 1):
                    k = kk - 1
                    l = ll - 1
                    m = mm - 1
                    n = nn - 1

                    val = 0.0

                    if kk == 0:
                        if mm == 0:
                            # (ss|ss)
                            val = ri[0]
                        elif nn == 0:
                            # (ss|ps)
                            val = ri[4] * r0[m]
                        else:
                            # (ss|pp)
                            val = ri[10] * r0[m] * r0[n] + ri[11] * (r1[m] * r1[n] + r2[m] * r2[n])

                    elif ll == 0:
                        if mm == 0:
                            # (ps|ss)
                            val = ri[1] * r0[k]
                        elif nn == 0:
                            # (ps|ps)
                            val = ri[5] * r0[k] * r0[m] + ri[6] * (r1[k] * r1[m] + r2[k] * r2[m])
                        else:
                            # (ps|pp)
                            t0 = r0[k] * r0[m] * r0[n]
                            t1 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k]
                            mix = r1[k] * (r1[n] * r0[m] + r1[m] * r0[n]) + r2[k] * (r2[m] * r0[n] + r2[n] * r0[m])
                            val = ri[12] * t0 + ri[13] * t1 + ri[14] * mix
                    else:
                        if mm == 0:
                            # (pp|ss)
                            val = ri[2] * r0[k] * r0[l] + ri[3] * (r1[k] * r1[l] + r2[k] * r2[l])
                        elif nn == 0:
                            # (pp|ps)
                            t0 = r0[k] * r0[l] * r0[m]
                            t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m]
                            t2a = r1[l] * r1[m] + r2[l] * r2[m]
                            t2b = r1[k] * r1[m] + r2[k] * r2[m]
                            val = ri[7] * t0 + ri[8] * t1 + ri[9] * (r0[k] * t2a + r0[l] * t2b)
                        else:
                            # (pp|pp)
                            t0 = r0[k] * r0[l] * r0[m] * r0[n]
                            t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m] * r0[n]
                            t2 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k] * r0[l]
                            quad = r1[k] * r1[l] * r1[m] * r1[n] + r2[k] * r2[l] * r2[m] * r2[n]

                            mix1 = r0[m] * (r1[l] * r1[n] + r2[l] * r2[n])
                            mix2 = r0[n] * (r1[l] * r1[m] + r2[l] * r2[m])
                            val5 = (r0[k] * (mix1 + mix2)
                                  + r0[l] * (r0[m] * (r1[k] * r1[n] + r2[k] * r2[n])
                                           + r0[n] * (r1[k] * r1[m] + r2[k] * r2[m])))

                            mix3 = r1[k] * r1[l] * r2[m] * r2[n] + r2[k] * r2[l] * r1[m] * r1[n]
                            cross = (r1[k] * r2[l] + r2[k] * r1[l]) * (r1[m] * r2[n] + r2[m] * r1[n])

                            val = (ri[15] * t0 + ri[16] * t1 + ri[17] * t2
                                 + ri[18] * quad + ri[19] * val5 + ri[20] * mix3 + ri[21] * cross)

                    # Store in the 4×4×4×4 w tensor
                    w[kk, ll, mm, nn] = val
                    w[ll, kk, mm, nn] = val
                    w[kk, ll, nn, mm] = val
                    w[ll, kk, nn, mm] = val

                    idx += 1

    # Core attractions from the rotated w matrix
    for mu in range(nA):
        for nu in range(mu + 1):
            e1b[mu, nu] = -ZB * w[mu, nu, 0, 0]
            e1b[nu, mu] = e1b[mu, nu]

    for mu in range(nB):
        for nu in range(mu + 1):
            e2a[mu, nu] = -ZA * w[0, 0, mu, nu]
            e2a[nu, mu] = e2a[mu, nu]

    return w, e1b, e2a
