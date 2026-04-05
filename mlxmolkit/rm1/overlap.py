"""
Slater-type orbital overlap integrals for NDDO methods.

Exact port of PYSEQM's diatom_overlap_matrix_PM6_SP:
  - SET function with z1=zeta_A, z2=zeta_B convention
  - S111, S211, S121, S221, S222 local-frame overlaps
  - Quaternion rotation to molecular frame (rotate_with_quaternion)
  - Assembly: di[p,s]=S211*c0, di[s,p]=-S121*c0,
              di[p,p]=-S221*r0⊗r0 + S222*(r1⊗r1+r2⊗r2)

Reference: diat_overlap_PM6_SP.py in PYSEQM (2025 update).
"""
from __future__ import annotations

import numpy as np
from .params import ElementParams, ANG_TO_BOHR
from .rotation import _rotation_matrix


def _aintgs(alpha: float, n_max: int = 5) -> np.ndarray:
    """A-integrals: a_k(alpha) = exp(-alpha)/alpha * recurrence.

    a_1 = exp(-alpha)/alpha
    a_k = a_1 + (k-1)*a_{k-1}/alpha

    Matches PYSEQM aintgs() exactly.
    """
    if abs(alpha) < 1e-10:
        return np.zeros(n_max)

    a = np.zeros(n_max)
    a[0] = np.exp(-alpha) / alpha
    for k in range(1, n_max):
        a[k] = a[0] + k * a[k - 1] / alpha
    return a


def _bintgs(beta: float, n_max: int = 5) -> np.ndarray:
    """B-integrals: b_k(beta).

    b_1 = 2*sinh(beta)/beta  (even function)
    b_k recurrence with alternating signs.

    Three regimes matching PYSEQM PM6_SP:
      |beta| <= 1e-6:  b(0) limit values
      1e-6 < |beta| <= 0.5:  Taylor series (x^2, x^4, x^6 terms)
      |beta| > 0.5:  exact recurrence
    """
    b = np.zeros(n_max)
    x = beta

    if abs(x) <= 1e-6:
        # Exact at beta=0
        b[0] = 2.0
        b[1] = 0.0
        b[2] = 2.0 / 3.0
        b[3] = 0.0
        if n_max > 4:
            b[4] = 2.0 / 5.0
        return b

    if abs(x) <= 0.5:
        # Taylor series expansion (PYSEQM PM6_SP lines 624-665)
        # Even-indexed b's: b_{i+1}(x) = sum_{m=0,2,4,6} x^m * 2/(m!*(m+i+1))
        x2 = x * x
        x4 = x2 * x2
        x6 = x2 * x4
        b[0] = 2.0 + x2 / 3.0 + x4 / 60.0 + x6 / 2520.0
        b[2] = 2.0 / 3.0 + x2 / 5.0 + x4 / 84.0 + x6 / 3240.0
        if n_max > 4:
            b[4] = 2.0 / 5.0 + x2 / 7.0 + x4 / 108.0 + x6 / 3960.0

        # Odd-indexed b's: b_{i+1}(x) = sum_{m=1,3,5} -x^m * 2/(m!*(m+i+1))
        x3 = x * x2
        x5 = x2 * x3
        b[1] = -2.0 / 3.0 * x - x3 / 15.0 - x5 / 420.0
        b[3] = -2.0 / 5.0 * x - x3 / 21.0 - x5 / 540.0
        return b

    # Exact recurrence for |beta| > 0.5
    tx = np.exp(x) / x
    tmx = -np.exp(-x) / x
    b[0] = tx + tmx                          # 2*sinh(beta)/beta
    b[1] = -tx + tmx + b[0] / x
    b[2] = tx + tmx + 2.0 * b[1] / x
    b[3] = -tx + tmx + 3.0 * b[2] / x
    if n_max > 4:
        b[4] = tx + tmx + 4.0 * b[3] / x
    return b


def overlap_molecular_frame(
    pA: ElementParams, pB: ElementParams,
    coordA: np.ndarray, coordB: np.ndarray,
) -> np.ndarray:
    """Compute overlap matrix in the molecular frame.

    Exact port of PYSEQM diatom_overlap_matrix_PM6_SP.

    Args:
        pA, pB: element parameters for atoms A and B
        coordA, coordB: coordinates in Angstrom

    Returns:
        di: (nA, nB) overlap matrix in molecular frame
            Indices: 0=s, 1=px, 2=py, 3=pz
    """
    R_vec = coordB - coordA
    R = np.linalg.norm(R_vec)
    nA = pA.n_basis
    nB = pB.n_basis

    if R < 1e-10:
        S = np.zeros((nA, nB))
        for k in range(min(nA, nB)):
            S[k, k] = 1.0
        return S

    R_bohr = R * ANG_TO_BOHR

    # Quantum numbers
    qnA = 1 if pA.Z <= 2 else 2
    qnB = 1 if pB.Z <= 2 else 2

    # PYSEQM convention: heavier atom first (qnA >= qnB)
    # If lighter atom is first, swap and transpose
    if qnA < qnB:
        di_swap = overlap_molecular_frame(pB, pA, coordB, coordA)
        return di_swap.T

    # Determine jcall (PYSEQM convention)
    if qnA == 1 and qnB == 1:
        jcall = 2
    elif qnA == 2 and qnB == 1:
        jcall = 3
    elif qnA == 2 and qnB == 2:
        jcall = 4
    else:
        raise ValueError(f"Unsupported pair: qn=({qnA},{qnB})")

    # ================================================================
    # SET function: z1 = zeta_A, z2 = zeta_B  (PYSEQM convention)
    # alpha = 0.5 * R * (z1 + z2)
    # beta  = 0.5 * R * (z1 - z2)
    # ================================================================

    # A111, B111: s_A - s_B
    alpha_ss = 0.5 * R_bohr * (pA.zeta_s + pB.zeta_s)
    beta_ss = 0.5 * R_bohr * (pA.zeta_s - pB.zeta_s)
    A111 = _aintgs(alpha_ss)
    B111 = _bintgs(beta_ss)

    # A211, B211: p_A - s_B  (z1 = zA_p, z2 = zB_s)
    if nA > 1:
        alpha_ps = 0.5 * R_bohr * (pA.zeta_p + pB.zeta_s)
        beta_ps = 0.5 * R_bohr * (pA.zeta_p - pB.zeta_s)
        A211 = _aintgs(alpha_ps)
        B211 = _bintgs(beta_ps)
    else:
        A211 = np.zeros(5)
        B211 = np.zeros(5)

    # A121, B121: s_A - p_B  (z1 = zA_s, z2 = zB_p)
    if nB > 1:
        alpha_sp = 0.5 * R_bohr * (pA.zeta_s + pB.zeta_p)
        beta_sp = 0.5 * R_bohr * (pA.zeta_s - pB.zeta_p)
        A121 = _aintgs(alpha_sp)
        B121 = _bintgs(beta_sp)
    else:
        A121 = np.zeros(5)
        B121 = np.zeros(5)

    # A22, B22: p_A - p_B  (z1 = zA_p, z2 = zB_p)
    if nA > 1 and nB > 1:
        alpha_pp = 0.5 * R_bohr * (pA.zeta_p + pB.zeta_p)
        beta_pp = 0.5 * R_bohr * (pA.zeta_p - pB.zeta_p)
        A22 = _aintgs(alpha_pp)
        B22 = _bintgs(beta_pp)
    else:
        A22 = np.zeros(5)
        B22 = np.zeros(5)

    # ================================================================
    # Local-frame overlaps (exact PYSEQM PM6_SP formulas)
    # ================================================================
    S111 = 0.0
    S211 = 0.0
    S121 = 0.0
    S221 = 0.0
    S222 = 0.0

    if jcall == 2:
        # (1s|1s) — PYSEQM jcall==2
        S111 = ((pA.zeta_s * pB.zeta_s * R_bohr ** 2) ** 1.5
                * (A111[2] * B111[0] - B111[2] * A111[0])
                / 4.0)

    elif jcall == 3:
        # (2s_A|1s_B) — PYSEQM jcall==3
        S111 = (pB.zeta_s ** 1.5 * pA.zeta_s ** 2.5 * R_bohr ** 4
                * (A111[3] * B111[0] - B111[3] * A111[0]
                   + A111[2] * B111[1] - B111[2] * A111[1])
                / (np.sqrt(3.0) * 8.0))

        # (2pσ_A|1s_B) — PYSEQM S211 jcall==3
        if nA > 1:
            S211 = (pB.zeta_s ** 1.5 * pA.zeta_p ** 2.5 * R_bohr ** 4
                    * (A211[2] * B211[0] - B211[2] * A211[0]
                       + A211[3] * B211[1] - B211[3] * A211[1])
                    / 8.0)

    elif jcall == 4:
        # (2s|2s) — PYSEQM jcall==4
        S111 = ((pA.zeta_s * pB.zeta_s) ** 2.5 * R_bohr ** 5
                * (A111[4] * B111[0] + B111[4] * A111[0]
                   - 2.0 * A111[2] * B111[2])
                / 48.0)

        if nA > 1 and nB > 1:
            # (2pσ_A|2s_B) = S211 — PYSEQM jcall==4
            S211 = ((pB.zeta_s * pA.zeta_p) ** 2.5 * R_bohr ** 5
                    * (A211[3] * (B211[0] - B211[2])
                       - A211[1] * (B211[2] - B211[4])
                       + B211[3] * (A211[0] - A211[2])
                       - B211[1] * (A211[2] - A211[4]))
                    / (16.0 * np.sqrt(3.0)))

            # (2s_A|2pσ_B) = S121 — PYSEQM jcall==4
            # NOTE: B-terms have OPPOSITE sign from S211 (PYSEQM convention)
            S121 = ((pB.zeta_p * pA.zeta_s) ** 2.5 * R_bohr ** 5
                    * (A121[3] * (B121[0] - B121[2])
                       - A121[1] * (B121[2] - B121[4])
                       - B121[3] * (A121[0] - A121[2])
                       + B121[1] * (A121[2] - A121[4]))
                    / (16.0 * np.sqrt(3.0)))

            # (2pσ|2pσ) = S221 — PYSEQM jcall==4
            # NOTE: This is NOT the same as our old S_local[1,1]!
            w = (pB.zeta_p * pA.zeta_p) ** 2.5 * R_bohr ** 5 / 16.0
            S221 = -w * (B22[2] * (A22[4] + A22[0])
                         - A22[2] * (B22[4] + B22[0]))

            # (2pπ|2pπ) = S222 — PYSEQM jcall==4
            S222 = (0.5 * w
                    * (A22[4] * (B22[0] - B22[2])
                       - B22[4] * (A22[0] - A22[2])
                       - A22[2] * B22[0]
                       + B22[2] * A22[0]))

    # ================================================================
    # Rotation to molecular frame (exact PYSEQM PM6_SP assembly)
    # ================================================================

    # Unit vector from A to B (positive direction, PYSEQM xij convention)
    v = R_vec / R

    # Rotation matrix: rotate v to x-axis
    # Same quaternion as PYSEQM's rotate_with_quaternion
    rot = _rotation_matrix(v)

    # PYSEQM: rot = rot.transpose(1,2), then c0 = rot[:,:,0]
    # c0 = column 0 of transposed rot = row 0 of original rot
    r0 = rot[0]  # sigma direction (along bond)
    r1 = rot[1]  # pi_x direction
    r2 = rot[2]  # pi_y direction

    # Assemble di (overlap in molecular frame)
    di = np.zeros((nA, nB))

    # s-s: always present
    di[0, 0] = S111

    if jcall == 3:
        # Heavy-H: di[p_A, s_B] = S211 * c0
        if nA > 1:
            for k in range(3):
                di[k + 1, 0] = S211 * r0[k]

    elif jcall == 4:
        if nA > 1 and nB > 1:
            # p_A - s_B: di[p, s] = S211 * c0
            for k in range(3):
                di[k + 1, 0] = S211 * r0[k]

            # s_A - p_B: di[s, p] = -S121 * c0  (PYSEQM minus sign!)
            for k in range(3):
                di[0, k + 1] = -S121 * r0[k]

            # p_A - p_B: di[p,p] = -S221*r0⊗r0 + S222*(r1⊗r1 + r2⊗r2)
            # PYSEQM: M = [-S221, S222, S222]
            #          B = rot_T * M
            #          di[1:,1:] = B @ rot
            for k in range(3):
                for l in range(3):
                    di[k + 1, l + 1] = (
                        -S221 * r0[k] * r0[l]
                        + S222 * (r1[k] * r1[l] + r2[k] * r2[l])
                    )

    return di
