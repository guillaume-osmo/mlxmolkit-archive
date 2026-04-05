"""
Slater-type orbital overlap integrals for NDDO methods.

Computes the diatomic overlap matrix S between basis functions on
atoms A and B, following MOPAC's diat.f / PYSEQM's diat_overlap.py.

The overlap in the local frame (bond axis = x) is computed using
auxiliary A and B integrals, then rotated to the molecular frame.

Reference: diat.f in MOPAC, diat_overlap.py in PYSEQM.
"""
from __future__ import annotations

import numpy as np
from .params import ElementParams, ANG_TO_BOHR
from .rotation import _rotation_matrix


def _aintgs(alpha: float, n_max: int = 5) -> np.ndarray:
    """A-integrals: a_k(alpha) = exp(-alpha)/alpha * recurrence.

    a_1 = exp(-alpha)/alpha
    a_k = a_1 + (k-1)*a_{k-1}/alpha
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

    b_1 = 2*sinh(beta)/beta
    b_k = (-1)^(k+1) * 2*cosh_or_sinh/beta + (k-1)*b_{k-1}/beta

    Uses exact recurrence for |beta| > 1e-6, Taylor for smaller.
    Matches PYSEQM's bintgs() implementation.
    """
    b = np.zeros(n_max)

    if abs(beta) < 1e-6:
        # Exact at beta=0: b_1=2, b_2=0, b_3=2/3, b_4=0, b_5=2/5
        b[0] = 2.0
        b[1] = 0.0
        b[2] = 2.0 / 3.0
        b[3] = 0.0
        b[4] = 2.0 / 5.0
        return b

    tx = np.exp(beta) / beta
    tmx = -np.exp(-beta) / beta
    b[0] = tx + tmx                         # 2*cosh(beta)/beta... wait
    # Actually: tx = exp(x)/x, tmx = -exp(-x)/x
    # b1 = tx + tmx = (exp(x) - exp(-x))/x = 2*sinh(x)/x
    b[1] = -tx + tmx + b[0] / beta          # b2 = -exp(x)/x - exp(-x)/x + b1/x
    b[2] = tx + tmx + 2.0 * b[1] / beta
    b[3] = -tx + tmx + 3.0 * b[2] / beta
    b[4] = tx + tmx + 4.0 * b[3] / beta
    return b


def overlap_local_frame(
    pA: ElementParams, pB: ElementParams, R_bohr: float,
) -> np.ndarray:
    """Compute overlap matrix in the local frame (bond axis = x).

    Returns S_local: (nA, nB) overlap matrix.
    For sp basis: up to 4×4 with elements:
      S[s,s], S[s,σ], S[σ,s], S[σ,σ], S[π,π]
    where σ = p_sigma (along bond), π = p_pi (perpendicular).
    """
    nA = pA.n_basis
    nB = pB.n_basis
    S = np.zeros((nA, nB))

    if R_bohr < 1e-10:
        np.fill_diagonal(S, 1.0)
        return S

    qnA = 1 if pA.Z <= 2 else 2
    qnB = 1 if pB.Z <= 2 else 2

    # s-s overlap
    # MOPAC/PYSEQM convention: z1 = zeta_B (lighter/second), z2 = zeta_A (heavier/first)
    # alpha = 0.5*R*(z1+z2), beta = 0.5*R*(z1-z2)
    # This means beta = 0.5*R*(zeta_B - zeta_A) — OPPOSITE sign from natural order
    alpha_ss = 0.5 * R_bohr * (pB.zeta_s + pA.zeta_s)  # same either way
    beta_ss = 0.5 * R_bohr * (pB.zeta_s - pA.zeta_s)   # PYSEQM: z1-z2 = zetaB-zetaA
    A_ss = _aintgs(alpha_ss)
    B_ss = _bintgs(beta_ss)

    if qnA == 1 and qnB == 1:
        # (1s|1s): PYSEQM jcall==2
        S[0, 0] = ((pA.zeta_s * pB.zeta_s * R_bohr ** 2) ** 1.5
                  * (A_ss[2] * B_ss[0] - B_ss[2] * A_ss[0]) / 4.0)

    elif qnA == 2 and qnB == 1:
        # (2s|1s): PYSEQM jcall==3, divisor = sqrt(3)*8
        S[0, 0] = (pB.zeta_s ** 1.5 * pA.zeta_s ** 2.5 * R_bohr ** 4
                  * (A_ss[3] * B_ss[0] - B_ss[3] * A_ss[0]
                   + A_ss[2] * B_ss[1] - B_ss[2] * A_ss[1])
                  / (np.sqrt(3.0) * 8.0))

        # (2pσ|1s): PYSEQM S211 jcall==3, divisor = 8
        if nA > 1:
            alpha_ps = 0.5 * R_bohr * (pB.zeta_s + pA.zeta_p)
            beta_ps = 0.5 * R_bohr * (pB.zeta_s - pA.zeta_p)
            A_ps = _aintgs(alpha_ps)
            B_ps = _bintgs(beta_ps)

            S[1, 0] = (pB.zeta_s ** 1.5 * pA.zeta_p ** 2.5 * R_bohr ** 4
                      * (A_ps[2] * B_ps[0] - B_ps[2] * A_ps[0]
                       + A_ps[3] * B_ps[1] - B_ps[3] * A_ps[1])
                      / 8.0)

    elif qnA == 1 and qnB == 2:
        # (1s|2s) — swap atoms
        S_swap = overlap_local_frame(pB, pA, R_bohr)
        S[0, 0] = S_swap[0, 0]
        if nB > 1:
            S[0, 1] = -S_swap[1, 0]  # sign flip for pσ direction

    elif qnA == 2 and qnB == 2:
        # (2s|2s): PYSEQM jcall==4, divisor = 48
        S[0, 0] = ((pA.zeta_s * pB.zeta_s) ** 2.5 * R_bohr ** 5
                  * (A_ss[4] * B_ss[0] + B_ss[4] * A_ss[0]
                   - 2.0 * A_ss[2] * B_ss[2])
                  / 48.0)

        if nA > 1 and nB > 1:
            # (2pσ|2s): PYSEQM S211 jcall==4, divisor = 16*sqrt(3)
            alpha_ps = 0.5 * R_bohr * (pB.zeta_s + pA.zeta_p)
            beta_ps = 0.5 * R_bohr * (pB.zeta_s - pA.zeta_p)
            A_ps = _aintgs(alpha_ps)
            B_ps = _bintgs(beta_ps)

            S[1, 0] = ((pA.zeta_p * pB.zeta_s) ** 2.5 * R_bohr ** 5
                      * (A_ps[3] * (B_ps[0] - B_ps[2])
                       - A_ps[1] * (B_ps[2] - B_ps[4])
                       + B_ps[3] * (A_ps[0] - A_ps[2])
                       - B_ps[1] * (A_ps[2] - A_ps[4]))
                      / (16.0 * np.sqrt(3.0)))

            # (2s|2pσ): PYSEQM S121 jcall==4
            alpha_sp = 0.5 * R_bohr * (pB.zeta_p + pA.zeta_s)
            beta_sp = 0.5 * R_bohr * (pB.zeta_p - pA.zeta_s)
            A_sp = _aintgs(alpha_sp)
            B_sp = _bintgs(beta_sp)

            S[0, 1] = -((pA.zeta_s * pB.zeta_p) ** 2.5 * R_bohr ** 5
                       * (A_sp[3] * (B_sp[0] - B_sp[2])
                        - A_sp[1] * (B_sp[2] - B_sp[4])
                        + B_sp[3] * (A_sp[0] - A_sp[2])
                        - B_sp[1] * (A_sp[2] - A_sp[4]))
                       / (16.0 * np.sqrt(3.0)))

            # (2pσ|2pσ): PYSEQM S221 jcall==4, divisor = 48
            alpha_pp = 0.5 * R_bohr * (pB.zeta_p + pA.zeta_p)
            beta_pp = 0.5 * R_bohr * (pB.zeta_p - pA.zeta_p)
            A_pp = _aintgs(alpha_pp)
            B_pp = _bintgs(beta_pp)

            S[1, 1] = ((pA.zeta_p * pB.zeta_p) ** 2.5 * R_bohr ** 5
                      * (A_pp[4] * B_pp[0] + B_pp[4] * A_pp[0]
                       - 2.0 * A_pp[2] * B_pp[2])
                      / 48.0)

            # (2pπ|2pπ): PYSEQM S222 jcall==4
            # S_pi = -(zeta_p^5 * R^5 * (A4*B0 - B4*A0 - 2*A2*B2 + ...) / 48
            # For pi: different recurrence
            S[2, 2] = -((pA.zeta_p * pB.zeta_p) ** 2.5 * R_bohr ** 5
                       * (A_pp[4] * B_pp[0] - B_pp[4] * A_pp[0])
                       / 48.0)
            S[3, 3] = S[2, 2]

    return S


def overlap_molecular_frame(
    pA: ElementParams, pB: ElementParams,
    coordA: np.ndarray, coordB: np.ndarray,
) -> np.ndarray:
    """Compute overlap matrix in the molecular frame.

    Rotates the local-frame overlap to the molecular frame using
    the same quaternion rotation as the two-electron integrals.

    Returns S: (nA, nB) overlap matrix.
    """
    R_vec = coordB - coordA
    R = np.linalg.norm(R_vec)
    R_bohr = R * ANG_TO_BOHR
    nA = pA.n_basis
    nB = pB.n_basis

    if R < 1e-10:
        S = np.zeros((nA, nB))
        for k in range(min(nA, nB)):
            S[k, k] = 1.0
        return S

    # Local frame overlap
    S_local = overlap_local_frame(pA, pB, R_bohr)

    # Rotation matrix (same as integrals)
    v = -(coordB - coordA) / R
    rot = _rotation_matrix(v)
    r0 = rot[0]  # sigma (along bond)
    r1 = rot[1]  # pi_x
    r2 = rot[2]  # pi_y

    # Rotate: S_mol[μ,ν] = Σ R_μa R_νb S_local[a,b]
    S = np.zeros((nA, nB))

    # s-s
    S[0, 0] = S_local[0, 0]

    if nA > 1 and nB == 1:
        # p_A - s_B
        for k in range(3):
            S[k + 1, 0] = S_local[1, 0] * r0[k]  # σ component

    elif nA == 1 and nB > 1:
        # s_A - p_B
        for k in range(3):
            S[0, k + 1] = S_local[0, 1] * r0[k]

    elif nA > 1 and nB > 1:
        # s-p and p-s
        for k in range(3):
            S[k + 1, 0] = S_local[1, 0] * r0[k]
            S[0, k + 1] = S_local[0, 1] * r0[k]

        # p-p: σ-σ + π-π components
        for k in range(3):
            for l in range(3):
                S[k + 1, l + 1] = (S_local[1, 1] * r0[k] * r0[l]
                                 + S_local[2, 2] * (r1[k] * r1[l] + r2[k] * r2[l]))

    return S
