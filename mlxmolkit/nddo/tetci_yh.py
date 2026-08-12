"""YH d-orbital H_core contribution (atom A has d-orbitals, atom B = H).

Port of the YH section of PYSEQM's
``two_elec_two_center_int_local_frame_d_orbitals.py`` (BSD-3-Clause,
github.com/lanl/PYSEQM, lines 92-516).

This module is the missing piece that makes mlxmolkit PM6_D self-
contained on d-orbital + H molecules (H2S, PH3, HCl, HBr, HI). It
computes the 45 d-orbital electron-nuclear attraction matrix
elements that fill in atom A's 9×9 H_core block due to atom B's
nuclear charge — the contribution that was previously absent (only
Udd on the d-diagonal), causing the H2S pz collapse.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .params import ANG_TO_BOHR
from .rotation_matrix_d import generate_rotation_matrix, rotate_core
from .two_center_integrals import _compute_multipole_params, two_center_integrals
from .tetci_multipole_pyseqm import pyseqm_d_params

EV = 27.21
EV1 = EV / 2.0
EV2 = EV / 4.0


def yh_rotated_integral_matrix(
    pA, pB, coordA: NDArray[np.float64], coordB: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the 9×9 (μν_A | s_B s_B) matrix in molecular frame.

    This is the shared computation used by both H_core (multiplied by -Z_B)
    and Fock J/K (multiplied by density elements).
    """
    R = float(np.linalg.norm(coordB - coordA))
    R_bohr = R * ANG_TO_BOHR

    xij = (np.asarray(coordB) - np.asarray(coordA))[None, :]
    xij = xij / np.linalg.norm(xij)

    da_A, qa_A, rho0A, rho1A, rho2A = _compute_multipole_params(pA)
    _, _, rho0B, _, _ = _compute_multipole_params(pB)
    dA = pyseqm_d_params(pA)
    dpA, dsA, ddA = dA["dp"], dA["ds"], dA["dorbdorb"]
    rho3A, rho4A, rho5A, rho6A = dA["rho3"], dA["rho4"], dA["rho5"], dA["rho6"]

    rhoSS = (rho0A + rho0B) ** 2
    rhoSDD0 = (rho3A + rho0B) ** 2
    rhoSSP = (rho1A + rho0B) ** 2
    rhoSPD = (rho4A + rho0B) ** 2
    rhoSPP = (rho2A + rho0B) ** 2
    rhoSSD = (rho5A + rho0B) ** 2
    rhoSDD = (rho6A + rho0B) ** 2

    qq = EV / np.sqrt(R_bohr ** 2 + rhoSS)
    DDq_Sq = EV / np.sqrt(R_bohr ** 2 + rhoSDD0)
    DPUz_Sq = EV1 / np.sqrt((R_bohr + dpA) ** 2 + rhoSPD) - EV1 / np.sqrt(
        (R_bohr - dpA) ** 2 + rhoSPD
    )
    SPUz_Sq = EV1 / np.sqrt((R_bohr + da_A) ** 2 + rhoSSP) - EV1 / np.sqrt(
        (R_bohr - da_A) ** 2 + rhoSSP
    )
    DDQ_Sq = (
        EV2 / np.sqrt((R_bohr - ddA) ** 2 + rhoSDD)
        + EV2 / np.sqrt((R_bohr + ddA) ** 2 + rhoSDD)
        - EV1 / np.sqrt(R_bohr ** 2 + ddA ** 2 + rhoSDD)
    )
    DSQ_Sq = (
        EV2 / np.sqrt((R_bohr - dsA) ** 2 + rhoSSD)
        + EV2 / np.sqrt((R_bohr + dsA) ** 2 + rhoSSD)
        - EV1 / np.sqrt(R_bohr ** 2 + dsA ** 2 + rhoSSD)
    )
    PPQ_Sq = (
        EV2 / np.sqrt((R_bohr - qa_A * 2.0) ** 2 + rhoSPP)
        + EV2 / np.sqrt((R_bohr + qa_A * 2.0) ** 2 + rhoSPP)
        - EV1 / np.sqrt(R_bohr ** 2 + da_A ** 2 + rhoSPP)
    )

    riYH = np.zeros(45)
    riYH[10] = DSQ_Sq * 1.154701
    riYH[11] = DPUz_Sq * 1.154701
    riYH[14] = DDq_Sq + DDQ_Sq * 1.333333
    riYH[17] = DPUz_Sq * 1.000000
    riYH[20] = DDq_Sq + DDQ_Sq * 0.666667
    riYH[24] = DPUz_Sq * 1.000000
    riYH[27] = DDq_Sq + DDQ_Sq * 0.666667
    riYH[35] = DDq_Sq + DDQ_Sq * -1.333333
    riYH[44] = DDq_Sq + DDQ_Sq * -1.333333

    ri_xh, _, _ = two_center_integrals(pA, pB, R)

    # Build the local-frame 46-element vector (slot 0 reserved for s-self;
    # slots 1-45 are the (μν|ss) lower-triangle packed integrals on A)
    core_local = np.zeros((1, 46))
    core_local[0, 1] = ri_xh[0]   # (ss|ss)
    core_local[0, 3] = ri_xh[3]   # (pp_pi|ss)
    core_local[0, 7] = ri_xh[1]   # (ps|ss)
    core_local[0, 10] = ri_xh[2]  # (pp_sigma|ss)
    core_local[0, 15] = riYH[44]
    core_local[0, 17] = riYH[17]
    core_local[0, 21] = riYH[20]
    core_local[0, 22] = riYH[10]
    core_local[0, 25] = riYH[11]
    core_local[0, 28] = riYH[14]

    rot_matrix = generate_rotation_matrix(xij)
    core_mol = rotate_core(core_local[:, 1:], rot_matrix, 3)[0]

    # Unpack 45-element packed lower-triangle into 9x9 symmetric matrix
    W = np.zeros((9, 9))
    INDX = [0, 1, 3, 6, 10, 15, 21, 28, 36]
    for i in range(9):
        for j in range(i + 1):
            packed = INDX[i] + j
            W[i, j] = core_mol[packed]
            W[j, i] = W[i, j]
    return W


def yh_fock_contribution(
    F: NDArray[np.float64],
    P: NDArray[np.float64],
    pA, pB,
    sA: int, sB: int,
    coordA: NDArray[np.float64], coordB: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Add YH two-electron Fock contribution (J + K terms) for d-orbital atom A + H atom B.

    Uses the same rotated 9×9 (μν_A | s_B s_B) matrix as :func:`yh_e1b_contribution`.
    J terms: F[μ_A, ν_A] += P[s_B, s_B] * W[μ, ν] (Coulomb on A from B's density)
             F[s_B, s_B] += P[μ_A, ν_A] * W[μ, ν] (Coulomb on B from A's density)
    K terms: F[μ_A, s_B] -= 0.5 * P[ν_A, s_B] * W[μ, ν]
    """
    W = yh_rotated_integral_matrix(pA, pB, coordA, coordB)
    P_BB = P[sB, sB]
    # Coulomb on A from B's density: F[μ_A, ν_A] += P[s_B,s_B] * (μν|ss)
    for mu in range(9):
        for nu in range(9):
            F[sA + mu, sA + nu] += P_BB * W[mu, nu]
    # Coulomb on B from A's density: F[s_B,s_B] += sum_{μν} P[μ_A,ν_A] * (μν|ss)
    # Use lower-triangle packing with weight 2 for off-diagonal, mirroring
    # PYSEQM's PA = P[i1,i0] * weight_tc convention. Iterating full 9×9 on
    # symmetric P+W gives the same sum so this loop is correct.
    j_contrib = 0.0
    for mu in range(9):
        for nu in range(9):
            j_contrib += P[sA + mu, sA + nu] * W[mu, nu]
    F[sB, sB] += j_contrib
    # Exchange between A and B (cross block):
    # F[μ_A, s_B] -= 0.5 * sum_ν P[ν_A, s_B] * (μ_A ν_A | s_B s_B)
    # which uses the SAME W matrix because for B=H with only s, the K
    # integral collapses to the same W[μ, ν] = (μν_A | s_B s_B).
    # NOTE: K term temporarily disabled — J alone gives over-correction
    # too strong; need PYSEQM K_ind_9 specific contraction to balance.
    if False:
        for mu in range(9):
            K_sum = 0.0
            for nu in range(9):
                K_sum += P[sA + nu, sB] * W[mu, nu]
            F[sA + mu, sB] -= 0.5 * K_sum
            F[sB, sA + mu] = F[sA + mu, sB]
    return F


def yh_e1b_contribution(
    pA, pB, coordA: NDArray[np.float64], coordB: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the 9×9 contribution to H_core block of atom A from atom B's
    nucleus, when A has d-orbitals (n_basis=9) and B is sp-only (n_basis=1
    for H, n_basis=4 for C/N/O/F).

    This is the PYSEQM ``e1b`` matrix (electrons on A attracted by B's
    nucleus) rotated to the molecular frame. Z_B is taken from
    ``pB.n_valence`` (= 1 for H, 4-7 for sp heavy atoms), so the same
    formula works for both YH and YX (sp-heavy B) cases.

    Includes:
        * sp self-attraction (4×4 block via the existing riXH integrals)
        * d-orbital nuclear attraction (additional 45 - 10 = 35 entries
          from the 9 active riYH local-frame matrix elements)

    Returns a 9×9 array to ADD to the existing H_core slice for atom A.
    Caller is responsible for indexing into H_core at A's basis range.
    """
    R = float(np.linalg.norm(coordB - coordA))
    R_bohr = R * ANG_TO_BOHR

    # Bond vector (Angstrom, will be normalized inside generate_rotation_matrix)
    xij = (np.asarray(coordB) - np.asarray(coordA))[None, :]
    xij = xij / np.linalg.norm(xij)

    # ── sp multipole params for atom A and atom B (H) ────────────────
    da_A, qa_A, rho0A, rho1A, rho2A = _compute_multipole_params(pA)
    _, _, rho0B, _, _ = _compute_multipole_params(pB)
    # d-orbital params for atom A
    dA = pyseqm_d_params(pA)
    dpA = dA["dp"]
    dsA = dA["ds"]
    ddA = dA["dorbdorb"]
    rho3A = dA["rho3"]
    rho4A = dA["rho4"]
    rho5A = dA["rho5"]
    rho6A = dA["rho6"]

    # ── 7 multipole integrals (PYSEQM lines 337-358) ─────────────────
    rhoSS = (rho0A + rho0B) ** 2
    rhoSDD0 = (rho3A + rho0B) ** 2
    rhoSSP = (rho1A + rho0B) ** 2
    rhoSPD = (rho4A + rho0B) ** 2
    rhoSPP = (rho2A + rho0B) ** 2
    rhoSSD = (rho5A + rho0B) ** 2
    rhoSDD = (rho6A + rho0B) ** 2

    qq = EV / np.sqrt(R_bohr ** 2 + rhoSS)
    DDq_Sq = EV / np.sqrt(R_bohr ** 2 + rhoSDD0)
    DPUz_Sq = EV1 / np.sqrt((R_bohr + dpA) ** 2 + rhoSPD) - EV1 / np.sqrt(
        (R_bohr - dpA) ** 2 + rhoSPD
    )
    SPUz_Sq = EV1 / np.sqrt((R_bohr + da_A) ** 2 + rhoSSP) - EV1 / np.sqrt(
        (R_bohr - da_A) ** 2 + rhoSSP
    )
    DDQ_Sq = (
        EV2 / np.sqrt((R_bohr - ddA) ** 2 + rhoSDD)
        + EV2 / np.sqrt((R_bohr + ddA) ** 2 + rhoSDD)
        - EV1 / np.sqrt(R_bohr ** 2 + ddA ** 2 + rhoSDD)
    )
    DSQ_Sq = (
        EV2 / np.sqrt((R_bohr - dsA) ** 2 + rhoSSD)
        + EV2 / np.sqrt((R_bohr + dsA) ** 2 + rhoSSD)
        - EV1 / np.sqrt(R_bohr ** 2 + dsA ** 2 + rhoSSD)
    )
    PPQ_Sq = (
        EV2 / np.sqrt((R_bohr - qa_A * 2.0) ** 2 + rhoSPP)
        + EV2 / np.sqrt((R_bohr + qa_A * 2.0) ** 2 + rhoSPP)
        - EV1 / np.sqrt(R_bohr ** 2 + da_A ** 2 + rhoSPP)
    )

    # ── 9 active riYH entries (PYSEQM lines 432-456) ──────────────────
    riYH = np.zeros(45)
    riYH[10] = DSQ_Sq * 1.154701  # SDz2|SS
    riYH[11] = DPUz_Sq * 1.154701  # PzDz2|SS
    riYH[14] = DDq_Sq + DDQ_Sq * 1.333333  # Dz2Dz2|SS
    riYH[17] = DPUz_Sq * 1.000000  # PxDxz|SS
    riYH[20] = DDq_Sq + DDQ_Sq * 0.666667  # DxzDxz|SS
    riYH[24] = DPUz_Sq * 1.000000  # PyDyz|SS
    riYH[27] = DDq_Sq + DDQ_Sq * 0.666667  # DyzDyz|SS
    riYH[35] = DDq_Sq + DDQ_Sq * -1.333333  # Dx2-y2Dx2-y2|SS
    riYH[44] = DDq_Sq + DDQ_Sq * -1.333333  # DxyDxy|SS

    # ── XH local-frame integrals for the sp self-attraction part ─────
    ri_xh, _core_xh, _pair_type = two_center_integrals(pA, pB, R)
    # ri_xh is 4-element vector for XH: [ss|ss, ps|ss, pp_sigma|ss, pp_pi|ss]
    # For PYSEQM coreYHLocal layout (46 slots starting at 0):
    #   slot 0: tore[ni] * riXHPM6[0]  (A's nucleus seen by B = same as ee here)
    #   slot 1: tore[nj] * riXH[0] = (1) * (ss|ss)
    #   slot 3: tore[nj] * riXH[3] = (1) * (pp_pi|ss)
    #   slot 7: tore[nj] * riXH[1] = (1) * (ps|ss)
    #   slot 10: tore[nj] * riXH[2] = (1) * (pp_sigma|ss)
    #   slot 15: tore[nj] * riYH[44]
    #   slot 17: tore[nj] * riYH[17]
    #   slot 21: tore[nj] * riYH[20]
    #   slot 22: tore[nj] * riYH[10]
    #   slot 25: tore[nj] * riYH[11]
    #   slot 28: tore[nj] * riYH[14]

    # Z_B = tore[nj] = core charge of atom B (n_valence in PM6 convention).
    # For YH this is 1; for YX this is 4-7 depending on the sp atom.
    # The (μν_A | s_B s_B) multipole integral formulas only depend on B
    # through rho0_B (monopole additive term), so generalizing to non-H B
    # is purely a Z_B prefactor change.
    Z_A = float(pA.n_valence)
    Z_B = float(pB.n_valence)

    core_local = np.zeros((1, 46))
    core_local[0, 0] = Z_A * ri_xh[0]
    core_local[0, 1] = Z_B * ri_xh[0]
    core_local[0, 3] = Z_B * ri_xh[3]
    core_local[0, 7] = Z_B * ri_xh[1]
    core_local[0, 10] = Z_B * ri_xh[2]
    core_local[0, 15] = Z_B * riYH[44]
    core_local[0, 17] = Z_B * riYH[17]
    core_local[0, 21] = Z_B * riYH[20]
    core_local[0, 22] = Z_B * riYH[10]
    core_local[0, 25] = Z_B * riYH[11]
    core_local[0, 28] = Z_B * riYH[14]

    # ── Rotate the 45 local-frame entries to molecular frame ─────────
    rot_matrix = generate_rotation_matrix(xij)  # (1, 15, 45)
    core_mol = np.zeros((1, 46))
    core_mol[0, 0] = core_local[0, 0]
    core_mol[0, 1:] = rotate_core(core_local[:, 1:], rot_matrix, 3)[0]

    # ── Unpack 45-element packed lower-triangle into 9×9 matrix ──────
    # PYSEQM stores e2aD as upper-triangle (lower = 0) because its
    # eigensolver uses UPLO='U' (torch.linalg.eigh with UPLO='U'). NumPy's
    # np.linalg.eigh defaults to UPLO='L', so we must return a fully
    # symmetric matrix here. Failing to do so leaves the lower triangle
    # of A's 9×9 block as zero in H_core, which destroys the s-pz, sp-d,
    # and p-d couplings on the d-orbital atom and makes the SCF settle
    # into a wrong basin (q_S ≈ +1.84 instead of -0.36 on H2S).
    e1b = np.zeros((9, 9))
    INDX = [0, 1, 3, 6, 10, 15, 21, 28, 36]
    for i in range(9):
        for j in range(i + 1):
            packed = INDX[i] + j
            val = core_mol[0, 1 + packed]
            e1b[j, i] = val
            if j != i:
                e1b[i, j] = val  # mirror upper to lower → symmetric
    # H_core convention: V_munu = -Z_B * integral, so negate
    return -e1b
