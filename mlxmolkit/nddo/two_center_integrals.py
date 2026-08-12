"""
Full 22-integral NDDO two-center two-electron repulsion integrals.

Implements the Klopman-Ohno-Dewar multipole approximation for sp-basis
following MOPAC/PYSEQM conventions. The 22 integrals are:

  (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5,
  (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10,
  (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15,
  (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20,
  (PP/P*P*)=21, (P*P/P*P)=22

where S=s, O=p_sigma, P=p_pi.

Reference: repp.f in MOPAC, two_elec_two_center_int_local_frame.py in PYSEQM.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np
from .params import RM1_PARAMS, ElementParams, ANG_TO_BOHR, principal_qn

EV = 27.21  # Hartree to eV (MOPAC convention)


def _compute_multipole_params(p: ElementParams) -> tuple[float, float, float, float, float]:
    """Charge separations and additive terms for one atom.

    Exact port of PYSEQM's cal_par.py (dd_qq, additive_term_rho1, rho2).

    A pure function of the element's own parameters — no geometry enters — yet
    it runs two five-step secant solvers, and every two-centre integral asked
    for it afresh. A single benzaldehyde gradient called it 14528 times for
    three distinct elements. It is memoised on the parameters it actually
    reads, so the answer is identical and computed once per element per method.

    Returns: da, qa, rho0, rho1, rho2
    """
    return _multipole_params_cached(
        p.Z, p.n_basis, p.gss, p.zeta_s, p.zeta_p, p.hsp, p.gpp, p.gp2)


@lru_cache(maxsize=None)
def _multipole_params_cached(Z, n_basis, gss, zeta_s, zeta_p, hsp, gpp, gp2):
    p = _MultipoleInputs(Z, n_basis, gss, zeta_s, zeta_p, hsp, gpp, gp2)
    return _compute_multipole_params_uncached(p)


class _MultipoleInputs(NamedTuple):
    """Exactly the fields :func:`_compute_multipole_params_uncached` reads.

    Naming them here is what makes the cache key provably complete: if the
    computation ever starts reading another parameter, it fails loudly on this
    object instead of silently returning a stale value.
    """
    Z: int
    n_basis: int
    gss: float
    zeta_s: float
    zeta_p: float
    hsp: float
    gpp: float
    gp2: float


def _compute_multipole_params_uncached(p) -> tuple[float, float, float, float, float]:
    """The actual computation. See :func:`_compute_multipole_params`."""
    # rho0: monopole additive term = 0.5*ev/gss (PYSEQM line 240)
    rho0 = 0.5 * EV / p.gss if p.gss > 1e-10 else 0.0

    if p.n_basis == 1:
        return 0.0, 0.0, rho0, 0.0, 0.0

    # PYSEQM uses periodic-row qn: 2 for C-F, 3 for P-Cl, 4 for Br, 5 for I.
    # Earlier this was hardcoded to 2 for all Z>2 → broke S/Cl/Br/I multipoles.
    qn = principal_qn(p.Z)
    zs = p.zeta_s
    zp = p.zeta_p

    # ================================================================
    # Charge separations: exact PYSEQM dd_qq formula (cal_par.py)
    # dd = (2*qn+1) * (4*zs*zp)^(qn+0.5) / (zs+zp)^(2*qn+2) / sqrt(3)
    # qq = sqrt((4*qn^2 + 6*qn + 2) / 20) / zp
    # ================================================================
    da = ((2.0 * qn + 1.0)
          * (4.0 * zs * zp) ** (qn + 0.5)
          / (zs + zp) ** (2.0 * qn + 2.0)
          / np.sqrt(3.0))
    qa = np.sqrt((4.0 * qn ** 2 + 6.0 * qn + 2.0) / 20.0) / zp

    # ================================================================
    # rho1: dipole additive term — Newton solver (PYSEQM additive_term_rho1)
    # Solves: hsp = ev * (d/2 - 1/2/sqrt(4*D1^2 + 1/d^2))  for d
    # Then rho1 = 0.5/d
    # ================================================================
    if p.hsp > 0:
        hsp_au = p.hsp / EV  # convert to atomic units
        D1 = da
        # Initial guess
        d1 = abs(hsp_au / D1 ** 2) ** (1.0 / 3.0)
        if hsp_au < 0:
            d1 = -d1
        d2 = d1 + 0.04
        for _ in range(5):
            hsp1 = 0.5 * d1 - 0.5 / np.sqrt(4.0 * D1 ** 2 + 1.0 / d1 ** 2)
            hsp2 = 0.5 * d2 - 0.5 / np.sqrt(4.0 * D1 ** 2 + 1.0 / d2 ** 2)
            if abs(hsp2 - hsp1) > 1e-16:
                d3 = d1 + (d2 - d1) * (hsp_au - hsp1) / (hsp2 - hsp1)
            else:
                d3 = d2
            d1, d2 = d2, d3
        rho1 = 0.5 / d2
    else:
        rho1 = 0.0

    # ================================================================
    # rho2: quadrupole additive term — Newton/secant solver
    # (PYSEQM additive_term_rho2)
    # hpp = 0.5*(gpp-gp2), clamped to min 0.1
    # Solves: hpp = ev * (q/4 - 1/2/sqrt(4*D2^2+1/q^2)
    #                     + 1/4/sqrt(8*D2^2+1/q^2))
    # rho2 = 0.5/q
    # ================================================================
    hpp = 0.5 * (p.gpp - p.gp2)
    hpp = max(hpp, 0.1)  # clamp_min(0.1) from PYSEQM
    hpp_au = hpp / EV
    D2 = qa
    # Initial guess
    q1 = abs(hpp_au / 3.0 / D2 ** 4) ** 0.2
    if hpp_au < 0:
        q1 = -q1
    q2 = q1 + 0.04
    for _ in range(5):
        hpp1 = (0.25 * q1
                - 0.5 / np.sqrt(4.0 * D2 ** 2 + 1.0 / q1 ** 2)
                + 0.25 / np.sqrt(8.0 * D2 ** 2 + 1.0 / q1 ** 2))
        hpp2 = (0.25 * q2
                - 0.5 / np.sqrt(4.0 * D2 ** 2 + 1.0 / q2 ** 2)
                + 0.25 / np.sqrt(8.0 * D2 ** 2 + 1.0 / q2 ** 2))
        if abs(hpp2 - hpp1) > 1e-16:
            q3 = q1 + (q2 - q1) * (hpp_au - hpp1) / (hpp2 - hpp1)
        else:
            q3 = q2
        q1, q2 = q2, q3
    rho2 = 0.5 / q2

    return da, qa, rho0, rho1, rho2


def two_center_integrals(pA: ElementParams, pB: ElementParams, R_ang: float):
    """Compute all two-center integrals for atom pair A-B.

    Args:
        pA, pB: element parameters
        R_ang: interatomic distance in Angstrom

    Returns:
        ri: array of two-electron integrals (eV)
            H-H: (1,), X-H: (4,), X-X: (22,)
        core: nuclear attraction integrals (eV)
            H-H: (2,), X-H: (5,), X-X: (8,)
        pair_type: 'HH', 'XH', 'HX', or 'XX'
    """
    R = R_ang * ANG_TO_BOHR
    daA, qaA, rho0A, rho1A, rho2A = _compute_multipole_params(pA)
    daB, qaB, rho0B, rho1B, rho2B = _compute_multipole_params(pB)

    ev1 = EV / 2.0
    ev2 = EV / 4.0
    ev3 = EV / 8.0
    ev4 = EV / 16.0

    nA = pA.n_basis
    nB = pB.n_basis
    ZA = float(pA.n_valence)
    ZB = float(pB.n_valence)

    if nA == 1 and nB == 1:
        return _ri_core_hh(rho0A, rho0B, R, ZA, ZB) + ('HH',)

    # Handle X-H case (ensure heavy atom is first)
    if nA == 1 and nB > 1:
        ri, core, _ = two_center_integrals(pB, pA, R_ang)
        # Swap core: core[0:4] was for B→A, core[4] was A→B
        # For HX: we need to return in swapped order
        return ri, core, 'HX'

    if nA > 1 and nB == 1:
        return _ri_core_xh(daA, qaA, rho0A, rho0B, rho1A, rho2A,
                           R, ZA, ZB) + ('XH',)

    return _ri_core_xx(daA, daB, qaA, qaB, rho0A, rho0B,
                       rho1A, rho1B, rho2A, rho2B, R, ZA, ZB) + ('XX',)


def _ri_core_xx(daA, daB, qaA, qaB, rho0A, rho0B, rho1A, rho1B, rho2A, rho2B,
                R, ZA, ZB):
    """The 22 heavy-heavy integrals and their core terms.

    Every operation is elementwise, so this runs unchanged on scalars or on
    arrays over a pair axis — ``ri[..., k]`` and ``np.stack(axis=-1)`` behave
    the same for a bare (22,) vector and a batched (P, 22). Extracted from
    :func:`two_center_integrals` so the scalar and batched entry points share
    one implementation rather than two transcriptions that can drift.
    """
    ev1, ev2, ev3, ev4 = EV / 2.0, EV / 4.0, EV / 8.0, EV / 16.0
    da, db = daA, daB
    qa, qb = qaA * 2.0, qaB * 2.0
    qa1, qb1 = qaA, qaB

    aee = (rho0A + rho0B) ** 2
    ade = (rho1A + rho0B) ** 2
    aqe = (rho2A + rho0B) ** 2
    aed = (rho0A + rho1B) ** 2
    aeq = (rho0A + rho2B) ** 2
    axx = (rho1A + rho1B) ** 2
    adq = (rho1A + rho2B) ** 2
    aqd = (rho2A + rho1B) ** 2
    aqq = (rho2A + rho2B) ** 2

    ee = EV / np.sqrt(R ** 2 + aee)
    dze = -ev1 / np.sqrt((R + da) ** 2 + ade) + ev1 / np.sqrt((R - da) ** 2 + ade)
    ev1dsqr6 = ev1 / np.sqrt(R ** 2 + aqe)
    qzze = ev2 / np.sqrt((R - qa) ** 2 + aqe) + ev2 / np.sqrt((R + qa) ** 2 + aqe) - ev1dsqr6
    qxxe = ev1 / np.sqrt(R ** 2 + qa ** 2 + aqe) - ev1dsqr6
    edz = -ev1 / np.sqrt((R - db) ** 2 + aed) + ev1 / np.sqrt((R + db) ** 2 + aed)
    ev1dsqr12 = ev1 / np.sqrt(R ** 2 + aeq)
    eqzz = ev2 / np.sqrt((R - qb) ** 2 + aeq) + ev2 / np.sqrt((R + qb) ** 2 + aeq) - ev1dsqr12
    eqxx = ev1 / np.sqrt(R ** 2 + qb ** 2 + aeq) - ev1dsqr12

    ev2dsqr20 = ev2 / np.sqrt((R + da) ** 2 + adq)
    ev2dsqr22 = ev2 / np.sqrt((R - da) ** 2 + adq)
    ev2dsqr24 = ev2 / np.sqrt((R - db) ** 2 + aqd)
    ev2dsqr26 = ev2 / np.sqrt((R + db) ** 2 + aqd)
    ev2dsqr36 = ev2 / np.sqrt(R ** 2 + aqq)
    ev2dsqr39 = ev2 / np.sqrt(R ** 2 + qa ** 2 + aqq)
    ev2dsqr40 = ev2 / np.sqrt(R ** 2 + qb ** 2 + aqq)
    ev3dsqr42 = ev3 / np.sqrt((R - qb) ** 2 + aqq)
    ev3dsqr44 = ev3 / np.sqrt((R + qb) ** 2 + aqq)
    ev3dsqr46 = ev3 / np.sqrt((R + qa) ** 2 + aqq)
    ev3dsqr48 = ev3 / np.sqrt((R - qa) ** 2 + aqq)

    ri = np.zeros(np.shape(R) + (22,))
    ri[..., 0] = ee                                                              # (SS|SS)
    ri[..., 1] = -dze                                                            # (SO|SS)
    ri[..., 2] = ee + qzze                                                       # (OO|SS)
    ri[..., 3] = ee + qxxe                                                       # (PP|SS)
    ri[..., 4] = -edz                                                            # (SS|OS)

    ri[..., 5] = (ev2 / np.sqrt((R + da - db) ** 2 + axx)                       # (SO|SO)
           + ev2 / np.sqrt((R - da + db) ** 2 + axx)
           - ev2 / np.sqrt((R - da - db) ** 2 + axx)
           - ev2 / np.sqrt((R + da + db) ** 2 + axx))

    ri[..., 6] = (ev1 / np.sqrt(R ** 2 + (da - db) ** 2 + axx)                  # (SP|SP)
           - ev1 / np.sqrt(R ** 2 + (da + db) ** 2 + axx))

    ri[..., 7] = (-edz                                                           # (OO|SO)
           + ev3 / np.sqrt((R + qa - db) ** 2 + aqd)
           - ev3 / np.sqrt((R + qa + db) ** 2 + aqd)
           + ev3 / np.sqrt((R - qa - db) ** 2 + aqd)
           - ev3 / np.sqrt((R - qa + db) ** 2 + aqd)
           - ev2dsqr24 + ev2dsqr26)

    ri[..., 8] = (-edz                                                           # (PP|SO)
           - ev2dsqr24
           + ev2 / np.sqrt((R - db) ** 2 + qa ** 2 + aqd)
           + ev2dsqr26
           - ev2 / np.sqrt((R + db) ** 2 + qa ** 2 + aqd))

    ri[..., 9] = (ev2 / np.sqrt((qa1 - db) ** 2 + (R + qa1) ** 2 + aqd)         # (PO|SP)
           - ev2 / np.sqrt((qa1 - db) ** 2 + (R - qa1) ** 2 + aqd)
           - ev2 / np.sqrt((qa1 + db) ** 2 + (R + qa1) ** 2 + aqd)
           + ev2 / np.sqrt((qa1 + db) ** 2 + (R - qa1) ** 2 + aqd))

    ri[..., 10] = ee + eqzz                                                      # (SS|OO)
    ri[..., 11] = ee + eqxx                                                      # (SS|PP)

    ri[..., 12] = (-dze                                                          # (SO|OO)
            + ev3 / np.sqrt((R + da - qb) ** 2 + adq)
            - ev3 / np.sqrt((R - da - qb) ** 2 + adq)
            + ev3 / np.sqrt((R + da + qb) ** 2 + adq)
            - ev3 / np.sqrt((R - da + qb) ** 2 + adq)
            + ev2dsqr22 - ev2dsqr20)

    ri[..., 13] = (-dze                                                          # (SO|PP)
            - ev2dsqr20
            + ev2 / np.sqrt((R + da) ** 2 + qb ** 2 + adq)
            + ev2dsqr22
            - ev2 / np.sqrt((R - da) ** 2 + qb ** 2 + adq))

    ri[..., 14] = (ev2 / np.sqrt((da - qb1) ** 2 + (R - qb1) ** 2 + adq)       # (SP|OP)
            - ev2 / np.sqrt((da - qb1) ** 2 + (R + qb1) ** 2 + adq)
            - ev2 / np.sqrt((da + qb1) ** 2 + (R - qb1) ** 2 + adq)
            + ev2 / np.sqrt((da + qb1) ** 2 + (R + qb1) ** 2 + adq))

    ri[..., 15] = (ee + eqzz + qzze                                             # (OO|OO)
            + ev4 / np.sqrt((R + qa - qb) ** 2 + aqq)
            + ev4 / np.sqrt((R + qa + qb) ** 2 + aqq)
            + ev4 / np.sqrt((R - qa - qb) ** 2 + aqq)
            + ev4 / np.sqrt((R - qa + qb) ** 2 + aqq)
            - ev3dsqr48 - ev3dsqr46 - ev3dsqr42 - ev3dsqr44 + ev2dsqr36)

    ri[..., 16] = (ee + eqzz + qxxe                                             # (PP|OO)
            + ev3 / np.sqrt((R - qb) ** 2 + qa ** 2 + aqq)
            + ev3 / np.sqrt((R + qb) ** 2 + qa ** 2 + aqq)
            - ev3dsqr42 - ev3dsqr44 - ev2dsqr39 + ev2dsqr36)

    ri[..., 17] = (ee + eqxx + qzze                                             # (OO|PP)
            + ev3 / np.sqrt((R + qa) ** 2 + qb ** 2 + aqq)
            + ev3 / np.sqrt((R - qa) ** 2 + qb ** 2 + aqq)
            - ev3dsqr46 - ev3dsqr48 - ev2dsqr40 + ev2dsqr36)

    qxxqxx = (ev3 / np.sqrt(R ** 2 + (qa - qb) ** 2 + aqq)
            + ev3 / np.sqrt(R ** 2 + (qa + qb) ** 2 + aqq)
            - ev2dsqr39 - ev2dsqr40 + ev2dsqr36)
    ri[..., 18] = ee + eqxx + qxxe + qxxqxx                                     # (PP|PP)

    ri[..., 19] = (ev3 / np.sqrt((R + qa1 - qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)  # (PO|PO)
            - ev3 / np.sqrt((R + qa1 + qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)
            - ev3 / np.sqrt((R - qa1 - qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)
            + ev3 / np.sqrt((R - qa1 + qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)
            - ev3 / np.sqrt((R + qa1 - qb1) ** 2 + (qa1 + qb1) ** 2 + aqq)
            + ev3 / np.sqrt((R + qa1 + qb1) ** 2 + (qa1 + qb1) ** 2 + aqq)
            + ev3 / np.sqrt((R - qa1 - qb1) ** 2 + (qa1 + qb1) ** 2 + aqq)
            - ev3 / np.sqrt((R - qa1 + qb1) ** 2 + (qa1 + qb1) ** 2 + aqq))

    qxxqyy = (ev2 / np.sqrt(R ** 2 + qa ** 2 + qb ** 2 + aqq)
            - ev2dsqr39 - ev2dsqr40 + ev2dsqr36)
    ri[..., 20] = ee + eqxx + qxxe + qxxqyy                                     # (PP|P*P*)

    ri[..., 21] = 0.5 * (qxxqxx - qxxqyy)                                       # (P*P|P*P)

    core = np.stack([
        ZB * ri[..., 0],    # core(ss/A, B) = Z_B * (SS|SS)
        ZB * ri[..., 1],    # core(sσ/A, B) = Z_B * (SO|SS)
        ZB * ri[..., 2],    # core(σσ/A, B) = Z_B * (OO|SS)
        ZB * ri[..., 3],    # core(ππ/A, B) = Z_B * (PP|SS)
        ZA * ri[..., 0],    # core(ss/B, A) = Z_A * (SS|SS)
        ZA * ri[..., 4],    # core(sσ/B, A) = Z_A * (SS|OS)
        ZA * ri[..., 10],   # core(σσ/B, A) = Z_A * (SS|OO)
        ZA * ri[..., 11],   # core(ππ/B, A) = Z_A * (SS|PP)
    ], axis=-1)

    return ri, core


def _ri_core_hh(rho0A, rho0B, R, ZA, ZB):
    """The single hydrogen-hydrogen integral and its core terms.

    Elementwise throughout, so it runs unchanged on scalars or on arrays
    over a pair axis. Shared by the scalar and batched entry points so the
    two cannot drift apart.
    """
    ev1, ev2, ev3, ev4 = EV / 2.0, EV / 4.0, EV / 8.0, EV / 16.0
    aee = (rho0A + rho0B) ** 2
    ri = (EV / np.sqrt(R ** 2 + aee))[..., None]
    core = np.stack([ZB * ri[..., 0], ZA * ri[..., 0]], axis=-1)
    return ri, core


def _ri_core_xh(daA, qaA, rho0A, rho0B, rho1A, rho2A, R, ZA, ZB):
    """The four heavy-hydrogen integrals and their core terms.

    Elementwise throughout, so it runs unchanged on scalars or on arrays
    over a pair axis. Shared by the scalar and batched entry points so the
    two cannot drift apart.
    """
    ev1, ev2, ev3, ev4 = EV / 2.0, EV / 4.0, EV / 8.0, EV / 16.0
    da = daA
    qa = qaA * 2.0
    aee = (rho0A + rho0B) ** 2
    ade = (rho1A + rho0B) ** 2
    aqe = (rho2A + rho0B) ** 2

    ri = np.zeros(np.shape(R) + (4,))
    ee = EV / np.sqrt(R ** 2 + aee)
    ri[..., 0] = ee  # (ss|ss)
    ri[..., 1] = ev1 / np.sqrt((R + da) ** 2 + ade) - ev1 / np.sqrt((R - da) ** 2 + ade)  # (sσ|ss)
    ev1dsqr6 = ev1 / np.sqrt(R ** 2 + aqe)
    ri[..., 2] = ee + ev2 / np.sqrt((R + qa) ** 2 + aqe) + ev2 / np.sqrt((R - qa) ** 2 + aqe) - ev1dsqr6  # (σσ|ss)
    ri[..., 3] = ee + ev1 / np.sqrt(R ** 2 + qa ** 2 + aqe) - ev1dsqr6  # (ππ|ss)

    core = np.stack([
        ZB * ri[..., 0],   # core(1,1)
        ZB * ri[..., 1],   # core(2,1)
        ZB * ri[..., 2],   # core(3,1)
        ZB * ri[..., 3],   # core(4,1)
        ZA * ri[..., 0],   # core(1,2)
    ])
    return ri, core



def two_center_integrals_batch(pair_params, R_ang):
    """Local-frame two-centre integrals for many pairs at once.

    Args:
        pair_params: sequence of (pA, pB) ElementParams.
        R_ang: (P,) separations in Angstrom.

    Returns:
        (ri, pair_type) where ri is (P, 22) zero-padded — HH fills column 0,
        XH/HX columns 0-3, XX all 22 — and pair_type is a length-P list of
        'HH' / 'XH' / 'HX' / 'XX' matching :func:`two_center_integrals`.

    Groups by pair type and calls the same ``_ri_core_*`` helpers the scalar
    entry point uses, so there is one implementation of the arithmetic.
    """
    P = len(pair_params)
    R = np.asarray(R_ang, dtype=np.float64) * ANG_TO_BOHR
    ri = np.zeros((P, 22))
    kinds = [''] * P

    mp = {}                                     # memoised per element
    def params_of(p):
        key = (p.Z, p.n_basis, p.gss, p.zeta_s, p.zeta_p, p.hsp, p.gpp, p.gp2)
        if key not in mp:
            mp[key] = _compute_multipole_params(p)
        return mp[key]

    groups: dict[str, list[int]] = {'HH': [], 'XH': [], 'HX': [], 'XX': []}
    for k, (pA, pB) in enumerate(pair_params):
        nA, nB = pA.n_basis, pB.n_basis
        kind = ('HH' if nA == 1 and nB == 1 else
                'HX' if nA == 1 else 'XH' if nB == 1 else 'XX')
        kinds[k] = kind
        groups[kind].append(k)

    for kind, idxs in groups.items():
        if not idxs:
            continue
        sel = np.array(idxs)
        # HX is XH with the pair reversed, exactly as the scalar handles it.
        first = [pair_params[i][1 if kind == 'HX' else 0] for i in sel]
        second = [pair_params[i][0 if kind == 'HX' else 1] for i in sel]
        A = [params_of(p) for p in first]
        B = [params_of(p) for p in second]
        col = lambda L, i: np.array([x[i] for x in L])
        ZA = np.array([float(p.n_valence) for p in first])
        ZB = np.array([float(p.n_valence) for p in second])
        Rs = R[sel]

        if kind == 'HH':
            block, _ = _ri_core_hh(col(A, 2), col(B, 2), Rs, ZA, ZB)
        elif kind == 'XX':
            block, _ = _ri_core_xx(col(A, 0), col(B, 0), col(A, 1), col(B, 1),
                                   col(A, 2), col(B, 2), col(A, 3), col(B, 3),
                                   col(A, 4), col(B, 4), Rs, ZA, ZB)
        else:
            block, _ = _ri_core_xh(col(A, 0), col(A, 1), col(A, 2), col(B, 2),
                                   col(A, 3), col(A, 4), Rs, ZA, ZB)
        ri[sel[:, None], np.arange(block.shape[1])[None, :]] = block

    return ri, kinds
