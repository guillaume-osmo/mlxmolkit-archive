"""PYSEQM-equivalent d-orbital multipole parameter derivation.

Replaces mlxmolkit's simplified compute_d_charge_separations with the
exact PYSEQM derivation chain:

  AIJ52, AIJ43, AIJ63   ← AIJL(zetap/s/d, qn0)
  ds_add, dp_add, ...    ← Slater-Condon parameters
  DD0, DP, DS, DD        ← POIJ(L, D, FG) — additive-term solver

Source: PYSEQM two_elec_two_center_int.py:31-96 (_pm6_d_param_from_key)
        + cal_par.py:283 (POIJ) + cal_par.py:377 (AIJL).
BSD-3-Clause / github.com/lanl/PYSEQM.

These are the params that PYSEQM passes to TETCILFDO as
rho3a/rho4a/rho5a/rho6a; mlxmolkit's previous values differed (e.g.
rho5a was 0.535 vs PYSEQM 3.230 — 6× off!), which corrupted the
d-orbital electron-nuclear attraction and the K-term.
"""

from __future__ import annotations

import math
import numpy as np

from .w_integrals import slater_condon_parameter as _SC

EV = 27.21
EV2 = EV / 4.0

# PYSEQM PM6 "tail" exponents for d-orbital atoms (s_orb_exp_tail, p_, d_)
# from parameters_PM6_MOPAC.csv. These differ from the main zeta_s/p/d
# exponents and are required by _pm6_d_param_from_key for the Slater-
# Condon parameter computation (PYSEQM uses MAIN zeta for AIJL,
# TAIL exponents for ds_add/dp_add/dd_add/dd0_add/dd4/dp3).
PM6_TAIL_EXPONENTS = {
    # Z: (s_tail, p_tail, d_tail)
    15: (6.04271, 2.37647, 7.14775),   # P
    16: (0.479722, 1.015507, 4.31747), # S
    17: (0.9563, 2.46407, 6.41033),    # Cl
    35: (3.09478, 3.06576, 2.82),      # Br
    53: (9.13524, 6.88819, 3.79152),   # I
}


def _aijl(z1: float, z2: float, n1: int, n2: int, L: int) -> float:
    """Port of PYSEQM AIJL — overlap-style scaling factor for s/p/d combinations."""
    if z1 == 0 or z2 == 0:
        return 0.0
    n1, n2 = int(n1), int(n2)
    a = math.factorial(n1 + n2 + L) / math.sqrt(math.factorial(2 * n1) * math.factorial(2 * n2))
    return (
        a
        * (2 * z1 / (z1 + z2)) ** n1
        * math.sqrt(2 * z1 / (z1 + z2))
        * (2 * z2 / (z1 + z2)) ** n2
        * math.sqrt(2 * z2 / (z1 + z2))
        / (z1 + z2) ** L
    )


def _poij(L: int, D: float, FG: float) -> float:
    """Port of PYSEQM POIJ — bisection-like search for the additive-term rho.

    L=0  : returns 0.5 * EV / FG (analytic)
    L=1  : iterate to find rho such that EV/4 * (1/rho - 1/sqrt(rho²+D²)) = FG
    L=2  : iterate to find rho such that EV/8 * (1/rho - 2/sqrt(rho²+D²/2) + 1/sqrt(rho²+D²)) = FG
    """
    if FG == 0.0:
        return 0.0
    if L == 0:
        return 0.5 * EV / FG

    EV4 = EV * 0.25
    EV8 = EV / 8.0
    dsq = D * D
    A1 = 0.1
    A2 = 5.0
    G1 = 0.3820
    G2 = 0.6180
    F1 = 0.0
    F2 = 0.0
    for _ in range(100):
        delta = A2 - A1
        if delta < 1e-8:
            return A2 if F1 >= F2 else A1
        Y1 = A1 + delta * G1
        Y2 = A1 + delta * G2
        if L == 1:
            F1 = (EV4 * (1.0 / Y1 - 1.0 / math.sqrt(Y1 ** 2 + dsq)) - FG) ** 2
            F2 = (EV4 * (1.0 / Y2 - 1.0 / math.sqrt(Y2 ** 2 + dsq)) - FG) ** 2
        else:  # L == 2
            F1 = (
                EV8
                * (
                    1.0 / Y1
                    - 2.0 / math.sqrt(Y1 ** 2 + dsq * 0.5)
                    + 1.0 / math.sqrt(Y1 ** 2 + dsq)
                )
                - FG
            ) ** 2
            F2 = (
                EV8
                * (
                    1.0 / Y2
                    - 2.0 / math.sqrt(Y2 ** 2 + dsq * 0.5)
                    + 1.0 / math.sqrt(Y2 ** 2 + dsq)
                )
                - FG
            ) ** 2
        if F1 < F2:
            A2 = Y2
        else:
            A1 = Y1
    return A2 if F1 >= F2 else A1


def pyseqm_d_params(pA) -> dict:
    """Return PYSEQM-equivalent (dpa, dsa, dda, rho3-6) for a d-orbital atom.

    Returns dict matching PYSEQM's TETCILFDO inputs:
      dp, ds, dorbdorb  (charge separations, derived from AIJ52/43/63)
      rho3 = DD0  (additive term L=0 for d-d quadrupole)
      rho4 = DP   (additive term L=1 for d-p dipole)
      rho5 = DS   (additive term L=2 for d-s quadrupole)
      rho6 = DD   (additive term L=2 for d-d quadrupole)
    """
    z = getattr(pA, "Z", 16)
    # MAIN zeta exponents (for AIJL)
    zeta_s = pA.zeta_s
    zeta_p = pA.zeta_p
    zeta_d = pA.zeta_d
    # TAIL exponents (for Slater-Condon parameters in _pm6_d_param_from_key)
    if z in PM6_TAIL_EXPONENTS:
        zs_tail, zp_tail, zd_tail = PM6_TAIL_EXPONENTS[z]
    else:
        # Fall back to main exponents for non-PM6-tail elements
        zs_tail, zp_tail, zd_tail = zeta_s, zeta_p, zeta_d
    # Principal quantum number
    qn0 = 3 if 11 <= z <= 17 else (
        4 if 30 <= z <= 36 else (
            5 if 50 <= z <= 56 else 3
        )
    )
    g2sd = getattr(pA, "G2SD", 0.0)

    # Slater-Condon parameters use TAIL exponents (PYSEQM convention)
    ds_add = 0.2 * _SC(2, qn0, zs_tail, qn0, zd_tail, qn0, zs_tail, qn0, zd_tail)
    dp_add = (4.0 / 15.0) * _SC(1, qn0, zp_tail, qn0, zd_tail, qn0, zp_tail, qn0, zd_tail)
    dd_add = (4.0 / 49.0) * _SC(2, qn0, zd_tail, qn0, zd_tail, qn0, zd_tail, qn0, zd_tail)
    dd0_add = _SC(0, qn0, zd_tail, qn0, zd_tail, qn0, zd_tail, qn0, zd_tail)
    dd4 = _SC(4, qn0, zd_tail, qn0, zd_tail, qn0, zd_tail, qn0, zd_tail)
    dp3 = (27.0 / 245.0) * _SC(3, qn0, zp_tail, qn0, zd_tail, qn0, zp_tail, qn0, zd_tail)
    # AIJL uses MAIN exponents
    aij52 = _aijl(zeta_p, zeta_d, qn0, qn0, 1)
    aij43 = _aijl(zeta_s, zeta_d, qn0, qn0, 2)
    aij63 = _aijl(zeta_d, zeta_d, qn0, qn0, 2)

    # dp = AIJ52/sqrt(5)
    dp = aij52 / math.sqrt(5.0)
    # ds = sqrt(AIJ43 * sqrt(1/15)) * sqrt(2)
    ds_charge = math.sqrt(aij43 * math.sqrt(1.0 / 15.0)) * math.sqrt(2.0)
    # dorbdorb = sqrt(2 * AIJ63 / 7)
    dorbdorb = math.sqrt(2.0 * aij63 / 7.0)

    # DS = POIJ(2, ds_charge, dsAdditiveTerm)
    DS = _poij(2, ds_charge, ds_add)

    # DD0 = POIJ(0, 1.0, 0.2 * (FG + 2*FG1 + 2*FG2))
    #   FG  = dd0_add + dd_add + 4/49 * dd4
    #   FG1 = dd0_add + 0.5 * dd_add - 24/441 * dd4
    #   FG2 = dd0_add - dd_add + 6/441 * dd4
    FG = dd0_add + dd_add + 4.0 / 49.0 * dd4
    FG1 = dd0_add + 0.5 * dd_add - 24.0 / 441.0 * dd4
    FG2 = dd0_add - dd_add + 6.0 / 441.0 * dd4
    DD0 = _poij(0, 1.0, 0.2 * (FG + 2.0 * FG1 + 2.0 * FG2))

    # DP = POIJ(1, dp, FG - 1.8 * FG1)
    #   FG  = dp_add + dp3
    #   FG1 = (3/49) * (245/27) * dp3
    FG_dp = dp_add + dp3
    FG1_dp = (3.0 / 49.0) * (245.0 / 27.0) * dp3
    DP = _poij(1, dp, FG_dp - 1.8 * FG1_dp)

    # DD = POIJ(2, dorbdorb, FG - (20/35) * FG1)
    #   FG  = 3/4 * dd_add + 20/441 * dd4
    #   FG1 = 35/441 * dd4
    FG_dd = 3.0 / 4.0 * dd_add + 20.0 / 441.0 * dd4
    FG1_dd = 35.0 / 441.0 * dd4
    DD = _poij(2, dorbdorb, FG_dd - (20.0 / 35.0) * FG1_dd)

    return {
        "dp": dp,
        "ds": ds_charge,
        "dorbdorb": dorbdorb,
        "rho3": DD0,
        "rho4": DP,
        "rho5": DS,
        "rho6": DD,
    }
