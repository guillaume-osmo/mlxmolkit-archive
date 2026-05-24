"""
D-orbital charge separations and additive terms for PM6 two-center integrals.

Computes dp, ds, dd, dd0 (charge separations) and their corresponding
additive terms (rho3-rho6) which enter the two-center Coulomb integrals.

Port of PYSEQM two_elec_two_center_int.py lines 130-270.
"""
from __future__ import annotations

import math
import numpy as np
from .w_integrals import slater_condon_parameter
from .two_center_integrals import _compute_multipole_params, EV
from .params import principal_qn


def _aijl(Z1: float, Z2: float, N1: int, N2: int, L: int) -> float:
    """AIJL function from MOPAC — orbital overlap factor.

    Port of PYSEQM cal_par.py AIJL.
    """
    N1 = int(N1)
    N2 = int(N2)
    if Z1 == 0 or Z2 == 0:
        return 0.0
    val = math.factorial(N1 + N2 + L) / math.sqrt(
        math.factorial(2 * N1) * math.factorial(2 * N2)
    )
    val *= (2 * Z1 / (Z1 + Z2)) ** N1 * math.sqrt(2 * Z1 / (Z1 + Z2))
    val *= (2 * Z2 / (Z1 + Z2)) ** N2 * math.sqrt(2 * Z2 / (Z1 + Z2))
    val /= (Z1 + Z2) ** L
    return val


def _poij_scalar(L: int, D: float, FG: float) -> float:
    """POIJ function for scalar inputs. Golden-section search."""
    if L == 0:
        return 0.5 * EV / FG if FG > 1e-10 else 0.0

    DSQ = D * D
    EV4 = EV * 0.25
    EV8 = EV / 8.0

    A1 = 0.1
    A2 = 5.0
    G1 = 0.3820
    G2 = 0.6180

    for _ in range(100):
        DELTA = A2 - A1
        if DELTA < 1e-8:
            break

        Y1 = A1 + DELTA * G1
        Y2 = A1 + DELTA * G2

        if L == 1:
            F1 = (EV4 * (1.0/Y1 - 1.0/math.sqrt(Y1**2 + DSQ)) - FG) ** 2
            F2 = (EV4 * (1.0/Y2 - 1.0/math.sqrt(Y2**2 + DSQ)) - FG) ** 2
        else:  # L == 2
            F1 = (EV8 * (1.0/Y1 - 2.0/math.sqrt(Y1**2 + DSQ*0.5) + 1.0/math.sqrt(Y1**2 + DSQ)) - FG) ** 2
            F2 = (EV8 * (1.0/Y2 - 2.0/math.sqrt(Y2**2 + DSQ*0.5) + 1.0/math.sqrt(Y2**2 + DSQ)) - FG) ** 2

        if F1 < F2:
            A2 = Y2
        else:
            A1 = Y1

    return A2 if F1 >= F2 else A1


def compute_d_charge_separations(p) -> dict:
    """Compute d-orbital charge separations and additive terms for one atom.

    For PM6 main-group (category B): Z in (13-17, 33-35, 51-53)
    qn_d = qn_sp (same principal quantum number)

    Returns dict with: dp, ds, dd0, dorbdorb (charge separations)
                       rho3(=DD0), rho4(=DP), rho5(=DS), rho6(=DD)
    """
    if not getattr(p, 'has_d', False) or p.zeta_d <= 0:
        return {'dp': 0, 'ds': 0, 'dd0': 0, 'dorbdorb': 0,
                'rho3': 0, 'rho4': 0, 'rho5': 0, 'rho6': 0}

    SC = slater_condon_parameter
    # PYSEQM convention: row of the periodic table (3 for S/Cl, 4 for Br, 5 for I)
    qn = principal_qn(p.Z)
    # For PM6 category B (main-group with d): qn_d = qn_sp
    qn_d = qn

    zs = p.zeta_s
    zp = p.zeta_p
    zd = p.zeta_d

    # Compute Slater-Condon-based additive terms
    G2SD = getattr(p, 'G2SD', 0.0)

    if abs(G2SD) > 1e-9:
        ds_add = 0.2 * G2SD
    else:
        ds_add = 0.2 * SC(2, qn, zs, qn_d, zd, qn, zs, qn_d, zd)

    dp_add = (4.0/15.0) * SC(1, qn, zp, qn_d, zd, qn, zp, qn_d, zd)

    dd_add = (4.0/49.0) * SC(2, qn_d, zd, qn_d, zd, qn_d, zd, qn_d, zd)

    dd0_add = SC(0, qn_d, zd, qn_d, zd, qn_d, zd, qn_d, zd)

    dd4 = SC(4, qn_d, zd, qn_d, zd, qn_d, zd, qn_d, zd)

    dp3 = (27.0/245.0) * SC(3, qn, zp, qn_d, zd, qn, zp, qn_d, zd)

    # AIJL orbital overlap factors
    aij52 = _aijl(zp, zd, qn, qn_d, 1)  # zetap, zetad
    aij43 = _aijl(zs, zd, qn, qn_d, 2)  # zetas, zetad
    aij63 = _aijl(zd, zd, qn_d, qn_d, 2)  # zetad, zetad

    # Charge separations
    dp = aij52 / math.sqrt(5.0)
    D_ds = math.sqrt(aij43 * math.sqrt(1.0/15.0)) * math.sqrt(2.0)
    ds = D_ds
    D_dd = aij63 / 7.0
    dorbdorb = math.sqrt(2.0 * D_dd) if D_dd > 0 else 0.0

    # Additive terms (rho3-rho6 via POIJ golden-section search)
    # DS (rho5)
    rho5 = _poij_scalar(2, D_ds, ds_add) if D_ds > 0 and ds_add > 0 else 0.0

    # DD0 (rho3)
    FG = dd0_add + dd_add + 4.0/49.0 * dd4
    FG1 = dd0_add + 0.5 * dd_add - 24.0/441.0 * dd4
    FG2 = dd0_add - dd_add + 6.0/441.0 * dd4
    rho3 = _poij_scalar(0, 1.0, 0.2 * (FG + 2.0*FG1 + 2.0*FG2))

    # DP (rho4)
    D_dp = aij52 / math.sqrt(5.0)
    FG_dp = dp_add + dp3
    FG1_dp = 3.0/49.0 * 245.0/27.0 * dp3
    rho4 = _poij_scalar(1, D_dp, FG_dp - 1.8 * FG1_dp) if D_dp > 0 else 0.0

    # DD (rho6)
    FG_dd = 0.75 * dd_add + 20.0/441.0 * dd4
    FG1_dd = 35.0/441.0 * dd4
    rho6 = _poij_scalar(2, dorbdorb, FG_dd - (20.0/35.0)*FG1_dd) if dorbdorb > 0 else 0.0

    return {
        'dp': dp, 'ds': ds, 'dd0': D_dd, 'dorbdorb': dorbdorb,
        'rho3': rho3,  # DD0 additive term
        'rho4': rho4,  # DP additive term
        'rho5': rho5,  # DS additive term
        'rho6': rho6,  # DD additive term
    }
