"""
Multi-method support for NDDO semi-empirical calculations.

Supported methods:
  - 'RM1': Recife Model 1 (Rocha et al. 2006) — default
  - 'AM1': Austin Model 1 (Dewar et al. 1985)
  - 'AM1_STAR': Geometry-corrected AM1 (Ong et al. 2025, CL=500)

Each method uses the same NDDO framework but different parameters.
Parameters are loaded from params.py (RM1) or defined here (AM1, AM1*).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from .params import ElementParams, RM1_PARAMS, _EISOL_COEFFICIENTS, _compute_eisol


# AM1 parameters from PYSEQM/MOPAC CSV (exact values)
AM1_PARAMS: Dict[int, ElementParams] = {
    1: ElementParams(
        Z=1, symbol="H", n_basis=1, n_valence=1, eheat=52.102,
        Uss=-11.396427, Upp=0.0,
        zeta_s=1.188078, zeta_p=0.0,
        beta_s=-6.173787, beta_p=0.0,
        gss=12.848, gsp=0.0, gpp=0.0, gp2=0.0, hsp=0.0,
        alpha=2.882324,
        gauss_K=[0.122796, 0.005090, -0.018336, 0.0],
        gauss_L=[5.0, 5.0, 2.0, 0.0],
        gauss_M=[1.2, 1.8, 2.1, 0.0],
    ),
    6: ElementParams(
        Z=6, symbol="C", n_basis=4, n_valence=4, eheat=170.89,
        Uss=-52.028658, Upp=-39.614239,
        zeta_s=1.808665, zeta_p=1.685116,
        beta_s=-15.715783, beta_p=-7.719283,
        gss=12.23, gsp=11.47, gpp=11.08, gp2=9.84, hsp=2.43,
        alpha=2.648274,
        gauss_K=[0.011355, 0.045924, -0.020061, -0.001260],
        gauss_L=[5.0, 5.0, 5.0, 5.0],
        gauss_M=[1.6, 1.85, 2.05, 2.65],
    ),
    7: ElementParams(
        Z=7, symbol="N", n_basis=4, n_valence=5, eheat=113.00,
        Uss=-71.860000, Upp=-57.167581,
        zeta_s=2.315410, zeta_p=2.157940,
        beta_s=-20.299110, beta_p=-18.238666,
        gss=13.59, gsp=12.66, gpp=12.98, gp2=11.59, hsp=3.14,
        alpha=2.947286,
        gauss_K=[0.025251, 0.028953, -0.005806, 0.0],
        gauss_L=[5.0, 5.0, 2.0, 0.0],
        gauss_M=[1.5, 2.1, 2.4, 0.0],
    ),
    8: ElementParams(
        Z=8, symbol="O", n_basis=4, n_valence=6, eheat=59.559,
        Uss=-97.830000, Upp=-78.262380,
        zeta_s=3.108032, zeta_p=2.524039,
        beta_s=-29.272773, beta_p=-29.272773,
        gss=15.42, gsp=14.48, gpp=14.52, gp2=12.98, hsp=3.94,
        alpha=4.455371,
        gauss_K=[0.280962, 0.081430, 0.0, 0.0],
        gauss_L=[5.0, 7.0, 0.0, 0.0],
        gauss_M=[0.847918, 1.445071, 0.0, 0.0],
    ),
}

# Compute Eisol for AM1
for _z, _p in AM1_PARAMS.items():
    _p.eisol = _compute_eisol(_p)


# AM1* (Ong et al. 2025) — geometry-corrected reparameterization
# CL=500 variant from Table 3(a) of the paper
# Only Uss, Upp, zeta_s, zeta_p, beta_s, beta_p, alpha, and Gaussians change.
# One-center integrals (gss, gsp, gpp, gp2, hsp) stay same as AM1.
AM1_STAR_PARAMS: Dict[int, ElementParams] = {
    1: ElementParams(
        Z=1, symbol="H", n_basis=1, n_valence=1, eheat=52.102,
        Uss=-11.2709, Upp=0.0,
        zeta_s=1.169675, zeta_p=0.0,
        beta_s=-5.93954, beta_p=0.0,
        gss=12.848, gsp=0.0, gpp=0.0, gp2=0.0, hsp=0.0,  # same as AM1
        alpha=2.65747,
        gauss_K=[3.844408, 1.315199, -4.36017, 0.0],
        gauss_L=[4.793111, 5.778936, 3.499363, 0.0],
        gauss_M=[1.452021, 1.867145, 1.540349, 0.0],
    ),
    6: ElementParams(
        Z=6, symbol="C", n_basis=4, n_valence=4, eheat=170.89,
        Uss=-51.7812, Upp=-39.9628,
        zeta_s=1.6352, zeta_p=1.733751,
        beta_s=-11.5065, beta_p=-9.19868,
        gss=12.23, gsp=11.47, gpp=11.08, gp2=9.84, hsp=2.43,  # same as AM1
        alpha=2.712553,
        gauss_K=[0.05463, -0.07112, 0.07237, -0.00307],
        gauss_L=[7.312193, 10.37091, 12.12756, 24.36623],
        gauss_M=[1.700631, 2.516281, 2.507685, 3.511127],
    ),
    7: ElementParams(
        Z=7, symbol="N", n_basis=4, n_valence=5, eheat=113.00,
        Uss=-66.567, Upp=-57.3715,
        zeta_s=2.469507, zeta_p=2.033593,
        beta_s=-15.8169, beta_p=-16.7438,
        gss=13.59, gsp=12.66, gpp=12.98, gp2=11.59, hsp=3.14,  # same as AM1
        alpha=3.097823,
        gauss_K=[0.084117, 0.436119, -0.4254, 0.0],
        gauss_L=[2.288517, 39.64693, 38.46668, 0.0],
        gauss_M=[1.217317, 2.414354, 2.414079, 0.0],
    ),
    8: ElementParams(
        Z=8, symbol="O", n_basis=4, n_valence=6, eheat=59.559,
        Uss=-82.4869, Upp=-78.643,
        zeta_s=11.81357, zeta_p=2.438453,
        beta_s=-150.235, beta_p=-26.3035,
        gss=15.42, gsp=14.48, gpp=14.52, gp2=12.98, hsp=3.94,  # same as AM1
        alpha=3.311738,
        gauss_K=[-0.03582, 0.008038, 0.0, 0.0],
        gauss_L=[26.29376, 155.0571, 0.0, 0.0],
        gauss_M=[0.822597, 1.33939, 0.0, 0.0],
    ),
}

# Compute Eisol for AM1*
for _z, _p in AM1_STAR_PARAMS.items():
    _p.eisol = _compute_eisol(_p)


# RM1* (Ong et al. 2025) — geometry-corrected reparameterization of RM1
# CL=300 variant from Table S8 of the SI
# ALL parameters re-optimized including gss, gsp, hsp, gpp, gp2
RM1_STAR_PARAMS: Dict[int, ElementParams] = {
    1: ElementParams(
        Z=1, symbol="H", n_basis=1, n_valence=1, eheat=52.102,
        Uss=-13.51263, Upp=0.0,
        zeta_s=1.111345, zeta_p=0.0,
        beta_s=-6.387845, beta_p=0.0,
        gss=17.958837, gsp=0.0, gpp=0.0, gp2=0.0, hsp=0.0,
        alpha=2.702005,
        gauss_K=[0.223322, 0.082594, -0.383675, 0.0],
        gauss_L=[5.82296, 11.344232, 1.027211, 0.0],
        gauss_M=[1.372451, 1.994656, 0.824151, 0.0],
    ),
    6: ElementParams(
        Z=6, symbol="C", n_basis=4, n_valence=4, eheat=170.89,
        Uss=-49.020966, Upp=-40.185318,
        zeta_s=1.693587, zeta_p=1.78194,
        beta_s=-10.659812, beta_p=-9.932403,
        gss=12.83272, gsp=10.293862, gpp=11.194386, gp2=10.393041, hsp=1.147843,
        alpha=2.702952,
        gauss_K=[-0.01131, 0.098069, -0.040013, -0.00453],
        gauss_L=[6.454638, 8.284788, 7.953951, 17.380582],
        gauss_M=[1.540066, 1.684195, 1.698655, 2.801171],
    ),
    7: ElementParams(
        Z=7, symbol="N", n_basis=4, n_valence=5, eheat=113.00,
        Uss=-61.716728, Upp=-48.58914,
        zeta_s=2.2132, zeta_p=2.224215,
        beta_s=-20.019847, beta_p=-18.289266,
        gss=17.854284, gsp=8.541609, gpp=12.820501, gp2=12.022664, hsp=5.504014,
        alpha=3.075064,
        gauss_K=[0.043236, 0.034417, -0.013481, 0.0],
        gauss_L=[6.76928, 21.837709, 12.575294, 0.0],
        gauss_M=[1.327038, 2.800138, 2.828134, 0.0],
    ),
    8: ElementParams(
        Z=8, symbol="O", n_basis=4, n_valence=6, eheat=59.559,
        Uss=-96.522323, Upp=-76.999427,
        zeta_s=5.100172, zeta_p=2.198391,
        beta_s=-50.873122, beta_p=-21.596604,
        gss=10.115357, gsp=15.012399, gpp=13.763601, gp2=11.274583, hsp=0.000367,
        alpha=4.194039,
        gauss_K=[0.123471, 0.152106, 0.0, 0.0],
        gauss_L=[7.859863, 2.467465, 0.0, 0.0],
        gauss_M=[0.784888, 0.993296, 0.0, 0.0],
    ),
}

# Compute Eisol for RM1*
for _z, _p in RM1_STAR_PARAMS.items():
    _p.eisol = _compute_eisol(_p)


# Method registry
METHOD_PARAMS: Dict[str, Dict[int, ElementParams]] = {
    'RM1': RM1_PARAMS,
    'AM1': AM1_PARAMS,
    'AM1_STAR': AM1_STAR_PARAMS,
    'RM1_STAR': RM1_STAR_PARAMS,
}


def get_params(method: str = 'RM1') -> Dict[int, ElementParams]:
    """Get parameter dictionary for a given method.

    Args:
        method: 'RM1', 'AM1', or 'AM1_STAR'

    Returns:
        Dict mapping atomic number → ElementParams
    """
    method = method.upper().replace('-', '_').replace('*', '_STAR')
    if method not in METHOD_PARAMS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHOD_PARAMS.keys())}")
    return METHOD_PARAMS[method]
