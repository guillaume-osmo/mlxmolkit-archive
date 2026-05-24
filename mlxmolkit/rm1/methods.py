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


# PM3 parameters from PYSEQM/MOPAC CSV (Stewart 1989)
# More polarized charges than AM1/RM1 — better for COSMO-RS
PM3_PARAMS: Dict[int, ElementParams] = {
    1: ElementParams(
        Z=1, symbol="H", n_basis=1, n_valence=1, eheat=52.102,
        Uss=-13.073321, Upp=0.0,
        zeta_s=0.967807, zeta_p=0.0,
        beta_s=-5.626512, beta_p=0.0,
        gss=14.794208, gsp=0.0, gpp=0.0, gp2=0.0, hsp=0.0,
        alpha=3.356386,
        gauss_K=[1.12875, -1.060329, 0.0, 0.0],
        gauss_L=[5.096282, 6.003788, 0.0, 0.0],
        gauss_M=[1.537465, 1.570189, 0.0, 0.0],
    ),
    6: ElementParams(
        Z=6, symbol="C", n_basis=4, n_valence=4, eheat=170.89,
        Uss=-47.27032, Upp=-36.266918,
        zeta_s=1.565085, zeta_p=1.842345,
        beta_s=-11.910015, beta_p=-9.802755,
        gss=11.200708, gsp=10.265027, gpp=10.796292, gp2=9.042566, hsp=2.29098,
        alpha=2.707807,
        gauss_K=[0.050107, 0.050733, 0.0, 0.0],
        gauss_L=[6.003165, 6.002979, 0.0, 0.0],
        gauss_M=[1.642214, 0.892488, 0.0, 0.0],
    ),
    7: ElementParams(
        Z=7, symbol="N", n_basis=4, n_valence=5, eheat=113.0,
        Uss=-49.335672, Upp=-47.509736,
        zeta_s=2.028094, zeta_p=2.313728,
        beta_s=-14.062521, beta_p=-20.043848,
        gss=11.904787, gsp=7.348565, gpp=11.754672, gp2=10.807277, hsp=1.136713,
        alpha=2.830545,
        gauss_K=[1.501674, -1.505772, 0.0, 0.0],
        gauss_L=[5.901148, 6.004658, 0.0, 0.0],
        gauss_M=[1.71074, 1.716149, 0.0, 0.0],
    ),
    8: ElementParams(
        Z=8, symbol="O", n_basis=4, n_valence=6, eheat=59.559,
        Uss=-86.993002, Upp=-71.87958,
        zeta_s=3.796544, zeta_p=2.389402,
        beta_s=-45.202651, beta_p=-24.752515,
        gss=15.75576, gsp=10.62116, gpp=13.654016, gp2=12.406095, hsp=0.593883,
        alpha=3.217102,
        gauss_K=[-1.131128, 1.137891, 0.0, 0.0],
        gauss_L=[6.002477, 5.950512, 0.0, 0.0],
        gauss_M=[1.607311, 1.598395, 0.0, 0.0],
    ),
    9: ElementParams(
        Z=9, symbol="F", n_basis=4, n_valence=7, eheat=18.86,
        Uss=-110.435303, Upp=-105.685047,
        zeta_s=4.708555, zeta_p=2.491178,
        beta_s=-48.405939, beta_p=-27.74466,
        gss=10.496667, gsp=16.073689, gpp=14.817256, gp2=14.418393, hsp=0.727763,
        alpha=3.358921,
        gauss_K=[-0.012166, -0.002852, 0.0, 0.0],
        gauss_L=[6.023574, 6.003717, 0.0, 0.0],
        gauss_M=[1.856859, 2.636158, 0.0, 0.0],
    ),
    15: ElementParams(
        Z=15, symbol="P", n_basis=4, n_valence=5, eheat=75.42,
        Uss=-40.413096, Upp=-29.593052,
        zeta_s=2.017563, zeta_p=1.504732,
        beta_s=-12.615879, beta_p=-4.16004,
        gss=7.801615, gsp=5.186949, gpp=6.618478, gp2=6.062002, hsp=1.542809,
        alpha=1.940534,
        gauss_K=[-0.611421, -0.093935, 0.0, 0.0],
        gauss_L=[1.997272, 1.99836, 0.0, 0.0],
        gauss_M=[0.794624, 1.910677, 0.0, 0.0],
    ),
    16: ElementParams(
        Z=16, symbol="S", n_basis=4, n_valence=6, eheat=66.4,
        Uss=-49.895371, Upp=-44.392583,
        zeta_s=1.891185, zeta_p=1.658972,
        beta_s=-8.827465, beta_p=-8.091415,
        gss=8.964667, gsp=6.785936, gpp=9.968164, gp2=7.970247, hsp=4.041836,
        alpha=2.269706,
        gauss_K=[-0.399191, -0.054899, 0.0, 0.0],
        gauss_L=[6.000669, 6.001845, 0.0, 0.0],
        gauss_M=[0.962123, 1.579944, 0.0, 0.0],
    ),
    17: ElementParams(
        Z=17, symbol="Cl", n_basis=4, n_valence=7, eheat=28.99,
        Uss=-100.626747, Upp=-53.614396,
        zeta_s=2.24621, zeta_p=2.15101,
        beta_s=-27.52856, beta_p=-11.593922,
        gss=16.013601, gsp=8.048115, gpp=7.522215, gp2=7.504154, hsp=3.481153,
        alpha=2.517296,
        gauss_K=[-0.171591, -0.013458, 0.0, 0.0],
        gauss_L=[6.000802, 1.966618, 0.0, 0.0],
        gauss_M=[1.087502, 2.292891, 0.0, 0.0],
    ),
    35: ElementParams(
        Z=35, symbol="Br", n_basis=4, n_valence=7, eheat=26.74,
        Uss=-116.619311, Upp=-74.227129,
        zeta_s=5.348457, zeta_p=2.12759,
        beta_s=-31.171342, beta_p=-6.814013,
        gss=15.943425, gsp=16.06168, gpp=8.282763, gp2=7.816849, hsp=0.578869,
        alpha=2.511842,
        gauss_K=[0.960458, -0.954916, 0.0, 0.0],
        gauss_L=[5.976508, 5.944703, 0.0, 0.0],
        gauss_M=[2.321654, 2.328142, 0.0, 0.0],
    ),
    53: ElementParams(
        Z=53, symbol="I", n_basis=4, n_valence=7, eheat=25.517,
        Uss=-96.454037, Upp=-61.091582,
        zeta_s=7.001013, zeta_p=2.454354,
        beta_s=-14.494234, beta_p=-5.894703,
        gss=13.631943, gsp=14.990406, gpp=7.28833, gp2=5.966407, hsp=2.630035,
        alpha=1.990185,
        gauss_K=[-0.131481, -0.036897, 0.0, 0.0],
        gauss_L=[5.206417, 6.010117, 0.0, 0.0],
        gauss_M=[1.748824, 2.710373, 0.0, 0.0],
    ),
}

for _z, _p in PM3_PARAMS.items():
    _p.eisol = _compute_eisol(_p)


# PM6 sp-only parameters (Stewart 2007) from PYSEQM/MOPAC CSV
# Uses sp basis like AM1/PM3 but PM6-optimized parameters
# More polarized charges than PM3 — better for COSMO-RS
PM6_SP_PARAMS: Dict[int, ElementParams] = {
    1: ElementParams(
        Z=1, symbol="H", n_basis=1, n_valence=1, eheat=52.102,
        Uss=-11.24696, Upp=0.0,
        zeta_s=1.26864, zeta_p=0.0,
        beta_s=-8.35298, beta_p=0.0,
        gss=14.44869, gsp=0.0, gpp=0.0, gp2=0.0, hsp=0.0,
        alpha=3.35639,
        gauss_K=[0.024184, 0.0, 0.0, 0.0],
        gauss_L=[3.055953, 0.0, 0.0, 0.0],
        gauss_M=[1.786011, 0.0, 0.0, 0.0],
    ),
    6: ElementParams(
        Z=6, symbol="C", n_basis=4, n_valence=4, eheat=170.89,
        Uss=-51.08965, Upp=-39.93792,
        zeta_s=2.04756, zeta_p=1.70284,
        beta_s=-15.38524, beta_p=-7.47193,
        gss=13.33552, gsp=11.52813, gpp=10.77833, gp2=9.48621, hsp=0.71732,
        alpha=2.70781,
        gauss_K=[0.0463, 0.0, 0.0, 0.0],
        gauss_L=[2.10021, 0.0, 0.0, 0.0],
        gauss_M=[1.33396, 0.0, 0.0, 0.0],
    ),
    7: ElementParams(
        Z=7, symbol="N", n_basis=4, n_valence=5, eheat=113.0,
        Uss=-57.78482, Upp=-49.89304,
        zeta_s=2.38041, zeta_p=1.99925,
        beta_s=-17.97938, beta_p=-15.05502,
        gss=12.35703, gsp=9.63619, gpp=12.57076, gp2=10.57643, hsp=2.87154,
        alpha=2.83054,
        gauss_K=[-0.001436, 0.0, 0.0, 0.0],
        gauss_L=[0.495196, 0.0, 0.0, 0.0],
        gauss_M=[1.704857, 0.0, 0.0, 0.0],
    ),
    8: ElementParams(
        Z=8, symbol="O", n_basis=4, n_valence=6, eheat=59.559,
        Uss=-91.67876, Upp=-70.46095,
        zeta_s=5.42175, zeta_p=2.27096,
        beta_s=-65.63514, beta_p=-21.6226,
        gss=11.30404, gsp=15.80742, gpp=13.6182, gp2=10.33277, hsp=5.0108,
        alpha=3.2171,
        gauss_K=[-0.01777, 0.0, 0.0, 0.0],
        gauss_L=[3.05831, 0.0, 0.0, 0.0],
        gauss_M=[1.89644, 0.0, 0.0, 0.0],
    ),
    9: ElementParams(
        Z=9, symbol="F", n_basis=4, n_valence=7, eheat=18.86,
        Uss=-140.22563, Upp=-98.77804,
        zeta_s=6.04385, zeta_p=2.90672,
        beta_s=-69.92259, beta_p=-30.44816,
        gss=12.44682, gsp=18.49608, gpp=8.41737, gp2=12.17982, hsp=2.60438,
        alpha=3.35892,
        gauss_K=[-0.01079, 0.0, 0.0, 0.0],
        gauss_L=[6.00465, 0.0, 0.0, 0.0],
        gauss_M=[1.84772, 0.0, 0.0, 0.0],
    ),
    15: ElementParams(
        Z=15, symbol="P", n_basis=4, n_valence=5, eheat=75.42,
        Uss=-48.72991, Upp=-40.35469,
        zeta_s=2.15803, zeta_p=1.80534,
        beta_s=-14.58378, beta_p=-11.74473,
        gss=8.75886, gsp=8.48368, gpp=8.66275, gp2=7.73426, hsp=0.87168,
        alpha=1.94053,
        gauss_K=[-0.03432, 0.0, 0.0, 0.0],
        gauss_L=[6.00139, 0.0, 0.0, 0.0],
        gauss_M=[2.29674, 0.0, 0.0, 0.0],
    ),
    16: ElementParams(
        Z=16, symbol="S", n_basis=4, n_valence=6, eheat=66.4,
        Uss=-47.530706, Upp=-39.191045,
        zeta_s=2.192844, zeta_p=1.841078,
        beta_s=-13.82744, beta_p=-7.66461,
        gss=9.17035, gsp=5.944296, gpp=8.165473, gp2=7.301878, hsp=5.005404,
        alpha=2.26971,
        gauss_K=[-0.036928, 0.0, 0.0, 0.0],
        gauss_L=[1.795067, 0.0, 0.0, 0.0],
        gauss_M=[2.082618, 0.0, 0.0, 0.0],
    ),
    17: ElementParams(
        Z=17, symbol="Cl", n_basis=4, n_valence=7, eheat=28.99,
        Uss=-61.38993, Upp=-54.4828,
        zeta_s=2.63705, zeta_p=2.11815,
        beta_s=-2.36799, beta_p=-13.80214,
        gss=11.14265, gsp=7.48788, gpp=9.55189, gp2=8.12844, hsp=5.00427,
        alpha=2.5173,
        gauss_K=[-0.01321, 0.0, 0.0, 0.0],
        gauss_L=[3.68702, 0.0, 0.0, 0.0],
        gauss_M=[2.54463, 0.0, 0.0, 0.0],
    ),
    35: ElementParams(
        Z=35, symbol="Br", n_basis=4, n_valence=7, eheat=26.74,
        Uss=-45.83436, Upp=-50.29368,
        zeta_s=4.67068, zeta_p=2.03563,
        beta_s=-32.13166, beta_p=-9.51448,
        gss=7.61679, gsp=5.01042, gpp=9.64922, gp2=8.34379, hsp=4.99655,
        alpha=2.51184,
        gauss_K=[-0.005, 0.0, 0.0, 0.0],
        gauss_L=[6.00129, 0.0, 0.0, 0.0],
        gauss_M=[2.89515, 0.0, 0.0, 0.0],
    ),
    53: ElementParams(
        Z=53, symbol="I", n_basis=4, n_valence=7, eheat=25.517,
        Uss=-59.97323, Upp=-56.45983,
        zeta_s=4.49865, zeta_p=1.91707,
        beta_s=-30.52248, beta_p=-5.94212,
        gss=7.23476, gsp=9.15441, gpp=9.87747, gp2=8.03592, hsp=5.00422,
        alpha=1.99019,
        gauss_K=[-0.03552, 0.0, 0.0, 0.0],
        gauss_L=[1.74439, 0.0, 0.0, 0.0],
        gauss_M=[1.22384, 0.0, 0.0, 0.0],
    ),
}

for _z, _p in PM6_SP_PARAMS.items():
    _p.eisol = _compute_eisol(_p)


# PM6 full with d-orbitals
from .pm6_params import PM6_FULL_PARAMS


# Method registry
METHOD_PARAMS: Dict[str, Dict[int, ElementParams]] = {
    'RM1': RM1_PARAMS,
    'AM1': AM1_PARAMS,
    'PM3': PM3_PARAMS,
    'PM6': PM6_SP_PARAMS,        # PM6 (sp-only, matches PYSEQM/MOPAC)
    'PM6_SP': PM6_SP_PARAMS,     # PM6 sp-only (alias)
    'PM6_D': PM6_FULL_PARAMS,    # PM6 with d-orbitals (experimental)
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
