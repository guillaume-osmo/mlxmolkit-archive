"""
PM6 full parameters with d-orbital support.

Elements with d-orbitals in PM6: P(15), S(16), Cl(17), Br(35), I(53)
and all transition metals (not included here).

For elements WITH d-orbitals: n_basis=9, has_d=True
For elements WITHOUT: n_basis=4 (or 1 for H), has_d=False
"""
from __future__ import annotations

from .params import ElementParams, _EISOL_COEFFICIENTS, _compute_eisol
from typing import Dict


PM6_FULL_PARAMS: Dict[int, ElementParams] = {
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
    # --- Elements WITH d-orbitals ---
    15: ElementParams(
        Z=15, symbol="P", n_basis=9, n_valence=5, eheat=75.42,
        Uss=-48.72991, Upp=-40.35469,
        zeta_s=2.15803, zeta_p=1.80534,
        beta_s=-14.58378, beta_p=-11.74473,
        gss=8.75886, gsp=8.48368, gpp=8.66275, gp2=7.73426, hsp=0.87168,
        alpha=1.94053,
        gauss_K=[-0.03432, 0.0, 0.0, 0.0],
        gauss_L=[6.00139, 0.0, 0.0, 0.0],
        gauss_M=[2.29674, 0.0, 0.0, 0.0],
        Udd=-7.34925, zeta_d=1.23036, beta_d=-20.09989,
        F0SD=0.0, G2SD=0.0, has_d=True,
    ),
    16: ElementParams(
        Z=16, symbol="S", n_basis=9, n_valence=6, eheat=66.4,
        Uss=-47.530706, Upp=-39.191045,
        zeta_s=2.192844, zeta_p=1.841078,
        beta_s=-13.82744, beta_p=-7.66461,
        gss=9.17035, gsp=5.944296, gpp=8.165473, gp2=7.301878, hsp=5.005404,
        alpha=2.26971,
        gauss_K=[-0.036928, 0.0, 0.0, 0.0],
        gauss_L=[1.795067, 0.0, 0.0, 0.0],
        gauss_M=[2.082618, 0.0, 0.0, 0.0],
        Udd=-46.306944, zeta_d=3.109401, beta_d=-9.986172,
        F0SD=0.0, G2SD=0.0, has_d=True,
    ),
    17: ElementParams(
        Z=17, symbol="Cl", n_basis=9, n_valence=7, eheat=28.99,
        Uss=-61.38993, Upp=-54.4828,
        zeta_s=2.63705, zeta_p=2.11815,
        beta_s=-2.36799, beta_p=-13.80214,
        gss=11.14265, gsp=7.48788, gpp=9.55189, gp2=8.12844, hsp=5.00427,
        alpha=2.5173,
        gauss_K=[-0.01321, 0.0, 0.0, 0.0],
        gauss_L=[3.68702, 0.0, 0.0, 0.0],
        gauss_M=[2.54463, 0.0, 0.0, 0.0],
        Udd=-38.25816, zeta_d=1.32403, beta_d=-4.03775,
        F0SD=0.0, G2SD=0.0, has_d=True,
    ),
    35: ElementParams(
        Z=35, symbol="Br", n_basis=9, n_valence=7, eheat=26.74,
        Uss=-45.83436, Upp=-50.29368,
        zeta_s=4.67068, zeta_p=2.03563,
        beta_s=-32.13166, beta_p=-9.51448,
        gss=7.61679, gsp=5.01042, gpp=9.64922, gp2=8.34379, hsp=4.99655,
        alpha=2.51184,
        gauss_K=[-0.005, 0.0, 0.0, 0.0],
        gauss_L=[6.00129, 0.0, 0.0, 0.0],
        gauss_M=[2.89515, 0.0, 0.0, 0.0],
        Udd=7.08674, zeta_d=1.52103, beta_d=-9.83912,
        F0SD=0.0, G2SD=0.0, has_d=True,
    ),
    53: ElementParams(
        Z=53, symbol="I", n_basis=9, n_valence=7, eheat=25.517,
        Uss=-59.97323, Upp=-56.45983,
        zeta_s=4.49865, zeta_p=1.91707,
        beta_s=-30.52248, beta_p=-5.94212,
        gss=7.23476, gsp=9.15441, gpp=9.87747, gp2=8.03592, hsp=5.00422,
        alpha=1.99019,
        gauss_K=[-0.03552, 0.0, 0.0, 0.0],
        gauss_L=[1.74439, 0.0, 0.0, 0.0],
        gauss_M=[1.22384, 0.0, 0.0, 0.0],
        Udd=-23.45792, zeta_d=2.72301, beta_d=-5.25221,
        F0SD=0.0, G2SD=0.0, has_d=True,
    ),
}

# Compute Eisol
for _z, _p in PM6_FULL_PARAMS.items():
    _p.eisol = _compute_eisol(_p)
