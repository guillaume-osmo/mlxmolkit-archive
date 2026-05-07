"""
RM1 semi-empirical parameters extracted from MOPAC (Apache 2.0 license).

Source: openmopac/mopac/src/models/parameters_for_RM1_C.F90
Reference: Rocha et al., J. Comput. Chem. 2006, 27, 1101-1111.

Parameters for: H, C, N, O, F, P, S, Cl, Br, I
Units: eV (energies), Bohr^-1 (orbital exponents), Angstrom^-1 (alpha)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class ElementParams:
    """RM1 parameters for one element."""
    Z: int              # atomic number
    symbol: str
    n_basis: int        # number of basis functions (1 for H, 4 for C/N/O/etc)

    # One-electron one-center integrals (eV)
    Uss: float          # s orbital
    Upp: float          # p orbital (0 for H)

    # Slater orbital exponents (Bohr^-1)
    zeta_s: float
    zeta_p: float       # 0 for H

    # Resonance integrals (eV)
    beta_s: float
    beta_p: float       # 0 for H

    # One-center two-electron integrals (eV)
    gss: float          # (ss|ss)
    gsp: float          # (ss|pp) — 0 for H
    gpp: float          # (pp|pp) — 0 for H
    gp2: float          # (pp'|pp') — 0 for H
    hsp: float          # (sp|sp) — 0 for H

    # Core-core repulsion parameter (Angstrom^-1)
    alpha: float

    # Gaussian correction terms: K, L, M for up to 4 Gaussians
    # E_core_core += Σ K * exp(-L * (R - M)^2)
    gauss_K: list       # [K1, K2, K3, K4]
    gauss_L: list       # [L1, L2, L3, L4]
    gauss_M: list       # [M1, M2, M3, M4]

    # Derived: number of valence electrons
    n_valence: int = 0

    # Derived: electron heat of formation (kcal/mol)
    eheat: float = 0.0

    # Isolated atom electronic energy (eV) — from MOPAC
    # E_isol = Uss * n_s + Upp * n_p + one-center integrals
    eisol: float = 0.0


# RM1 parameters from MOPAC (Apache 2.0)
# Atomic number → ElementParams
RM1_PARAMS: Dict[int, ElementParams] = {
    # Hydrogen (Z=1): 1s → 1 basis function, 1 valence electron
    1: ElementParams(
        Z=1, symbol="H", n_basis=1, n_valence=1, eheat=52.102,
        Uss=-11.9606770, Upp=0.0,
        zeta_s=1.0826737, zeta_p=0.0,
        beta_s=-5.7654447, beta_p=0.0,
        gss=13.9832130, gsp=0.0, gpp=0.0, gp2=0.0, hsp=0.0,
        alpha=3.0683595,
        gauss_K=[0.1028888, 0.0645745, -0.0356739, 0.0],
        gauss_L=[5.9017227, 6.4178567, 2.8047313, 0.0],
        gauss_M=[1.1750118, 1.9384448, 1.6365524, 0.0],
    ),
    # Carbon (Z=6): 2s,2px,2py,2pz → 4 basis functions, 4 valence electrons
    6: ElementParams(
        Z=6, symbol="C", n_basis=4, n_valence=4, eheat=170.89,
        Uss=-51.7255603, Upp=-39.4072894,
        zeta_s=1.8501880, zeta_p=1.7683009,
        beta_s=-15.4593243, beta_p=-8.2360864,
        gss=13.0531244, gsp=11.3347939, gpp=10.9511374, gp2=9.7239510, hsp=1.5521513,
        alpha=2.7928208,
        gauss_K=[0.0746227, 0.0117705, 0.0372066, -0.0027066],
        gauss_L=[5.7392160, 6.9240173, 6.2615894, 9.0000373],
        gauss_M=[1.0439698, 1.6615957, 1.6315872, 2.7955790],
    ),
    # Nitrogen (Z=7): 4 basis functions, 5 valence electrons
    7: ElementParams(
        Z=7, symbol="N", n_basis=4, n_valence=5, eheat=113.00,
        Uss=-70.8512372, Upp=-57.9773092,
        zeta_s=2.3744716, zeta_p=1.9781257,
        beta_s=-20.8712455, beta_p=-16.6717185,
        gss=13.0873623, gsp=13.2122683, gpp=13.6992432, gp2=11.9410395, hsp=5.0000085,
        alpha=2.9642254,
        gauss_K=[0.0607338, 0.0243856, -0.0228343, 0.0],
        gauss_L=[4.5889295, 4.6273052, 2.0527466, 0.0],
        gauss_M=[1.3787388, 2.0837070, 1.8676382, 0.0],
    ),
    # Oxygen (Z=8): 4 basis functions, 6 valence electrons
    8: ElementParams(
        Z=8, symbol="O", n_basis=4, n_valence=6, eheat=59.559,
        Uss=-96.9494807, Upp=-77.8909298,
        zeta_s=3.1793691, zeta_p=2.5536191,
        beta_s=-29.8510121, beta_p=-29.1510131,
        gss=14.0024279, gsp=14.9562504, gpp=14.1451514, gp2=12.7032550, hsp=3.9321716,
        alpha=4.1719672,
        gauss_K=[0.2309355, 0.0585987, 0.0, 0.0],
        gauss_L=[5.2182874, 7.4293293, 0.0, 0.0],
        gauss_M=[0.9036356, 1.5175461, 0.0, 0.0],
    ),
    # Fluorine (Z=9): 4 basis functions, 7 valence electrons
    9: ElementParams(
        Z=9, symbol="F", n_basis=4, n_valence=7, eheat=18.86,
        Uss=-134.1836959, Upp=-107.8466092,
        zeta_s=4.4033791, zeta_p=2.6484156,
        beta_s=-70.0000051, beta_p=-32.6798271,
        gss=16.7209132, gsp=16.7614263, gpp=15.2258103, gp2=14.8657868, hsp=1.9976617,
        alpha=6.0000006,
        gauss_K=[0.4030203, 0.0708583, 0.0, 0.0],
        gauss_L=[7.2044196, 9.0000156, 0.0, 0.0],
        gauss_M=[0.8165301, 1.4380238, 0.0, 0.0],
    ),
    # Phosphorus (Z=15): 4 basis functions, 5 valence electrons
    15: ElementParams(
        Z=15, symbol="P", n_basis=4, n_valence=5, eheat=75.42,
        Uss=-41.8153318, Upp=-34.3834253,
        zeta_s=2.1224012, zeta_p=1.7432795,
        beta_s=-6.1351497, beta_p=-5.9444213,
        gss=11.0805926, gsp=5.6833920, gpp=7.6041756, gp2=7.4026518, hsp=1.1618179,
        alpha=1.9099329,
        gauss_K=[-0.4106347, -0.1629929, -0.0488713, 0.0],
        gauss_L=[6.0875283, 7.0947260, 8.9997931, 0.0],
        gauss_M=[1.3165026, 1.9072132, 2.6585778, 0.0],
    ),
    # Sulfur (Z=16): 4 basis functions, 6 valence electrons
    16: ElementParams(
        Z=16, symbol="S", n_basis=4, n_valence=6, eheat=66.40,
        Uss=-55.1677512, Upp=-46.5293042,
        zeta_s=2.1334431, zeta_p=1.8746065,
        beta_s=-1.9591072, beta_p=-8.7743065,
        gss=12.4882841, gsp=8.5691057, gpp=8.5230117, gp2=7.6686330, hsp=3.8897893,
        alpha=2.4401564,
        gauss_K=[-0.7460106, -0.0651929, -0.0065598, 0.0],
        gauss_L=[4.8103800, 7.2076086, 9.0000018, 0.0],
        gauss_M=[0.5938013, 1.2949201, 1.8006015, 0.0],
    ),
    # Chlorine (Z=17): 4 basis functions, 7 valence electrons
    17: ElementParams(
        Z=17, symbol="Cl", n_basis=4, n_valence=7, eheat=28.99,
        Uss=-118.4730692, Upp=-76.3533034,
        zeta_s=3.8649107, zeta_p=1.8959314,
        beta_s=-19.9243043, beta_p=-11.5293520,
        gss=15.3602310, gsp=13.3067117, gpp=12.5650264, gp2=9.6639708, hsp=1.7648990,
        alpha=3.6935883,
        gauss_K=[0.1294711, 0.0028890, 0.0, 0.0],
        gauss_L=[2.9772442, 7.0982759, 0.0, 0.0],
        gauss_M=[1.4674978, 2.5000272, 0.0, 0.0],
    ),
    # Bromine (Z=35): 4 basis functions, 7 valence electrons
    35: ElementParams(
        Z=35, symbol="Br", n_basis=4, n_valence=7, eheat=26.74,
        Uss=-113.4839818, Upp=-76.1872002,
        zeta_s=5.7315721, zeta_p=2.0314758,
        beta_s=-1.3413984, beta_p=-8.2022599,
        gss=17.1156307, gsp=15.6241925, gpp=10.7354629, gp2=8.8605620, hsp=2.2351276,
        alpha=2.8671053,
        gauss_K=[0.9868994, -0.9273125, 0.0, 0.0],
        gauss_L=[4.2848419, 4.5400591, 0.0, 0.0],
        gauss_M=[2.0001970, 2.0161770, 0.0, 0.0],
    ),
    # Iodine (Z=53): 4 basis functions, 7 valence electrons
    53: ElementParams(
        Z=53, symbol="I", n_basis=4, n_valence=7, eheat=25.517,
        Uss=-74.8999784, Upp=-51.4102380,
        zeta_s=2.5300375, zeta_p=2.3173868,
        beta_s=-4.1931615, beta_p=-4.4003841,
        gss=19.9997413, gsp=7.6895767, gpp=7.3048834, gp2=6.8542461, hsp=1.4160294,
        alpha=2.1415709,
        gauss_K=[-0.0814772, 0.0591499, 0.0, 0.0],
        gauss_L=[1.5606507, 5.7611127, 0.0, 0.0],
        gauss_M=[2.0000206, 2.2048880, 0.0, 0.0],
    ),
}

# Convenience: symbol → Z mapping
SYMBOL_TO_Z = {p.symbol: p.Z for p in RM1_PARAMS.values()}

# Physical constants (matching MOPAC conventions)
EV_TO_KCAL = 23.061      # 1 eV = 23.061 kcal/mol
BOHR_TO_ANG = 0.529167   # 1 bohr = 0.529167 Angstrom
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


# ================================================================
# Compute Eisol for each element using PYSEQM/MOPAC coefficients
# Eisol = ussc*Uss + uppc*Upp + gssc*gss + gppc*gpp + gspc*gsp
#         + gp2c*gp2 + hspc*hsp
# Coefficients from PYSEQM Constants (calpar.f90):
# ================================================================
_EISOL_COEFFICIENTS = {
    # Z: (ussc, uppc, gssc, gppc, gspc, gp2c, hspc)
    1:  (1.0,  0.0,  0.0,   0.0,   0.0,  0.0,   0.0),     # H: 1s^1
    6:  (2.0,  2.0,  1.0,  -0.5,   4.0,  1.5,  -2.0),     # C: 2s^2 2p^2
    7:  (2.0,  3.0,  1.0,  -1.5,   6.0,  4.5,  -3.0),     # N: 2s^2 2p^3
    8:  (2.0,  4.0,  1.0,  -0.5,   8.0,  6.5,  -4.0),     # O: 2s^2 2p^4
    9:  (2.0,  5.0,  1.0,   0.5,  10.0,  9.5,  -5.0),     # F: 2s^2 2p^5
    15: (2.0,  3.0,  1.0,  -1.5,   6.0,  4.5,  -3.0),     # P: 3s^2 3p^3 (same as N)
    16: (2.0,  4.0,  1.0,  -0.5,   8.0,  6.5,  -4.0),     # S: 3s^2 3p^4 (same as O)
    17: (2.0,  5.0,  1.0,   0.5,  10.0,  9.5,  -5.0),     # Cl: 3s^2 3p^5 (same as F)
    35: (2.0,  5.0,  1.0,   0.5,  10.0,  9.5,  -5.0),     # Br: same as Cl
    53: (2.0,  5.0,  1.0,   0.5,  10.0,  9.5,  -5.0),     # I: same as Cl
}


def _compute_eisol(p: ElementParams) -> float:
    """Compute isolated atom electronic energy using MOPAC/PYSEQM coefficients."""
    if p.Z not in _EISOL_COEFFICIENTS:
        return 0.0
    ussc, uppc, gssc, gppc, gspc, gp2c, hspc = _EISOL_COEFFICIENTS[p.Z]
    return (ussc * p.Uss + uppc * p.Upp
            + gssc * p.gss + gppc * p.gpp + gspc * p.gsp
            + gp2c * p.gp2 + hspc * p.hsp)


# Set Eisol for all elements
for _z, _p in RM1_PARAMS.items():
    _p.eisol = _compute_eisol(_p)
