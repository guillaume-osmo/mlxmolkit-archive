"""
COSMO-RS parameters and van der Waals radii.

Sources:
- VdW radii: openCOSMO-RS cpcm_radii.inp (ORCA convention)
- COSMO-RS model: openCOSMO-RS_py parameterization.py (ORCA default + 24a)
- Physical constants: scipy.constants
"""
from __future__ import annotations

import numpy as np

# ================================================================
# Van der Waals radii (Angstrom) from ORCA CPCM
# Used to build the molecular cavity
# ================================================================
VDW_RADII = {
    1: 1.300,   # H
    2: 1.601,   # He
    5: 2.048,   # B
    6: 2.000,   # C
    7: 1.830,   # N
    8: 1.720,   # O
    9: 1.720,   # F
    10: 1.771,  # Ne
    13: 2.152,  # Al
    14: 2.200,  # Si
    15: 2.106,  # P
    16: 2.160,  # S
    17: 2.050,  # Cl
    18: 2.184,  # Ar
    22: 2.261,  # Ti
    26: 2.195,  # Fe
    27: 2.153,  # Co
    31: 2.172,  # Ga
    32: 2.700,  # Ge
    33: 2.148,  # As
    34: 2.200,  # Se
    35: 2.160,  # Br
    36: 2.354,  # Kr
    49: 2.245,  # In
    50: 2.550,  # Sn
    51: 2.402,  # Sb
    53: 2.320,  # I
    54: 2.524,  # Xe
    80: 1.978,  # Hg
    82: 2.360,  # Pb
    86: 2.573,  # Rn
}

# Cavity scaling factor (COSMO convention: scale VdW radii by this)
CAVITY_SCALING = 1.2

# ================================================================
# COSMO-RS model parameters (openCOSMO-RS ORCA preset)
# ================================================================
# Effective contact area
A_EFF = 6.226  # Angstrom^2

# Sigma averaging radius
R_AV = 0.5  # Angstrom

# Misfit energy parameters
# NOTE: For semi-empirical (RM1/PM3) charges, use MF_ALPHA_SE which is ~10x the
# DFT value to compensate for less polarized Mulliken charges.
MF_ALPHA_DFT = 7.579075e6   # J/(mol * Angstrom^2 * e^2) — for DFT/ORCA COSMO
MF_ALPHA_SIMPLE = 7.579075e7  # Tuned for simple COSMO + PM3 charges
MF_ALPHA_DDCOSMO = 1.0e8      # Tuned for ddCOSMO direct + PM3 charges
MF_ALPHA_PM6 = 3.2e7          # Tuned for PM6 (more polarized, lower alpha)
MF_ALPHA_SH4 = 5.0e6          # Tuned for ddCOSMO SH lmax=4 (sigma ~3x larger)
MF_ALPHA_SH6 = 1.5e6          # Tuned for ddCOSMO SH lmax=6 (sigma ~5x larger)
MF_ALPHA = 1.0e8             # Default: ddCOSMO direct + RM1/PM3
MF_F_CORR = 2.4             # sigma_orth correction factor
MF_R_AV_CORR = 1.0          # averaging radius correction

# Hydrogen bonding parameters
HB_C = 2.7488747e7     # J/(mol * Angstrom^2 * e^2)
HB_C_T = 1.5           # temperature dependence factor
HB_SIGMA_THRESH = 7.686e-3  # e/Angstrom^2

# Combinatorial (Staverman-Guggenheim) parameters
COMB_SG_Z_COORD = 10.0     # coordination number / 2
COMB_SG_A_STD = 47.999      # standard segment area (Angstrom^2)

# Sigma profile grid
SIGMA_GRID_MIN = -0.15  # e/Angstrom^2
SIGMA_GRID_MAX = 0.15
SIGMA_GRID_STEP = 0.001
SIGMA_GRID = np.arange(SIGMA_GRID_MIN, SIGMA_GRID_MAX + SIGMA_GRID_STEP / 2, SIGMA_GRID_STEP)

# Dielectric constant for water (default solvent)
EPSILON_WATER = 78.39

# Physical constants
R_GAS = 8.314462  # J/(mol*K)
BOHR_TO_ANG = 0.529177  # Bohr to Angstrom
HARTREE_TO_KJMOL = 2625.5  # Hartree to kJ/mol
EV_TO_KJMOL = 96.485  # eV to kJ/mol

# H-bond donor/acceptor element classification
# In COSMO-RS, H atoms bonded to electronegative atoms (N, O, F) are HB donors
# Electronegative atoms (N, O, F, S) with lone pairs are HB acceptors
HB_DONOR_ELEMENTS = {1}      # H (when bonded to N, O, F)
HB_ACCEPTOR_ELEMENTS = {7, 8, 9, 16}  # N, O, F, S
