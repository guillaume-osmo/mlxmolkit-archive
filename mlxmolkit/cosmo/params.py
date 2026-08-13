"""
COSMO-RS parameters and van der Waals radii.

Sources:
- VdW radii: openCOSMO-RS cpcm_radii.inp (ORCA convention)
- COSMO-RS model: openCOSMO-RS_py parameterization.py (ORCA default + 24a)
- Physical constants: scipy.constants
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# van der Waals radii (Angstrom), keyed by atomic number.
# ---------------------------------------------------------------------------
VDW_RADII = {
    1: 1.3,
    2: 1.601,
    5: 2.048,
    6: 2.0,
    7: 1.83,
    8: 1.72,
    9: 1.72,
    10: 1.771,
    13: 2.152,
    14: 2.2,
    15: 2.106,
    16: 2.16,
    17: 2.05,
    18: 2.184,
    22: 2.261,
    26: 2.195,
    27: 2.153,
    31: 2.172,
    32: 2.7,
    33: 2.148,
    34: 2.2,
    35: 2.16,
    36: 2.354,
    49: 2.245,
    50: 2.55,
    51: 2.402,
    53: 2.32,
    54: 2.524,
    80: 1.978,
    82: 2.36,
    86: 2.573,
}

# Cavity radii are scaled vdW radii.
CAVITY_SCALING = 1.2

# ---------------------------------------------------------------------------
# COSMO-RS segment parameters
# ---------------------------------------------------------------------------
# Effective contact area of a standard surface segment (Angstrom^2).
A_EFF = 6.226

# Averaging radius used when smoothing sigma over the surface (Angstrom).
R_AV = 0.5

# ---------------------------------------------------------------------------
# Misfit prefactor alpha', per charge source / segment descriptor.
# ---------------------------------------------------------------------------
MF_ALPHA_DFT = 7579075.0
MF_ALPHA_SIMPLE = 75790750.0
MF_ALPHA_DDCOSMO = 100000000.0
MF_ALPHA_PM6 = 32000000.0
MF_ALPHA_SH4 = 5000000.0
MF_ALPHA_SH6 = 1500000.0
MF_ALPHA = 100000000.0
MF_F_CORR = 2.4
MF_R_AV_CORR = 1.0

# Hydrogen bonding
HB_C = 27488747.0
HB_C_T = 1.5
HB_SIGMA_THRESH = 0.007686

# Combinatorial (Staverman-Guggenheim) contribution
COMB_SG_Z_COORD = 10.0
COMB_SG_A_STD = 47.999

# Sigma profile grid (e / Angstrom^2)
SIGMA_GRID_MIN = -0.15
SIGMA_GRID_MAX = 0.15
SIGMA_GRID_STEP = 0.001
SIGMA_GRID = np.arange(SIGMA_GRID_MIN, SIGMA_GRID_MAX + SIGMA_GRID_STEP / 2, SIGMA_GRID_STEP)

# Dielectric constant of water at 298.15 K
EPSILON_WATER = 78.39

# Physical constants / unit conversions
R_GAS = 8.314462
BOHR_TO_ANG = 0.529177
HARTREE_TO_KJMOL = 2625.5
EV_TO_KJMOL = 96.485

# ---------------------------------------------------------------------------
# Hydrogen bonding element classification
# ---------------------------------------------------------------------------
HB_DONOR_ELEMENTS = {1}
HB_ACCEPTOR_ELEMENTS = {7, 8, 9, 16}
