"""Atomic heats of formation, from openMOPAC.

MOPAC's ``src/models/parameters_C.F90`` ``data eheat(Z)`` table, in kcal/mol.
This is the experimental heat of formation of the gas-phase atom, added once per
atom in the heat-of-formation assembly:

    HoF = (E_total - sum eisol) * 23.06 + sum eheat

Unlike ``eisol`` — which MOPAC *computes* from the atomic parameters in
``calpar.F90``, and which our tables already reproduce exactly — this one is
purely experimental data and has to be transcribed.

Elements added to PM6 from the bundled MOPAC CSV used to carry ``eheat=0.0``, a
deliberate placeholder documented as "charge-only". That is fine for Mulliken
charges, which never touch ``eheat``, but it makes the heat of formation
silently wrong by the atom's whole heat of formation — 108.4 kcal/mol for a
single silicon, 135.7 for a boron. Silently, because nothing raises: the SCF,
the charges and the total energy are all unaffected.
"""
from __future__ import annotations

EHEAT_MOPAC: dict[int, float] = {
    1: 52.102,      # H
    2: 0.000,       # He
    3: 38.410,      # Li
    4: 76.960,      # Be
    5: 135.700,     # B
    6: 170.890,     # C
    7: 113.000,     # N
    8: 59.559,      # O
    9: 18.890,      # F
    10: 0.000,      # Ne
    11: 25.650,     # Na
    12: 35.000,     # Mg
    13: 79.490,     # Al
    14: 108.390,    # Si
    15: 75.570,     # P
    16: 66.400,     # S
    17: 28.990,     # Cl
    18: 0.000,      # Ar
    19: 21.420,     # K
    20: 42.600,     # Ca
    21: 90.300,     # Sc
    22: 112.300,    # Ti
    23: 122.900,    # V
    24: 95.000,     # Cr
    25: 67.700,     # Mn
    26: 99.300,     # Fe
    27: 102.400,    # Co
    28: 102.800,    # Ni
    29: 80.700,     # Cu
    30: 31.170,     # Zn
    31: 65.400,     # Ga
    32: 89.500,     # Ge
    33: 72.300,     # As
    34: 54.300,     # Se
    35: 26.740,     # Br
    36: 0.000,      # Kr
    37: 19.600,     # Rb
    38: 39.100,     # Sr
    39: 101.500,    # Y
    40: 145.500,    # Zr
    41: 172.400,    # Nb
    42: 157.300,    # Mo
    43: 162.000,    # Tc
    44: 155.500,    # Ru
    45: 133.000,    # Rh
    46: 90.000,     # Pd
    47: 68.100,     # Ag
    48: 26.720,     # Cd
    49: 58.000,     # In
    50: 72.200,     # Sn
    51: 63.200,     # Sb
    52: 47.000,     # Te
    53: 25.517,     # I
    54: 0.000,      # Xe
    55: 18.700,     # Cs
    56: 42.500,     # Ba
    57: 103.011,    # ?
    58: 101.004,    # ?
    59: 84.990,     # ?
    60: 78.298,     # ?
    61: 83.174,     # ?
    62: 49.402,     # ?
    63: 41.898,     # ?
    64: 95.007,     # ?
    65: 92.902,     # ?
    66: 69.407,     # ?
    67: 71.893,     # ?
    68: 75.791,     # ?
    69: 55.500,     # ?
    70: 36.358,     # ?
    71: 102.199,    # ?
    72: 148.000,    # ?
    73: 186.900,    # ?
    74: 203.100,    # ?
    75: 185.000,    # ?
    76: 188.000,    # ?
    77: 160.000,    # ?
    78: 135.200,    # ?
    79: 88.000,     # ?
    80: 14.690,     # Hg
    81: 43.550,     # Tl
    82: 46.620,     # Pb
    83: 50.100,     # Bi
    86: 0.000,      # ?
}


def atomic_heat(z: int) -> float:
    """Heat of formation of the gas-phase atom Z, kcal/mol.

    Returns 0.0 for elements MOPAC does not tabulate, which is what the old
    placeholder did — but now that is the documented answer for an untabulated
    element rather than the default for every element.
    """
    return EHEAT_MOPAC.get(int(z), 0.0)
