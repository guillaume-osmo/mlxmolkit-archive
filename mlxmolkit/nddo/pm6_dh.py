"""PM6-DH+ post-SCF corrections, ported from openMOPAC.

PM6-DH+ is plain PM6 electronically — same parameters, same SCF, same density
and charges — plus two post-SCF terms that the D3 family does not share:

    correction = PM6_DH_Dispersion(...) + Hydrogen_bond_corrections(...)

(``src/corrections/post_scf_corrections.F90``.)

The dispersion here is **not** D3. It is the older Slater-Kirkwood form with its
own element tables, ported from
``src/corrections/H_bond_correction_PM6_DH_Dispersion.F90``.
"""
from __future__ import annotations

import math

import numpy as np

# --- element tables, verbatim from H_bond_correction_PM6_DH_Dispersion.F90 ---
# C6, in J.nm^6/mol. Zero means "not parameterised"; such pairs are skipped.
_C6 = {1: 0.16, 2: 0.084, 5: 5.79, 6: 1.65, 7: 1.11, 8: 0.70, 9: 0.57,
       10: 0.45, 15: 3.25, 16: 5.79, 17: 5.97, 18: 3.71, 27: 0.04,
       35: 11.60, 36: 4.47, 53: 25.80, 54: 16.50}
# R0, in pm.
_R0 = {1: 156.0, 2: 140.0, 5: 180.0, 6: 170.0, 7: 155.0, 8: 152.0, 9: 147.0,
       10: 154.0, 15: 180.0, 16: 180.0, 17: 175.0, 18: 188.0, 27: 140.0,
       35: 185.0, 36: 202.0, 53: 198.0, 54: 216.0}
# Slater-Kirkwood effective electron numbers.
_NEFF = {1: 0.80, 2: 1.42, 5: 2.16, 6: 2.50, 7: 2.82, 8: 3.15, 9: 3.48,
         10: 3.81, 15: 4.50, 16: 4.80, 17: 5.10, 18: 5.40, 27: 2.90,
         35: 6.00, 36: 6.30, 53: 6.95, 54: 7.25}

# PM6 branch of the alpha/s/cscale selection (PM7 has its own values).
_ALPHA = 20.0
_S = 1.04
_CSCALE = 0.89


def _neighbour_counts(atoms, coords, scale: float = 1.25):
    """Bond counts, standing in for MOPAC's `nbonds`.

    Only carbon's coordination is consulted below, to pick its C6.
    """
    from .pm6_d3h4 import RCOV

    xyz = np.asarray(coords, dtype=np.float64)
    radii = np.array([RCOV[int(z)] if int(z) < len(RCOV) else 1.5 for z in atoms])
    counts = [0] * len(atoms)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if np.linalg.norm(xyz[i] - xyz[j]) < scale * (radii[i] + radii[j]):
                counts[i] += 1
                counts[j] += 1
    return counts


def pm6_dh_dispersion(atoms, coords) -> float:
    """PM6-DH+ dispersion energy, kcal/mol (negative).

    Port of ``PM6_DH_Disp``. The combining rule and damping are

        C6   = 2 (C6i^2 C6j^2 Ni Nj)^(1/3) /
               ((C6i Nj^2)^(1/3) + (C6j Ni^2)^(1/3))
        R0   = (Ri^3 + Rj^3)/(Ri^2 + Rj^2) / 1000 * 2        [pm -> nm]
        damp = 1 / (1 + exp(-alpha (Rij/(s R0) - 1)))
        E   -= C6 / Rij^6 * damp / (1000 * 4.184)            [-> kcal/mol]

    with the whole sum scaled by `cscale` at the end. Distances are in **nm**
    here, not Angstrom or Bohr, and C6 is in J.nm^6/mol — the units are baked
    into the two divisions above and do not survive being "tidied up".

    Carbon takes a coordination-dependent C6 — 0.95 with four bonds, 1.65
    otherwise — rather than its table value, which is MOPAC's way of
    distinguishing sp3 from sp2/sp carbon. Every other element uses the table.
    """
    z = [int(a) for a in atoms]
    xyz = np.asarray(coords, dtype=np.float64)
    counts = _neighbour_counts(z, xyz)
    total = 0.0
    for i in range(len(z)):
        zi = z[i]
        if zi > 86:
            continue
        c6i = (0.95 if counts[i] == 4 else 1.65) if zi == 6 else _C6.get(zi, 0.0)
        ni = _NEFF.get(zi, 0.0)
        ri = _R0.get(zi, 0.0)
        for j in range(i + 1, len(z)):
            zj = z[j]
            if zj > 86:
                continue
            c6j = (0.95 if counts[j] == 4 else 1.65) if zj == 6 else _C6.get(zj, 0.0)
            nj = _NEFF.get(zj, 0.0)
            rj = _R0.get(zj, 0.0)
            # MOPAC skips a pair when the *second* atom is unparameterised.
            if rj == 0.0 or _C6.get(zj, 0.0) == 0.0 or nj == 0.0:
                continue
            if ri == 0.0 or ni == 0.0:
                continue
            c6 = (2.0 * (c6i ** 2 * c6j ** 2 * ni * nj) ** (1.0 / 3.0)
                  / ((c6i * nj ** 2) ** (1.0 / 3.0)
                     + (c6j * ni ** 2) ** (1.0 / 3.0)))
            r0 = (ri ** 3 + rj ** 3) / (ri ** 2 + rj ** 2) / 1000.0 * 2.0
            rij = float(np.linalg.norm(xyz[j] - xyz[i])) * 0.1      # A -> nm
            damp = 1.0 / (1.0 + math.exp(-_ALPHA * (rij / (_S * r0) - 1.0)))
            total -= c6 / rij ** 6 * damp / (1000.0 * 4.184)
    return total * _CSCALE
