"""PM6-D3H4 post-SCF correction.

Implements Grimme's D3 dispersion (zero-damping variant as used by MOPAC's
PM6-D3H4) and Rezáč & Hobza's H4 hydrogen-bond correction, plus a
short-range H-H repulsion poly term. Port of MOPAC's
``src/corrections/dftd3.F90`` + ``dftd3_bits.F90`` + ``H_bonds4.F90``
(Apache-2.0, github.com/openmopac/MOPAC).

Both corrections are **post-SCF and density-independent** — they're added
to the total energy without entering the Fock matrix or affecting Mulliken
charges. The SCF density is the same as plain PM6_D.

References
----------
- Grimme, S. et al. (2010), J. Chem. Phys. 132, 154104 — D3 dispersion.
- Rezáč, J. & Hobza, P. (2012), J. Chem. Theory Comput. 8, 141-151 — H4 + HH-rep.

MOPAC parameters for PM6-D3H4 (from parameters_for_PM6_C.F90 lines 7-11):
    s6 = 0.880, alp = 22.0, rs6 = 1.180, s8 = 0.0  (E8 term disabled)

Tables
------
- ``c6ab[94, 94, 5, 5, 3]``     — D3 C6 reference values (loaded from data/c6ab_d3.npz)
- ``r0ab[94, 94]``              — D3 cutoff radii in Bohr (data/r0ab_d3.npz)
- ``RCOV[94]``                  — Pauling covalent radii in Å (hardcoded)
- ``R2R4[94]``                  — PBE0/def2-QZVP <r²>/<r⁴> (hardcoded)
- ``COVALENT_RADII_H4[118]``    — Rezáč's covalent radii for H4 valence (hardcoded)
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOHR = 0.5291772083            # Å per Bohr (MOPAC's a0)
AU_TO_KCAL = 627.5094740631    # 1 Hartree → kcal/mol
KCAL_TO_EV = 1.0 / 23.060547830619
AU_TO_EV = AU_TO_KCAL * KCAL_TO_EV  # ≈ 27.211

# PM6-D3H4 dispersion parameters (MOPAC parameters_for_PM6 v_par6(7..11))
PM6_D3H4_DISP = dict(s6=0.880, alp=22.0, rs6=1.180, s8=0.0, rs8=1.0)

# PM6-D3H4 H4 parameters (Rezáč & Hobza 2012)
PM6_D3H4_H4 = dict(
    para_oh_o=2.32, para_oh_n=3.10,
    para_nh_o=1.07, para_nh_n=2.01,
    multiplier_wh_o=0.42, multiplier_nh4=3.61, multiplier_coo=1.41,
)

# Rezáč's HH-rep is also part of PM6-D3H4. Its polynomial coefficients are
# hard-wired in MOPAC's ``poly`` function (H_bonds4.F90 lines 365-395).

# Pauling/Pyykkö covalent radii (Å), 94 elements, from MOPAC dftd3.F90:74-84
RCOV = np.array([
    0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67,
    1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96, 1.76, 1.54,
    1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09,
    1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39,
    1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26,
    1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57,
    1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,
    1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32,
    1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58,
    1.52, 1.53, 1.54, 1.55,
], dtype=np.float64)

# PBE0/def2-QZVP <r²>/<r⁴> atomic values (raw, before sqrt transform).
# MOPAC dftd3.F90:55-69
R2R4_RAW = np.array([
    8.0589,  3.4698, 29.0974, 14.8517, 11.8799,  7.8715,  5.5588,
    4.7566,  3.8025,  3.1036, 26.1552, 17.2304, 17.7210, 12.7442,
    9.5361,  8.1652,  6.7463,  5.6004, 29.2012, 22.3934, 19.0598,
   16.8590, 15.4023, 12.5589, 13.4788, 12.2309, 11.2809, 10.5569,
   10.1428,  9.4907, 13.4606, 10.8544,  8.9386,  8.1350,  7.1251,
    6.1971, 30.0162, 24.4103, 20.3537, 17.4780, 13.5528, 11.8451,
   11.0355, 10.1997,  9.5414,  9.0061,  8.6417,  8.9975, 14.0834,
   11.8333, 10.0179,  9.3844,  8.4110,  7.5152, 32.7622, 27.5708,
   23.1671, 21.6003, 20.9615, 20.4562, 20.1010, 19.7475, 19.4828,
   15.6013, 19.2362, 17.4717, 17.8321, 17.4237, 17.1954, 17.1631,
   14.5716, 15.8758, 13.8989, 12.4834, 11.4421, 10.2671,  8.3549,
    7.8496,  7.3278,  7.4820, 13.5124, 11.6554, 10.0959,  9.7340,
    8.8584,  8.0125, 29.8135, 26.3157, 19.1885, 15.8542, 16.1305,
   15.6161, 15.1226, 16.1576,
], dtype=np.float64)

# Pre-compute the sqrt transform applied by MOPAC at load time:
#   r2r4[i] = sqrt(0.5 * raw_r2r4[i] * sqrt(i+1))   (Fortran 1-based: float(i))
R2R4 = np.sqrt(0.5 * R2R4_RAW * np.sqrt(np.arange(1, len(R2R4_RAW) + 1, dtype=np.float64)))

# Rezáč's covalent radii for the H4 valence calculation (Å, 118 elements)
# from H_bonds4.F90:17-27
COVALENT_RADII_H4 = np.array([
    0.37, 0.32, 1.34, 0.90, 0.82, 0.77, 0.75, 0.73, 0.71, 0.69, 1.54, 1.30,
    1.18, 1.11, 1.06, 1.02, 0.99, 0.97, 1.96, 1.74, 1.44, 1.36, 1.25, 1.27,
    1.39, 1.25, 1.26, 1.21, 1.38, 1.31, 1.26, 1.22, 1.19, 1.16, 1.14, 1.10,
    2.11, 1.92, 1.62, 1.48, 1.37, 1.45, 1.56, 1.26, 1.35, 1.31, 1.53, 1.48,
    1.44, 1.41, 1.38, 1.35, 1.33, 1.30, 2.25, 1.98, 1.69, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.60, 1.50,
    1.38, 1.46, 1.59, 1.28, 1.37, 1.28, 1.44, 1.49, 0.00, 0.00, 1.46, 0.00,
    0.00, 1.45, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# C6 and r0ab table loaders (cached)
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@lru_cache(maxsize=1)
def _load_c6ab():
    """Return (c6ab, mxc) loaded from the bundled npz."""
    d = np.load(os.path.join(_DATA_DIR, "c6ab_d3.npz"))
    return d["c6ab"].astype(np.float64), d["mxc"].astype(np.int64)


@lru_cache(maxsize=1)
def _load_r0ab():
    """Return r0ab[94, 94] in Bohr."""
    d = np.load(os.path.join(_DATA_DIR, "r0ab_d3.npz"))
    return d["r0ab"].astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pauling_coordination(atoms, coords_bohr, rcov_bohr):
    """Compute fractional coordination number for each atom (Pauling-style).

    ``cn[i] = Σ_{j≠i} 1 / (1 + exp(-16 * ((rcov_i + rcov_j) / r_ij - 1)))``

    This is the ncoord routine from Grimme's D3 paper.
    """
    n = len(atoms)
    cn = np.zeros(n, dtype=np.float64)
    k1 = 16.0
    for i in range(n):
        rcov_i = rcov_bohr[atoms[i] - 1]
        for j in range(n):
            if i == j:
                continue
            rij = float(np.linalg.norm(coords_bohr[j] - coords_bohr[i]))
            if rij < 1e-12:
                continue
            rco = rcov_i + rcov_bohr[atoms[j] - 1]
            damp = 1.0 / (1.0 + np.exp(-k1 * (rco / rij - 1.0)))
            cn[i] += damp
    return cn


def _getc6(iat, jat, cn_i, cn_j, c6ab, mxc):
    """Interpolate the (i, j) C6 from the reference table via Gaussian weights.

    Mirrors MOPAC ``getc6`` exactly.
    """
    c6mem = -1e99
    rsum = 0.0
    csum = 0.0
    for i in range(mxc[iat - 1]):
        for j in range(mxc[jat - 1]):
            c6 = c6ab[iat - 1, jat - 1, i, j, 0]
            if c6 > 0.0:
                c6mem = c6
                cn1 = c6ab[iat - 1, jat - 1, i, j, 1]
                cn2 = c6ab[iat - 1, jat - 1, i, j, 2]
                r = (cn1 - cn_i) ** 2 + (cn2 - cn_j) ** 2
                w = np.exp(-4.0 * r)
                rsum += w
                csum += w * c6
    if rsum > 0:
        return csum / rsum
    return c6mem


# ---------------------------------------------------------------------------
# D3 dispersion (zero-damping)
# ---------------------------------------------------------------------------

def d3_energy(atoms, coords, params=None, rthr=15.0):
    """Compute D3 dispersion energy (kcal/mol).

    Parameters
    ----------
    atoms : sequence[int]
        Atomic numbers (1 ≤ Z ≤ 94).
    coords : array (N, 3)
        Cartesian coordinates in Å.
    params : dict
        ``{s6, alp, rs6, s8, rs8}``. Defaults to PM6-D3H4 values.
    rthr : float
        Pairwise cutoff in Å (MOPAC default 15.0).

    Returns
    -------
    dict with keys 'e6', 'e8', 'e_disp' (all in kcal/mol).
    """
    if params is None:
        params = PM6_D3H4_DISP
    s6 = params['s6']; alp = params['alp']
    rs6 = params['rs6']; s8 = params['s8']; rs8 = params['rs8']
    alp6 = alp; alp8 = alp + 2.0

    coords = np.asarray(coords, dtype=np.float64)
    n = len(atoms)
    if n < 2:
        return {'e6': 0.0, 'e8': 0.0, 'e_disp': 0.0}

    # Convert to Bohr for internal calc (MOPAC does the same)
    coords_bohr = coords / BOHR
    rthr_bohr = rthr / BOHR
    rthr_bohr_sq = rthr_bohr ** 2
    rcov_bohr = 4.0 / 3.0 * RCOV / BOHR  # MOPAC's scaling

    c6ab, mxc = _load_c6ab()
    r0ab = _load_r0ab()

    cn = _pauling_coordination(atoms, coords_bohr, rcov_bohr)

    e6 = 0.0
    e8 = 0.0
    for i in range(n - 1):
        zi = atoms[i]
        for j in range(i + 1, n):
            zj = atoms[j]
            rij = float(np.linalg.norm(coords_bohr[j] - coords_bohr[i]))
            rij2 = rij * rij
            if rij2 > rthr_bohr_sq:
                continue
            rr = r0ab[zj - 1, zi - 1] / rij
            tmp6 = rs6 * rr
            damp6 = 1.0 / (1.0 + 6.0 * tmp6 ** alp6)
            tmp8 = rs8 * rr
            damp8 = 1.0 / (1.0 + 6.0 * tmp8 ** alp8)
            c6 = _getc6(zi, zj, cn[i], cn[j], c6ab, mxc)
            r6 = rij2 ** 3
            r8 = r6 * rij2
            c8 = 3.0 * c6 * R2R4[zi - 1] * R2R4[zj - 1]
            e6 += c6 * damp6 / r6
            e8 += c8 * damp8 / r8

    e6 *= -s6
    e8 *= -s8
    e_disp = (e6 + e8) * AU_TO_KCAL
    return {
        'e6': e6 * AU_TO_KCAL,
        'e8': e8 * AU_TO_KCAL,
        'e_disp': e_disp,
    }


# ---------------------------------------------------------------------------
# H4 hydrogen bond correction (Rezáč-Hobza 2012)
# ---------------------------------------------------------------------------

def _cvalence_contribution(atoms, coords_ang, atom_a, atom_b):
    """Valence contribution of (atom_a, atom_b) bond — Rezáč's smooth step.

    Returns 1 if r < r0 (covalent), 0 if r > r0*1.6, smooth interpolation
    between. r0 = sum of Rezáč's covalent radii.
    """
    ri = COVALENT_RADII_H4[atoms[atom_a] - 1]
    rj = COVALENT_RADII_H4[atoms[atom_b] - 1]
    r0 = ri + rj
    r1 = r0 * 1.6
    r = float(np.linalg.norm(coords_ang[atom_b] - coords_ang[atom_a]))
    if r == 0.0 or r >= r1:
        return 0.0
    if r <= r0:
        return 1.0
    x = (r - r0) / (r1 - r0)
    return 1.0 - (-20.0 * x ** 7 + 70.0 * x ** 6 - 84.0 * x ** 5 + 35.0 * x ** 4)


def _h_bonds4_triple(atoms, coords_ang, h_i, atom_i, atom_j, params):
    """Compute the H4 correction for one (atom_i — h_i — atom_j) triple.

    Returns energy in kcal/mol (negative means stabilizing).
    """
    coord_i = coords_ang[atom_i]
    coord_j = coords_ang[atom_j]
    coord_h = coords_ang[h_i]
    rda = float(np.linalg.norm(coord_j - coord_i))
    rih = float(np.linalg.norm(coord_h - coord_i))
    rjh = float(np.linalg.norm(coord_h - coord_j))
    # D-H...A angle (apex at H)
    v1 = coord_i - coord_h
    v2 = coord_j - coord_h
    cos_a = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
    cos_a = max(min(cos_a, 1.0), -1.0)
    angle = np.arccos(cos_a)
    angle = np.pi - angle
    if angle >= np.pi / 2.0:
        return 0.0

    # Donor = the heavy atom closer to H
    if rih < rjh:
        d_i, a_i = atom_i, atom_j
        rdh, rah = rih, rjh
    else:
        d_i, a_i = atom_j, atom_i
        rdh, rah = rjh, rih

    # Radial term (7th-order polynomial; min at rda = 5.5 Å)
    e_radial = (
        -0.00303407407407313510 * rda ** 7
        + 0.07357629629627092382 * rda ** 6
        - 0.70087111111082800452 * rda ** 5
        + 3.25309629629461749545 * rda ** 4
        - 7.20687407406838786983 * rda ** 3
        + 5.31754666665572184314 * rda ** 2
        + 3.40736000001102778967 * rda
        - 4.68512000000450434811
    )

    # Angular term: 1 − x² where x is 7th-order in (angle / π/2)
    a = angle / (np.pi / 2.0)
    x = -20.0 * a ** 7 + 70.0 * a ** 6 - 84.0 * a ** 5 + 35.0 * a ** 4
    e_angular = 1.0 - x * x

    # Donor/acceptor type parameter
    zd, za = atoms[d_i], atoms[a_i]
    e_para = 0.0
    if zd == 8 and za == 8: e_para = params['para_oh_o']
    elif zd == 8 and za == 7: e_para = params['para_oh_n']
    elif zd == 7 and za == 8: e_para = params['para_nh_o']
    elif zd == 7 and za == 7: e_para = params['para_nh_n']
    if e_para == 0.0:
        return 0.0

    # Bond switching (suppress for D-H > 1.15 Å)
    if rdh > 1.15:
        rdhs = rdh - 1.15
        ravgs = 0.5 * rdh + 0.5 * rah - 1.15
        x = rdhs / max(ravgs, 1e-12)
        e_bond_switch = 1.0 - (
            -20.0 * x ** 7 + 70.0 * x ** 6 - 84.0 * x ** 5 + 35.0 * x ** 4
        )
    else:
        e_bond_switch = 1.0

    # Water scaling (O-O donors only)
    e_scale_w = 1.0
    if zd == 8 and za == 8:
        hydrogens = sum(
            _cvalence_contribution(atoms, coords_ang, d_i, k)
            for k in range(len(atoms)) if atoms[k] == 1
        )
        others = sum(
            _cvalence_contribution(atoms, coords_ang, d_i, k)
            for k in range(len(atoms)) if atoms[k] != 1
        )
        if hydrogens >= 1.0:
            slope = params['multiplier_wh_o'] - 1.0
            v = hydrogens
            fv = 0.0
            if 1.0 < v <= 2.0: fv = v - 1.0
            if 2.0 < v < 3.0: fv = 3.0 - v
            fv2 = max(1.0 - others, 0.0)
            e_scale_w = 1.0 + slope * fv * fv2

    # NH4+ donor scaling
    e_scale_chd = 1.0
    if zd == 7:
        slope = params['multiplier_nh4'] - 1.0
        v = sum(_cvalence_contribution(atoms, coords_ang, d_i, k)
                for k in range(len(atoms)))
        v = (v - 3.0) if v > 3.0 else 0.0
        e_scale_chd = 1.0 + slope * v

    # COO⁻ acceptor scaling (O attached to a C that has another O)
    e_scale_cha = 1.0
    if za == 8:
        slope = params['multiplier_coo'] - 1.0
        # Search closest C bonded to a_i
        cdist = float('inf')
        cv_o1 = 0.0
        cc = -1
        for k in range(len(atoms)):
            v = _cvalence_contribution(atoms, coords_ang, a_i, k)
            cv_o1 += v
            if v > 0.0 and atoms[k] == 6:
                d = float(np.linalg.norm(coords_ang[k] - coords_ang[a_i]))
                if d < cdist:
                    cdist = d; cc = k
        if cc != -1:
            # Find the second O bonded to that C
            odist = float('inf')
            cv_cc = 0.0
            o2 = -1
            for k in range(len(atoms)):
                v = _cvalence_contribution(atoms, coords_ang, cc, k)
                cv_cc += v
                if v > 0.0 and k != a_i and atoms[k] == 8:
                    d = float(np.linalg.norm(coords_ang[k] - coords_ang[cc]))
                    if d < odist:
                        odist = d; o2 = k
            if o2 != -1:
                cv_o2 = sum(_cvalence_contribution(atoms, coords_ang, o2, k)
                            for k in range(len(atoms)))
                f_o1 = max(1.0 - abs(1.0 - cv_o1), 0.0)
                f_o2 = max(1.0 - abs(1.0 - cv_o2), 0.0)
                f_cc = max(1.0 - abs(3.0 - cv_cc), 0.0)
                e_scale_cha = 1.0 + slope * f_o1 * f_o2 * f_cc

    return (e_para * e_radial * e_angular * e_bond_switch
            * e_scale_w * e_scale_chd * e_scale_cha)


def h4_energy(atoms, coords, params=None):
    """Compute H4 hydrogen-bond correction energy (kcal/mol).

    Loops over every (D, A, H) triple where:
      - H is a hydrogen atom
      - D, A are N/O heavy atoms (the e_para table is 0 for other types)
      - the D-H...A angle is < 90° (cosine > 0)

    Returns
    -------
    e_hb : float (kcal/mol)
    """
    if params is None:
        params = PM6_D3H4_H4
    coords = np.asarray(coords, dtype=np.float64)
    e_hb = 0.0
    n = len(atoms)
    h_indices = [k for k in range(n) if atoms[k] == 1]
    heavy_no = [k for k in range(n) if atoms[k] in (7, 8)]
    for h in h_indices:
        for ai in heavy_no:
            for aj in heavy_no:
                if ai >= aj:
                    continue
                e_hb += _h_bonds4_triple(atoms, coords, h, ai, aj, params)
    return e_hb


# ---------------------------------------------------------------------------
# Short-range H-H repulsion (Rezáč)
# ---------------------------------------------------------------------------

def _poly_hh(r):
    """Pairwise H-H repulsion polynomial in Å. Returns kcal/mol-ish unit.

    Matches MOPAC ``poly`` (H_bonds4.F90:365-395). The actual code uses
    these values directly as energy contributions (already in kcal/mol).
    """
    if r <= 1.0:
        return 25.46293603147693
    if r < 1.5:
        return (
            -2714.952351603469651 * r ** 5
            + 17103.650110591705015 * r ** 4
            - 42511.857982217959943 * r ** 3
            + 52063.196799138342612 * r ** 2
            - 31430.658335972289933 * r
            + 7516.084696095140316
        )
    return 118.7326 * np.exp(-1.53965 * (r ** 1.72905))


def hh_repulsion(atoms, coords):
    """Total H-H repulsion energy (kcal/mol) summed over all H-H pairs."""
    coords = np.asarray(coords, dtype=np.float64)
    h_idx = [k for k in range(len(atoms)) if atoms[k] == 1]
    e = 0.0
    for ii, i in enumerate(h_idx):
        for j in h_idx[:ii]:
            r = float(np.linalg.norm(coords[i] - coords[j]))
            e += _poly_hh(r)
    return e


# ---------------------------------------------------------------------------
# Combined PM6-D3H4 correction
# ---------------------------------------------------------------------------

def pm6_d3h4_correction(atoms, coords):
    """Return the full PM6-D3H4 post-SCF correction.

    Parameters
    ----------
    atoms : sequence[int]
        Atomic numbers.
    coords : array (N, 3)
        Coordinates in Å.

    Returns
    -------
    dict with:
      - 'e_disp' : D3 dispersion (kcal/mol, negative)
      - 'e_hb'   : H4 hydrogen-bond correction (kcal/mol, usually negative)
      - 'e_hh'   : H-H repulsion (kcal/mol, positive)
      - 'e_total': sum, ready to add to PM6 SCF total energy
    """
    disp = d3_energy(atoms, coords, params=PM6_D3H4_DISP)
    e_hb = h4_energy(atoms, coords, params=PM6_D3H4_H4)
    e_hh = hh_repulsion(atoms, coords)
    return {
        'e_disp': disp['e_disp'],
        'e_hb': e_hb,
        'e_hh': e_hh,
        'e_total': disp['e_disp'] + e_hb + e_hh,
    }
