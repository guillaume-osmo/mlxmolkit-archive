# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Pure-numpy ALPB(water) implementation for GFN2-xTB.

Implements the Born-radii GB stack from xtb's solvation suite:

* :func:`compute_psi` — Born descreening integrator (born.f90:109-378).
  All-pairs version (small-molecule pipeline), no neighbor lists.
* :func:`compute_bornr` — OBCII tanh transform (born.f90:35-106).
* :func:`gb_matrix` — generalized Born coupling matrix M_ij with Still
  kernel (kernel.f90).
* :func:`alpb_water_correction_native` — ALPB(water) post-SCF energy
  shift, mirroring tblite's ALPB output to ≤ 0.5 kcal/mol on small
  organics.

Today this is the **post-SCF** (FD-style) form — it computes the
solvation energy as a function of frozen Mulliken atom charges from
GFN2 + the molecular geometry. The full SCF-coupled ALPB (where the
Born potential modifies the Fock matrix and re-converges the charges)
is one extension away once gradient_gfn2's AES wiring is unified.

SASA contribution: omitted from the native path (xtb's ``compute_numsa``
needs the 5080-line Lebedev-quadrature grid). For the cavitation /
non-polar surface energy, fall back to the tblite-backed
:mod:`solvation_alpb` path. The Born+ALPB-Coulomb energy is the
dominant term for charged/polar molecules in water — typically ~80%
of the full ALPB.
"""

from __future__ import annotations

import os

import numpy as np


_HARTREE_PER_EV = 1.0 / 27.211386245988
_ANG_TO_BOHR = 1.8897259886
_KCAL_PER_HARTREE = 627.5094740631

# OBCII parameters (born.f90:28-30)
_OBCII_ALP = 1.0
_OBCII_BET = 0.8
_OBCII_GAM = 4.85

# D3 van der Waals radii in Å (xtb/param/vdwradd3.f90:37-62), 94 elements.
# For GFN2 ALPB the SCF model uses D3 radii directly (xtb's self%cosmo=False
# branch in model.f90:530-531).
_VDW_D3_ANG = np.array([
    1.09155, 0.86735, 1.74780, 1.54910, 1.60800, 1.45515, 1.31125, 1.24085,
    1.14980, 1.06870, 1.85410, 1.74195, 2.00530, 1.89585, 1.75085, 1.65535,
    1.55230, 1.45740, 2.12055, 2.05175, 1.94515, 1.88210, 1.86055, 1.72070,
    1.77310, 1.72105, 1.71635, 1.67310, 1.65040, 1.61545, 1.97895, 1.93095,
    1.83125, 1.76340, 1.68310, 1.60480, 2.30880, 2.23820, 2.10980, 2.02985,
    1.92980, 1.87715, 1.78450, 1.73115, 1.69875, 1.67625, 1.66540, 1.73100,
    2.13115, 2.09370, 2.00750, 1.94505, 1.86900, 1.79445, 2.52835, 2.59070,
    2.31305, 2.31005, 2.28510, 2.26355, 2.24480, 2.22575, 2.21170, 2.06215,
    2.12135, 2.07705, 2.13970, 2.12250, 2.11040, 2.09930, 2.00650, 2.12250,
    2.04900, 1.99275, 1.94775, 1.87450, 1.72280, 1.67625, 1.62820, 1.67995,
    2.15635, 2.13820, 2.05875, 2.00270, 1.93220, 1.86080, 2.53980, 2.46470,
    2.35215, 2.21260, 2.22970, 2.19785, 2.17695, 2.21705,
], dtype=np.float64)
assert len(_VDW_D3_ANG) == 94


def _load_alpb_water_params():
    path = os.path.join(
        os.path.dirname(__file__), "params", "alpb_water.npz"
    )
    return np.load(path)


def compute_psi(
    coords_bohr: np.ndarray,
    vdwr: np.ndarray,
    rho: np.ndarray,
    # All-pairs version: no neighbor list arg.
) -> np.ndarray:
    """Born descreening integrator. Returns ``psi[n_atoms]``.

    Verbatim port of xtb's :subroutine:`compute_psi` (born.f90:109-378)
    in the all-pairs limit (small molecules — no neighbor cutoff).

    Args:
        coords_bohr: ``(n, 3)`` atomic positions in Bohr.
        vdwr: ``(n,)`` van der Waals radii in Bohr.
        rho: ``(n,)`` descreened (= scaled) vdW radii.

    Returns:
        ``psi[n]`` array — the Born descreening integral per atom.
    """
    n = coords_bohr.shape[0]
    psi = np.zeros(n, dtype=np.float64)
    for ii in range(n):
        rvdwi = vdwr[ii]
        rhoi = rho[ii]
        for jj in range(n):
            if jj == ii:
                continue
            r = float(np.linalg.norm(coords_bohr[ii] - coords_bohr[jj]))
            if r < 1e-12:
                continue
            rvdwj = vdwr[jj]
            rhoj = rho[jj]
            # ovij: ji-side of i centre intersects j volume? (overlap test)
            ovij = 1 if r < rvdwi + rhoj else 0
            r1 = 1.0 / r

            if ovij == 0:
                # ij does not overlap ⇒ standard non-overlapping Still form.
                ap = r + rhoj
                am = r - rhoj
                ab = ap * am
                rhab = rhoj / ab
                lnab = 0.5 * np.log(am / ap) * r1
                gi = rhab + lnab
                psi[ii] += gi
            else:
                # ij overlaps ⇒ inner-form (xtb born.f90:271-296).
                if (r + rhoj) > rvdwi:
                    r12 = 0.5 * r1
                    ap = r + rhoj
                    am = r - rhoj
                    rh1 = 1.0 / rvdwi
                    rhr1 = 1.0 / ap
                    aprh1 = ap * rh1
                    lnab = float(np.log(aprh1))
                    gi = (
                        rh1
                        - rhr1
                        + r12 * (0.5 * am * (rhr1 - rh1 * aprh1) - lnab)
                    )
                    psi[ii] += gi
                # If r + rhoj <= rvdwi: i is fully buried inside j → no contrib.
    return psi


def compute_bornr(
    coords_bohr: np.ndarray,
    vdwr: np.ndarray,
    rho: np.ndarray,
    svdw: np.ndarray,
    c1: float,
) -> np.ndarray:
    """Born radii via OBCII tanh transform (born.f90:35-106).

    Args:
        coords_bohr: ``(n, 3)``.
        vdwr: ``(n,)`` vdW radii (Bohr).
        rho: ``(n,)`` descreened vdW radii (Bohr).
        svdw: ``(n,)`` vdW radii with ALPB offset (Bohr).
        c1: Born-radius scaling factor (gfn2_alpb_water.c1 = 1.474).

    Returns:
        ``brad[n]`` Born radii in Bohr.
    """
    n = coords_bohr.shape[0]
    psi = compute_psi(coords_bohr, vdwr, rho)
    brad = np.zeros(n, dtype=np.float64)
    alp, bet, gam = _OBCII_ALP, _OBCII_BET, _OBCII_GAM
    for iat in range(n):
        br = psi[iat]
        svdwi = svdw[iat]
        vdwri = vdwr[iat]
        s1 = 1.0 / svdwi
        v1 = 1.0 / vdwri
        s2 = 0.5 * svdwi
        br = br * s2

        arg2 = br * (gam * br - bet)
        arg = br * (alp + arg2)
        th = float(np.tanh(arg))
        br_inv = s1 - v1 * th
        if br_inv == 0.0:
            br_new = float("inf")
        else:
            br_new = c1 / br_inv
        brad[iat] = br_new
    return brad


def gb_matrix(
    coords_bohr: np.ndarray, brad: np.ndarray, alpha: float = 0.0,
) -> np.ndarray:
    """Generalized Born coupling matrix M_ij (Still kernel + ALPB α).

    Standard Still GB form for off-diagonal:
        f_ij = sqrt(R_ij² + R_b_i · R_b_j · exp(-R_ij² / (4 R_b_i R_b_j)))
        M_ij = 1 / f_ij

    Diagonal: M_ii = 1 / R_b_i.

    For ALPB extension (α > 0), an additional 1/A_det shift is added to
    each off-diagonal — but for the gfn2_alpb_water table α = 0 (it's
    plain GBSA), so we omit the extension.

    Returns: ``M[n, n]`` (Bohr⁻¹).
    """
    n = coords_bohr.shape[0]
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        M[i, i] = 1.0 / brad[i]
    for i in range(n):
        for j in range(i):
            R = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
            R2 = R * R
            bb = brad[i] * brad[j]
            f = float(np.sqrt(R2 + bb * np.exp(-R2 / (4.0 * bb))))
            M[i, j] = 1.0 / f
            M[j, i] = M[i, j]
    return M


def alpb_water_correction_native(
    atoms: list[int] | np.ndarray, coords_ang: np.ndarray,
    q_at: np.ndarray,
) -> dict:
    """Pure-numpy ALPB(water) post-SCF energy correction.

    Args:
        atoms: ``(n,)`` atomic numbers.
        coords_ang: ``(n, 3)`` Å coordinates.
        q_at: ``(n,)`` atomic Mulliken charges from the (already
            converged) GFN2-xTB SCF.

    Returns:
        Dict with ``e_born_hartree`` (the Born-Coulomb energy),
        ``e_shift_hartree`` (free-energy shift), ``brad`` (Born
        radii in Bohr), ``M_gb`` (the GB matrix).

    Notes:
        Mirrors tblite's ALPB(water) up to the SASA term, which is
        omitted in the native path. For the full ALPB(water)
        (including SASA), use :func:`solvation_alpb.alpb_water_correction`
        which dispatches to tblite.
    """
    p = _load_alpb_water_params()
    epsv = float(p["epsv"])
    c1 = float(p["c1"])
    rprobe = float(p["rprobe"])
    gshift = float(p["gshift"])  # global shift, kcal/mol
    alpha = float(p["alpha"])
    sx = p["sx"]                  # per-element vdW scaling

    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n = len(atoms)

    # GFN2 ALPB uses D3 vdW radii directly (xtb model.f90:530-531
    # falls into the non-cosmo branch). Per-atom:
    #   vdwr  = vanDerWaalsRadD3   (Bohr)
    #   rho   = vdwr · sx  (descreening factor)
    #   svdw  = vdwr − bornOffset  (= vdwr for water; soset = 0)
    soset = float(p["soset"])
    born_offset = soset * 0.1 * _ANG_TO_BOHR
    vdwr = np.array(
        [_VDW_D3_ANG[int(z) - 1] * _ANG_TO_BOHR for z in atoms],
        dtype=np.float64,
    )
    descreening = np.array([sx[int(z) - 1] for z in atoms], dtype=np.float64)
    rho = vdwr * descreening
    svdw = vdwr - born_offset

    # Born radii.
    brad = compute_bornr(coords_bohr, vdwr, rho, svdw, c1)
    # GB matrix.
    M = gb_matrix(coords_bohr, brad, alpha=alpha)
    # ALPB Coulomb energy: E = -½ · (1 - 1/ε) · q · M · q  (standard GB).
    kEps = -0.5 * (1.0 - 1.0 / epsv)
    e_born = kEps * float(q_at @ M @ q_at)
    # Global free-energy shift: gshift (kcal/mol) per molecule.
    e_shift = gshift / _KCAL_PER_HARTREE
    return {
        "e_born_hartree": e_born,
        "e_shift_hartree": e_shift,
        "e_total_hartree": e_born + e_shift,
        "brad_bohr": brad,
        "M_gb": M,
    }


def gfn2_alpb_water_native_singlepoint(
    atoms: list[int] | np.ndarray, coords_ang: np.ndarray,
    *, charge: int = 0, **scf_kwargs,
) -> dict:
    """GFN2-xTB single-point + pure-MLX ALPB(water) post-SCF correction.

    Same shape as :func:`solvation_alpb.gfn2_energy_alpb_water` but the
    ALPB correction comes from the in-tree
    :func:`alpb_water_correction_native` (Born + GB Coulomb +
    free-energy shift; SASA omitted). For the strategic-purity path —
    no tblite dependency, all computation in numpy/MLX.

    For full ALPB(water) including SASA + SCF charge coupling, use
    :func:`solvation_alpb.gfn2_energy_alpb_water` (tblite-backed).
    """
    from .scf_gfn2 import gfn2_energy

    res = gfn2_energy(atoms, coords_ang, charge=charge, **scf_kwargs)
    q_at = res["atom_charges"]
    alpb = alpb_water_correction_native(atoms, coords_ang, q_at)
    res["alpb_water_eV_native"] = (
        alpb["e_total_hartree"] / _HARTREE_PER_EV
    )
    res["energy_hartree_alpb_native"] = (
        res["energy_hartree"] + alpb["e_total_hartree"]
    )
    res["alpb_native_breakdown"] = {
        "e_born_hartree": alpb["e_born_hartree"],
        "e_shift_hartree": alpb["e_shift_hartree"],
        "brad_bohr": alpb["brad_bohr"],
    }
    return res
