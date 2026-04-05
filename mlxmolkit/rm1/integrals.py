"""
NDDO integral computation for RM1 (sp-only, no d orbitals).

Implements the Dewar-Klopman-Ohno multipole approximation for
two-center two-electron integrals, following MOPAC/PYSEQM conventions.

Key equations:
  - One-center: directly from Slater-Condon parameters (gss, gsp, gpp, gp2, hsp)
  - Two-center: multipole expansion with charge separations (da, qa)
  - Core Hamiltonian: H_core = overlap * (beta_A + beta_B) / 2

Units: eV for energies, Bohr for distances (internal), Angstrom for input.
"""
from __future__ import annotations

import numpy as np
from .params import RM1_PARAMS, ElementParams, ANG_TO_BOHR, EV_TO_KCAL

# Physical constants
EV = 27.21  # Hartree to eV (MOPAC convention)
A0 = 0.529167  # Bohr radius in Angstrom


def _charge_separations(p: ElementParams) -> tuple[float, float]:
    """Compute dipole (da) and quadrupole (qa) charge separations.

    da = charge separation for sp hybrid dipole
    qa = charge separation for pp quadrupole
    """
    if p.n_basis == 1:  # H: no p orbitals
        return 0.0, 0.0

    # Dipole: da from hsp
    # hsp = e²/(4*da²+rho1²)^0.5 approximately
    # Exact: da = (2*qn+1) * sqrt(4*zeta_s*zeta_p) / ((zeta_s+zeta_p)^2 * sqrt(3))
    # MOPAC uses: dd = 2^(qn) * sqrt(Fk) where Fk from Slater integrals
    # Simplified for RM1: compute from hsp and gsp
    if p.hsp > 0 and p.gsp > 0:
        # da² = (ev/hsp)² - (ev/gsp)²  ... approximately
        # More precisely, from MOPAC's gettab.f:
        # dd[i] = (2*qn+1) * sqrt( s_exp * p_exp ) / ((s_exp + p_exp)^2 * sqrt(3))
        qn = 2 if p.Z > 2 else 1  # principal quantum number
        n2 = 2 * qn + 1
        ss = p.zeta_s
        pp = p.zeta_p
        # Slater dipole integral
        da = n2 * np.sqrt(ss * pp) / ((ss + pp) ** 2 * np.sqrt(3.0))
        # Convert to atomic units (already in Bohr^-1 for zeta)
        # da is in Bohr
    else:
        da = 0.0

    # Quadrupole: qa from gpp, gp2
    if p.gpp > 0 and p.gp2 > 0:
        qn = 2 if p.Z > 2 else 1
        n2 = 2 * qn + 1
        pp = p.zeta_p
        # qa² from Slater quadrupole integral
        qa = np.sqrt(n2 * (n2 - 1)) / (4.0 * pp * np.sqrt(5.0))
    else:
        qa = 0.0

    return da, qa


def _additive_terms(p: ElementParams) -> tuple[float, float, float]:
    """Compute additive terms rho0, rho1, rho2 from one-center integrals.

    These ensure correct one-center limit for two-center integrals.
    rho0 = 0.5/am  where am = gss/ev
    rho1 = 0.5/ad  where ad from hsp
    rho2 = 0.5/aq  where aq from (gpp-gp2)
    """
    # Monopole additive term
    am = p.gss / EV  # in atomic units
    rho0 = 0.5 / am if am > 1e-10 else 0.0

    if p.n_basis == 1:
        return rho0, 0.0, 0.0

    # Dipole additive term
    if p.hsp > 0:
        ad = p.hsp / EV
        rho1 = 0.5 / ad if ad > 1e-10 else 0.0
    else:
        rho1 = 0.0

    # Quadrupole additive term
    dd = p.gpp - p.gp2  # in eV
    if dd > 0:
        aq = dd / EV
        rho2 = 0.5 / aq if aq > 1e-10 else 0.0
    else:
        rho2 = 0.0

    return rho0, rho1, rho2


def compute_one_center_integrals(p: ElementParams) -> np.ndarray:
    """One-center two-electron integral matrix (n_basis x n_basis x n_basis x n_basis).

    For sp basis: 4x4x4x4 but most elements zero (NDDO approximation).
    Stored as flat (10,) for unique integrals:
      0: (ss|ss) = gss
      1: (ss|pp) = gsp
      2: (pp|ss) = gsp
      3: (pp|pp) = gpp
      4: (pp'|pp') = gp2
      5: (sp|sp) = hsp  (exchange)
    """
    n = p.n_basis
    # Store as dict for clarity
    eri = {
        'gss': p.gss,
        'gsp': p.gsp,
        'gpp': p.gpp,
        'gp2': p.gp2,
        'hsp': p.hsp,
    }
    return eri


def compute_two_center_integrals_pair(
    pA: ElementParams, pB: ElementParams,
    R_ang: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-center two-electron integrals for atom pair A-B.

    Uses Klopman-Ohno-Dewar multipole approximation.

    Args:
        pA, pB: parameters for atoms A and B
        R_ang: interatomic distance in Angstrom

    Returns:
        ri: (22,) two-electron repulsion integrals in eV
        core: (4, 2) electron-core attraction integrals in eV
    """
    R = R_ang * ANG_TO_BOHR  # convert to Bohr

    daA, qaA = _charge_separations(pA)
    daB, qaB = _charge_separations(pB)
    rho0A, rho1A, rho2A = _additive_terms(pA)
    rho0B, rho1B, rho2B = _additive_terms(pB)

    # Klopman-Ohno formula: (μν|λσ) = ev / sqrt(R² + (rho_μν + rho_λσ)²)
    ev1 = EV / 2.0
    ev2 = EV / 4.0

    # The 22 unique integrals depend on which atom pair type:
    ri = np.zeros(22)
    core = np.zeros((4, 2))

    nA = pA.n_basis  # 1 for H, 4 for heavy
    nB = pB.n_basis

    # (ss|ss) — always present
    aee = (rho0A + rho0B) ** 2
    ri[0] = EV / np.sqrt(R ** 2 + aee)

    if nA == 1 and nB == 1:
        # H-H: only (ss|ss)
        core[0, 0] = pB.n_valence * ri[0]
        core[0, 1] = pA.n_valence * ri[0]

    elif nA > 1 and nB == 1:
        # Heavy-H: 4 integrals
        ade = (rho1A + rho0B) ** 2
        aqe = (rho2A + rho0B) ** 2
        da = daA
        qa = qaA * 2.0

        ri[0] = EV / np.sqrt(R ** 2 + aee)  # (ss|ss)
        ri[1] = ev1 / np.sqrt((R + da) ** 2 + ade) - ev1 / np.sqrt((R - da) ** 2 + ade)  # (sσ|ss)
        ri[2] = ev1 / np.sqrt((R + da) ** 2 + ade) + ev1 / np.sqrt((R - da) ** 2 + ade) - EV / np.sqrt(R ** 2 + ade)  # (σσ|ss) ≈ (pp_sigma|ss)
        # Quadrupole term
        qterm = ev2 / np.sqrt(R ** 2 + aqe)
        ri[3] = qterm  # (pp_pi|ss)

        core[0, 0] = pB.n_valence * ri[0]
        core[1, 0] = pB.n_valence * ri[1]
        core[2, 0] = pB.n_valence * ri[2]
        core[3, 0] = pB.n_valence * ri[3]
        core[0, 1] = pA.n_valence * ri[0]

    elif nA > 1 and nB > 1:
        # Heavy-Heavy: 22 integrals (full sp-sp interaction)
        da = daA
        db = daB
        qa = qaA * 2.0
        qb = qaB * 2.0

        ade_a = (rho1A + rho0B) ** 2
        ade_b = (rho0A + rho1B) ** 2
        aqe_a = (rho2A + rho0B) ** 2
        aqe_b = (rho0A + rho2B) ** 2
        add = (rho1A + rho1B) ** 2
        adq = (rho1A + rho2B) ** 2
        aqd = (rho2A + rho1B) ** 2
        aqq = (rho2A + rho2B) ** 2

        # (ss|ss)
        ri[0] = EV / np.sqrt(R ** 2 + aee)

        # Build all 22 integrals following MOPAC's repp.f conventions
        # This is simplified — full implementation follows PYSEQM's local frame code
        # For now, compute the key integrals for the Fock matrix

        # Monopole-monopole
        core[0, 0] = pB.n_valence * ri[0]
        core[0, 1] = pA.n_valence * ri[0]

    return ri, core


def compute_core_hamiltonian(
    atoms: list[int],
    coords: np.ndarray,
) -> np.ndarray:
    """Compute the core Hamiltonian matrix H_core.

    H_core[μ,ν] = Uss (diagonal, same atom)
                 = 0.5 * (beta_μ + beta_ν) * S_μν (off-diagonal, different atoms)

    Args:
        atoms: list of atomic numbers
        coords: (N, 3) coordinates in Angstrom

    Returns:
        H_core: (n_basis_total, n_basis_total) matrix in eV
    """
    n_atoms = len(atoms)
    params = [RM1_PARAMS[z] for z in atoms]
    n_basis = sum(p.n_basis for p in params)

    H = np.zeros((n_basis, n_basis))

    # Diagonal: one-electron one-center integrals
    mu = 0
    for i, p in enumerate(params):
        H[mu, mu] = p.Uss  # s orbital
        if p.n_basis > 1:
            for k in range(1, 4):
                H[mu + k, mu + k] = p.Upp  # px, py, pz
        mu += p.n_basis

    # Off-diagonal: resonance integrals (simplified)
    # Full implementation needs overlap integrals S_μν
    # H_μν = 0.5 * (beta_μ + beta_ν) * S_μν

    return H


def compute_nuclear_repulsion(
    atoms: list[int],
    coords: np.ndarray,
    param_dict: dict = None,
) -> float:
    """Compute core-core (nuclear) repulsion energy.

    E_nuc = Σ_{A<B} Z_A * Z_B * (ss|ss)_AB * f(R_AB)

    where f includes the AM1/RM1 Gaussian correction terms.

    Returns energy in eV.
    """
    if param_dict is None:
        param_dict = RM1_PARAMS
    n_atoms = len(atoms)
    params = [param_dict[z] for z in atoms]
    E_nuc = 0.0

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pA = params[i]
            pB = params[j]
            R = np.linalg.norm(coords[i] - coords[j])  # Angstrom
            R_bohr = R * ANG_TO_BOHR

            # (ss|ss) integral
            rho0A = 0.5 * EV / pA.gss if pA.gss > 0 else 0.0
            rho0B = 0.5 * EV / pB.gss if pB.gss > 0 else 0.0
            aee = (rho0A + rho0B) ** 2
            ssss = EV / np.sqrt(R_bohr ** 2 + aee)

            ZA = pA.n_valence
            ZB = pB.n_valence

            # MOPAC/PYSEQM AM1 convention:
            # t1 = Z_A * Z_B * (ss|ss)
            # t2 = exp(-alpha_A * R_angstrom)
            # t3 = exp(-alpha_B * R_angstrom)
            # Gaussian: t4 = Z_A * Z_B / R_angstrom (Coulomb-like, NOT ssss)
            # t5 = Σ K_A * exp(-L_A * (R_ang - M_A)²)
            # t6 = Σ K_B * exp(-L_B * (R_ang - M_B)²)
            # E_nuc_pair = t1 * (1 + t2 + t3) + t4 * (t5 + t6)

            t1 = ZA * ZB * ssss

            # MOPAC special case: N-H and O-H pairs
            # exp(-alpha*R) becomes exp(-alpha*R)*R for the heavy atom
            # PYSEQM: XH = ((ni==7)|(ni==8)) & (nj==1)
            is_NH_OH_A = (pA.Z in (7, 8)) and (pB.Z == 1)
            is_NH_OH_B = (pB.Z in (7, 8)) and (pA.Z == 1)

            t2 = np.exp(-pA.alpha * R)
            if is_NH_OH_A:
                t2 *= R  # special N-H/O-H convention
            t3 = np.exp(-pB.alpha * R)
            if is_NH_OH_B:
                t3 *= R  # special N-H/O-H convention

            t4 = ZA * ZB / R  # Coulomb Z*Z/R in Angstrom (eV·A units from Gaussian terms)
            t5 = sum(pA.gauss_K[k] * np.exp(-pA.gauss_L[k] * (R - pA.gauss_M[k]) ** 2)
                    for k in range(4) if pA.gauss_K[k] != 0)
            t6 = sum(pB.gauss_K[k] * np.exp(-pB.gauss_L[k] * (R - pB.gauss_M[k]) ** 2)
                    for k in range(4) if pB.gauss_K[k] != 0)

            E_nuc += t1 * (1.0 + t2 + t3) + t4 * (t5 + t6)

    return E_nuc
