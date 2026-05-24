"""
COSMO-RS thermodynamics: activity coefficients from sigma profiles.

Port of openCOSMO-RS_py cosmors.py core algorithm.

Steps:
1. Compute interaction matrices A_mf (misfit) and A_hb (hydrogen bonding)
2. COSMOspace iteration: Γ_new = 1 / ((X·Γ)·τᵀ) — successive substitution
3. Activity coefficients: ln(γ) = Σ_k n_k · [ln(Γ_k) - ln(Γ_k^pure)]
4. Combinatorial (Staverman-Guggenheim) contribution

Reference: Klamt, COSMO-RS: From Quantum Chemistry to Fluid Phase
           Thermodynamics and Drug Design, Elsevier 2005.
"""
from __future__ import annotations

import numpy as np
from . import params as _P


def _compute_interaction_matrices(
    sigma_arr: np.ndarray,
    hb_type_arr: np.ndarray,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute misfit and hydrogen bonding interaction matrices.

    Args:
        sigma_arr: (n_segtp,) sigma values for each segment type
        hb_type_arr: (n_segtp,) 0=non-HB, 1=donor, 2=acceptor
        T: temperature in Kelvin

    Returns:
        A_mf: (n_segtp, n_segtp) misfit interaction in J/mol
        A_hb: (n_segtp, n_segtp) HB interaction in J/mol
    """
    n = len(sigma_arr)

    # Misfit: A_mf[i,j] = 0.5 · α · a_eff · (σ_i + σ_j)²
    sigma_sum = sigma_arr[:, np.newaxis] + sigma_arr[np.newaxis, :]  # (n, n)
    A_mf = 0.5 * _P.MF_ALPHA * _P.A_EFF * sigma_sum ** 2

    # Hydrogen bonding: donor-acceptor interaction
    A_hb = np.zeros((n, n))

    # Temperature-dependent prefactor
    hb_c_at_T = _P.HB_C * (1.0 - _P.HB_C_T + _P.HB_C_T * (298.15 / T))

    # Donor: sigma < 0 (negative charge → positive potential on H)
    # Acceptor: sigma > 0 (positive charge → negative potential on lone pair)
    for i in range(n):
        for j in range(n):
            # Donor i, acceptor j
            if hb_type_arr[i] == 1 and hb_type_arr[j] == 2:
                del_d = sigma_arr[i] + _P.HB_SIGMA_THRESH  # σ_D + threshold (< 0)
                del_a = sigma_arr[j] - _P.HB_SIGMA_THRESH  # σ_A - threshold (> 0)
                if del_d < 0 and del_a > 0:
                    A_hb[i, j] = hb_c_at_T * _P.A_EFF * del_d * del_a
            # Acceptor i, donor j
            elif hb_type_arr[i] == 2 and hb_type_arr[j] == 1:
                del_d = sigma_arr[j] + _P.HB_SIGMA_THRESH
                del_a = sigma_arr[i] - _P.HB_SIGMA_THRESH
                if del_d < 0 and del_a > 0:
                    A_hb[i, j] = hb_c_at_T * _P.A_EFF * del_d * del_a

    return A_mf, A_hb


def cosmospace(
    X: np.ndarray,
    tau: np.ndarray,
    max_iter: int = 1000,
    conv_thresh: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """COSMOspace iteration: solve for segment activity coefficients.

    Γ_new = 1 / ((X · Γ) · τᵀ)

    Successive substitution with damping.

    Args:
        X: (n_segtp,) segment type mole fractions
        tau: (n_segtp, n_segtp) Boltzmann factors exp(-A/(RT))
        max_iter: maximum iterations
        conv_thresh: relative convergence threshold

    Returns:
        Gamma: (n_segtp,) segment activity coefficients
        n_iter: number of iterations
    """
    n = len(X)
    Gamma = np.ones(n)

    for iteration in range(max_iter):
        XG = X * Gamma  # (n,)
        denom = XG @ tau.T  # (n,) — matrix-vector product
        Gamma_new = 1.0 / (denom + 1e-30)

        # Check convergence
        rel_change = np.max(np.abs(Gamma_new - Gamma) / (np.abs(Gamma) + 1e-30))
        if rel_change < conv_thresh:
            return Gamma_new, iteration + 1

        # Damped update (0.7 mixing)
        Gamma = 0.7 * (Gamma_new - Gamma) + Gamma

    return Gamma, max_iter


def activity_coefficients(
    mol_profiles: list[dict],
    x: np.ndarray,
    T: float = 298.15,
    refst: str = 'pure_component',
) -> np.ndarray:
    """Compute activity coefficients for a mixture.

    Args:
        mol_profiles: list of sigma analysis dicts (from sigma.full_sigma_analysis)
            Each must have: 'sigma_grid', 'sigma_profile', 'total_area',
                           'seg_sigma_av', 'seg_hb_type'
        x: (n_mol,) mole fractions
        T: temperature in Kelvin
        refst: reference state ('pure_component' or 'cosmo')

    Returns:
        lng: (n_mol,) logarithmic activity coefficients
    """
    n_mol = len(mol_profiles)
    sigma_grid = _P.SIGMA_GRID
    n_bins = len(sigma_grid)

    # Build unified segment type list from all molecules
    # For simplicity, use the sigma grid bins as segment types
    # Each bin gets an HB classification based on majority of contributing segments

    # Collect per-molecule: area in each sigma bin, HB type
    mol_area_profiles = np.zeros((n_mol, n_bins))
    mol_hb_profiles = np.zeros((n_mol, n_bins), dtype=np.int32)

    for m, prof in enumerate(mol_profiles):
        mol_area_profiles[m] = prof['sigma_profile']

        # Assign HB type per sigma bin: donor for σ < -thresh, acceptor for σ > thresh
        for k in range(n_bins):
            s = sigma_grid[k]
            if s < -_P.HB_SIGMA_THRESH:
                mol_hb_profiles[m, k] = 1  # potential donor
            elif s > _P.HB_SIGMA_THRESH:
                mol_hb_profiles[m, k] = 2  # potential acceptor

    # Mixture segment mole fractions
    total_area_per_mol = np.array([p['total_area'] for p in mol_profiles])
    weighted_area = x * total_area_per_mol  # (n_mol,)

    # X[k] = Σ_m x_m · A_m(k) / Σ_m x_m · A_total_m
    mixture_profile = np.zeros(n_bins)
    for m in range(n_mol):
        mixture_profile += weighted_area[m] * mol_area_profiles[m] / (total_area_per_mol[m] + 1e-30)

    X = mixture_profile / (np.sum(mixture_profile) + 1e-30)

    # Majority HB type
    hb_type_arr = np.zeros(n_bins, dtype=np.int32)
    for k in range(n_bins):
        # Use most common HB type across molecules
        hb_counts = [0, 0, 0]
        for m in range(n_mol):
            if mol_area_profiles[m, k] > 0:
                hb_counts[mol_hb_profiles[m, k]] += mol_area_profiles[m, k]
        hb_type_arr[k] = np.argmax(hb_counts)

    # Interaction matrices
    A_mf, A_hb = _compute_interaction_matrices(sigma_grid, hb_type_arr, T)
    A_int = A_mf + A_hb

    # Boltzmann factors
    tau = np.exp(-A_int / (_P.R_GAS * T))

    # COSMOspace for mixture
    Gamma_mix, n_iter_mix = cosmospace(X, tau)

    # Activity coefficients
    lng = np.zeros(n_mol)

    if refst == 'pure_component':
        # Reference: pure component Γ
        for m in range(n_mol):
            X_pure = mol_area_profiles[m] / (np.sum(mol_area_profiles[m]) + 1e-30)
            Gamma_pure, _ = cosmospace(X_pure, tau)

            # ln(γ_m) = Σ_k n_k · [ln(Γ_k^mix) - ln(Γ_k^pure)]
            n_k = mol_area_profiles[m] / (_P.A_EFF + 1e-30)
            lng[m] = np.sum(n_k * (np.log(Gamma_mix + 1e-30) - np.log(Gamma_pure + 1e-30)))
    else:
        # COSMO reference state
        for m in range(n_mol):
            n_k = mol_area_profiles[m] / (_P.A_EFF + 1e-30)
            lng[m] = np.sum(n_k * np.log(Gamma_mix + 1e-30))

    # Combinatorial contribution (Staverman-Guggenheim)
    r = total_area_per_mol / _P.COMB_SG_A_STD
    phi = x * r / (np.sum(x * r) + 1e-30)
    theta = x * total_area_per_mol / (np.sum(x * total_area_per_mol) + 1e-30)

    z_half = _P.COMB_SG_Z_COORD / 2.0
    lng_comb = np.log(phi / (x + 1e-30) + 1e-30) + 1.0 - phi / (x + 1e-30) \
               - z_half * r * (np.log(phi / (theta + 1e-30) + 1e-30) + 1.0 - phi / (theta + 1e-30))

    # Handle x=0 cases
    lng_comb[x < 1e-30] = 0.0

    return lng + lng_comb
