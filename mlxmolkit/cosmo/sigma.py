"""
Sigma profile generation and averaging for COSMO-RS.

Port of openCOSMO-RS molecules.py sigma averaging algorithm.

Steps:
1. Gaussian-weight averaging of raw sigma over local region (r_av = 0.5 Å)
2. Classify segments by (sigma, element, HB type)
3. Bin into sigma profile histogram
"""
from __future__ import annotations

import numpy as np
from .params import (R_AV, SIGMA_GRID, SIGMA_GRID_STEP,
                     HB_DONOR_ELEMENTS, HB_ACCEPTOR_ELEMENTS)


def average_sigma(
    seg_pos: np.ndarray,
    seg_area: np.ndarray,
    seg_sigma_raw: np.ndarray,
    r_av: float = R_AV,
) -> np.ndarray:
    """Gaussian-weighted averaging of sigma charges.

    σ_av(i) = Σ_j σ_raw(j) · A_j · w(i,j) / Σ_j A_j · w(i,j)
    where w(i,j) = r_av² / (r_seg_i² + r_av²) · exp(-d²_ij / (r_seg_i² + r_av²))

    Uses scipy cdist (C-optimized) for squared distances.
    """
    from scipy.spatial.distance import cdist

    r_av_sq = r_av * r_av
    r_seg_sq = seg_area / np.pi
    inv_rad = 1.0 / (r_seg_sq + r_av_sq)

    # Squared distances via C-optimized cdist
    dist_sq = cdist(seg_pos, seg_pos, 'sqeuclidean')  # (n, n) — no 3D intermediate

    # Gaussian weights
    exp_arr = np.exp(-dist_sq * inv_rad[:, np.newaxis])
    weight = (r_av_sq * inv_rad[:, np.newaxis] * exp_arr).T  # transposed for matmul

    # Weighted average via matmul
    sr = seg_sigma_raw * r_seg_sq
    numerator = sr @ weight
    denominator = r_seg_sq @ weight

    return numerator / (denominator + 1e-30)


def compute_sigma_profile(
    seg_area: np.ndarray,
    seg_sigma: np.ndarray,
    sigma_grid: np.ndarray = SIGMA_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sigma profile p(σ) = histogram of segment areas by sigma.

    Vectorized: no Python loops over segments.
    """
    n_bins = len(sigma_grid)

    # Continuous bin positions
    idx_float = (seg_sigma - sigma_grid[0]) / SIGMA_GRID_STEP
    idx = np.floor(idx_float).astype(np.int64)
    frac = idx_float - idx

    # Clamp to valid range
    idx = np.clip(idx, 0, n_bins - 2)
    frac = np.clip(frac, 0.0, 1.0)

    # Scatter-add: np.add.at for lower and upper bins
    profile = np.zeros(n_bins)
    np.add.at(profile, idx, seg_area * (1.0 - frac))
    np.add.at(profile, idx + 1, seg_area * frac)

    return sigma_grid, profile


def classify_segments(
    atoms: list[int],
    seg_sigma: np.ndarray,
    seg_atom: np.ndarray,
    adjacency: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify segments as HB donor, HB acceptor, or non-HB. Vectorized."""
    atoms_arr = np.array(atoms)
    seg_element = atoms_arr[seg_atom]
    seg_hb_type = np.zeros(len(seg_sigma), dtype=np.int32)

    # HB donors: H atoms with negative sigma
    is_H = np.isin(seg_element, list(HB_DONOR_ELEMENTS))
    seg_hb_type[is_H & (seg_sigma < 0)] = 1

    # HB acceptors: electronegative atoms with positive sigma
    is_acc = np.isin(seg_element, list(HB_ACCEPTOR_ELEMENTS))
    seg_hb_type[is_acc & (seg_sigma > 0)] = 2

    return seg_hb_type, seg_element


def full_sigma_analysis(cosmo_result: dict, atoms: list[int]) -> dict:
    """Complete sigma profile analysis from COSMO surface data.

    Args:
        cosmo_result: output from cavity.cosmo_surface()
        atoms: atomic numbers

    Returns:
        dict with sigma profile, HB classification, and statistics
    """
    seg_pos = cosmo_result['seg_pos']
    seg_area = cosmo_result['seg_area']
    seg_sigma_raw = cosmo_result['seg_sigma']
    seg_atom = cosmo_result['seg_atom']

    # Average sigma
    seg_sigma_av = average_sigma(seg_pos, seg_area, seg_sigma_raw)

    # Sigma profile
    sigma_grid, profile = compute_sigma_profile(seg_area, seg_sigma_av)

    # HB classification
    seg_hb_type, seg_element = classify_segments(atoms, seg_sigma_av, seg_atom)

    # Sigma moments
    total_area = np.sum(seg_area)
    p_norm = profile / (total_area + 1e-30)
    sigma_moment_0 = total_area
    sigma_moment_1 = np.sum(sigma_grid * p_norm) * total_area  # dipole
    sigma_moment_2 = np.sum(sigma_grid**2 * p_norm) * total_area  # variance

    return {
        'sigma_grid': sigma_grid,
        'sigma_profile': profile,
        'seg_sigma_av': seg_sigma_av,
        'seg_hb_type': seg_hb_type,
        'seg_element': seg_element,
        'total_area': total_area,
        'sigma_moment_0': sigma_moment_0,
        'sigma_moment_1': sigma_moment_1,
        'sigma_moment_2': sigma_moment_2,
    }
