"""
ddCOSMO-inspired COSMO solver with smooth switching function.

Improvements over simple COSMO (cavity.py):
1. Smooth 5th-degree polynomial switching function (xTB convention)
   - No sharp burial cutoff → smoother sigma profiles
   - Width parameter eta=0.2 Å
2. Weighted surface charges: q(i) = ui(i) * q_raw(i)
   - Partially buried segments contribute proportionally
3. Better cavity area from weighted quadrature

Reference: Stahn et al., J. Phys. Chem. A 2023, 127, 5555-5567
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from .params import VDW_RADII, CAVITY_SCALING, EPSILON_WATER, BOHR_TO_ANG
from .lebedev import get_lebedev_grid
from .spherical_harmonics import real_spherical_harmonics, project_to_harmonics, expand_from_harmonics


def _switching_function(t: np.ndarray, eta: float = 0.2) -> np.ndarray:
    """Smooth 5th-degree polynomial switching function (xTB convention).

    χ(t) = 1  if t <= 1-eta  (fully inside neighbor → buried)
    χ(t) = 0  if t >= 1      (fully outside → exposed)
    χ(t) = smooth transition for 1-eta < t < 1

    Args:
        t: normalized distance t = |r - r_j| / R_j
        eta: switching width (default 0.2)

    Returns:
        chi: switching function values in [0, 1]
    """
    chi = np.zeros_like(t)

    # Fully buried: t <= 1 - eta
    chi[t <= 1.0 - eta] = 1.0

    # Transition region: 1-eta < t < 1
    mask = (t > 1.0 - eta) & (t < 1.0)
    if np.any(mask):
        # Normalize to [0, 1] range
        s = (t[mask] - (1.0 - eta)) / eta
        # 5th degree: smooth C2 transition from 1 to 0
        chi[mask] = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5

    return chi


def build_ddcosmo_cavity(
    atoms: list[int],
    coords: np.ndarray,
    n_points_per_atom: int = 194,
    scaling: float = CAVITY_SCALING,
    eta: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build COSMO cavity with smooth switching function.

    Returns:
        seg_pos: (n_seg, 3) segment positions in Angstrom
        seg_area: (n_seg,) segment areas (weighted by ui)
        seg_normal: (n_seg, 3) outward normals
        seg_atom: (n_seg,) atom index
        seg_ui: (n_seg,) switching function values (0=buried, 1=exposed)
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    sphere_pts, sphere_weights = get_lebedev_grid(n_points_per_atom)
    n_pts = len(sphere_pts)

    radii = np.array([VDW_RADII.get(z, 2.0) * scaling for z in atoms])

    all_pos = []
    all_area = []
    all_normal = []
    all_atom_idx = []
    all_ui = []

    for i in range(n_atoms):
        r_i = radii[i]
        pts = coords[i] + r_i * sphere_pts  # (n_pts, 3)

        # Compute switching function fi(n) = Σ_j χ(t_n^j)
        fi = np.zeros(n_pts)
        for j in range(n_atoms):
            if j == i:
                continue
            # Normalized distance: t = |pt - center_j| / R_j
            dists = np.linalg.norm(pts - coords[j], axis=1)
            t = dists / radii[j]
            fi += _switching_function(t, eta)

        # Exposure indicator: ui = max(0, 1 - fi)
        ui = np.maximum(0.0, 1.0 - fi)

        # Keep segments with ui > threshold (partially or fully exposed)
        threshold = 0.001
        mask = ui > threshold

        if np.sum(mask) == 0:
            continue

        # Weighted areas: weight * r² * ui
        areas = sphere_weights[mask] * r_i * r_i * ui[mask]

        all_pos.append(pts[mask])
        all_area.append(areas)
        all_normal.append(sphere_pts[mask])
        all_atom_idx.append(np.full(np.sum(mask), i, dtype=np.int32))
        all_ui.append(ui[mask])

    if not all_pos:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)),
                np.zeros(0, dtype=np.int32), np.zeros(0))

    return (np.vstack(all_pos), np.concatenate(all_area),
            np.vstack(all_normal), np.concatenate(all_atom_idx),
            np.concatenate(all_ui))


def _jacobi_diis_solve(
    A: np.ndarray,
    rhs: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
    n_diis: int = 20,
) -> tuple[np.ndarray, int, bool]:
    """Jacobi iteration with DIIS extrapolation for A·q = rhs.

    Split A = D + O (diagonal + off-diagonal).
    Iterate: q_new = D⁻¹ · (rhs - O · q)
    DIIS: extrapolate from error history for faster convergence.

    O(n²) per iteration (matvec) vs O(n³) for dense solve.

    Args:
        A: (n, n) coefficient matrix
        rhs: (n,) right-hand side
        max_iter: maximum Jacobi iterations
        tol: convergence tolerance (rms of increment)
        n_diis: DIIS history size

    Returns:
        q: (n,) solution
        n_iter: iterations used
        converged: whether tolerance was reached
    """
    n = len(rhs)
    D_inv = 1.0 / np.diag(A)  # (n,)
    O = A.copy()
    np.fill_diagonal(O, 0.0)   # off-diagonal only

    # Initial guess: D⁻¹ · rhs
    q = D_inv * rhs

    # DIIS history
    diis_q = []
    diis_err = []

    for iteration in range(max_iter):
        # Jacobi step: q_new = D⁻¹ · (rhs - O · q)
        residual = rhs - O @ q
        q_new = D_inv * residual

        # Error = increment
        dq = q_new - q
        rms_dq = np.sqrt(np.mean(dq * dq))

        if rms_dq < tol:
            return q_new, iteration + 1, True

        # DIIS extrapolation (after a few plain Jacobi steps)
        if iteration >= 2:
            diis_q.append(q_new.copy())
            diis_err.append(dq.copy())

            if len(diis_q) > n_diis:
                diis_q.pop(0)
                diis_err.pop(0)

            nd = len(diis_q)
            if nd >= 2:
                # Build DIIS B matrix: B[i,j] = err_i · err_j
                B = np.zeros((nd + 1, nd + 1))
                for i in range(nd):
                    for j in range(i, nd):
                        B[i, j] = np.dot(diis_err[i], diis_err[j])
                        B[j, i] = B[i, j]
                B[:nd, nd] = -1.0
                B[nd, :nd] = -1.0

                rhs_diis = np.zeros(nd + 1)
                rhs_diis[nd] = -1.0

                try:
                    c = np.linalg.solve(B, rhs_diis)
                    q_new = sum(c[i] * diis_q[i] for i in range(nd))
                except np.linalg.LinAlgError:
                    pass  # Skip DIIS this step, use plain Jacobi

        # SOR-like damped update (omega=0.5 for stability with dense Coulomb)
        q = 0.5 * q_new + 0.5 * q

    return q, max_iter, False


def ddcosmo_charges(
    atoms: list[int],
    coords: np.ndarray,
    mulliken_charges: np.ndarray,
    seg_pos: np.ndarray,
    seg_area: np.ndarray,
    seg_ui: np.ndarray,
    epsilon: float = EPSILON_WATER,
    solver: str = 'auto',
) -> np.ndarray:
    """Solve COSMO equation with ddCOSMO-weighted segments.

    Args:
        solver: 'jacobi' for Jacobi/DIIS iterative,
                'direct' for np.linalg.solve,
                'auto' picks based on system size (Jacobi for n>400)
    """
    n_seg = len(seg_pos)
    if n_seg == 0:
        return np.zeros(0)

    seg_pos_b = seg_pos / BOHR_TO_ANG
    seg_area_b = seg_area / (BOHR_TO_ANG ** 2)
    coords_b = coords / BOHR_TO_ANG

    # A matrix
    dist = cdist(seg_pos_b, seg_pos_b)
    np.fill_diagonal(dist, 1.0)
    A = 1.0 / dist
    np.fill_diagonal(A, 1.07 * np.sqrt(4.0 * np.pi / np.maximum(seg_area_b, 1e-30)))

    # Phi potential
    dist_ac = np.maximum(cdist(seg_pos_b, coords_b), 1e-10)
    Phi = (mulliken_charges[np.newaxis, :] / dist_ac).sum(axis=1)

    # Weighted RHS
    rhs = -Phi * seg_ui

    # Solve
    if solver == 'auto':
        # Direct (numpy Accelerate) is faster for n < 3000 on Apple Silicon
        # Jacobi/DIIS only for very large systems where O(n³) dominates
        solver = 'jacobi' if n_seg > 3000 else 'direct'

    if solver == 'jacobi':
        q, n_iter, converged = _jacobi_diis_solve(A, rhs, max_iter=200, tol=1e-8)
    else:
        q = np.linalg.solve(A, rhs)

    keps = (epsilon - 1.0) / (epsilon + 0.5)
    return keps * q


def ddcosmo_charges_sh(
    atoms: list[int],
    coords: np.ndarray,
    mulliken_charges: np.ndarray,
    seg_pos: np.ndarray,
    seg_area: np.ndarray,
    seg_normal: np.ndarray,
    seg_atom: np.ndarray,
    seg_ui: np.ndarray,
    epsilon: float = EPSILON_WATER,
    lmax: int = 6,
) -> np.ndarray:
    """Solve COSMO in spherical harmonic basis (xTB-style).

    Instead of n_seg × n_seg dense system, projects onto
    (lmax+1)² = 49 basis per atom → much smaller system.

    Steps:
    1. Project Phi onto SH basis per atom: rhs_{lm,A} = Σ_i w_i ui_i Phi_i Y_{lm}(θ_i)
    2. Build coupling matrix L in SH basis
    3. Solve L·σ_sh = rhs_sh (direct — matrix is small)
    4. Expand σ back to segments: σ_i = Σ_{lm} σ_{lm,A} · Y_{lm}(θ_i)

    Args:
        lmax: max angular momentum (6 → 49 basis per atom)
    """
    n_seg = len(seg_pos)
    if n_seg == 0:
        return np.zeros(0)

    n_atoms = len(atoms)
    n_ylm = (lmax + 1) ** 2
    n_basis_total = n_atoms * n_ylm

    coords = np.asarray(coords, dtype=np.float64)
    seg_pos_b = seg_pos / BOHR_TO_ANG
    seg_area_b = seg_area / (BOHR_TO_ANG ** 2)
    coords_b = coords / BOHR_TO_ANG

    radii = np.array([VDW_RADII.get(z, 2.0) * CAVITY_SCALING for z in atoms])

    # Get Lebedev grid used for cavity
    sphere_pts, sphere_weights = get_lebedev_grid(194)

    # Compute SH basis for each grid point
    Y = real_spherical_harmonics(lmax, sphere_pts)  # (n_ylm, n_grid)

    # Phi at each segment
    dist_ac = np.maximum(cdist(seg_pos_b, coords_b), 1e-10)
    Phi = (mulliken_charges[np.newaxis, :] / dist_ac).sum(axis=1)

    # --- Project RHS to SH basis per atom ---
    rhs_sh = np.zeros(n_basis_total)

    # Map segments back to per-atom grid points
    # We need to know which grid point each segment came from
    # Since we kept all segments with ui > threshold, we need to track this

    # For now: project Phi * ui weighted by area onto SH per atom
    for a in range(n_atoms):
        mask = seg_atom == a
        if not np.any(mask):
            continue

        # Local positions → unit vectors on atom's sphere
        local_pos = seg_pos[mask] - coords[a]
        local_r = np.linalg.norm(local_pos, axis=1, keepdims=True)
        local_unit = local_pos / np.maximum(local_r, 1e-30)

        # SH values at these segment positions
        Y_local = real_spherical_harmonics(lmax, local_unit)  # (n_ylm, n_seg_a)

        # Weighted potential
        phi_weighted = Phi[mask] * seg_ui[mask]
        w_local = seg_area_b[mask] / (radii[a] / BOHR_TO_ANG) ** 2  # approximate weights

        # Project: rhs_{lm} = -Σ_i w_i · phi_i · Y_{lm,i}
        rhs_sh[a * n_ylm:(a + 1) * n_ylm] = -Y_local @ (phi_weighted * w_local)

    # --- Build coupling matrix L in SH basis ---
    # L_{(a,lm),(b,l'm')} = <Y_{lm}^a | A | Y_{l'm'}^b>
    # For on-sphere: diagonal = self-interaction
    # For cross-sphere: Coulomb coupling

    L = np.zeros((n_basis_total, n_basis_total))

    for a in range(n_atoms):
        # Diagonal block (same atom): self-interaction
        # L_{aa} ≈ diag(1.07 * sqrt(4π/a_eff)) * delta_{lm,l'm'}
        # Simplified: use average self-interaction
        r_a_b = radii[a] / BOHR_TO_ANG
        a_eff_b = 4.0 * np.pi * r_a_b ** 2 / len(sphere_pts)
        self_int = 1.07 * np.sqrt(4.0 * np.pi / a_eff_b)
        for lm in range(n_ylm):
            L[a * n_ylm + lm, a * n_ylm + lm] = self_int

        # Off-diagonal blocks (atom a ↔ atom b)
        for b in range(n_atoms):
            if b == a:
                continue
            # Coulomb coupling between SH on atoms a and b
            # Sample: compute 1/|r_i^a - r_j^b| for Lebedev points, project onto SH
            mask_a = seg_atom == a
            mask_b = seg_atom == b
            if not np.any(mask_a) or not np.any(mask_b):
                continue

            pos_a_b = seg_pos_b[mask_a]
            pos_b_b = seg_pos_b[mask_b]

            # Coulomb matrix between a's and b's segments
            dist_ab = np.maximum(cdist(pos_a_b, pos_b_b), 1e-10)
            coulomb_ab = 1.0 / dist_ab  # (n_a, n_b)

            # Project: L_{(a,lm),(b,l'm')} = Σ_{i,j} Y_{lm}(i) * w_i * coulomb(i,j) * w_j * Y_{l'm'}(j)
            local_a = seg_pos[mask_a] - coords[a]
            r_a = np.linalg.norm(local_a, axis=1, keepdims=True)
            Y_a = real_spherical_harmonics(lmax, local_a / np.maximum(r_a, 1e-30))

            local_b = seg_pos[mask_b] - coords[b]
            r_b = np.linalg.norm(local_b, axis=1, keepdims=True)
            Y_b = real_spherical_harmonics(lmax, local_b / np.maximum(r_b, 1e-30))

            w_a = seg_area_b[mask_a] / (radii[a] / BOHR_TO_ANG) ** 2
            w_b = seg_area_b[mask_b] / (radii[b] / BOHR_TO_ANG) ** 2

            # L_ab = Y_a · diag(w_a) · coulomb · diag(w_b) · Y_b^T
            L_ab = (Y_a * w_a[np.newaxis, :]) @ coulomb_ab @ (Y_b * w_b[np.newaxis, :]).T

            L[a * n_ylm:(a + 1) * n_ylm,
              b * n_ylm:(b + 1) * n_ylm] = L_ab

    # --- Solve L · σ_sh = rhs_sh ---
    sigma_sh = np.linalg.solve(L, rhs_sh)

    # --- Expand back to segments ---
    q = np.zeros(n_seg)
    for a in range(n_atoms):
        mask = seg_atom == a
        if not np.any(mask):
            continue

        local_pos = seg_pos[mask] - coords[a]
        local_r = np.linalg.norm(local_pos, axis=1, keepdims=True)
        Y_local = real_spherical_harmonics(lmax, local_pos / np.maximum(local_r, 1e-30))

        coeffs = sigma_sh[a * n_ylm:(a + 1) * n_ylm]
        q[mask] = expand_from_harmonics(coeffs, Y_local) * seg_ui[mask]

    keps = (epsilon - 1.0) / (epsilon + 0.5)
    return keps * q


def ddcosmo_surface(
    atoms: list[int],
    coords: np.ndarray,
    density: np.ndarray,
    n_points: int = 194,
    epsilon: float = EPSILON_WATER,
    eta: float = 0.2,
    solver: str = 'direct',
    lmax: int = 6,
) -> dict:
    """Complete ddCOSMO surface calculation.

    Args:
        solver: 'direct' (dense np.linalg.solve on segments),
                'sh' (spherical harmonic basis — smaller matrix),
                'jacobi' (iterative on segments),
                'auto' (picks best for system size)
        lmax: max angular momentum for SH basis (6→49 per atom)
    """
    from .cavity import _mulliken_charges
    from ..rm1.params import RM1_PARAMS

    coords = np.asarray(coords, dtype=np.float64)

    seg_pos, seg_area, seg_normal, seg_atom, seg_ui = build_ddcosmo_cavity(
        atoms, coords, n_points_per_atom=n_points, eta=eta,
    )

    n_basis_per = [RM1_PARAMS[z].n_basis for z in atoms]
    mulliken = _mulliken_charges(atoms, density, n_basis_per)

    if solver == 'sh':
        seg_charge = ddcosmo_charges_sh(
            atoms, coords, mulliken, seg_pos, seg_area,
            seg_normal, seg_atom, seg_ui,
            epsilon=epsilon, lmax=lmax,
        )
    else:
        seg_charge = ddcosmo_charges(
            atoms, coords, mulliken, seg_pos, seg_area, seg_ui,
            epsilon=epsilon, solver=solver,
        )

    seg_sigma = np.zeros_like(seg_charge)
    nonzero = seg_area > 1e-30
    seg_sigma[nonzero] = seg_charge[nonzero] / seg_area[nonzero]

    cavity_area = np.sum(seg_area)
    r_dot_n = np.sum((seg_pos - np.mean(coords, axis=0)) * seg_normal, axis=1)
    cavity_volume = np.abs(np.sum(r_dot_n * seg_area) / 3.0)

    return {
        'seg_pos': seg_pos, 'seg_area': seg_area,
        'seg_charge': seg_charge, 'seg_sigma': seg_sigma,
        'seg_normal': seg_normal, 'seg_atom': seg_atom,
        'seg_ui': seg_ui, 'mulliken_charges': mulliken,
        'cavity_area': cavity_area, 'cavity_volume': cavity_volume,
        'n_seg': len(seg_pos),
    }
