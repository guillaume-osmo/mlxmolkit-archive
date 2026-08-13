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

import warnings

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

    # Fully buried
    chi[t <= 1.0 - eta] = 1.0

    # Transition region
    mask = (t > 1.0 - eta) & (t < 1.0)
    if np.any(mask):
        # Map onto [0, 1]
        s = (t[mask] - (1.0 - eta)) / eta
        # 5th-degree polynomial with zero 1st and 2nd derivatives at both ends
        chi[mask] = 1.0 - 10.0 * s ** 3 + 15.0 * s ** 4 - 6.0 * s ** 5

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
        pts = coords[i] + r_i * sphere_pts

        # Accumulated burial from every other sphere
        fi = np.zeros(n_pts)
        for j in range(n_atoms):
            if j == i:
                continue

            dists = np.linalg.norm(pts - coords[j], axis=1)
            t = dists / radii[j]
            fi += _switching_function(t, eta)

        # Exposure weight in [0, 1]
        ui = np.maximum(0.0, 1.0 - fi)

        # Drop segments that are effectively fully buried
        threshold = 0.001
        mask = ui > threshold

        if np.sum(mask) == 0:
            continue

        # Areas are weighted by the exposure — this is the ddCOSMO part
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
    tol: float = 1e-08,
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
    D_inv = 1.0 / np.diag(A)
    O = A.copy()
    np.fill_diagonal(O, 0.0)

    # Diagonal-only starting guess
    q = D_inv * rhs

    # DIIS history
    diis_q = []
    diis_err = []

    for iteration in range(max_iter):
        # Jacobi sweep
        residual = rhs - O @ q
        q_new = D_inv * residual

        # Convergence on the rms increment
        dq = q_new - q
        rms_dq = np.sqrt(np.mean(dq * dq))

        if rms_dq < tol:
            return q_new, iteration + 1, True

        # Let a couple of plain Jacobi steps build the history first
        if iteration >= 2:
            diis_q.append(q_new.copy())
            diis_err.append(dq.copy())

            if len(diis_q) > n_diis:
                diis_q.pop(0)
                diis_err.pop(0)

            nd = len(diis_q)
            if nd >= 2:
                # Error-overlap matrix with the Lagrange multiplier border
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
                    pass

        # Damped update
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
                'auto' always uses 'direct' (see below)

    Note on 'jacobi': the COSMO matrix pairs a 1/r off-diagonal block with a
    1.07*sqrt(4*pi/a) diagonal and is nowhere near diagonally dominant — the
    spectral radius of D^-1*O is ~33 for ethanol at 194 points, so the
    iteration diverges and DIIS only slows it down. If it is requested and
    fails to converge, this falls back to the direct solve rather than
    returning the diverged iterate.
    """
    n_seg = len(seg_pos)
    if n_seg == 0:
        return np.zeros(0)

    seg_pos_b = seg_pos / BOHR_TO_ANG
    seg_area_b = seg_area / BOHR_TO_ANG ** 2
    coords_b = coords / BOHR_TO_ANG

    # Segment-segment Coulomb matrix
    dist = cdist(seg_pos_b, seg_pos_b)
    np.fill_diagonal(dist, 1.0)
    A = 1.0 / dist
    np.fill_diagonal(A, 1.07 * np.sqrt(4.0 * np.pi / np.maximum(seg_area_b, 1e-30)))

    # Potential from the atomic point charges
    dist_ac = np.maximum(cdist(seg_pos_b, coords_b), 1e-10)
    Phi = (mulliken_charges[np.newaxis, :] / dist_ac).sum(axis=1)

    # Weight the rhs by the exposure — buried segments carry no charge
    rhs = -Phi * seg_ui

    # Pick a solver. 'auto' used to hand anything over 3000 segments to
    # Jacobi on the theory that the O(n²) matvec beats a dense factorisation;
    # that is only true if the iteration converges, and here it does not.
    if solver == 'auto':
        solver = 'direct'

    q = None
    if solver == 'jacobi':
        q, n_iter, converged = _jacobi_diis_solve(A, rhs, max_iter=200, tol=1e-08)
        if not converged:
            warnings.warn(
                f"Jacobi/DIIS did not converge in {n_iter} iterations on a "
                f"{n_seg}-segment COSMO matrix; falling back to a direct "
                f"solve. The COSMO matrix is not diagonally dominant, so "
                f"this is expected — prefer solver='direct' or 'sh'.",
                RuntimeWarning, stacklevel=2,
            )
            q = None

    if q is None:
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
    seg_area_b = seg_area / BOHR_TO_ANG ** 2
    coords_b = coords / BOHR_TO_ANG

    radii = np.array([VDW_RADII.get(z, 2.0) * CAVITY_SCALING for z in atoms])

    # Reference grid for the effective segment area
    sphere_pts, sphere_weights = get_lebedev_grid(194)

    # Harmonics on the reference grid
    Y = real_spherical_harmonics(lmax, sphere_pts)

    # Potential at every segment from the atomic point charges
    dist_ac = np.maximum(cdist(seg_pos_b, coords_b), 1e-10)
    Phi = (mulliken_charges[np.newaxis, :] / dist_ac).sum(axis=1)

    # Project the potential onto the per-atom SH basis
    rhs_sh = np.zeros(n_basis_total)

    # For each atom, take its own segments, express their directions in the
    # atom-local frame, and integrate Phi against Y_lm over that patch. The
    # ui weights fold the burial in, so buried segments drop out of the
    # projection smoothly instead of being cut off at a sharp radius
    # threshold.
    for a in range(n_atoms):
        mask = seg_atom == a
        if not np.any(mask):
            continue

        # Directions relative to this atom's centre
        local_pos = seg_pos[mask] - coords[a]
        local_r = np.linalg.norm(local_pos, axis=1, keepdims=True)
        local_unit = local_pos / np.maximum(local_r, 1e-30)

        # Harmonics at this atom's own segment directions
        Y_local = real_spherical_harmonics(lmax, local_unit)

        # Quadrature weight = area in units of the atom's own r²
        phi_weighted = Phi[mask] * seg_ui[mask]
        w_local = seg_area_b[mask] / (radii[a] / BOHR_TO_ANG) ** 2

        # rhs_{lm,a} = -Σ_i Y_{lm}(θ_i) w_i ui_i Phi_i
        rhs_sh[a * n_ylm:(a + 1) * n_ylm] = -Y_local @ (phi_weighted * w_local)

    # Coupling matrix in the SH basis.
    #
    # Diagonal blocks are the on-atom self-interaction; off-diagonal blocks
    # are the atom-atom segment Coulomb matrix projected onto the harmonic
    # basis from both sides.
    L = np.zeros((n_basis_total, n_basis_total))

    for a in range(n_atoms):

        # Self-interaction, same Klamt convention as the segment-basis solver
        # but with a_eff taken from the reference grid.
        r_a_b = radii[a] / BOHR_TO_ANG
        a_eff_b = 4.0 * np.pi * r_a_b ** 2 / len(sphere_pts)
        self_int = 1.07 * np.sqrt(4.0 * np.pi / a_eff_b)
        for lm in range(n_ylm):
            L[a * n_ylm + lm, a * n_ylm + lm] = self_int

        # Off-diagonal atom-atom blocks
        for b in range(n_atoms):
            if b == a:
                continue

            # Segments belonging to each atom
            mask_a = seg_atom == a
            mask_b = seg_atom == b
            if not np.any(mask_a) or not np.any(mask_b):
                continue

            pos_a_b = seg_pos_b[mask_a]
            pos_b_b = seg_pos_b[mask_b]

            # Segment-segment Coulomb between the two patches
            dist_ab = np.maximum(cdist(pos_a_b, pos_b_b), 1e-10)
            coulomb_ab = 1.0 / dist_ab

            # Harmonics in each atom's local frame
            local_a = seg_pos[mask_a] - coords[a]
            r_a = np.linalg.norm(local_a, axis=1, keepdims=True)
            Y_a = real_spherical_harmonics(lmax, local_a / np.maximum(r_a, 1e-30))

            local_b = seg_pos[mask_b] - coords[b]
            r_b = np.linalg.norm(local_b, axis=1, keepdims=True)
            Y_b = real_spherical_harmonics(lmax, local_b / np.maximum(r_b, 1e-30))

            w_a = seg_area_b[mask_a] / (radii[a] / BOHR_TO_ANG) ** 2
            w_b = seg_area_b[mask_b] / (radii[b] / BOHR_TO_ANG) ** 2

            # Sandwich the Coulomb block between both harmonic bases
            L_ab = (Y_a * w_a[np.newaxis, :]) @ coulomb_ab @ (Y_b * w_b[np.newaxis, :]).T

            L[a * n_ylm:(a + 1) * n_ylm,
              b * n_ylm:(b + 1) * n_ylm] = L_ab

    # Small dense solve in the SH basis
    sigma_sh = np.linalg.solve(L, rhs_sh)

    # Expand back onto the segments
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
    method: str = 'RM1',
) -> dict:
    """Complete ddCOSMO surface calculation.

    Args:
        solver: 'direct' (dense np.linalg.solve on segments),
                'sh' (spherical harmonic basis — smaller matrix),
                'jacobi' (iterative on segments; see the warning in
                ddcosmo_charges — it does not converge on COSMO matrices),
                'auto' (picks best for system size)
        lmax: max angular momentum for SH basis (6→49 per atom)
        method: NDDO method the density came from (see cavity.cosmo_surface)
    """
    from .cavity import _mulliken_charges
    from ..nddo.methods import get_params

    coords = np.asarray(coords, dtype=np.float64)

    seg_pos, seg_area, seg_normal, seg_atom, seg_ui = build_ddcosmo_cavity(
        atoms, coords, n_points_per_atom=n_points, eta=eta
    )

    method_params = get_params(method)
    n_basis_per = [method_params[z].n_basis for z in atoms]
    mulliken = _mulliken_charges(atoms, density, n_basis_per)

    if solver == 'sh':
        seg_charge = ddcosmo_charges_sh(
            atoms, coords, mulliken, seg_pos, seg_area,
            seg_normal, seg_atom, seg_ui,
            epsilon=epsilon, lmax=lmax
        )
    else:
        seg_charge = ddcosmo_charges(
            atoms, coords, mulliken, seg_pos, seg_area, seg_ui,
            epsilon=epsilon, solver=solver
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
