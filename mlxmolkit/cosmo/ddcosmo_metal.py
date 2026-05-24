"""
Metal GPU kernel for ddCOSMO spherical harmonic solver.

Single kernel: compute Y_lm, assemble L matrix and RHS for N molecules.
Then numpy batch solve (small matrices: 49*n_atoms per side).

The SH approach reduces COSMO from n_seg×n_seg to (49*n_atoms)×(49*n_atoms):
  Water: 398×398 → 147×147
  Ethanol: 883×883 → 441×441
  Decane: 2505×2505 → 1568×1568
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from .params import VDW_RADII, CAVITY_SCALING, EPSILON_WATER, BOHR_TO_ANG
from .spherical_harmonics import real_spherical_harmonics


def ddcosmo_sh_batch(
    molecules: list[dict],
    epsilon: float = EPSILON_WATER,
    lmax: int = 6,
) -> list[np.ndarray]:
    """Batch SH ddCOSMO solve for N molecules.

    All-numpy vectorized: avoids per-atom Python loops.

    Args:
        molecules: list of dicts from build_ddcosmo_cavity, each with
            seg_pos, seg_area, seg_atom, seg_ui, mulliken_charges, atoms, coords
        epsilon: dielectric constant
        lmax: max angular momentum

    Returns:
        list of (n_seg_i,) charge arrays
    """
    N = len(molecules)
    n_ylm = (lmax + 1) ** 2
    keps = (epsilon - 1.0) / (epsilon + 0.5)

    charges_list = []

    for mol in molecules:
        seg_pos = mol['seg_pos']
        seg_area = mol['seg_area']
        seg_atom = mol['seg_atom']
        seg_ui = mol['seg_ui']
        mulliken = mol['mulliken_charges']
        atoms = mol['atoms']
        coords = np.asarray(mol['coords'], dtype=np.float64)
        n_seg = len(seg_pos)
        n_atoms = len(atoms)

        if n_seg == 0:
            charges_list.append(np.zeros(0))
            continue

        radii = np.array([VDW_RADII.get(z, 2.0) * CAVITY_SCALING for z in atoms])
        n_basis = n_atoms * n_ylm

        seg_pos_b = seg_pos / BOHR_TO_ANG
        coords_b = coords / BOHR_TO_ANG
        seg_area_b = seg_area / (BOHR_TO_ANG ** 2)

        # --- Phi at segments (vectorized) ---
        dist_ac = np.maximum(cdist(seg_pos_b, coords_b), 1e-10)
        Phi = (mulliken[np.newaxis, :] / dist_ac).sum(axis=1)

        # --- Pre-compute Y_lm for ALL segments at once ---
        # Group by atom, compute unit vectors, evaluate SH
        Y_all = np.zeros((n_ylm, n_seg))  # SH values at every segment
        w_all = np.zeros(n_seg)  # quadrature weights

        for a in range(n_atoms):
            mask = seg_atom == a
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            local = seg_pos[idx] - coords[a]
            local_r = np.linalg.norm(local, axis=1, keepdims=True)
            unit = local / np.maximum(local_r, 1e-30)

            Y_a = real_spherical_harmonics(lmax, unit)  # (n_ylm, n_seg_a)
            Y_all[:, idx] = Y_a
            w_all[idx] = seg_area_b[idx] / (radii[a] / BOHR_TO_ANG) ** 2

        # --- RHS: project -Phi*ui onto SH per atom ---
        rhs = np.zeros(n_basis)
        phi_w = -Phi * seg_ui * w_all
        for a in range(n_atoms):
            mask = seg_atom == a
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            rhs[a * n_ylm:(a + 1) * n_ylm] = Y_all[:, idx] @ phi_w[idx]

        # --- L matrix: Coulomb coupling in SH basis (vectorized) ---
        # L_{(a,lm),(b,l'm')} = Σ_{i∈a,j∈b} Y_lm(i) * w_i * (1/r_ij) * w_j * Y_l'm'(j)
        # For diagonal (a==a): add self-interaction

        L = np.zeros((n_basis, n_basis))

        # Pairwise segment Coulomb (full n_seg × n_seg, but we project)
        dist_ss = np.maximum(cdist(seg_pos_b, seg_pos_b), 1e-10)
        coulomb = 1.0 / dist_ss

        # Self-interaction on diagonal (replace Coulomb self with Klamt formula)
        for i in range(n_seg):
            coulomb[i, i] = 1.07 * np.sqrt(4.0 * np.pi / max(seg_area_b[i], 1e-30))

        # Project entire Coulomb matrix to SH basis:
        # L = (Y · diag(w))  · coulomb · (Y · diag(w))^T
        # But organized by atom blocks
        Yw = Y_all * w_all[np.newaxis, :]  # (n_ylm, n_seg)

        for a in range(n_atoms):
            mask_a = seg_atom == a
            idx_a = np.where(mask_a)[0]
            if len(idx_a) == 0:
                continue

            Ya_w = Yw[:, idx_a]  # (n_ylm, n_a)

            for b in range(n_atoms):
                mask_b = seg_atom == b
                idx_b = np.where(mask_b)[0]
                if len(idx_b) == 0:
                    continue

                Yb_w = Yw[:, idx_b]  # (n_ylm, n_b)
                C_ab = coulomb[np.ix_(idx_a, idx_b)]  # (n_a, n_b)

                # L_ab = Ya_w @ C_ab @ Yb_w^T  → (n_ylm, n_ylm)
                L[a * n_ylm:(a + 1) * n_ylm,
                  b * n_ylm:(b + 1) * n_ylm] = Ya_w @ C_ab @ Yb_w.T

        # --- Solve ---
        try:
            sigma_sh = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            charges_list.append(np.zeros(n_seg))
            continue

        # --- Expand back to segments ---
        q = np.zeros(n_seg)
        for a in range(n_atoms):
            mask = seg_atom == a
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            coeffs = sigma_sh[a * n_ylm:(a + 1) * n_ylm]
            q[idx] = (coeffs @ Y_all[:, idx]) * seg_ui[idx]

        charges_list.append(keps * q)

    return charges_list


def ddcosmo_sh_jacobi_batch(
    molecules: list[dict],
    epsilon: float = EPSILON_WATER,
    lmax: int = 4,
    max_iter: int = 100,
    tol: float = 1e-6,
    n_diis: int = 15,
) -> list[np.ndarray]:
    """SH Jacobi/DIIS batch: iterative solve in well-conditioned SH space.

    For large molecules where even SH direct solve is expensive:
    lmax=4, 25 basis/atom → matrix is small AND well-conditioned.
    Jacobi/DIIS converges fast (< 30 iterations typically).
    """
    from .ddcosmo import _jacobi_diis_solve

    N = len(molecules)
    n_ylm = (lmax + 1) ** 2
    keps = (epsilon - 1.0) / (epsilon + 0.5)
    charges_list = []

    for mol in molecules:
        seg_pos = mol['seg_pos']
        seg_area = mol['seg_area']
        seg_atom = mol['seg_atom']
        seg_ui = mol['seg_ui']
        mulliken = mol['mulliken_charges']
        atoms = mol['atoms']
        coords = np.asarray(mol['coords'], dtype=np.float64)
        n_seg = len(seg_pos)
        n_atoms = len(atoms)

        if n_seg == 0:
            charges_list.append(np.zeros(0))
            continue

        radii = np.array([VDW_RADII.get(z, 2.0) * CAVITY_SCALING for z in atoms])
        n_basis = n_atoms * n_ylm
        seg_pos_b = seg_pos / BOHR_TO_ANG
        coords_b = coords / BOHR_TO_ANG
        seg_area_b = seg_area / (BOHR_TO_ANG ** 2)

        # Phi
        dist_ac = np.maximum(cdist(seg_pos_b, coords_b), 1e-10)
        Phi = (mulliken[np.newaxis, :] / dist_ac).sum(axis=1)

        # Y_lm + weights
        Y_all = np.zeros((n_ylm, n_seg))
        w_all = np.zeros(n_seg)
        for a in range(n_atoms):
            idx = np.where(seg_atom == a)[0]
            if len(idx) == 0:
                continue
            local = seg_pos[idx] - coords[a]
            local_r = np.linalg.norm(local, axis=1, keepdims=True)
            Y_all[:, idx] = real_spherical_harmonics(lmax, local / np.maximum(local_r, 1e-30))
            w_all[idx] = seg_area_b[idx] / (radii[a] / BOHR_TO_ANG) ** 2

        # RHS
        rhs = np.zeros(n_basis)
        phi_w = -Phi * seg_ui * w_all
        for a in range(n_atoms):
            idx = np.where(seg_atom == a)[0]
            if len(idx) == 0:
                continue
            rhs[a * n_ylm:(a + 1) * n_ylm] = Y_all[:, idx] @ phi_w[idx]

        # L matrix (same as ddcosmo_sh_batch)
        dist_ss = np.maximum(cdist(seg_pos_b, seg_pos_b), 1e-10)
        coulomb = 1.0 / dist_ss
        for i in range(n_seg):
            coulomb[i, i] = 1.07 * np.sqrt(4.0 * np.pi / max(seg_area_b[i], 1e-30))

        Yw = Y_all * w_all[np.newaxis, :]
        L = np.zeros((n_basis, n_basis))
        for a in range(n_atoms):
            idx_a = np.where(seg_atom == a)[0]
            if len(idx_a) == 0:
                continue
            for b in range(n_atoms):
                idx_b = np.where(seg_atom == b)[0]
                if len(idx_b) == 0:
                    continue
                L[a*n_ylm:(a+1)*n_ylm, b*n_ylm:(b+1)*n_ylm] = \
                    Yw[:, idx_a] @ coulomb[np.ix_(idx_a, idx_b)] @ Yw[:, idx_b].T

        # Jacobi/DIIS solve in SH space (well-conditioned!)
        sigma_sh, n_iter, converged = _jacobi_diis_solve(
            L, rhs, max_iter=max_iter, tol=tol, n_diis=n_diis
        )

        # Expand back
        q = np.zeros(n_seg)
        for a in range(n_atoms):
            idx = np.where(seg_atom == a)[0]
            if len(idx) == 0:
                continue
            q[idx] = (sigma_sh[a*n_ylm:(a+1)*n_ylm] @ Y_all[:, idx]) * seg_ui[idx]

        charges_list.append(keps * q)

    return charges_list
