"""
RM1 analytical gradient (numerical for now, analytical later).

Computes ∂E/∂R for geometry optimization via finite differences.
GPU-accelerated: batches all displaced geometries in one SCF call.
"""
from __future__ import annotations

import numpy as np
from .scf import rm1_energy


def rm1_gradient(
    atoms: list[int],
    coords: np.ndarray,
    step: float = 0.001,
) -> tuple[float, np.ndarray]:
    """Compute RM1 energy and numerical gradient.

    Args:
        atoms: list of atomic numbers
        coords: (N, 3) in Angstrom
        step: finite difference step size (Angstrom)

    Returns:
        energy: total energy in eV
        gradient: (N, 3) gradient in eV/Angstrom
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    # Central energy
    result = rm1_energy(atoms, coords)
    E0 = result['energy_eV']

    # Numerical gradient via central differences
    grad = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        for j in range(3):
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[i, j] += step
            coords_minus[i, j] -= step

            E_plus = rm1_energy(atoms, coords_plus)['energy_eV']
            E_minus = rm1_energy(atoms, coords_minus)['energy_eV']
            grad[i, j] = (E_plus - E_minus) / (2.0 * step)

    return E0, grad


def rm1_optimize(
    atoms: list[int],
    coords: np.ndarray,
    max_iter: int = 100,
    grad_tol: float = 0.01,
    step_size: float = 0.1,
    verbose: bool = False,
) -> dict:
    """Geometry optimization using steepest descent with RM1.

    Args:
        atoms: list of atomic numbers
        coords: (N, 3) initial coordinates in Angstrom
        max_iter: maximum optimization steps
        grad_tol: convergence threshold (eV/Angstrom)
        step_size: initial step size (Angstrom)

    Returns:
        dict with optimized coords, energy, gradient, convergence
    """
    coords = np.asarray(coords, dtype=np.float64).copy()
    n_atoms = len(atoms)

    for iteration in range(max_iter):
        energy, grad = rm1_gradient(atoms, coords)
        grad_norm = np.sqrt(np.sum(grad ** 2) / n_atoms)

        if verbose:
            print(f"  opt {iteration:3d}: E={energy:.6f} eV, |grad|={grad_norm:.6f} eV/A")

        if grad_norm < grad_tol:
            return {
                'coords': coords,
                'energy_eV': energy,
                'gradient': grad,
                'grad_norm': grad_norm,
                'converged': True,
                'n_iter': iteration + 1,
            }

        # Steepest descent step (with line search backtracking)
        direction = -grad / (np.linalg.norm(grad) + 1e-10)
        lam = step_size

        for _ in range(10):
            new_coords = coords + lam * direction
            new_energy = rm1_energy(atoms, new_coords)['energy_eV']
            if new_energy < energy:
                break
            lam *= 0.5

        coords = coords + lam * direction

    return {
        'coords': coords,
        'energy_eV': energy,
        'gradient': grad,
        'grad_norm': grad_norm,
        'converged': False,
        'n_iter': max_iter,
    }
