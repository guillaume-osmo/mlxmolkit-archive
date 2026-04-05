"""
NDDO gradient and geometry optimizer (batch-capable).

Numerical gradient via central differences.
L-BFGS geometry optimizer.
Batch version: optimize N molecules simultaneously.
"""
from __future__ import annotations

import numpy as np
from .scf import rm1_energy, rm1_energy_batch


def nddo_gradient(
    atoms: list[int],
    coords: np.ndarray,
    step: float = 0.001,
    method: str = 'RM1',
) -> tuple[float, np.ndarray]:
    """Compute energy and numerical gradient (single molecule)."""
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    result = rm1_energy(atoms, coords, method=method)
    E0 = result['energy_eV']

    grad = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        for j in range(3):
            cp = coords.copy(); cp[i, j] += step
            cm = coords.copy(); cm[i, j] -= step
            Ep = rm1_energy(atoms, cp, method=method)['energy_eV']
            Em = rm1_energy(atoms, cm, method=method)['energy_eV']
            grad[i, j] = (Ep - Em) / (2.0 * step)

    return E0, grad


def nddo_gradient_batch(
    molecules: list[tuple[list[int], np.ndarray]],
    step: float = 0.0005,
    method: str = 'RM1',
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute energies and numerical gradients for N molecules (batched).

    All displaced geometries are run in one batch call for efficiency.

    Returns:
        energies: (N,) array of energies in eV
        gradients: list of (n_atoms_i, 3) gradient arrays
    """
    N = len(molecules)

    # Build all displaced geometries
    # For each molecule: 1 center + 6*n_atoms displaced = 6*n_atoms+1 evals
    all_mols = []
    mol_info = []  # (mol_idx, n_atoms, center_idx, grad_start_idx)

    idx = 0
    for mol_idx, (atoms, coords) in enumerate(molecules):
        coords = np.asarray(coords, dtype=np.float64)
        n_at = len(atoms)
        center_idx = idx

        # Center geometry
        all_mols.append((atoms, coords))
        idx += 1

        grad_start = idx
        # +/- displacements
        for i in range(n_at):
            for j in range(3):
                cp = coords.copy(); cp[i, j] += step
                all_mols.append((atoms, cp))
                idx += 1
                cm = coords.copy(); cm[i, j] -= step
                all_mols.append((atoms, cm))
                idx += 1

        mol_info.append((mol_idx, n_at, center_idx, grad_start))

    # One batch call for ALL displaced geometries
    results = rm1_energy_batch(all_mols, method=method, use_metal=True)

    # Extract energies and gradients
    energies = np.zeros(N)
    gradients = []

    for mol_idx, n_at, center_idx, grad_start in mol_info:
        energies[mol_idx] = results[center_idx]['energy_eV']

        grad = np.zeros((n_at, 3))
        k = grad_start
        for i in range(n_at):
            for j in range(3):
                Ep = results[k]['energy_eV']
                Em = results[k + 1]['energy_eV']
                grad[i, j] = (Ep - Em) / (2.0 * step)
                k += 2

        gradients.append(grad)

    return energies, gradients


def nddo_optimize_batch(
    molecules: list[tuple[list[int], np.ndarray]],
    max_iter: int = 50,
    grad_tol: float = 0.005,
    method: str = 'RM1',
    verbose: bool = False,
) -> list[dict]:
    """L-BFGS geometry optimization for N molecules simultaneously.

    Pipeline: all molecules share the same opt iteration count,
    converged molecules are skipped.

    Returns:
        list of result dicts with optimized coords and energies
    """
    N = len(molecules)
    atoms_list = [atoms for atoms, _ in molecules]
    coords_list = [np.asarray(c, dtype=np.float64).copy() for _, c in molecules]

    # L-BFGS state per molecule
    m = 6  # history size
    s_hist = [[] for _ in range(N)]
    y_hist = [[] for _ in range(N)]
    rho_hist = [[] for _ in range(N)]

    converged = np.zeros(N, dtype=bool)
    n_iter_arr = np.full(N, max_iter, dtype=np.int32)

    # Initial gradients (batch)
    mols_current = [(atoms_list[i], coords_list[i]) for i in range(N)]
    energies, gradients = nddo_gradient_batch(mols_current, method=method)
    grad_flats = [g.flatten() for g in gradients]

    for iteration in range(max_iter):
        # Check convergence
        for i in range(N):
            if converged[i]:
                continue
            g_rms = np.sqrt(np.mean(grad_flats[i] ** 2))
            if g_rms < grad_tol:
                converged[i] = True
                n_iter_arr[i] = iteration

        if verbose and (iteration % 5 == 0 or np.all(converged)):
            n_conv = np.sum(converged)
            print(f"  geom opt {iteration:3d}: {n_conv}/{N} converged")

        if np.all(converged):
            break

        # L-BFGS direction for each active molecule
        directions = [None] * N
        for i in range(N):
            if converged[i]:
                continue

            q = grad_flats[i].copy()
            alphas = []

            for k in range(len(s_hist[i]) - 1, -1, -1):
                a = rho_hist[i][k] * np.dot(s_hist[i][k], q)
                alphas.append(a)
                q -= a * y_hist[i][k]

            if len(s_hist[i]) > 0:
                sy = np.dot(s_hist[i][-1], y_hist[i][-1])
                yy = np.dot(y_hist[i][-1], y_hist[i][-1])
                gamma = sy / yy if yy > 1e-10 else 1.0
            else:
                gamma = 0.1

            r = gamma * q
            alphas.reverse()
            for k in range(len(s_hist[i])):
                beta = rho_hist[i][k] * np.dot(y_hist[i][k], r)
                r += (alphas[k] - beta) * s_hist[i][k]

            d = -r
            slope = np.dot(grad_flats[i], d)
            if slope > 0:
                d = -grad_flats[i]
            directions[i] = d

        # Line search: try step=1.0 for all active molecules (batch)
        step = 1.0
        trial_mols = []
        trial_indices = []
        for i in range(N):
            if converged[i]:
                continue
            n_at = len(atoms_list[i])
            new_c = coords_list[i] + step * directions[i].reshape(n_at, 3)
            trial_mols.append((atoms_list[i], new_c))
            trial_indices.append(i)

        if len(trial_mols) > 0:
            trial_results = rm1_energy_batch(trial_mols, method=method, use_metal=True)

            # Accept or reduce step per molecule
            for k, i in enumerate(trial_indices):
                new_E = trial_results[k]['energy_eV']
                n_at = len(atoms_list[i])
                if np.isfinite(new_E) and new_E < energies[i]:
                    coords_list[i] = coords_list[i] + step * directions[i].reshape(n_at, 3)
                else:
                    # Tiny step as fallback
                    coords_list[i] = coords_list[i] + 0.01 * directions[i].reshape(n_at, 3)

        # New gradients (batch)
        old_grad_flats = [g.copy() for g in grad_flats]
        mols_current = [(atoms_list[i], coords_list[i]) for i in range(N)]
        energies, gradients = nddo_gradient_batch(mols_current, method=method)
        grad_flats = [g.flatten() for g in gradients]

        # L-BFGS history update
        for i in range(N):
            if converged[i]:
                continue
            n_at = len(atoms_list[i])
            # s_k = actual displacement
            s_k = step * directions[i] if energies[i] < energies[i] + 1 else 0.01 * directions[i]
            y_k = grad_flats[i] - old_grad_flats[i]
            sy = np.dot(s_k, y_k)
            if sy > 1e-10:
                s_hist[i].append(s_k)
                y_hist[i].append(y_k)
                rho_hist[i].append(1.0 / sy)
                if len(s_hist[i]) > m:
                    s_hist[i].pop(0)
                    y_hist[i].pop(0)
                    rho_hist[i].pop(0)

    # Final energies
    final_results = rm1_energy_batch(
        [(atoms_list[i], coords_list[i]) for i in range(N)],
        method=method, use_metal=True,
    )

    results = []
    for i in range(N):
        r = final_results[i]
        r['coords'] = coords_list[i]
        r['opt_converged'] = bool(converged[i])
        r['opt_n_iter'] = int(n_iter_arr[i])
        r['opt_grad_rms'] = float(np.sqrt(np.mean(grad_flats[i] ** 2)))
        results.append(r)

    return results


def nddo_optimize(
    atoms: list[int],
    coords: np.ndarray,
    max_iter: int = 50,
    grad_tol: float = 0.005,
    method: str = 'RM1',
    verbose: bool = False,
) -> dict:
    """L-BFGS geometry optimization (single molecule). See nddo_optimize_batch."""
    results = nddo_optimize_batch(
        [(atoms, coords)], max_iter=max_iter, grad_tol=grad_tol,
        method=method, verbose=verbose,
    )
    return results[0]


# Backward-compatible aliases
def rm1_gradient(atoms, coords, step=0.001):
    return nddo_gradient(atoms, coords, step=step, method='RM1')

def rm1_optimize(atoms, coords, max_iter=100, grad_tol=0.01, step_size=0.1, verbose=False):
    return nddo_optimize(atoms, coords, max_iter=max_iter, grad_tol=grad_tol,
                        method='RM1', verbose=verbose)
