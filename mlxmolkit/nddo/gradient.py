"""
NDDO gradient and geometry optimizer (batch-capable).

Numerical gradient via central differences.
L-BFGS geometry optimizer.
Batch version: optimize N molecules simultaneously.
"""
from __future__ import annotations

import numpy as np
from .scf import nddo_energy, nddo_energy_batch


def nddo_gradient(
    atoms: list[int],
    coords: np.ndarray,
    step: float = 0.0005,
    method: str = 'RM1',
    analytical: bool = True,
    molecular_charge: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Compute energy and gradient (single molecule).

    Uses analytical (frozen-density) gradient by default — 6x faster.
    """
    if analytical:
        from .anal_grad import analytical_gradient
        result, grad = analytical_gradient(
            atoms,
            coords,
            method=method,
            molecular_charge=molecular_charge,
        )
        return result['energy_eV'], grad

    # Numerical fallback
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    result = nddo_energy(atoms, coords, method=method, molecular_charge=molecular_charge)
    E0 = result['energy_eV']

    grad = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        for j in range(3):
            cp = coords.copy(); cp[i, j] += step
            cm = coords.copy(); cm[i, j] -= step
            Ep = nddo_energy(atoms, cp, method=method, molecular_charge=molecular_charge)['energy_eV']
            Em = nddo_energy(atoms, cm, method=method, molecular_charge=molecular_charge)['energy_eV']
            grad[i, j] = (Ep - Em) / (2.0 * step)

    return E0, grad


def nddo_gradient_batch(
    molecules: list[tuple[list[int], np.ndarray]],
    step: float = 0.0005,
    method: str = 'RM1',
    molecular_charges: list[float] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute energies and numerical gradients for N molecules (batched).

    All displaced geometries are run in one batch call for efficiency.

    Returns:
        energies: (N,) array of energies in eV
        gradients: list of (n_atoms_i, 3) gradient arrays
    """
    normalized_molecules = []
    tuple_charges = []
    for molecule in molecules:
        if len(molecule) == 2:
            atoms, coords = molecule
            charge = 0.0
        elif len(molecule) == 3:
            atoms, coords, charge = molecule
        else:
            raise ValueError("molecules must contain (atoms, coords) or (atoms, coords, charge) tuples")
        normalized_molecules.append((atoms, coords))
        tuple_charges.append(float(charge))

    if molecular_charges is None:
        charge_values = tuple_charges
    else:
        if len(molecular_charges) != len(normalized_molecules):
            raise ValueError("molecular_charges must match the number of molecules")
        charge_values = [float(charge) for charge in molecular_charges]

    molecules = normalized_molecules
    N = len(molecules)

    # The reference geometries are solved in a single batched SCF; each
    # molecule's gradient then comes from its own converged density by the
    # frozen-density route.
    #
    # This used to batch the *numerical* gradient instead: a full SCF at each
    # of the 6N+1 displaced geometries, all in one dispatch. Batching does not
    # rescue that — 10 small molecules cost 13.4 s each batched against 1.5 s
    # each solved one at a time by the frozen-density route, because the
    # batched version was doing 55 full SCFs per molecule where the other does
    # one. Batch the cheap algorithm, not the expensive one.
    from .anal_grad import analytical_gradient

    scf = nddo_energy_batch(molecules, method=method, use_metal=True,
                            molecular_charges=charge_values)

    energies = np.zeros(N)
    gradients = []
    for idx, ((atoms, coords), result) in enumerate(zip(molecules, scf)):
        res, grad = analytical_gradient(
            atoms, np.asarray(coords, dtype=np.float64), method=method,
            molecular_charge=charge_values[idx], scf_result=result,
        )
        energies[idx] = res['energy_eV']
        gradients.append(grad)

    return energies, gradients


def nddo_optimize_batch(
    molecules: list[tuple[list[int], np.ndarray]],
    # Must match nddo_optimize's default. At 50 against the single path's 200,
    # menthol came out 0.385 kcal/mol apart depending on which entry point you
    # called — 9x the MOPAC agreement — because batch stopped it short. See #28.
    max_iter: int = 200,
    grad_tol: float = 0.005,
    method: str = 'RM1',
    verbose: bool = False,
    molecular_charges: list[float] | None = None,
) -> list[dict]:
    """L-BFGS geometry optimization for N molecules simultaneously.

    Pipeline: all molecules share the same opt iteration count,
    converged molecules are skipped.

    Returns:
        list of result dicts with optimized coords and energies
    """
    normalized_molecules = []
    tuple_charges = []
    for molecule in molecules:
        if len(molecule) == 2:
            atoms, coords = molecule
            charge = 0.0
        elif len(molecule) == 3:
            atoms, coords, charge = molecule
        else:
            raise ValueError("molecules must contain (atoms, coords) or (atoms, coords, charge) tuples")
        normalized_molecules.append((atoms, coords))
        tuple_charges.append(float(charge))
    if molecular_charges is None:
        charge_values = tuple_charges
    else:
        if len(molecular_charges) != len(normalized_molecules):
            raise ValueError("molecular_charges must match the number of molecules")
        charge_values = [float(charge) for charge in molecular_charges]

    molecules = normalized_molecules
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
    energies, gradients = nddo_gradient_batch(
        mols_current,
        method=method,
        molecular_charges=charge_values,
    )
    grad_flats = [g.flatten() for g in gradients]

    for iteration in range(max_iter):
        # Check convergence
        for i in range(N):
            if converged[i]:
                continue
            g_rms = np.sqrt(np.mean(grad_flats[i] ** 2))
            if g_rms < grad_tol:
                converged[i] = True
                # iteration + 1, matching nddo_optimize, which returns
                # `iteration + 1` from the same top-of-loop check. Recording
                # `iteration` made batch report one fewer for an identical
                # optimization.
                n_iter_arr[i] = iteration + 1

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
        steps_taken: dict[int, float] = {}
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
            trial_results = nddo_energy_batch(
                trial_mols,
                method=method,
                use_metal=True,
                molecular_charges=[charge_values[i] for i in trial_indices],
            )

            # Accept or reduce step per molecule, remembering which was used:
            # the L-BFGS update below needs the displacement that actually
            # happened, and a fallback step recorded as a full one poisons the
            # curvature pair.
            for k, i in enumerate(trial_indices):
                new_E = trial_results[k]['energy_eV']
                n_at = len(atoms_list[i])
                accepted = np.isfinite(new_E) and new_E < energies[i]
                steps_taken[i] = step if accepted else 0.01
                coords_list[i] = (coords_list[i]
                                  + steps_taken[i] * directions[i].reshape(n_at, 3))

        # New gradients (batch)
        old_grad_flats = [g.copy() for g in grad_flats]
        mols_current = [(atoms_list[i], coords_list[i]) for i in range(N)]
        energies, gradients = nddo_gradient_batch(
            mols_current,
            method=method,
            molecular_charges=charge_values,
        )
        grad_flats = [g.flatten() for g in gradients]

        # L-BFGS history update
        for i in range(N):
            if converged[i]:
                continue
            n_at = len(atoms_list[i])
            # s_k must be the displacement actually applied above. This read
            # `step * d if energies[i] < energies[i] + 1 else 0.01 * d`, whose
            # condition is x < x + 1 — always true — so a molecule that fell
            # back to the 0.01 step still recorded a full one, 100x too large,
            # and every curvature pair after a rejected step was wrong.
            s_k = steps_taken.get(i, 0.0) * directions[i]
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
    final_results = nddo_energy_batch(
        [(atoms_list[i], coords_list[i]) for i in range(N)],
        method=method, use_metal=True,
        molecular_charges=charge_values,
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


def _optimize_result(result, coords, energy, grad, g_rms, converged, n_iter, method):
    """Assemble nddo_optimize's return dict.

    The SCF `result` is merged **first** so the optimizer's own keys win. It
    used to be splatted last, which let the SCF's `converged`, `n_iter` and
    `method` overwrite the optimizer's — so a geometry optimization that ran
    out of iterations still reported `converged=True`, because that was the
    SCF's flag for the final single-point. The SCF's values are kept under
    `scf_*` rather than dropped.

    `converged`/`n_iter` describe the **optimization**, which is what a caller
    of a function named `optimize` means by them. `opt_converged`/`opt_n_iter`
    are retained as explicit aliases.
    """
    out = {k: v for k, v in result.items() if k != 'coords'}
    if 'converged' in out:
        out['scf_converged'] = out['converged']
    if 'n_iter' in out:
        out['scf_n_iter'] = out['n_iter']
    out.update(
        coords=coords,
        energy_eV=energy,
        heat_of_formation_kcal=result['heat_of_formation_kcal'],
        gradient=grad,
        grad_rms=g_rms,
        converged=converged,
        opt_converged=converged,
        n_iter=n_iter,
        opt_n_iter=n_iter,
        method=method,
    )
    return out


def nddo_optimize(
    atoms: list[int],
    coords: np.ndarray,
    # 200, not 50. The loop returns as soon as g_rms < grad_tol, so this is a
    # bound on the work rather than a cost: chlorobenzene still exits at 16.
    # At 50, menthol ran out of iterations at g_rms=0.014 — 3x the tolerance —
    # while its energy was still falling, and then reported success because of
    # the key collision above. It converges at 94. See #28.
    max_iter: int = 200,
    grad_tol: float = 0.005,
    method: str = 'RM1',
    verbose: bool = False,
    molecular_charge: float = 0.0,
) -> dict:
    """L-BFGS geometry optimization using analytical gradient.

    Returns a dict whose `converged` and `n_iter` refer to the geometry
    optimization; the SCF's own values are under `scf_converged`/`scf_n_iter`.
    """
    from .anal_grad import analytical_gradient

    coords = np.asarray(coords, dtype=np.float64).copy()
    n_atoms = len(atoms)
    n_vars = n_atoms * 3

    m = 8
    s_hist, y_hist, rho_hist = [], [], []

    result, grad = analytical_gradient(
        atoms,
        coords,
        method=method,
        molecular_charge=molecular_charge,
    )
    energy = result['energy_eV']
    grad_flat = grad.flatten()

    for iteration in range(max_iter):
        g_rms = np.sqrt(np.mean(grad_flat ** 2))
        if verbose and (iteration % 5 == 0 or g_rms < grad_tol):
            print(f"  opt {iteration:3d}: E={energy:.6f}, Hf={result['heat_of_formation_kcal']:.2f}, |g|={g_rms:.5f}")

        if g_rms < grad_tol:
            return _optimize_result(result, coords, energy, grad, g_rms,
                                    converged=True, n_iter=iteration + 1,
                                    method=method)

        # L-BFGS direction
        q = grad_flat.copy()
        alphas = []
        for k in range(len(s_hist) - 1, -1, -1):
            a = rho_hist[k] * np.dot(s_hist[k], q)
            alphas.append(a)
            q -= a * y_hist[k]

        gamma = (np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1])
                 if s_hist else 0.1)
        r = gamma * q
        alphas.reverse()
        for k in range(len(s_hist)):
            beta = rho_hist[k] * np.dot(y_hist[k], r)
            r += (alphas[k] - beta) * s_hist[k]

        direction = -r
        if np.dot(grad_flat, direction) > 0:
            direction = -grad_flat
            step = 0.05
        else:
            step = 1.0

        # Backtracking line search
        for ls in range(15):
            new_coords = coords + step * direction.reshape(n_atoms, 3)
            new_result, new_grad = analytical_gradient(
                atoms,
                new_coords,
                method=method,
                molecular_charge=molecular_charge,
            )
            if new_result['energy_eV'] <= energy + 1e-4 * step * np.dot(grad_flat, direction):
                break
            step *= 0.5
        else:
            step = 1e-4
            new_coords = coords + step * direction.reshape(n_atoms, 3)
            new_result, new_grad = analytical_gradient(
                atoms,
                new_coords,
                method=method,
                molecular_charge=molecular_charge,
            )

        old_grad = grad_flat.copy()
        s_k = step * direction
        coords = new_coords
        result = new_result
        energy = result['energy_eV']
        grad = new_grad
        grad_flat = grad.flatten()

        y_k = grad_flat - old_grad
        sy = np.dot(s_k, y_k)
        if sy > 1e-10:
            s_hist.append(s_k)
            y_hist.append(y_k)
            rho_hist.append(1.0 / sy)
            if len(s_hist) > m:
                s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)

    # Ran out of iterations. This used to hardcode 'converged': True.
    return _optimize_result(result, coords, energy, grad,
                            np.sqrt(np.mean(grad_flat ** 2)),
                            converged=False, n_iter=max_iter, method=method)


# Backward-compatible aliases
def rm1_gradient(atoms, coords, step=0.001):
    return nddo_gradient(atoms, coords, step=step, method='RM1')

def rm1_optimize(atoms, coords, max_iter=None, grad_tol=None, step_size=0.1,
                 verbose=False):
    """Deprecated alias for :func:`nddo_optimize` with method='RM1'.

    Used to hardcode max_iter=100 and grad_tol=0.01 — a third set of defaults
    alongside the single and batch paths, and a tolerance 2x looser than
    either. Defaults now delegate, so all three agree. `step_size` is accepted
    and ignored, as it always was.
    """
    kwargs = {}
    if max_iter is not None:
        kwargs['max_iter'] = max_iter
    if grad_tol is not None:
        kwargs['grad_tol'] = grad_tol
    return nddo_optimize(atoms, coords, method='RM1', verbose=verbose, **kwargs)
