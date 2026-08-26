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
    max_iter: int = 500,
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


# --- Eigenvector following (EF) ------------------------------------------
# Mirrors MOPAC's ef.F90: a diagonal initial Hessian (gethes), BFGS/Powell
# updates (updhes), and a P-RFO step under a trust radius (formd). MOPAC's
# header credits "JACK SIMONS P-RFO ALGORITHM AS IMPLEMENTED BY JON BAKER
# (J.COMP.CHEM. 7, 385)".
#
# Why it pays here and not in MOPAC: MOPAC's own wall times have L-BFGS
# beating EF (menthol 0.28 s vs 0.37 s) because the eigensolve shows up when
# the SCF is cheap. Our gradients cost 79-373 ms and swamp a 93x93
# eigendecomposition, so fewer cycles converts directly into less wall time.

_EF_DMAX0 = 0.2      # ef.F90: dmax = 0.2d0
_EF_DDMAX = 0.5      # ef.F90: ddmax = 0.5d0
_EF_DDMIN = 1e-3
_EF_H0 = 16.0        # initial diagonal curvature, eV/Angstrom^2


def _ef_rfo_step(hessian, grad, dmax):
    """P-RFO step: lowest eigenvector of [[H, g], [g^T, 0]], trust-clipped."""
    n = len(grad)
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = hessian
    augmented[:n, n] = grad
    augmented[n, :n] = grad
    _w, vectors = np.linalg.eigh(augmented)
    v = vectors[:, 0]
    # v[n] is the augmented coordinate; near zero means the RFO step is
    # undefined, so fall back on steepest descent rather than dividing by it.
    step = -grad if abs(v[n]) < 1e-12 else v[:n] / v[n]
    norm = float(np.linalg.norm(step))
    if norm > dmax:
        step = step * (dmax / norm)
    return step, norm


def _ef_update_hessian(hessian, s_k, y_k):
    """BFGS while the curvature is positive, Powell (PSB) otherwise."""
    sy = float(np.dot(s_k, y_k))
    hs = hessian @ s_k
    shs = float(np.dot(s_k, hs))
    if sy > 1e-10 and shs > 1e-10:
        return hessian - np.outer(hs, hs) / shs + np.outer(y_k, y_k) / sy
    ss = float(np.dot(s_k, s_k))
    if ss < 1e-14:
        return hessian
    d = y_k - hs
    return (hessian + (np.outer(d, s_k) + np.outer(s_k, d)) / ss
            - float(np.dot(d, s_k)) * np.outer(s_k, s_k) / (ss * ss))


def _ef_optimize(atoms, coords, max_iter, grad_tol, method, verbose,
                 molecular_charge, h0=_EF_H0):
    """Geometry optimization by eigenvector following. See nddo_optimize."""
    from .anal_grad import analytical_gradient

    coords = np.asarray(coords, dtype=np.float64).copy()
    n_atoms = len(atoms)
    hessian = np.eye(n_atoms * 3) * h0
    dmax = _EF_DMAX0

    result, grad = analytical_gradient(atoms, coords, method=method,
                                       molecular_charge=molecular_charge)
    energy = result['energy_eV']
    grad_flat = grad.flatten()

    for iteration in range(max_iter):
        g_rms = float(np.sqrt(np.mean(grad_flat ** 2)))
        if verbose and iteration % 5 == 0:
            print(f"  ef {iteration:3d}: E={energy:.6f}, |g|={g_rms:.5f}, "
                  f"trust={dmax:.3f}")
        if g_rms < grad_tol:
            return _optimize_result(result, coords, energy, grad, g_rms,
                                    converged=True, n_iter=iteration + 1,
                                    method=method)

        step, raw_norm = _ef_rfo_step(hessian, grad_flat, dmax)
        predicted = float(np.dot(grad_flat, step)
                          + 0.5 * step @ hessian @ step)

        new_coords = coords + step.reshape(n_atoms, 3)
        new_result, new_grad = analytical_gradient(
            atoms, new_coords, method=method,
            molecular_charge=molecular_charge,
            P_init=result.get('density'))
        actual = new_result['energy_eV'] - energy

        if actual > 0:
            # Uphill: shrink the trust radius and retry from the same point.
            shrunk = max(_EF_DDMIN, min(dmax, float(np.linalg.norm(step))) / 2)
            if shrunk > _EF_DDMIN:
                dmax = shrunk
                continue
            dmax = shrunk           # already at the floor; take the step
        else:
            ratio = actual / predicted if abs(predicted) > 1e-12 else 0.0
            if ratio > 0.75 and raw_norm > dmax * 0.99:
                dmax = min(_EF_DDMAX, dmax * 1.5)
            elif ratio < 0.25:
                dmax = max(_EF_DDMIN, dmax * 0.5)

        hessian = _ef_update_hessian(hessian, step,
                                     new_grad.flatten() - grad_flat)
        coords, result = new_coords, new_result
        energy = result['energy_eV']
        grad = new_grad
        grad_flat = grad.flatten()

    return _optimize_result(result, coords, energy, grad,
                            float(np.sqrt(np.mean(grad_flat ** 2))),
                            converged=False, n_iter=max_iter, method=method)


def _strong_wolfe(evaluate, phi0, dphi0, c1=1e-4, c2=0.9, max_evals=12,
                  step_init=1.0, step_max=16.0):
    """Strong-Wolfe line search: Nocedal & Wright Alg. 3.5 with 3.6 'zoom'.

    Backtracking-Armijo alone only guarantees *decrease*. It can accept a step
    so short that y_k = g_new - g_old barely moves, which makes the curvature
    pair (s_k, y_k) nearly singular and degrades the L-BFGS Hessian
    approximation. The curvature condition |phi'(a)| <= c2 |phi'(0)| is what
    rules that out, and it is why MOPAC's own L-BFGS uses a Wolfe search
    (`lnsrlb` -> `dcsrch`, ftol=1e-3, gtol=0.9) rather than backtracking.

    Normally the curvature test costs an extra gradient per trial. Here it is
    free: `analytical_gradient` returns the energy and the gradient together,
    so every trial point already has both.

    Args:
        evaluate: step -> (phi, dphi, payload); phi is the energy, dphi the
            directional derivative g(x + step*d) . d, payload whatever the
            caller needs to reuse the accepted point.
        phi0, dphi0: energy and directional derivative at step 0. dphi0 must
            be negative (a descent direction).
        c1, c2: Armijo and curvature constants, 0 < c1 < c2 < 1.
        max_evals: cap on trial points, since each is a full gradient.

    Returns:
        (step, payload) for a step satisfying both conditions, or the best
        point that at least decreased the energy, or None if nothing did.
    """
    if dphi0 >= 0:
        return None

    def armijo(step, phi):
        return phi <= phi0 + c1 * step * dphi0

    def curvature(dphi):
        return abs(dphi) <= c2 * abs(dphi0)

    evals = 0
    best = None                      # fallback: any step that decreased phi

    def record(step, phi, payload):
        nonlocal best
        if phi < phi0 and (best is None or phi < best[1]):
            best = (step, phi, payload)

    def zoom(lo, hi, phi_lo):
        """Shrink a bracket known to contain an acceptable step."""
        nonlocal evals
        while evals < max_evals:
            step = 0.5 * (lo + hi)          # bisection: robust, no extra evals
            if step <= 1e-12:
                return None
            phi, dphi, payload = evaluate(step)
            evals += 1
            record(step, phi, payload)
            if not armijo(step, phi) or phi >= phi_lo:
                hi = step
            else:
                if curvature(dphi):
                    return step, payload
                if dphi * (hi - lo) >= 0:
                    hi = lo
                lo, phi_lo = step, phi
        return None

    prev_step, prev_phi = 0.0, phi0
    step = step_init
    while evals < max_evals:
        phi, dphi, payload = evaluate(step)
        evals += 1
        record(step, phi, payload)

        if not armijo(step, phi) or (evals > 1 and phi >= prev_phi):
            found = zoom(prev_step, step, prev_phi)
            return found if found is not None else _wolfe_fallback(best)

        if curvature(dphi):
            return step, payload

        if dphi >= 0:
            found = zoom(step, prev_step, phi)
            return found if found is not None else _wolfe_fallback(best)

        prev_step, prev_phi = step, phi
        step = min(2.0 * step, step_max)

    return _wolfe_fallback(best)


def _wolfe_fallback(best):
    """Neither condition met within budget — keep any step that went downhill.

    Losing the curvature condition costs L-BFGS quality; losing the decrease
    would cost correctness, so a decreasing step is still worth taking.
    """
    if best is None:
        return None
    step, _phi, payload = best
    return step, payload


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
    # 500. The loop returns as soon as g_rms < grad_tol, so this bounds the
    # work rather than causing it — chlorobenzene still exits at 15, and a
    # batch measured at cap 100 and cap 200 took the same 50 s.
    #
    # Sized from the worst case, not a formula. Iterations do not track atom
    # count: indole and anisole are both 16 atoms and need 14 and 45. They
    # track flexibility, as everything else in this optimizer does. So the
    # only defensible default is one generous enough for the worst molecule
    # anyone is likely to hand it — cholesterol (74 atoms) needs 184, which
    # left the old cap of 200 with 8% headroom.
    #
    # Not larger than 500: the cap is free for a molecule that converges, but
    # a pathological one pays it in full, and at ~3.6 s per gradient on a
    # 74-atom system that is the difference between 30 and 60 minutes. #65
    # made `converged` truthful, so a rare truncation now reports itself.
    max_iter: int = 500,
    grad_tol: float = 0.005,
    method: str = 'RM1',
    verbose: bool = False,
    molecular_charge: float = 0.0,
    optimizer: str = 'lbfgs',
) -> dict:
    """Geometry optimization using the analytical gradient.

    Args:
        optimizer: 'lbfgs' (default) or 'ef'. EF is eigenvector following —
            an explicit Hessian with a P-RFO step under a trust radius, as in
            MOPAC's ef.F90. It needs **21% fewer gradient calls** on a
            held-out set (391 -> 309), and the split is systematic: it wins on
            flexible molecules, where the cost actually is (geraniol
            133 -> 91, butyl acetate 88 -> 61), and loses on rigid ones, which
            are cheap anyway (indole 17 -> 24). It carries a 3Nx3N Hessian and
            an O(N^3) eigensolve per step, negligible against a gradient at
            these sizes. Minima agree with L-BFGS to 0.016 kcal/mol.

    Returns a dict whose `converged` and `n_iter` refer to the geometry
    optimization; the SCF's own values are under `scf_converged`/`scf_n_iter`.
    """
    if optimizer == 'ef':
        return _ef_optimize(atoms, coords, max_iter, grad_tol, method,
                            verbose, molecular_charge)
    if optimizer != 'lbfgs':
        raise ValueError(f"unknown optimizer {optimizer!r}; use 'lbfgs' or 'ef'")

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
            step_init = 0.05
        else:
            step_init = 1.0

        def trial(step, _d=direction):
            """Energy, directional derivative and payload at coords + step*d."""
            trial_coords = coords + step * _d.reshape(n_atoms, 3)
            trial_result, trial_grad = analytical_gradient(
                atoms,
                trial_coords,
                method=method,
                molecular_charge=molecular_charge,
                P_init=result.get('density'),
            )
            trial_flat = trial_grad.flatten()
            return (trial_result['energy_eV'],
                    float(np.dot(trial_flat, _d)),
                    (trial_coords, trial_result, trial_grad))

        found = _strong_wolfe(trial, energy, float(np.dot(grad_flat, direction)),
                              step_init=step_init)
        if found is None:
            # No trial decreased the energy. Take the tiny step the old
            # backtracking loop fell back on rather than stalling.
            step = 1e-4
            new_coords = coords + step * direction.reshape(n_atoms, 3)
            new_result, new_grad = analytical_gradient(
                atoms,
                new_coords,
                method=method,
                molecular_charge=molecular_charge,
                P_init=result.get('density'),
            )
        else:
            step, (new_coords, new_result, new_grad) = found

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
