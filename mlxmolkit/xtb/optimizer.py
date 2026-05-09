# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Geometry optimizer — RFO step on an approximate Hessian.

Method-agnostic: accepts any callable ``calc(atoms, coords) -> (E, ∇E)``
returning energy in Hartree and gradient in Hartree/Å. Designed to
replicate xtb's ``--opt`` ANCopt qualitatively:

    1. Build initial approximate Hessian (currently a simple diagonal;
       Lindh model is a TODO refinement that improves the first few
       steps but isn't required for correctness).
    2. Each iteration:
       a. Diagonalize the Hessian (in Cartesians for v1; Lindh-derived
          ANC normal coordinates for v2).
       b. Take an RFO (Rational Function Optimization) step that
          regularizes near zero-eigenvalue directions, naturally
          handling the rotation/translation modes.
       c. Update the Hessian via BFGS (or SR1) using the new
          (gradient, step) pair.
    3. Convergence on max |grad| < gtol and max |step| < stol.

For ``xtb --opt`` the equivalent thresholds (``opt_level=normal``):
    energy change      < 5e-6 Ha
    gradient max norm  < 1e-3 Ha/Bohr  (≈ 1.89e-3 Ha/Å)
    displacement max   < 1e-3 Bohr

Returns the optimized coordinates + a trajectory dict.
"""

from __future__ import annotations

import math
import numpy as np


_ANG_TO_BOHR = 1.8897259886


def _rfo_step(grad_flat: np.ndarray, hess: np.ndarray, max_step: float) -> np.ndarray:
    """Rational-Function-Optimization step.

    Solves the augmented eigenproblem
        [[H,  g],     [[s],
         [gᵀ, 0]]  ·   [1]]   = λ · [[s],
                                     [1]]
    and returns the step ``s`` that descends along the lowest
    eigenvalue direction. RFO naturally damps near-zero modes
    (rotations/translations) by mixing them with the gradient.

    Args:
        grad_flat: ``(3N,)`` gradient.
        hess: ``(3N, 3N)`` symmetric approximate Hessian.
        max_step: hard cap on step magnitude (Å).

    Returns:
        ``(3N,)`` step vector, capped to ``max_step``.
    """
    n = grad_flat.size
    # Build the augmented (n+1) matrix
    aug = np.zeros((n + 1, n + 1), dtype=np.float64)
    aug[:n, :n] = hess
    aug[:n, n] = grad_flat
    aug[n, :n] = grad_flat
    # Symmetric eigh
    w, V = np.linalg.eigh(aug)
    # Lowest eigenvalue → step (scale so last component = 1).
    v = V[:, 0]
    if abs(v[n]) < 1e-12:
        # Degenerate — fall back to simple Newton step.
        try:
            step = -np.linalg.solve(hess, grad_flat)
        except np.linalg.LinAlgError:
            step = -grad_flat
    else:
        step = v[:n] / v[n]
    # Cap.
    step_norm = float(np.linalg.norm(step))
    if step_norm > max_step:
        step *= max_step / step_norm
    return step


def _bfgs_update(
    H: np.ndarray, s: np.ndarray, y: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """BFGS Hessian update. ``s = x_new - x_old``, ``y = g_new - g_old``."""
    sy = float(s @ y)
    if abs(sy) < eps:
        return H  # skip update when (s,y) nearly orthogonal — keeps H PD
    Hs = H @ s
    sHs = float(s @ Hs)
    return H + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs


def _initial_hessian(n_atoms: int, scale: float = 0.5) -> np.ndarray:
    """Simple ``c·I`` initial Hessian. v2 will replace with Lindh-1995.

    The scale ``c = 0.5 Ha/Bohr²`` is what xtb uses as the default
    "ddvopt" stretch force constant for unrelaxed initial guesses. In
    Hartree/Å² units it's ``0.5 / Bohr²`` ≈ ``0.14 Ha/Å²``.
    """
    n3 = 3 * n_atoms
    return np.eye(n3, dtype=np.float64) * scale / (_ANG_TO_BOHR ** 2)


def optimize(
    atoms: list[int],
    coords_ang: np.ndarray,
    calc,
    *,
    max_iter: int = 200,
    gtol: float = 1.89e-3,         # Ha/Å (≈ 1e-3 Ha/Bohr — xtb default)
    etol: float = 5e-6,            # Ha
    stol: float = 5.3e-4,          # Å (1e-3 Bohr)
    max_step: float = 0.3,         # Å — hard step cap
    initial_hessian: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """Optimize the geometry against an arbitrary calculator.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` initial Angstrom coordinates.
        calc: callable ``calc(atoms, coords) -> (E_hartree, grad_ha_per_ang)``.
        max_iter: max number of optimization steps.
        gtol: convergence on max |gradient| (Ha/Å).
        etol: convergence on |ΔE| between steps (Ha).
        stol: convergence on max step component (Å).
        max_step: per-step distance cap (Å).
        initial_hessian: optional ``(3N, 3N)`` initial Hessian. Defaults
            to ``0.5/Bohr² · I`` (uniform diagonal). Replace with a
            Lindh model Hessian when available for fewer iterations.
        verbose: print per-iteration diagnostic.

    Returns:
        ``dict`` with keys:
            ``coords`` (Å), ``energy`` (Ha), ``gradient`` (Ha/Å),
            ``converged`` (bool), ``n_iter`` (int),
            ``trajectory`` (list of (coords, E)).
    """
    coords = np.asarray(coords_ang, dtype=np.float64).copy()
    n_atoms = coords.shape[0]
    n3 = 3 * n_atoms
    H = (
        np.asarray(initial_hessian, dtype=np.float64).copy()
        if initial_hessian is not None
        else _initial_hessian(n_atoms)
    )

    E, grad = calc(atoms, coords)
    grad_flat = grad.flatten().astype(np.float64)

    trajectory: list[tuple[np.ndarray, float]] = [(coords.copy(), float(E))]
    converged = False
    iter_done = 0
    if verbose:
        print(f"  iter   0:  E = {E:.8f}  |grad|max = {np.max(np.abs(grad)):.2e}")

    for it in range(1, max_iter + 1):
        step_flat = _rfo_step(grad_flat, H, max_step)
        step = step_flat.reshape(coords.shape)
        max_step_comp = float(np.max(np.abs(step)))

        coords_new = coords + step
        E_new, grad_new = calc(atoms, coords_new)
        grad_new_flat = grad_new.flatten().astype(np.float64)

        dE = float(E_new - E)
        gmax = float(np.max(np.abs(grad_new)))
        if verbose:
            print(
                f"  iter {it:3d}:  E = {E_new:.8f}  ΔE = {dE:+.2e}  "
                f"|grad|max = {gmax:.2e}  step_max = {max_step_comp:.2e}"
            )

        # Convergence check (xtb uses all three — energy, gradient, step).
        if abs(dE) < etol and gmax < gtol and max_step_comp < stol:
            coords = coords_new
            E = E_new
            grad = grad_new
            grad_flat = grad_new_flat
            trajectory.append((coords.copy(), E))
            converged = True
            iter_done = it
            break

        # BFGS update.
        y = grad_new_flat - grad_flat
        s = step_flat
        H = _bfgs_update(H, s, y)

        coords = coords_new
        E = E_new
        grad = grad_new
        grad_flat = grad_new_flat
        trajectory.append((coords.copy(), E))
        iter_done = it

    return {
        "coords": coords,
        "energy": float(E),
        "gradient": grad,
        "converged": converged,
        "n_iter": iter_done,
        "trajectory": trajectory,
    }
