# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""ANCopt geometry optimizer — port of xtb's optimizer.f90 ancopt routine.

Mirrors xtb's algorithm (xtb/src/optimizer.f90:122-613 + type/anc.f90):

    1. Build a model Hessian (Lindh 1995 by default — see lindh.py).
    2. Project out the 6 (or 5 for linear molecules) translation/
       rotation modes by diagonalization + zeroing of those eigenvalues.
    3. Build the Approximate-Normal-Coordinate (ANC) basis:
       B is the matrix of *retained* eigenvectors (shape (3N, nvar)),
       with eigenvalues clamped to [hlow, hmax]. nvar = 3N − 6
       (or 3N − 5 for linear).
    4. Iterate the relax loop:
       a. transform Cartesian gradient to ANC: g_anc = Bᵀ · g_cart
       b. RFO step in ANC space (Hessian is full nvar×nvar after BFGS
          updates; starts diagonal from the model)
       c. update internal coords and recompute Cartesians:
          xyz_new = xyz_ref + B · anc_coord
       d. evaluate (E, ∇E) at the new geometry.
       e. BFGS update on the ANC Hessian using (s, y) = (Δcoord, Δg_anc).
       f. check convergence on |ΔE|, |g_anc|max, |step|max.

Defaults match xtb's opt_level=normal:
    energy threshold:    5e-6 Ha
    gradient threshold:  1e-3 Ha/Bohr (≈ 1.89e-3 Ha/Å)
    displacement thresh: 1e-3 Bohr (≈ 5.3e-4 Å)
    max micro-iter:      50
"""

from __future__ import annotations

import numpy as np

from .lindh import model_hessian, LindhParams


_ANG_TO_BOHR = 1.8897259886
_BOHR_TO_ANG = 1.0 / _ANG_TO_BOHR


# ---------------------------------------------------------------------------
# Translation / rotation mode detection (port of xtb/src/detrotra.f90:detrotra8).
# Mark modes that are predominantly trans/rot by their overlap with the
# 6 (or 5 for linear) global trans/rot directions, then zero their
# eigenvalues. Cartesian eigenvectors are unchanged.
# ---------------------------------------------------------------------------

def _build_trotra_modes(coords_b: np.ndarray) -> tuple[np.ndarray, bool]:
    """Build the 5 or 6 translation/rotation modes (in Cartesians).

    Returns (T, is_linear) where T is shape (3N, ntr) with ntr=5 or 6.
    Modes are NOT orthonormalized — they're used only for projection.
    """
    n = coords_b.shape[0]
    n3 = 3 * n
    # Center
    com = coords_b.mean(axis=0)
    rc = coords_b - com[None, :]
    # Translations: ones along each axis, broadcast over atoms.
    Tt = np.zeros((n3, 3))
    for axis in range(3):
        v = np.zeros((n, 3))
        v[:, axis] = 1.0
        Tt[:, axis] = v.flatten()
    # Rotations: r × ê_axis, evaluated at each atom.
    Tr = np.zeros((n3, 3))
    for axis in range(3):
        v = np.zeros((n, 3))
        e = np.zeros(3); e[axis] = 1.0
        for i in range(n):
            v[i] = np.cross(rc[i], e)
        Tr[:, axis] = v.flatten()
    # Linearity check: if ‖Tr along one axis‖ ~ 0 the molecule is linear.
    is_linear = False
    for axis in range(3):
        if np.linalg.norm(Tr[:, axis]) < 1e-8:
            is_linear = True
    if is_linear:
        # Drop the zero rotation column.
        keep = [axis for axis in range(3) if np.linalg.norm(Tr[:, axis]) > 1e-8]
        Tr = Tr[:, keep]
    T = np.concatenate([Tt, Tr], axis=1)
    # Orthonormalize via QR for stability.
    T, _ = np.linalg.qr(T)
    return T, is_linear


def _project_trotra(
    H: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Zero out eigenvalues whose eigenvectors have large overlap with
    the trans/rot subspace ``T`` (i.e. ‖Tᵀ v_i‖ near 1 means v_i is
    a trans/rot mode). Returns the modified eigvals (in place)."""
    overlaps = np.linalg.norm(T.T @ eigvecs, axis=0)  # (n3,)
    # Modes with overlap > 0.5 are dominated by trans/rot.
    mask = overlaps > 0.5
    eigvals[mask] = 0.0
    return eigvals


# ---------------------------------------------------------------------------
# ANC type — Approximate Normal Coordinates.
# ---------------------------------------------------------------------------

class ANC:
    """Approximate Normal Coordinate basis.

    Holds:
        xyz_ref: reference Cartesians (Å) at which the ANC was built.
        B: ``(3N, nvar)`` transformation matrix; columns are the
           retained Cartesian eigenvectors.
        hess: ``(nvar, nvar)`` Hessian in ANC space (diagonal at init,
              gets BFGS-updated during the optimization).
        eigv: ``(nvar,)`` clamped initial eigenvalues (Ha/Bohr²).
        coord: ``(nvar,)`` internal coordinates (start at zero).
        is_linear: True if the molecule is linear.
    """

    def __init__(
        self,
        atoms: list[int],
        xyz_ref_ang: np.ndarray,
        H_cart: np.ndarray,
        hlow: float = 1e-3,
        hmax: float = 5.0,
    ):
        self.xyz_ref = xyz_ref_ang.copy()
        n = xyz_ref_ang.shape[0]
        n3 = 3 * n

        coords_b = xyz_ref_ang * _ANG_TO_BOHR
        T, is_linear = _build_trotra_modes(coords_b)
        self.is_linear = is_linear

        # Diagonalize Hessian
        eigvals, eigvecs = np.linalg.eigh(H_cart)

        # Project out trans/rot modes (zero their eigenvalues so they get
        # filtered out below).
        eigvals = _project_trotra(H_cart, eigvals, eigvecs, T)

        # Determine nvar (= 3N - ntr).
        ntr = 5 if is_linear else 6
        nvar = n3 - ntr
        self.nvar = nvar

        # Find lowest non-zero eigenvalue and shift all up by hlow if needed.
        thr_zero = 1e-10
        nonzero = np.abs(eigvals) > thr_zero
        if np.any(nonzero):
            elow = float(np.min(eigvals[nonzero]))
            damp = max(hlow - elow, 0.0)
            eigvals[nonzero] += damp

        # Pick the nvar eigvectors with largest |λ| (drop the smallest
        # |λ| ones, which correspond to the projected-out trans/rot
        # directions).
        order = np.argsort(np.abs(eigvals))
        keep_idx = order[ntr:][:nvar]                    # indices to keep
        keep_idx = keep_idx[np.argsort(eigvals[keep_idx])]  # sort by eigval ascending

        self.B = eigvecs[:, keep_idx]                     # (3N, nvar)
        clamped = np.clip(eigvals[keep_idx], hlow, hmax)
        self.eigv = clamped.copy()
        self.hess = np.diag(clamped)                      # (nvar, nvar)
        self.coord = np.zeros(nvar, dtype=np.float64)

    def get_cartesian(self) -> np.ndarray:
        """Return current Cartesians (Å) = xyz_ref + B · coord (in Bohr,
        then convert)."""
        displ_b = self.B @ self.coord                     # (3N,) Bohr
        n = self.xyz_ref.shape[0]
        displ_ang = displ_b.reshape(n, 3) * _BOHR_TO_ANG
        return self.xyz_ref + displ_ang

    def get_normal(self, g_cart: np.ndarray) -> np.ndarray:
        """Transform Cartesian gradient (Ha/Å) → ANC gradient (Ha/Bohr).

        g_anc = Bᵀ · g_cart_in_Bohr_units. Note B has Cartesian rows
        in Bohr-displacement units (since we built it from a Hessian
        in Ha/Bohr²); converting g_cart from Ha/Å to Ha/Bohr means
        dividing by Bohr_per_Ang."""
        g_b = g_cart.flatten() * _BOHR_TO_ANG          # Ha/Bohr
        return self.B.T @ g_b


# ---------------------------------------------------------------------------
# RFO step — port of the relax() inner loop. The "full" RFO step solves
# the augmented eigenproblem; near-zero modes are damped automatically
# because they end up with small eigenvalue → small step.
# ---------------------------------------------------------------------------

def _rfo_step(
    g: np.ndarray, H: np.ndarray, max_step: float
) -> np.ndarray:
    """RFO step for the augmented eigenproblem
       [[H, g], [gᵀ, 0]] [s, 1] = λ [s, 1]"""
    n = g.size
    aug = np.zeros((n + 1, n + 1), dtype=np.float64)
    aug[:n, :n] = H
    aug[:n, n] = g
    aug[n, :n] = g
    w, V = np.linalg.eigh(aug)
    v = V[:, 0]
    if abs(v[n]) < 1e-12:
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = -g
    else:
        step = v[:n] / v[n]
    s_norm = float(np.linalg.norm(step))
    if s_norm > max_step:
        step *= max_step / s_norm
    return step


def _bfgs_update(
    H: np.ndarray, s: np.ndarray, y: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """BFGS update of the Hessian.

    Skip the rank-2 correction when (s, y) is nearly orthogonal — keeps H
    positive-definite under degenerate steps.
    """
    sy = float(s @ y)
    if abs(sy) < eps:
        return H
    Hs = H @ s
    sHs = float(s @ Hs)
    if abs(sHs) < eps:
        return H
    return H + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs


# ---------------------------------------------------------------------------
# Top-level ancopt
# ---------------------------------------------------------------------------

def ancopt(
    atoms: list[int],
    coords_ang: np.ndarray,
    calc,
    *,
    max_iter: int = 200,
    max_micro: int = 50,
    gtol_bohr: float = 1e-3,         # Ha/Bohr — xtb opt_level=normal
    etol: float = 5e-6,              # Ha
    stol_bohr: float = 1e-3,         # Bohr
    max_step_bohr: float = 0.3,      # Bohr per micro-iter
    hlow: float = 1e-3,
    hmax: float = 5.0,
    lindh_params: LindhParams | None = None,
    verbose: bool = False,
) -> dict:
    """xtb-style ANCopt geometry optimization.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n, 3)`` initial Angstrom coordinates.
        calc: callable ``calc(atoms, coords_ang) -> (E_hartree, grad_Ha_per_Ang)``.
        max_iter: outer loop cap (each outer step rebuilds the model
            Hessian and ANC).
        max_micro: micro-iterations within a single ANC subspace.
        gtol_bohr / etol / stol_bohr: convergence thresholds (xtb
            opt_level=normal defaults).
        max_step_bohr: hard cap on RFO step length per micro-iter.
        hlow / hmax: ANC eigenvalue clamps (Ha/Bohr²).
        lindh_params: override for the Lindh model Hessian.
        verbose: per-iter diagnostics.

    Returns:
        ``dict`` with ``coords`` (Å), ``energy`` (Ha), ``gradient`` (Ha/Å),
        ``converged`` (bool), ``n_iter`` (total micro-iter count),
        ``trajectory`` (list of (coords, E)).
    """
    coords = np.asarray(coords_ang, dtype=np.float64).copy()
    E, grad = calc(atoms, coords)
    trajectory: list[tuple[np.ndarray, float]] = [(coords.copy(), float(E))]
    total_iter = 0
    converged = False

    if verbose:
        print(
            f"  iter   0:  E = {E:.8f}  |grad|max = {np.max(np.abs(grad)):.2e} Ha/Å"
        )

    for outer in range(max_iter):
        # Rebuild model Hessian + ANC at current geometry.
        H_cart = model_hessian(atoms, coords, params=lindh_params)
        anc = ANC(atoms, coords, H_cart, hlow=hlow, hmax=hmax)
        # Track Cartesian coords; ANC.coord starts at zero.
        E_prev = E
        g_anc = anc.get_normal(grad)

        for micro in range(max_micro):
            total_iter += 1
            step = _rfo_step(g_anc, anc.hess, max_step_bohr)
            anc.coord += step
            coords = anc.get_cartesian()
            E_new, grad_new = calc(atoms, coords)
            g_anc_new = anc.get_normal(grad_new)
            dE = float(E_new - E)
            gmax_bohr = float(np.max(np.abs(g_anc_new)))   # ANC gradient (Ha/Bohr)
            smax_bohr = float(np.max(np.abs(step)))
            if verbose:
                print(
                    f"  iter {total_iter:3d}:  E = {E_new:.8f}  ΔE = {dE:+.2e}  "
                    f"|g_anc|max = {gmax_bohr:.2e}  step_max = {smax_bohr:.2e}"
                )
            # BFGS update on ANC Hessian.
            anc.hess = _bfgs_update(anc.hess, step, g_anc_new - g_anc)
            E = E_new
            grad = grad_new
            g_anc = g_anc_new
            trajectory.append((coords.copy(), E))

            if abs(dE) < etol and gmax_bohr < gtol_bohr and smax_bohr < stol_bohr:
                converged = True
                break

        if converged:
            break

    return {
        "coords": coords,
        "energy": float(E),
        "gradient": grad,
        "converged": converged,
        "n_iter": total_iter,
        "trajectory": trajectory,
    }
