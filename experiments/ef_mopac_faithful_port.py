"""Faithful port of MOPAC's EF for a minimum search (ef.F90) — NOT LANDED.

Kept because a negative result written down is cheaper than one re-derived.
This reaches 409 gradient calls on the held-out set where our own L-BFGS
takes 391, and it does not reproduce MOPAC's cycle counts (42/36/45/95
against MOPAC's 39/20/34/75 on ethanol/benzaldehyde/thioanisole/menthol at
matched tolerance), so something is still missing. `ireclc` — MOPAC's
periodic recalculation of the exact Hessian — is the leading suspect.

See experiments/README.md section 7 for why the comparison that motivated
this port was itself invalid.

Every constant below is read out of the source, not chosen:

  efstr,  minimum search   iupd=2 (BFGS), rmin=0, rmax=1e3,
                           dmax=0.2, ddmax=0.5, ddmin=1e-4
  gethes, igthes=0, XYZ    flat diagonal 200 kcal/mol/A^2
  ef,     demin=1e-2 kcal/mol, gmin=5.0 kcal/mol/A
  formd                    lambda found by bisection so |d| = dmax
"""
import numpy as np
from mlxmolkit.nddo.anal_grad import analytical_gradient

KCAL_PER_EV = 23.060547830619026

H0_DIAG = 200.0 / KCAL_PER_EV      # gethes: XYZ -> flat 200 kcal/mol/A^2
DMAX0, DDMAX, DDMIN = 0.2, 0.5, 1.0e-4
DEMIN = 1.0e-2 / KCAL_PER_EV       # ef.F90 parameter, kcal/mol -> eV
GMIN = 5.0 / KCAL_PER_EV           # kcal/mol/A -> eV/A
RMIN, RMAX = 0.0, 1.0e3


def _formd(eigval, fx, dmax):
    """MOPAC formd, minimum branch: shift lambda until |d| == dmax.

    d_i = -F_i / (b_i - lambda), lambda < b_1 keeps every denominator
    positive so the step descends. Bisection on lambda rather than clipping
    the natural RFO step: shortening by scaling keeps the direction, whereas
    re-solving rotates it toward steepest descent, which is the point of a
    trust region.
    """
    eone = eigval[0]
    step = -fx / eigval
    if np.linalg.norm(step) <= dmax:
        # Newton step is inside the trust radius: lambda = 0.
        return step, float(np.linalg.norm(step)), 0.0

    lo, hi = eone - 1.0e4, min(0.0, eone) - 1.0e-8
    lam = hi
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        denom = eigval - lam
        denom[np.abs(denom) < 1e-12] = 1e-12
        d = -fx / denom
        length = float(np.linalg.norm(d))
        if abs(length - dmax) < 1e-10 * max(dmax, 1.0):
            break
        if length > dmax:
            hi = lam          # more negative shift -> shorter step
        else:
            lo = lam
    denom = eigval - lam
    denom[np.abs(denom) < 1e-12] = 1e-12
    d = -fx / denom
    ddx = float(np.linalg.norm(d))
    if ddx > dmax:            # formd's final safeguard: skal = dmax/ddx
        d *= dmax / ddx
        ddx = dmax
    return d, ddx, lam


def _updhes_bfgs(hess, d, y):
    """updhes iupd=2: H += y y^T/(y.d) - (Hd)(Hd)^T/(d.Hd)."""
    tvec = hess @ d
    dds = float(np.dot(y, d))
    ddtd = float(np.dot(d, tvec))
    if abs(dds) < 1e-20 or abs(ddtd) < 1e-20:
        return hess
    return hess + np.outer(y, y) / dds - np.outer(tvec, tvec) / ddtd


def ef_mopac(atoms, coords, max_iter=200, grad_tol=0.005, method='RM1',
             molecular_charge=0.0):
    coords = np.asarray(coords, dtype=np.float64).copy()
    n_atoms = len(atoms)
    n = n_atoms * 3
    hess = np.eye(n) * H0_DIAG
    dmax = DMAX0
    grads = 0

    result, grad = analytical_gradient(atoms, coords, method=method,
                                       molecular_charge=molecular_charge)
    grads += 1
    energy = result['energy_eV']
    g = grad.flatten()

    for iteration in range(max_iter):
        g_rms = float(np.sqrt(np.mean(g ** 2)))
        if g_rms < grad_tol:
            return dict(converged=True, n_iter=iteration + 1, grads=grads,
                        grad_rms=g_rms, coords=coords,
                        heat_of_formation_kcal=result['heat_of_formation_kcal'])

        eigval, U = np.linalg.eigh(hess)
        fx = U.T @ g
        d_eig, ddx, _lam = _formd(eigval, fx, dmax)
        d = U @ d_eig
        depre = float(np.dot(g, d) + 0.5 * d @ hess @ d)

        new_coords = coords + d.reshape(n_atoms, 3)
        new_result, new_grad = analytical_gradient(
            atoms, new_coords, method=method, molecular_charge=molecular_charge)
        grads += 1
        deact = new_result['energy_eV'] - energy
        ratio = deact / depre if abs(depre) > 1e-20 else 1.0

        reject = ((ratio < RMIN or ratio > RMAX)
                  and (abs(depre) > DEMIN or abs(deact) > DEMIN))
        if reject and abs(dmax - DDMIN) < 1e-20:
            reject = False                      # at the floor: accept anyway
        if reject:
            dmax = max(DDMIN, min(dmax, ddx) / 2.0)
            continue

        y = new_grad.flatten() - g
        hess = _updhes_bfgs(hess, d, y)
        coords, result = new_coords, new_result
        energy = result['energy_eV']
        grad = new_grad
        g = grad.flatten()

        # Trust radius. Note MOPAC does NOT shrink on a poor ratio for a
        # minimum search; both growth tests can fire in the same cycle.
        if ratio >= 0.5 and ddx > dmax - 1.0e-6:
            dmax *= np.sqrt(2.0)
        if abs(ratio - 1.0) < 0.1:
            dmax *= np.sqrt(2.0)
        dmax = min(max(dmax, DDMIN), DDMAX)
        if (float(np.abs(g).max()) < GMIN
                and abs(depre) < DEMIN and abs(deact) < DEMIN):
            dmax = max(dmax, 0.1)               # end-game

    return dict(converged=False, n_iter=max_iter, grads=grads, coords=coords,
                grad_rms=float(np.sqrt(np.mean(g ** 2))),
                heat_of_formation_kcal=result['heat_of_formation_kcal'])
