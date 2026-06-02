"""openCOSMO-RS activity-coefficient helpers for mixtures.

This module is intentionally a thin wrapper around ``opencosmorspy``.  It
mirrors the FACCTS OPI notebook section-6 workflow:

    clear_jobs() -> add_job(x, T, refst="pure_component") -> calculate()

The main addition is explicit support for solvent mixtures in the solubility
iteration: the non-solute mole fractions are kept at a fixed solvent ratio
while the solute mole fraction is varied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


R_J_PER_MOL_K = 8.31446261815324
WALDEN_DELTA_S_FUSION_J_MOL_K = 56.5


@dataclass(frozen=True)
class ActivityResult:
    labels: tuple[str, ...]
    x: np.ndarray
    T: float
    ln_gamma: np.ndarray
    gamma: np.ndarray


def _normalized_x(x: Sequence[float], *, n: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("mole fractions must be a one-dimensional vector")
    if n is not None and arr.size != n:
        raise ValueError(f"expected {n} mole fractions, got {arr.size}")
    if np.any(arr < 0.0):
        raise ValueError("mole fractions must be non-negative")
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("mole fractions must have a positive finite sum")
    return arr / total


def make_cosmors(component_cosmo_paths: Sequence[str | Path]):
    """Create an ``opencosmorspy.COSMORS`` object loaded with components."""

    from opencosmorspy.cosmors import COSMORS
    from opencosmorspy.parameterization import openCOSMORS24a

    crs = COSMORS(par=openCOSMORS24a())
    for path in component_cosmo_paths:
        crs.add_molecule([str(Path(path))])
    return crs


def activity_coefficients(
    component_cosmo_paths: Sequence[str | Path],
    x: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    T: float = 298.15,
    refst: str = "pure_component",
) -> ActivityResult:
    """Return activity coefficients for all components at a composition.

    ``x`` is normalized before it is sent to openCOSMO-RS.  The returned
    ``ln_gamma`` vector has one entry per component and uses the requested
    reference state, usually ``pure_component`` for liquid mixtures.
    """

    paths = [Path(p) for p in component_cosmo_paths]
    x_arr = _normalized_x(x, n=len(paths))
    labels_tuple = tuple(labels) if labels is not None else tuple(p.stem for p in paths)
    if len(labels_tuple) != len(paths):
        raise ValueError(f"expected {len(paths)} labels, got {len(labels_tuple)}")

    crs = make_cosmors(paths)
    crs.clear_jobs()
    crs.add_job(x_arr, float(T), refst=refst)
    res = crs.calculate()
    ln_gamma = np.asarray(res["tot"]["lng"], dtype=np.float64)[0]
    return ActivityResult(
        labels=labels_tuple,
        x=x_arr,
        T=float(T),
        ln_gamma=ln_gamma,
        gamma=np.exp(ln_gamma),
    )


def ideal_solid_solubility_ln_x(
    *,
    delta_h_fus_J_mol: float,
    T_fus_K: float,
    T_K: float = 298.15,
) -> float:
    """Ideal solid-liquid solubility term used in the FACCTS notebook."""

    return -float(delta_h_fus_J_mol) / R_J_PER_MOL_K * (1.0 / float(T_K) - 1.0 / float(T_fus_K))


def estimate_delta_h_fusion_walden(
    T_fus_K: float,
    *,
    delta_s_fusion_J_mol_K: float = WALDEN_DELTA_S_FUSION_J_MOL_K,
) -> float:
    """Estimate ``ΔH_fus`` from ``T_fus`` via Walden's-rule entropy.

    This is a fallback for screening, not a replacement for measured
    ``ΔH_fus``.  The approximation assumes roughly constant fusion entropy:

        ΔH_fus ≈ T_fus · ΔS_fus
    """

    return float(T_fus_K) * float(delta_s_fusion_J_mol_K)


def solute_solvent_mixture_x(solute_x: float, solvent_x: Sequence[float]) -> np.ndarray:
    """Composition for one solute followed by a fixed-ratio solvent mixture."""

    xs = float(np.clip(solute_x, 1.0e-15, 1.0 - 1.0e-15))
    s = _normalized_x(solvent_x)
    return np.concatenate([[xs], (1.0 - xs) * s])


def solubility_in_solvent_mixture(
    component_cosmo_paths: Sequence[str | Path],
    solvent_x: Sequence[float],
    *,
    delta_h_fus_J_mol: float,
    T_fus_K: float,
    T: float = 298.15,
    labels: Sequence[str] | None = None,
    iterative: bool = True,
    infinite_dilution_x: float = 1.0e-8,
) -> dict[str, object]:
    """Predict solute mole-fraction solubility in a fixed solvent mixture.

    Components must be ordered as ``[solute, solvent_1, solvent_2, ...]``.
    ``solvent_x`` gives the solvent ratio among the non-solute components.
    The non-iterative approximation evaluates ``ln(gamma_solute)`` at
    infinite dilution.  The iterative solution updates ``ln(gamma_solute)`` at
    the finite solubility composition.
    """

    from scipy.optimize import brentq, minimize_scalar

    paths = [Path(p) for p in component_cosmo_paths]
    if len(paths) < 2:
        raise ValueError("need one solute and at least one solvent component")
    s = _normalized_x(solvent_x, n=len(paths) - 1)
    labels_tuple = tuple(labels) if labels is not None else tuple(p.stem for p in paths)
    rhs = ideal_solid_solubility_ln_x(
        delta_h_fus_J_mol=delta_h_fus_J_mol,
        T_fus_K=T_fus_K,
        T_K=T,
    )

    crs = make_cosmors(paths)

    def ln_gamma_solute(x_solute: float) -> float:
        x = solute_solvent_mixture_x(x_solute, s)
        crs.clear_jobs()
        crs.add_job(x, float(T), refst="pure_component")
        res = crs.calculate()
        return float(np.asarray(res["tot"]["lng"], dtype=np.float64)[0, 0])

    ln_gamma_inf = ln_gamma_solute(float(infinite_dilution_x))
    x_noniter = float(np.exp(rhs - ln_gamma_inf))
    if not iterative:
        x = solute_solvent_mixture_x(x_noniter, s)
        return {
            "labels": labels_tuple,
            "solubility_x": x_noniter,
            "composition": x,
            "ln_gamma_solute": ln_gamma_inf,
            "ln_x_ideal": rhs,
            "iterative": False,
        }

    def residual(x_solute: float) -> float:
        x_safe = float(np.clip(x_solute, 1.0e-15, 1.0 - 1.0e-15))
        return ln_gamma_solute(x_safe) + np.log(x_safe) - rhs

    lo, hi = 1.0e-12, 1.0 - 1.0e-12
    f_lo, f_hi = residual(lo), residual(hi)
    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0.0:
        x_sol = float(brentq(residual, lo, hi, xtol=1.0e-12, rtol=1.0e-10, maxiter=80))
    else:
        opt = minimize_scalar(lambda x: residual(x) ** 2, bounds=(lo, hi), method="bounded")
        x_sol = float(opt.x)

    x = solute_solvent_mixture_x(x_sol, s)
    return {
        "labels": labels_tuple,
        "solubility_x": x_sol,
        "composition": x,
        "ln_gamma_solute": ln_gamma_solute(x_sol),
        "ln_x_ideal": rhs,
        "iterative": True,
    }
