# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2-xTB ALPB (Analytical Linearized Poisson-Boltzmann) implicit
solvent — water.

This is the implicit solvation model used by ORCA's
``! XTB2 OPT ALPB(water)`` invocation, which is the actual upstream
call we're replacing in the openCOSMO-RS conformer pipeline.

ALPB extends the Generalized Born (GB) model with a linear
Poisson-Boltzmann correction; it adds:

1. **Polar (GB-LPB)** contribution — the dominant term, coupling
   atomic charges through a damped Coulomb kernel modulated by
   atomic Born radii and the dielectric constant.
2. **Non-polar (SASA)** contribution — surface area integral with
   per-element surface tensions.
3. **HB correction** for hydrogen bonds in protic solvents (small).

The full pure-MLX port (Born radii self-consistency via OBC-II,
Lebedev-quadrature SASA, the GB Fock-matrix coupling for charge
self-consistency in the SCF) is substantial — xtb's
``src/solv/gbsa.f90`` + ``born.f90`` + ``sasa.f90`` total ~1700 lines
and require careful sign and convention work.

For the strategic destination (replacing the ORCA pipeline call) we
ship a **tblite-backed wrapper** here that delegates to the canonical
GFN2-xTB + ALPB(water) implementation in ``libtblite``. This is
analogous to the ``simple-dftd4`` backend in :mod:`dispersion_d4`:
it's a separate, MIT-licensed library with the right physics, used as
a single-call boundary at the energy level — not in the SCF inner
loop. The pure-MLX path is tracked as the next major refactor.

Usage:
    >>> from mlxmolkit.xtb.solvation_alpb import alpb_water_correction
    >>> e_solv = alpb_water_correction([8, 1, 1], coords_ang, charge=0)
    >>> e_total_solvated = e_total_vacuum + e_solv

The returned correction is the ``E_solvated − E_vacuum`` energy
difference computed by tblite under identical method settings — i.e.
exactly what gets added when you flip the ALPB switch on.
"""

from __future__ import annotations

import numpy as np

_ANG_TO_BOHR = 1.8897259886


def alpb_water_correction(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    method: str = "GFN2-xTB",
) -> float:
    """Return the ALPB(water) energy correction in Hartree.

    Args:
        atoms: list of atomic numbers.
        coords_ang: (n_atoms, 3) Angstrom coordinates.
        charge: integer net charge.
        method: tblite method name. Default ``'GFN2-xTB'`` matches the
            ORCA ``XTB2`` invocation. Also accepts ``'GFN1-xTB'``.

    Returns:
        ``E_solvated - E_vacuum`` (Hartree). Negative means stabilizing
        — typical for charged/polar species in water.

    Notes:
        Calls the ``tblite`` Python library twice (vacuum vs ALPB) with
        the same SCF settings. Energy difference isolates the ALPB
        contribution as a clean post-SCF correction.

        The fully self-consistent SCF coupling (where ALPB modifies the
        Fock matrix and re-converges the charges) is not yet ported;
        the difference vs the SCF-coupled result is small for typical
        organic conformer-pipeline geometries (sub-kcal/mol on most
        small molecules, larger for ions and zwitterions). When that
        precision matters, use a tblite-backed singlepoint until the
        pure-MLX port lands.
    """
    try:
        from tblite.interface import Calculator
    except ImportError as e:
        raise ImportError(
            "ALPB(water) backend currently relies on tblite-python. "
            "Install via 'conda install -c conda-forge tblite-python' "
            "or 'pip install tblite'."
        ) from e
    nums = np.asarray(atoms, dtype=np.int32)
    pos_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR

    # Vacuum singlepoint
    c_vac = Calculator(method, nums, pos_bohr, charge=float(charge))
    c_vac.set("verbosity", 0)
    e_vac = float(c_vac.singlepoint().get("energy"))

    # Solvated singlepoint
    c_sol = Calculator(method, nums, pos_bohr, charge=float(charge))
    c_sol.set("verbosity", 0)
    c_sol.add("alpb-solvation", "water")
    e_sol = float(c_sol.singlepoint().get("energy"))

    return e_sol - e_vac


def gfn2_energy_alpb_water(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    **scf_kwargs,
) -> dict:
    """Convenience: GFN2-xTB single-point + ALPB(water) correction.

    Calls :func:`mlxmolkit.xtb.scf_gfn2.gfn2_energy` (our pure-MLX
    SCF, sub-3 kcal/mol parity vs tblite-vacuum) and adds the ALPB
    correction from :func:`alpb_water_correction`.
    """
    from .scf_gfn2 import gfn2_energy
    r = gfn2_energy(atoms, coords_ang, charge=charge, **scf_kwargs)
    e_alpb = alpb_water_correction(atoms, coords_ang, charge=charge)
    r["alpb_water_eV"] = e_alpb * 27.211386245988
    r["energy_hartree_alpb"] = r["energy_hartree"] + e_alpb
    r["energy_hartree"] = r["energy_hartree_alpb"]
    r["energy_kcal"] = r["energy_hartree"] * 627.5094740631
    r["energy_eV"] = r["energy_hartree"] * 27.211386245988
    return r


def _tblite_alpb_water_calc(method: str = "GFN2-xTB"):
    """Build a closure that evaluates ``(E_hartree, grad_Ha_per_Ang)``
    at a given geometry using tblite's GFN2-xTB + ALPB(water) backend.
    Matches the ``ancopt`` calc signature.
    """
    try:
        from tblite.interface import Calculator
    except ImportError as e:
        raise ImportError(
            "tblite-python required for the ALPB(water) optimizer. "
            "Install via 'conda install -c conda-forge tblite-python'."
        ) from e

    def _calc(atoms, coords_ang, charge: int = 0):
        nums = np.asarray(atoms, dtype=np.int32)
        pos_b = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
        c = Calculator(method, nums, pos_b, charge=float(charge))
        c.set("verbosity", 0)
        c.add("alpb-solvation", "water")
        r = c.singlepoint()
        e = float(r.get("energy"))
        g_b = np.asarray(r.get("gradient"), dtype=np.float64)
        return e, g_b * _ANG_TO_BOHR  # Ha/Bohr → Ha/Å for ANCopt

    return _calc


def _opt_worker(args):
    """Top-level worker for ProcessPoolExecutor (must be importable)."""
    a, c, chg, mi, gt, et, st = args
    r = gfn2_alpb_water_optimize(
        a, c, charge=chg, max_iter=mi,
        gtol_bohr=gt, etol=et, stol_bohr=st,
    )
    r.pop("trajectory", None)  # strip trajectory for IPC efficiency
    return r


def gfn2_alpb_water_optimize_batch(
    conformers: list[tuple],
    *,
    charge: int = 0,
    max_iter: int = 200,
    gtol_bohr: float = 1e-3,
    etol: float = 5e-6,
    stol_bohr: float = 1e-3,
    n_workers: int | None = None,
    use_processes: bool = False,
) -> list[dict]:
    """Optimize a batch of conformers in parallel.

    Bread-and-butter routine for the openCOSMO-RS pipeline: takes a
    list of (atoms, coords) conformer tuples and returns optimized
    geometries + energies for each, parallelized across cores.

    Default backend is :class:`ThreadPoolExecutor` (works well because
    tblite's analytical-gradient call releases the GIL during the
    C-side compute). Pass ``use_processes=True`` for
    :class:`ProcessPoolExecutor` if you need fork isolation.

    Args:
        conformers: list of ``(atoms, coords_ang)`` tuples or
            ``(atoms, coords_ang, charge)`` triples (per-conformer
            charge override).
        charge: default integer net charge for any conformer that
            doesn't specify its own.
        max_iter / gtol_bohr / etol / stol_bohr: ANCopt tolerances.
        n_workers: number of workers. ``None`` → ``os.cpu_count()``.
        use_processes: True for process pool (fork isolation), False
            for thread pool (default — lower overhead).

    Returns:
        List of per-conformer result dicts (same shape as
        :func:`gfn2_alpb_water_optimize`).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    jobs = []
    for c in conformers:
        if len(c) == 2:
            atoms_i, coords_i = c
            chg_i = charge
        else:
            atoms_i, coords_i, chg_i = c
        jobs.append((atoms_i, coords_i, chg_i, max_iter, gtol_bohr, etol, stol_bohr))

    Pool = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with Pool(max_workers=n_workers) as ex:
        return list(ex.map(_opt_worker, jobs))


def gfn2_alpb_water_optimize(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    max_iter: int = 200,
    gtol_bohr: float = 1e-3,
    etol: float = 5e-6,
    stol_bohr: float = 1e-3,
    verbose: bool = False,
) -> dict:
    """ANCopt geometry optimization with GFN2-xTB + ALPB(water).

    Production entry point for the openCOSMO-RS conformer pipeline —
    replaces external ``xtb --opt --alpb water`` calls with an
    in-process ANCopt loop driven by tblite's analytical gradient.

    Args:
        atoms: atomic numbers.
        coords_ang: ``(n, 3)`` initial Angstrom coordinates.
        charge: integer net charge.
        max_iter / gtol_bohr / etol / stol_bohr: ANCopt convergence
            knobs (defaults match xtb ``opt_level=normal``).
        verbose: per-iter diagnostics from ANCopt.

    Returns:
        Dict from :func:`mlxmolkit.xtb.optimizer.ancopt`. Key fields:
        ``coords`` (Å), ``energy`` (Ha, including ALPB), ``converged``,
        ``n_iter``, ``trajectory``.

    Notes:
        Each ANCopt micro-iter calls tblite once for ``(E, ∇E)``. The
        tblite path is used because (1) ALPB analytical gradient
        already exists upstream, (2) it's the canonical reference for
        the openCOSMO-RS pipeline. The pure-MLX SCF (faithfully
        reproducing tblite's GFN2 to sub-3 kcal/mol) is exposed
        separately via :func:`gfn2_energy_alpb_water` for analysis,
        and a fully MLX-native opt path will land alongside the
        analytical multipole-derivative kernels and a pure-MLX ALPB.
    """
    from .optimizer import ancopt

    coords = np.asarray(coords_ang, dtype=np.float64).copy()
    calc = _tblite_alpb_water_calc(method="GFN2-xTB")

    def _calc_charged(atoms_, coords_):
        return calc(atoms_, coords_, charge=charge)

    return ancopt(
        atoms,
        coords,
        _calc_charged,
        max_iter=max_iter,
        gtol_bohr=gtol_bohr,
        etol=etol,
        stol_bohr=stol_bohr,
        verbose=verbose,
    )
