"""
MMFF force field optimization using RDKit energy/gradient + Metal L-BFGS.

Pipeline matching nvMolKit's conformer_generation_optimization_workflow:
  1. Load/generate molecules (RDKit)
  2. Embed conformers (ETKDG, RDKit CPU)
  3. MMFF optimization (RDKit energy/grad + Metal L-BFGS)

The RDKit ForceField provides the MMFF94 energy and gradient. Our Metal
L-BFGS handles the optimization loop, which is faster than RDKit's
built-in optimizer for large molecules (the Hessian direction computation
and line search benefit from Metal).

Reference:
  - nvMolKit workflow: https://github.com/NVIDIA-Digital-Bio/nvMolKit/blob/main/examples/conformer_generation_optimization_workflow.ipynb
  - MMFF94: Halgren, J. Comput. Chem. 1996, 17, 490-519
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mlx.core as mx

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from mlxmolkit.bfgs_metal import BfgsResult, lbfgs_minimize, bfgs_minimize, EnergyGradFn


@dataclass
class MMFFResult:
    """Result of MMFF optimization for a single conformer."""
    energy: float
    positions: np.ndarray
    grad_norm: float
    n_iters: int
    converged: bool


@dataclass
class ConformerResult:
    """Result of conformer generation + MMFF optimization."""
    mol: Chem.Mol
    energies: list[float]
    n_conformers: int
    embed_time: float
    optimize_time: float


def _make_mmff_energy_grad(mol: Chem.Mol, conf_id: int = 0):
    """
    Create energy/gradient function backed by RDKit's MMFF94 ForceField.

    IMPORTANT: The ForceField is recreated on each call because RDKit's
    MMFF ForceField stores an internal copy of positions at creation time.
    Merely calling conf.SetAtomPosition() does NOT correctly update all
    ForceField internal state (non-bonded pair lists, cached distances).
    Only a fresh ForceField gives correct energy and gradient.

    Args:
        mol: RDKit molecule with at least one conformer.
        conf_id: conformer ID to optimize.

    Returns:
        (energy_grad_fn, n_atoms) tuple.
    """
    mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if mmff_props is None:
        raise ValueError("Could not compute MMFF properties for molecule")

    n_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer(conf_id)

    def energy_grad_fn(pos_flat: mx.array) -> tuple[mx.array, mx.array]:
        pos_np = np.array(pos_flat, dtype=np.float64).reshape(n_atoms, 3)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, pos_np[i].tolist())

        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id,
        )
        e = ff.CalcEnergy()
        g = np.array(ff.CalcGrad(), dtype=np.float32)
        return mx.array([e], dtype=mx.float32), mx.array(g, dtype=mx.float32)

    return energy_grad_fn, n_atoms


def mmff_optimize(
    mol: Chem.Mol,
    conf_id: int = 0,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    method: str = "bfgs",
    m: int = 10,
    scale_grads: bool = True,
) -> MMFFResult:
    """
    Optimize a single conformer with MMFF94 using Metal BFGS.

    Matches nvMolKit: gradient scaling is handled INSIDE the BFGS loop
    (not in the energy function), so the optimizer sees unscaled energy
    and raw gradient, then applies scale_grads per nvMolKit's scaleGradKernel.

    Args:
        mol: RDKit molecule with at least one conformer.
        conf_id: conformer ID to optimize.
        max_iters: maximum BFGS/L-BFGS iterations.
        grad_tol: gradient convergence tolerance.
        method: "bfgs" (default, like nvMolKit) or "lbfgs".
        m: L-BFGS history size (only for method="lbfgs").
        scale_grads: apply nvMolKit-style gradient scaling (default True).

    Returns:
        MMFFResult with optimized energy, positions, etc.
    """
    energy_grad_fn, n_atoms = _make_mmff_energy_grad(mol, conf_id)

    conf = mol.GetConformer(conf_id)
    pos = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(n_atoms)],
        dtype=np.float32,
    ).flatten()
    x0 = mx.array(pos, dtype=mx.float32)

    if method == "lbfgs":
        result = lbfgs_minimize(x0, energy_grad_fn, max_iters=max_iters,
                                grad_tol=grad_tol, m=m, scale_grads=scale_grads)
    else:
        result = bfgs_minimize(x0, energy_grad_fn, max_iters=max_iters,
                               grad_tol=grad_tol, scale_grads=scale_grads)

    final_pos = result.x.reshape(n_atoms, 3)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, final_pos[i].tolist())

    return MMFFResult(
        energy=result.energy,
        positions=final_pos,
        grad_norm=result.grad_norm,
        n_iters=result.n_iters,
        converged=result.converged,
    )


def mmff_optimize_confs(
    mol: Chem.Mol,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    method: str = "bfgs",
    m: int = 10,
    scale_grads: bool = True,
) -> list[MMFFResult]:
    """
    Optimize all conformers of a molecule with MMFF94.

    Args:
        mol: RDKit molecule with one or more conformers.
        max_iters: maximum iterations per conformer.
        grad_tol: gradient tolerance.
        method: "bfgs" (default) or "lbfgs".
        m: L-BFGS history size.
        scale_grads: apply nvMolKit-style gradient scaling.

    Returns:
        List of MMFFResult, one per conformer.
    """
    results = []
    for conf in mol.GetConformers():
        r = mmff_optimize(mol, conf_id=conf.GetId(), max_iters=max_iters,
                          grad_tol=grad_tol, method=method, m=m,
                          scale_grads=scale_grads)
        results.append(r)
    return results


def generate_and_optimize(
    smiles: str,
    n_conformers: int = 5,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    random_seed: int = 42,
    method: str = "bfgs",
    scale_grads: bool = True,
) -> ConformerResult:
    """
    Full pipeline: SMILES → conformers → MMFF optimization.

    Matches nvMolKit's workflow:
      1. Parse SMILES, add Hs
      2. Embed conformers with ETKDG
      3. MMFF94 optimization with Metal BFGS + gradient scaling

    Args:
        smiles: SMILES string.
        n_conformers: number of conformers to generate.
        max_iters: max BFGS iterations per conformer.
        grad_tol: gradient tolerance.
        random_seed: random seed for ETKDG.
        method: "bfgs" (default) or "lbfgs".
        scale_grads: apply nvMolKit-style gradient scaling (default True).

    Returns:
        ConformerResult with optimized molecule and energies.
    """
    import time

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True

    t0 = time.perf_counter()
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    embed_time = time.perf_counter() - t0

    if len(conf_ids) == 0:
        raise RuntimeError(f"Failed to generate conformers for {smiles}")

    t0 = time.perf_counter()
    results = mmff_optimize_confs(mol, max_iters=max_iters, grad_tol=grad_tol,
                                  method=method, scale_grads=scale_grads)
    optimize_time = time.perf_counter() - t0

    energies = [r.energy for r in results]

    return ConformerResult(
        mol=mol,
        energies=energies,
        n_conformers=len(conf_ids),
        embed_time=embed_time,
        optimize_time=optimize_time,
    )
