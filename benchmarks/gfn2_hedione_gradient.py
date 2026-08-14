#!/usr/bin/env python3
"""Benchmark GFN2 energy/gradient on Hedione.

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \
        benchmarks/gfn2_hedione_gradient.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from mlxmolkit.xtb.gradient_gfn2 import gfn2_gradient_analytical
from mlxmolkit.xtb.gradient_gfn2_fast import gfn2_gradient_analytical_fast
from mlxmolkit.xtb.multipole_integrals_cpp import CPP_AVAILABLE
from mlxmolkit.xtb.scf_gfn2 import gfn2_energy
from mlxmolkit.xtb.scf_gfn2_fast import gfn2_energy_fast
from mlxmolkit.xtb.scf_gfn2_mlx import gfn2_energy_mlx


MOLECULES = {
    # Hedione / methyl dihydrojasmonate, C13H22O3.
    "hedione": "CCCCCC1C(CCC1=O)CC(=O)OC",
    # Small perfume-like controls.
    "linalool": "CC(=CCCC(C)(C=C)O)C",
    "vanillin": "COc1cc(C=O)ccc1O",
}


def rdkit_coords(smiles: str, seed: int = 42) -> tuple[list[int], np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    conf = mol.GetConformer()
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    return atoms, coords


def timed(fn, repeat: int) -> tuple[object, list[float]]:
    out = None
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", choices=sorted(MOLECULES), default="hedione")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--conv-tol", type=float, default=1e-7)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Also time gfn2_gradient_analytical. This is much slower than energy.",
    )
    parser.add_argument(
        "--compare-mlx",
        action="store_true",
        help="Also time the experimental MLX float32 sidecar energy path.",
    )
    parser.add_argument(
        "--compare-fast",
        action="store_true",
        help="Also time the vectorized float64 fast sidecar energy path.",
    )
    parser.add_argument(
        "--fast-gradient",
        action="store_true",
        help="When --gradient is set, also time the fast gradient sidecar.",
    )
    parser.add_argument(
        "--only-fast-gradient",
        action="store_true",
        help="With --gradient, skip the slow reference gradient and time only the fast sidecar.",
    )
    parser.add_argument("--fd-workers", type=int, default=None)
    args = parser.parse_args()

    atoms, coords = rdkit_coords(MOLECULES[args.molecule])
    scf_kwargs = {"conv_tol": args.conv_tol, "max_iter": args.max_iter}
    base_energy_fn = gfn2_energy_fast if args.only_fast_gradient else gfn2_energy
    base_energy_label = "energy_fast" if args.only_fast_gradient else "energy"

    # Warm Metal/MLX and SciPy imports once before timing.
    warm = base_energy_fn(atoms, coords, **scf_kwargs)

    e_out, e_times = timed(lambda: base_energy_fn(atoms, coords, **scf_kwargs), args.repeat)
    assert e_out is not None

    print(f"molecule: {args.molecule}")
    print(f"smiles:   {MOLECULES[args.molecule]}")
    print(f"atoms:    {len(atoms)}")
    print(f"basis:    {warm['n_basis']}")
    print(f"cpp multipole: {CPP_AVAILABLE}")
    print(f"{base_energy_label}:   {e_out['energy_hartree']:.12f} Ha")
    print(f"scf iters last: {e_out['n_iter']} converged={e_out['converged']}")
    print(
        f"{base_energy_label} timing (s): "
        f"median={statistics.median(e_times):.4f} "
        f"min={min(e_times):.4f} max={max(e_times):.4f} "
        f"runs={[round(t, 4) for t in e_times]}"
    )

    if args.compare_mlx:
        mlx_warm = gfn2_energy_mlx(atoms, coords, allow_float32=True, **scf_kwargs)
        mlx_out, mlx_times = timed(
            lambda: gfn2_energy_mlx(
                atoms, coords, allow_float32=True, **scf_kwargs
            ),
            args.repeat,
        )
        assert mlx_out is not None
        print("energy_mlx note: experimental MLX GPU float32 sidecar")
        print(f"energy_mlx: {mlx_out['energy_hartree']:.12f} Ha")
        print(f"energy Δ mlx-ref: {mlx_out['energy_hartree'] - e_out['energy_hartree']:.3e} Ha")
        print(f"scf iters mlx last: {mlx_out['n_iter']} converged={mlx_out['converged']}")
        print(
            "energy_mlx timing (s): "
            f"median={statistics.median(mlx_times):.4f} "
            f"min={min(mlx_times):.4f} max={max(mlx_times):.4f} "
            f"runs={[round(t, 4) for t in mlx_times]}"
        )

    if args.compare_fast:
        fast_warm = gfn2_energy_fast(atoms, coords, **scf_kwargs)
        fast_out, fast_times = timed(
            lambda: gfn2_energy_fast(atoms, coords, **scf_kwargs), args.repeat
        )
        assert fast_out is not None
        print(f"energy_fast: {fast_out['energy_hartree']:.12f} Ha")
        print(f"energy Δ fast-ref: {fast_out['energy_hartree'] - e_out['energy_hartree']:.3e} Ha")
        print(f"scf iters fast last: {fast_out['n_iter']} converged={fast_out['converged']}")
        print(
            "energy_fast timing (s): "
            f"median={statistics.median(fast_times):.4f} "
            f"min={min(fast_times):.4f} max={max(fast_times):.4f} "
            f"runs={[round(t, 4) for t in fast_times]}"
        )

    if args.gradient:
        g_out = None
        if not args.only_fast_gradient:
            g_out, g_times = timed(
                lambda: gfn2_gradient_analytical(atoms, coords, scf_kwargs=scf_kwargs),
                args.repeat,
            )
            assert g_out is not None
            grad = g_out["gradient"]
            print(f"gradient max |g|: {np.max(np.abs(grad)):.6e} Ha/A")
            print(
                "gradient timing (s): "
                f"median={statistics.median(g_times):.4f} "
                f"min={min(g_times):.4f} max={max(g_times):.4f} "
                f"runs={[round(t, 4) for t in g_times]}"
            )
            print("component max |.| (Ha/A):")
            for name, value in g_out["components"].items():
                print(f"  {name:>14}: {np.max(np.abs(value)):.6e}")

        if args.fast_gradient or args.only_fast_gradient:
            gf_out, gf_times = timed(
                lambda: gfn2_gradient_analytical_fast(
                    atoms, coords, scf_kwargs=scf_kwargs, fd_workers=args.fd_workers
                ),
                args.repeat,
            )
            assert gf_out is not None
            print(f"gradient_fast max |g|: {np.max(np.abs(gf_out['gradient'])):.6e} Ha/A")
            print(
                "gradient_fast timing (s): "
                f"median={statistics.median(gf_times):.4f} "
                f"min={min(gf_times):.4f} max={max(gf_times):.4f} "
                f"runs={[round(t, 4) for t in gf_times]}"
            )
            if g_out is not None:
                print(
                    "gradient_fast max |Δ ref|: "
                    f"{np.max(np.abs(gf_out['gradient'] - g_out['gradient'])):.6e} Ha/A"
                )


if __name__ == "__main__":
    main()
