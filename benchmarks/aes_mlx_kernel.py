#!/usr/bin/env python3
"""Kernel-level parity/timing for experimental AES MLX functions."""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

from benchmarks.gfn2_hedione_gradient import MOLECULES, rdkit_coords
from mlxmolkit.xtb.aes import fockelectro, get_radcn, mmomgabzero, mmompop, setvsdq
from mlxmolkit.xtb.aes_mlx import (
    fockelectro_mlx,
    get_radcn_mlx,
    mmomgabzero_mlx,
    mmompop_mlx,
)
from mlxmolkit.xtb.gradient_gfn2 import _ANG_TO_BOHR
from mlxmolkit.xtb.scf_gfn2 import gfn2_energy


def bench(fn, repeat: int) -> tuple[object, list[float]]:
    out = None
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out) if isinstance(out, mx.array) else None
        times.append(time.perf_counter() - t0)
    return out, times


def eval_tree(x):
    if isinstance(x, tuple):
        mx.eval(*[v for v in x if isinstance(v, mx.array)])
    elif isinstance(x, mx.array):
        mx.eval(x)


def bench_tree(fn, repeat: int) -> tuple[object, list[float]]:
    out = None
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        eval_tree(out)
        times.append(time.perf_counter() - t0)
    return out, times


def fmt(times: list[float]) -> str:
    return (
        f"median={statistics.median(times) * 1e3:.3f} ms "
        f"min={min(times) * 1e3:.3f} ms max={max(times) * 1e3:.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", choices=sorted(MOLECULES), default="hedione")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--conv-tol", type=float, default=1e-7)
    args = parser.parse_args()

    atoms, coords = rdkit_coords(MOLECULES[args.molecule])
    res = gfn2_energy(atoms, coords, conv_tol=args.conv_tol)
    coords_bohr = coords * _ANG_TO_BOHR
    aoat = np.array([b.atom_idx for b in res["sao_basis"]], dtype=np.int64)

    radcn = get_radcn(atoms, res["coordination_number"])
    gab3, gab5 = mmomgabzero(coords_bohr, radcn)
    dipm, qp = mmompop(
        res["density"], res["S"], res["dpint"], res["qpint"], aoat, coords_bohr
    )
    vs, vd, vq = setvsdq(
        atoms, coords_bohr, res["atom_charges"], dipm, qp, gab3, gab5
    )

    P_mx = mx.array(res["density"])
    S_mx = mx.array(res["S"])
    dp_mx = mx.array(res["dpint"])
    qpint_mx = mx.array(res["qpint"])
    aoat_mx = mx.array(aoat)
    vs_mx = mx.array(vs)
    vd_mx = mx.array(vd)
    vq_mx = mx.array(vq)
    atoms_mx = mx.array(np.asarray(atoms, dtype=np.int32))
    coords_mx = mx.array(coords_bohr)
    cn_mx = mx.array(res["coordination_number"])

    F_np, e_np = fockelectro(
        res["density"], res["S"], res["dpint"], res["qpint"], aoat, vs, vd, vq
    )
    F_mx, e_mx = fockelectro_mlx(P_mx, S_mx, dp_mx, qpint_mx, aoat_mx, vs_mx, vd_mx, vq_mx)
    mx.eval(F_mx, e_mx)

    rad_mx = get_radcn_mlx(atoms_mx, cn_mx)
    g3_mx, g5_mx = mmomgabzero_mlx(coords_mx, rad_mx)
    dipm_mx, qp_mx = mmompop_mlx(P_mx, S_mx, dp_mx, qpint_mx, aoat_mx, coords_mx)
    mx.eval(rad_mx, g3_mx, g5_mx, dipm_mx, qp_mx)

    print(f"molecule: {args.molecule}")
    print(f"atoms:    {len(atoms)}")
    print(f"basis:    {res['n_basis']}")
    print("note:     MLX kernels use GPU float32; NumPy reference uses float64")
    print(f"F parity max |Δ|: {np.max(np.abs(np.asarray(F_mx) - F_np)):.3e}")
    print(f"E parity |Δ|:     {abs(float(np.asarray(e_mx)) - e_np):.3e}")
    print(f"radcn max |Δ|:    {np.max(np.abs(np.asarray(rad_mx) - radcn)):.3e}")
    print(f"gab3 max |Δ|:     {np.max(np.abs(np.asarray(g3_mx) - gab3)):.3e}")
    print(f"gab5 max |Δ|:     {np.max(np.abs(np.asarray(g5_mx) - gab5)):.3e}")
    print(f"dipm max |Δ|:     {np.max(np.abs(np.asarray(dipm_mx) - dipm)):.3e}")
    print(f"qp max |Δ|:       {np.max(np.abs(np.asarray(qp_mx) - qp)):.3e}")

    _, t_np = bench_tree(
        lambda: fockelectro(
            res["density"], res["S"], res["dpint"], res["qpint"], aoat, vs, vd, vq
        ),
        args.repeat,
    )
    _, t_mx = bench_tree(
        lambda: fockelectro_mlx(
            P_mx, S_mx, dp_mx, qpint_mx, aoat_mx, vs_mx, vd_mx, vq_mx
        ),
        args.repeat,
    )
    _, t_gab_np = bench_tree(lambda: mmomgabzero(coords_bohr, radcn), args.repeat)
    _, t_gab_mx = bench_tree(lambda: mmomgabzero_mlx(coords_mx, rad_mx), args.repeat)
    _, t_mpop_np = bench_tree(
        lambda: mmompop(
            res["density"], res["S"], res["dpint"], res["qpint"], aoat, coords_bohr
        ),
        args.repeat,
    )
    _, t_mpop_mx = bench_tree(
        lambda: mmompop_mlx(P_mx, S_mx, dp_mx, qpint_mx, aoat_mx, coords_mx),
        args.repeat,
    )

    print(f"fockelectro numpy: {fmt(t_np)}")
    print(f"fockelectro MLX:   {fmt(t_mx)}")
    print(f"gab numpy:         {fmt(t_gab_np)}")
    print(f"gab MLX:           {fmt(t_gab_mx)}")
    print(f"mmompop numpy:     {fmt(t_mpop_np)}")
    print(f"mmompop MLX:       {fmt(t_mpop_mx)}")


if __name__ == "__main__":
    main()
