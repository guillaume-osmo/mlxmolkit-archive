#!/usr/bin/env python3
"""Assemble deltaHvapv2 ORCA/COSMORS sigma tensors from the molecule cache only.

This is the non-launching companion to ``build_delta_hvap_v2_orca_sigma.py``:
it never starts xTB or ORCA. It scans the stable InChIKey cache folders and
writes the same row-aligned NPZ/status files, with missing rows marked invalid.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_delta_hvap_v2_orca_sigma import (
    DEFAULT_CACHE,
    DEFAULT_OUT,
    DEFAULT_UNION,
    _assemble,
    _cache_paths,
    _mol_identity,
    _row_smiles,
    _select_rows,
    _valid_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-csv", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--subset", choices=["calc-only", "autovap", "overlap", "all"], default="calc-only")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT.with_name("deltaHvapv2_orca_cosmors_calc_only_sigma_cache_partial.npz"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-mode", choices=["auto", "single", "deep"], default="auto")
    parser.add_argument("--method", default="BP86")
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--solvent", default="water")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--orca-cores", type=int, default=1)
    parser.add_argument("--deep-n-conformers", type=int, default=20)
    parser.add_argument("--deep-n-keep", type=int, default=5)
    parser.add_argument("--deep-prune-rms", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.union_csv)
    selected_mask = _select_rows(df, args.subset)
    selected_indices = np.flatnonzero(selected_mask)
    status_path = args.out.with_suffix(".status.csv")
    run_args = {
        "run_mode": args.run_mode,
        "method": args.method,
        "basis": args.basis,
        "solvent": args.solvent,
        "seed": int(args.seed),
        "acc": float(args.acc),
        "orca_cores": int(args.orca_cores),
        "deep_n_conformers": int(args.deep_n_conformers),
        "deep_n_keep": int(args.deep_n_keep),
        "deep_prune_rms": float(args.deep_prune_rms),
    }

    results: list[dict[str, object]] = []
    for row_index in selected_indices:
        row = df.iloc[int(row_index)]
        smiles = _row_smiles(row)
        identity = _mol_identity(smiles, str(row["canonical_smiles"]))
        if not smiles or int(identity["n_fragments"]) != 1:
            results.append(
                {
                    "row_index": int(row_index),
                    "ok": False,
                    "cached": False,
                    "wall_s": math.nan,
                    "error": f"missing_or_non_single_fragment:n_fragments={identity['n_fragments']}",
                    "inchikey": str(identity["inchikey"]),
                    "mode": "",
                    "reason": "",
                }
            )
            continue

        cache_path, error_path, _metadata_path = _cache_paths(args.cache_dir, str(identity["inchikey"]), run_args)
        if cache_path.exists() and _valid_cache(cache_path):
            results.append(
                {
                    "row_index": int(row_index),
                    "ok": True,
                    "cached": True,
                    "cache_path": str(cache_path),
                    "wall_s": math.nan,
                    "mode": "",
                    "reason": "",
                    "inchikey": str(identity["inchikey"]),
                }
            )
        else:
            error = "cache_missing"
            if error_path.exists():
                error = f"cache_error:{error_path}"
            results.append(
                {
                    "row_index": int(row_index),
                    "ok": False,
                    "cached": False,
                    "wall_s": math.nan,
                    "error": error,
                    "inchikey": str(identity["inchikey"]),
                    "mode": "",
                    "reason": "",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _assemble(
        df,
        selected_mask,
        results,
        args.cache_dir,
        args.out,
        status_path,
        subset=args.subset,
        run_args=run_args,
    )

    ok = sum(1 for result in results if result.get("ok", False))
    failed = sum(1 for result in results if not result.get("ok", False))
    selected = int(np.sum(selected_mask))
    print(
        f"[deltaHvapv2-orca-cache-assemble] wrote {args.out} and {status_path} "
        f"valid_selected={ok}/{selected} missing_or_failed={failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
