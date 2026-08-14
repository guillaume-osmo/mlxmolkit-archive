#!/usr/bin/env python3
"""Build an incremental deltaHvapv3 training table with completed ORCA rows.

The output appends rows from the v3-new queue that already have a completed
ORCA/COSMORS cache to the current 3147-row training table. The sigma NPZ is
row-aligned to the appended CSV: current rows come from an existing hybrid
NPZ, new rows come directly from the completed ORCA cache files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT_CSV = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv3_current3147_strict_train.csv"
DEFAULT_CURRENT_SIGMA = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_hybrid_gxtb_trusted_orca_cache866_sigma.npz"
DEFAULT_QUEUE_CSV = REPO_ROOT / "benchmarks/delta_hvap_v3_conflict_web_review/deltaHvapv3_new1347_sigma_queue.csv"
DEFAULT_CACHE = REPO_ROOT / "data/delta_hvap_v2/orca_cosmors_v3_new1347_molcache"
DEFAULT_OUT_CSV = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv3_current3147_plus_orca_new_completed_strict_train.csv"
DEFAULT_OUT_NPZ = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv3_current3147_plus_orca_new_completed_sigma.npz"


def scalar(value: object) -> object:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def load_completed_cache(cache_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in cache_dir.glob("*/*/*.npz"):
        try:
            z = np.load(path, allow_pickle=True)
            inchikey = str(scalar(z["inchikey"]))
        except Exception:
            continue
        if inchikey:
            out[inchikey] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-csv", type=Path, default=DEFAULT_CURRENT_CSV)
    parser.add_argument("--current-sigma", type=Path, default=DEFAULT_CURRENT_SIGMA)
    parser.add_argument("--queue-csv", type=Path, default=DEFAULT_QUEUE_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ)
    args = parser.parse_args()

    current = pd.read_csv(args.current_csv, low_memory=False)
    queue = pd.read_csv(args.queue_csv, low_memory=False)
    base = np.load(args.current_sigma, allow_pickle=True)

    current_canon = current["canonical_smiles"].astype(str).to_numpy()
    base_canon = np.asarray(base["canonical_smiles"]).astype(str)
    if len(current_canon) != len(base_canon) or not np.array_equal(current_canon, base_canon):
        raise ValueError("current CSV and current sigma NPZ are not row-aligned")

    completed = load_completed_cache(args.cache_dir)
    queue = queue.copy()
    queue["_cache_path"] = queue["autovap_inchikey"].astype(str).map(lambda x: str(completed.get(x, "")))
    new_rows = queue[queue["_cache_path"].astype(bool)].copy()
    new_rows = new_rows[~new_rows["canonical_smiles"].astype(str).isin(set(current["canonical_smiles"].astype(str)))].copy()
    new_rows = new_rows.drop_duplicates("canonical_smiles").reset_index(drop=True)

    out_df = pd.concat([current, new_rows.reindex(columns=current.columns, fill_value=np.nan)], ignore_index=True)
    for col in new_rows.columns:
        if col not in out_df.columns:
            out_df[col] = pd.Series([np.nan] * len(out_df), dtype=object)
    # Fill appended rows with all queue columns where current did not have them.
    start = len(current)
    for col in new_rows.columns:
        out_df.loc[start:, col] = new_rows[col].to_numpy()
    out_df["target_source"] = out_df["target_source"].fillna("")
    out_df["sample_weight"] = pd.to_numeric(out_df["sample_weight"], errors="coerce").fillna(1.0)
    out_df["trusted_target_kJmol"] = pd.to_numeric(out_df["trusted_target_kJmol"], errors="coerce")
    out_df["curated_target_kJmol"] = pd.to_numeric(out_df["curated_target_kJmol"], errors="coerce").fillna(out_df["trusted_target_kJmol"])

    base_mu = np.asarray(base["mu_J_per_mol"], dtype=np.float64)
    base_profile = np.asarray(base["profile_area_A2"], dtype=np.float64)
    grid = np.asarray(base["sigma_grid_e_per_A2"], dtype=np.float64)
    new_mu: list[np.ndarray] = []
    new_profile: list[np.ndarray] = []
    new_valid: list[bool] = []
    new_area: list[float] = []
    new_volume: list[float] = []
    new_cache_paths: list[str] = []
    for path_str in new_rows["_cache_path"].astype(str):
        z = np.load(path_str, allow_pickle=True)
        cache_grid = np.asarray(z["sigma_grid_e_per_A2"], dtype=np.float64)
        if cache_grid.shape != grid.shape or not np.allclose(cache_grid, grid):
            raise ValueError(f"sigma grid mismatch for {path_str}")
        new_mu.append(np.asarray(z["mu_J_per_mol"], dtype=np.float64))
        new_profile.append(np.asarray(z["profile_area_A2"], dtype=np.float64))
        new_valid.append(True)
        new_area.append(float(scalar(z["area_A2"])))
        new_volume.append(float(scalar(z["volume_A3"])))
        new_cache_paths.append(path_str)

    n_new = len(new_rows)
    mu = np.concatenate([base_mu, np.vstack(new_mu) if n_new else np.empty((0, base_mu.shape[1]))], axis=0)
    profile = np.concatenate([base_profile, np.vstack(new_profile) if n_new else np.empty((0, base_profile.shape[1]))], axis=0)
    valid = np.concatenate([np.asarray(base["valid_mask"], dtype=bool), np.asarray(new_valid, dtype=bool)])
    source_method = np.concatenate([
        np.asarray(base["source_method"]).astype(object) if "source_method" in base.files else np.full(len(current), "hybrid_base", dtype=object),
        np.full(n_new, "orca_cosmors_water_new", dtype=object),
    ])
    cache_path = np.concatenate([
        np.asarray(base["cache_path"]).astype(object) if "cache_path" in base.files else np.full(len(current), "", dtype=object),
        np.asarray(new_cache_paths, dtype=object),
    ])
    area = np.concatenate([
        np.asarray(base["area_A2"], dtype=np.float64) if "area_A2" in base.files else np.full(len(current), np.nan),
        np.asarray(new_area, dtype=np.float64),
    ])
    volume = np.concatenate([
        np.asarray(base["volume_A3"], dtype=np.float64) if "volume_A3" in base.files else np.full(len(current), np.nan),
        np.asarray(new_volume, dtype=np.float64),
    ])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    np.savez_compressed(
        args.out_npz,
        method=np.asarray("deltaHvapv3_current_plus_completed_orca_new"),
        sigma_grid_e_per_A2=grid,
        mu_J_per_mol=mu,
        profile_area_A2=profile,
        valid_mask=valid,
        canonical_smiles=out_df["canonical_smiles"].astype(str).to_numpy(dtype=object),
        smiles=out_df.get("calc_smiles", out_df["canonical_smiles"]).astype(str).to_numpy(dtype=object),
        trusted_target_kJmol=out_df["trusted_target_kJmol"].to_numpy(dtype=np.float64),
        sample_weight=out_df["sample_weight"].to_numpy(dtype=np.float64),
        target_source=out_df["target_source"].astype(str).to_numpy(dtype=object),
        source_method=source_method,
        cache_path=cache_path,
        area_A2=area,
        volume_A3=volume,
    )

    summary = {
        "out_csv": str(args.out_csv),
        "out_npz": str(args.out_npz),
        "current_rows": int(len(current)),
        "new_completed_rows": int(n_new),
        "total_rows": int(len(out_df)),
        "new_target_source_counts": {str(k): int(v) for k, v in new_rows["target_source"].value_counts().items()},
        "source_method_counts": {str(k): int(v) for k, v in pd.Series(source_method).value_counts().items()},
    }
    args.out_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
