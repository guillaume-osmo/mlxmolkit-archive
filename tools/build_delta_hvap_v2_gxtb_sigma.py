#!/usr/bin/env python3
"""Build degraded g-xTB/GFN2-tmCOSMO sigma tensors for deltaHvapv2.

This is the fast, reproducible baseline path:

    canonical SMILES -> RDKit 3D -> g-xTB geometry opt -> GFN2 tmCOSMO water

The expensive/fine-tuned ORCA COSMORS tensors remain separate. This script is
intentionally row-aligned to ``deltaHvapv2_homoset_union.csv`` so the 866
calcphyschemprop-only molecules can be generated first and compared later
against the 3162-row union without changing indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNION = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_homoset_union.csv"
DEFAULT_LEGACY_CACHE = REPO_ROOT / "data/delta_hvap_v2/gxtb_tmcosmo_cache"
DEFAULT_CACHE = REPO_ROOT / "data/delta_hvap_v2/gxtb_tmcosmo_molcache"
DEFAULT_XTB = Path("/tmp/gxtb-v2-macos/bin/xtb")
SIGMA_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)
METHOD = "rdkit3d_gxtb_opt_gfn2_tmcosmo"


def _ensure_repo_on_path() -> None:
    candidates = [
        str(REPO_ROOT),
        "/Users/guillaume-osmo/Github/mlx-addons/src",
    ]
    for item in reversed(candidates):
        if item and item not in sys.path:
            sys.path.insert(0, item)


def _as_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _row_smiles(row: pd.Series) -> str:
    """Choose the executable SMILES for a canonical union row."""

    for col in ("calc_smiles", "autovap_smiles", "canonical_smiles"):
        text = _as_text(row.get(col, ""))
        if not text:
            continue
        # Joined duplicate raw SMILES are pipe-separated in the homoset files.
        return text.split("|")[0].strip()
    return ""


def _select_rows(df: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones(len(df), dtype=bool)
    if subset == "calc-only":
        return df["target_source"].eq("calcphyschemprop_calibrated_pseudo").to_numpy()
    if subset == "autovap":
        return df["target_source"].eq("autovap_trusted").to_numpy()
    if subset == "overlap":
        return (
            df["target_source"].eq("autovap_trusted")
            & df["calc_deltaHvap_source_kJmol"].notna()
        ).to_numpy()
    raise ValueError(f"unknown subset: {subset}")


def _method_slug(solvent: str, seed: int, acc: float) -> str:
    acc_tag = f"{float(acc):g}".replace(".", "p").replace("-", "m")
    solvent_tag = str(solvent).replace("/", "_")
    return f"{METHOD}_{solvent_tag}_seed{int(seed)}_acc{acc_tag}"


def _mol_identity(smiles: str, canonical_hint: str = "") -> dict[str, object]:
    """Stable molecule identity for cache paths.

    Cache keys must not depend on row indices, because dataset curation changes
    row order. InChIKey gives us the durable molecule folder; canonical SMILES
    and a short SHA are persisted as safety metadata.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(canonical_hint)) if canonical_hint else None
    if mol is None:
        mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        digest = hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()[:20].upper()
        return {
            "canonical_smiles": str(canonical_hint or smiles),
            "inchikey": f"NOINCHIKEY-{digest}",
            "n_fragments": -1,
            "smiles_sha20": digest,
        }

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20].upper()
    inchikey = Chem.MolToInchiKey(mol) if hasattr(Chem, "MolToInchiKey") else f"NOINCHIKEY-{digest}"
    return {
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "n_fragments": len(Chem.GetMolFrags(mol)),
        "smiles_sha20": digest,
    }


def _cache_paths(cache_dir: Path, inchikey: str, solvent: str, seed: int, acc: float) -> tuple[Path, Path, Path]:
    prefix = str(inchikey)[:3].upper()
    mol_dir = cache_dir / prefix / str(inchikey)
    stem = _method_slug(solvent, seed, acc)
    return mol_dir / f"{stem}.npz", mol_dir / f"{stem}.error.json", mol_dir / "metadata.json"


def _sigma_profile_on_grid(cosmo: Any, grid: np.ndarray) -> np.ndarray:
    step = float(grid[1] - grid[0])
    edges = np.empty(grid.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = float(grid[0]) - 0.5 * step
    edges[-1] = float(grid[-1]) + 0.5 * step
    profile, _ = np.histogram(
        np.asarray(cosmo.segments_sigma, dtype=np.float64),
        bins=edges,
        weights=np.asarray(cosmo.segments_area, dtype=np.float64),
    )
    return np.asarray(profile, dtype=np.float64)


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _valid_cache(path: Path) -> bool:
    try:
        loaded = _load_npz(path)
        return (
            np.asarray(loaded["mu_J_per_mol"]).shape == SIGMA_GRID.shape
            and np.asarray(loaded["profile_area_A2"]).shape == SIGMA_GRID.shape
            and np.all(np.isfinite(np.asarray(loaded["mu_J_per_mol"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(loaded["profile_area_A2"], dtype=np.float64)))
        )
    except Exception:
        return False


def _scalar_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size:
        return str(arr.reshape(-1)[0])
    return ""


def _migrate_legacy_cache(
    legacy_dir: Path,
    mol_cache_dir: Path,
    *,
    solvent: str,
    seed: int,
    acc: float,
) -> dict[str, int]:
    """Copy old row-index caches into stable InChIKey folders."""

    stats = {"seen": 0, "valid": 0, "migrated": 0, "skipped_existing": 0, "skipped_fragment": 0, "invalid": 0}
    if not legacy_dir.exists():
        return stats

    for legacy_path in legacy_dir.glob("row*.npz"):
        stats["seen"] += 1
        if not _valid_cache(legacy_path):
            stats["invalid"] += 1
            continue
        stats["valid"] += 1
        loaded = _load_npz(legacy_path)
        smiles = _scalar_str(loaded.get("smiles", ""))
        canonical = _scalar_str(loaded.get("canonical_smiles", smiles))
        identity = _mol_identity(smiles, canonical)
        if int(identity["n_fragments"]) != 1:
            stats["skipped_fragment"] += 1
            continue
        target_path, _, metadata_path = _cache_paths(
            mol_cache_dir,
            str(identity["inchikey"]),
            solvent,
            seed,
            acc,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and _valid_cache(target_path):
            stats["skipped_existing"] += 1
            continue
        tmp_path = target_path.with_name(f".{target_path.stem}.migrate.tmp.npz")
        shutil.copy2(legacy_path, tmp_path)
        tmp_path.replace(target_path)
        metadata = {
            "inchikey": str(identity["inchikey"]),
            "canonical_smiles": str(identity["canonical_smiles"]),
            "source_smiles_examples": [smiles],
            "method": METHOD,
            "solvent": solvent,
            "seed": seed,
            "acc": acc,
            "n_fragments": int(identity["n_fragments"]),
            "smiles_sha20": str(identity["smiles_sha20"]),
            "cache_file": target_path.name,
            "migrated_from": str(legacy_path),
            "updated_at_unix": time.time(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        stats["migrated"] += 1
    return stats


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    _ensure_repo_on_path()

    row_index = int(task["row_index"])
    canonical_smiles = str(task["canonical_smiles"])
    smiles = str(task["smiles"])
    solvent = str(task["solvent"])
    seed = int(task["seed"])
    inchikey = str(task["inchikey"])
    identity_canonical = str(task["identity_canonical_smiles"])
    n_fragments = int(task["n_fragments"])
    smiles_sha20 = str(task["smiles_sha20"])
    xtb_path = Path(task["xtb_path"])
    cache_dir = Path(task["cache_dir"])
    keep_workdir = bool(task["keep_workdir"])
    force = bool(task["force"])
    acc = float(task["acc"])
    gxtb_timeout = float(task["gxtb_timeout"])

    cache_path, error_path, metadata_path = _cache_paths(cache_dir, inchikey, solvent, seed, acc)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force and _valid_cache(cache_path):
        loaded = _load_npz(cache_path)
        return {
            "row_index": row_index,
            "ok": True,
            "cache_path": str(cache_path),
            "cached": True,
            "wall_s": float(np.asarray(loaded.get("wall_s", np.nan)).reshape(-1)[0]),
            "inchikey": inchikey,
        }
    if cache_path.exists() and not force:
        cache_path.unlink(missing_ok=True)

    start = time.perf_counter()
    try:
        from mlxmolkit.xtb.cosmo_sigma import hybrid_gxtb_gfn2_cosmo_from_smiles, sigma_potential

        out = hybrid_gxtb_gfn2_cosmo_from_smiles(
            smiles,
            solvent=solvent,
            seed=seed,
            xtb_path=xtb_path,
            keep_workdir=keep_workdir,
            acc=acc,
            gxtb_timeout=gxtb_timeout,
        )
        cosmo = out["cosmo"]
        grid, mu = sigma_potential(cosmo, sigma_grid_e_per_A2=SIGMA_GRID)
        profile = _sigma_profile_on_grid(cosmo, SIGMA_GRID)
        wall_s = time.perf_counter() - start

        tmp_cache_path = cache_path.with_name(f".{cache_path.stem}.tmp.npz")
        np.savez_compressed(
            tmp_cache_path,
            row_index=np.asarray(row_index, dtype=np.int64),
            canonical_smiles=np.asarray(canonical_smiles),
            identity_canonical_smiles=np.asarray(identity_canonical),
            inchikey=np.asarray(inchikey),
            smiles_sha20=np.asarray(smiles_sha20),
            smiles=np.asarray(smiles),
            method=np.asarray(METHOD),
            solvent=np.asarray(solvent),
            sigma_grid_e_per_A2=np.asarray(grid, dtype=np.float64),
            mu_J_per_mol=np.asarray(mu, dtype=np.float64),
            profile_area_A2=np.asarray(profile, dtype=np.float64),
            area_A2=np.asarray(float(cosmo.area), dtype=np.float64),
            volume_A3=np.asarray(float(cosmo.volume), dtype=np.float64),
            total_screening_charge=np.asarray(float(cosmo.total_screening_charge), dtype=np.float64),
            total_energy_hartree=np.asarray(float(cosmo.total_energy_hartree), dtype=np.float64),
            dielectric_energy_hartree=np.asarray(float(cosmo.dielectric_energy_hartree), dtype=np.float64),
            gxtb_energy_hartree=np.asarray(float(out["gxtb_energy_hartree"]), dtype=np.float64),
            n_atoms=np.asarray(len(out["atoms"]), dtype=np.int64),
            n_segments=np.asarray(len(cosmo.segments_sigma), dtype=np.int64),
            n_fragments=np.asarray(n_fragments, dtype=np.int64),
            wall_s=np.asarray(wall_s, dtype=np.float64),
        )
        tmp_cache_path.replace(cache_path)
        metadata = {
            "inchikey": inchikey,
            "canonical_smiles": identity_canonical,
            "source_smiles_examples": [smiles],
            "method": METHOD,
            "solvent": solvent,
            "seed": seed,
            "acc": acc,
            "n_fragments": n_fragments,
            "smiles_sha20": smiles_sha20,
            "cache_file": str(cache_path.name),
            "updated_at_unix": time.time(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        if error_path.exists():
            error_path.unlink()
        return {
            "row_index": row_index,
            "ok": True,
            "cache_path": str(cache_path),
            "cached": False,
            "wall_s": wall_s,
            "inchikey": inchikey,
        }
    except Exception as exc:  # noqa: BLE001 - persist full failure for later audit.
        wall_s = time.perf_counter() - start
        payload = {
            "row_index": row_index,
            "canonical_smiles": canonical_smiles,
            "smiles": smiles,
            "inchikey": inchikey,
            "identity_canonical_smiles": identity_canonical,
            "n_fragments": n_fragments,
            "solvent": solvent,
            "seed": seed,
            "wall_s": wall_s,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        error_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return {
            "row_index": row_index,
            "ok": False,
            "cache_path": "",
            "error_path": str(error_path),
            "cached": False,
            "wall_s": wall_s,
            "error": repr(exc),
            "inchikey": inchikey,
        }


def _assemble_output(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    results: list[dict[str, Any]],
    cache_dir: Path,
    out_path: Path,
    status_path: Path,
    *,
    subset: str,
    solvent: str,
    seed: int,
) -> None:
    n = len(df)
    b = SIGMA_GRID.size
    mu = np.full((n, b), np.nan, dtype=np.float64)
    profile = np.full((n, b), np.nan, dtype=np.float64)
    area = np.full(n, np.nan, dtype=np.float64)
    volume = np.full(n, np.nan, dtype=np.float64)
    gxtb_e = np.full(n, np.nan, dtype=np.float64)
    cosmo_e = np.full(n, np.nan, dtype=np.float64)
    diel_e = np.full(n, np.nan, dtype=np.float64)
    n_atoms = np.full(n, -1, dtype=np.int64)
    n_segments = np.full(n, -1, dtype=np.int64)
    wall_s = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    cached = np.zeros(n, dtype=bool)
    error = np.full(n, "", dtype=object)
    cache_path_arr = np.full(n, "", dtype=object)
    inchikey_arr = np.full(n, "", dtype=object)

    by_row = {int(r["row_index"]): r for r in results}
    for row_index, result in by_row.items():
        wall_s[row_index] = float(result.get("wall_s", np.nan))
        cached[row_index] = bool(result.get("cached", False))
        inchikey_arr[row_index] = str(result.get("inchikey", ""))
        if not result.get("ok", False):
            error[row_index] = str(result.get("error", "failed"))
            continue
        path = Path(str(result["cache_path"]))
        cache_path_arr[row_index] = str(path)
        try:
            loaded = _load_npz(path)
            mu[row_index] = np.asarray(loaded["mu_J_per_mol"], dtype=np.float64)
            profile[row_index] = np.asarray(loaded["profile_area_A2"], dtype=np.float64)
            area[row_index] = float(np.asarray(loaded["area_A2"]).reshape(-1)[0])
            volume[row_index] = float(np.asarray(loaded["volume_A3"]).reshape(-1)[0])
            gxtb_e[row_index] = float(np.asarray(loaded["gxtb_energy_hartree"]).reshape(-1)[0])
            cosmo_e[row_index] = float(np.asarray(loaded["total_energy_hartree"]).reshape(-1)[0])
            diel_e[row_index] = float(np.asarray(loaded["dielectric_energy_hartree"]).reshape(-1)[0])
            n_atoms[row_index] = int(np.asarray(loaded["n_atoms"]).reshape(-1)[0])
            n_segments[row_index] = int(np.asarray(loaded["n_segments"]).reshape(-1)[0])
            if "inchikey" in loaded:
                inchikey_arr[row_index] = str(np.asarray(loaded["inchikey"]).reshape(-1)[0])
            valid[row_index] = True
        except Exception as exc:  # noqa: BLE001
            error[row_index] = f"cache_load_failed: {exc!r}"

    canonical = df["canonical_smiles"].astype(str).to_numpy(dtype=object)
    smiles = df.apply(_row_smiles, axis=1).astype(str).to_numpy(dtype=object)
    target = pd.to_numeric(df["trusted_target_kJmol"], errors="coerce").to_numpy(dtype=np.float64)
    curated = pd.to_numeric(df["curated_target_kJmol"], errors="coerce").to_numpy(dtype=np.float64)
    sample_weight = pd.to_numeric(df["sample_weight"], errors="coerce").to_numpy(dtype=np.float64)

    np.savez_compressed(
        out_path,
        method=np.asarray(METHOD),
        subset=np.asarray(subset),
        solvent=np.asarray(solvent),
        seed=np.asarray(seed, dtype=np.int64),
        cache_dir=np.asarray(str(cache_dir)),
        sigma_grid_e_per_A2=SIGMA_GRID,
        mu_J_per_mol=mu,
        profile_area_A2=profile,
        area_A2=area,
        volume_A3=volume,
        gxtb_energy_hartree=gxtb_e,
        tmcosmo_total_energy_hartree=cosmo_e,
        tmcosmo_dielectric_energy_hartree=diel_e,
        n_atoms=n_atoms,
        n_segments=n_segments,
        wall_s=wall_s,
        valid_mask=valid,
        selected_mask=selected_mask,
        cached_mask=cached,
        canonical_smiles=canonical,
        smiles=smiles,
        inchikey=inchikey_arr,
        trusted_target_kJmol=target,
        curated_target_kJmol=curated,
        sample_weight=sample_weight,
        target_source=df["target_source"].astype(str).to_numpy(dtype=object),
        curation_action=df["curation_action"].astype(str).to_numpy(dtype=object),
        error=error,
        cache_path=cache_path_arr,
    )

    status = pd.DataFrame(
        {
            "row_index": np.arange(n, dtype=np.int64),
            "selected": selected_mask,
            "valid": valid,
            "cached": cached,
            "canonical_smiles": canonical,
            "smiles": smiles,
            "inchikey": inchikey_arr,
            "target_source": df["target_source"].astype(str),
            "trusted_target_kJmol": target,
            "curation_action": df["curation_action"].astype(str),
            "gxtb_energy_hartree": gxtb_e,
            "tmcosmo_total_energy_hartree": cosmo_e,
            "area_A2": area,
            "volume_A3": volume,
            "n_atoms": n_atoms,
            "n_segments": n_segments,
            "wall_s": wall_s,
            "cache_path": cache_path_arr,
            "error": error,
        }
    )
    status.to_csv(status_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-csv", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--subset", choices=["calc-only", "autovap", "overlap", "all"], default="calc-only")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--legacy-cache-dir", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--no-migrate-legacy-cache", action="store_true")
    parser.add_argument("--xtb-path", type=Path, default=DEFAULT_XTB)
    parser.add_argument("--solvent", default="water")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--limit", type=int, default=0, help="debug limit after subset selection")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--gxtb-timeout", type=float, default=600.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if not args.xtb_path.exists():
        raise FileNotFoundError(f"xtb binary not found: {args.xtb_path}")

    df = pd.read_csv(args.union_csv)
    selected_mask = _select_rows(df, args.subset)
    selected_indices = np.flatnonzero(selected_mask)
    if args.limit and args.limit > 0:
        selected_indices = selected_indices[: args.limit]
        limited_mask = np.zeros(len(df), dtype=bool)
        limited_mask[selected_indices] = True
        selected_mask = limited_mask

    if args.out is None:
        suffix = args.subset.replace("-", "_")
        if args.limit and args.limit > 0:
            suffix += f"_limit{args.limit}"
        args.out = REPO_ROOT / f"data/delta_hvap_v2/deltaHvapv2_gxtb_tmcosmo_{suffix}_sigma.npz"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out.with_suffix(".status.csv")

    if not args.no_migrate_legacy_cache and args.legacy_cache_dir.exists():
        mig = _migrate_legacy_cache(
            args.legacy_cache_dir,
            args.cache_dir,
            solvent=args.solvent,
            seed=int(args.seed),
            acc=float(args.acc),
        )
        print(f"[deltaHvapv2-gxtb-sigma] migrated legacy row cache: {mig}", flush=True)

    tasks: list[dict[str, Any]] = []
    skipped_no_smiles: list[dict[str, Any]] = []
    for row_index in selected_indices:
        row = df.iloc[int(row_index)]
        smiles = _row_smiles(row)
        if not smiles:
            skipped_no_smiles.append(
                {
                    "row_index": int(row_index),
                    "ok": False,
                    "cached": False,
                    "cache_path": "",
                    "wall_s": math.nan,
                    "error": "missing_smiles",
                }
            )
            continue
        identity = _mol_identity(smiles, str(row["canonical_smiles"]))
        if int(identity["n_fragments"]) != 1:
            skipped_no_smiles.append(
                {
                    "row_index": int(row_index),
                    "ok": False,
                    "cached": False,
                    "cache_path": "",
                    "wall_s": math.nan,
                    "error": f"non_single_fragment_smiles:n_fragments={identity['n_fragments']}",
                    "inchikey": str(identity["inchikey"]),
                }
            )
            continue
        tasks.append(
            {
                "row_index": int(row_index),
                "canonical_smiles": str(row["canonical_smiles"]),
                "smiles": smiles,
                "identity_canonical_smiles": str(identity["canonical_smiles"]),
                "inchikey": str(identity["inchikey"]),
                "n_fragments": int(identity["n_fragments"]),
                "smiles_sha20": str(identity["smiles_sha20"]),
                "solvent": args.solvent,
                "seed": int(args.seed),
                "xtb_path": str(args.xtb_path),
                "cache_dir": str(args.cache_dir),
                "keep_workdir": bool(args.keep_workdir),
                "force": bool(args.force),
                "acc": float(args.acc),
                "gxtb_timeout": float(args.gxtb_timeout),
            }
        )

    print(
        f"[deltaHvapv2-gxtb-sigma] subset={args.subset} selected={len(selected_indices)} "
        f"tasks={len(tasks)} workers={args.workers} cache={args.cache_dir}",
        flush=True,
    )

    start = time.perf_counter()
    results: list[dict[str, Any]] = list(skipped_no_smiles)
    if tasks:
        if args.workers <= 1:
            for i, task in enumerate(tasks, 1):
                results.append(_worker(task))
                if args.progress_every and (i % args.progress_every == 0 or i == len(tasks)):
                    ok = sum(1 for r in results if r.get("ok", False))
                    print(
                        f"[deltaHvapv2-gxtb-sigma] done={i}/{len(tasks)} ok={ok} "
                        f"elapsed={time.perf_counter() - start:.1f}s",
                        flush=True,
                    )
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                futures = [pool.submit(_worker, task) for task in tasks]
                for i, fut in enumerate(as_completed(futures), 1):
                    results.append(fut.result())
                    if args.progress_every and (i % args.progress_every == 0 or i == len(futures)):
                        ok = sum(1 for r in results if r.get("ok", False))
                        failed = sum(1 for r in results if not r.get("ok", False))
                        print(
                            f"[deltaHvapv2-gxtb-sigma] done={i}/{len(futures)} ok={ok} "
                            f"failed={failed} elapsed={time.perf_counter() - start:.1f}s",
                            flush=True,
                        )

    _assemble_output(
        df,
        selected_mask,
        results,
        args.cache_dir,
        args.out,
        status_path,
        subset=args.subset,
        solvent=args.solvent,
        seed=args.seed,
    )

    ok = sum(1 for r in results if r.get("ok", False))
    failed = sum(1 for r in results if not r.get("ok", False))
    print(
        f"[deltaHvapv2-gxtb-sigma] wrote {args.out} and {status_path} "
        f"ok={ok} failed={failed} elapsed={time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
