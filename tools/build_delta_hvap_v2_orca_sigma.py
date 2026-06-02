#!/usr/bin/env python3
"""Build trusted ORCA/COSMORS sigma tensors for deltaHvapv2.

Pipeline:

    canonical SMILES -> RDKit 3D -> g-xTB opt -> ORCA BP86/def2-TZVP COSMORS

Auto mode matches ``cosmors_sigma_potential_auto``: simple molecules use a
single conformer; complex/flexible molecules use deep multi-conformer ORCA and
Boltzmann-weighted sigma tensors. Results are row-aligned to
``deltaHvapv2_homoset_union.csv`` and cached by stable InChIKey folders:

    data/delta_hvap_v2/orca_cosmors_molcache/ABC/INCHIKEY/*.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNION = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_homoset_union.csv"
DEFAULT_CACHE = REPO_ROOT / "data/delta_hvap_v2/orca_cosmors_molcache"
DEFAULT_OUT = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_orca_cosmors_calc_only_sigma.npz"
DEFAULT_XTB = Path("/tmp/gxtb-v2-macos/bin/xtb")
DEFAULT_ORCA = Path.home() / "Library/orca_6_1_0/orca"
SIGMA_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)
METHOD = "rdkit3d_gxtb_opt_orca_cosmors"


def _ensure_repo_on_path() -> None:
    for item in reversed((str(REPO_ROOT), "/Users/guillaume-osmo/Github/mlx-addons/src")):
        if item and item not in sys.path:
            sys.path.insert(0, item)


def _as_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _row_smiles(row: pd.Series) -> str:
    for col in ("calc_smiles", "autovap_smiles", "canonical_smiles"):
        text = _as_text(row.get(col, ""))
        if text:
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


def _mol_identity(smiles: str, canonical_hint: str = "") -> dict[str, object]:
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
            "formal_charge": 0,
            "uhf": 0,
            "n_electrons": -1,
        }
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20].upper()
    inchikey = Chem.MolToInchiKey(mol)
    formal_charge = int(Chem.GetFormalCharge(mol))
    mol_h = Chem.AddHs(mol)
    n_electrons = int(sum(atom.GetAtomicNum() for atom in mol_h.GetAtoms()) - formal_charge)
    n_radical = int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))
    uhf = n_radical if n_radical > 0 else int(n_electrons % 2)
    return {
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "n_fragments": len(Chem.GetMolFrags(mol)),
        "smiles_sha20": digest,
        "formal_charge": formal_charge,
        "uhf": uhf,
        "n_electrons": n_electrons,
    }


def _method_slug(args: dict[str, Any]) -> str:
    method = str(args["method"]).replace("/", "_")
    basis = str(args["basis"]).replace("/", "_")
    solvent = str(args["solvent"]).replace("/", "_")
    acc = f"{float(args['acc']):g}".replace(".", "p").replace("-", "m")
    return (
        f"{METHOD}_{args['run_mode']}_{method}_{basis}_{solvent}"
        f"_seed{int(args['seed'])}_acc{acc}"
        f"_dn{int(args['deep_n_conformers'])}_dk{int(args['deep_n_keep'])}"
    )


def _cache_paths(cache_dir: Path, inchikey: str, args: dict[str, Any]) -> tuple[Path, Path, Path]:
    mol_dir = cache_dir / str(inchikey)[:3].upper() / str(inchikey)
    stem = _method_slug(args)
    return mol_dir / f"{stem}.npz", mol_dir / f"{stem}.error.json", mol_dir / "metadata.json"


def _profile_on_grid(cosmo: Any, grid: np.ndarray) -> np.ndarray:
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
        d = _load_npz(path)
        return (
            np.asarray(d["mu_J_per_mol"]).shape == SIGMA_GRID.shape
            and np.asarray(d["profile_area_A2"]).shape == SIGMA_GRID.shape
            and np.all(np.isfinite(np.asarray(d["mu_J_per_mol"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(d["profile_area_A2"], dtype=np.float64)))
        )
    except Exception:
        return False


def _scalar(data: dict[str, Any], key: str, default: Any = np.nan) -> Any:
    if key not in data:
        return default
    arr = np.asarray(data[key])
    if arr.shape == ():
        return arr.item()
    return arr.reshape(-1)[0] if arr.size else default


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    _ensure_repo_on_path()

    row_index = int(task["row_index"])
    smiles = str(task["smiles"])
    canonical_smiles = str(task["canonical_smiles"])
    inchikey = str(task["inchikey"])
    charge = int(task.get("charge", 0))
    uhf = int(task.get("uhf", 0))
    cache_dir = Path(task["cache_dir"])
    force = bool(task["force"])
    run_args = dict(task["run_args"])
    cache_path, error_path, metadata_path = _cache_paths(cache_dir, inchikey, run_args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force and _valid_cache(cache_path):
        d = _load_npz(cache_path)
        return {
            "row_index": row_index,
            "ok": True,
            "cached": True,
            "cache_path": str(cache_path),
            "wall_s": float(_scalar(d, "wall_s", np.nan)),
            "mode": str(_scalar(d, "mode", "")),
            "reason": str(_scalar(d, "reason", "")),
            "inchikey": inchikey,
            "charge": charge,
            "uhf": uhf,
        }
    if cache_path.exists() and not force:
        cache_path.unlink(missing_ok=True)

    start = time.perf_counter()
    workdirs_to_cleanup: list[Path] = []
    try:
        from mlxmolkit.xtb.cosmo_sigma import (
            cosmosegments_from_orcacosmo,
            is_complex_case,
            sigma_potential,
            sigma_potential_ensemble,
            tiered_gxtb_orca_cosmors_from_smiles,
            tiered_multiconformer_gxtb_orca,
        )

        xtb_path = Path(str(task["xtb_path"]))
        orca_path = Path(str(task["orca_path"]))
        complex_flag, reason = is_complex_case(smiles)
        run_mode = str(run_args["run_mode"])
        use_deep = run_mode == "deep" or (run_mode == "auto" and bool(complex_flag))

        if use_deep:
            mp = tiered_multiconformer_gxtb_orca(
                smiles,
                n_conformers=int(run_args["deep_n_conformers"]),
                n_keep=int(run_args["deep_n_keep"]),
                seed=int(run_args["seed"]),
                xtb_path=xtb_path,
                orca_path=orca_path,
                method=str(run_args["method"]),
                basis=str(run_args["basis"]),
                solvent=str(run_args["solvent"]),
                charge=charge,
                uhf=uhf,
                orca_cores=int(run_args["orca_cores"]),
                acc=float(run_args["acc"]),
                screen_with_solvent=True,
                use_exp_torsion_prefs=False,
                prune_rms_thresh=float(run_args["deep_prune_rms"]),
            )
            grid, mu, weights = sigma_potential_ensemble(
                mp["cosmos"],
                mp["energies_screen_hartree"],
                sigma_grid_e_per_A2=SIGMA_GRID,
            )
            weights_arr = np.asarray(weights, dtype=np.float64)
            profiles = np.asarray([_profile_on_grid(cs, SIGMA_GRID) for cs in mp["cosmos"]], dtype=np.float64)
            profile = np.sum(weights_arr[:, None] * profiles, axis=0)
            area = float(np.sum(weights_arr * np.asarray([cs.area for cs in mp["cosmos"]], dtype=np.float64)))
            volume = float(np.sum(weights_arr * np.asarray([cs.volume for cs in mp["cosmos"]], dtype=np.float64)))
            gxtb_energy = float(np.min(mp["energies_gxtb_hartree"]))
            orca_energy = float(np.sum(weights_arr * np.asarray([cs.total_energy_hartree for cs in mp["cosmos"]], dtype=np.float64)))
            dielectric_energy = float(np.sum(weights_arr * np.asarray([cs.dielectric_energy_hartree for cs in mp["cosmos"]], dtype=np.float64)))
            n_kept = int(mp["n_kept"])
            n_optimized = int(mp["n_optimized"])
            n_conformers_generated = int(mp["n_conformers_generated"])
            orcacosmo_paths = [str(p) for p in mp["orcacosmo_paths"]]
            mode = "deep"
        else:
            # The single-conformer ORCA path is parsed by this worker after the
            # tiered run returns, so cleanup has to live here, after the cache
            # NPZ is committed.  Otherwise macOS temp fills with multi-GB
            # tiered-cosmors-* directories during long queues.
            single_workdir = Path(tempfile.mkdtemp(prefix=f"tiered-cosmors-{inchikey[:8]}-"))
            workdirs_to_cleanup.append(single_workdir)
            sp = tiered_gxtb_orca_cosmors_from_smiles(
                smiles,
                seed=int(run_args["seed"]),
                xtb_path=xtb_path,
                orca_path=orca_path,
                method=str(run_args["method"]),
                basis=str(run_args["basis"]),
                solvent=str(run_args["solvent"]),
                charge=charge,
                uhf=uhf,
                n_cores=int(run_args["orca_cores"]),
                acc=float(run_args["acc"]),
                workdir=single_workdir,
                keep_workdir=True,
            )
            cs = cosmosegments_from_orcacosmo(sp["orcacosmo_path"])
            grid, mu = sigma_potential(cs, sigma_grid_e_per_A2=SIGMA_GRID)
            profile = _profile_on_grid(cs, SIGMA_GRID)
            weights_arr = np.asarray([1.0], dtype=np.float64)
            area = float(cs.area)
            volume = float(cs.volume)
            gxtb_energy = float(sp["gxtb_energy_hartree"])
            orca_energy = float(cs.total_energy_hartree)
            dielectric_energy = float(cs.dielectric_energy_hartree)
            n_kept = 1
            n_optimized = 1
            n_conformers_generated = 1
            orcacosmo_paths = [str(sp["orcacosmo_path"])]
            mode = "single"

        wall_s = time.perf_counter() - start
        tmp_path = cache_path.with_name(f".{cache_path.stem}.tmp.npz")
        np.savez_compressed(
            tmp_path,
            row_index=np.asarray(row_index, dtype=np.int64),
            canonical_smiles=np.asarray(canonical_smiles),
            smiles=np.asarray(smiles),
            inchikey=np.asarray(inchikey),
            method=np.asarray(METHOD),
            mode=np.asarray(mode),
            reason=np.asarray(reason),
            is_complex=np.asarray(bool(complex_flag)),
            charge=np.asarray(charge, dtype=np.int64),
            uhf=np.asarray(uhf, dtype=np.int64),
            solvent=np.asarray(str(run_args["solvent"])),
            sigma_grid_e_per_A2=np.asarray(grid, dtype=np.float64),
            mu_J_per_mol=np.asarray(mu, dtype=np.float64),
            profile_area_A2=np.asarray(profile, dtype=np.float64),
            area_A2=np.asarray(area, dtype=np.float64),
            volume_A3=np.asarray(volume, dtype=np.float64),
            gxtb_energy_hartree=np.asarray(gxtb_energy, dtype=np.float64),
            orca_total_energy_hartree=np.asarray(orca_energy, dtype=np.float64),
            orca_dielectric_energy_hartree=np.asarray(dielectric_energy, dtype=np.float64),
            n_kept=np.asarray(n_kept, dtype=np.int64),
            n_optimized=np.asarray(n_optimized, dtype=np.int64),
            n_conformers_generated=np.asarray(n_conformers_generated, dtype=np.int64),
            weights=np.asarray(weights_arr, dtype=np.float64),
            orcacosmo_paths=np.asarray(orcacosmo_paths, dtype=str),
            wall_s=np.asarray(wall_s, dtype=np.float64),
        )
        tmp_path.replace(cache_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "inchikey": inchikey,
                    "canonical_smiles": canonical_smiles,
                    "source_smiles_examples": [smiles],
                    "method": METHOD,
                    "run_args": run_args,
                    "charge": charge,
                    "uhf": uhf,
                    "cache_file": cache_path.name,
                    "mode": mode,
                    "reason": reason,
                    "n_kept": n_kept,
                    "orcacosmo_paths": orcacosmo_paths,
                    "updated_at_unix": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if error_path.exists():
            error_path.unlink()
        for workdir in workdirs_to_cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
        return {
            "row_index": row_index,
            "ok": True,
            "cached": False,
            "cache_path": str(cache_path),
            "wall_s": wall_s,
            "mode": mode,
            "reason": reason,
            "inchikey": inchikey,
            "charge": charge,
            "uhf": uhf,
        }
    except Exception as exc:  # noqa: BLE001
        wall_s = time.perf_counter() - start
        payload = {
            "row_index": row_index,
            "canonical_smiles": canonical_smiles,
            "smiles": smiles,
            "inchikey": inchikey,
            "run_args": run_args,
            "charge": charge,
            "uhf": uhf,
            "wall_s": wall_s,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        error_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        for workdir in workdirs_to_cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
        return {
            "row_index": row_index,
            "ok": False,
            "cached": False,
            "cache_path": "",
            "error_path": str(error_path),
            "wall_s": wall_s,
            "error": repr(exc),
            "mode": "",
            "reason": "",
            "inchikey": inchikey,
            "charge": charge,
            "uhf": uhf,
        }


def _assemble(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    results: list[dict[str, Any]],
    cache_dir: Path,
    out_path: Path,
    status_path: Path,
    *,
    subset: str,
    run_args: dict[str, Any],
) -> None:
    n = len(df)
    b = SIGMA_GRID.size
    mu = np.full((n, b), np.nan, dtype=np.float64)
    profile = np.full((n, b), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    cached = np.zeros(n, dtype=bool)
    wall_s = np.full(n, np.nan, dtype=np.float64)
    area = np.full(n, np.nan, dtype=np.float64)
    volume = np.full(n, np.nan, dtype=np.float64)
    gxtb_e = np.full(n, np.nan, dtype=np.float64)
    orca_e = np.full(n, np.nan, dtype=np.float64)
    diel_e = np.full(n, np.nan, dtype=np.float64)
    n_kept = np.full(n, -1, dtype=np.int64)
    n_optimized = np.full(n, -1, dtype=np.int64)
    n_conformers = np.full(n, -1, dtype=np.int64)
    mode = np.full(n, "", dtype=object)
    reason = np.full(n, "", dtype=object)
    inchikey = np.full(n, "", dtype=object)
    charge = np.full(n, 0, dtype=np.int64)
    uhf = np.full(n, 0, dtype=np.int64)
    error = np.full(n, "", dtype=object)
    cache_path_arr = np.full(n, "", dtype=object)

    for result in results:
        row_index = int(result["row_index"])
        wall_s[row_index] = float(result.get("wall_s", np.nan))
        cached[row_index] = bool(result.get("cached", False))
        mode[row_index] = str(result.get("mode", ""))
        reason[row_index] = str(result.get("reason", ""))
        inchikey[row_index] = str(result.get("inchikey", ""))
        charge[row_index] = int(result.get("charge", 0))
        uhf[row_index] = int(result.get("uhf", 0))
        if not result.get("ok", False):
            error[row_index] = str(result.get("error", "failed"))
            continue
        path = Path(str(result["cache_path"]))
        cache_path_arr[row_index] = str(path)
        try:
            d = _load_npz(path)
            mu[row_index] = np.asarray(d["mu_J_per_mol"], dtype=np.float64)
            profile[row_index] = np.asarray(d["profile_area_A2"], dtype=np.float64)
            area[row_index] = float(_scalar(d, "area_A2"))
            volume[row_index] = float(_scalar(d, "volume_A3"))
            gxtb_e[row_index] = float(_scalar(d, "gxtb_energy_hartree"))
            orca_e[row_index] = float(_scalar(d, "orca_total_energy_hartree"))
            diel_e[row_index] = float(_scalar(d, "orca_dielectric_energy_hartree"))
            n_kept[row_index] = int(_scalar(d, "n_kept", -1))
            n_optimized[row_index] = int(_scalar(d, "n_optimized", -1))
            n_conformers[row_index] = int(_scalar(d, "n_conformers_generated", -1))
            mode[row_index] = str(_scalar(d, "mode", mode[row_index]))
            reason[row_index] = str(_scalar(d, "reason", reason[row_index]))
            inchikey[row_index] = str(_scalar(d, "inchikey", inchikey[row_index]))
            charge[row_index] = int(_scalar(d, "charge", charge[row_index]))
            uhf[row_index] = int(_scalar(d, "uhf", uhf[row_index]))
            valid[row_index] = True
        except Exception as exc:  # noqa: BLE001
            error[row_index] = f"cache_load_failed:{exc!r}"

    canonical = df["canonical_smiles"].astype(str).to_numpy(dtype=object)
    smiles = df.apply(_row_smiles, axis=1).astype(str).to_numpy(dtype=object)
    target = pd.to_numeric(df["trusted_target_kJmol"], errors="coerce").to_numpy(dtype=np.float64)
    sample_weight = pd.to_numeric(df["sample_weight"], errors="coerce").to_numpy(dtype=np.float64)

    np.savez_compressed(
        out_path,
        method=np.asarray(METHOD),
        subset=np.asarray(subset),
        run_args_json=np.asarray(json.dumps(run_args, sort_keys=True)),
        cache_dir=np.asarray(str(cache_dir)),
        sigma_grid_e_per_A2=SIGMA_GRID,
        mu_J_per_mol=mu,
        profile_area_A2=profile,
        valid_mask=valid,
        selected_mask=selected_mask,
        cached_mask=cached,
        canonical_smiles=canonical,
        smiles=smiles,
        inchikey=inchikey,
        target_source=df["target_source"].astype(str).to_numpy(dtype=object),
        trusted_target_kJmol=target,
        sample_weight=sample_weight,
        charge=charge,
        uhf=uhf,
        area_A2=area,
        volume_A3=volume,
        gxtb_energy_hartree=gxtb_e,
        orca_total_energy_hartree=orca_e,
        orca_dielectric_energy_hartree=diel_e,
        mode=mode,
        reason=reason,
        n_kept=n_kept,
        n_optimized=n_optimized,
        n_conformers_generated=n_conformers,
        wall_s=wall_s,
        cache_path=cache_path_arr,
        error=error,
    )

    status = pd.DataFrame(
        {
            "row_index": np.arange(n, dtype=np.int64),
            "selected": selected_mask,
            "valid": valid,
            "cached": cached,
            "canonical_smiles": canonical,
            "smiles": smiles,
            "inchikey": inchikey,
            "target_source": df["target_source"].astype(str),
            "trusted_target_kJmol": target,
            "charge": charge,
            "uhf": uhf,
            "mode": mode,
            "reason": reason,
            "n_kept": n_kept,
            "n_optimized": n_optimized,
            "n_conformers_generated": n_conformers,
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--xtb-path", type=Path, default=DEFAULT_XTB)
    parser.add_argument("--orca-path", type=Path, default=DEFAULT_ORCA)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--orca-cores", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-mode", choices=["auto", "single", "deep"], default="auto")
    parser.add_argument("--method", default="BP86")
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--solvent", default="water")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--deep-n-conformers", type=int, default=20)
    parser.add_argument("--deep-n-keep", type=int, default=5)
    parser.add_argument("--deep-prune-rms", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if not args.xtb_path.exists():
        raise FileNotFoundError(f"xtb binary not found: {args.xtb_path}")
    if not args.orca_path.exists():
        raise FileNotFoundError(f"ORCA binary not found: {args.orca_path}")

    df = pd.read_csv(args.union_csv)
    selected_mask = _select_rows(df, args.subset)
    selected_indices = np.flatnonzero(selected_mask)
    if args.limit and args.limit > 0:
        selected_indices = selected_indices[: int(args.limit)]
        selected_mask = np.zeros(len(df), dtype=bool)
        selected_mask[selected_indices] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
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

    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_index in selected_indices:
        row = df.iloc[int(row_index)]
        smiles = _row_smiles(row)
        identity = _mol_identity(smiles, str(row["canonical_smiles"]))
        if not smiles or int(identity["n_fragments"]) != 1:
            skipped.append(
                {
                    "row_index": int(row_index),
                    "ok": False,
                    "cached": False,
                    "wall_s": math.nan,
                    "error": f"missing_or_non_single_fragment:n_fragments={identity['n_fragments']}",
                    "inchikey": str(identity["inchikey"]),
                    "charge": int(identity["formal_charge"]),
                    "uhf": int(identity["uhf"]),
                    "mode": "",
                    "reason": "",
                }
            )
            continue
        tasks.append(
            {
                "row_index": int(row_index),
                "canonical_smiles": str(row["canonical_smiles"]),
                "smiles": smiles,
                "inchikey": str(identity["inchikey"]),
                "charge": int(identity["formal_charge"]),
                "uhf": int(identity["uhf"]),
                "xtb_path": str(args.xtb_path),
                "orca_path": str(args.orca_path),
                "cache_dir": str(args.cache_dir),
                "force": bool(args.force),
                "run_args": run_args,
            }
        )

    print(
        f"[deltaHvapv2-orca-sigma] subset={args.subset} selected={len(selected_indices)} "
        f"tasks={len(tasks)} workers={args.workers} orca_cores={args.orca_cores} "
        f"cache={args.cache_dir}",
        flush=True,
    )

    start = time.perf_counter()
    results: list[dict[str, Any]] = list(skipped)
    if tasks:
        if args.workers <= 1:
            for i, task in enumerate(tasks, 1):
                results.append(_worker(task))
                if args.progress_every and (i % args.progress_every == 0 or i == len(tasks)):
                    ok = sum(1 for r in results if r.get("ok", False))
                    failed = sum(1 for r in results if not r.get("ok", False))
                    print(
                        f"[deltaHvapv2-orca-sigma] done={i}/{len(tasks)} ok={ok} "
                        f"failed={failed} elapsed={time.perf_counter() - start:.1f}s",
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
                            f"[deltaHvapv2-orca-sigma] done={i}/{len(futures)} ok={ok} "
                            f"failed={failed} elapsed={time.perf_counter() - start:.1f}s",
                            flush=True,
                        )

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

    ok = sum(1 for r in results if r.get("ok", False))
    failed = sum(1 for r in results if not r.get("ok", False))
    print(
        f"[deltaHvapv2-orca-sigma] wrote {args.out} and {status_path} "
        f"ok={ok} failed={failed} elapsed={time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
