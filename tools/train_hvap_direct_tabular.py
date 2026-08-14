#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


AUTOVAP_EXCLUDE = {"CAS", "dvap", "num", "External", "SMILES", "Key", "Family", "VOC"}


@dataclass
class FoldPrep:
    keep_mask: np.ndarray
    arcsinh_mask: np.ndarray
    threshold: float


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _as_float_matrix(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def load_or_compute_chemeleon_fps(cache_path: Path, smiles: Sequence[str], batch_size: int) -> np.ndarray:
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_smiles = [str(s) for s in z["smiles"].tolist()]
        if cached_smiles == list(smiles):
            return _as_float_matrix(z["fingerprints"])

    try:
        from chemeleon_smd import fingerprint
    except Exception as exc:
        raise ImportError(
            "ChemeleonSMD fingerprints requested but `chemeleon_smd` is not importable. "
            "Run with PYTHONPATH including /Users/guillaume-osmo/Github/ChemeleonSMD."
        ) from exc

    chunks: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(smiles), batch_size):
        stop = min(start + batch_size, len(smiles))
        fps = fingerprint(smiles[start:stop])
        chunks.append(_as_float_matrix(np.asarray(fps)))
        print(f"[chemeleon] fingerprints {stop}/{len(smiles)} seconds={time.time() - t0:.1f}", flush=True)
    x = np.concatenate(chunks, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, fingerprints=x, smiles=np.asarray(smiles, dtype=object))
    return x


def load_autovap_data(path: Path, max_mols: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["dvap"].notna()].reset_index(drop=True)
    if max_mols is not None and max_mols < len(df):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(np.arange(len(df)), size=max_mols, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
    return df


def _canonical_smiles(smiles: str, *, isomeric: bool) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def _feature_parts(mode: str) -> set[str]:
    if mode == "all":
        return {"rdkit", "sigma", "chemeleon"}
    aliases = {"autovap": "rdkit", "chaos": "sigma", "chaos3d": "sigma", "smd": "chemeleon"}
    return {aliases.get(part, part) for part in mode.split("_") if part}


def load_chaos_sigma_index(path: Path) -> tuple[dict[str, int], dict[str, int], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    smiles = np.asarray(z["canonical_smiles"])
    mu = np.asarray(z["mu_J_per_mol"], dtype=np.float32)
    grid = np.asarray(z["sigma_grid_e_per_A2"], dtype=np.float64)
    iso_index = {str(smi): i for i, smi in enumerate(smiles)}
    no_stereo_index: dict[str, int] = {}
    for i, smi in enumerate(smiles):
        key = _canonical_smiles(str(smi), isomeric=False)
        if key and key not in no_stereo_index:
            no_stereo_index[key] = i
    return iso_index, no_stereo_index, mu, grid


def attach_chaos_sigma_matches(df: pd.DataFrame, chaos_matrix: Path) -> tuple[pd.DataFrame, dict]:
    iso_index, no_stereo_index, _mu, _grid = load_chaos_sigma_index(chaos_matrix)
    rows: list[int] = []
    modes: list[str] = []
    for smiles in df["SMILES"].astype(str):
        canon_iso = _canonical_smiles(smiles, isomeric=True)
        canon_no_stereo = _canonical_smiles(smiles, isomeric=False)
        if canon_iso in iso_index:
            rows.append(int(iso_index[canon_iso]))
            modes.append("isomeric")
        elif canon_no_stereo in no_stereo_index:
            rows.append(int(no_stereo_index[canon_no_stereo]))
            modes.append("no_stereo")
        else:
            rows.append(-1)
            modes.append("none")
    out = df.copy()
    out["_chaos_row"] = np.asarray(rows, dtype=np.int64)
    out["_chaos_match_mode"] = modes
    matched = out["_chaos_row"].to_numpy(dtype=np.int64) >= 0
    report = {
        "chaos_matrix": str(chaos_matrix),
        "input_rows": int(len(df)),
        "matched_rows": int(matched.sum()),
        "missing_rows": int((~matched).sum()),
        "isomeric_matches": int(np.sum(np.asarray(modes) == "isomeric")),
        "no_stereo_matches": int(np.sum(np.asarray(modes) == "no_stereo")),
    }
    return out, report


def attach_filled_sigma_matches(df: pd.DataFrame, filled_sigma: Path) -> tuple[pd.DataFrame, dict]:
    z = np.load(filled_sigma, allow_pickle=True)
    valid = np.asarray(z["valid_mask"], dtype=bool)
    if valid.shape[0] != len(df):
        raise ValueError(f"filled sigma row count {valid.shape[0]} does not match data rows {len(df)}")
    out = df.copy()
    out["_filled_sigma_row"] = np.arange(len(df), dtype=np.int64)
    out["_filled_sigma_valid"] = valid
    source = np.asarray(z["source"]).astype(str) if "source" in z.files else np.full(len(df), "filled", dtype=str)
    report = {
        "filled_sigma": str(filled_sigma),
        "input_rows": int(len(df)),
        "matched_rows": int(valid.sum()),
        "missing_rows": int((~valid).sum()),
        "source_counts": {str(k): int(v) for k, v in pd.Series(source[valid]).value_counts().items()},
    }
    return out, report


def load_features(
    df: pd.DataFrame,
    mode: str,
    cache_dir: Path,
    chemeleon_batch_size: int,
    chaos_matrix: Path,
    filled_sigma: Path | None,
) -> tuple[np.ndarray, list[str], dict]:
    parts = _feature_parts(mode)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    meta: dict = {"mode": mode, "blocks": []}

    if "rdkit" in parts:
        rdkit_names = [c for c in df.columns if c not in AUTOVAP_EXCLUDE]
        rdkit_names = [c for c in rdkit_names if not c.startswith("_")]
        x = df[rdkit_names].to_numpy(dtype=np.float32)
        blocks.append(x)
        names.extend(rdkit_names)
        meta["blocks"].append({"name": "autovap_rdkit", "features": len(rdkit_names)})

    if "sigma" in parts:
        if filled_sigma is not None:
            z = np.load(filled_sigma, allow_pickle=True)
            mu = np.asarray(z["mu_J_per_mol"], dtype=np.float32)
            grid = np.asarray(z["sigma_grid_e_per_A2"], dtype=np.float64)
            rows = df["_filled_sigma_row"].to_numpy(dtype=np.int64)
            sigma = mu[rows]
        elif "_chaos_row" in df.columns:
            _iso, _no_stereo, mu, grid = load_chaos_sigma_index(chaos_matrix)
            rows = df["_chaos_row"].to_numpy(dtype=np.int64)
            if np.any(rows < 0):
                raise ValueError("sigma features requested with unmatched CHAOS rows; filter first")
            sigma = mu[rows]
        else:
            raise ValueError("sigma features requested but CHAOS sigma matches were not attached")
        blocks.append(sigma)
        sigma_names = [f"chaos_sigma_mu_{i:02d}" for i in range(sigma.shape[1])]
        names.extend(sigma_names)
        meta["blocks"].append(
            {
                "name": "chaos_sigma_mu",
                "features": int(sigma.shape[1]),
                "grid_min": float(grid[0]),
                "grid_max": float(grid[-1]),
            }
        )

    if "chemeleon" in parts:
        smiles = df["SMILES"].astype(str).tolist()
        x = load_or_compute_chemeleon_fps(cache_dir / "autovap_chemeleon_smd_fingerprints.npz", smiles, chemeleon_batch_size)
        blocks.append(x)
        names.extend([f"chemeleon_smd_{i:04d}" for i in range(x.shape[1])])
        meta["blocks"].append({"name": "chemeleon_smd", "features": int(x.shape[1])})

    if not blocks:
        raise ValueError(f"feature mode {mode!r} produced no features")
    x_all = np.concatenate(blocks, axis=1).astype(np.float32)
    meta["raw_features"] = int(x_all.shape[1])
    return x_all, names, meta


def fit_fold_prep(x_train: np.ndarray, threshold: float, min_std: float) -> tuple[np.ndarray, FoldPrep, dict]:
    x64 = np.asarray(x_train, dtype=np.float64)
    finite = np.all(np.isfinite(x64), axis=0)
    x_keep = x64[:, finite]
    if x_keep.size:
        std = x_keep.std(axis=0)
        nonconstant = std > min_std
        x_keep = x_keep[:, nonconstant].copy()
        max_abs = np.max(np.abs(x_keep), axis=0)
        arcsinh = max_abs > threshold
        x_keep[:, arcsinh] = np.arcsinh(x_keep[:, arcsinh] / threshold)
    else:
        nonconstant = np.zeros((0,), dtype=bool)
        arcsinh = np.zeros((0,), dtype=bool)

    keep_mask = np.zeros_like(finite, dtype=bool)
    finite_idx = np.flatnonzero(finite)
    keep_mask[finite_idx[nonconstant]] = True
    prep = FoldPrep(keep_mask=keep_mask, arcsinh_mask=arcsinh.astype(bool), threshold=float(threshold))
    meta = {
        "input_features": int(x64.shape[1]),
        "dropped_nonfinite_train": int((~finite).sum()),
        "dropped_constant_train": int((finite.sum() - nonconstant.sum()) if finite.sum() else 0),
        "clean_features": int(keep_mask.sum()),
        "arcsinh_features": int(arcsinh.sum()),
    }
    return x_keep.astype(np.float32), prep, meta


def apply_fold_prep(x: np.ndarray, prep: FoldPrep) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)[:, prep.keep_mask].copy()
    x64[~np.isfinite(x64)] = 0.0
    if np.any(prep.arcsinh_mask):
        x64[:, prep.arcsinh_mask] = np.arcsinh(x64[:, prep.arcsinh_mask] / prep.threshold)
    return x64.astype(np.float32)


def make_model(name: str, args: argparse.Namespace, seed: int):
    if name == "rf_autovap_script":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=20,
            random_state=47,
            n_jobs=args.jobs,
        )
    if name == "rf_autovap_best":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            max_depth=20,
            bootstrap=False,
            random_state=seed,
            n_jobs=args.jobs,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=args.rf_trees,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features,
            max_depth=args.rf_max_depth,
            bootstrap=args.rf_bootstrap,
            random_state=seed,
            n_jobs=args.jobs,
        )
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=args.xgb_trees,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=args.jobs,
        )
    raise ValueError(f"unknown model {name!r}")


def iter_splits(n: int, protocol: str, folds: int, repeats: int, seed: int):
    if protocol == "autovap_random60":
        for iteration in range(repeats):
            tr, va = train_test_split(np.arange(n), test_size=0.60, random_state=iteration)
            yield iteration, np.asarray(tr, dtype=int), np.asarray(va, dtype=int)
        return
    if protocol == "kfold":
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(splitter.split(np.arange(n))):
            yield fold, tr.astype(int), va.astype(int)
        return
    raise ValueError(f"unknown protocol={protocol!r}")


def run_combo(
    x_raw: np.ndarray,
    names: list[str],
    y: np.ndarray,
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    oof_sum = np.zeros_like(y, dtype=np.float64)
    oof_count = np.zeros_like(y, dtype=np.float64)
    fold_rows: list[dict] = []
    importances = np.zeros(x_raw.shape[1], dtype=np.float64)
    importance_counts = np.zeros(x_raw.shape[1], dtype=np.float64)

    for split_id, tr, va in iter_splits(len(y), args.protocol, args.folds, args.repeats, args.seed):
        t0 = time.time()
        x_tr, prep, prep_meta = fit_fold_prep(x_raw[tr], args.arcsinh_threshold, args.min_std)
        x_va = apply_fold_prep(x_raw[va], prep)
        model = make_model(model_name, args, seed=args.seed + split_id)
        model.fit(x_tr, y[tr])
        pred = np.asarray(model.predict(x_va), dtype=np.float64)
        oof_sum[va] += pred
        oof_count[va] += 1.0

        if hasattr(model, "feature_importances_"):
            clean_idx = np.flatnonzero(prep.keep_mask)
            imp = np.asarray(model.feature_importances_, dtype=np.float64)
            importances[clean_idx] += np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
            importance_counts[clean_idx] += 1.0

        row = {
            "feature_set": feature_set,
            "model": model_name,
            "split_id": split_id,
            "protocol": args.protocol,
            "seconds": time.time() - t0,
            **metrics(y[va], pred),
            **prep_meta,
        }
        fold_rows.append(row)
        print(
            f"[{feature_set}/{model_name}] split={split_id} mae={row['mae']:.3f} "
            f"rmse={row['rmse']:.3f} r2={row['r2']:.4f} clean={row['clean_features']} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )

    observed = oof_count > 0
    y_pred = np.zeros_like(y, dtype=np.float64)
    y_pred[observed] = oof_sum[observed] / oof_count[observed]
    overall = {
        "feature_set": feature_set,
        "model": model_name,
        "protocol": args.protocol,
        "n": int(observed.sum()),
        "splits": int(len(fold_rows)),
        "raw_features": int(x_raw.shape[1]),
        **metrics(y[observed], y_pred[observed]),
    }

    pred_df = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=int),
            "SMILES": df["SMILES"].astype(str).to_numpy(),
            "y_true_dvap_kjmol": y,
            "y_pred_dvap_kjmol": y_pred,
            "observed": observed,
            "feature_set": feature_set,
            "model": model_name,
        }
    )
    pred_df.to_csv(out_dir / f"predictions_{feature_set}_{model_name}_{args.protocol}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"split_metrics_{feature_set}_{model_name}_{args.protocol}.csv", index=False)

    if np.any(importance_counts > 0):
        avg = np.divide(importances, np.maximum(importance_counts, 1.0))
        imp_df = pd.DataFrame({"feature": names, "importance": avg})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df.to_csv(out_dir / f"feature_importance_{feature_set}_{model_name}_{args.protocol}.csv", index=False)

    return overall


def parse_csv_arg(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-only RF/XGB benchmark for AutoVap ΔHvap/dvap.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/autovap/source/AutoVapOnline/Datasets/Database-Global.csv"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-sets", default="rdkit,rdkit_chemeleon")
    parser.add_argument("--models", default="rf_autovap_script,rf_autovap_best,xgb")
    parser.add_argument("--protocol", choices=["autovap_random60", "kfold"], default="autovap_random60")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-mols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/autovap"))
    parser.add_argument("--chaos-matrix", type=Path, default=Path("data/chaos_25a_mu_matrix.npz"))
    parser.add_argument("--filled-sigma", type=Path, default=None)
    parser.add_argument("--chemeleon-batch-size", type=int, default=128)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--min-std", type=float, default=1.0e-12)

    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=20)
    parser.add_argument("--rf-max-features", default="sqrt")
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-bootstrap", action="store_true")

    parser.add_argument("--xgb-trees", type=int, default=700)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.035)
    parser.add_argument("--xgb-subsample", type=float, default=0.90)
    parser.add_argument("--xgb-colsample", type=float, default=0.80)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_autovap_data(args.data, args.max_mols, args.seed)
    needs_sigma = any("sigma" in _feature_parts(fs) for fs in parse_csv_arg(args.feature_sets))
    sigma_match_report: dict | None = None
    if needs_sigma:
        if args.filled_sigma is not None:
            df, sigma_match_report = attach_filled_sigma_matches(df, args.filled_sigma)
            valid = df["_filled_sigma_valid"].to_numpy(dtype=bool)
            before = len(df)
            df = df[valid].reset_index(drop=True)
            print(
                f"[filled sigma] valid {len(df)}/{before} AutoVap rows "
                f"using {args.filled_sigma}",
                flush=True,
            )
        else:
            df, sigma_match_report = attach_chaos_sigma_matches(df, args.chaos_matrix)
            before = len(df)
            df = df[df["_chaos_row"].to_numpy(dtype=np.int64) >= 0].reset_index(drop=True)
            print(
                f"[chaos sigma] matched {len(df)}/{before} AutoVap rows "
                f"using {args.chaos_matrix}",
                flush=True,
            )
    y = df["dvap"].to_numpy(dtype=np.float64)
    feature_sets = parse_csv_arg(args.feature_sets)
    models = parse_csv_arg(args.models)

    summaries: list[dict] = []
    feature_meta: dict = {}
    for feature_set in feature_sets:
        x, names, meta = load_features(
            df,
            feature_set,
            args.cache_dir,
            args.chemeleon_batch_size,
            args.chaos_matrix,
            args.filled_sigma,
        )
        feature_meta[feature_set] = meta
        print(f"[features] {feature_set}: X={x.shape}", flush=True)
        for model_name in models:
            summaries.append(run_combo(x, names, y, df, feature_set, model_name, args, args.out_dir))

    summary = pd.DataFrame(summaries).sort_values(["r2", "mae"], ascending=[False, True])
    summary.to_csv(args.out_dir / "hvap_direct_summary.csv", index=False)
    report = {
        "args": vars(args)
        | {
            "data": str(args.data),
            "out_dir": str(args.out_dir),
            "cache_dir": str(args.cache_dir),
            "chaos_matrix": str(args.chaos_matrix),
            "filled_sigma": str(args.filled_sigma) if args.filled_sigma else None,
        },
        "dataset": {
            "rows": int(len(df)),
            "target": "dvap",
            "target_unit": "kJ/mol",
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
        },
        "feature_sets": feature_meta,
        "sigma_match_report": sigma_match_report,
    }
    (args.out_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
