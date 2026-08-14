#!/usr/bin/env python3
"""Train/check sublimation enthalpy as calcphyschemprop target #30.

The Ramprasad/Khazana data bundle contains the 845-row sublimation training
set with SMILES, RDKit descriptors, and DFT sublimation enthalpies.  This
script benchmarks direct tabular models on:

  rdkit
  rdkit + calcphyschemprop cascade predictions
  rdkit + calcphyschemprop cascade predictions + CHAOS sigma potential

It follows the same fold-local cleaning rule used in the Hvap benchmarks:
remove non-finite/constant columns and apply arcsinh(x / 100) to large-scale
features.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from train_hvap_direct_tabular import (
    apply_fold_prep,
    fit_fold_prep,
    iter_splits,
    make_model,
    metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data/sublimation/khazana/Sublimaion_enthalpy_Yifan/845dataset_rdkit.csv"
DEFAULT_CASCADE_WIDE = REPO_ROOT / "benchmarks/sublimation_ramprasad/calcphys_cascade_845/physchem_cascade_predictions_wide.csv"
DEFAULT_CHAOS = REPO_ROOT / "data/chaos_25a_mu_matrix.npz"
DEFAULT_OFFICIAL_MODEL = REPO_ROOT / "data/sublimation/845model.pkl"
DEFAULT_OFFICIAL_SCALER = REPO_ROOT / "data/sublimation/845scaler.save"
DEFAULT_OFFICIAL_ZERO = REPO_ROOT / "data/sublimation/all_zero_features.txt"

TARGET_COL = "sublimation_enthalpy_kj/mol"
ID_COLS = {"index", "Unnamed: 0", "SMILES", TARGET_COL}


def parse_csv_arg(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_smiles(smiles: object, *, isomeric: bool = True) -> str:
    if pd.isna(smiles):
        return ""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def is_single_valid_fragment(smiles: object) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None and len(Chem.GetMolFrags(mol)) == 1


def load_cascade_module(path: Path, model_root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("osmo_physchem_cascade", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import cascade module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUTPUT_DIR = model_root
    mod.CURRENT_VERSION = model_root.name
    return mod


def ensure_cascade_predictions(
    df: pd.DataFrame,
    cascade_wide: Path,
    cascade_module: Path,
    model_root: Path,
) -> pd.DataFrame:
    if cascade_wide.exists():
        return pd.read_csv(cascade_wide, low_memory=False)

    print(f"[cascade] {cascade_wide} missing; running calcphyschemprop inference", flush=True)
    mod = load_cascade_module(cascade_module, model_root)
    smiles = df["SMILES"].astype(str).tolist()
    pred_map = mod.run_cascade_inference(smiles, up_to_target=None)
    target_names = ["MW", "MR"] + [target for target, *_ in mod.CASCADE_ORDER]
    rows: list[dict[str, object]] = []
    for i, row in df.iterrows():
        smi = str(row["SMILES"])
        preds = pred_map.get(smi, {})
        out = {
            "row_index": row.get("index", i),
            "smiles": smi,
            "canonical_smiles": canonical_smiles(smi, isomeric=True),
            "canonical_smiles_nostereo": canonical_smiles(smi, isomeric=False),
        }
        for target in target_names:
            out[f"pred_{target}"] = preds.get(target, np.nan)
        rows.append(out)
    cascade_wide.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cascade_wide, index=False)
    return pd.DataFrame(rows)


def load_chaos_sigma_index(path: Path) -> tuple[dict[str, int], dict[str, int], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    smiles = np.asarray(z["canonical_smiles"]).astype(str)
    mu = np.asarray(z["mu_J_per_mol"], dtype=np.float32)
    grid = np.asarray(z["sigma_grid_e_per_A2"], dtype=np.float64)
    iso_index = {str(smi): i for i, smi in enumerate(smiles)}
    no_stereo_index: dict[str, int] = {}
    for i, smi in enumerate(smiles):
        key = canonical_smiles(smi, isomeric=False)
        if key and key not in no_stereo_index:
            no_stereo_index[key] = i
    return iso_index, no_stereo_index, mu, grid


def attach_sigma(df: pd.DataFrame, chaos_matrix: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    iso_index, no_stereo_index, _mu, _grid = load_chaos_sigma_index(chaos_matrix)
    rows: list[int] = []
    modes: list[str] = []
    for smi in df["SMILES"].astype(str):
        iso = canonical_smiles(smi, isomeric=True)
        no_stereo = canonical_smiles(smi, isomeric=False)
        if iso in iso_index:
            rows.append(int(iso_index[iso]))
            modes.append("isomeric")
        elif no_stereo in no_stereo_index:
            rows.append(int(no_stereo_index[no_stereo]))
            modes.append("no_stereo")
        else:
            rows.append(-1)
            modes.append("missing")
    out = df.copy()
    out["_sigma_row"] = np.asarray(rows, dtype=np.int64)
    out["_sigma_match"] = modes
    matched = out["_sigma_row"].to_numpy(dtype=np.int64) >= 0
    return out, {
        "chaos_matrix": str(chaos_matrix),
        "matched": int(matched.sum()),
        "missing": int((~matched).sum()),
        "mode_counts": {str(k): int(v) for k, v in pd.Series(modes).value_counts().items()},
    }


def source_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


def official_gpr_apparent_metrics(df: pd.DataFrame, model_path: Path, scaler_path: Path, zero_path: Path) -> dict[str, Any] | None:
    if not (model_path.exists() and scaler_path.exists() and zero_path.exists()):
        return None
    feature_cols = source_feature_columns(df)
    zero_idx = [int(x) for x in zero_path.read_text().split()]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    if x.shape[1] <= max(zero_idx):
        return None
    x = np.delete(x, zero_idx, axis=1)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    pred, std = model.predict(scaler.transform(x), return_std=True)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    return {
        "feature_set": "official_rdkit193",
        "model": "official_gpr",
        "protocol": "apparent_train",
        "n": int(len(y)),
        "splits": 1,
        "raw_features": int(x.shape[1]),
        "uncertainty_mean": float(np.mean(std)),
        **metrics(y, np.asarray(pred, dtype=np.float64)),
    }


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str,
    cascade_wide: pd.DataFrame,
    chaos_matrix: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, Any]]:
    parts = set(parse_csv_arg(feature_set.replace("+", ",")))
    if feature_set == "all":
        parts = {"rdkit", "calcphys", "sigma"}

    work = df.copy()
    blocks: list[np.ndarray] = []
    names: list[str] = []
    meta: dict[str, Any] = {"feature_set": feature_set, "blocks": []}

    if "calcphys" in parts:
        pred_cols = [c for c in cascade_wide.columns if c.startswith("pred_")]
        calc = cascade_wide[["row_index", *pred_cols]].copy()
        calc["row_index"] = pd.to_numeric(calc["row_index"], errors="coerce")
        work = work.merge(calc, left_on="index", right_on="row_index", how="left")
        finite = np.all(np.isfinite(work[pred_cols].to_numpy(dtype=np.float64)), axis=1)
        before = len(work)
        work = work[finite].reset_index(drop=True)
        meta["calcphys_rows_kept"] = int(len(work))
        meta["calcphys_rows_dropped"] = int(before - len(work))

    if "sigma" in parts:
        work, sigma_report = attach_sigma(work, chaos_matrix)
        before = len(work)
        work = work[work["_sigma_row"].to_numpy(dtype=np.int64) >= 0].reset_index(drop=True)
        meta["sigma_report"] = sigma_report
        meta["sigma_rows_kept"] = int(len(work))
        meta["sigma_rows_dropped"] = int(before - len(work))

    if "rdkit" in parts:
        rdkit_cols = source_feature_columns(work)
        # Do not accidentally include merged prediction columns as RDKit source.
        rdkit_cols = [c for c in rdkit_cols if not c.startswith("pred_") and c != "row_index"]
        x = work[rdkit_cols].to_numpy(dtype=np.float32)
        blocks.append(x)
        names.extend([f"rdkit_{c}" for c in rdkit_cols])
        meta["blocks"].append({"name": "rdkit_source", "features": len(rdkit_cols)})

    if "calcphys" in parts:
        pred_cols = [c for c in work.columns if c.startswith("pred_")]
        x = work[pred_cols].to_numpy(dtype=np.float32)
        blocks.append(x)
        names.extend(pred_cols)
        meta["blocks"].append({"name": "calcphyschemprop_cascade", "features": len(pred_cols)})

    if "sigma" in parts:
        _iso, _no_stereo, mu, grid = load_chaos_sigma_index(chaos_matrix)
        rows = work["_sigma_row"].to_numpy(dtype=np.int64)
        x = mu[rows].astype(np.float32)
        blocks.append(x)
        sigma_names = [f"chaos_mu_{i:02d}" for i in range(x.shape[1])]
        names.extend(sigma_names)
        meta["blocks"].append(
            {"name": "chaos_sigma_mu", "features": int(x.shape[1]), "grid_min": float(grid[0]), "grid_max": float(grid[-1])}
        )

    if not blocks:
        raise ValueError(f"feature_set {feature_set!r} produced no features")
    x_all = np.concatenate(blocks, axis=1).astype(np.float32)
    y = work[TARGET_COL].to_numpy(dtype=np.float64)
    finite_y = np.isfinite(y)
    work = work[finite_y].reset_index(drop=True)
    x_all = x_all[finite_y]
    meta["rows"] = int(len(work))
    meta["raw_features"] = int(x_all.shape[1])
    return work, x_all, names, meta


def run_combo(
    x_raw: np.ndarray,
    names: list[str],
    y: np.ndarray,
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    oof_sum = np.zeros_like(y, dtype=np.float64)
    oof_count = np.zeros_like(y, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []

    for split_id, tr, va in iter_splits(len(y), args.protocol, args.folds, args.repeats, args.seed):
        t0 = time.time()
        x_tr, prep, prep_meta = fit_fold_prep(x_raw[tr], args.arcsinh_threshold, args.min_std)
        x_va = apply_fold_prep(x_raw[va], prep)
        model = make_model(model_name, args, seed=args.seed + split_id)
        model.fit(x_tr, y[tr])
        pred = np.asarray(model.predict(x_va), dtype=np.float64)
        oof_sum[va] += pred
        oof_count[va] += 1.0
        row = {
            "feature_set": feature_set,
            "model": model_name,
            "split_id": int(split_id),
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
    safe_name = feature_set.replace("+", "_")
    pd.DataFrame(
        {
            "index": df["index"].to_numpy(dtype=int),
            "SMILES": df["SMILES"].astype(str).to_numpy(),
            "y_true_deltaHsub_kJmol": y,
            "y_pred_deltaHsub_kJmol": y_pred,
            "observed": observed,
            "feature_set": feature_set,
            "model": model_name,
        }
    ).to_csv(out_dir / f"predictions_{safe_name}_{model_name}_{args.protocol}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"split_metrics_{safe_name}_{model_name}_{args.protocol}.csv", index=False)
    return overall


def train_final_artifact(
    x_raw: np.ndarray,
    names: list[str],
    y: np.ndarray,
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Path:
    x_clean, prep, prep_meta = fit_fold_prep(x_raw, args.arcsinh_threshold, args.min_std)
    model = make_model(model_name, args, seed=args.seed)
    model.fit(x_clean, y)
    artifact = {
        "target": "deltaHsub",
        "target_unit": "kJ/mol",
        "feature_set": feature_set,
        "model_name": model_name,
        "feature_names": names,
        "prep": {
            "keep_mask": prep.keep_mask.astype(bool),
            "arcsinh_mask": prep.arcsinh_mask.astype(bool),
            "threshold": float(prep.threshold),
            "meta": prep_meta,
        },
        "model": model,
        "training_rows": int(len(y)),
        "training_index": df["index"].to_numpy(dtype=int),
    }
    out_path = out_dir / "deltaHsub_target30_model.joblib"
    joblib.dump(artifact, out_path)
    return out_path


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cascade-wide", type=Path, default=DEFAULT_CASCADE_WIDE)
    parser.add_argument("--cascade-module", type=Path, default=Path("/Users/guillaume-osmo/Github/osmo/src/runway/physchemprops/train_all_25_models_v34_xgboost.py"))
    parser.add_argument("--model-root", type=Path, default=Path("/Users/guillaume-osmo/Github/osmo/src/runway/physchemprops/models"))
    parser.add_argument("--chaos-matrix", type=Path, default=DEFAULT_CHAOS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-sets", default="rdkit,rdkit+calcphys,rdkit+calcphys+sigma")
    parser.add_argument("--models", default="xgb,rf")
    parser.add_argument("--protocol", choices=["kfold", "autovap_random60"], default="kfold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--single-fragment-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--min-std", type=float, default=1.0e-12)
    parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--artifact-feature-set",
        default="rdkit+calcphys",
        help="preferred feature set for the saved target #30 artifact; use 'best' for lowest CV MAE",
    )
    parser.add_argument("--artifact-model", default="xgb")

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
    df_all = pd.read_csv(args.data, low_memory=False)
    before = len(df_all)
    if args.single_fragment_only:
        mask = np.asarray([is_single_valid_fragment(s) for s in df_all["SMILES"].astype(str)], dtype=bool)
        df = df_all[mask].reset_index(drop=True)
    else:
        df = df_all.copy()
    print(f"[data] rows={len(df)}/{before} single_fragment_only={args.single_fragment_only}", flush=True)

    cascade = ensure_cascade_predictions(df_all, args.cascade_wide, args.cascade_module, args.model_root)
    feature_sets = parse_csv_arg(args.feature_sets)
    models = parse_csv_arg(args.models)

    summaries: list[dict[str, Any]] = []
    official = official_gpr_apparent_metrics(df_all, DEFAULT_OFFICIAL_MODEL, DEFAULT_OFFICIAL_SCALER, DEFAULT_OFFICIAL_ZERO)
    if official is not None:
        summaries.append(official)

    feature_meta: dict[str, Any] = {}
    built: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]] = {}
    for feature_set in feature_sets:
        work, x, names, meta = build_feature_matrix(df, feature_set, cascade, args.chaos_matrix)
        y = work[TARGET_COL].to_numpy(dtype=np.float64)
        feature_meta[feature_set] = meta
        print(f"[features] {feature_set}: rows={len(work)} X={x.shape}", flush=True)
        for model_name in models:
            summaries.append(run_combo(x, names, y, work, feature_set, model_name, args, args.out_dir))
            built[(feature_set, model_name)] = (work, x, names, y)

    summary = pd.DataFrame(summaries).sort_values(["protocol", "mae", "rmse"], ascending=[True, True, True])
    summary_path = args.out_dir / "sublimation_target30_summary.csv"
    summary.to_csv(summary_path, index=False)

    artifact_path = None
    cv_summary = summary[summary["protocol"].eq(args.protocol)].copy()
    if args.save_best_model and not cv_summary.empty:
        preferred = (str(args.artifact_feature_set), str(args.artifact_model))
        if str(args.artifact_feature_set).lower() == "best" or preferred not in built:
            best = cv_summary.sort_values(["mae", "rmse"], ascending=[True, True]).iloc[0]
            key = (str(best["feature_set"]), str(best["model"]))
        else:
            key = preferred
        if key in built:
            work, x, names, y = built[key]
            artifact_path = train_final_artifact(x, names, y, work, key[0], key[1], args, args.out_dir)
            print(f"[artifact] wrote {artifact_path}", flush=True)

    report = {
        "args": vars(args)
        | {
            "data": str(args.data),
            "cascade_wide": str(args.cascade_wide),
            "chaos_matrix": str(args.chaos_matrix),
            "out_dir": str(args.out_dir),
        },
        "dataset": {
            "rows_source": int(before),
            "rows_modelled": int(len(df)),
            "target": "deltaHsub",
            "target_unit": "kJ/mol",
            "target_mean": float(df[TARGET_COL].mean()),
            "target_std": float(df[TARGET_COL].std()),
            "single_fragment_only": bool(args.single_fragment_only),
        },
        "feature_sets": feature_meta,
        "summary_csv": str(summary_path),
        "best_model_artifact": str(artifact_path) if artifact_path else None,
    }
    report_path = args.out_dir / "run_report.json"
    report_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
