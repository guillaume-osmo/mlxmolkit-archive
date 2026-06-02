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
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


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


def make_folds(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr.astype(int), va.astype(int)) for tr, va in splitter.split(np.arange(n))]


def load_dataset(data_dir: Path, max_mols: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "dipole_sigma_chemprop.csv")
    df = df[df["dipole_debye"].notna()].reset_index(drop=True)
    if max_mols is not None and max_mols < len(df):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(np.arange(len(df)), size=max_mols, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
    return df


def load_or_compute_chemeleon_fps(data_dir: Path, smiles: Sequence[str], batch_size: int) -> np.ndarray:
    cache_path = data_dir / "chemeleon_smd_fingerprints.npz"
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
    np.savez_compressed(cache_path, fingerprints=x, smiles=np.asarray(smiles, dtype=object))
    return x


def _mode_parts(mode: str) -> set[str]:
    if mode == "none":
        return set()
    if mode == "all":
        return {"osmordred", "sigma", "chemeleon"}
    parts = set(mode.split("_"))
    aliases = {
        "chaos": "sigma",
        "chaos3d": "sigma",
        "smd": "chemeleon",
    }
    return {aliases.get(part, part) for part in parts}


def load_feature_set(data_dir: Path, df: pd.DataFrame, mode: str, chemeleon_batch_size: int) -> tuple[np.ndarray, list[str], dict]:
    parts = _mode_parts(mode)
    source = df["feature_index"].to_numpy(dtype=np.int64) if "feature_index" in df.columns else df.index.to_numpy(dtype=np.int64)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    meta: dict = {"mode": mode, "blocks": []}

    if "osmordred" in parts:
        z = np.load(data_dir / "osmordred_features.npz", allow_pickle=True)
        x = _as_float_matrix(z["X"])[source]
        raw_names = [str(n) for n in z["names"].tolist()]
        keep = np.asarray(
            [
                ("calcphyschemprop" not in name.lower())
                and ("pred_debye" not in name.lower())
                and ("dipole_debye" not in name.lower())
                for name in raw_names
            ],
            dtype=bool,
        )
        blocks.append(x[:, keep])
        kept_names = [name for name, use in zip(raw_names, keep) if bool(use)]
        names.extend(kept_names)
        meta["blocks"].append({"name": "osmordred", "features": len(kept_names), "dropped_teacher_like": int((~keep).sum())})

    if "sigma" in parts:
        z = np.load(data_dir / "sigma_features.npz", allow_pickle=False)
        mu = _as_float_matrix(z["mu_J_per_mol"])[source]
        profile = _as_float_matrix(z["profile_area_A2"])[source]
        blocks.extend([mu, profile])
        mu_names = [f"sigma_mu_{i:02d}" for i in range(mu.shape[1])]
        profile_names = [f"sigma_profile_{i:02d}" for i in range(profile.shape[1])]
        names.extend(mu_names)
        names.extend(profile_names)
        meta["blocks"].append({"name": "sigma", "features": len(mu_names) + len(profile_names)})

    if "chemeleon" in parts:
        all_df = pd.read_csv(data_dir / "dipole_sigma_chemprop.csv")
        all_smiles = all_df["canonical_smiles"].astype(str).tolist()
        fps = load_or_compute_chemeleon_fps(data_dir, all_smiles, batch_size=chemeleon_batch_size)[source]
        blocks.append(fps)
        fp_names = [f"chemeleon_smd_{i:04d}" for i in range(fps.shape[1])]
        names.extend(fp_names)
        meta["blocks"].append({"name": "chemeleon_smd", "features": fps.shape[1]})

    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float32), [], meta
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
        keep_inside = nonconstant
        x_keep = x_keep[:, keep_inside].copy()
        max_abs = np.max(np.abs(x_keep), axis=0)
        arcsinh = max_abs > threshold
        x_keep[:, arcsinh] = np.arcsinh(x_keep[:, arcsinh] / threshold)
    else:
        keep_inside = np.zeros((0,), dtype=bool)
        arcsinh = np.zeros((0,), dtype=bool)

    keep_mask = np.zeros_like(finite, dtype=bool)
    finite_idx = np.flatnonzero(finite)
    keep_mask[finite_idx[keep_inside]] = True
    prep = FoldPrep(keep_mask=keep_mask, arcsinh_mask=arcsinh.astype(bool), threshold=float(threshold))
    meta = {
        "input_features": int(x64.shape[1]),
        "dropped_nonfinite_train": int((~finite).sum()),
        "dropped_constant_train": int((finite.sum() - keep_inside.sum()) if finite.sum() else 0),
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
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            max_features=args.rf_max_features,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            bootstrap=args.rf_bootstrap,
            random_state=seed,
            n_jobs=args.jobs,
        )
    if name == "rf_autovap":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=False,
            random_state=seed,
            n_jobs=args.jobs,
        )
    if name == "etr":
        return ExtraTreesRegressor(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            max_features=args.rf_max_features,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            bootstrap=False,
            random_state=seed,
            n_jobs=args.jobs,
        )
    raise ValueError(f"unknown model {name!r}")


def select_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict]:
    n_features = x_train.shape[1]
    if args.select == "none" or args.select_top_k <= 0 or args.select_top_k >= n_features:
        return np.arange(n_features, dtype=np.int64), {"select": "none", "selected_features": int(n_features)}

    top_k = min(int(args.select_top_k), n_features)
    if args.select == "model_importance":
        selector = clone(model)
        selector.fit(x_train, y_train)
        importance = np.asarray(getattr(selector, "feature_importances_", np.zeros(n_features)), dtype=np.float64)
    elif args.select == "permutation":
        tr, va = train_test_split(np.arange(len(y_train)), test_size=args.select_valid_size, random_state=seed)
        selector = clone(model)
        selector.fit(x_train[tr], y_train[tr])
        perm = permutation_importance(
            selector,
            x_train[va],
            y_train[va],
            n_repeats=args.select_permutation_repeats,
            random_state=seed,
            n_jobs=args.jobs,
            scoring="neg_mean_absolute_error",
        )
        importance = np.asarray(perm.importances_mean, dtype=np.float64)
    else:
        raise ValueError(f"unknown select mode {args.select!r}")

    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(importance)[::-1]
    if args.select_min_importance > 0:
        keep = order[importance[order] > args.select_min_importance]
        selected = keep[:top_k] if len(keep) else order[:top_k]
    else:
        selected = order[:top_k]
    selected = np.sort(selected.astype(np.int64))
    meta = {
        "select": args.select,
        "selected_features": int(len(selected)),
        "top_k": int(top_k),
        "importance_sum": float(importance[selected].sum()),
        "importance_max": float(importance.max()) if importance.size else 0.0,
    }
    return selected, meta


def compare_autovap(data_dir: Path, autovap_dir: Path | None, our_feature_names: Sequence[str]) -> dict:
    report: dict = {
        "our_dipole_rows": int(pd.read_csv(data_dir / "dipole_sigma_chemprop.csv")["dipole_debye"].notna().sum()),
        "our_feature_count_current_set": int(len(our_feature_names)),
    }
    if autovap_dir is None:
        return report
    global_path = autovap_dir / "Datasets" / "Database-Global.csv"
    results_path = autovap_dir / "Results" / "results.csv"
    if global_path.exists():
        df = pd.read_csv(global_path)
        exclude_cols = {"CAS", "dvap", "num", "External", "SMILES", "Key", "Family", "VOC"}
        av_features = [c for c in df.columns if c not in exclude_cols]
        our = set(our_feature_names)
        overlap = sorted(set(av_features).intersection(our))
        report.update(
            {
                "autovap_rows": int(len(df)),
                "autovap_target": "dvap",
                "autovap_feature_count": int(len(av_features)),
                "autovap_overlap_with_current_features": int(len(overlap)),
                "autovap_overlap_examples": overlap[:30],
            }
        )
    if results_path.exists():
        res = pd.read_csv(results_path)
        for col in ["MAE", "RMSE", "rsq_test"]:
            if col in res.columns:
                report[f"autovap_{col.lower()}_mean"] = float(res[col].mean())
                report[f"autovap_{col.lower()}_std"] = float(res[col].std())
    return report


def run_one_feature_model(
    x_raw: np.ndarray,
    names: list[str],
    y: np.ndarray,
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    folds = make_folds(len(y), args.folds, args.seed)
    oof = np.zeros_like(y, dtype=np.float64)
    fold_rows: list[dict] = []
    importance_sum = np.zeros(x_raw.shape[1], dtype=np.float64)
    importance_count = np.zeros(x_raw.shape[1], dtype=np.float64)

    for fold, (tr, va) in enumerate(folds):
        t0 = time.time()
        x_tr, prep, prep_meta = fit_fold_prep(x_raw[tr], args.arcsinh_threshold, args.min_std)
        x_va = apply_fold_prep(x_raw[va], prep)
        clean_names = [name for name, use in zip(names, prep.keep_mask) if bool(use)]
        model = make_model(model_name, args, seed=args.seed + fold)
        selected, select_meta = select_features(x_tr, y[tr], model, args, seed=args.seed + 1000 + fold)
        x_tr_sel = x_tr[:, selected]
        x_va_sel = x_va[:, selected]
        selected_names = [clean_names[int(i)] for i in selected]

        model = make_model(model_name, args, seed=args.seed + fold)
        model.fit(x_tr_sel, y[tr])
        pred = np.asarray(model.predict(x_va_sel), dtype=np.float64)
        oof[va] = pred

        if hasattr(model, "feature_importances_"):
            local_imp = np.asarray(model.feature_importances_, dtype=np.float64)
            clean_idx = np.flatnonzero(prep.keep_mask)[selected]
            importance_sum[clean_idx] += np.nan_to_num(local_imp, nan=0.0, posinf=0.0, neginf=0.0)
            importance_count[clean_idx] += 1.0

        row = {
            "feature_set": feature_set,
            "model": model_name,
            "fold": fold,
            "seconds": time.time() - t0,
            **metrics(y[va], pred),
            **prep_meta,
            **select_meta,
        }
        fold_rows.append(row)
        print(
            f"[{feature_set}/{model_name}] fold={fold} mae={row['mae']:.4f} "
            f"rmse={row['rmse']:.4f} r2={row['r2']:.4f} clean={row['clean_features']} "
            f"selected={row['selected_features']} seconds={row['seconds']:.1f}",
            flush=True,
        )
        if args.save_selected_features:
            (out_dir / f"selected_{feature_set}_{model_name}_fold{fold}.txt").write_text("\n".join(selected_names) + "\n")

    overall = {
        "feature_set": feature_set,
        "model": model_name,
        "n": int(len(y)),
        "folds": int(args.folds),
        "raw_features": int(x_raw.shape[1]),
        "select": args.select,
        "select_top_k": int(args.select_top_k),
        **metrics(y, oof),
    }

    pred_df = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=int),
            "canonical_smiles": df["canonical_smiles"].astype(str).to_numpy(),
            "y_true_dipole_debye": y,
            "y_pred_dipole_debye": oof,
            "feature_set": feature_set,
            "model": model_name,
        }
    )
    pred_df.to_csv(out_dir / f"predictions_{feature_set}_{model_name}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"fold_metrics_{feature_set}_{model_name}.csv", index=False)

    if np.any(importance_count > 0):
        avg = np.divide(importance_sum, np.maximum(importance_count, 1.0))
        imp_df = pd.DataFrame({"feature": names, "importance": avg})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df.to_csv(out_dir / f"feature_importance_{feature_set}_{model_name}.csv", index=False)

    return overall


def parse_csv_arg(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-only XGB/RF dipole benchmark on sigma/Osmordred/Chemeleon features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dipole_physchem_chaos3d_graph"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-sets", type=str, default="osmordred,osmordred_sigma,osmordred_sigma_chemeleon")
    parser.add_argument("--models", type=str, default="xgb,rf_autovap")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-mols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--min-std", type=float, default=1.0e-12)
    parser.add_argument("--chemeleon-batch-size", type=int, default=128)
    parser.add_argument("--autovap-dir", type=Path, default=None)

    parser.add_argument("--select", choices=["none", "model_importance", "permutation"], default="none")
    parser.add_argument("--select-top-k", type=int, default=0)
    parser.add_argument("--select-min-importance", type=float, default=0.0)
    parser.add_argument("--select-valid-size", type=float, default=0.25)
    parser.add_argument("--select-permutation-repeats", type=int, default=2)
    parser.add_argument("--save-selected-features", action="store_true")

    parser.add_argument("--xgb-trees", type=int, default=900)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.90)
    parser.add_argument("--xgb-colsample", type=float, default=0.80)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)

    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=20)
    parser.add_argument("--rf-max-features", default="sqrt")
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-bootstrap", action="store_true")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_dir, args.max_mols, args.seed)
    y = df["dipole_debye"].to_numpy(dtype=np.float64)
    feature_sets = parse_csv_arg(args.feature_sets)
    models = parse_csv_arg(args.models)

    summaries: list[dict] = []
    feature_meta: dict = {}
    first_names: list[str] = []

    for feature_set in feature_sets:
        x_raw, names, meta = load_feature_set(args.data_dir, df, feature_set, args.chemeleon_batch_size)
        if not first_names:
            first_names = names
        feature_meta[feature_set] = meta
        if x_raw.shape[1] == 0:
            print(f"[skip] feature_set={feature_set} has no features", flush=True)
            continue
        print(f"[features] {feature_set}: X={x_raw.shape}", flush=True)
        for model_name in models:
            summaries.append(run_one_feature_model(x_raw, names, y, df, feature_set, model_name, args, args.out_dir))

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
    summary_df.to_csv(args.out_dir / "direct_tabular_summary.csv", index=False)

    report = {
        "args": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir), "autovap_dir": str(args.autovap_dir) if args.autovap_dir else None},
        "dataset": {
            "rows": int(len(df)),
            "target": "dipole_debye",
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
        },
        "feature_sets": feature_meta,
        "autovap_compare": compare_autovap(args.data_dir, args.autovap_dir, first_names),
    }
    (args.out_dir / "run_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
