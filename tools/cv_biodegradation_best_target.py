#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from annotate_biodegradation_protocol_unknown_pdl_etr import force_protocol_features
from build_biodegradation_best_protocol_target import (
    DEFAULT_HOMOSET_MOL,
    DEFAULT_HOMOSET_OBS,
    build_targets,
    load_strict_protocol_measurements,
)
from train_biodegradation_pdl_ordered import (
    apply_feature_prep,
    apply_feature_selector,
    build_features,
    build_pair_features,
    fit_feature_prep,
    fit_feature_selector,
    make_regressor,
    predict_pdl,
    sample_pairs,
)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true[mask], dtype=np.float64)
    y_pred = np.asarray(y_pred[mask], dtype=np.float64)
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    y60 = (y_true >= 60.0).astype(int)
    p60 = (y_pred >= 60.0).astype(int)
    y70 = (y_true >= 70.0).astype(int)
    p70 = (y_pred >= 70.0).astype(int)
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "bacc60": float(balanced_accuracy_score(y60, p60)) if len(np.unique(y60)) > 1 else np.nan,
        "bacc70": float(balanced_accuracy_score(y70, p70)) if len(np.unique(y70)) > 1 else np.nan,
    }
    for thr, yy in [(60, y60), (70, y70)]:
        try:
            out[f"roc_auc{thr}"] = float(roc_auc_score(yy, y_pred)) if len(np.unique(yy)) > 1 else np.nan
        except Exception:
            out[f"roc_auc{thr}"] = np.nan
    return out


def make_best_training_frame(best: pd.DataFrame, *, include_protocol_features: bool, target_col: str) -> pd.DataFrame:
    df = best.copy()
    df["y_percent"] = pd.to_numeric(df[target_col], errors="coerce").clip(0.0, 100.0)
    df["duration_days"] = pd.to_numeric(df["duration_round"], errors="coerce")
    df["protocol_group"] = df["best_protocol_group"].astype(str)
    df["guideline_norm"] = df["best_guideline_norm"].astype(str)
    if not include_protocol_features:
        df["protocol_group"] = "best"
        df["guideline_norm"] = "BEST_ACCEPTED"
    df["source"] = "best_accepted_protocol"
    df["raw_smiles"] = df["canonical_smiles"]
    return df.reset_index(drop=True)


def run_variant(args: argparse.Namespace, *, include_protocol_features: bool, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    t0 = time.time()
    measurements = load_strict_protocol_measurements(args)
    _, best = build_targets(measurements)
    df = make_best_training_frame(best, include_protocol_features=include_protocol_features, target_col=target_col)
    y = df["y_percent"].to_numpy(dtype=np.float32)
    guideline_categories = sorted(df["guideline_norm"].dropna().astype(str).unique().tolist())
    x_raw, feature_names = build_features(
        df,
        n_bits=args.n_bits,
        radius=args.radius,
        include_biowin=False,
        include_epi_physchem=False,
        guideline_categories=guideline_categories,
        molecular_feature_stack=args.molecular_feature_stack,
        feature_blocks=[b.strip().upper() for b in args.feature_blocks.split(",") if b.strip()],
        n_jobs=args.jobs,
        use_osmo_cache=not args.no_osmo_cache,
    )

    direct_pred = np.full(len(df), np.nan, dtype=np.float32)
    pdl_pred = np.full(len(df), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    selected_names: list[str] = []
    groups = df["canonical_smiles"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=args.folds).split(np.arange(len(df)), groups=groups))
    for fold_id, (tr, va) in enumerate(folds, start=1):
        x_train_prepared, prep = fit_feature_prep(x_raw[tr], args.arcsinh_threshold)
        x_val = apply_feature_prep(x_raw[va], prep)
        y_train = y[tr]
        x_train, selector = fit_feature_selector(
            x_train_prepared,
            y_train,
            method=args.select_method,
            k=args.select_k,
            seed=args.seed + fold_id,
            jobs=args.jobs,
            trees=args.select_trees,
        )
        forced_added = 0
        if args.force_protocol_features and include_protocol_features:
            selector, forced_added = force_protocol_features(selector, prep, feature_names)
            x_train = apply_feature_selector(x_train_prepared, selector)
        x_val = apply_feature_selector(x_val, selector)
        if fold_id == 1:
            src_idx = np.where(prep.finite_mask)[0][prep.keep_mask][selector.indices]
            selected_names = [feature_names[int(i)] for i in src_idx]

        direct = make_regressor("etr", args, args.seed + 100 + fold_id)
        direct.fit(x_train, y_train)
        direct_pred[va] = np.clip(direct.predict(x_val), 0.0, 100.0).astype(np.float32)

        pair_a, pair_b, dy = sample_pairs(
            y_train,
            n_pairs=args.pairs_per_fold,
            min_abs_dy=args.min_abs_dy,
            seed=args.seed + 1000 + fold_id,
        )
        pair_raw = build_pair_features(
            x_train[pair_a],
            x_train[pair_b],
            y_train[pair_a],
            include_abs_delta=args.include_abs_delta,
        )
        pair_train, pair_prep = fit_feature_prep(pair_raw, args.arcsinh_threshold)
        pdl = make_regressor("etr", args, args.seed + 2000 + fold_id)
        pdl.fit(pair_train, dy)
        pdl_pred[va] = np.clip(
            predict_pdl(
                pdl,
                pair_prep,
                x_train,
                y_train,
                x_val,
                anchors=args.anchors,
                include_abs_delta=args.include_abs_delta,
                aggregate=args.aggregate,
                batch_size=args.predict_batch_size,
            ),
            0.0,
            100.0,
        ).astype(np.float32)
        fold_rows.append(
            {
                "fold": fold_id,
                "variant": f"{target_col}__{'with_best_protocol_features' if include_protocol_features else 'target_only'}",
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "features": int(x_train.shape[1]),
                "forced_protocol_features_added": int(forced_added),
                "pairs": int(len(dy)),
                "elapsed_seconds": float(time.time() - t0),
                **{f"direct_{k}": v for k, v in metric_row(y[va], direct_pred[va]).items()},
                **{f"pdl_{k}": v for k, v in metric_row(y[va], pdl_pred[va]).items()},
            }
        )
        print(
            f"[biodeg-best-cv] variant={fold_rows[-1]['variant']} fold={fold_id}/{len(folds)} "
            f"features={x_train.shape[1]} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    pred = df.copy()
    pred["direct_pred"] = direct_pred
    pred["pdl_pred"] = pdl_pred
    pred["include_best_protocol_features"] = bool(include_protocol_features)

    summary_rows: list[dict[str, Any]] = []
    for method, values in [("direct_etr", direct_pred), ("pdl_etr", pdl_pred)]:
        row = {
            "variant": f"{target_col}__{'with_best_protocol_features' if include_protocol_features else 'target_only'}",
            "method": method,
            "n_rows": int(len(df)),
            "n_molecules": int(df["canonical_smiles"].nunique()),
            "n_ready_best": int(df["best_protocol_group"].eq("ready").sum()),
            "n_inherent_best": int(df["best_protocol_group"].eq("inherent").sum()),
            "target_col": target_col,
            "features_raw": int(x_raw.shape[1]),
            "selected_features_fold1": int(len(selected_names)),
        }
        row.update(metric_row(y, values))
        summary_rows.append(row)
    meta = {
        "strict_measurement_rows": int(len(measurements)),
        "strict_measurement_molecules": int(measurements["canonical_smiles"].nunique()),
        "best_target_rows": int(len(df)),
        "best_target_molecules": int(df["canonical_smiles"].nunique()),
        "multi_protocol_duration_rows": int(df["n_protocol_candidates"].gt(1).sum()),
        "best_protocol_counts": df["best_protocol_group"].value_counts().to_dict(),
        "include_protocol_features": bool(include_protocol_features),
        "target_col": target_col,
        "elapsed_seconds": float(time.time() - t0),
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pred, selected_names, meta


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [False, True] if args.run_with_and_without_protocol else [args.include_best_protocol_features]
    target_cols = [x.strip() for x in args.target_cols.split(",") if x.strip()]
    summaries = []
    folds = []
    metas = []
    for target_col in target_cols:
      for include_protocol in variants:
        tag = f"{target_col}__{'with_best_protocol_features' if include_protocol else 'target_only'}"
        summary, fold_df, pred, selected, meta = run_variant(
            args,
            include_protocol_features=include_protocol,
            target_col=target_col,
        )
        summary.to_csv(out_dir / f"{tag}_summary.csv", index=False)
        fold_df.to_csv(out_dir / f"{tag}_fold_metrics.csv", index=False)
        pred.to_csv(out_dir / f"{tag}_predictions.csv", index=False)
        (out_dir / f"{tag}_selected_features_fold1.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
        summaries.append(summary)
        folds.append(fold_df)
        metas.append({**meta, "variant": tag})
    all_summary = pd.concat(summaries, ignore_index=True).sort_values("mae")
    all_folds = pd.concat(folds, ignore_index=True)
    all_summary.to_csv(out_dir / "summary.csv", index=False)
    all_folds.to_csv(out_dir / "fold_metrics.csv", index=False)
    (out_dir / "run_report.json").write_text(json.dumps(metas, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(all_summary.to_string(index=False), flush=True)
    print(f"[biodeg-best-cv] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV-5 on best accepted biodegradation target.")
    parser.add_argument("--homoset-observation-csv", default=str(DEFAULT_HOMOSET_OBS))
    parser.add_argument("--homoset-molecule-csv", default=str(DEFAULT_HOMOSET_MOL))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_protocol_annotation/cv5_best_protocol_target_v1")
    parser.add_argument("--direct-percent-only", action="store_true", default=True)
    parser.add_argument("--allow-bound-percent", dest="direct_percent_only", action="store_false")
    parser.add_argument("--strict-protocol-guidelines", action="store_true", default=True)
    parser.add_argument("--allow-loose-guidelines", dest="strict_protocol_guidelines", action="store_false")
    parser.add_argument("--include-best-protocol-features", action="store_true", default=False)
    parser.add_argument("--run-with-and-without-protocol", action="store_true", default=True)
    parser.add_argument("--target-cols", default="y_best,upper_consensus_y_percent")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--molecular-feature-stack", choices=["osmo", "rdkit"], default="osmo")
    parser.add_argument("--feature-blocks", default="RDKIT217,OSMO,GOLD,ABRAHAM,FUNCGROUPS")
    parser.add_argument("--no-osmo-cache", action="store_true")
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--select-method", choices=["none", "rpcholesky", "variance", "f_regression", "mutual_info", "etr_importance"], default="rpcholesky")
    parser.add_argument("--select-k", type=int, default=768)
    parser.add_argument("--select-trees", type=int, default=50)
    parser.add_argument("--force-protocol-features", action="store_true", default=True)
    parser.add_argument("--pairs-per-fold", type=int, default=50000)
    parser.add_argument("--min-abs-dy", type=float, default=3.0)
    parser.add_argument("--include-abs-delta", action="store_true")
    parser.add_argument("--anchors", type=int, default=96)
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--predict-batch-size", type=int, default=64)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run(parse_args())
