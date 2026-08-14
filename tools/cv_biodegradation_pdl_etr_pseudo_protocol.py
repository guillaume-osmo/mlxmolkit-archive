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
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

from annotate_biodegradation_protocol_unknown_pdl_etr import (
    PROTOCOL_CANDIDATES,
    force_protocol_features,
    load_clean_known,
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


DEFAULT_HOMOSET_OBS = Path(
    "benchmarks/biodegradation_homoset_audit/"
    "rifm_excelra_2308_v3_curated_targets/biodegradation_observation_homoset.csv"
)
DEFAULT_HOMOSET_MOL = Path(
    "benchmarks/biodegradation_homoset_audit/"
    "rifm_excelra_2308_v3_curated_targets/biodegradation_molecule_curated_targets.csv"
)


DEFAULT_ANNOTATION = Path(
    "benchmarks/biodegradation_protocol_annotation/"
    "cas_unknown_clean_known_plus_cas_known_pdl_etr_v2_force_protocol/"
    "unknown_protocol_pdl_etr_annotations.csv"
)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }
    y_bin = (y_true >= 60.0).astype(int)
    p_bin = (y_pred >= 60.0).astype(int)
    out["bacc60"] = float(balanced_accuracy_score(y_bin, p_bin)) if len(np.unique(y_bin)) > 1 else np.nan
    try:
        out["roc_auc60"] = float(roc_auc_score(y_bin, y_pred)) if len(np.unique(y_bin)) > 1 else np.nan
    except Exception:
        out["roc_auc60"] = np.nan
    return out


def load_pseudo_annotations(args: argparse.Namespace) -> pd.DataFrame:
    if not args.annotation_csv:
        return pd.DataFrame()
    ann = pd.read_csv(args.annotation_csv, low_memory=False)
    ann = ann[
        ann["canonical_smiles"].notna()
        & ann["y_percent"].notna()
        & ann["predicted_protocol_group"].isin(PROTOCOL_CANDIDATES)
    ].copy()
    ann["best_abs_error_percent"] = pd.to_numeric(ann.get("best_abs_error_percent"), errors="coerce")
    ann["error_margin_percent"] = pd.to_numeric(ann.get("error_margin_percent"), errors="coerce")
    if args.max_best_error >= 0:
        ann = ann[ann["best_abs_error_percent"].le(args.max_best_error)].copy()
    if args.min_margin > 0:
        ann = ann[ann["error_margin_percent"].ge(args.min_margin)].copy()

    ann["source"] = "cas_unknown_pseudo_protocol"
    ann["protocol_group"] = ann["predicted_protocol_group"].astype(str)
    ann["guideline_norm"] = ann["protocol_group"].map(PROTOCOL_CANDIDATES).fillna("OTHER")
    ann["y_percent"] = pd.to_numeric(ann["y_percent"], errors="coerce").clip(0.0, 100.0)
    ann["duration_days"] = pd.to_numeric(ann["duration_days"], errors="coerce")
    ann["is_pseudo_protocol"] = True
    ann["pseudo_annotation_csv"] = str(args.annotation_csv)
    return ann.reset_index(drop=True)


def selected_feature_names(prep: Any, selector: Any, feature_names: list[str]) -> list[str]:
    source_idx = np.where(prep.finite_mask)[0][prep.keep_mask][selector.indices]
    return [feature_names[int(i)] for i in source_idx]


def run_cv(args: argparse.Namespace, *, include_pseudo: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    known = load_clean_known(args)
    known["is_pseudo_protocol"] = False
    pseudo = load_pseudo_annotations(args) if include_pseudo else pd.DataFrame()

    all_rows = pd.concat([known, pseudo.reindex(columns=known.columns.union(pseudo.columns))], ignore_index=True, sort=False)
    n_known = len(known)
    pseudo_indices = np.arange(n_known, len(all_rows), dtype=np.int64)

    guideline_categories = sorted(all_rows["guideline_norm"].dropna().astype(str).unique().tolist())
    x_raw, feature_names = build_features(
        all_rows,
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
    y_all = all_rows["y_percent"].to_numpy(dtype=np.float32)
    groups = known["canonical_smiles"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=args.folds).split(np.arange(n_known), groups=groups))

    direct_pred = np.full(n_known, np.nan, dtype=np.float32)
    pdl_pred = np.full(n_known, np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    selected_names: list[str] = []

    for fold_id, (tr_known, va_known) in enumerate(folds, start=1):
        val_smiles = set(known.iloc[va_known]["canonical_smiles"].astype(str))
        if include_pseudo and len(pseudo_indices):
            pseudo_train = pseudo_indices[
                ~all_rows.iloc[pseudo_indices]["canonical_smiles"].astype(str).isin(val_smiles).to_numpy()
            ]
            tr = np.concatenate([tr_known.astype(np.int64), pseudo_train])
        else:
            tr = tr_known.astype(np.int64)
        va = va_known.astype(np.int64)

        x_train_prepared, prep = fit_feature_prep(x_raw[tr], args.arcsinh_threshold)
        x_val = apply_feature_prep(x_raw[va], prep)
        y_train = y_all[tr]

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
        if args.force_protocol_features:
            selector, forced_added = force_protocol_features(selector, prep, feature_names)
            x_train = apply_feature_selector(x_train_prepared, selector)
        x_val = apply_feature_selector(x_val, selector)
        if fold_id == 1:
            selected_names = selected_feature_names(prep, selector, feature_names)

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
                "train_known_rows": int(len(tr_known)),
                "train_pseudo_rows": int(max(0, len(tr) - len(tr_known))),
                "val_known_rows": int(len(va)),
                "features": int(x_train.shape[1]),
                "forced_protocol_features_added": int(forced_added),
                "pairs": int(len(dy)),
                "elapsed_seconds": float(time.time() - t0),
                **{f"direct_{k}": v for k, v in metric_row(y_all[va], direct_pred[va]).items()},
                **{f"pdl_{k}": v for k, v in metric_row(y_all[va], pdl_pred[va]).items()},
            }
        )
        print(
            f"[biodeg-cv] include_pseudo={include_pseudo} fold={fold_id}/{len(folds)} "
            f"train_known={len(tr_known)} train_pseudo={max(0, len(tr)-len(tr_known))} "
            f"val={len(va)} features={x_train.shape[1]} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    pred = known.copy()
    pred["direct_pred"] = direct_pred
    pred["pdl_pred"] = pdl_pred
    pred["include_pseudo_training"] = bool(include_pseudo)

    summary_rows = []
    for method, values in [("direct_etr", direct_pred), ("pdl_etr", pdl_pred)]:
        row = {
            "variant": "known_plus_pseudo_train" if include_pseudo else "known_only",
            "method": method,
            "n_eval_known_rows": int(n_known),
            "n_eval_known_molecules": int(known["canonical_smiles"].nunique()),
            "n_train_pseudo_rows_total": int(len(pseudo) if include_pseudo else 0),
            "n_train_pseudo_molecules_total": int(pseudo["canonical_smiles"].nunique() if include_pseudo and len(pseudo) else 0),
            "folds": int(args.folds),
            "selected_features_fold1": int(len(selected_names)),
        }
        row.update(metric_row(known["y_percent"].to_numpy(dtype=np.float32), values))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    meta = {
        "include_pseudo": bool(include_pseudo),
        "known_rows": int(len(known)),
        "known_molecules": int(known["canonical_smiles"].nunique()),
        "pseudo_rows": int(len(pseudo) if include_pseudo else 0),
        "pseudo_molecules": int(pseudo["canonical_smiles"].nunique() if include_pseudo and len(pseudo) else 0),
        "annotation_csv": str(args.annotation_csv),
        "min_margin": float(args.min_margin),
        "max_best_error": float(args.max_best_error),
        "elapsed_seconds": float(time.time() - t0),
    }
    return summary, pd.DataFrame(fold_rows), pred, selected_names, meta


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    folds = []
    metas = []
    for include_pseudo in ([False, True] if args.run_baseline else [True]):
        tag = "known_plus_pseudo_train" if include_pseudo else "known_only"
        summary, fold_df, pred, selected_names, meta = run_cv(args, include_pseudo=include_pseudo)
        summary.to_csv(out_dir / f"{tag}_summary.csv", index=False)
        fold_df.to_csv(out_dir / f"{tag}_fold_metrics.csv", index=False)
        pred.to_csv(out_dir / f"{tag}_predictions.csv", index=False)
        (out_dir / f"{tag}_selected_features_fold1.txt").write_text("\n".join(selected_names) + "\n", encoding="utf-8")
        summaries.append(summary)
        folds.append(fold_df.assign(variant=tag))
        metas.append({**meta, "variant": tag})

    all_summary = pd.concat(summaries, ignore_index=True)
    all_fold = pd.concat(folds, ignore_index=True)
    all_summary.to_csv(out_dir / "summary.csv", index=False)
    all_fold.to_csv(out_dir / "fold_metrics.csv", index=False)
    (out_dir / "run_report.json").write_text(json.dumps(metas, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(all_summary.to_string(index=False), flush=True)
    print(f"[biodeg-cv] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="True known-row CV-5 for PDL-ETR with optional pseudo protocol training rows.")
    parser.add_argument("--homoset-observation-csv", default=str(DEFAULT_HOMOSET_OBS))
    parser.add_argument("--homoset-molecule-csv", default=str(DEFAULT_HOMOSET_MOL))
    parser.add_argument("--annotation-csv", default=str(DEFAULT_ANNOTATION))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_protocol_annotation/cv5_pdl_etr_pseudo_protocol_v1")
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--no-clean-only", dest="clean_only", action="store_false")
    parser.add_argument("--exclude-unknown-source-like", action="store_true")
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--max-best-error", type=float, default=-1.0)
    parser.add_argument("--run-baseline", action="store_true", default=True)
    parser.add_argument("--no-run-baseline", dest="run_baseline", action="store_false")
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
    parser.add_argument("--no-force-protocol-features", dest="force_protocol_features", action="store_false")
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
