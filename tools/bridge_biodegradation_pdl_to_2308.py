#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from train_biodegradation_pdl_ordered import (
    DEFAULT_RIFM_2026,
    apply_feature_prep,
    apply_feature_selector,
    build_features,
    build_pair_features,
    canonical_smiles,
    fit_feature_prep,
    fit_feature_selector,
    load_rifm,
    make_regressor,
    predict_pdl,
    sample_pairs,
)


DEFAULT_TRAIN = Path("/Users/guillaume-osmo/Github/transformer-CNN-osmoai/transformer_cnn/biodegradation_2308_train.csv")
DEFAULT_VAL = Path("/Users/guillaume-osmo/Github/transformer-CNN-osmoai/transformer_cnn/biodegradation_2308_val.csv")


PRESETS = {
    "ready_mean": [
        ("OECD301B", 28.0),
        ("OECD301D", 28.0),
        ("OECD301F", 28.0),
        ("OECD310", 28.0),
    ],
    "oecd301f_28": [("OECD301F", 28.0)],
    "oecd301b_28": [("OECD301B", 28.0)],
    "oecd301d_28": [("OECD301D", 28.0)],
    "oecd310_28": [("OECD310", 28.0)],
}

PRESET_PROTOCOL_THRESHOLDS = {
    "ready_mean": 60.0,
    "oecd301f_28": 60.0,
    "oecd301b_28": 60.0,
    "oecd301d_28": 60.0,
    "oecd310_28": 60.0,
}


def load_2308(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "smiles" not in df.columns or "Result0" not in df.columns:
        raise ValueError(f"{path} must contain smiles and Result0 columns")
    df = df.copy()
    df["canonical_smiles"] = df["smiles"].map(canonical_smiles)
    df["y_binary"] = pd.to_numeric(df["Result0"], errors="coerce")
    df = df[df["canonical_smiles"].notna() & df["y_binary"].notna()].reset_index(drop=True)
    df["y_binary"] = df["y_binary"].astype(int)
    return df


def make_virtual_rows(df: pd.DataFrame, contexts: list[tuple[str, float]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for preset_id, (guideline, days) in enumerate(contexts):
        block = pd.DataFrame(
            {
                "row_id": np.arange(len(df), dtype=int),
                "preset_id": preset_id,
                "canonical_smiles": df["canonical_smiles"].to_numpy(),
                "SMILES": df["canonical_smiles"].to_numpy(),
                "Test guideline": guideline,
                "guideline_norm": guideline,
                "Duration": f"{days:g} days",
                "duration_days": float(days),
                "Unit": "%",
            }
        )
        rows.append(block)
    return pd.concat(rows, axis=0, ignore_index=True)


def aggregate_virtual(values: np.ndarray, virtual: pd.DataFrame, n_rows: int, agg: str) -> np.ndarray:
    out = np.full((n_rows,), np.nan, dtype=np.float32)
    for row_id, group in virtual.groupby("row_id", sort=False):
        vals = np.asarray(values[group.index], dtype=np.float32)
        out[int(row_id)] = np.nanmean(vals) if agg == "mean" else np.nanmedian(vals)
    return out


def choose_threshold(y: np.ndarray, score: np.ndarray, metric: str) -> tuple[float, float]:
    finite = np.isfinite(score)
    y = y[finite].astype(int)
    score = score[finite]
    thresholds = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 199)))
    best_t = float(np.median(score))
    best = -1.0
    for t in thresholds:
        pred = (score >= t).astype(int)
        if metric == "f1":
            val = f1_score(y, pred, zero_division=0)
        else:
            val = balanced_accuracy_score(y, pred)
        if val > best:
            best = float(val)
            best_t = float(t)
    return best_t, best


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    finite = np.isfinite(score)
    y = y[finite].astype(int)
    score = score[finite]
    pred = (score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    row = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    if len(np.unique(y)) == 2:
        row["roc_auc"] = float(roc_auc_score(y, score))
        row["average_precision"] = float(average_precision_score(y, score))
    else:
        row["roc_auc"] = np.nan
        row["average_precision"] = np.nan
    return row


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rifm = load_rifm(Path(args.rifm_xlsx), clip_target=True)
    guideline_categories = sorted(rifm["guideline_norm"].astype(str).unique().tolist())
    x_rifm_raw, feature_names = build_features(
        rifm,
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
    y_rifm = rifm["y_percent"].to_numpy(dtype=np.float32)
    x_rifm, prep = fit_feature_prep(x_rifm_raw, args.arcsinh_threshold)
    x_rifm, selector = fit_feature_selector(
        x_rifm,
        y_rifm,
        method=args.select_method,
        k=args.select_k,
        seed=args.seed + 77,
        jobs=args.jobs,
        trees=args.select_trees,
    )

    direct = make_regressor(args.model, args, args.seed)
    direct.fit(x_rifm, y_rifm)

    pair_a, pair_b, dy = sample_pairs(
        y_rifm,
        n_pairs=args.pairs,
        min_abs_dy=args.min_abs_dy,
        seed=args.seed + 99,
    )
    pair_raw = build_pair_features(
        x_rifm[pair_a],
        x_rifm[pair_b],
        y_rifm[pair_a],
        include_abs_delta=args.include_abs_delta,
    )
    pair_train, pair_prep = fit_feature_prep(pair_raw, args.arcsinh_threshold)
    pdl = make_regressor(args.model, args, args.seed + 1)
    pdl.fit(pair_train, dy)

    train = load_2308(Path(args.train_csv))
    val = load_2308(Path(args.val_csv))

    summary_rows: list[dict] = []
    pred_blocks: list[pd.DataFrame] = []

    for preset_name, contexts in PRESETS.items():
        for split_name, split_df in [("train", train), ("val", val)]:
            virtual = make_virtual_rows(split_df, contexts)
            x_virtual_raw, _ = build_features(
                virtual,
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
            x_virtual = apply_feature_prep(x_virtual_raw, prep)
            x_virtual = apply_feature_selector(x_virtual, selector)
            direct_virtual = direct.predict(x_virtual).astype(np.float32)
            pdl_virtual = predict_pdl(
                pdl,
                pair_prep,
                x_rifm,
                y_rifm,
                x_virtual,
                anchors=args.anchors,
                include_abs_delta=args.include_abs_delta,
                aggregate=args.anchor_aggregate,
                batch_size=args.predict_batch_size,
            )
            block = split_df[["smiles", "canonical_smiles", "y_binary"]].copy()
            block["split"] = split_name
            block["preset"] = preset_name
            block["direct_percent_score"] = aggregate_virtual(direct_virtual, virtual, len(split_df), args.preset_aggregate)
            block["pdl_percent_score"] = aggregate_virtual(pdl_virtual, virtual, len(split_df), args.preset_aggregate)
            pred_blocks.append(block)

    pred = pd.concat(pred_blocks, axis=0, ignore_index=True)

    for preset_name in PRESETS:
        train_pred = pred[(pred["split"] == "train") & (pred["preset"] == preset_name)]
        val_pred = pred[(pred["split"] == "val") & (pred["preset"] == preset_name)]
        for score_col in ["direct_percent_score", "pdl_percent_score"]:
            threshold, train_bal = choose_threshold(
                train_pred["y_binary"].to_numpy(),
                train_pred[score_col].to_numpy(dtype=np.float32),
                args.threshold_metric,
            )
            threshold_variants = [("calibrated", threshold, train_bal)]
            if preset_name in PRESET_PROTOCOL_THRESHOLDS:
                threshold_variants.append(("protocol_fixed", PRESET_PROTOCOL_THRESHOLDS[preset_name], np.nan))
            for threshold_kind, threshold_value, train_score in threshold_variants:
                for split_name, split_pred in [("train", train_pred), ("val", val_pred)]:
                    row = {
                        "method": score_col.replace("_percent_score", ""),
                        "preset": preset_name,
                        "split": split_name,
                        "threshold_kind": threshold_kind,
                        "train_threshold_score": float(train_score) if np.isfinite(train_score) else np.nan,
                    }
                    row.update(
                        classification_metrics(
                            split_pred["y_binary"].to_numpy(),
                            split_pred[score_col].to_numpy(dtype=np.float32),
                            threshold_value,
                        )
                    )
                    summary_rows.append(row)

    # Protocol decision helper for the continuous model. OECD 301/310 ready
    # tests usually use 60% for CO2/BOD/ThOD endpoints, while DOC removal
    # protocols use 70%. The current virtual presets are all 60% endpoint
    # variants; calibration is still reported next to this fixed rule.
    protocol_thresholds = PRESET_PROTOCOL_THRESHOLDS

    for preset_name, fixed_threshold in protocol_thresholds.items():
        pass

    # A direct binary baseline on the 2308 split, using the same virtual context.
    base_context = PRESETS[args.binary_baseline_preset]
    train_virtual = make_virtual_rows(train, base_context)
    val_virtual = make_virtual_rows(val, base_context)
    x_train_raw, _ = build_features(
        train_virtual,
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
    x_val_raw, _ = build_features(
        val_virtual,
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
    x_train_bin, bin_prep = fit_feature_prep(x_train_raw, args.arcsinh_threshold)
    x_val_bin = apply_feature_prep(x_val_raw, bin_prep)
    y_train_bin = train_virtual.merge(train[["y_binary"]], left_on="row_id", right_index=True)["y_binary"].to_numpy(dtype=int)
    x_train_bin, bin_selector = fit_feature_selector(
        x_train_bin,
        y_train_bin.astype(np.float32),
        method=args.select_method,
        k=args.select_k,
        seed=args.seed + 88,
        jobs=args.jobs,
        trees=args.select_trees,
    )
    x_val_bin = apply_feature_selector(x_val_bin, bin_selector)
    clf = ExtraTreesClassifier(
        n_estimators=args.trees,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.jobs,
    )
    clf.fit(x_train_bin, y_train_bin)
    bin_train_score = clf.predict_proba(x_train_bin)[:, 1]
    bin_val_score = clf.predict_proba(x_val_bin)[:, 1]
    threshold, train_bal = choose_threshold(
        y_train_bin,
        bin_train_score,
        args.threshold_metric,
    )
    for split_name, y_true, score in [
        ("train", y_train_bin, bin_train_score),
        ("val", val_virtual.merge(val[["y_binary"]], left_on="row_id", right_index=True)["y_binary"].to_numpy(dtype=int), bin_val_score),
    ]:
        row = {
            "method": "binary_etr_2308",
            "preset": args.binary_baseline_preset,
            "split": split_name,
            "threshold_kind": "calibrated",
            "train_threshold_score": float(train_bal),
        }
        row.update(classification_metrics(y_true, score, threshold))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["split", "balanced_accuracy", "roc_auc"], ascending=[True, False, False]).reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pred.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    meta = {
        "rifm_xlsx": str(Path(args.rifm_xlsx)),
        "train_csv": str(Path(args.train_csv)),
        "val_csv": str(Path(args.val_csv)),
        "rifm_rows": int(len(rifm)),
        "rifm_unique_molecules": int(rifm["canonical_smiles"].nunique()),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "pairs": int(args.pairs),
        "trees": int(args.trees),
        "select_method": args.select_method,
        "select_k": int(args.select_k),
        "protocol_thresholds": protocol_thresholds,
        "presets": PRESETS,
        "elapsed_seconds": float(time.time() - t0),
        "args": vars(args),
    }
    (out_dir / "run_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[biodeg-bridge] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge RIFM pairwise % biodegradation regression to the 2308 binary split.")
    parser.add_argument("--rifm-xlsx", default=str(DEFAULT_RIFM_2026))
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val-csv", default=str(DEFAULT_VAL))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_pdl_ordered/rifm_to_2308_binary_bridge")
    parser.add_argument("--model", default="etr")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--molecular-feature-stack", choices=["osmo", "rdkit"], default="osmo")
    parser.add_argument(
        "--feature-blocks",
        default="RDKIT217,OSMO,GOLD,ABRAHAM,FUNCGROUPS",
        help="OSMO stack blocks; GOLD is the calcPhysChem/v34 cascade feature block.",
    )
    parser.add_argument("--no-osmo-cache", action="store_true")
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--select-method",
        choices=["none", "rpcholesky", "variance", "f_regression", "mutual_info", "etr_importance"],
        default="rpcholesky",
    )
    parser.add_argument("--select-k", type=int, default=1024)
    parser.add_argument("--select-trees", type=int, default=50)
    parser.add_argument("--xgb-estimators", type=int, default=250)
    parser.add_argument("--xgb-depth", type=int, default=4)
    parser.add_argument("--xgb-lr", type=float, default=0.04)
    parser.add_argument("--ensemble-models", default="etr,rf,xgb,catboost,svm,lgbm")
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-lr", type=float, default=0.04)
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--lgbm-depth", type=int, default=-1)
    parser.add_argument("--lgbm-lr", type=float, default=0.04)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-epsilon", type=float, default=0.1)
    parser.add_argument("--svm-max-iter", type=int, default=5000)
    parser.add_argument("--svm-tol", type=float, default=1.0e-4)
    parser.add_argument("--pairs", type=int, default=30_000)
    parser.add_argument("--min-abs-dy", type=float, default=3.0)
    parser.add_argument("--include-abs-delta", action="store_true")
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--anchor-aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--preset-aggregate", choices=["median", "mean"], default="mean")
    parser.add_argument("--predict-batch-size", type=int, default=96)
    parser.add_argument("--threshold-metric", choices=["balanced_accuracy", "f1"], default="balanced_accuracy")
    parser.add_argument("--binary-baseline-preset", choices=list(PRESETS), default="oecd301f_28")
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
