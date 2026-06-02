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

from annotate_biodegradation_protocol_unknown_pdl_etr import (
    PROTOCOL_CANDIDATES,
    force_protocol_features,
    load_clean_known,
)
from cv_biodegradation_pdl_etr_pseudo_protocol import DEFAULT_HOMOSET_MOL, DEFAULT_HOMOSET_OBS
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

STRICT_PROTOCOL_RE = r"OECD301|OECD310|OECD302|BODIS|METHODC\.4|DIRECTIVE84/449/EEC,C\.4|C\.4"


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true[mask], dtype=np.float64)
    y_pred = np.asarray(y_pred[mask], dtype=np.float64)
    if len(y_true) == 0:
        return {k: np.nan for k in ["mae", "rmse", "r2", "spearman", "bacc60", "roc_auc60"]}
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    y_bin = (y_true >= 60.0).astype(int)
    p_bin = (y_pred >= 60.0).astype(int)
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "bacc60": float(balanced_accuracy_score(y_bin, p_bin)) if len(np.unique(y_bin)) > 1 else np.nan,
    }
    try:
        out["roc_auc60"] = float(roc_auc_score(y_bin, y_pred)) if len(np.unique(y_bin)) > 1 else np.nan
    except Exception:
        out["roc_auc60"] = np.nan
    return out


def selected_feature_count(prep: Any, selector: Any) -> int:
    return int(len(selector.indices))


def fit_predict_block(
    args: argparse.Namespace,
    *,
    x_raw: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    pairs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(train_idx) < args.min_train_rows or len(val_idx) == 0:
        return (
            np.full(len(val_idx), np.nan, dtype=np.float32),
            np.full(len(val_idx), np.nan, dtype=np.float32),
            {"status": "skipped", "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx))},
        )

    x_train_prepared, prep = fit_feature_prep(x_raw[train_idx], args.arcsinh_threshold)
    x_val = apply_feature_prep(x_raw[val_idx], prep)
    y_train = y[train_idx]

    x_train, selector = fit_feature_selector(
        x_train_prepared,
        y_train,
        method=args.select_method,
        k=args.select_k,
        seed=seed,
        jobs=args.jobs,
        trees=args.select_trees,
    )
    forced_added = 0
    if args.force_protocol_features:
        selector, forced_added = force_protocol_features(selector, prep, feature_names)
        x_train = apply_feature_selector(x_train_prepared, selector)
    x_val = apply_feature_selector(x_val, selector)

    direct = make_regressor("etr", args, seed + 100)
    direct.fit(x_train, y_train)
    direct_pred = np.clip(direct.predict(x_val), 0.0, 100.0).astype(np.float32)

    pdl_pred = np.full(len(val_idx), np.nan, dtype=np.float32)
    pdl_status = "ok"
    try:
        pair_a, pair_b, dy = sample_pairs(
            y_train,
            n_pairs=pairs,
            min_abs_dy=args.min_abs_dy,
            seed=seed + 1000,
        )
        pair_raw = build_pair_features(
            x_train[pair_a],
            x_train[pair_b],
            y_train[pair_a],
            include_abs_delta=args.include_abs_delta,
        )
        pair_train, pair_prep = fit_feature_prep(pair_raw, args.arcsinh_threshold)
        pdl = make_regressor("etr", args, seed + 2000)
        pdl.fit(pair_train, dy)
        pdl_pred = np.clip(
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
        n_pairs = int(len(dy))
    except Exception as exc:  # noqa: BLE001
        pdl_status = repr(exc)
        n_pairs = 0

    return direct_pred, pdl_pred, {
        "status": "ok",
        "pdl_status": pdl_status,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "features": selected_feature_count(prep, selector),
        "forced_protocol_features_added": int(forced_added),
        "pairs": int(n_pairs),
    }


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_known(args)
    allowed_protocols = [p.strip() for p in args.protocol_groups.split(",") if p.strip()]
    df = df[df["protocol_group"].isin(allowed_protocols)].copy()
    if args.direct_percent_only:
        df = df[df["y_percent_source"].astype(str).eq("direct")].copy()
    if args.strict_protocol_guidelines:
        guideline = df["guideline_norm"].astype(str).str.upper()
        df = df[guideline.str.contains(STRICT_PROTOCOL_RE, regex=True, na=False)].copy()
    df = df.reset_index(drop=True)
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

    groups = df["canonical_smiles"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=args.folds).split(np.arange(len(df)), groups=groups))
    protocols = allowed_protocols

    global_direct = np.full(len(df), np.nan, dtype=np.float32)
    global_pdl = np.full(len(df), np.nan, dtype=np.float32)
    sub_direct = np.full(len(df), np.nan, dtype=np.float32)
    sub_pdl = np.full(len(df), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (tr, va) in enumerate(folds, start=1):
        gd, gp, info = fit_predict_block(
            args,
            x_raw=x_raw,
            y=y,
            feature_names=feature_names,
            train_idx=tr.astype(np.int64),
            val_idx=va.astype(np.int64),
            seed=args.seed + fold_id,
            pairs=args.global_pairs_per_fold,
        )
        global_direct[va] = gd
        global_pdl[va] = gp
        fold_rows.append({"fold": fold_id, "model_scope": "global", "protocol_group": "ALL", **info})

        for protocol in protocols:
            tr_p = tr[df.iloc[tr]["protocol_group"].astype(str).eq(protocol).to_numpy()]
            va_p = va[df.iloc[va]["protocol_group"].astype(str).eq(protocol).to_numpy()]
            sd, sp, pinfo = fit_predict_block(
                args,
                x_raw=x_raw,
                y=y,
                feature_names=feature_names,
                train_idx=tr_p.astype(np.int64),
                val_idx=va_p.astype(np.int64),
                seed=args.seed + 100 * fold_id + protocols.index(protocol),
                pairs=args.protocol_pairs_per_fold,
            )
            sub_direct[va_p] = sd
            sub_pdl[va_p] = sp
            fold_rows.append({"fold": fold_id, "model_scope": "protocol_submodel", "protocol_group": protocol, **pinfo})

        print(
            f"[biodeg-protocol-submodels] fold={fold_id}/{len(folds)} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    pred = df.copy()
    pred["global_direct_pred"] = global_direct
    pred["global_pdl_pred"] = global_pdl
    pred["protocol_submodel_direct_pred"] = sub_direct
    pred["protocol_submodel_pdl_pred"] = sub_pdl

    summary_rows: list[dict[str, Any]] = []
    for method, values in [
        ("global_direct_etr", global_direct),
        ("global_pdl_etr", global_pdl),
        ("protocol_submodel_direct_etr", sub_direct),
        ("protocol_submodel_pdl_etr", sub_pdl),
    ]:
        row = {
            "method": method,
            "n_rows": int(np.isfinite(values).sum()),
            "n_molecules": int(df.loc[np.isfinite(values), "canonical_smiles"].nunique()),
            "features_raw": int(x_raw.shape[1]),
            "force_protocol_features": bool(args.force_protocol_features),
        }
        row.update(metric_row(y, values))
        summary_rows.append(row)

    by_protocol_rows: list[dict[str, Any]] = []
    for protocol in protocols:
        mask = df["protocol_group"].astype(str).eq(protocol).to_numpy()
        for method, values in [
            ("global_direct_etr", global_direct),
            ("global_pdl_etr", global_pdl),
            ("protocol_submodel_direct_etr", sub_direct),
            ("protocol_submodel_pdl_etr", sub_pdl),
        ]:
            row = {"protocol_group": protocol, "method": method, "n_rows": int(np.sum(mask & np.isfinite(values)))}
            row.update(metric_row(y[mask], values[mask]))
            by_protocol_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("mae")
    by_protocol = pd.DataFrame(by_protocol_rows).sort_values(["protocol_group", "mae"])
    fold_df = pd.DataFrame(fold_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    by_protocol.to_csv(out_dir / "by_protocol_metrics.csv", index=False)
    fold_df.to_csv(out_dir / "fold_model_status.csv", index=False)
    pred.to_csv(out_dir / "predictions.csv", index=False)
    report = {
        "rows": int(len(df)),
        "molecules": int(df["canonical_smiles"].nunique()),
        "protocol_counts": df["protocol_group"].value_counts().to_dict(),
        "args": vars(args),
        "elapsed_seconds": float(time.time() - t0),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[biodeg-protocol-submodels] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV-5 biodegradation percent with global vs per-protocol ETR/PDL-ETR submodels.")
    parser.add_argument("--homoset-observation-csv", default=str(DEFAULT_HOMOSET_OBS))
    parser.add_argument("--homoset-molecule-csv", default=str(DEFAULT_HOMOSET_MOL))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_protocol_annotation/cv5_protocol_submodels_v1")
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--no-clean-only", dest="clean_only", action="store_false")
    parser.add_argument("--exclude-unknown-source-like", action="store_true")
    parser.add_argument("--protocol-groups", default="ready,inherent")
    parser.add_argument("--direct-percent-only", action="store_true", default=True)
    parser.add_argument("--allow-bound-percent", dest="direct_percent_only", action="store_false")
    parser.add_argument("--strict-protocol-guidelines", action="store_true", default=True)
    parser.add_argument("--allow-loose-guidelines", dest="strict_protocol_guidelines", action="store_false")
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
    parser.add_argument("--global-pairs-per-fold", type=int, default=50000)
    parser.add_argument("--protocol-pairs-per-fold", type=int, default=20000)
    parser.add_argument("--min-train-rows", type=int, default=20)
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
