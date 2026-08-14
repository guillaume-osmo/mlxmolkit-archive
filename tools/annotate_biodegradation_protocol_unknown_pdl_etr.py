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
DEFAULT_CAS_USABLE = Path(
    "benchmarks/biodegradation_homoset_audit/"
    "cas_source_biodegradation_csv_v1/biodegradation_cas_observations_usable_percent.csv"
)

PROTOCOL_CANDIDATES = {
    "ready": "OECD301F",
    "inherent": "OECD302B",
    "other": "OTHER",
    "simulation": "OECD303A",
}

PROTOCOL_FEATURE_PREFIXES = ("exp_", "guideline_")


def _bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    text = series.fillna(str(default)).astype(str).str.lower()
    return text.isin({"true", "1", "yes", "y"})


def load_clean_known(args: argparse.Namespace) -> pd.DataFrame:
    obs = pd.read_csv(args.homoset_observation_csv, low_memory=False)
    mol = pd.read_csv(args.homoset_molecule_csv, low_memory=False)
    mol_cols = [
        "canonical_smiles",
        "curation_status",
        "use_for_percent_training",
        "high_percent_conflict",
        "percent_range",
        "percent_std",
    ]
    obs = obs.merge(mol[[c for c in mol_cols if c in mol.columns]], on="canonical_smiles", how="left")
    obs["use_for_percent_training"] = _bool_series(obs["use_for_percent_training"], default=True)
    obs["high_percent_conflict"] = _bool_series(obs["high_percent_conflict"], default=False)
    known = obs[
        obs["canonical_smiles"].notna()
        & obs["y_percent"].notna()
        & obs["is_known_protocol"].astype(bool)
        & obs["protocol_group"].isin(PROTOCOL_CANDIDATES)
        & obs["guideline_norm"].notna()
    ].copy()
    if args.clean_only:
        known = known[known["use_for_percent_training"] & ~known["high_percent_conflict"]].copy()
    if args.exclude_unknown_source_like:
        known = known[known["source"].ne("biodegradation_cas_csv")].copy()
    known["y_percent"] = pd.to_numeric(known["y_percent"], errors="coerce").clip(0.0, 100.0)
    known["duration_days"] = pd.to_numeric(known["duration_days"], errors="coerce")
    known["guideline_norm"] = known["guideline_norm"].astype(str)
    known = known[known["y_percent"].notna()].reset_index(drop=True)
    return known


def maybe_add_cas_known(train: pd.DataFrame, cas: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.include_cas_known_training:
        return train
    cas_known = cas[
        cas["canonical_smiles"].notna()
        & cas["y_percent"].notna()
        & cas["protocol_group"].isin(PROTOCOL_CANDIDATES)
        & cas["guideline_norm"].notna()
    ].copy()
    if len(cas_known) == 0:
        return train
    cas_known["source"] = "biodegradation_cas_csv_known"
    cas_known["raw_smiles"] = cas_known["canonical_smiles"]
    cas_known["duration_days"] = pd.to_numeric(cas_known["duration_days"], errors="coerce")
    cas_known["y_percent"] = pd.to_numeric(cas_known["y_percent"], errors="coerce").clip(0.0, 100.0)
    keep_cols = sorted(set(train.columns).union(cas_known.columns))
    return pd.concat(
        [train.reindex(columns=keep_cols), cas_known.reindex(columns=keep_cols)],
        ignore_index=True,
    )


def load_unknown(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    cas = pd.read_csv(args.cas_usable_csv, low_memory=False)
    cas["y_percent"] = pd.to_numeric(cas["y_percent"], errors="coerce").clip(0.0, 100.0)
    cas["duration_days"] = pd.to_numeric(cas["duration_days"], errors="coerce")
    unknown = cas[
        cas["canonical_smiles"].notna()
        & cas["y_percent"].notna()
        & cas["protocol_group"].fillna("unknown").eq("unknown")
    ].copy()
    unknown["source"] = "biodegradation_cas_csv_unknown"
    unknown["raw_smiles"] = unknown["canonical_smiles"]
    return cas, unknown.reset_index(drop=True)


def candidate_rows(unknown: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for protocol_group, guideline in PROTOCOL_CANDIDATES.items():
        frame = unknown.copy()
        frame["candidate_protocol_group"] = protocol_group
        frame["guideline_norm"] = guideline
        frame["protocol_group"] = protocol_group
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out["_unknown_row_index"] = np.tile(np.arange(len(unknown), dtype=np.int64), len(PROTOCOL_CANDIDATES))
    return out


def choose_protocol(pred_matrix: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protocols = np.asarray(list(PROTOCOL_CANDIDATES), dtype=object)
    errors = np.abs(pred_matrix - observed[:, None])
    order = np.argsort(errors, axis=1)
    best_idx = order[:, 0]
    second_idx = order[:, 1] if pred_matrix.shape[1] > 1 else order[:, 0]
    best = protocols[best_idx]
    best_err = errors[np.arange(len(errors)), best_idx]
    margin = errors[np.arange(len(errors)), second_idx] - best_err
    return best, best_err.astype(np.float32), margin.astype(np.float32)


def force_protocol_features(selector: Any, prep: Any, feature_names: list[str]) -> tuple[Any, int]:
    prepared_source_idx = np.where(prep.finite_mask)[0][prep.keep_mask]
    prepared_names = [feature_names[int(i)] for i in prepared_source_idx]
    forced = np.asarray(
        [
            i
            for i, name in enumerate(prepared_names)
            if any(str(name).startswith(prefix) for prefix in PROTOCOL_FEATURE_PREFIXES)
        ],
        dtype=int,
    )
    if forced.size == 0:
        return selector, 0
    merged = np.unique(np.concatenate([selector.indices.astype(int), forced]))
    added = int(len(merged) - len(selector.indices))
    selector.indices = merged.astype(int)
    return selector, added


def score_ready_from_percent(protocol: pd.Series, y: pd.Series) -> pd.Series:
    yv = pd.to_numeric(y, errors="coerce")
    p = protocol.astype(str)
    out = pd.Series(np.nan, index=y.index, dtype="float64")
    out[p.eq("ready")] = (yv[p.eq("ready")] >= 60.0).astype(float)
    out[p.eq("inherent")] = (yv[p.eq("inherent")] >= 70.0).astype(float)
    return out


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cas, unknown = load_unknown(args)
    train = load_clean_known(args)
    train = maybe_add_cas_known(train, cas, args)
    if len(train) < 50:
        raise RuntimeError("not enough clean known protocol rows for PDL-ETR")
    if len(unknown) == 0:
        raise RuntimeError("no unknown CAS protocol rows to annotate")

    cand = candidate_rows(unknown)
    guideline_categories = sorted(set(train["guideline_norm"].astype(str)) | set(cand["guideline_norm"].astype(str)))
    all_df = pd.concat([train, cand.reindex(columns=train.columns.union(cand.columns))], ignore_index=True, sort=False)
    for col in ["canonical_smiles", "guideline_norm", "duration_days"]:
        if col not in all_df.columns:
            raise RuntimeError(f"missing required feature column {col!r}")

    x_raw, feature_names = build_features(
        all_df,
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
    n_train = len(train)
    x_train_raw = x_raw[:n_train]
    x_cand_raw = x_raw[n_train:]
    y_train = train["y_percent"].to_numpy(dtype=np.float32)

    x_train_prepared, prep = fit_feature_prep(x_train_raw, args.arcsinh_threshold)
    x_cand = apply_feature_prep(x_cand_raw, prep)
    x_train, selector = fit_feature_selector(
        x_train_prepared,
        y_train,
        method=args.select_method,
        k=args.select_k,
        seed=args.seed,
        jobs=args.jobs,
        trees=args.select_trees,
    )
    forced_protocol_features_added = 0
    if args.force_protocol_features:
        selector, forced_protocol_features_added = force_protocol_features(selector, prep, feature_names)
        x_train = apply_feature_selector(x_train_prepared, selector)
    x_cand = apply_feature_selector(x_cand, selector)

    pair_a, pair_b, dy = sample_pairs(
        y_train,
        n_pairs=args.pairs,
        min_abs_dy=args.min_abs_dy,
        seed=args.seed + 1000,
    )
    pair_raw = build_pair_features(
        x_train[pair_a],
        x_train[pair_b],
        y_train[pair_a],
        include_abs_delta=args.include_abs_delta,
    )
    pair_train, pair_prep = fit_feature_prep(pair_raw, args.arcsinh_threshold)
    pdl = make_regressor("etr", args, args.seed + 2000)
    pdl.fit(pair_train, dy)
    cand_pred = np.clip(
        predict_pdl(
            pdl,
            pair_prep,
            x_train,
            y_train,
            x_cand,
            anchors=args.anchors,
            include_abs_delta=args.include_abs_delta,
            aggregate=args.aggregate,
            batch_size=args.predict_batch_size,
        ),
        0.0,
        100.0,
    )

    n_unknown = len(unknown)
    protocols = list(PROTOCOL_CANDIDATES)
    pred_matrix = cand_pred.reshape(len(protocols), n_unknown).T
    observed = unknown["y_percent"].to_numpy(dtype=np.float32)
    best, best_error, margin = choose_protocol(pred_matrix, observed)

    annotated = unknown.copy()
    annotated["predicted_protocol_group"] = best
    annotated["best_abs_error_percent"] = best_error
    annotated["error_margin_percent"] = margin
    annotated["candidate_protocols"] = "|".join(protocols)
    annotated["ready_label_from_predicted_protocol"] = score_ready_from_percent(
        annotated["predicted_protocol_group"],
        annotated["y_percent"],
    )
    for j, protocol in enumerate(protocols):
        annotated[f"pdl_pred_percent_if_{protocol}"] = pred_matrix[:, j]
        annotated[f"abs_err_if_{protocol}"] = np.abs(pred_matrix[:, j] - observed)

    train_status = train[
        [
            c
            for c in [
                "source",
                "source_row",
                "canonical_smiles",
                "cas",
                "name",
                "protocol_group",
                "guideline_norm",
                "duration_days",
                "y_percent",
                "curation_status",
            ]
            if c in train.columns
        ]
    ].copy()

    annotated.to_csv(out_dir / "unknown_protocol_pdl_etr_annotations.csv", index=False)
    train_status.to_csv(out_dir / "clean_known_training_rows.csv", index=False)
    pd.Series([feature_names[i] for i in np.where(prep.finite_mask)[0][prep.keep_mask][selector.indices]], name="feature_name").to_csv(
        out_dir / "selected_features.csv",
        index=False,
    )
    summary = {
        "train_rows_clean_known": int(len(train)),
        "train_molecules": int(train["canonical_smiles"].nunique()),
        "unknown_rows_annotated": int(len(annotated)),
        "unknown_molecules": int(annotated["canonical_smiles"].nunique()),
        "candidate_protocols": PROTOCOL_CANDIDATES,
        "predicted_protocol_counts": annotated["predicted_protocol_group"].value_counts().to_dict(),
        "median_best_abs_error_percent": float(np.median(best_error)),
        "mean_best_abs_error_percent": float(np.mean(best_error)),
        "median_error_margin_percent": float(np.median(margin)),
        "selected_features": int(x_train.shape[1]),
        "forced_protocol_features_added": int(forced_protocol_features_added),
        "pairs": int(len(dy)),
        "anchors": int(args.anchors),
        "include_cas_known_training": bool(args.include_cas_known_training),
        "clean_only": bool(args.clean_only),
        "elapsed_seconds": float(time.time() - t0),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "run_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[biodeg-protocol-pdl] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate unknown CAS biodegradation protocols with clean-known PDL-ETR.")
    parser.add_argument("--homoset-observation-csv", default=str(DEFAULT_HOMOSET_OBS))
    parser.add_argument("--homoset-molecule-csv", default=str(DEFAULT_HOMOSET_MOL))
    parser.add_argument("--cas-usable-csv", default=str(DEFAULT_CAS_USABLE))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_protocol_annotation/cas_unknown_clean_known_pdl_etr_v1")
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--no-clean-only", dest="clean_only", action="store_false")
    parser.add_argument("--include-cas-known-training", action="store_true")
    parser.add_argument("--exclude-unknown-source-like", action="store_true")
    parser.add_argument("--molecular-feature-stack", choices=["osmo", "rdkit"], default="osmo")
    parser.add_argument("--feature-blocks", default="RDKIT217,OSMO,GOLD,ABRAHAM,FUNCGROUPS")
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--no-osmo-cache", action="store_true")
    parser.add_argument("--select-method", choices=["none", "rpcholesky", "variance", "f_regression", "mutual_info", "etr_importance"], default="rpcholesky")
    parser.add_argument("--select-k", type=int, default=768)
    parser.add_argument("--select-trees", type=int, default=50)
    parser.add_argument("--force-protocol-features", action="store_true", default=True)
    parser.add_argument("--no-force-protocol-features", dest="force_protocol_features", action="store_false")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=80000)
    parser.add_argument("--min-abs-dy", type=float, default=3.0)
    parser.add_argument("--include-abs-delta", action="store_true")
    parser.add_argument("--anchors", type=int, default=96)
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--predict-batch-size", type=int, default=64)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run(parse_args())
