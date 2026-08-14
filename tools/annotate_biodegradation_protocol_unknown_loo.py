#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut

from train_biodegradation_pdl_ordered import (
    apply_feature_prep,
    apply_feature_selector,
    fit_feature_prep,
    fit_feature_selector,
    morgan_count_matrix,
    osmo_feature_matrix,
)


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


DEFAULT_CAS_USABLE = Path(
    "benchmarks/biodegradation_homoset_audit/"
    "cas_source_biodegradation_csv_v1/biodegradation_cas_observations_usable_percent.csv"
)

KNOWN_GROUPS = ("ready", "inherent", "other", "simulation")


def _to_float_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _text_flag(df: pd.DataFrame, col: str, pattern: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return df[col].astype(str).str.contains(pattern, case=False, regex=True, na=False).to_numpy(dtype=np.float32)


def build_protocol_features(
    df: pd.DataFrame,
    *,
    n_bits: int,
    radius: int,
    molecular_feature_stack: str,
    feature_blocks: list[str],
    n_jobs: int,
    use_osmo_cache: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build protocol-inference features without method/guideline leakage."""

    smiles = df["canonical_smiles"].astype(str).tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        bad = [s for s, m in zip(smiles, mols) if m is None][:5]
        raise RuntimeError(f"canonical SMILES failed RDKit parsing, examples={bad}")

    blocks: list[np.ndarray] = []
    names: list[str] = []

    if molecular_feature_stack == "osmo":
        x_osmo, osmo_names = osmo_feature_matrix(
            smiles,
            blocks=feature_blocks,
            n_jobs=n_jobs,
            use_cache=use_osmo_cache,
        )
        blocks.append(x_osmo)
        names.extend(osmo_names)
    elif molecular_feature_stack == "none":
        pass
    else:
        raise ValueError(f"unknown molecular_feature_stack={molecular_feature_stack!r}")

    if n_bits > 0:
        blocks.append(morgan_count_matrix(mols, n_bits=n_bits, radius=radius))
        names.extend([f"morgan_count_r{radius}_{i:04d}" for i in range(n_bits)])

    y = _to_float_series(df, "y_percent").to_numpy(dtype=np.float32)
    duration = _to_float_series(df, "duration_days").to_numpy(dtype=np.float32)
    mw = _to_float_series(df, "molecular_weight").to_numpy(dtype=np.float32)
    year = _to_float_series(df, "year").to_numpy(dtype=np.float32)
    ready_result = _to_float_series(df, "ready_label_from_result").to_numpy(dtype=np.float32)

    duration_filled = np.nan_to_num(duration, nan=28.0)
    y_filled = np.nan_to_num(y, nan=0.0)
    obs = np.stack(
        [
            y_filled,
            y_filled / 100.0,
            np.sqrt(np.clip(y_filled, 0.0, 100.0) / 100.0),
            duration_filled,
            np.log1p(np.clip(duration_filled, 0.0, None)),
            duration_filled / 28.0,
            np.clip(duration_filled / 28.0, 0.0, 1.0),
            np.maximum(duration_filled - 28.0, 0.0) / 28.0,
            np.isfinite(duration).astype(np.float32),
            np.nan_to_num(mw, nan=np.nanmedian(mw[np.isfinite(mw)]) if np.isfinite(mw).any() else 0.0),
            np.nan_to_num(year, nan=0.0),
            np.isfinite(year).astype(np.float32),
            np.nan_to_num(ready_result, nan=-1.0),
            np.isfinite(ready_result).astype(np.float32),
            _text_flag(df, "result", r"readily biodegradable"),
            _text_flag(df, "result", r"no biodegradation|not biodegradable"),
            _text_flag(df, "type", r"aerobic"),
            _text_flag(df, "type", r"anaerobic"),
            _text_flag(df, "glp", r"yes|true|y"),
            _text_flag(df, "reliability", r"1|reliable"),
        ],
        axis=1,
    ).astype(np.float32)
    blocks.append(obs)
    names.extend(
        [
            "obs_y_percent",
            "obs_y_fraction",
            "obs_sqrt_y_fraction",
            "obs_duration_days",
            "obs_log1p_duration_days",
            "obs_duration_over_28",
            "obs_duration_progress_0_28",
            "obs_duration_after_28",
            "obs_duration_present",
            "obs_molecular_weight",
            "obs_year",
            "obs_year_present",
            "obs_ready_label_from_result",
            "obs_ready_label_present",
            "obs_result_ready_text",
            "obs_result_not_biodeg_text",
            "obs_type_aerobic",
            "obs_type_anaerobic",
            "obs_glp_positive",
            "obs_reliability_positive",
        ]
    )

    y_source = pd.get_dummies(df.get("y_percent_source", pd.Series("", index=df.index)).astype(str), prefix="ysrc")
    if len(y_source.columns):
        blocks.append(y_source.to_numpy(dtype=np.float32))
        names.extend(list(y_source.columns))

    x = np.concatenate(blocks, axis=1).astype(np.float32)
    x = np.where(np.isinf(x), np.nan, x)
    return x, names


def make_classifier(args: argparse.Namespace, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.trees,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        bootstrap=False,
        random_state=seed,
        n_jobs=args.jobs,
    )


def loo_predict(args: argparse.Namespace, x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    pred = np.empty(n, dtype=object)
    proba = np.zeros((n, len(labels)), dtype=np.float32)
    loo = LeaveOneOut()
    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(loo.split(x), start=1):
        model = make_classifier(args, args.seed + fold)
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx[0]] = model.predict(x[test_idx])[0]
        local = model.predict_proba(x[test_idx])[0]
        for cls, p in zip(model.classes_, local):
            proba[test_idx[0], int(np.where(labels == cls)[0][0])] = float(p)
        if fold % args.progress_every == 0 or fold == n:
            elapsed = time.time() - t0
            print(f"[protocol-loo] fold {fold}/{n} elapsed={elapsed:.1f}s", flush=True)
    return pred, proba


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df[df["canonical_smiles"].notna() & df["y_percent"].notna()].copy()
    df["protocol_group"] = df["protocol_group"].fillna("unknown").astype(str)

    known = df[df["protocol_group"].isin(KNOWN_GROUPS)].copy().reset_index(drop=True)
    unknown = df[df["protocol_group"].eq("unknown")].copy().reset_index(drop=True)
    if len(known) < 10:
        raise RuntimeError("not enough known-protocol rows for LOO annotation")

    feature_blocks = [b.strip().upper() for b in args.feature_blocks.split(",") if b.strip()]
    all_rows = pd.concat([known, unknown], axis=0, ignore_index=True)
    x_raw, feature_names = build_protocol_features(
        all_rows,
        n_bits=args.n_bits,
        radius=args.radius,
        molecular_feature_stack=args.molecular_feature_stack,
        feature_blocks=feature_blocks,
        n_jobs=args.jobs,
        use_osmo_cache=not args.no_osmo_cache,
    )
    x_known_raw = x_raw[: len(known)]
    x_unknown_raw = x_raw[len(known) :]

    x_known, prep = fit_feature_prep(x_known_raw, args.arcsinh_threshold)
    x_unknown = apply_feature_prep(x_unknown_raw, prep)

    y = known["protocol_group"].to_numpy(dtype=object)
    x_known_sel, selector = fit_feature_selector(
        x_known,
        np.arange(len(y), dtype=np.float32),
        method=args.feature_select,
        k=args.select_k,
        seed=args.seed,
        jobs=args.jobs,
        trees=max(20, min(args.trees, 80)),
    )
    x_unknown_sel = apply_feature_selector(x_unknown, selector)
    kept_names = [feature_names[np.where(prep.finite_mask)[0][prep.keep_mask][i]] for i in selector.indices]

    labels = np.array(sorted(pd.unique(y)), dtype=object)
    print(
        json.dumps(
            {
                "input_rows": int(len(df)),
                "known_rows": int(len(known)),
                "unknown_rows": int(len(unknown)),
                "known_class_counts": known["protocol_group"].value_counts().to_dict(),
                "raw_features": int(x_raw.shape[1]),
                "prepared_features": int(x_known.shape[1]),
                "selected_features": int(x_known_sel.shape[1]),
                "labels": labels.tolist(),
            },
            indent=2,
        ),
        flush=True,
    )

    loo_pred, loo_proba = loo_predict(args, x_known_sel, y, labels)
    loo = known.copy()
    loo["loo_predicted_protocol_group"] = loo_pred
    loo["loo_correct"] = loo["protocol_group"].to_numpy(dtype=object) == loo_pred
    for j, label in enumerate(labels):
        loo[f"loo_p_{label}"] = loo_proba[:, j]
    loo["loo_confidence"] = loo[[f"loo_p_{label}" for label in labels]].max(axis=1)

    metrics_rows = []
    metrics = {
        "accuracy": float(np.mean(loo["loo_correct"])),
        "balanced_accuracy": float(balanced_accuracy_score(y, loo_pred)),
        "macro_f1": float(f1_score(y, loo_pred, average="macro")),
        "weighted_f1": float(f1_score(y, loo_pred, average="weighted")),
        "n_known": int(len(known)),
        "n_unknown": int(len(unknown)),
    }
    metrics_rows.append({"split": "loo_known_protocol", **metrics})

    cm = pd.DataFrame(
        confusion_matrix(y, loo_pred, labels=labels),
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )

    final_model = make_classifier(args, args.seed)
    final_model.fit(x_known_sel, y)
    unknown_proba = final_model.predict_proba(x_unknown_sel) if len(unknown) else np.zeros((0, len(final_model.classes_)))
    unknown_pred = final_model.predict(x_unknown_sel) if len(unknown) else np.array([], dtype=object)

    annotated = unknown.copy()
    annotated["predicted_protocol_group"] = unknown_pred
    for label in labels:
        annotated[f"p_{label}"] = 0.0
    for j, cls in enumerate(final_model.classes_):
        annotated[f"p_{cls}"] = unknown_proba[:, j] if len(unknown) else []
    p_cols = [f"p_{label}" for label in labels]
    annotated["protocol_confidence"] = annotated[p_cols].max(axis=1) if len(annotated) else []
    annotated["protocol_margin"] = (
        np.sort(annotated[p_cols].to_numpy(dtype=np.float32), axis=1)[:, -1]
        - np.sort(annotated[p_cols].to_numpy(dtype=np.float32), axis=1)[:, -2]
        if len(annotated) and len(p_cols) > 1
        else np.nan
    )

    loo.to_csv(out_dir / "known_protocol_loo_predictions.csv", index=False)
    annotated.to_csv(out_dir / "unknown_protocol_annotations.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "protocol_loo_metrics.csv", index=False)
    cm.to_csv(out_dir / "protocol_loo_confusion_matrix.csv")
    pd.Series(kept_names, name="feature_name").to_csv(out_dir / "selected_features.csv", index=False)

    report = {
        **metrics,
        "input_csv": str(Path(args.input_csv)),
        "out_dir": str(out_dir),
        "known_class_counts": known["protocol_group"].value_counts().to_dict(),
        "unknown_predicted_counts": annotated["predicted_protocol_group"].value_counts().to_dict()
        if len(annotated)
        else {},
        "mean_unknown_confidence": float(annotated["protocol_confidence"].mean()) if len(annotated) else None,
        "feature_select": args.feature_select,
        "selected_features": int(x_known_sel.shape[1]),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[protocol-loo] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOO ExtraTrees annotation for unknown biodegradation protocol rows.")
    parser.add_argument("--input-csv", default=str(DEFAULT_CAS_USABLE))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_protocol_annotation/cas_source_loo_etr_v1")
    parser.add_argument("--molecular-feature-stack", choices=["osmo", "none"], default="osmo")
    parser.add_argument("--feature-blocks", default="RDKIT217,OSMO,GOLD,ABRAHAM,FUNCGROUPS")
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--no-osmo-cache", action="store_true")
    parser.add_argument("--feature-select", choices=["none", "rpcholesky", "variance"], default="rpcholesky")
    parser.add_argument("--select-k", type=int, default=768)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run(parse_args())
