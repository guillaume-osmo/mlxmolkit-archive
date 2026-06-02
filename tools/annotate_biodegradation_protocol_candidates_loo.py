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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut

from train_biodegradation_pdl_ordered import (
    apply_feature_prep,
    apply_feature_selector,
    experiment_feature_matrix,
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

PROTOCOLS = ("ready", "inherent", "other", "simulation")
PROTO_GUIDELINE = {
    "ready": "OECD301",
    "inherent": "OECD302",
    "simulation": "OECD303",
    "other": "OTHER",
}


def _float_col(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)


def _text_flag(df: pd.DataFrame, col: str, pattern: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return df[col].astype(str).str.contains(pattern, case=False, regex=True, na=False).to_numpy(dtype=np.float32)


def build_molecular_features(
    df: pd.DataFrame,
    *,
    n_bits: int,
    radius: int,
    molecular_feature_stack: str,
    feature_blocks: list[str],
    n_jobs: int,
    use_osmo_cache: bool,
) -> tuple[np.ndarray, list[str]]:
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

    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float32), []
    x = np.concatenate(blocks, axis=1).astype(np.float32)
    x = np.where(np.isinf(x), np.nan, x)
    return x, names


def apply_candidate_protocol(df: pd.DataFrame, protocol: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if protocol is None:
        proto = out["protocol_group"].astype(str).where(out["protocol_group"].isin(PROTOCOLS), "other")
    else:
        proto = pd.Series(protocol, index=out.index)
    out["candidate_protocol_group"] = proto.astype(str)
    out["guideline_norm"] = out["candidate_protocol_group"].map(PROTO_GUIDELINE).fillna("OTHER")
    return out


def make_candidate_rows(df: pd.DataFrame, protocols: tuple[str, ...] = PROTOCOLS) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[pd.DataFrame] = []
    proto_index: list[int] = []
    for i, (_, row) in enumerate(df.iterrows()):
        for protocol in protocols:
            one = row.to_frame().T
            one = apply_candidate_protocol(one, protocol)
            rows.append(one)
            proto_index.append(i)
    if not rows:
        return pd.DataFrame(columns=df.columns), np.zeros(0, dtype=np.int64)
    return pd.concat(rows, ignore_index=True), np.asarray(proto_index, dtype=np.int64)


def build_context_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    exp, exp_names = experiment_feature_matrix(df)
    group = df["candidate_protocol_group"].astype(str)
    group_x = np.stack([(group == p).to_numpy(dtype=np.float32) for p in PROTOCOLS], axis=1)
    group_names = [f"candidate_protocol_{p}" for p in PROTOCOLS]

    duration = _float_col(df, "duration_days")
    mw = _float_col(df, "molecular_weight")
    year = _float_col(df, "year")
    duration_filled = np.nan_to_num(duration, nan=28.0)
    mw_filled = np.nan_to_num(mw, nan=np.nanmedian(mw[np.isfinite(mw)]) if np.isfinite(mw).any() else 0.0)
    year_filled = np.nan_to_num(year, nan=0.0)
    obs = np.stack(
        [
            duration_filled,
            np.log1p(np.clip(duration_filled, 0.0, None)),
            duration_filled / 28.0,
            np.clip(duration_filled / 28.0, 0.0, 1.0),
            np.maximum(duration_filled - 28.0, 0.0) / 28.0,
            np.isfinite(duration).astype(np.float32),
            mw_filled,
            year_filled,
            np.isfinite(year).astype(np.float32),
            _text_flag(df, "type", r"aerobic"),
            _text_flag(df, "type", r"anaerobic"),
            _text_flag(df, "glp", r"yes|true|y"),
            _text_flag(df, "reliability", r"1|reliable"),
        ],
        axis=1,
    ).astype(np.float32)
    obs_names = [
        "obs_duration_days",
        "obs_log1p_duration_days",
        "obs_duration_over_28",
        "obs_duration_progress_0_28",
        "obs_duration_after_28",
        "obs_duration_present",
        "obs_molecular_weight",
        "obs_year",
        "obs_year_present",
        "obs_type_aerobic",
        "obs_type_anaerobic",
        "obs_glp_positive",
        "obs_reliability_positive",
    ]
    x = np.concatenate([exp, group_x, obs], axis=1).astype(np.float32)
    return x, exp_names + group_names + obs_names


def combine_features(x_mol: np.ndarray, context_df: pd.DataFrame, row_index: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x_ctx, ctx_names = build_context_features(context_df)
    x = np.concatenate([x_mol[row_index], x_ctx], axis=1).astype(np.float32)
    return x, ctx_names


def make_regressor(args: argparse.Namespace, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.trees,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        random_state=seed,
        n_jobs=args.jobs,
    )


def reshape_candidate_predictions(pred: np.ndarray, n_rows: int) -> np.ndarray:
    return np.asarray(pred, dtype=np.float32).reshape(n_rows, len(PROTOCOLS))


def choose_protocol(pred_matrix: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = np.abs(pred_matrix - observed[:, None])
    order = np.argsort(errors, axis=1)
    best_idx = order[:, 0]
    second_idx = order[:, 1] if len(PROTOCOLS) > 1 else order[:, 0]
    labels = np.asarray(PROTOCOLS, dtype=object)[best_idx]
    best_error = errors[np.arange(len(errors)), best_idx]
    margin = errors[np.arange(len(errors)), second_idx] - best_error
    return labels, best_error.astype(np.float32), margin.astype(np.float32)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df[df["canonical_smiles"].notna() & df["y_percent"].notna()].copy()
    df["protocol_group"] = df["protocol_group"].fillna("unknown").astype(str)
    df["y_percent"] = pd.to_numeric(df["y_percent"], errors="coerce").clip(0.0, 100.0)
    df = df[df["y_percent"].notna()].reset_index(drop=True)

    known = df[df["protocol_group"].isin(PROTOCOLS)].copy().reset_index(drop=True)
    unknown = df[df["protocol_group"].eq("unknown")].copy().reset_index(drop=True)
    if len(known) < 10:
        raise RuntimeError("not enough known-protocol rows")

    feature_blocks = [b.strip().upper() for b in args.feature_blocks.split(",") if b.strip()]
    base = pd.concat([known, unknown], ignore_index=True)
    x_mol, mol_names = build_molecular_features(
        base,
        n_bits=args.n_bits,
        radius=args.radius,
        molecular_feature_stack=args.molecular_feature_stack,
        feature_blocks=feature_blocks,
        n_jobs=args.jobs,
        use_osmo_cache=not args.no_osmo_cache,
    )

    known_actual = apply_candidate_protocol(known, None)
    x_actual_raw, ctx_names = combine_features(x_mol, known_actual, np.arange(len(known), dtype=np.int64))
    y = known["y_percent"].to_numpy(dtype=np.float32)

    known_candidates, known_candidate_base_idx = make_candidate_rows(known)
    x_known_candidates_raw, _ = combine_features(x_mol, known_candidates, known_candidate_base_idx)

    unknown_candidates, unknown_candidate_base_idx = make_candidate_rows(unknown)
    if len(unknown):
        x_unknown_candidates_raw, _ = combine_features(
            x_mol,
            unknown_candidates,
            unknown_candidate_base_idx + len(known),
        )
    else:
        x_unknown_candidates_raw = np.zeros((0, x_actual_raw.shape[1]), dtype=np.float32)

    feature_names = mol_names + ctx_names
    x_actual, prep = fit_feature_prep(x_actual_raw, args.arcsinh_threshold)
    x_known_candidates = apply_feature_prep(x_known_candidates_raw, prep)
    x_unknown_candidates = apply_feature_prep(x_unknown_candidates_raw, prep) if len(unknown) else x_unknown_candidates_raw

    x_actual_sel, selector = fit_feature_selector(
        x_actual,
        y,
        method=args.feature_select,
        k=args.select_k,
        seed=args.seed,
        jobs=args.jobs,
        trees=max(20, min(args.trees, 80)),
    )
    x_known_candidates_sel = apply_feature_selector(x_known_candidates, selector)
    x_unknown_candidates_sel = (
        apply_feature_selector(x_unknown_candidates, selector) if len(unknown) else x_unknown_candidates
    )
    kept_source_idx = np.where(prep.finite_mask)[0][prep.keep_mask][selector.indices]
    kept_names = [feature_names[int(i)] for i in kept_source_idx]

    print(
        json.dumps(
            {
                "input_rows": int(len(df)),
                "known_rows": int(len(known)),
                "unknown_rows": int(len(unknown)),
                "known_class_counts": known["protocol_group"].value_counts().to_dict(),
                "raw_features": int(x_actual_raw.shape[1]),
                "prepared_features": int(x_actual.shape[1]),
                "selected_features": int(x_actual_sel.shape[1]),
                "protocols": list(PROTOCOLS),
            },
            indent=2,
        ),
        flush=True,
    )

    loo_pred_matrix = np.zeros((len(known), len(PROTOCOLS)), dtype=np.float32)
    loo_true_protocol_pred = np.zeros(len(known), dtype=np.float32)
    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(x_actual_sel), start=1):
        i = int(test_idx[0])
        model = make_regressor(args, args.seed + fold)
        model.fit(x_actual_sel[train_idx], y[train_idx])
        start = i * len(PROTOCOLS)
        stop = start + len(PROTOCOLS)
        preds = np.clip(model.predict(x_known_candidates_sel[start:stop]), 0.0, 100.0)
        loo_pred_matrix[i] = preds.astype(np.float32)
        true_j = PROTOCOLS.index(str(known.loc[i, "protocol_group"]))
        loo_true_protocol_pred[i] = float(preds[true_j])
        if fold % args.progress_every == 0 or fold == len(known):
            print(f"[protocol-candidate-loo] fold {fold}/{len(known)} elapsed={time.time() - t0:.1f}s", flush=True)

    loo_best, loo_best_error, loo_margin = choose_protocol(loo_pred_matrix, y)
    known_out = known.copy()
    known_out["loo_predicted_protocol_group"] = loo_best
    known_out["loo_correct"] = known_out["protocol_group"].to_numpy(dtype=object) == loo_best
    known_out["loo_best_abs_error_percent"] = loo_best_error
    known_out["loo_error_margin_percent"] = loo_margin
    known_out["loo_true_protocol_pred_percent"] = loo_true_protocol_pred
    known_out["loo_true_protocol_abs_error_percent"] = np.abs(loo_true_protocol_pred - y)
    for j, protocol in enumerate(PROTOCOLS):
        known_out[f"pred_y_if_{protocol}"] = loo_pred_matrix[:, j]
        known_out[f"abs_err_if_{protocol}"] = np.abs(loo_pred_matrix[:, j] - y)

    model = make_regressor(args, args.seed)
    model.fit(x_actual_sel, y)
    if len(unknown):
        unknown_pred_matrix = reshape_candidate_predictions(
            np.clip(model.predict(x_unknown_candidates_sel), 0.0, 100.0),
            len(unknown),
        )
        unknown_y = unknown["y_percent"].to_numpy(dtype=np.float32)
        unknown_best, unknown_best_error, unknown_margin = choose_protocol(unknown_pred_matrix, unknown_y)
    else:
        unknown_pred_matrix = np.zeros((0, len(PROTOCOLS)), dtype=np.float32)
        unknown_best = np.asarray([], dtype=object)
        unknown_best_error = np.asarray([], dtype=np.float32)
        unknown_margin = np.asarray([], dtype=np.float32)

    unknown_out = unknown.copy()
    unknown_out["predicted_protocol_group"] = unknown_best
    unknown_out["best_abs_error_percent"] = unknown_best_error
    unknown_out["error_margin_percent"] = unknown_margin
    for j, protocol in enumerate(PROTOCOLS):
        unknown_out[f"pred_y_if_{protocol}"] = unknown_pred_matrix[:, j] if len(unknown) else []
        unknown_out[f"abs_err_if_{protocol}"] = (
            np.abs(unknown_pred_matrix[:, j] - unknown["y_percent"].to_numpy(dtype=np.float32)) if len(unknown) else []
        )

    true_labels = known["protocol_group"].astype(str).to_numpy()
    metrics = {
        "accuracy": float(np.mean(known_out["loo_correct"])),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, loo_best)),
        "macro_f1": float(f1_score(true_labels, loo_best, average="macro")),
        "weighted_f1": float(f1_score(true_labels, loo_best, average="weighted")),
        "true_protocol_mae_percent": float(mean_absolute_error(y, loo_true_protocol_pred)),
        "chosen_protocol_mae_percent": float(np.mean(loo_best_error)),
        "median_error_margin_percent": float(np.median(loo_margin)),
        "n_known": int(len(known)),
        "n_unknown": int(len(unknown)),
    }

    labels = list(PROTOCOLS)
    cm = pd.DataFrame(
        confusion_matrix(true_labels, loo_best, labels=labels),
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    by_group = (
        known_out.groupby("protocol_group", dropna=False)
        .agg(
            n=("protocol_group", "size"),
            accuracy=("loo_correct", "mean"),
            true_protocol_mae_percent=("loo_true_protocol_abs_error_percent", "mean"),
            chosen_protocol_mae_percent=("loo_best_abs_error_percent", "mean"),
            median_error_margin_percent=("loo_error_margin_percent", "median"),
        )
        .reset_index()
    )

    known_out.to_csv(out_dir / "known_protocol_loo_candidate_predictions.csv", index=False)
    unknown_out.to_csv(out_dir / "unknown_protocol_candidate_annotations.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "protocol_candidate_loo_metrics.csv", index=False)
    cm.to_csv(out_dir / "protocol_candidate_loo_confusion_matrix.csv")
    by_group.to_csv(out_dir / "protocol_candidate_loo_by_group.csv", index=False)
    pd.Series(kept_names, name="feature_name").to_csv(out_dir / "selected_features.csv", index=False)

    report = {
        **metrics,
        "input_csv": str(Path(args.input_csv)),
        "out_dir": str(out_dir),
        "known_class_counts": known["protocol_group"].value_counts().to_dict(),
        "unknown_predicted_counts": unknown_out["predicted_protocol_group"].value_counts().to_dict()
        if len(unknown_out)
        else {},
        "feature_select": args.feature_select,
        "selected_features": int(x_actual_sel.shape[1]),
        "protocols": list(PROTOCOLS),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[protocol-candidate-loo] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOO ETR counterfactual protocol annotation for CAS biodegradation rows.")
    parser.add_argument("--input-csv", default=str(DEFAULT_CAS_USABLE))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_protocol_annotation/cas_source_candidate_loo_etr_v1")
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
