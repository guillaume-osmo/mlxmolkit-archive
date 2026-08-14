#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from bridge_biodegradation_pdl_to_2308 import choose_threshold
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
    normalize_guideline,
    predict_pdl,
    sample_pairs,
)


DEFAULT_EXCELRA = Path("/Users/guillaume-osmo/Github/data/Biodegradation-cleaned-dataset_Excelra_240630.csv")
DEFAULT_HOMOSET_CURATED = Path(
    "benchmarks/biodegradation_homoset_audit/rifm_excelra_2308_v3_curated_targets/biodegradation_molecule_curated_targets.csv"
)
DEFAULT_HOMOSET_OBSERVATIONS = Path(
    "benchmarks/biodegradation_homoset_audit/rifm_excelra_2308_v3_curated_targets/biodegradation_observation_homoset.csv"
)


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\xa0", " ").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group(0)) if match else np.nan


def load_excelra(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"smiles", "label", "perc_biodeg", "num_days", "compliance"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df.copy()
    out["canonical_smiles"] = out["smiles"].map(canonical_smiles)
    out["y_percent_direct"] = out["perc_biodeg"].map(parse_number)
    lower = out["perc_biodeg_lower"].map(parse_number) if "perc_biodeg_lower" in out.columns else np.nan
    upper = out["perc_biodeg_upper"].map(parse_number) if "perc_biodeg_upper" in out.columns else np.nan
    midpoint = (pd.Series(lower, index=out.index) + pd.Series(upper, index=out.index)) / 2.0
    out["y_percent"] = out["y_percent_direct"].where(out["y_percent_direct"].notna(), midpoint)
    out["y_percent_source"] = np.where(out["y_percent_direct"].notna(), "direct", "bound_midpoint")
    out.loc[out["y_percent"].isna(), "y_percent_source"] = "missing"
    out["y_percent"] = pd.to_numeric(out["y_percent"], errors="coerce").clip(0.0, 100.0)

    out["duration_days"] = out["num_days"].map(parse_number)
    out["guideline_norm"] = out["compliance"].map(normalize_guideline)
    out["SMILES"] = out["canonical_smiles"]
    out["Test guideline"] = out["compliance"].fillna("UNKNOWN")
    out["Duration"] = out["duration_days"].map(lambda x: f"{x:g} days" if np.isfinite(x) else np.nan)
    out["Unit"] = "%"
    out["label_norm"] = out["label"].astype(str).str.strip().str.lower()

    # Clean ready/non-ready benchmark: inherent/ultimate labels are not the
    # same decision boundary as OECD ready biodegradability, so keep them for
    # ordinal diagnostics but exclude them from this binary control.
    ready_clean = out["label_norm"].map(
        {
            "readily biodegradable": 1.0,
            "non-biodegradable": 0.0,
            "poorly biodegradable": 0.0,
        }
    )
    out["y_ready_clean"] = ready_clean
    out["y_ready_strict"] = out["label_norm"].map(
        {
            "readily biodegradable": 1.0,
            "ultimately biodegradable": 0.0,
            "inherently biodegradable": 0.0,
            "poorly biodegradable": 0.0,
            "non-biodegradable": 0.0,
        }
    )
    out["y_any_biodeg"] = out["label_norm"].map(
        {
            "readily biodegradable": 1.0,
            "ultimately biodegradable": 1.0,
            "inherently biodegradable": 1.0,
            "poorly biodegradable": 0.0,
            "non-biodegradable": 0.0,
        }
    )
    out["y_order"] = out["label_norm"].map(
        {
            "non-biodegradable": 0.0,
            "poorly biodegradable": 0.0,
            "inherently biodegradable": 1.0,
            "ultimately biodegradable": 2.0,
            "readily biodegradable": 2.0,
        }
    )

    out = out[out["canonical_smiles"].notna()].reset_index(drop=True)
    return out


def ready_thresholds(guideline_norm: pd.Series) -> np.ndarray:
    text = guideline_norm.astype(str).str.upper()
    doc_like = text.str.contains(r"301A|301E|C\.4A|DOC", regex=True)
    return np.where(doc_like.to_numpy(), 70.0, 60.0).astype(np.float32)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    finite = np.isfinite(y) & np.isfinite(pred)
    y = y[finite]
    pred = pred[finite]
    row = {
        "n": int(len(y)),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "spearman": np.nan,
    }
    if len(y) == 0:
        return row
    row["mae"] = float(mean_absolute_error(y, pred))
    row["rmse"] = float(mean_squared_error(y, pred, squared=False))
    row["r2"] = float(r2_score(y, pred)) if len(y) >= 2 else np.nan
    row["spearman"] = float(spearmanr(y, pred).statistic) if len(np.unique(y)) >= 2 else np.nan
    return row


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float | np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    if np.ndim(threshold) == 0:
        threshold_arr = np.full_like(score, float(threshold), dtype=np.float32)
        threshold_value = float(threshold)
    else:
        threshold_arr = np.asarray(threshold, dtype=np.float32)
        threshold_value = float(np.nanmean(threshold_arr))

    finite = np.isfinite(y) & np.isfinite(score) & np.isfinite(threshold_arr)
    y = y[finite].astype(int)
    score = score[finite]
    threshold_arr = threshold_arr[finite]
    pred = (score >= threshold_arr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    row = {
        "n": int(len(y)),
        "threshold": threshold_value,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "mcc": np.nan,
        "roc_auc": np.nan,
    }
    if len(y) == 0:
        return row
    row.update(
        {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "mcc": float(matthews_corrcoef(y, pred)),
        }
    )
    if len(np.unique(y)) == 2:
        row["roc_auc"] = float(roc_auc_score(y, score))
    return row


def order_metrics(y_order: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y_order, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    finite = np.isfinite(y) & np.isfinite(score)
    y = y[finite]
    score = score[finite]
    return {
        "n": int(len(y)),
        "spearman_order": float(spearmanr(y, score).statistic) if len(np.unique(y)) >= 2 else np.nan,
    }


def build_anchor_quality_weights(rifm: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray | None, dict]:
    path_text = str(args.anchor_quality_csv).strip()
    if not path_text:
        return None, {"enabled": False}

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"anchor quality CSV not found: {path}")

    quality = pd.read_csv(path)
    if "canonical_smiles" not in quality.columns or "curation_status" not in quality.columns:
        raise ValueError(f"{path} must contain canonical_smiles and curation_status columns")

    status_map = quality.set_index("canonical_smiles")["curation_status"].astype(str).to_dict()
    high_conflict_map = (
        quality.set_index("canonical_smiles")["high_percent_conflict"].astype(bool).to_dict()
        if "high_percent_conflict" in quality.columns
        else {}
    )
    weight_by_status = {
        "single_source": float(args.anchor_single_source_weight),
        "multi_source_agree": float(args.anchor_multi_source_weight),
        "conflict": float(args.anchor_conflict_weight),
        "no_target": float(args.anchor_no_target_weight),
    }

    statuses = rifm["canonical_smiles"].astype(str).map(status_map).fillna("missing")
    weights = statuses.map(lambda s: weight_by_status.get(str(s), float(args.anchor_missing_quality_weight))).to_numpy(
        dtype=np.float32
    )
    if high_conflict_map:
        high = rifm["canonical_smiles"].astype(str).map(high_conflict_map).fillna(False).to_numpy(dtype=bool)
        weights[high] = np.minimum(weights[high], float(args.anchor_high_percent_conflict_weight))

    obs_path_text = str(args.anchor_observation_homoset_csv).strip()
    obs_summary: dict[str, int | float | str | bool] = {"enabled": False}
    if obs_path_text and "rifm_source_row" in rifm.columns:
        obs_path = Path(obs_path_text)
        if obs_path.exists():
            obs = pd.read_csv(obs_path, low_memory=False)
            needed = {"source", "source_row", "canonical_smiles", "protocol_group", "duration_bucket", "y_percent"}
            if needed.issubset(obs.columns):
                rifm_obs = obs[obs["source"].eq("rifm2026")].copy()
                excelra_obs = obs[obs["source"].eq("excelra")].copy()
                key = ["canonical_smiles", "protocol_group", "duration_bucket"]
                excelra_exact = (
                    excelra_obs.dropna(subset=["y_percent"])
                    .groupby(key, dropna=False)["y_percent"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "excelra_exact_percent", "count": "excelra_exact_n"})
                    .reset_index()
                )
                rifm_exact = rifm_obs[["source_row", *key, "y_percent"]].merge(excelra_exact, on=key, how="left")
                exact_map = rifm_exact.set_index("source_row")
                source_rows = rifm["rifm_source_row"].astype(int).to_numpy()
                aligned = exact_map.reindex(source_rows)
                exact_percent = aligned["excelra_exact_percent"].to_numpy(dtype=np.float32)
                exact_n = aligned["excelra_exact_n"].fillna(0).to_numpy(dtype=np.float32)
                rifm_percent = rifm["y_percent"].to_numpy(dtype=np.float32)
                has_exact = np.isfinite(exact_percent) & (exact_n > 0)
                delta = np.abs(rifm_percent - exact_percent)
                exact_agree = has_exact & (delta <= float(args.anchor_observation_agree_percent))
                exact_conflict = has_exact & (delta >= float(args.anchor_observation_conflict_percent))
                weights[exact_agree] = np.maximum(weights[exact_agree], float(args.anchor_observation_agree_weight))
                weights[exact_conflict] = np.minimum(
                    weights[exact_conflict], float(args.anchor_observation_conflict_weight)
                )
                obs_summary = {
                    "enabled": True,
                    "path": str(obs_path),
                    "n_exact_protocol_duration_overlap": int(has_exact.sum()),
                    "n_exact_agree": int(exact_agree.sum()),
                    "n_exact_conflict": int(exact_conflict.sum()),
                    "agree_percent_threshold": float(args.anchor_observation_agree_percent),
                    "conflict_percent_threshold": float(args.anchor_observation_conflict_percent),
                    "agree_weight": float(args.anchor_observation_agree_weight),
                    "conflict_weight": float(args.anchor_observation_conflict_weight),
                }
            else:
                obs_summary = {"enabled": False, "path": str(obs_path), "reason": "missing required columns"}
        else:
            obs_summary = {"enabled": False, "path": str(obs_path), "reason": "file not found"}

    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0).astype(np.float32)
    summary = {
        "enabled": True,
        "path": str(path),
        "counts_by_status": statuses.value_counts(dropna=False).to_dict(),
        "weight_min": float(np.min(weights)) if len(weights) else np.nan,
        "weight_max": float(np.max(weights)) if len(weights) else np.nan,
        "weight_mean": float(np.mean(weights)) if len(weights) else np.nan,
        "n_zero_weight": int(np.sum(weights <= 0.0)),
        "anchor_candidate_factor": int(args.anchor_candidate_factor),
        "anchor_quality_power": float(args.anchor_quality_power),
        "weight_by_status": weight_by_status,
        "observation_quality": obs_summary,
    }
    return weights, summary


def molecule_level(pred: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "smiles": "first",
        "label": "first",
        "label_norm": "first",
        "guideline_norm": "first",
        "known_protocol": "max",
        "seen_in_rifm_train": "max",
        "duration_days": "median",
        "y_percent": "mean",
        "direct_percent_score": "mean",
        "pdl_percent_score": "mean",
        "y_ready_clean": "max",
        "y_ready_strict": "max",
        "y_any_biodeg": "max",
        "y_order": "max",
        "protocol_ready_threshold": "median",
    }
    return pred.groupby("canonical_smiles", as_index=False).agg(agg)


def train_rifm_models(args: argparse.Namespace):
    rifm = load_rifm(Path(args.rifm_xlsx), clip_target=True)
    anchor_weights, anchor_weight_summary = build_anchor_quality_weights(rifm, args)
    guideline_categories = sorted(rifm["guideline_norm"].astype(str).unique().tolist())
    feature_blocks = [b.strip().upper() for b in args.feature_blocks.split(",") if b.strip()]
    x_rifm_raw, feature_names = build_features(
        rifm,
        n_bits=args.n_bits,
        radius=args.radius,
        include_biowin=False,
        include_epi_physchem=False,
        guideline_categories=guideline_categories,
        molecular_feature_stack=args.molecular_feature_stack,
        feature_blocks=feature_blocks,
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

    direct_name = args.direct_model or args.model
    pdl_name = args.pdl_model or args.model
    direct = make_regressor(direct_name, args, args.seed)
    direct.fit(x_rifm, y_rifm)

    pair_a, pair_b, dy = sample_pairs(
        y_rifm,
        n_pairs=args.pairs,
        min_abs_dy=args.min_abs_dy,
        seed=args.seed + 99,
        row_weights=anchor_weights if args.weight_training_pairs else None,
    )
    pair_raw = build_pair_features(
        x_rifm[pair_a],
        x_rifm[pair_b],
        y_rifm[pair_a],
        include_abs_delta=args.include_abs_delta,
    )
    pair_train, pair_prep = fit_feature_prep(pair_raw, args.arcsinh_threshold)
    pdl = make_regressor(pdl_name, args, args.seed + 1)
    pdl.fit(pair_train, dy)
    return {
        "rifm": rifm,
        "guideline_categories": guideline_categories,
        "feature_blocks": feature_blocks,
        "feature_names": feature_names,
        "x_rifm": x_rifm,
        "y_rifm": y_rifm,
        "prep": prep,
        "selector": selector,
        "direct": direct,
        "pdl": pdl,
        "direct_model_name": direct_name,
        "pdl_model_name": pdl_name,
        "pair_prep": pair_prep,
        "anchor_weights": anchor_weights,
        "anchor_weight_summary": anchor_weight_summary,
    }


def predict_control(args: argparse.Namespace, bundle: dict, control: pd.DataFrame) -> pd.DataFrame:
    x_raw, _ = build_features(
        control,
        n_bits=args.n_bits,
        radius=args.radius,
        include_biowin=False,
        include_epi_physchem=False,
        guideline_categories=bundle["guideline_categories"],
        molecular_feature_stack=args.molecular_feature_stack,
        feature_blocks=bundle["feature_blocks"],
        n_jobs=args.jobs,
        use_osmo_cache=not args.no_osmo_cache,
    )
    x = apply_feature_prep(x_raw, bundle["prep"])
    x = apply_feature_selector(x, bundle["selector"])
    pred = control.copy()
    pred["direct_percent_score"] = bundle["direct"].predict(x).astype(np.float32)
    pred["pdl_percent_score"] = predict_pdl(
        bundle["pdl"],
        bundle["pair_prep"],
        bundle["x_rifm"],
        bundle["y_rifm"],
        x,
        anchors=args.anchors,
        include_abs_delta=args.include_abs_delta,
        aggregate=args.anchor_aggregate,
        batch_size=args.predict_batch_size,
        anchor_weights=bundle.get("anchor_weights") if args.weight_inference_anchors else None,
        anchor_candidate_factor=args.anchor_candidate_factor,
        anchor_quality_power=args.anchor_quality_power,
    )
    pred["protocol_ready_threshold"] = ready_thresholds(pred["guideline_norm"])
    return pred


def summarize(pred: pd.DataFrame, level: str) -> pd.DataFrame:
    rows: list[dict] = []
    for method, col in [("direct", "direct_percent_score"), ("pdl", "pdl_percent_score")]:
        row = {"level": level, "task": "percent_regression", "method": method}
        row.update(regression_metrics(pred["y_percent"].to_numpy(), pred[col].to_numpy()))
        rows.append(row)

        for label_col, task, fixed_threshold in [
            ("y_ready_clean", "ready_clean_fixed60", 60.0),
            ("y_ready_strict", "ready_strict_fixed60", 60.0),
            ("y_any_biodeg", "any_biodeg_fixed20", 20.0),
        ]:
            row = {"level": level, "task": task, "method": method}
            row.update(binary_metrics(pred[label_col].to_numpy(), pred[col].to_numpy(), fixed_threshold))
            rows.append(row)

        row = {"level": level, "task": "ready_clean_protocol_threshold", "method": method}
        row.update(
            binary_metrics(
                pred["y_ready_clean"].to_numpy(),
                pred[col].to_numpy(),
                pred["protocol_ready_threshold"].to_numpy(),
            )
        )
        rows.append(row)

        y_ready = pred["y_ready_clean"].to_numpy(dtype=np.float32)
        score = pred[col].to_numpy(dtype=np.float32)
        finite = np.isfinite(y_ready) & np.isfinite(score)
        if np.any(finite):
            threshold, train_score = choose_threshold(y_ready[finite], score[finite], "balanced_accuracy")
            row = {
                "level": level,
                "task": "ready_clean_oracle_calibrated_excelra",
                "method": method,
                "train_threshold_score": float(train_score),
            }
            row.update(binary_metrics(y_ready, score, threshold))
            rows.append(row)

        row = {"level": level, "task": "label_order_spearman", "method": method}
        row.update(order_metrics(pred["y_order"].to_numpy(), pred[col].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    control = load_excelra(Path(args.excelra_csv))
    bundle = train_rifm_models(args)
    pred_obs = predict_control(args, bundle, control)
    rifm_molecules = set(bundle["rifm"]["canonical_smiles"].dropna().astype(str))
    pred_obs["seen_in_rifm_train"] = pred_obs["canonical_smiles"].astype(str).isin(rifm_molecules)
    pred_obs["known_protocol"] = ~pred_obs["guideline_norm"].isin(["UNKNOWN", "OTHER"])
    pred_mol = molecule_level(pred_obs)

    known_protocol = pred_obs["known_protocol"].to_numpy(dtype=bool)
    seen = pred_obs["seen_in_rifm_train"].to_numpy(dtype=bool)
    mol_known = pred_mol["known_protocol"].to_numpy(dtype=bool)
    mol_seen = pred_mol["seen_in_rifm_train"].to_numpy(dtype=bool)
    summary_blocks = [
        summarize(pred_obs, "observation"),
        summarize(pred_obs[seen].copy(), "observation_seen_in_rifm"),
        summarize(pred_obs[~seen].copy(), "observation_novel"),
        summarize(pred_obs[known_protocol].copy(), "observation_known_protocol"),
        summarize(pred_obs[known_protocol & seen].copy(), "observation_known_protocol_seen_in_rifm"),
        summarize(pred_obs[known_protocol & ~seen].copy(), "observation_known_protocol_novel"),
        summarize(pred_obs[~known_protocol].copy(), "observation_unknown_or_other"),
        summarize(pred_mol, "molecule"),
        summarize(pred_mol[mol_seen].copy(), "molecule_seen_in_rifm"),
        summarize(pred_mol[~mol_seen].copy(), "molecule_novel"),
        summarize(pred_mol[mol_known].copy(), "molecule_known_protocol"),
        summarize(pred_mol[mol_known & mol_seen].copy(), "molecule_known_protocol_seen_in_rifm"),
        summarize(pred_mol[mol_known & ~mol_seen].copy(), "molecule_known_protocol_novel"),
    ]
    summary = pd.concat(summary_blocks, axis=0, ignore_index=True)
    sort_cols = [c for c in ["level", "task", "balanced_accuracy", "r2", "spearman"] if c in summary.columns]
    if sort_cols:
        ascending = [True, True] + [False] * (len(sort_cols) - 2)
        summary = summary.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)

    pred_obs.to_csv(out_dir / "excelra_observation_predictions.csv", index=False)
    pred_mol.to_csv(out_dir / "excelra_molecule_predictions.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "feature_names.txt").write_text("\n".join(bundle["feature_names"]) + "\n", encoding="utf-8")
    report = {
        "rifm_xlsx": str(Path(args.rifm_xlsx)),
        "excelra_csv": str(Path(args.excelra_csv)),
        "rifm_rows": int(len(bundle["rifm"])),
        "rifm_unique_molecules": int(bundle["rifm"]["canonical_smiles"].nunique()),
        "excelra_observations": int(len(pred_obs)),
        "excelra_molecules": int(pred_mol["canonical_smiles"].nunique()),
        "excelra_seen_in_rifm_observations": int(seen.sum()),
        "excelra_novel_observations": int((~seen).sum()),
        "excelra_seen_in_rifm_molecules": int(mol_seen.sum()),
        "excelra_novel_molecules": int((~mol_seen).sum()),
        "excelra_label_counts": pred_obs["label_norm"].value_counts(dropna=False).to_dict(),
        "excelra_percent_source_counts": pred_obs["y_percent_source"].value_counts(dropna=False).to_dict(),
        "excelra_known_protocol_observations": int(known_protocol.sum()),
        "excelra_unknown_or_other_observations": int((~known_protocol).sum()),
        "excelra_known_protocol_novel_observations": int((known_protocol & ~seen).sum()),
        "excelra_known_protocol_novel_molecules": int((mol_known & ~mol_seen).sum()),
        "pairs": int(args.pairs),
        "trees": int(args.trees),
        "direct_model_name": bundle.get("direct_model_name"),
        "pdl_model_name": bundle.get("pdl_model_name"),
        "ensemble_models": getattr(bundle.get("pdl"), "model_names", None),
        "select_method": args.select_method,
        "select_k": int(args.select_k),
        "anchor_weight_summary": bundle.get("anchor_weight_summary", {"enabled": False}),
        "elapsed_seconds": float(time.time() - t0),
        "args": vars(args),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[biodeg-excelra] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RIFM biodegradation % models on the Excelra control dataset.")
    parser.add_argument("--rifm-xlsx", default=str(DEFAULT_RIFM_2026))
    parser.add_argument("--excelra-csv", default=str(DEFAULT_EXCELRA))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_pdl_ordered/rifm2026_osmostack_rpc_excelra_control")
    parser.add_argument("--model", default="etr")
    parser.add_argument("--direct-model", default="etr")
    parser.add_argument("--pdl-model", default="")
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
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--min-abs-dy", type=float, default=3.0)
    parser.add_argument("--include-abs-delta", action="store_true")
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--anchor-aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--anchor-quality-csv", default="")
    parser.add_argument("--anchor-observation-homoset-csv", default=str(DEFAULT_HOMOSET_OBSERVATIONS))
    parser.add_argument("--anchor-conflict-weight", type=float, default=0.05)
    parser.add_argument("--anchor-single-source-weight", type=float, default=0.75)
    parser.add_argument("--anchor-multi-source-weight", type=float, default=1.0)
    parser.add_argument("--anchor-no-target-weight", type=float, default=0.5)
    parser.add_argument("--anchor-missing-quality-weight", type=float, default=0.75)
    parser.add_argument("--anchor-high-percent-conflict-weight", type=float, default=0.2)
    parser.add_argument("--anchor-observation-agree-percent", type=float, default=10.0)
    parser.add_argument("--anchor-observation-conflict-percent", type=float, default=20.0)
    parser.add_argument("--anchor-observation-agree-weight", type=float, default=1.25)
    parser.add_argument("--anchor-observation-conflict-weight", type=float, default=0.02)
    parser.add_argument("--anchor-candidate-factor", type=int, default=1)
    parser.add_argument("--anchor-quality-power", type=float, default=1.0)
    parser.add_argument("--no-weight-training-pairs", dest="weight_training_pairs", action="store_false")
    parser.add_argument("--no-weight-inference-anchors", dest="weight_inference_anchors", action="store_false")
    parser.set_defaults(weight_training_pairs=True, weight_inference_anchors=True)
    parser.add_argument("--predict-batch-size", type=int, default=96)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
