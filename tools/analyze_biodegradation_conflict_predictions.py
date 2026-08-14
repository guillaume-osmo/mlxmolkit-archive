#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error


DEFAULT_HOMOSET_DIR = Path("benchmarks/biodegradation_homoset_audit/rifm_excelra_2308_v3_curated_targets")
DEFAULT_PRED_CSV = Path(
    "benchmarks/biodegradation_pdl_ordered/"
    "rifm2026_osmostack_rpc_excelra_control_etr50_pairs20k_leakage_audit/"
    "excelra_observation_predictions.csv"
)


def _source_fallback(df: pd.DataFrame, stem: str) -> pd.Series:
    a = f"rifm2026_{stem}"
    b = f"rifm2024_{stem}"
    if a not in df:
        df[a] = np.nan
    if b not in df:
        df[b] = np.nan
    return df[a].where(df[a].notna(), df[b])


def classification_summary(sub: pd.DataFrame, name: str) -> list[dict]:
    rows: list[dict] = []
    y_excelra = sub["excelra_ready_bin"].to_numpy(dtype=float)
    y_rifm = sub["rifm_ready_bin"].to_numpy(dtype=float)
    for method in ["direct", "pdl"]:
        for pred_col in [f"{method}_pred60", f"{method}_pred_protocol"]:
            pred = sub[pred_col].to_numpy(dtype=float)
            m_excelra = np.isfinite(y_excelra) & np.isfinite(pred)
            m_rifm = np.isfinite(y_rifm) & np.isfinite(pred)
            row: dict[str, float | int | str] = {
                "subset": name,
                "method": pred_col,
                "n_excelra": int(m_excelra.sum()),
                "n_rifm": int(m_rifm.sum()),
            }
            if m_excelra.sum() and len(np.unique(y_excelra[m_excelra])) > 1:
                row["bal_acc_vs_excelra"] = float(
                    balanced_accuracy_score(y_excelra[m_excelra].astype(int), pred[m_excelra].astype(int))
                )
                row["acc_vs_excelra"] = float(
                    accuracy_score(y_excelra[m_excelra].astype(int), pred[m_excelra].astype(int))
                )
            if m_rifm.sum() and len(np.unique(y_rifm[m_rifm])) > 1:
                row["bal_acc_vs_rifm"] = float(
                    balanced_accuracy_score(y_rifm[m_rifm].astype(int), pred[m_rifm].astype(int))
                )
                row["acc_vs_rifm"] = float(accuracy_score(y_rifm[m_rifm].astype(int), pred[m_rifm].astype(int)))

            disagree = np.isfinite(y_excelra) & np.isfinite(y_rifm) & (y_excelra != y_rifm) & np.isfinite(pred)
            row["n_rifm_excelra_disagree"] = int(disagree.sum())
            if disagree.sum():
                pred_disagree = pred[disagree]
                row["model_matches_excelra_in_disagree"] = float(np.mean(pred_disagree == y_excelra[disagree]))
                row["model_matches_rifm_in_disagree"] = float(np.mean(pred_disagree == y_rifm[disagree]))
                row["excelra_positive_rate_disagree"] = float(np.mean(y_excelra[disagree]))
                row["rifm_positive_rate_disagree"] = float(np.mean(y_rifm[disagree]))
            rows.append(row)
    return rows


def percent_summary(sub: pd.DataFrame, name: str) -> list[dict]:
    rows: list[dict] = []
    y_excelra = sub["y_percent"].to_numpy(dtype=float)
    y_rifm = sub["rifm_percent_ref"].to_numpy(dtype=float)
    for method, col in [("direct", "direct_percent_score"), ("pdl", "pdl_percent_score")]:
        pred = sub[col].to_numpy(dtype=float)
        m_excelra = np.isfinite(y_excelra) & np.isfinite(pred)
        m_rifm = np.isfinite(y_rifm) & np.isfinite(pred)
        row: dict[str, float | int | str] = {
            "subset": name,
            "method": method,
            "n_excelra_percent": int(m_excelra.sum()),
            "n_rifm_percent": int(m_rifm.sum()),
        }
        if m_excelra.sum():
            row["mae_vs_excelra"] = float(mean_absolute_error(y_excelra[m_excelra], pred[m_excelra]))
            row["rmse_vs_excelra"] = float(mean_squared_error(y_excelra[m_excelra], pred[m_excelra], squared=False))
            row["rho_vs_excelra"] = (
                float(spearmanr(y_excelra[m_excelra], pred[m_excelra]).statistic)
                if len(np.unique(y_excelra[m_excelra])) > 1
                else np.nan
            )
        if m_rifm.sum():
            row["mae_vs_rifm"] = float(mean_absolute_error(y_rifm[m_rifm], pred[m_rifm]))
            row["rmse_vs_rifm"] = float(mean_squared_error(y_rifm[m_rifm], pred[m_rifm], squared=False))
            row["rho_vs_rifm"] = (
                float(spearmanr(y_rifm[m_rifm], pred[m_rifm]).statistic)
                if len(np.unique(y_rifm[m_rifm])) > 1
                else np.nan
            )

        both = np.isfinite(y_excelra) & np.isfinite(y_rifm) & np.isfinite(pred)
        diff = np.abs(y_excelra - y_rifm)
        conflict = both & (diff >= 20.0)
        row["n_percent_rifm_excelra_diff_ge20"] = int(conflict.sum())
        if conflict.sum():
            excelra_err = np.abs(pred[conflict] - y_excelra[conflict])
            rifm_err = np.abs(pred[conflict] - y_rifm[conflict])
            row["closer_to_excelra_frac_ge20"] = float(np.mean(excelra_err < rifm_err))
            row["closer_to_rifm_frac_ge20"] = float(np.mean(rifm_err < excelra_err))
            row["mean_abs_diff_rifm_excelra_ge20"] = float(np.mean(diff[conflict]))
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> None:
    homoset_dir = Path(args.homoset_dir)
    out_dir = Path(args.out_dir) if args.out_dir else homoset_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.excelra_predictions, low_memory=False)
    mol = pd.read_csv(homoset_dir / "biodegradation_molecule_homoset.csv", low_memory=False)
    curated = pd.read_csv(homoset_dir / "biodegradation_molecule_curated_targets.csv", low_memory=False)

    curated_cols = [
        c
        for c in [
            "canonical_smiles",
            "curation_status",
            "ready_vote_disagreement",
            "high_percent_conflict",
            "ready_vote_mean",
            "percent_consensus",
            "percent_range",
            "has_rifm2026",
            "has_excelra",
            "has_2308",
        ]
        if c in curated.columns
    ]
    source_cols = [
        c
        for c in mol.columns
        if c == "canonical_smiles"
        or c.endswith("_ready_best_mean")
        or c.endswith("_ready_rule_mean")
        or c.endswith("_percent_mean")
    ]
    meta = curated[curated_cols].merge(mol[source_cols], on="canonical_smiles", how="left")
    df = pred.merge(meta, on="canonical_smiles", how="left")

    rifm_ready = _source_fallback(df, "ready_best_mean")
    rifm_percent = _source_fallback(df, "percent_mean")
    df["rifm_ready_bin"] = (rifm_ready >= 0.5).astype(float).where(rifm_ready.notna(), np.nan)
    df["rifm_percent_ref"] = rifm_percent
    df["excelra_ready_bin"] = df["y_ready_clean"]
    for method, col in [("direct", "direct_percent_score"), ("pdl", "pdl_percent_score")]:
        df[f"{method}_pred60"] = (df[col] >= 60.0).astype(float)
        df[f"{method}_pred_protocol"] = (df[col] >= df["protocol_ready_threshold"]).astype(float)

    subsets = {
        "all_excelra": df,
        "known_protocol": df[df["known_protocol"] == True],  # noqa: E712
        "conflict_molecules": df[df["curation_status"].eq("conflict")],
        "conflict_known_protocol": df[df["curation_status"].eq("conflict") & (df["known_protocol"] == True)],  # noqa: E712
        "conflict_known_seen_rifm": df[
            df["curation_status"].eq("conflict") & (df["known_protocol"] == True) & (df["seen_in_rifm_train"] == True)
        ],
        "conflict_known_novel": df[
            df["curation_status"].eq("conflict") & (df["known_protocol"] == True) & (df["seen_in_rifm_train"] == False)
        ],
        "nonconflict_known_seen_rifm": df[
            ~df["curation_status"].eq("conflict") & (df["known_protocol"] == True) & (df["seen_in_rifm_train"] == True)
        ],
    }

    class_df = pd.DataFrame([row for name, sub in subsets.items() for row in classification_summary(sub, name)])
    percent_df = pd.DataFrame([row for name, sub in subsets.items() for row in percent_summary(sub, name)])
    class_df.to_csv(out_dir / "conflict_prediction_classification_summary.csv", index=False)
    percent_df.to_csv(out_dir / "conflict_prediction_percent_summary.csv", index=False)

    mask = (
        df["curation_status"].eq("conflict")
        & df["known_protocol"].eq(True)
        & df["excelra_ready_bin"].notna()
        & df["rifm_ready_bin"].notna()
        & (df["excelra_ready_bin"] != df["rifm_ready_bin"])
    )
    detail_cols = [
        "canonical_smiles",
        "material_name",
        "cas_number",
        "compliance",
        "num_days",
        "label",
        "perc_biodeg",
        "y_percent",
        "direct_percent_score",
        "pdl_percent_score",
        "protocol_ready_threshold",
        "excelra_ready_bin",
        "rifm_ready_bin",
        "rifm2026_ready_best_mean",
        "rifm2024_ready_best_mean",
        "rifm2026_percent_mean",
        "rifm2024_percent_mean",
        "seen_in_rifm_train",
        "percent_range",
        "ready_vote_mean",
    ]
    detail = df.loc[mask, [c for c in detail_cols if c in df.columns]].copy()
    detail["direct_matches_excelra_60"] = df.loc[mask, "direct_pred60"].to_numpy() == df.loc[mask, "excelra_ready_bin"].to_numpy()
    detail["direct_matches_rifm_60"] = df.loc[mask, "direct_pred60"].to_numpy() == df.loc[mask, "rifm_ready_bin"].to_numpy()
    detail["pdl_matches_excelra_60"] = df.loc[mask, "pdl_pred60"].to_numpy() == df.loc[mask, "excelra_ready_bin"].to_numpy()
    detail.to_csv(out_dir / "conflict_prediction_disagreement_rows.csv", index=False)

    print(class_df.to_string(index=False), flush=True)
    print(percent_df.to_string(index=False), flush=True)
    print(f"[biodeg-conflict] wrote summaries to {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether RIFM-trained models side with RIFM or Excelra on conflicts.")
    parser.add_argument("--homoset-dir", default=str(DEFAULT_HOMOSET_DIR))
    parser.add_argument("--excelra-predictions", default=str(DEFAULT_PRED_CSV))
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
