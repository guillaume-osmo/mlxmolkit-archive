#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_HOMOSET_OBS = Path(
    "benchmarks/biodegradation_homoset_audit/"
    "rifm_excelra_2308_v3_curated_targets/biodegradation_observation_homoset.csv"
)
DEFAULT_HOMOSET_MOL = Path(
    "benchmarks/biodegradation_homoset_audit/"
    "rifm_excelra_2308_v3_curated_targets/biodegradation_molecule_curated_targets.csv"
)
STRICT_PROTOCOL_RE = r"OECD301|OECD310|OECD302|BODIS|METHODC\.4|DIRECTIVE84/449/EEC,C\.4|C\.4"


def _bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    text = series.fillna(str(default)).astype(str).str.lower()
    return text.isin({"true", "1", "yes", "y"})


def load_strict_protocol_measurements(args: argparse.Namespace) -> pd.DataFrame:
    obs = pd.read_csv(args.homoset_observation_csv, low_memory=False)
    mol = pd.read_csv(args.homoset_molecule_csv, low_memory=False)
    obs = obs.merge(
        mol[
            [
                "canonical_smiles",
                "use_for_percent_training",
                "high_percent_conflict",
                "curation_status",
            ]
        ],
        on="canonical_smiles",
        how="left",
    )
    df = obs[
        obs["canonical_smiles"].notna()
        & obs["is_known_protocol"].astype(bool)
        & obs["y_percent"].notna()
        & obs["protocol_group"].isin(["ready", "inherent"])
    ].copy()
    df = df[
        _bool_series(df["use_for_percent_training"], default=True)
        & ~_bool_series(df["high_percent_conflict"], default=False)
    ].copy()
    if args.direct_percent_only:
        df = df[df["y_percent_source"].astype(str).eq("direct")].copy()
    if args.strict_protocol_guidelines:
        guideline = df["guideline_norm"].astype(str).str.upper()
        df = df[guideline.str.contains(STRICT_PROTOCOL_RE, regex=True, na=False)].copy()
    df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
    df["y_percent"] = pd.to_numeric(df["y_percent"], errors="coerce").clip(0.0, 100.0)
    df = df[df["duration_days"].notna() & df["y_percent"].notna()].copy()
    df["duration_round"] = df["duration_days"].round().astype(int)
    return df.reset_index(drop=True)


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # First collapse duplicated evidence inside the same protocol/guideline.
    protocol = (
        df.groupby(
            ["canonical_smiles", "duration_round", "protocol_group", "guideline_norm"],
            dropna=False,
        )
        .agg(
            y_protocol_median=("y_percent", "median"),
            y_protocol_max=("y_percent", "max"),
            y_protocol_min=("y_percent", "min"),
            n_protocol_obs=("y_percent", "size"),
            sources=("source", lambda x: "|".join(sorted(set(map(str, x))))),
            name=("name", "first"),
            cas=("cas", "first"),
        )
        .reset_index()
    )

    best_idx = protocol.groupby(["canonical_smiles", "duration_round"])["y_protocol_median"].idxmax()
    best = protocol.loc[best_idx].copy()
    best = best.rename(
        columns={
            "protocol_group": "best_protocol_group",
            "guideline_norm": "best_guideline_norm",
            "y_protocol_median": "best_y_percent",
        }
    )
    aggregate = (
        protocol.groupby(["canonical_smiles", "duration_round"])
        .agg(
            n_protocol_candidates=("y_protocol_median", "size"),
            protocol_groups=("protocol_group", lambda x: "|".join(sorted(set(map(str, x))))),
            y_best=("y_protocol_median", "max"),
            y_median_across_protocols=("y_protocol_median", "median"),
            y_min=("y_protocol_median", "min"),
            y_range=("y_protocol_median", lambda x: float(np.max(x) - np.min(x))),
            n_total_obs=("n_protocol_obs", "sum"),
        )
        .reset_index()
    )
    best = best.merge(aggregate, on=["canonical_smiles", "duration_round"], how="left")
    best["ready_any_pass_60"] = best["y_best"].ge(60.0)
    best["inherent_any_pass_70"] = best["y_best"].ge(70.0)
    best["protocol_disagreement_ge10"] = best["y_range"].ge(10.0)
    best["protocol_disagreement_ge20"] = best["y_range"].ge(20.0)
    best["upper_consensus_y_percent"] = np.where(
        best["n_protocol_candidates"].le(1) | best["y_range"].le(20.0),
        best["y_best"],
        best["y_median_across_protocols"],
    )
    best["upper_consensus_used_median"] = best["n_protocol_candidates"].gt(1) & best["y_range"].gt(20.0)
    best["upper_consensus_ready_pass_60"] = best["upper_consensus_y_percent"] >= 60.0
    best["upper_consensus_inherent_pass_70"] = best["upper_consensus_y_percent"] >= 70.0
    best["y_best_fraction"] = best["y_best"].clip(0.0, 100.0) / 100.0
    best["upper_consensus_y_fraction"] = best["upper_consensus_y_percent"].clip(0.0, 100.0) / 100.0
    # 0.001 in fraction space preserves 0.1 percentage-point resolution.
    best["y_best_fraction_round3"] = best["y_best_fraction"].round(3)
    best["upper_consensus_y_fraction_round3"] = best["upper_consensus_y_fraction"].round(3)
    best["y_best_percent_round1"] = best["y_best"].round(1)
    best["upper_consensus_y_percent_round1"] = best["upper_consensus_y_percent"].round(1)
    best["protocol_conflict_fraction"] = (best["y_range"].clip(lower=0.0) / 100.0).clip(0.0, 1.0)
    best["protocol_conflict_weight"] = 1.0 / (1.0 + (best["y_range"].clip(lower=0.0) / 20.0) ** 2)
    return protocol, best


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_strict_protocol_measurements(args)
    protocol, best = build_targets(df)
    protocol.to_csv(out_dir / "protocol_collapsed_same_molecule_duration.csv", index=False)
    best.to_csv(out_dir / "best_same_molecule_duration_target.csv", index=False)
    report = {
        "strict_observation_rows": int(len(df)),
        "strict_molecules": int(df["canonical_smiles"].nunique()),
        "protocol_collapsed_rows": int(len(protocol)),
        "best_target_rows": int(len(best)),
        "best_target_molecules": int(best["canonical_smiles"].nunique()),
        "multi_protocol_duration_rows": int(best["n_protocol_candidates"].gt(1).sum()),
        "protocol_disagreement_ge10": int(best["protocol_disagreement_ge10"].sum()),
        "protocol_disagreement_ge20": int(best["protocol_disagreement_ge20"].sum()),
        "best_protocol_counts": best["best_protocol_group"].value_counts(dropna=False).to_dict(),
        "ready_any_pass_60": best["ready_any_pass_60"].value_counts(dropna=False).to_dict(),
        "inherent_any_pass_70": best["inherent_any_pass_70"].value_counts(dropna=False).to_dict(),
        "upper_consensus_used_median": int(best["upper_consensus_used_median"].sum()),
        "upper_consensus_ready_pass_60": best["upper_consensus_ready_pass_60"].value_counts(dropna=False).to_dict(),
        "upper_consensus_inherent_pass_70": best["upper_consensus_inherent_pass_70"].value_counts(dropna=False).to_dict(),
        "median_protocol_conflict_weight": float(best["protocol_conflict_weight"].median()),
        "mean_protocol_conflict_weight": float(best["protocol_conflict_weight"].mean()),
        "args": vars(args),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build best accepted biodegradation target at molecule+duration level.")
    parser.add_argument("--homoset-observation-csv", default=str(DEFAULT_HOMOSET_OBS))
    parser.add_argument("--homoset-molecule-csv", default=str(DEFAULT_HOMOSET_MOL))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_protocol_annotation/best_protocol_same_duration_v1")
    parser.add_argument("--direct-percent-only", action="store_true", default=True)
    parser.add_argument("--allow-bound-percent", dest="direct_percent_only", action="store_false")
    parser.add_argument("--strict-protocol-guidelines", action="store_true", default=True)
    parser.add_argument("--allow-loose-guidelines", dest="strict_protocol_guidelines", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
