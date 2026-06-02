#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from evaluate_biodegradation_excelra_control import DEFAULT_EXCELRA, parse_number  # noqa: E402
from train_biodegradation_pdl_ordered import (  # noqa: E402
    DEFAULT_RIFM_2026,
    canonical_smiles,
    normalize_guideline,
    parse_days,
)


DEFAULT_RIFM_2024 = Path("/Users/guillaume-osmo/Downloads/BioDegradationData2024 (2).xlsx")
DEFAULT_2308_TRAIN = Path("/Users/guillaume-osmo/Github/transformer-CNN-osmoai/transformer_cnn/biodegradation_2308_train.csv")
DEFAULT_2308_VAL = Path("/Users/guillaume-osmo/Github/transformer-CNN-osmoai/transformer_cnn/biodegradation_2308_val.csv")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path


def inchikey_from_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def protocol_family(guideline: object) -> str:
    text = normalize_guideline(guideline)
    if text in {"UNKNOWN", "NAN", ""}:
        return "UNKNOWN"
    if "BIODEG2308" in text:
        return "BIODEG2308_READY_BINARY"
    if "OTHER" == text:
        return "OTHER"
    for token in [
        "OECD301A",
        "OECD301B",
        "OECD301C",
        "OECD301D",
        "OECD301E",
        "OECD301F",
        "OECD302A",
        "OECD302B",
        "OECD302C",
        "OECD303A",
        "OECD310",
    ]:
        if token in text:
            return token
    if re.search(r"C\.?4A", text):
        return "OECD301A"
    if re.search(r"C\.?4C", text):
        return "OECD301B"
    if re.search(r"C\.?4F", text):
        return "OECD301C"
    if re.search(r"C\.?4E", text):
        return "OECD301D"
    if re.search(r"C\.?4B", text):
        return "OECD302A"
    if re.search(r"C\.?4D", text):
        return "OECD302B"
    if "ISO7827" in text:
        return "ISO7827"
    if "BODIS" in text:
        return "BODIS"
    return text[:64]


def protocol_group(family: str) -> str:
    if family.startswith("OECD301") or family == "OECD310":
        return "ready"
    if family.startswith("OECD302") or family in {"OECD303A", "ISO7827"}:
        return "inherent"
    if family == "BIODEG2308_READY_BINARY":
        return "ready_binary"
    if family in {"UNKNOWN", "OTHER"}:
        return "unknown"
    return "other_known"


def ready_threshold(family: str) -> float:
    return 70.0 if family in {"OECD301A", "OECD301E", "ISO7827"} else 60.0


def duration_bucket(days: float) -> str:
    if not np.isfinite(days):
        return "missing"
    if days <= 2:
        return "0_2d"
    if days <= 7:
        return "3_7d"
    if days <= 14:
        return "8_14d"
    if days <= 21:
        return "15_21d"
    if days <= 35:
        return "22_35d"
    if days <= 70:
        return "36_70d"
    return "gt70d"


def continuous_to_ready(row: pd.Series) -> float:
    if not np.isfinite(row.get("y_percent", np.nan)):
        return np.nan
    if row.get("protocol_group") != "ready":
        return np.nan
    return float(row["y_percent"] >= ready_threshold(str(row["protocol_family"])))


def continuous_to_inherent(row: pd.Series) -> float:
    if not np.isfinite(row.get("y_percent", np.nan)):
        return np.nan
    if row.get("protocol_group") != "inherent":
        return np.nan
    return float(row["y_percent"] >= 20.0)


def load_rifm(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame(index=df.index)
    out["source"] = source
    out["source_row"] = np.arange(len(df), dtype=int)
    out["raw_smiles"] = df["SMILES"]
    out["canonical_smiles"] = df["SMILES"].map(canonical_smiles)
    out["name"] = df.get("Chemical_name")
    out["cas"] = df.get("CAS")
    out["guideline_norm"] = df["Test guideline"].map(normalize_guideline)
    out["protocol_family"] = out["guideline_norm"].map(protocol_family)
    out["protocol_group"] = out["protocol_family"].map(protocol_group)
    out["duration_days"] = df["Duration"].map(parse_days)
    out["duration_bucket"] = out["duration_days"].map(duration_bucket)
    unit = df["Unit"].astype(str).str.strip()
    y = pd.to_numeric(df["Reviewed Data/Results"], errors="coerce")
    out["y_percent"] = y.where(unit.eq("%") | df["Unit"].isna()).clip(0.0, 100.0)
    out["y_percent_source"] = "direct"
    out.loc[out["y_percent"].isna(), "y_percent_source"] = "missing"
    out["y_ready_label"] = np.nan
    out["y_ready_rule"] = out.apply(continuous_to_ready, axis=1)
    out["y_inherent_rule"] = out.apply(continuous_to_inherent, axis=1)
    out["label_norm"] = np.nan
    out["split"] = source
    out["is_known_protocol"] = ~out["protocol_family"].isin(["UNKNOWN", "OTHER"])
    out = out[out["canonical_smiles"].notna()].reset_index(drop=True)
    out["inchikey"] = out["canonical_smiles"].map(inchikey_from_smiles)
    return out


def load_excelra(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=df.index)
    out["source"] = "excelra"
    out["source_row"] = np.arange(len(df), dtype=int)
    out["raw_smiles"] = df["smiles"]
    out["canonical_smiles"] = df["smiles"].map(canonical_smiles)
    out["name"] = df.get("material_name")
    out["cas"] = df.get("cas_number")
    out["guideline_norm"] = df["compliance"].map(normalize_guideline)
    out["protocol_family"] = out["guideline_norm"].map(protocol_family)
    out["protocol_group"] = out["protocol_family"].map(protocol_group)
    out["duration_days"] = df["num_days"].map(parse_number)
    out["duration_bucket"] = out["duration_days"].map(duration_bucket)
    y_direct = df["perc_biodeg"].map(parse_number)
    lower = df["perc_biodeg_lower"].map(parse_number) if "perc_biodeg_lower" in df.columns else np.nan
    upper = df["perc_biodeg_upper"].map(parse_number) if "perc_biodeg_upper" in df.columns else np.nan
    midpoint = (pd.Series(lower, index=df.index) + pd.Series(upper, index=df.index)) / 2.0
    out["y_percent"] = y_direct.where(pd.Series(y_direct).notna(), midpoint).clip(0.0, 100.0)
    out["y_percent_source"] = np.where(pd.Series(y_direct).notna(), "direct", "bound_midpoint")
    out.loc[out["y_percent"].isna(), "y_percent_source"] = "missing"
    out["label_norm"] = df["label"].astype(str).str.strip().str.lower()
    out["y_ready_label"] = out["label_norm"].map(
        {
            "readily biodegradable": 1.0,
            "non-biodegradable": 0.0,
            "poorly biodegradable": 0.0,
        }
    )
    out["y_ready_rule"] = out.apply(continuous_to_ready, axis=1)
    out["y_inherent_rule"] = out.apply(continuous_to_inherent, axis=1)
    out["split"] = "excelra"
    out["is_known_protocol"] = ~out["protocol_family"].isin(["UNKNOWN", "OTHER"])
    out = out[out["canonical_smiles"].notna()].reset_index(drop=True)
    out["inchikey"] = out["canonical_smiles"].map(inchikey_from_smiles)
    return out


def load_2308(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(index=df.index)
    out["source"] = f"biodeg2308_{split}"
    out["source_row"] = np.arange(len(df), dtype=int)
    out["raw_smiles"] = df["smiles"]
    out["canonical_smiles"] = df["smiles"].map(canonical_smiles)
    out["name"] = np.nan
    out["cas"] = np.nan
    out["guideline_norm"] = "BIODEG2308_READY_BINARY"
    out["protocol_family"] = "BIODEG2308_READY_BINARY"
    out["protocol_group"] = "ready_binary"
    out["duration_days"] = np.nan
    out["duration_bucket"] = "missing"
    out["y_percent"] = np.nan
    out["y_percent_source"] = "missing"
    out["label_norm"] = np.nan
    out["y_ready_label"] = pd.to_numeric(df["Result0"], errors="coerce")
    out["y_ready_rule"] = np.nan
    out["y_inherent_rule"] = np.nan
    out["split"] = split
    out["is_known_protocol"] = False
    out = out[out["canonical_smiles"].notna() & out["y_ready_label"].notna()].reset_index(drop=True)
    out["inchikey"] = out["canonical_smiles"].map(inchikey_from_smiles)
    return out


def agreement_metrics(df: pd.DataFrame, a: str, b: str, key: str) -> dict:
    pivot = (
        df[df["source"].isin([a, b])]
        .groupby([key, "source"], dropna=False)
        .agg(
            y_percent=("y_percent", "mean"),
            y_ready_label=("y_ready_label", "mean"),
            y_ready_rule=("y_ready_rule", "mean"),
            ready_best=("ready_best", "mean"),
            n=("source", "size"),
        )
        .reset_index()
    )
    wide = pivot.pivot(index=key, columns="source")
    row: dict[str, float | int | str] = {"source_a": a, "source_b": b, "key": key}
    if ("y_percent", a) in wide.columns and ("y_percent", b) in wide.columns:
        ya = wide[("y_percent", a)]
        yb = wide[("y_percent", b)]
        mask = ya.notna() & yb.notna()
        diff = (ya[mask] - yb[mask]).astype(float)
        row["n_percent_overlap"] = int(mask.sum())
        row["percent_mae"] = float(np.mean(np.abs(diff))) if len(diff) else np.nan
        row["percent_rmse"] = float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan
        row["percent_bias_a_minus_b"] = float(np.mean(diff)) if len(diff) else np.nan
        row["percent_spearman"] = float(spearmanr(ya[mask], yb[mask]).statistic) if mask.sum() >= 3 else np.nan
    for target in ["y_ready_label", "y_ready_rule"]:
        if (target, a) in wide.columns and (target, b) in wide.columns:
            ya = wide[(target, a)]
            yb = wide[(target, b)]
            mask = ya.notna() & yb.notna()
            if mask.sum():
                a_bin = (ya[mask].astype(float) >= 0.5).astype(int)
                b_bin = (yb[mask].astype(float) >= 0.5).astype(int)
                row[f"n_{target}_overlap"] = int(mask.sum())
                row[f"{target}_accuracy"] = float(accuracy_score(a_bin, b_bin))
                row[f"{target}_balanced_accuracy"] = float(balanced_accuracy_score(a_bin, b_bin))
                if len(np.unique(a_bin)) == 2 and len(np.unique(yb[mask])) > 1:
                    row[f"{target}_roc_auc_b_scores_a_labels"] = float(roc_auc_score(a_bin, yb[mask]))
    if ("ready_best", a) in wide.columns and ("ready_best", b) in wide.columns:
        ya = wide[("ready_best", a)]
        yb = wide[("ready_best", b)]
        mask = ya.notna() & yb.notna()
        if mask.sum():
            a_bin = (ya[mask].astype(float) >= 0.5).astype(int)
            b_bin = (yb[mask].astype(float) >= 0.5).astype(int)
            row["n_ready_best_overlap"] = int(mask.sum())
            row["ready_best_accuracy"] = float(accuracy_score(a_bin, b_bin))
            row["ready_best_balanced_accuracy"] = float(balanced_accuracy_score(a_bin, b_bin))
            if len(np.unique(a_bin)) == 2 and len(np.unique(yb[mask])) > 1:
                row["ready_best_roc_auc_b_scores_a_labels"] = float(roc_auc_score(a_bin, yb[mask]))
    return row


def agreement_by_columns(df: pd.DataFrame, a: str, b: str, keys: list[str]) -> dict:
    tmp = df[df["source"].isin([a, b])].dropna(subset=keys).copy()
    pivot = (
        tmp.groupby(keys + ["source"], dropna=False)
        .agg(
            y_percent=("y_percent", "mean"),
            ready_best=("ready_best", "mean"),
            n=("source", "size"),
        )
        .reset_index()
    )
    wide = pivot.pivot(index=keys, columns="source")
    row: dict[str, float | int | str] = {"source_a": a, "source_b": b, "keys": "|".join(keys)}
    if ("y_percent", a) in wide.columns and ("y_percent", b) in wide.columns:
        ya = wide[("y_percent", a)]
        yb = wide[("y_percent", b)]
        mask = ya.notna() & yb.notna()
        diff = (ya[mask] - yb[mask]).astype(float)
        row["n_percent_overlap"] = int(mask.sum())
        row["percent_mae"] = float(np.mean(np.abs(diff))) if len(diff) else np.nan
        row["percent_rmse"] = float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan
        row["percent_bias_a_minus_b"] = float(np.mean(diff)) if len(diff) else np.nan
        row["percent_spearman"] = float(spearmanr(ya[mask], yb[mask]).statistic) if mask.sum() >= 3 else np.nan
    if ("ready_best", a) in wide.columns and ("ready_best", b) in wide.columns:
        ya = wide[("ready_best", a)]
        yb = wide[("ready_best", b)]
        mask = ya.notna() & yb.notna()
        if mask.sum():
            a_bin = (ya[mask].astype(float) >= 0.5).astype(int)
            b_bin = (yb[mask].astype(float) >= 0.5).astype(int)
            row["n_ready_best_overlap"] = int(mask.sum())
            row["ready_best_accuracy"] = float(accuracy_score(a_bin, b_bin))
            row["ready_best_balanced_accuracy"] = float(balanced_accuracy_score(a_bin, b_bin))
    return row


def molecule_homoset(obs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for smi, g in obs.groupby("canonical_smiles", sort=False):
        row: dict[str, object] = {
            "canonical_smiles": smi,
            "inchikey": g["inchikey"].dropna().iloc[0] if g["inchikey"].notna().any() else None,
            "n_observations": int(len(g)),
            "sources": "|".join(sorted(g["source"].unique())),
            "n_sources": int(g["source"].nunique()),
            "has_rifm2026": bool((g["source"] == "rifm2026").any()),
            "has_excelra": bool((g["source"] == "excelra").any()),
            "has_2308": bool(g["source"].str.startswith("biodeg2308").any()),
            "has_known_protocol": bool(g["is_known_protocol"].any()),
            "has_unknown_protocol": bool((~g["is_known_protocol"]).any()),
        }
        for source in ["rifm2026", "rifm2024", "excelra", "biodeg2308_train", "biodeg2308_val"]:
            sg = g[g["source"] == source]
            row[f"n_{source}"] = int(len(sg))
            row[f"{source}_percent_mean"] = float(sg["y_percent"].mean()) if sg["y_percent"].notna().any() else np.nan
            row[f"{source}_ready_label_mean"] = (
                float(sg["y_ready_label"].mean()) if sg["y_ready_label"].notna().any() else np.nan
            )
            row[f"{source}_ready_rule_mean"] = (
                float(sg["y_ready_rule"].mean()) if sg["y_ready_rule"].notna().any() else np.nan
            )
            row[f"{source}_ready_best_mean"] = (
                float(sg["ready_best"].mean()) if sg["ready_best"].notna().any() else np.nan
            )
        ready_values = pd.concat([g["y_ready_label"], g["y_ready_rule"]], axis=0).dropna().astype(float)
        row["ready_votes_n"] = int(len(ready_values))
        row["ready_vote_mean"] = float(ready_values.mean()) if len(ready_values) else np.nan
        row["ready_vote_disagreement"] = bool(len(ready_values) and ready_values.min() < 0.5 and ready_values.max() >= 0.5)
        row["ready_consensus"] = (
            float(ready_values.mean() >= 0.5) if len(ready_values) and not row["ready_vote_disagreement"] else np.nan
        )
        percent = g["y_percent"].dropna().astype(float)
        row["percent_n"] = int(len(percent))
        row["percent_mean"] = float(percent.mean()) if len(percent) else np.nan
        row["percent_std"] = float(percent.std(ddof=0)) if len(percent) else np.nan
        row["percent_range"] = float(percent.max() - percent.min()) if len(percent) else np.nan
        row["high_percent_conflict"] = bool(len(percent) >= 2 and (percent.max() - percent.min()) >= 40.0)
        row["percent_consensus"] = row["percent_mean"] if len(percent) and not row["high_percent_conflict"] else np.nan
        if row["ready_vote_disagreement"] or row["high_percent_conflict"]:
            status = "conflict"
        elif row["n_sources"] >= 2:
            status = "multi_source_agree"
        elif row["ready_votes_n"] or row["percent_n"]:
            status = "single_source"
        else:
            status = "no_target"
        row["curation_status"] = status
        row["use_for_ready_training"] = bool(row["ready_votes_n"] and not row["ready_vote_disagreement"])
        row["use_for_percent_training"] = bool(row["percent_n"] and not row["high_percent_conflict"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Biodegradation homoset/data-quality audit across RIFM, Excelra, and 2308.")
    parser.add_argument("--rifm2026", default=str(DEFAULT_RIFM_2026))
    parser.add_argument("--rifm2024", default=str(DEFAULT_RIFM_2024))
    parser.add_argument("--excelra", default=str(DEFAULT_EXCELRA))
    parser.add_argument("--train2308", default=str(DEFAULT_2308_TRAIN))
    parser.add_argument("--val2308", default=str(DEFAULT_2308_VAL))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_homoset_audit")
    args = parser.parse_args()

    frames = [
        load_rifm(Path(args.rifm2026), "rifm2026"),
        load_excelra(Path(args.excelra)),
        load_2308(Path(args.train2308), "train"),
        load_2308(Path(args.val2308), "val"),
    ]
    if Path(args.rifm2024).exists():
        frames.append(load_rifm(Path(args.rifm2024), "rifm2024"))
    obs = pd.concat(frames, axis=0, ignore_index=True)
    obs["ready_best"] = np.nan
    biodeg2308 = obs["source"].astype(str).str.startswith("biodeg2308")
    excelra = obs["source"].eq("excelra")
    rifm = obs["source"].astype(str).str.startswith("rifm")
    obs.loc[biodeg2308, "ready_best"] = obs.loc[biodeg2308, "y_ready_label"]
    obs.loc[excelra, "ready_best"] = obs.loc[excelra, "y_ready_label"]
    obs.loc[rifm, "ready_best"] = obs.loc[rifm, "y_ready_rule"]
    mol = molecule_homoset(obs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out_dir / "biodegradation_observation_homoset.csv", index=False)
    mol.to_csv(out_dir / "biodegradation_molecule_homoset.csv", index=False)
    curated_cols = [
        "canonical_smiles",
        "inchikey",
        "sources",
        "n_sources",
        "n_observations",
        "curation_status",
        "use_for_ready_training",
        "ready_consensus",
        "ready_votes_n",
        "ready_vote_mean",
        "ready_vote_disagreement",
        "use_for_percent_training",
        "percent_consensus",
        "percent_n",
        "percent_std",
        "percent_range",
        "high_percent_conflict",
        "has_rifm2026",
        "has_excelra",
        "has_2308",
        "has_known_protocol",
    ]
    mol[curated_cols].to_csv(out_dir / "biodegradation_molecule_curated_targets.csv", index=False)

    agreements = []
    pairs = [
        ("rifm2026", "excelra"),
        ("rifm2026", "biodeg2308_train"),
        ("rifm2026", "biodeg2308_val"),
        ("excelra", "biodeg2308_train"),
        ("excelra", "biodeg2308_val"),
    ]
    for a, b in pairs:
        for key in ["canonical_smiles", "inchikey"]:
            agreements.append(agreement_metrics(obs, a, b, key))
    agreements_df = pd.DataFrame(agreements)
    agreements_df.to_csv(out_dir / "source_agreement.csv", index=False)

    exact_agreements = []
    exact_keys = [
        ["canonical_smiles", "protocol_family"],
        ["canonical_smiles", "protocol_family", "duration_bucket"],
        ["canonical_smiles", "protocol_group"],
    ]
    for keys in exact_keys:
        exact_agreements.append(agreement_by_columns(obs, "rifm2026", "excelra", keys))
        exact_agreements.append(agreement_by_columns(obs, "rifm2024", "excelra", keys))
    exact_agreements_df = pd.DataFrame(exact_agreements)
    exact_agreements_df.to_csv(out_dir / "source_agreement_exact_protocol.csv", index=False)

    conflict_cols = [
        "canonical_smiles",
        "inchikey",
        "sources",
        "n_observations",
        "ready_votes_n",
        "ready_vote_mean",
        "ready_vote_disagreement",
        "percent_n",
        "percent_mean",
        "percent_std",
        "percent_range",
        "high_percent_conflict",
        "rifm2026_percent_mean",
        "excelra_percent_mean",
        "rifm2026_ready_rule_mean",
        "excelra_ready_label_mean",
        "excelra_ready_best_mean",
        "biodeg2308_train_ready_label_mean",
        "biodeg2308_train_ready_best_mean",
        "biodeg2308_val_ready_label_mean",
        "biodeg2308_val_ready_best_mean",
    ]
    conflicts = mol[mol["ready_vote_disagreement"] | mol["high_percent_conflict"]].copy()
    conflicts = conflicts.sort_values(["ready_vote_disagreement", "percent_range"], ascending=[False, False])
    conflicts[conflict_cols].to_csv(out_dir / "molecule_conflicts.csv", index=False)

    source_counts = obs.groupby("source").agg(
        observations=("source", "size"),
        molecules=("canonical_smiles", "nunique"),
        known_protocol_obs=("is_known_protocol", "sum"),
        percent_obs=("y_percent", lambda s: int(s.notna().sum())),
        ready_label_obs=("y_ready_label", lambda s: int(s.notna().sum())),
        ready_rule_obs=("y_ready_rule", lambda s: int(s.notna().sum())),
        ready_best_obs=("ready_best", lambda s: int(s.notna().sum())),
    )
    overlap = {
        "source_counts": source_counts.reset_index().to_dict(orient="records"),
        "n_observations_total": int(len(obs)),
        "n_molecules_union": int(mol["canonical_smiles"].nunique()),
        "n_molecules_all_three_rifm_excelra_2308": int((mol["has_rifm2026"] & mol["has_excelra"] & mol["has_2308"]).sum()),
        "n_molecules_rifm_excelra": int((mol["has_rifm2026"] & mol["has_excelra"]).sum()),
        "n_molecules_rifm_2308": int((mol["has_rifm2026"] & mol["has_2308"]).sum()),
        "n_molecules_excelra_2308": int((mol["has_excelra"] & mol["has_2308"]).sum()),
        "n_molecules_ready_vote_conflict": int(mol["ready_vote_disagreement"].sum()),
        "n_molecules_high_percent_conflict_ge40": int(mol["high_percent_conflict"].sum()),
        "curation_status_counts": mol["curation_status"].value_counts(dropna=False).to_dict(),
        "n_molecules_use_for_ready_training": int(mol["use_for_ready_training"].sum()),
        "n_molecules_use_for_percent_training": int(mol["use_for_percent_training"].sum()),
        "protocol_family_counts": obs["protocol_family"].value_counts(dropna=False).to_dict(),
        "protocol_group_counts": obs["protocol_group"].value_counts(dropna=False).to_dict(),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")

    print(json.dumps(overlap, indent=2), flush=True)
    print("\nSOURCE AGREEMENT", flush=True)
    print(agreements_df.to_string(index=False), flush=True)
    print("\nSOURCE AGREEMENT EXACT PROTOCOL", flush=True)
    print(exact_agreements_df.to_string(index=False), flush=True)
    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
