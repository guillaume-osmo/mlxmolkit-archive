#!/usr/bin/env python3
"""Build a source-audited deltaHvap v3 curation table.

The goal is to stop overwriting labels ad hoc.  Each molecule receives every
available source label, explicit disagreement statistics, and two training
weights:

* strict_sample_weight: high-confidence experimental labels only
* broad_sample_weight: experimental labels plus low-weight pseudo labels
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/delta_hvap_v2"
BENCH_DIR = REPO_ROOT / "benchmarks/delta_hvap_v3_curation"


@dataclass(frozen=True)
class SourceLabel:
    key: str
    canonical_smiles: str
    canonical_smiles_no_stereo: str
    source: str
    value: float
    reliability: float
    provenance: str
    note: str = ""
    uncertainty: float = math.nan


def canonical_pair(smiles: object) -> tuple[str, str]:
    if smiles is None or pd.isna(smiles):
        return "", ""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "", ""
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    no_stereo = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return iso, no_stereo


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def add_label(
    labels: list[SourceLabel],
    *,
    key: str,
    canonical_smiles: str,
    canonical_smiles_no_stereo: str = "",
    source: str,
    value: object,
    reliability: float,
    provenance: str,
    note: str = "",
    uncertainty: object = math.nan,
) -> None:
    v = finite_float(value)
    if not key or not canonical_smiles or not math.isfinite(v):
        return
    if not canonical_smiles_no_stereo:
        _iso, canonical_smiles_no_stereo = canonical_pair(canonical_smiles)
    labels.append(
        SourceLabel(
            key=key,
            canonical_smiles=canonical_smiles,
            canonical_smiles_no_stereo=canonical_smiles_no_stereo,
            source=source,
            value=v,
            reliability=float(reliability),
            provenance=provenance,
            note=note,
            uncertainty=finite_float(uncertainty),
        )
    )


def weighted_mean(labels: Iterable[SourceLabel]) -> float:
    items = list(labels)
    weights = np.asarray([x.reliability for x in items], dtype=np.float64)
    values = np.asarray([x.value for x in items], dtype=np.float64)
    if weights.sum() <= 0:
        return float(np.mean(values))
    return float(np.sum(weights * values) / np.sum(weights))


def source_values(labels: Iterable[SourceLabel], source: str) -> list[float]:
    return [x.value for x in labels if x.source == source]


def load_current_union(path: Path) -> tuple[pd.DataFrame, list[SourceLabel], dict[str, str], dict[str, dict]]:
    df = pd.read_csv(path)
    labels: list[SourceLabel] = []
    canonical_by_key: dict[str, str] = {}
    union_meta: dict[str, dict] = {}
    for idx, row in df.iterrows():
        iso, key = canonical_pair(row["canonical_smiles"])
        no_stereo = key
        key = iso
        if not key:
            continue
        canonical_by_key.setdefault(key, iso)
        union_meta[key] = {
            "in_current_union": True,
            "current_union_row": int(idx),
            "current_target_source": str(row.get("target_source", "")),
            "current_trusted_target_kJmol": finite_float(row.get("trusted_target_kJmol")),
            "current_sample_weight": finite_float(row.get("sample_weight")),
            "autovap_smiles": row.get("autovap_smiles", ""),
            "calc_smiles": row.get("calc_smiles", ""),
        }
        if str(row.get("target_source", "")) == "autovap_trusted":
            add_label(
                labels,
                key=key,
                canonical_smiles=iso,
                canonical_smiles_no_stereo=no_stereo,
                source="autovap_trusted",
                value=row.get("autovap_dvap_kJmol", row.get("trusted_target_kJmol")),
                reliability=0.90,
                provenance="deltaHvapv2_homoset_union/autovap_trusted",
                note=f"union_row={idx}",
            )
        if str(row.get("target_source", "")) == "calcphyschemprop_calibrated_pseudo":
            add_label(
                labels,
                key=key,
                canonical_smiles=iso,
                canonical_smiles_no_stereo=no_stereo,
                source="calcphyschemprop_pseudo",
                value=row.get("calc_deltaHvap_source_kJmol", row.get("trusted_target_kJmol")),
                reliability=0.20,
                provenance="deltaHvapv2_homoset_union/calcphyschemprop",
                note=f"union_row={idx}",
            )
    return df, labels, canonical_by_key, union_meta


def load_zenodo(path: Path) -> list[SourceLabel]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    labels: list[SourceLabel] = []
    for idx, row in df.iterrows():
        iso = str(row.get("_canonical_smiles", "") or "")
        key = str(row.get("_canonical_smiles_no_stereo", "") or "")
        if not key:
            iso, key = canonical_pair(row.get("SMILES", ""))
        no_stereo = key
        key = iso
        uncertainty = finite_float(row.get("Uncertainty / kJ mol  -1"))
        reliability = 1.00 if not math.isfinite(uncertainty) else max(0.70, min(1.10, 1.0 / (1.0 + uncertainty / 5.0)))
        add_label(
            labels,
            key=key,
            canonical_smiles=iso,
            canonical_smiles_no_stereo=no_stereo,
            source="zenodo8132046_nist",
            value=row.get("_deltaHvap_kJmol"),
            reliability=reliability,
            provenance=f"Zenodo 8132046 {_safe_text(row.get('_sheet'))}",
            note=f"row={idx}; type={_safe_text(row.get('Type'))}; refs={_safe_text(row.get('NIST References'))}",
            uncertainty=uncertainty,
        )
    return labels


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_naef2017(path: Path, outliers_path: Path) -> list[SourceLabel]:
    if not path.exists():
        return []
    outlier_names: set[str] = set()
    if outliers_path.exists():
        try:
            out = pd.read_excel(outliers_path)
            for value in out.iloc[:, 0].dropna().astype(str):
                low = value.strip().lower()
                if low and "molecule" not in low and "outlier" not in low:
                    outlier_names.add(low)
        except Exception:
            outlier_names = set()
    df = pd.read_csv(path)
    labels: list[SourceLabel] = []
    for idx, row in df.iterrows():
        if not bool(row.get("ok", False)):
            continue
        iso = _safe_text(row.get("canonical_smiles"))
        no_stereo = _safe_text(row.get("canonical_smiles_no_stereo"))
        key = iso
        name = _safe_text(row.get("name"))
        is_outlier = name.lower() in outlier_names
        add_label(
            labels,
            key=key,
            canonical_smiles=iso,
            canonical_smiles_no_stereo=no_stereo,
            source="naef2017_sdf",
            value=row.get("deltaHvap_2017_sdf_kJmol"),
            reliability=0.85 if not is_outlier else 0.55,
            provenance="Molecules 2017 22 1059 S2 SDF",
            note=f"sdf_index={row.get('sdf_index')}; name={name}; model_outlier={is_outlier}",
        )
    return labels


def load_mdpi2021(path: Path) -> list[SourceLabel]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    labels: list[SourceLabel] = []
    for idx, row in df.iterrows():
        iso, key = canonical_pair(row.get("canonical_smiles"))
        no_stereo = key
        key = iso
        add_label(
            labels,
            key=key,
            canonical_smiles=iso,
            canonical_smiles_no_stereo=no_stereo,
            source="mdpi2021_s06",
            value=row.get("deltaH_vap_exp_s06"),
            reliability=0.80,
            provenance="Molecules 2021 26 1045 S06",
            note=f"row={idx}; name={_safe_text(row.get('mdpi_name'))}",
        )
    return labels


def choose_row(key: str, labels: list[SourceLabel], canonical_by_key: dict[str, str], union_meta: dict[str, dict]) -> dict:
    experimental = [x for x in labels if x.source != "calcphyschemprop_pseudo"]
    pseudo = [x for x in labels if x.source == "calcphyschemprop_pseudo"]
    canonical = canonical_by_key.get(key) or (labels[0].canonical_smiles if labels else "")

    row: dict[str, object] = {
        "canonical_smiles": canonical,
        "canonical_smiles_no_stereo": canonical_pair(canonical)[1],
        "in_current_union": bool(union_meta.get(key, {}).get("in_current_union", False)),
        "current_union_row": union_meta.get(key, {}).get("current_union_row", -1),
        "current_target_source": union_meta.get(key, {}).get("current_target_source", ""),
        "n_source_labels": len(labels),
        "n_experimental_labels": len(experimental),
        "n_pseudo_labels": len(pseudo),
        "source_names": "|".join(sorted({x.source for x in labels})),
    }
    for source in (
        "autovap_trusted",
        "zenodo8132046_nist",
        "naef2017_sdf",
        "mdpi2021_s06",
        "calcphyschemprop_pseudo",
    ):
        vals = source_values(labels, source)
        row[f"{source}_n"] = len(vals)
        row[f"{source}_value_kJmol"] = float(np.median(vals)) if vals else math.nan

    chosen_source = ""
    strict_weight = 0.0
    broad_weight = 0.0
    confidence = "none"
    action = "no_label"
    target = math.nan
    label_spread = math.nan
    label_std = math.nan

    if experimental:
        exp_values = np.asarray([x.value for x in experimental], dtype=np.float64)
        label_spread = float(np.max(exp_values) - np.min(exp_values))
        label_std = float(np.std(exp_values)) if exp_values.size > 1 else 0.0
        target = weighted_mean(experimental)
        chosen_source = "weighted_experimental"
        broad_weight = 1.0
        if label_spread <= 2.0:
            confidence = "high"
            strict_weight = 1.0
            action = "use_experimental_agree_le2"
        elif label_spread <= 5.0:
            confidence = "medium"
            strict_weight = 0.70
            broad_weight = 0.85
            action = "use_experimental_mild_conflict_le5"
        elif label_spread <= 10.0:
            confidence = "review"
            strict_weight = 0.0
            broad_weight = 0.35
            action = "flag_experimental_conflict_le10"
        else:
            confidence = "exclude_conflict"
            strict_weight = 0.0
            broad_weight = 0.0
            action = "exclude_experimental_conflict_gt10"
    elif pseudo:
        target = weighted_mean(pseudo)
        chosen_source = "calcphyschemprop_pseudo"
        confidence = "pseudo"
        strict_weight = 0.0
        broad_weight = 0.20
        action = "use_low_weight_pseudo_only"

    row.update(
        {
            "curated_deltaHvap_kJmol": target,
            "chosen_source": chosen_source,
            "confidence": confidence,
            "curation_action": action,
            "strict_sample_weight": strict_weight,
            "broad_sample_weight": broad_weight,
            "experimental_spread_kJmol": label_spread,
            "experimental_std_kJmol": label_std,
            "source_detail": " || ".join(
                f"{x.source}:{x.value:.4g}:w{x.reliability:.2g}:{x.note}" for x in sorted(labels, key=lambda z: (z.source, z.value))
            ),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-union", type=Path, default=DATA_DIR / "deltaHvapv2_homoset_union.csv")
    parser.add_argument("--zenodo-csv", type=Path, default=BENCH_DIR.parent / "delta_hvap_v2_zenodo8132046_check/zenodo8132046_all_rows.csv")
    parser.add_argument("--naef2017-csv", type=Path, default=DATA_DIR / "source_mdpi_molecules_2017_22_1059/supplementary/S2_heat_vaporization_sdf.parsed.csv")
    parser.add_argument("--naef2017-outliers", type=Path, default=DATA_DIR / "source_mdpi_molecules_2017_22_1059/supplementary/S3. Compounds List of Heat-of-Vaporization Outliers.xls")
    parser.add_argument("--mdpi2021-csv", type=Path, default=REPO_ROOT / "benchmarks/delta_hvap_v2_mdpi_source_check/mdpi_thermo_reconstructed.csv")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    args = parser.parse_args()

    union, current_labels, canonical_by_key, union_meta = load_current_union(args.current_union)
    current_ns_to_key = {
        canonical_pair(smi)[1]: canonical_pair(smi)[0]
        for smi in union["canonical_smiles"].astype(str)
        if canonical_pair(smi)[0]
    }
    all_labels = []
    all_labels.extend(current_labels)
    all_labels.extend(load_zenodo(args.zenodo_csv))
    all_labels.extend(load_naef2017(args.naef2017_csv, args.naef2017_outliers))
    all_labels.extend(load_mdpi2021(args.mdpi2021_csv))

    remapped_labels: list[SourceLabel] = []
    for label in all_labels:
        if label.source != "calcphyschemprop_pseudo" and label.source != "autovap_trusted":
            current_key = current_ns_to_key.get(label.canonical_smiles_no_stereo)
            if current_key:
                label = SourceLabel(
                    key=current_key,
                    canonical_smiles=current_key,
                    canonical_smiles_no_stereo=label.canonical_smiles_no_stereo,
                    source=label.source,
                    value=label.value,
                    reliability=label.reliability,
                    provenance=label.provenance,
                    note=f"{label.note}; remapped_to_current_union_by_no_stereo",
                    uncertainty=label.uncertainty,
                )
        remapped_labels.append(label)
    all_labels = remapped_labels

    grouped: dict[str, list[SourceLabel]] = defaultdict(list)
    for label in all_labels:
        grouped[label.key].append(label)
        canonical_by_key.setdefault(label.key, label.canonical_smiles)

    rows = [choose_row(key, labels, canonical_by_key, union_meta) for key, labels in sorted(grouped.items())]
    curated = pd.DataFrame(rows).sort_values(["in_current_union", "canonical_smiles"], ascending=[False, True])

    current_keys = union["canonical_smiles"].map(lambda smi: canonical_pair(smi)[0]).tolist()
    current = pd.DataFrame({"canonical_smiles": current_keys, "_row_order": np.arange(len(union))})
    model_ready = current.merge(curated, on="canonical_smiles", how="left").sort_values("_row_order")
    model_ready = pd.concat([union.add_prefix("v2_"), model_ready.drop(columns=["_row_order"])], axis=1)
    new_missing_sigma = curated[~curated["in_current_union"]].copy()
    conflicts = curated[
        curated["n_experimental_labels"].gt(1)
        & curated["experimental_spread_kJmol"].fillna(0).gt(5.0)
    ].sort_values("experimental_spread_kJmol", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.bench_dir.mkdir(parents=True, exist_ok=True)
    curated_path = args.out_dir / "deltaHvapv3_curated_sources.csv"
    model_path = args.out_dir / "deltaHvapv3_model_ready_current3147.csv"
    strict_train_path = args.out_dir / "deltaHvapv3_current3147_strict_train.csv"
    broad_train_path = args.out_dir / "deltaHvapv3_current3147_broad_train.csv"
    new_path = args.out_dir / "deltaHvapv3_experimental_new_missing_sigma.csv"
    conflict_path = args.bench_dir / "deltaHvapv3_conflicts_for_review.csv"
    curated.to_csv(curated_path, index=False)
    model_ready.to_csv(model_path, index=False)
    new_missing_sigma.to_csv(new_path, index=False)
    conflicts.to_csv(conflict_path, index=False)

    def training_view(weight_col: str, *, include_pseudo: bool) -> pd.DataFrame:
        out = union.copy()
        out["trusted_target_kJmol"] = pd.to_numeric(model_ready["curated_deltaHvap_kJmol"], errors="coerce")
        out["curated_target_kJmol"] = out["trusted_target_kJmol"]
        out["sample_weight"] = pd.to_numeric(model_ready[weight_col], errors="coerce").fillna(0.0)
        out["target_source"] = "excluded_by_v3_curation"
        strict_mask = pd.to_numeric(model_ready["strict_sample_weight"], errors="coerce").fillna(0.0).gt(0)
        broad_mask = pd.to_numeric(model_ready["broad_sample_weight"], errors="coerce").fillna(0.0).gt(0)
        pseudo_mask = model_ready["confidence"].astype(str).eq("pseudo")
        out.loc[strict_mask, "target_source"] = "autovap_trusted"
        if include_pseudo:
            out.loc[broad_mask & ~strict_mask, "target_source"] = "calcphyschemprop_calibrated_pseudo"
            out.loc[pseudo_mask & broad_mask, "target_source"] = "calcphyschemprop_calibrated_pseudo"
        out["v3_confidence"] = model_ready["confidence"].astype(str)
        out["v3_curation_action"] = model_ready["curation_action"].astype(str)
        out["v3_source_names"] = model_ready["source_names"].astype(str)
        out["v3_experimental_spread_kJmol"] = model_ready["experimental_spread_kJmol"]
        return out

    training_view("strict_sample_weight", include_pseudo=False).to_csv(strict_train_path, index=False)
    training_view("broad_sample_weight", include_pseudo=True).to_csv(broad_train_path, index=False)

    summary = {
        "curated_rows": int(len(curated)),
        "current_union_rows": int(len(union)),
        "current_union_model_ready_rows": int(len(model_ready)),
        "new_missing_sigma_rows": int(len(new_missing_sigma)),
        "strict_train_rows": int(curated["strict_sample_weight"].gt(0).sum()),
        "broad_train_rows": int(curated["broad_sample_weight"].gt(0).sum()),
        "pseudo_only_rows": int(curated["confidence"].eq("pseudo").sum()),
        "conflict_gt5_rows": int(len(conflicts)),
        "conflict_gt10_rows": int(curated["curation_action"].eq("exclude_experimental_conflict_gt10").sum()),
        "current_union_strict_rows": int(model_ready["strict_sample_weight"].gt(0).sum()),
        "current_union_broad_rows": int(model_ready["broad_sample_weight"].gt(0).sum()),
        "current_union_conflict_gt10_rows": int(model_ready["curation_action"].eq("exclude_experimental_conflict_gt10").sum()),
        "source_label_counts": pd.Series([x.source for x in all_labels]).value_counts().to_dict(),
        "outputs": {
            "curated_sources": str(curated_path),
            "model_ready_current3147": str(model_path),
            "strict_train_current3147": str(strict_train_path),
            "broad_train_current3147": str(broad_train_path),
            "new_missing_sigma": str(new_path),
            "conflicts_for_review": str(conflict_path),
        },
    }
    (args.bench_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
