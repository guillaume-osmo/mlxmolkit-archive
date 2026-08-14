#!/usr/bin/env python3
"""Build deltaHvapv2 from AutoVap and calcphyschemprop.

AutoVap is treated as the trusted target. calcphyschemprop deltaHvap is joined
on canonical SMILES to create an alignment overlap, audited for drift/outliers,
and used as a teacher/source only. The coordinated homoset is the union: it
keeps AutoVap labels where available and uses a calibrated calcphyschemprop
pseudo-label only for calc-only molecules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem


DEFAULT_AUTOVAP = Path("data/autovap/source/AutoVapOnline/Datasets/Database-Global.csv")
DEFAULT_CALC = Path("/Users/guillaume-osmo/Github/osmo/src/runway/physchemprops/models/deltaHvap/deltaHvap_predictions.csv")
DEFAULT_HOMOSET_CODE = Path(
    "/Users/guillaume-osmo/Github/osmo/src/sandbox/guillaume/formulabloomclassifier/homoset_alignment.py"
)
DEFAULT_OUT = Path("data/delta_hvap_v2")


def canonical_smiles(smiles: str, *, isomeric: bool = True) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def fragment_count(smiles: object) -> int:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return -1
    return len(Chem.GetMolFrags(mol))


def load_homoset_ps_critical(path: Path):
    """Load the existing Homoset PS threshold function when available."""
    if path.exists():
        spec = importlib.util.spec_from_file_location("osmo_homoset_alignment", path)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "compute_ps_critical"):
                return mod.compute_ps_critical, str(path)

    def compute_ps_critical(n: int, alpha: float = 0.008) -> float:
        from scipy.stats import chi2

        if n < 2:
            return float("nan")
        y = chi2.ppf(alpha, df=n - 1)
        return 1.0 - math.sqrt(y / (3.0 * n))

    return compute_ps_critical, "local_fallback"


def join_strings(values: Iterable[object]) -> str:
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return "|".join(out)


def counts_string(values: Iterable[object]) -> str:
    counts = pd.Series([str(v).strip() for v in values if not pd.isna(v)]).value_counts()
    return "|".join(f"{k}:{int(v)}" for k, v in counts.items())


def number_list_string(values: Iterable[object]) -> str:
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        out.append(f"{float(value):.12g}")
    return "|".join(out)


def parse_number_list(text: object) -> list[float]:
    if pd.isna(text):
        return []
    vals: list[float] = []
    for part in str(text).split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            continue
    return vals


def load_autovap(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"SMILES", "dvap"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"AutoVap file {path} is missing columns: {missing}")

    df = df.copy()
    df["autovap_row_index"] = np.arange(len(df), dtype=np.int64)
    df["canonical_smiles"] = df["SMILES"].map(canonical_smiles)
    df = df[df["canonical_smiles"] != ""].copy()
    df["n_fragments"] = df["SMILES"].map(fragment_count)
    df["dvap"] = pd.to_numeric(df["dvap"], errors="coerce")
    df = df[df["dvap"].notna()].copy()

    group = df.groupby("canonical_smiles", sort=True)
    agg = group.agg(
        autovap_dvap_kJmol=("dvap", "mean"),
        autovap_n_rows=("dvap", "size"),
        autovap_std_kJmol=("dvap", "std"),
        autovap_min_kJmol=("dvap", "min"),
        autovap_max_kJmol=("dvap", "max"),
        autovap_values_kJmol=("dvap", number_list_string),
        autovap_smiles=("SMILES", join_strings),
        autovap_n_fragments=("n_fragments", "max"),
        autovap_row_indices=("autovap_row_index", lambda s: "|".join(map(str, s))),
    ).reset_index()
    agg["autovap_std_kJmol"] = agg["autovap_std_kJmol"].fillna(0.0)
    agg["autovap_range_kJmol"] = agg["autovap_max_kJmol"] - agg["autovap_min_kJmol"]

    for col, out_col in [
        ("CAS", "autovap_cas"),
        ("Key", "autovap_inchikey"),
        ("Family", "autovap_family"),
    ]:
        if col in df.columns:
            extra = group[col].agg(join_strings).reset_index(name=out_col)
            agg = agg.merge(extra, on="canonical_smiles", how="left")
    for col, out_col in [("External", "autovap_external_counts"), ("VOC", "autovap_voc_counts")]:
        if col in df.columns:
            extra = group[col].agg(counts_string).reset_index(name=out_col)
            agg = agg.merge(extra, on="canonical_smiles", how="left")
    return df, agg


def load_calcphyschemprop(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"smiles", "y_true", "y_pred_final"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"calcphyschemprop file {path} is missing columns: {missing}")

    df = df.copy()
    df["calc_row_index"] = np.arange(len(df), dtype=np.int64)
    df["canonical_smiles"] = df["smiles"].map(canonical_smiles)
    df = df[df["canonical_smiles"] != ""].copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred_final"] = pd.to_numeric(df["y_pred_final"], errors="coerce")
    df = df[df["y_true"].notna() & df["y_pred_final"].notna()].copy()

    group = df.groupby("canonical_smiles", sort=True)
    agg = group.agg(
        calc_deltaHvap_source_kJmol=("y_true", "mean"),
        calc_deltaHvap_pred_kJmol=("y_pred_final", "mean"),
        calc_n_rows=("y_true", "size"),
        calc_source_std_kJmol=("y_true", "std"),
        calc_pred_std_kJmol=("y_pred_final", "std"),
        calc_smiles=("smiles", join_strings),
        calc_row_indices=("calc_row_index", lambda s: "|".join(map(str, s))),
    ).reset_index()
    agg["calc_source_std_kJmol"] = agg["calc_source_std_kJmol"].fillna(0.0)
    agg["calc_pred_std_kJmol"] = agg["calc_pred_std_kJmol"].fillna(0.0)
    return df, agg


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return {"slope": 1.0, "intercept": 0.0, "r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(math.sqrt(np.mean((y - pred) ** 2))),
    }


def apply_linear(x: np.ndarray | pd.Series, params: dict[str, float]) -> np.ndarray:
    return float(params["slope"]) * np.asarray(x, dtype=np.float64) + float(params["intercept"])


def metric_summary(diff: pd.Series) -> dict[str, float]:
    arr = np.asarray(diff.dropna(), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean_signed_kJmol": float(np.mean(arr)),
        "median_signed_kJmol": float(np.median(arr)),
        "mae_kJmol": float(np.mean(np.abs(arr))),
        "rmse_kJmol": float(math.sqrt(np.mean(arr**2))),
        "p90_abs_kJmol": float(np.quantile(np.abs(arr), 0.90)),
        "p95_abs_kJmol": float(np.quantile(np.abs(arr), 0.95)),
        "p99_abs_kJmol": float(np.quantile(np.abs(arr), 0.99)),
        "max_abs_kJmol": float(np.max(np.abs(arr))),
    }


def homoset_l_scale(*values: pd.Series) -> float:
    arr = np.concatenate([np.asarray(v.dropna(), dtype=np.float64) for v in values])
    if arr.size == 0:
        return 1.0
    return max(float((np.max(arr) - np.min(arr)) / 2.0), 0.5)


def reference_homoset_stats(
    reference: pd.Series,
    source: pd.Series,
    *,
    alpha: float,
    l_scale: float,
    ps_critical_fn,
) -> dict[str, float | int | bool]:
    """Homoset-style offset/PS gate with the trusted reference fixed.

    This mirrors the existing HomosetAligner equations but avoids its default
    "largest source is reference" rule, which is wrong for deltaHvapv2 because
    AutoVap is the trusted experimental source.
    """
    ref = np.asarray(reference, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    mask = np.isfinite(ref) & np.isfinite(src)
    diffs = src[mask] - ref[mask]
    if diffs.size < 2:
        return {
            "n_overlap": int(diffs.size),
            "offset_kJmol": float("nan"),
            "rms_after_offset_kJmol": float("nan"),
            "ps": float("nan"),
            "ps_critical": float("nan"),
            "accepted": False,
            "L_kJmol": float(l_scale),
            "alpha": float(alpha),
        }
    offset = float(np.median(diffs))
    rms = float(np.sqrt(np.mean((diffs - offset) ** 2)))
    ps = float(1.0 - rms / l_scale)
    ps_critical = float(ps_critical_fn(int(diffs.size), alpha))
    return {
        "n_overlap": int(diffs.size),
        "offset_kJmol": offset,
        "rms_after_offset_kJmol": rms,
        "ps": ps,
        "ps_critical": ps_critical,
        "accepted": bool(np.isfinite(ps) and np.isfinite(ps_critical) and ps >= ps_critical),
        "L_kJmol": float(l_scale),
        "alpha": float(alpha),
    }


def nearest_raw_autovap_value(row: pd.Series) -> float:
    values = parse_number_list(row.get("autovap_values_kJmol", ""))
    if not values:
        return float(row["autovap_dvap_kJmol"])
    anchors = [
        row.get("calc_deltaHvap_source_kJmol", np.nan),
        row.get("calc_deltaHvap_pred_kJmol", np.nan),
    ]
    anchors = [float(v) for v in anchors if pd.notna(v) and np.isfinite(float(v))]
    if not anchors:
        return float(np.median(values))
    anchor = float(np.median(anchors))
    return float(min(values, key=lambda v: abs(v - anchor)))


def add_outlier_curation(homoset: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add row-level curation columns for the AutoVap/calcphyschemprop overlap."""
    df = homoset.copy()
    source_conflict = (
        (df["abs_source_minus_autovap_kJmol"] >= threshold)
        | (df["abs_pred_minus_autovap_kJmol"] >= threshold)
    )
    duplicate_conflict = df["autovap_range_kJmol"] >= threshold

    df["is_homoset_outlier"] = source_conflict | duplicate_conflict
    df["is_source_conflict_outlier"] = source_conflict
    df["is_autovap_duplicate_conflict"] = duplicate_conflict
    df["curated_target_kJmol"] = df["autovap_dvap_kJmol"]
    df["curation_action"] = "keep_autovap"
    df["curation_note"] = ""

    dup_only = duplicate_conflict & ~source_conflict
    if dup_only.any():
        df.loc[dup_only, "curated_target_kJmol"] = df.loc[dup_only].apply(nearest_raw_autovap_value, axis=1)
        df.loc[dup_only, "curation_action"] = "resolve_autovap_duplicate_to_nearest_raw"
        df.loc[dup_only, "curation_note"] = (
            "canonical key merged conflicting AutoVap rows; selected raw AutoVap value nearest calcphyschemprop"
        )

    df.loc[source_conflict, "curation_action"] = "correct_calcphyschemprop_to_autovap"
    df.loc[source_conflict, "curation_note"] = (
        "calcphyschemprop source/pred disagrees with trusted AutoVap beyond threshold"
    )

    df["calc_deltaHvap_source_curated_kJmol"] = df["calc_deltaHvap_source_kJmol"]
    df["calc_deltaHvap_pred_curated_kJmol"] = df["calc_deltaHvap_pred_kJmol"]
    df.loc[source_conflict, "calc_deltaHvap_source_curated_kJmol"] = df.loc[source_conflict, "curated_target_kJmol"]
    df.loc[source_conflict, "calc_deltaHvap_pred_curated_kJmol"] = df.loc[source_conflict, "curated_target_kJmol"]
    df["source_curated_minus_target_kJmol"] = (
        df["calc_deltaHvap_source_curated_kJmol"] - df["curated_target_kJmol"]
    )
    df["pred_curated_minus_target_kJmol"] = (
        df["calc_deltaHvap_pred_curated_kJmol"] - df["curated_target_kJmol"]
    )
    df["abs_source_curated_minus_target_kJmol"] = df["source_curated_minus_target_kJmol"].abs()
    df["abs_pred_curated_minus_target_kJmol"] = df["pred_curated_minus_target_kJmol"].abs()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autovap", type=Path, default=DEFAULT_AUTOVAP)
    parser.add_argument("--calcphyschemprop", type=Path, default=DEFAULT_CALC)
    parser.add_argument("--homoset-code", type=Path, default=DEFAULT_HOMOSET_CODE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--outlier-threshold-kjmol", type=float, default=10.0)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument("--homoset-alpha", type=float, default=0.008)
    args = parser.parse_args()

    auto_rows, auto = load_autovap(args.autovap)
    calc_rows, calc = load_calcphyschemprop(args.calcphyschemprop)
    ps_critical_fn, homoset_code_used = load_homoset_ps_critical(args.homoset_code)

    autovap_multifragment = auto[auto["autovap_n_fragments"] != 1].copy()
    auto = auto[auto["autovap_n_fragments"] == 1].copy()

    homoset = auto.merge(calc, on="canonical_smiles", how="inner")
    autovap_only = auto.merge(calc[["canonical_smiles"]], on="canonical_smiles", how="left", indicator=True)
    autovap_only = autovap_only[autovap_only["_merge"] == "left_only"].drop(columns=["_merge"])
    calc_only = calc.merge(auto[["canonical_smiles"]], on="canonical_smiles", how="left", indicator=True)
    calc_only = calc_only[calc_only["_merge"] == "left_only"].drop(columns=["_merge"])

    homoset["source_minus_autovap_kJmol"] = homoset["calc_deltaHvap_source_kJmol"] - homoset["autovap_dvap_kJmol"]
    homoset["pred_minus_autovap_kJmol"] = homoset["calc_deltaHvap_pred_kJmol"] - homoset["autovap_dvap_kJmol"]
    homoset["abs_source_minus_autovap_kJmol"] = homoset["source_minus_autovap_kJmol"].abs()
    homoset["abs_pred_minus_autovap_kJmol"] = homoset["pred_minus_autovap_kJmol"].abs()

    l_source = homoset_l_scale(homoset["autovap_dvap_kJmol"], homoset["calc_deltaHvap_source_kJmol"])
    l_pred = homoset_l_scale(homoset["autovap_dvap_kJmol"], homoset["calc_deltaHvap_pred_kJmol"])
    homoset_source_stats = reference_homoset_stats(
        homoset["autovap_dvap_kJmol"],
        homoset["calc_deltaHvap_source_kJmol"],
        alpha=float(args.homoset_alpha),
        l_scale=l_source,
        ps_critical_fn=ps_critical_fn,
    )
    homoset_pred_stats = reference_homoset_stats(
        homoset["autovap_dvap_kJmol"],
        homoset["calc_deltaHvap_pred_kJmol"],
        alpha=float(args.homoset_alpha),
        l_scale=l_pred,
        ps_critical_fn=ps_critical_fn,
    )
    source_offset = float(homoset_source_stats["offset_kJmol"])
    pred_offset = float(homoset_pred_stats["offset_kJmol"])
    homoset["calc_deltaHvap_source_homoset_offset_kJmol"] = source_offset
    homoset["calc_deltaHvap_pred_homoset_offset_kJmol"] = pred_offset
    homoset["calc_deltaHvap_source_homoset_aligned_kJmol"] = homoset["calc_deltaHvap_source_kJmol"] - source_offset
    homoset["calc_deltaHvap_pred_homoset_aligned_kJmol"] = homoset["calc_deltaHvap_pred_kJmol"] - pred_offset
    homoset["source_homoset_aligned_minus_autovap_kJmol"] = (
        homoset["calc_deltaHvap_source_homoset_aligned_kJmol"] - homoset["autovap_dvap_kJmol"]
    )
    homoset["pred_homoset_aligned_minus_autovap_kJmol"] = (
        homoset["calc_deltaHvap_pred_homoset_aligned_kJmol"] - homoset["autovap_dvap_kJmol"]
    )
    homoset["homoset_consensus_source_kJmol"] = homoset[
        ["autovap_dvap_kJmol", "calc_deltaHvap_source_homoset_aligned_kJmol"]
    ].median(axis=1)
    homoset["homoset_consensus_pred_kJmol"] = homoset[
        ["autovap_dvap_kJmol", "calc_deltaHvap_pred_homoset_aligned_kJmol"]
    ].median(axis=1)

    threshold = float(args.outlier_threshold_kjmol)
    homoset["is_source_outlier"] = homoset["abs_source_minus_autovap_kJmol"] >= threshold
    homoset["is_pred_outlier"] = homoset["abs_pred_minus_autovap_kJmol"] >= threshold
    homoset = add_outlier_curation(homoset, threshold)
    homoset["trusted_target_kJmol"] = homoset["curated_target_kJmol"]
    homoset["teacher_deltaHvap_kJmol"] = homoset["calc_deltaHvap_pred_curated_kJmol"]
    homoset["split_group"] = "homoset"

    outliers = homoset[homoset["is_homoset_outlier"]].copy()
    outliers = outliers.sort_values(
        ["abs_source_minus_autovap_kJmol", "abs_pred_minus_autovap_kJmol", "autovap_range_kJmol"],
        ascending=False,
    )
    outliers_corrected = outliers.copy()

    duplicate_conflicts = auto[auto["autovap_n_rows"] > 1].copy()
    duplicate_conflicts = duplicate_conflicts.sort_values("autovap_range_kJmol", ascending=False)

    inlier = homoset["abs_source_minus_autovap_kJmol"] < threshold
    calibration_all = fit_linear(homoset["calc_deltaHvap_source_kJmol"].to_numpy(), homoset["autovap_dvap_kJmol"].to_numpy())
    calibration_inlier = fit_linear(
        homoset.loc[inlier, "calc_deltaHvap_source_kJmol"].to_numpy(),
        homoset.loc[inlier, "autovap_dvap_kJmol"].to_numpy(),
    )

    auto_union = auto.copy()
    curation_by_key = homoset.set_index("canonical_smiles")
    target_map = curation_by_key["curated_target_kJmol"].to_dict()
    action_map = curation_by_key["curation_action"].to_dict()
    note_map = curation_by_key["curation_note"].to_dict()
    auto_union["trusted_target_kJmol"] = auto_union["canonical_smiles"].map(target_map).fillna(auto_union["autovap_dvap_kJmol"])
    auto_union["curated_target_kJmol"] = auto_union["trusted_target_kJmol"]
    auto_union["curation_action"] = auto_union["canonical_smiles"].map(action_map).fillna("keep_autovap")
    auto_union["curation_note"] = auto_union["canonical_smiles"].map(note_map).fillna("")
    auto_union["target_source"] = "autovap_trusted"
    auto_union["sample_weight"] = 1.0
    for col in [
        "calc_deltaHvap_source_kJmol",
        "calc_deltaHvap_pred_kJmol",
        "calc_deltaHvap_source_homoset_aligned_kJmol",
        "calc_deltaHvap_pred_homoset_aligned_kJmol",
        "calc_deltaHvap_source_curated_kJmol",
        "calc_deltaHvap_pred_curated_kJmol",
        "calc_smiles",
    ]:
        auto_union[col] = auto_union["canonical_smiles"].map(curation_by_key[col].to_dict())
    auto_union["pseudo_calibration"] = ""

    calc_pseudo = calc_only.copy()
    calc_pseudo["trusted_target_kJmol"] = calc_pseudo["calc_deltaHvap_source_kJmol"] - source_offset
    calc_pseudo["curated_target_kJmol"] = calc_pseudo["trusted_target_kJmol"]
    calc_pseudo["calc_deltaHvap_source_homoset_aligned_kJmol"] = calc_pseudo["calc_deltaHvap_source_kJmol"] - source_offset
    calc_pseudo["calc_deltaHvap_pred_homoset_aligned_kJmol"] = calc_pseudo["calc_deltaHvap_pred_kJmol"] - pred_offset
    calc_pseudo["calc_deltaHvap_source_curated_kJmol"] = calc_pseudo["calc_deltaHvap_source_homoset_aligned_kJmol"]
    calc_pseudo["calc_deltaHvap_pred_curated_kJmol"] = calc_pseudo["calc_deltaHvap_pred_homoset_aligned_kJmol"]
    calc_pseudo["curation_action"] = "pseudo_from_calcphyschemprop_homoset_offset"
    calc_pseudo["curation_note"] = "calcphyschemprop-only molecule; no AutoVap overlap for outlier correction"
    calc_pseudo["target_source"] = "calcphyschemprop_calibrated_pseudo"
    calc_pseudo["sample_weight"] = float(args.pseudo_weight)
    calc_pseudo["pseudo_calibration"] = "homoset_offset_source_to_autovap"
    for col in [
        "autovap_dvap_kJmol", "autovap_n_rows", "autovap_std_kJmol", "autovap_min_kJmol",
        "autovap_max_kJmol", "autovap_values_kJmol", "autovap_range_kJmol", "autovap_smiles",
        "autovap_n_fragments", "autovap_row_indices",
        "autovap_cas", "autovap_inchikey", "autovap_family", "autovap_external_counts",
        "autovap_voc_counts",
    ]:
        if col not in calc_pseudo.columns:
            calc_pseudo[col] = np.nan if "kJmol" in col or col.endswith("_rows") else ""

    union_cols = [
        "canonical_smiles",
        "trusted_target_kJmol",
        "curated_target_kJmol",
        "target_source",
        "sample_weight",
        "pseudo_calibration",
        "curation_action",
        "curation_note",
        "autovap_dvap_kJmol",
        "calc_deltaHvap_source_kJmol",
        "calc_deltaHvap_pred_kJmol",
        "calc_deltaHvap_source_homoset_aligned_kJmol",
        "calc_deltaHvap_pred_homoset_aligned_kJmol",
        "calc_deltaHvap_source_curated_kJmol",
        "calc_deltaHvap_pred_curated_kJmol",
        "autovap_n_rows",
        "autovap_std_kJmol",
        "autovap_range_kJmol",
        "autovap_n_fragments",
        "autovap_cas",
        "autovap_inchikey",
        "autovap_family",
        "autovap_smiles",
        "calc_smiles",
    ]
    union = pd.concat([auto_union, calc_pseudo], axis=0, ignore_index=True, sort=False)
    for col in union_cols:
        if col not in union.columns:
            union[col] = ""
    union = union[union_cols].sort_values(["target_source", "canonical_smiles"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    homoset_path = args.out_dir / "deltaHvapv2_homoset.csv"
    alignment_overlap_path = args.out_dir / "deltaHvapv2_alignment_overlap.csv"
    union_path = args.out_dir / "deltaHvapv2_union.csv"
    homoset_union_path = args.out_dir / "deltaHvapv2_homoset_union.csv"
    outliers_path = args.out_dir / "deltaHvapv2_outliers.csv"
    outliers_corrected_path = args.out_dir / "deltaHvapv2_outliers_corrected.csv"
    auto_only_path = args.out_dir / "deltaHvapv2_autovap_only.csv"
    calc_only_path = args.out_dir / "deltaHvapv2_calcphyschemprop_only.csv"
    dup_path = args.out_dir / "deltaHvapv2_autovap_duplicate_conflicts.csv"
    multifrag_path = args.out_dir / "deltaHvapv2_autovap_multifragment_excluded.csv"
    align_path = args.out_dir / "deltaHvapv2_homoset_alignment.csv"
    summary_path = args.out_dir / "deltaHvapv2_summary.json"

    homoset.to_csv(homoset_path, index=False)
    homoset.to_csv(alignment_overlap_path, index=False)
    union.to_csv(union_path, index=False)
    union.to_csv(homoset_union_path, index=False)
    outliers.to_csv(outliers_path, index=False)
    outliers_corrected.to_csv(outliers_corrected_path, index=False)
    autovap_only.to_csv(auto_only_path, index=False)
    calc_only.to_csv(calc_only_path, index=False)
    duplicate_conflicts.to_csv(dup_path, index=False)
    autovap_multifragment.to_csv(multifrag_path, index=False)
    pd.DataFrame(
        [
            {"source": "calcphyschemprop_source", **homoset_source_stats},
            {"source": "calcphyschemprop_pred", **homoset_pred_stats},
        ]
    ).to_csv(align_path, index=False)

    summary = {
        "autovap_path": str(args.autovap),
        "calcphyschemprop_path": str(args.calcphyschemprop),
        "homoset_code_path": homoset_code_used,
        "homoset_reference_source": "autovap_trusted",
        "autovap_rows": int(len(auto_rows)),
        "autovap_multifragment_excluded_unique_canonical": int(len(autovap_multifragment)),
        "autovap_unique_canonical": int(len(auto)),
        "calcphyschemprop_rows": int(len(calc_rows)),
        "calcphyschemprop_unique_canonical": int(len(calc)),
        "alignment_overlap_unique_canonical": int(len(homoset)),
        "homoset_unique_canonical": int(len(union)),
        "autovap_only_unique_canonical": int(len(autovap_only)),
        "calcphyschemprop_only_unique_canonical": int(len(calc_only)),
        "union_rows": int(len(union)),
        "homoset_union_rows": int(len(union)),
        "outlier_threshold_kJmol": threshold,
        "outliers_rows": int(len(outliers)),
        "outliers_source_conflict_rows": int(outliers["is_source_conflict_outlier"].sum()),
        "outliers_autovap_duplicate_conflict_rows": int(outliers["is_autovap_duplicate_conflict"].sum()),
        "outliers_remaining_after_curation_rows": int(
            (
                (outliers["abs_source_curated_minus_target_kJmol"] >= threshold)
                | (outliers["abs_pred_curated_minus_target_kJmol"] >= threshold)
            ).sum()
        ),
        "outlier_curation_actions": {
            str(k): int(v) for k, v in outliers["curation_action"].value_counts(dropna=False).items()
        },
        "autovap_duplicate_canonical": int((auto["autovap_n_rows"] > 1).sum()),
        "autovap_duplicate_conflict_ge_threshold": int((duplicate_conflicts["autovap_range_kJmol"] >= threshold).sum()),
        "source_minus_autovap": metric_summary(homoset["source_minus_autovap_kJmol"]),
        "pred_minus_autovap": metric_summary(homoset["pred_minus_autovap_kJmol"]),
        "source_curated_minus_target": metric_summary(homoset["source_curated_minus_target_kJmol"]),
        "pred_curated_minus_target": metric_summary(homoset["pred_curated_minus_target_kJmol"]),
        "source_homoset_aligned_minus_autovap": metric_summary(homoset["source_homoset_aligned_minus_autovap_kJmol"]),
        "pred_homoset_aligned_minus_autovap": metric_summary(homoset["pred_homoset_aligned_minus_autovap_kJmol"]),
        "homoset_ps_gating": {
            "calcphyschemprop_source": homoset_source_stats,
            "calcphyschemprop_pred": homoset_pred_stats,
        },
        "calibration_source_to_autovap_all": calibration_all,
        "calibration_source_to_autovap_inlier": calibration_inlier,
        "pseudo_weight": float(args.pseudo_weight),
        "pseudo_calibration_default": "homoset_offset_source_to_autovap",
        "outputs": {
            "alignment_overlap": str(alignment_overlap_path),
            "homoset_legacy_overlap": str(homoset_path),
            "homoset_union": str(homoset_union_path),
            "union": str(union_path),
            "outliers": str(outliers_path),
            "outliers_corrected": str(outliers_corrected_path),
            "autovap_only": str(auto_only_path),
            "calcphyschemprop_only": str(calc_only_path),
            "duplicate_conflicts": str(dup_path),
            "autovap_multifragment_excluded": str(multifrag_path),
            "homoset_alignment": str(align_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    if len(outliers):
        cols = [
            "canonical_smiles", "autovap_dvap_kJmol", "calc_deltaHvap_source_kJmol",
            "source_minus_autovap_kJmol", "autovap_range_kJmol", "autovap_family",
        ]
        print("\nTop outliers:")
        print(outliers[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
