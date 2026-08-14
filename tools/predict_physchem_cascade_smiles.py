#!/usr/bin/env python3
"""Predict the physchemprop cascade for a small SMILES table and join source labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OSMO_CASCADE = Path(
    "/Users/guillaume-osmo/Github/osmo/src/runway/physchemprops/train_all_25_models_v34_xgboost.py"
)
DEFAULT_MODEL_ROOT = Path("/Users/guillaume-osmo/Github/osmo/src/runway/physchemprops/models")
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks/physchem_cascade_smiles"
DEFAULT_DELTA_HSUB_MODEL = REPO_ROOT / "data/sublimation/deltaHsub_target30_model.joblib"


def canonical(smiles: object, *, isomeric: bool = True) -> str:
    if pd.isna(smiles):
        return ""
    text = str(smiles).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown", "unknownsmiles"}:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def load_cascade_module(path: Path, model_root: Path):
    spec = importlib.util.spec_from_file_location("osmo_physchem_cascade", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import cascade module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUTPUT_DIR = model_root
    mod.CURRENT_VERSION = model_root.name
    return mod


def find_smiles_col(df: pd.DataFrame) -> str | None:
    for col in ("SMILES", "smiles", "cansmi", "CanonicalSMILES", "canonical_smiles"):
        if col in df.columns:
            return col
    return None


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


@lru_cache(maxsize=None)
def load_csv_cached(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text, low_memory=False)


def target_column(mod: Any, target: str, preferred: str, df: pd.DataFrame) -> str | None:
    if preferred in df.columns:
        return preferred
    return mod.find_column_name(target, df.columns)


def add_canonical_columns(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    out = df.copy()
    smiles = out[smiles_col].astype(str)
    out["_canon_iso"] = [canonical(s, isomeric=True) for s in smiles]
    out["_canon_nostereo"] = [canonical(s, isomeric=False) for s in smiles]
    out = out[out["_canon_iso"] != ""].copy()
    return out


def lookup_values(df: pd.DataFrame, row: pd.Series, value_col: str) -> tuple[pd.DataFrame, str]:
    iso = str(row["canonical_smiles"])
    nostereo = str(row["canonical_smiles_nostereo"])
    exact = df[df["_canon_iso"].eq(iso)]
    if not exact.empty:
        return exact, "isomeric"
    loose = df[df["_canon_nostereo"].eq(nostereo)]
    return loose, "no_stereo" if not loose.empty else "missing"


def summarize_values(values: list[float], source: str, match: str) -> dict[str, object]:
    if not values:
        return {"true_mean": np.nan, "true_min": np.nan, "true_max": np.nan, "true_n": 0, "true_source": "", "true_match": "missing"}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "true_mean": float(np.mean(arr)),
        "true_min": float(np.min(arr)),
        "true_max": float(np.max(arr)),
        "true_n": int(arr.shape[0]),
        "true_source": source,
        "true_match": match,
    }


def _descriptor_function_map() -> dict[str, Any]:
    return {name: func for name, func in Descriptors.descList}


def predict_delta_hsub_target30(
    input_df: pd.DataFrame,
    pred_map: dict[str, dict[str, float]],
    model_path: Path,
) -> dict[int, float]:
    if not model_path.exists():
        return {}

    artifact = joblib.load(model_path)
    feature_names = [str(x) for x in artifact["feature_names"]]
    prep = artifact["prep"]
    keep_mask = np.asarray(prep["keep_mask"], dtype=bool)
    arcsinh_mask = np.asarray(prep["arcsinh_mask"], dtype=bool)
    threshold = float(prep["threshold"])
    model = artifact["model"]
    desc_map = _descriptor_function_map()

    rows: list[list[float]] = []
    indices: list[int] = []
    for i, row in input_df.iterrows():
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            continue
        preds = pred_map.get(smi, {})
        values: list[float] = []
        for name in feature_names:
            if name.startswith("rdkit_"):
                desc_name = name.removeprefix("rdkit_")
                func = desc_map.get(desc_name)
                if func is None:
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(func(mol)))
                    except Exception:
                        values.append(np.nan)
            elif name.startswith("pred_"):
                target = name.removeprefix("pred_")
                values.append(float(preds.get(target, np.nan)))
            else:
                values.append(np.nan)
        rows.append(values)
        indices.append(int(i))

    if not rows:
        return {}
    x = np.asarray(rows, dtype=np.float64)
    x = x[:, keep_mask].copy()
    x[~np.isfinite(x)] = 0.0
    if np.any(arcsinh_mask):
        x[:, arcsinh_mask] = np.arcsinh(x[:, arcsinh_mask] / threshold)
    pred = np.asarray(model.predict(x.astype(np.float32)), dtype=np.float64)
    return {idx: float(val) for idx, val in zip(indices, pred)}


def collect_truth_from_cascade_datasets(mod: Any, input_df: pd.DataFrame) -> dict[tuple[int, str], dict[str, object]]:
    truth: dict[tuple[int, str], dict[str, object]] = {}
    prepared: dict[tuple[str, str], pd.DataFrame] = {}

    for target, file_key, preferred_col, _deps in mod.CASCADE_ORDER:
        path = Path(mod.DATA_PATHS[file_key])
        if not path.exists():
            for i in input_df.index:
                truth[(i, target)] = summarize_values([], "", "missing")
            continue

        df = load_csv_cached(str(path))
        smiles_col = find_smiles_col(df)
        if smiles_col is None:
            for i in input_df.index:
                truth[(i, target)] = summarize_values([], str(path), "missing")
            continue

        col = target_column(mod, target, preferred_col, df)
        work = df
        source_note = f"{path.name}:{col or preferred_col}"

        if target == "HansenTotal" and col is None:
            d_cols = [mod.find_column_name(name, df.columns) for name in ("dD", "dH", "dP")]
            if all(d_cols):
                work = df.copy()
                vals = [
                    pd.to_numeric(work[d_cols[0]], errors="coerce"),
                    pd.to_numeric(work[d_cols[1]], errors="coerce"),
                    pd.to_numeric(work[d_cols[2]], errors="coerce"),
                ]
                work["_computed_HansenTotal"] = np.sqrt(vals[0] ** 2 + vals[1] ** 2 + vals[2] ** 2)
                col = "_computed_HansenTotal"
                source_note = f"{path.name}:computed_from_dD_dH_dP"

        if col is None or col not in work.columns or str(preferred_col).startswith("_COMPUTED_"):
            for i in input_df.index:
                truth[(i, target)] = summarize_values([], str(path), "missing")
            continue

        cache_key = (str(path), smiles_col)
        if cache_key not in prepared:
            prepared[cache_key] = add_canonical_columns(work, smiles_col)
        else:
            # Keep the latest computed column if needed.
            if col not in prepared[cache_key].columns and col in work.columns:
                prepared[cache_key] = add_canonical_columns(work, smiles_col)
        canon_df = prepared[cache_key]
        vals_numeric = pd.to_numeric(canon_df[col], errors="coerce")
        canon_df = canon_df.assign(_target_value=vals_numeric).dropna(subset=["_target_value"])

        for i, row in input_df.iterrows():
            hits, match = lookup_values(canon_df, row, "_target_value")
            values = [v for v in (finite_float(x) for x in hits["_target_value"].tolist()) if v is not None]
            truth[(i, target)] = summarize_values(values, source_note, match)

    # Add deterministic computed modularity as an extra "true" value for that computed target.
    if hasattr(mod, "compute_molecular_modularity"):
        smiles = input_df["smiles"].astype(str).tolist()
        try:
            values, valid_idx = mod.compute_molecular_modularity(smiles)
            for value, local_idx in zip(values, valid_idx):
                i = int(input_df.index[local_idx])
                truth[(i, "Modularity")] = summarize_values(
                    [float(value)], "computed_rdkit_modularity", "computed"
                )
        except Exception:
            pass

    return truth


def collect_extra_delta_hvap_truth(input_df: pd.DataFrame) -> dict[tuple[int, str], dict[str, object]]:
    extras: dict[tuple[int, str], dict[str, object]] = {}
    sources = [
        (
            "deltaHvap_AutoVap",
            REPO_ROOT / "data/autovap/source/AutoVapOnline/Datasets/Database-Global.csv",
            "dvap",
        ),
        (
            "deltaHvap_MDPI_exp",
            REPO_ROOT / "benchmarks/delta_hvap_v2_mdpi_source_check/mdpi_thermo_reconstructed.csv",
            "deltaH_vap_exp_s06",
        ),
        (
            "deltaHvap_MDPI_calc",
            REPO_ROOT / "benchmarks/delta_hvap_v2_mdpi_source_check/mdpi_thermo_reconstructed.csv",
            "deltaH_vap_calc_s06",
        ),
        (
            "deltaHvap_calcphyschemprop_y_true",
            Path("/Users/guillaume-osmo/Github/osmo/src/runway/physchemprops/models/deltaHvap/deltaHvap_predictions.csv"),
            "y_true",
        ),
    ]
    for target, path, value_col in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        smiles_col = find_smiles_col(df)
        if smiles_col is None or value_col not in df.columns:
            continue
        canon_df = add_canonical_columns(df, smiles_col)
        canon_df = canon_df.assign(_target_value=pd.to_numeric(canon_df[value_col], errors="coerce")).dropna(
            subset=["_target_value"]
        )
        for i, row in input_df.iterrows():
            hits, match = lookup_values(canon_df, row, "_target_value")
            values = [v for v in (finite_float(x) for x in hits["_target_value"].tolist()) if v is not None]
            extras[(i, target)] = summarize_values(values, f"{path.name}:{value_col}", match)
    return extras


def collect_input_delta_hsub_truth(input_df: pd.DataFrame) -> dict[tuple[int, str], dict[str, object]]:
    for col in ("deltaHsub_kJmol", "sublimation_enthalpy_kj/mol", "sublimation enthalpy_kj/mol"):
        if col in input_df.columns:
            out: dict[tuple[int, str], dict[str, object]] = {}
            vals = pd.to_numeric(input_df[col], errors="coerce")
            for i, value in vals.items():
                if finite_float(value) is None:
                    out[(int(i), "deltaHsub")] = summarize_values([], "", "missing")
                else:
                    out[(int(i), "deltaHsub")] = summarize_values([float(value)], f"input:{col}", "input")
            return out
    return {}


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with at least smiles; optional code/vendor/name.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cascade-module", type=Path, default=DEFAULT_OSMO_CASCADE)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--delta-hsub-model", type=Path, default=DEFAULT_DELTA_HSUB_MODEL)
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    if "smiles" not in input_df.columns:
        raise KeyError("--input must contain a smiles column")
    input_df = input_df.copy()
    input_df["canonical_smiles"] = [canonical(s, isomeric=True) for s in input_df["smiles"]]
    input_df["canonical_smiles_nostereo"] = [canonical(s, isomeric=False) for s in input_df["smiles"]]

    mod = load_cascade_module(args.cascade_module, args.model_root)
    pred_map = mod.run_cascade_inference(input_df["smiles"].astype(str).tolist(), up_to_target=None)
    delta_hsub = predict_delta_hsub_target30(input_df, pred_map, args.delta_hsub_model)
    truth = collect_truth_from_cascade_datasets(mod, input_df)
    truth.update(collect_extra_delta_hvap_truth(input_df))
    truth.update(collect_input_delta_hsub_truth(input_df))

    target_names = ["MW", "MR"] + [target for target, *_ in mod.CASCADE_ORDER]
    if delta_hsub:
        target_names.append("deltaHsub")
    extra_truth_targets = sorted({target for (_i, target) in truth if target not in target_names})
    all_truth_targets = target_names + extra_truth_targets

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for i, row in input_df.iterrows():
        smi = str(row["smiles"])
        preds = pred_map.get(smi, {})
        base = row.to_dict()
        wide = dict(base)
        for target in target_names:
            if target == "deltaHsub":
                wide[f"pred_{target}"] = delta_hsub.get(int(i), np.nan)
            else:
                wide[f"pred_{target}"] = preds.get(target, np.nan)
        for target in all_truth_targets:
            info = truth.get((i, target), summarize_values([], "", "missing"))
            wide[f"true_{target}"] = info["true_mean"]
            wide[f"true_{target}_n"] = info["true_n"]
            wide[f"true_{target}_source"] = info["true_source"]
            wide[f"true_{target}_match"] = info["true_match"]
        wide_rows.append(wide)

        for target in target_names:
            info = truth.get((i, target), summarize_values([], "", "missing"))
            if target == "deltaHsub":
                pred = finite_float(delta_hsub.get(int(i), np.nan))
            else:
                pred = finite_float(preds.get(target, np.nan))
            true_mean = finite_float(info["true_mean"])
            long = {
                **base,
                "target": target,
                "pred": pred if pred is not None else np.nan,
                **info,
                "abs_error": abs(pred - true_mean) if pred is not None and true_mean is not None else np.nan,
            }
            long_rows.append(long)
        for target in extra_truth_targets:
            info = truth.get((i, target), summarize_values([], "", "missing"))
            long_rows.append({**base, "target": target, "pred": np.nan, **info, "abs_error": np.nan})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wide_path = args.out_dir / "physchem_cascade_predictions_wide.csv"
    long_path = args.out_dir / "physchem_cascade_predictions_long.csv"
    pd.DataFrame(wide_rows).to_csv(wide_path, index=False)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)

    summary = {
        "input": str(args.input),
        "out_dir": str(args.out_dir),
        "n_molecules": int(len(input_df)),
        "n_prediction_targets": int(len(target_names)),
        "prediction_targets": target_names,
        "extra_truth_targets": extra_truth_targets,
        "delta_hsub_model": str(args.delta_hsub_model) if args.delta_hsub_model.exists() else "",
        "wide_csv": str(wide_path),
        "long_csv": str(long_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
