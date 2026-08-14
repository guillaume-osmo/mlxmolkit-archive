#!/usr/bin/env python3
"""Train/evaluate deltaHvapv2 using the coordinated Homoset union correctly.

Evaluation is always on trusted AutoVap labels in the AutoVap×calcphyschemprop
alignment overlap. calcphyschemprop-only rows in the Homoset union may be used
as lower-weight pseudo-labels in the training fold, never in validation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


DEFAULT_DATA_DIR = Path("data/delta_hvap_v2")
DEFAULT_OUT_DIR = Path("benchmarks/delta_hvap_v2_homoset")
DEFAULT_GXTB_SIGMA = DEFAULT_DATA_DIR / "deltaHvapv2_gxtb_tmcosmo_all_sigma.npz"
TRUSTED_TARGET_SOURCES = {"autovap_trusted", "deltaHvapv3_experimental_new"}


@dataclass
class FoldPrep:
    keep_mask: np.ndarray
    arcsinh_mask: np.ndarray
    threshold: float


def canonical_smiles(smiles: str, *, isomeric: bool = True) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def first_smiles(row: pd.Series) -> str:
    for col in ("autovap_smiles", "calc_smiles", "canonical_smiles"):
        val = row.get(col, "")
        if pd.notna(val):
            text = str(val).strip()
            if text:
                return text.split("|")[0]
    return ""


def descriptor_names() -> list[str]:
    return [name for name, _func in Descriptors._descList]


def compute_rdkit_features(smiles: Sequence[str], cache_path: Path) -> tuple[np.ndarray, list[str]]:
    names = descriptor_names()
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_smiles = [str(x) for x in z["smiles"].tolist()]
        cached_names = [str(x) for x in z["names"].tolist()]
        if cached_smiles == list(smiles) and cached_names == names:
            return np.asarray(z["x"], dtype=np.float32), names

    calc = MolecularDescriptorCalculator(names)
    x = np.full((len(smiles), len(names)), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            vals = calc.CalcDescriptors(mol)
        except Exception:
            continue
        x[i] = np.asarray(vals, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, smiles=np.asarray(smiles, dtype=object), names=np.asarray(names, dtype=object))
    return x, names


def load_chaos_sigma(path: Path) -> tuple[dict[str, int], dict[str, int], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    smiles = np.asarray(z["canonical_smiles"]).astype(str)
    mu = np.asarray(z["mu_J_per_mol"], dtype=np.float32)
    grid = np.asarray(z["sigma_grid_e_per_A2"], dtype=np.float64)
    iso: dict[str, int] = {}
    no_stereo: dict[str, int] = {}
    for i, smi in enumerate(smiles):
        if not smi:
            continue
        iso[smi] = i
        ns = canonical_smiles(smi, isomeric=False)
        if ns:
            no_stereo.setdefault(ns, i)
    return iso, no_stereo, mu, grid


def attach_sigma(smiles: Sequence[str], chaos_matrix: Path) -> tuple[np.ndarray, list[str], dict[str, int]]:
    iso, no_stereo, mu, grid = load_chaos_sigma(chaos_matrix)
    x = np.zeros((len(smiles), mu.shape[1] + 1), dtype=np.float32)
    modes: dict[str, int] = {"isomeric": 0, "no_stereo": 0, "missing": 0}
    for i, smi in enumerate(smiles):
        hit = -1
        canon_iso = canonical_smiles(smi, isomeric=True)
        canon_no_stereo = canonical_smiles(smi, isomeric=False)
        if canon_iso in iso:
            hit = iso[canon_iso]
            modes["isomeric"] += 1
        elif canon_no_stereo in no_stereo:
            hit = no_stereo[canon_no_stereo]
            modes["no_stereo"] += 1
        else:
            modes["missing"] += 1
        if hit >= 0:
            x[i, : mu.shape[1]] = mu[hit]
            x[i, -1] = 1.0
    names = [f"chaos_mu_{j:02d}" for j in range(mu.shape[1])] + ["has_chaos_sigma"]
    return x, names, {"grid_min": float(grid[0]), "grid_max": float(grid[-1]), **modes}


def attach_row_aligned_gxtb_sigma(
    df: pd.DataFrame,
    sigma_npz: Path,
    *,
    include_profile: bool,
) -> tuple[np.ndarray, list[str], dict]:
    z = np.load(sigma_npz, allow_pickle=True)
    valid = np.asarray(z["valid_mask"], dtype=bool)
    canonical = np.asarray(z["canonical_smiles"]).astype(str)
    expected = df["canonical_smiles"].astype(str).to_numpy()
    if canonical.shape[0] != len(df) or not np.array_equal(canonical, expected):
        raise ValueError(
            f"{sigma_npz} is not row-aligned to the current homoset union: "
            f"npz_rows={canonical.shape[0]} df_rows={len(df)}"
        )
    if not np.all(valid):
        raise ValueError(f"{sigma_npz} has invalid selected rows: valid={int(valid.sum())}/{len(valid)}")

    blocks = [np.asarray(z["mu_J_per_mol"], dtype=np.float32)]
    names = [f"gxtb_mu_{i:02d}" for i in range(blocks[0].shape[1])]
    meta = {
        "sigma_npz": str(sigma_npz),
        "valid_rows": int(valid.sum()),
        "grid_min": float(np.asarray(z["sigma_grid_e_per_A2"])[0]),
        "grid_max": float(np.asarray(z["sigma_grid_e_per_A2"])[-1]),
        "mu_features": int(blocks[0].shape[1]),
    }
    if include_profile:
        profile = np.asarray(z["profile_area_A2"], dtype=np.float32)
        blocks.append(profile)
        names.extend([f"gxtb_profile_area_{i:02d}" for i in range(profile.shape[1])])
        meta["profile_features"] = int(profile.shape[1])
    x = np.concatenate(blocks, axis=1).astype(np.float32)
    return x, names, meta


def fit_fold_prep(x_train: np.ndarray, threshold: float, min_std: float) -> tuple[np.ndarray, FoldPrep, dict]:
    x64 = np.asarray(x_train, dtype=np.float64)
    finite = np.all(np.isfinite(x64), axis=0)
    x_keep = x64[:, finite]
    if x_keep.size:
        std = x_keep.std(axis=0)
        nonconstant = std > min_std
        x_keep = x_keep[:, nonconstant].copy()
        max_abs = np.max(np.abs(x_keep), axis=0)
        arcsinh = max_abs > threshold
        x_keep[:, arcsinh] = np.arcsinh(x_keep[:, arcsinh] / threshold)
    else:
        nonconstant = np.zeros((0,), dtype=bool)
        arcsinh = np.zeros((0,), dtype=bool)

    keep_mask = np.zeros_like(finite, dtype=bool)
    finite_idx = np.flatnonzero(finite)
    keep_mask[finite_idx[nonconstant]] = True
    prep = FoldPrep(keep_mask=keep_mask, arcsinh_mask=arcsinh.astype(bool), threshold=float(threshold))
    meta = {
        "input_features": int(x64.shape[1]),
        "dropped_nonfinite_train": int((~finite).sum()),
        "dropped_constant_train": int((finite.sum() - nonconstant.sum()) if finite.sum() else 0),
        "clean_features": int(keep_mask.sum()),
        "arcsinh_features": int(arcsinh.sum()),
    }
    return x_keep.astype(np.float32), prep, meta


def apply_fold_prep(x: np.ndarray, prep: FoldPrep) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)[:, prep.keep_mask].copy()
    x64[~np.isfinite(x64)] = 0.0
    if np.any(prep.arcsinh_mask):
        x64[:, prep.arcsinh_mask] = np.arcsinh(x64[:, prep.arcsinh_mask] / prep.threshold)
    return x64.astype(np.float32)


def make_model(name: str, args: argparse.Namespace, seed: int):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            max_features=args.rf_max_features,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=False,
            random_state=seed,
            n_jobs=args.jobs,
        )
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=args.xgb_trees,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            min_child_weight=1.0,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=args.jobs,
        )
    raise ValueError(f"unknown model {name!r}")


def parse_csv_arg(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def training_mask_for_mode(union: pd.DataFrame, mode: str, *, exclude_keys: set[str] | None = None) -> np.ndarray:
    if mode not in {"trusted", "trusted_pseudo"}:
        raise ValueError(f"unknown training mode {mode!r}")
    mask = union["_trusted"].to_numpy(dtype=bool).copy()
    if mode == "trusted_pseudo":
        mask |= union["target_source"].astype(str).eq("calcphyschemprop_calibrated_pseudo").to_numpy(dtype=bool)
    if exclude_keys:
        mask &= ~union["canonical_smiles"].astype(str).isin(exclude_keys).to_numpy(dtype=bool)
    return mask


def build_feature_matrix(df: pd.DataFrame, feature_set: str, args: argparse.Namespace) -> tuple[np.ndarray, list[str], dict]:
    smiles = df["_feature_smiles"].astype(str).tolist()
    blocks: list[np.ndarray] = []
    names: list[str] = []
    meta: dict = {"feature_set": feature_set, "blocks": []}

    if "rdkit" in feature_set.split("_"):
        x, n = compute_rdkit_features(smiles, args.cache_dir / "deltaHvapv2_rdkit_features.npz")
        blocks.append(x)
        names.extend([f"rdkit_{name}" for name in n])
        meta["blocks"].append({"name": "rdkit", "features": int(x.shape[1])})

    parts = set(feature_set.split("_"))

    if "sigma" in parts:
        x, n, sigma_meta = attach_sigma(smiles, args.chaos_matrix)
        blocks.append(x)
        names.extend(n)
        meta["blocks"].append({"name": "chaos_sigma_mu", "features": int(x.shape[1]), **sigma_meta})

    if "gxtb" in parts:
        x, n, gxtb_meta = attach_row_aligned_gxtb_sigma(
            df,
            args.gxtb_sigma_npz,
            include_profile="profile" in parts,
        )
        blocks.append(x)
        names.extend(n)
        meta["blocks"].append({"name": "gxtb_tmcosmo_sigma", "features": int(x.shape[1]), **gxtb_meta})

    if not blocks:
        raise ValueError(f"feature_set={feature_set!r} produced no blocks")
    x_all = np.concatenate(blocks, axis=1).astype(np.float32)
    meta["features"] = int(x_all.shape[1])
    return x_all, names, meta


def save_final_model(
    *,
    out_dir: Path,
    feature_set: str,
    training_mode: str,
    model_name: str,
    x_all: np.ndarray,
    feature_names: list[str],
    union: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    import joblib

    train_mask = training_mask_for_mode(union, training_mode)
    tr_idx = np.flatnonzero(train_mask)
    y_train = union.loc[tr_idx, "trusted_target_kJmol"].to_numpy(dtype=np.float64)
    w_train = union.loc[tr_idx, "sample_weight"].to_numpy(dtype=np.float64)
    x_train, prep, prep_meta = fit_fold_prep(x_all[tr_idx], args.arcsinh_threshold, args.min_std)

    model = make_model(model_name, args, args.seed)
    model.fit(x_train, y_train, sample_weight=w_train)

    stem = f"deltaHvapv2_{feature_set}_{training_mode}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{stem}.joblib"
    meta_path = out_dir / f"{stem}.json"
    npz_path = out_dir / f"{stem}_prep.npz"

    joblib.dump(
        {
            "model": model,
            "feature_set": feature_set,
            "training_mode": training_mode,
            "model_name": model_name,
            "feature_names": feature_names,
            "prep": {
                "keep_mask": prep.keep_mask,
                "arcsinh_mask": prep.arcsinh_mask,
                "threshold": prep.threshold,
            },
            "target": "deltaHvap_kJmol",
            "dataset": "deltaHvapv2_curated",
        },
        model_path,
    )
    np.savez_compressed(
        npz_path,
        keep_mask=prep.keep_mask,
        arcsinh_mask=prep.arcsinh_mask,
        threshold=np.asarray([prep.threshold], dtype=np.float64),
        feature_names=np.asarray(feature_names, dtype=object),
        train_indices=tr_idx.astype(np.int64),
    )
    meta = {
        "model_path": str(model_path),
        "prep_path": str(npz_path),
        "feature_set": feature_set,
        "training_mode": training_mode,
        "model": model_name,
        "n_train": int(len(tr_idx)),
        "n_train_pseudo": int(np.sum(union.loc[tr_idx, "target_source"].astype(str).eq("calcphyschemprop_calibrated_pseudo"))),
        "target_min_kJmol": float(np.min(y_train)),
        "target_max_kJmol": float(np.max(y_train)),
        "target_mean_kJmol": float(np.mean(y_train)),
        **prep_meta,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--union-csv", type=Path, default=None)
    parser.add_argument("--overlap-csv", type=Path, default=None)
    parser.add_argument("--feature-sets", default="rdkit,rdkit_sigma")
    parser.add_argument("--models", default="xgb,rf")
    parser.add_argument("--training-modes", default="trusted,trusted_pseudo")
    parser.add_argument("--eval-scope", choices=["overlap", "all-trusted"], default="overlap")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/delta_hvap_v2"))
    parser.add_argument("--chaos-matrix", type=Path, default=Path("data/chaos_25a_mu_matrix.npz"))
    parser.add_argument("--gxtb-sigma-npz", type=Path, default=DEFAULT_GXTB_SIGMA)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--min-std", type=float, default=1.0e-12)
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--rf-max-depth", type=int, default=20)
    parser.add_argument("--rf-max-features", default="sqrt")
    parser.add_argument("--xgb-trees", type=int, default=600)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.035)
    parser.add_argument("--xgb-subsample", type=float, default=0.90)
    parser.add_argument("--xgb-colsample", type=float, default=0.80)
    parser.add_argument("--save-final-models", action="store_true")
    parser.add_argument("--final-dir", type=Path, default=None)
    parser.add_argument("--final-training-modes", default="trusted")
    args = parser.parse_args()

    overlap_path = args.overlap_csv or (args.data_dir / "deltaHvapv2_alignment_overlap.csv")
    if not overlap_path.exists():
        overlap_path = args.data_dir / "deltaHvapv2_homoset.csv"
    alignment_overlap = pd.read_csv(overlap_path)
    union_path = args.union_csv or (args.data_dir / "deltaHvapv2_homoset_union.csv")
    if not union_path.exists():
        union_path = args.data_dir / "deltaHvapv2_union.csv"
    union = pd.read_csv(union_path)
    homoset_keys = set(alignment_overlap["canonical_smiles"].astype(str))

    union = union.copy()
    union["_feature_smiles"] = union.apply(first_smiles, axis=1)
    union["_eval_homoset"] = union["canonical_smiles"].astype(str).isin(homoset_keys)
    union["_trusted"] = union["target_source"].astype(str).isin(TRUSTED_TARGET_SOURCES)

    if args.eval_scope == "overlap":
        eval_mask = union["_trusted"] & union["_eval_homoset"]
    elif args.eval_scope == "all-trusted":
        eval_mask = union["_trusted"]
    else:
        raise ValueError(f"unknown eval scope {args.eval_scope!r}")

    eval_df = union[eval_mask].copy().reset_index(drop=True)
    eval_keys = eval_df["canonical_smiles"].astype(str).to_numpy()
    eval_y = eval_df["trusted_target_kJmol"].to_numpy(dtype=np.float64)

    baseline_rows: list[dict] = []
    if args.eval_scope == "overlap":
        calc_baseline = alignment_overlap.set_index("canonical_smiles").loc[eval_keys]
        baseline_rows.extend(
            [
                {
                    "feature_set": "calcphyschemprop_source",
                    "training_mode": "baseline",
                    "model": "none",
                    "n_eval": int(len(eval_y)),
                    **metrics(eval_y, calc_baseline["calc_deltaHvap_source_kJmol"].to_numpy(dtype=np.float64)),
                },
                {
                    "feature_set": "calcphyschemprop_pred",
                    "training_mode": "baseline",
                    "model": "none",
                    "n_eval": int(len(eval_y)),
                    **metrics(eval_y, calc_baseline["calc_deltaHvap_pred_kJmol"].to_numpy(dtype=np.float64)),
                },
            ]
        )
        if "calc_deltaHvap_source_homoset_aligned_kJmol" in calc_baseline.columns:
            baseline_rows.append(
                {
                    "feature_set": "calcphyschemprop_source_homoset_aligned",
                    "training_mode": "baseline",
                    "model": "none",
                    "n_eval": int(len(eval_y)),
                    **metrics(eval_y, calc_baseline["calc_deltaHvap_source_homoset_aligned_kJmol"].to_numpy(dtype=np.float64)),
                }
            )
        if "calc_deltaHvap_pred_homoset_aligned_kJmol" in calc_baseline.columns:
            baseline_rows.append(
                {
                    "feature_set": "calcphyschemprop_pred_homoset_aligned",
                    "training_mode": "baseline",
                    "model": "none",
                    "n_eval": int(len(eval_y)),
                    **metrics(eval_y, calc_baseline["calc_deltaHvap_pred_homoset_aligned_kJmol"].to_numpy(dtype=np.float64)),
                }
            )
        if "calc_deltaHvap_source_curated_kJmol" in calc_baseline.columns:
            baseline_rows.append(
                {
                    "feature_set": "calcphyschemprop_source_curated",
                    "training_mode": "baseline",
                    "model": "none",
                    "n_eval": int(len(eval_y)),
                    **metrics(eval_y, calc_baseline["calc_deltaHvap_source_curated_kJmol"].to_numpy(dtype=np.float64)),
                }
            )
        if "calc_deltaHvap_pred_curated_kJmol" in calc_baseline.columns:
            baseline_rows.append(
                {
                    "feature_set": "calcphyschemprop_pred_curated",
                    "training_mode": "baseline",
                    "model": "none",
                    "n_eval": int(len(eval_y)),
                    **metrics(eval_y, calc_baseline["calc_deltaHvap_pred_curated_kJmol"].to_numpy(dtype=np.float64)),
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = parse_csv_arg(args.feature_sets)
    models = parse_csv_arg(args.models)
    training_modes = parse_csv_arg(args.training_modes)
    summaries: list[dict] = baseline_rows.copy()
    all_fold_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    feature_meta: dict[str, dict] = {}
    final_artifacts: list[dict] = []
    final_training_modes = set(parse_csv_arg(args.final_training_modes))
    final_dir = args.final_dir or (args.out_dir / "final_models")

    for feature_set in feature_sets:
        x_all, names, meta = build_feature_matrix(union, feature_set, args)
        feature_meta[feature_set] = meta
        key_to_index = {key: i for i, key in enumerate(union["canonical_smiles"].astype(str))}
        eval_indices = np.asarray([key_to_index[key] for key in eval_keys], dtype=np.int64)

        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for training_mode in training_modes:
            if training_mode not in {"trusted", "trusted_pseudo"}:
                raise ValueError(f"unknown training mode {training_mode!r}")
            for model_name in models:
                oof = np.zeros_like(eval_y, dtype=np.float64)
                oof_seen = np.zeros_like(eval_y, dtype=bool)
                for fold, (tr_eval_local, va_eval_local) in enumerate(splitter.split(eval_indices)):
                    val_keys = set(eval_keys[va_eval_local])
                    train_mask = training_mask_for_mode(union, training_mode, exclude_keys=val_keys)

                    tr_idx = np.flatnonzero(train_mask)
                    va_idx = eval_indices[va_eval_local]
                    y_train = union.loc[tr_idx, "trusted_target_kJmol"].to_numpy(dtype=np.float64)
                    w_train = union.loc[tr_idx, "sample_weight"].to_numpy(dtype=np.float64)
                    x_train, prep, prep_meta = fit_fold_prep(x_all[tr_idx], args.arcsinh_threshold, args.min_std)
                    x_val = apply_fold_prep(x_all[va_idx], prep)

                    model = make_model(model_name, args, args.seed + fold)
                    model.fit(x_train, y_train, sample_weight=w_train)
                    pred = np.asarray(model.predict(x_val), dtype=np.float64)
                    oof[va_eval_local] = pred
                    oof_seen[va_eval_local] = True
                    row = {
                        "feature_set": feature_set,
                        "training_mode": training_mode,
                        "model": model_name,
                        "fold": fold,
                        "n_train": int(len(tr_idx)),
                        "n_train_pseudo": int(np.sum(union.loc[tr_idx, "target_source"].astype(str).eq("calcphyschemprop_calibrated_pseudo"))),
                        "n_val": int(len(va_idx)),
                        **metrics(eval_y[va_eval_local], pred),
                        **prep_meta,
                    }
                    all_fold_rows.append(row)
                    print(
                        f"[{feature_set}/{training_mode}/{model_name}] fold={fold} "
                        f"mae={row['mae']:.3f} rmse={row['rmse']:.3f} r2={row['r2']:.4f} "
                        f"train={row['n_train']} pseudo={row['n_train_pseudo']}",
                        flush=True,
                    )

                assert np.all(oof_seen)
                summary = {
                    "feature_set": feature_set,
                    "training_mode": training_mode,
                    "model": model_name,
                    "n_eval": int(len(eval_y)),
                    **metrics(eval_y, oof),
                }
                summaries.append(summary)
                predictions.append(
                    pd.DataFrame(
                        {
                            "canonical_smiles": eval_keys,
                            "y_true_autovap_dvap_kJmol": eval_y,
                            "y_pred_kJmol": oof,
                            "feature_set": feature_set,
                            "training_mode": training_mode,
                            "model": model_name,
                        }
                    )
                )
                if args.save_final_models and training_mode in final_training_modes:
                    artifact = save_final_model(
                        out_dir=final_dir,
                        feature_set=feature_set,
                        training_mode=training_mode,
                        model_name=model_name,
                        x_all=x_all,
                        feature_names=names,
                        union=union,
                        args=args,
                    )
                    final_artifacts.append(artifact)
                    print(
                        f"[final/{feature_set}/{training_mode}/{model_name}] "
                        f"saved={artifact['model_path']} train={artifact['n_train']} "
                        f"pseudo={artifact['n_train_pseudo']}",
                        flush=True,
                    )

    summary_df = pd.DataFrame(summaries).sort_values(["mae", "rmse"], ascending=[True, True])
    folds_df = pd.DataFrame(all_fold_rows)
    preds_df = pd.concat(predictions, axis=0, ignore_index=True) if predictions else pd.DataFrame()
    summary_df.to_csv(args.out_dir / "deltaHvapv2_homoset_summary.csv", index=False)
    folds_df.to_csv(args.out_dir / "deltaHvapv2_homoset_folds.csv", index=False)
    preds_df.to_csv(args.out_dir / "deltaHvapv2_homoset_predictions.csv", index=False)
    report = {
        "args": jsonable_args(args) | {
            "data_dir": str(args.data_dir),
            "out_dir": str(args.out_dir),
            "cache_dir": str(args.cache_dir),
            "chaos_matrix": str(args.chaos_matrix),
            "gxtb_sigma_npz": str(args.gxtb_sigma_npz),
            "final_dir": str(final_dir),
        },
        "n_union": int(len(union)),
        "n_homoset_union": int(len(union)),
        "n_eval_alignment_overlap": int(len(eval_df)),
        "n_eval_homoset": int(len(eval_df)),
        "eval_scope": args.eval_scope,
        "trusted_target_sources": sorted(TRUSTED_TARGET_SOURCES),
        "n_autovap_trusted": int(union["target_source"].astype(str).eq("autovap_trusted").sum()),
        "n_trusted_total": int(union["_trusted"].sum()),
        "n_pseudo": int(union["target_source"].astype(str).eq("calcphyschemprop_calibrated_pseudo").sum()),
        "feature_meta": feature_meta,
        "feature_name_count": {fs: int(feature_meta[fs]["features"]) for fs in feature_meta},
        "final_artifacts": final_artifacts,
    }
    (args.out_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== deltaHvapv2 homoset summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
