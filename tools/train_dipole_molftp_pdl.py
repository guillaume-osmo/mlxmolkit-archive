#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from molftp.pdl import common_unique_pair_features, fold_fit_transform, sample_oriented_pairs


@dataclass
class FeaturePrep:
    kept_mask: np.ndarray
    arcsinh_mask: np.ndarray
    threshold: float


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_folds(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr.astype(int), va.astype(int)) for tr, va in splitter.split(np.arange(n))]


def load_dataset(data_dir: Path, max_mols: int | None, seed: int, smiles_column: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "dipole_sigma_chemprop.csv")
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column {smiles_column!r} not found; available={list(df.columns)}")
    df = df[df["dipole_debye"].notna()].reset_index(drop=True)
    if max_mols is not None and max_mols < len(df):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(np.arange(len(df)), size=max_mols, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
    return df


def _as_float_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_or_compute_chemeleon_fps(data_dir: Path, smiles: list[str], batch_size: int) -> np.ndarray:
    cache_path = data_dir / "chemeleon_smd_fingerprints.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_smiles = [str(s) for s in z["smiles"].tolist()]
        if cached_smiles == list(smiles):
            return _as_float_matrix(z["fingerprints"])

    try:
        from chemeleon_smd import fingerprint
    except Exception as exc:
        raise ImportError(
            "ChemeleonSMD fingerprints requested but `chemeleon_smd` is not importable. "
            "Run with PYTHONPATH including /Users/guillaume-osmo/Github/ChemeleonSMD."
        ) from exc

    chunks: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(smiles), batch_size):
        stop = min(start + batch_size, len(smiles))
        fps = fingerprint(smiles[start:stop])
        chunks.append(_as_float_matrix(np.asarray(fps)))
        print(f"[chemeleon] fingerprints {stop}/{len(smiles)} seconds={time.time() - t0:.1f}", flush=True)
    x = np.concatenate(chunks, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, fingerprints=x, smiles=np.asarray(smiles, dtype=object))
    return x


def load_extra_features(data_dir: Path, df: pd.DataFrame, mode: str) -> tuple[np.ndarray, list[str]]:
    if mode == "none":
        return np.zeros((len(df), 0), dtype=np.float32), []

    mode_parts = set(mode.split("_"))
    source = df["feature_index"].to_numpy(dtype=np.int64) if "feature_index" in df.columns else df.index.to_numpy(dtype=np.int64)
    blocks: list[np.ndarray] = []
    names: list[str] = []

    if "osmordred" in mode_parts:
        z = np.load(data_dir / "osmordred_features.npz", allow_pickle=True)
        x = _as_float_matrix(z["X"])[source]
        raw_names = [str(n) for n in z["names"].tolist()]
        keep = np.asarray(
            [
                ("calcphyschemprop" not in name.lower())
                and ("pred_debye" not in name.lower())
                and ("dipole_debye" not in name.lower())
                for name in raw_names
            ],
            dtype=bool,
        )
        blocks.append(x[:, keep])
        names.extend([name for name, use in zip(raw_names, keep) if bool(use)])

    if "sigma" in mode_parts:
        z = np.load(data_dir / "sigma_features.npz", allow_pickle=False)
        mu = _as_float_matrix(z["mu_J_per_mol"])[source]
        profile = _as_float_matrix(z["profile_area_A2"])[source]
        blocks.extend([mu, profile])
        names.extend([f"sigma_mu_{i:02d}" for i in range(mu.shape[1])])
        names.extend([f"sigma_profile_{i:02d}" for i in range(profile.shape[1])])

    if "chemeleon" in mode_parts:
        all_df = pd.read_csv(data_dir / "dipole_sigma_chemprop.csv")
        all_smiles = all_df["canonical_smiles"].astype(str).tolist()
        fps = load_or_compute_chemeleon_fps(data_dir, all_smiles, batch_size=128)[source]
        blocks.append(fps)
        names.extend([f"chemeleon_smd_{i:04d}" for i in range(fps.shape[1])])

    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float32), []
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def make_threshold_labels(y: np.ndarray, quantiles: Iterable[float]) -> tuple[np.ndarray, list[str], list[float]]:
    qs = [float(q) for q in quantiles]
    thresholds = np.quantile(np.asarray(y, dtype=np.float64), qs)
    unique: list[float] = []
    names: list[str] = []
    for q, t in zip(qs, thresholds):
        t = float(t)
        if any(abs(t - old) < 1.0e-12 for old in unique):
            continue
        unique.append(t)
        names.append(f"dipole_ge_q{int(round(q * 100)):02d}")
    labels = np.stack([(y >= t).astype(float) for t in unique], axis=1)
    return labels, names, unique


def fit_feature_prep(x: np.ndarray, threshold: float) -> tuple[np.ndarray, FeaturePrep]:
    x64 = np.asarray(x, dtype=np.float64)
    finite = np.all(np.isfinite(x64), axis=0)
    x_keep = x64[:, finite].copy()
    if x_keep.size:
        max_abs = np.max(np.abs(x_keep), axis=0)
        arcsinh = max_abs > threshold
        x_keep[:, arcsinh] = np.arcsinh(x_keep[:, arcsinh] / threshold)
    else:
        arcsinh = np.zeros((0,), dtype=bool)
    prep = FeaturePrep(finite, arcsinh.astype(bool), float(threshold))
    return x_keep.astype(np.float32), prep


def apply_feature_prep(x: np.ndarray, prep: FeaturePrep) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)[:, prep.kept_mask].copy()
    x64[~np.isfinite(x64)] = 0.0
    if np.any(prep.arcsinh_mask):
        x64[:, prep.arcsinh_mask] = np.arcsinh(x64[:, prep.arcsinh_mask] / prep.threshold)
    return x64.astype(np.float32)


def standardize_for_knn(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def apply_outer_fold_pca(
    x_train_full: np.ndarray,
    x_train_oof: np.ndarray,
    x_test: np.ndarray,
    *,
    n_components: int,
    whiten: bool,
    arcsinh_threshold: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if n_components <= 0:
        return x_train_full, x_train_oof, x_test, {"enabled": False}

    x_fit, prep = fit_feature_prep(x_train_full, arcsinh_threshold)
    x_oof = apply_feature_prep(x_train_oof, prep)
    x_val = apply_feature_prep(x_test, prep)

    mean = x_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_fit.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std)
    x_fit_z = (x_fit - mean) / std
    x_oof_z = (x_oof - mean) / std
    x_val_z = (x_val - mean) / std

    max_components = min(x_fit_z.shape[0], x_fit_z.shape[1])
    n_pca = int(min(max(n_components, 1), max_components))
    pca = PCA(n_components=n_pca, whiten=whiten, random_state=seed, svd_solver="auto")
    train_full_pca = pca.fit_transform(x_fit_z).astype(np.float32)
    train_oof_pca = pca.transform(x_oof_z).astype(np.float32)
    test_pca = pca.transform(x_val_z).astype(np.float32)
    meta = {
        "enabled": True,
        "requested_components": int(n_components),
        "components": n_pca,
        "raw_features": int(x_train_full.shape[1]),
        "clean_features": int(x_fit.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "whiten": bool(whiten),
    }
    return train_full_pca, train_oof_pca, test_pca, meta


def make_extra_trees(args: argparse.Namespace, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.trees,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        random_state=seed,
        n_jobs=args.jobs,
        bootstrap=False,
    )


def make_direct_model(args: argparse.Namespace, seed: int):
    if args.direct_model == "etr":
        return make_extra_trees(args, seed)
    if args.direct_model == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=args.xgb_trees,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=args.jobs,
        )
    if args.direct_model == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 17))
    raise ValueError(f"unknown direct_model={args.direct_model!r}")


def build_pair_matrix(x_a: np.ndarray, x_b: np.ndarray, y_a: np.ndarray, include_abs_delta: bool) -> np.ndarray:
    pair = common_unique_pair_features(x_a, x_b, include_abs_delta=include_abs_delta)
    return np.concatenate([pair, np.asarray(y_a, dtype=np.float32).reshape(-1, 1)], axis=1)


def predict_with_anchors(
    model: ExtraTreesRegressor,
    pair_prep: FeaturePrep,
    x_train_full: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    anchor_idx: np.ndarray,
    include_abs_delta: bool,
    batch_size: int,
    aggregate: str,
    pair_target: str,
) -> np.ndarray:
    preds = np.empty((x_test.shape[0],), dtype=np.float64)
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        anchors = anchor_idx[start:stop]
        k = anchors.shape[1]
        flat_anchor = anchors.reshape(-1)
        a = x_train_full[flat_anchor]
        b = np.repeat(x_test[start:stop], k, axis=0)
        y_a = y_train[flat_anchor]
        pair_raw = build_pair_matrix(a, b, y_a, include_abs_delta)
        pair = apply_feature_prep(pair_raw, pair_prep)
        pair_pred = model.predict(pair).reshape(stop - start, k)
        if pair_target == "delta":
            candidates = y_train[anchors] + pair_pred
        else:
            candidates = pair_pred
        if aggregate == "mean":
            preds[start:stop] = candidates.mean(axis=1)
        else:
            preds[start:stop] = np.median(candidates, axis=1)
    return preds


def run_cv(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_dataset(args.data_dir, args.max_mols, args.seed, args.smiles_column)
    smiles = df[args.smiles_column].astype(str).tolist()
    y = df["dipole_debye"].to_numpy(dtype=np.float64)
    extra_x, extra_names = load_extra_features(args.data_dir, df, args.extra_features)
    if args.skip_molftp and extra_x.shape[1] == 0:
        raise ValueError("--skip-molftp requires --extra-features other than 'none'")
    labels, task_names, thresholds = make_threshold_labels(y, args.quantiles)
    folds = make_folds(len(df), args.folds, args.seed)
    feature_prefix = "extra" if args.skip_molftp else "molftp"
    direct_model_name = f"{feature_prefix}_direct_{args.direct_model}"
    pdl_model_name = f"{feature_prefix}_pdl_{args.pair_target}_etr"

    print(
        f"Loaded {len(df)} molecules, target=dipole_debye, smiles={args.smiles_column}, "
        f"molFTP={'off' if args.skip_molftp else 'on'} "
        f"tasks={0 if args.skip_molftp else len(task_names)} "
        f"thresholds={[] if args.skip_molftp else [round(t, 4) for t in thresholds]}, "
        f"extra_features={args.extra_features} ({extra_x.shape[1]} cols)",
        flush=True,
    )

    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    t_all = time.time()
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        t0 = time.time()
        y_train = y[train_idx]
        y_test = y[test_idx]
        if args.skip_molftp:
            molftp_train_full = np.zeros((len(train_idx), 0), dtype=np.float32)
            molftp_train_oof = np.zeros((len(train_idx), 0), dtype=np.float32)
            molftp_test = np.zeros((len(test_idx), 0), dtype=np.float32)
        else:
            fold = fold_fit_transform(
                smiles,
                labels,
                train_idx,
                test_idx,
                task_names=task_names,
                inner_folds=args.inner_folds,
                random_state=args.seed + fold_id,
                generator_kwargs={
                    "radius": args.radius,
                    "nBits": args.nbits,
                    "sim_thresh": args.sim_thresh,
                    "method": args.molftp_method,
                    "num_threads": args.molftp_threads,
                    "counting_method": args.counting_method,
                    "verbose": args.verbose_molftp,
                },
            )
            molftp_train_full = fold.X_train_full
            molftp_train_oof = fold.X_train_oof
            molftp_test = fold.X_test
        extra_train_raw = extra_x[train_idx].astype(np.float32, copy=False)
        extra_test_raw = extra_x[test_idx].astype(np.float32, copy=False)
        if extra_train_raw.shape[1]:
            extra_train, extra_prep = fit_feature_prep(extra_train_raw, args.arcsinh_threshold)
            extra_test = apply_feature_prep(extra_test_raw, extra_prep)
        else:
            extra_train = np.zeros((len(train_idx), 0), dtype=np.float32)
            extra_test = np.zeros((len(test_idx), 0), dtype=np.float32)

        combined_train_full = np.concatenate([molftp_train_full, extra_train], axis=1)
        combined_train_oof = np.concatenate([molftp_train_oof, extra_train], axis=1)
        combined_test = np.concatenate([molftp_test, extra_test], axis=1)

        x_train_full_model, x_train_oof_model, x_test_model, pca_meta = apply_outer_fold_pca(
            combined_train_full,
            combined_train_oof,
            combined_test,
            n_components=args.pca_components,
            whiten=args.pca_whiten,
            arcsinh_threshold=args.arcsinh_threshold,
            seed=args.seed + 4000 + fold_id,
        )

        mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=np.float64)
        mean_m = metrics(y_test, mean_pred)
        fold_rows.append({"model": "train_mean", "fold": fold_id, **mean_m, "seconds": 0.0})

        direct_train_source = combined_train_full if args.direct_raw_features else x_train_full_model
        direct_test_source = combined_test if args.direct_raw_features else x_test_model
        x_direct_train, direct_prep = fit_feature_prep(direct_train_source, args.arcsinh_threshold)
        x_direct_test = apply_feature_prep(direct_test_source, direct_prep)
        direct = make_direct_model(args, args.seed + 1000 + fold_id)
        direct.fit(x_direct_train, y_train)
        direct_pred = direct.predict(x_direct_test)
        direct_m = metrics(y_test, direct_pred)
        fold_rows.append(
            {
                "model": direct_model_name,
                "fold": fold_id,
                **direct_m,
                "seconds": time.time() - t0,
                "n_features": int(x_direct_train.shape[1]),
                "raw_combined_features": int(combined_train_full.shape[1]),
                "molftp_features": int(molftp_train_full.shape[1]),
                "extra_features": int(extra_train.shape[1]),
                "direct_raw_features": bool(args.direct_raw_features),
                "pca_components": pca_meta.get("components"),
                "pca_var": pca_meta.get("explained_variance_ratio_sum"),
            }
        )

        n_pairs = min(args.pairs_per_fold, max(1, len(train_idx) * (len(train_idx) - 1)))
        pair_a, pair_b, dy = sample_oriented_pairs(
            np.arange(len(train_idx)),
            y_train,
            n_pairs,
            min_abs_dy=args.min_abs_dy,
            random_state=args.seed + 2000 + fold_id,
        )
        pair_raw = build_pair_matrix(
            x_train_oof_model[pair_a],
            x_train_oof_model[pair_b],
            y_train[pair_a],
            args.include_abs_delta,
        )
        pair_train, pair_prep = fit_feature_prep(pair_raw, args.arcsinh_threshold)
        pair_target = dy if args.pair_target == "delta" else y_train[pair_b]
        pdl = make_extra_trees(args, args.seed + 3000 + fold_id)
        pdl.fit(pair_train, pair_target)

        x_knn_train, x_knn_test = standardize_for_knn(x_train_full_model, x_test_model)
        k_anchor = min(args.predict_anchors, len(train_idx))
        nn = NearestNeighbors(n_neighbors=k_anchor, algorithm="auto", metric="euclidean")
        nn.fit(x_knn_train)
        anchor_idx = nn.kneighbors(x_knn_test, return_distance=False).astype(int)
        pdl_pred = predict_with_anchors(
            pdl,
            pair_prep,
            x_train_full_model,
            y_train,
            x_test_model,
            anchor_idx,
            args.include_abs_delta,
            args.predict_batch_size,
            args.aggregate,
            args.pair_target,
        )
        pdl_m = metrics(y_test, pdl_pred)
        fold_rows.append(
            {
                "model": pdl_model_name,
                "fold": fold_id,
                **pdl_m,
                "seconds": time.time() - t0,
                "n_features": int(pair_train.shape[1]),
                "raw_combined_features": int(combined_train_full.shape[1]),
                "molftp_features": int(molftp_train_full.shape[1]),
                "extra_features": int(extra_train.shape[1]),
                "n_pairs": int(pair_train.shape[0]),
                "anchors": int(k_anchor),
                "pca_components": pca_meta.get("components"),
                "pca_var": pca_meta.get("explained_variance_ratio_sum"),
            }
        )

        for local_pos, row_idx in enumerate(test_idx):
            base = {
                "fold": fold_id,
                "row_index": int(row_idx),
                "smiles": smiles[int(row_idx)],
                "dipole_debye": float(y[int(row_idx)]),
            }
            pred_rows.append({**base, "model": "train_mean", "prediction": float(mean_pred[local_pos])})
            pred_rows.append({**base, "model": direct_model_name, "prediction": float(direct_pred[local_pos])})
            pred_rows.append({**base, "model": pdl_model_name, "prediction": float(pdl_pred[local_pos])})

        print(
            f"[fold {fold_id + 1}/{len(folds)}] "
            f"direct MAE={direct_m['mae']:.4f} RMSE={direct_m['rmse']:.4f} R2={direct_m['r2']:.4f}; "
            f"PDL MAE={pdl_m['mae']:.4f} RMSE={pdl_m['rmse']:.4f} R2={pdl_m['r2']:.4f}; "
            f"seconds={time.time() - t0:.1f}",
            flush=True,
        )

    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for model, g in pred_df.groupby("model", sort=False):
        summary_rows.append(
            {
                "model": model,
                **metrics(g["dipole_debye"].to_numpy(), g["prediction"].to_numpy()),
                "n": int(len(g)),
            }
        )
    if "calcphyschemprop_pred_debye" in df.columns:
        summary_rows.append(
            {
                "model": "calcphyschemprop_pred_debye",
                **metrics(df["dipole_debye"].to_numpy(), df["calcphyschemprop_pred_debye"].to_numpy()),
                "n": int(len(df)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mae").reset_index(drop=True)
    args_json = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        "data_dir": str(args.data_dir),
        "n_molecules": int(len(df)),
        "smiles_column": args.smiles_column,
        "folds": int(args.folds),
        "seed": int(args.seed),
        "thresholds": thresholds,
        "task_names": task_names,
        "skip_molftp": bool(args.skip_molftp),
        "extra_feature_mode": args.extra_features,
        "extra_feature_count": int(extra_x.shape[1]),
        "extra_feature_names": extra_names,
        "seconds": time.time() - t_all,
        "args": args_json,
    }
    return summary, pred_df, {"folds": fold_df, "config": config}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test leakage-free PDL-molFTP on DipoleMoment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dipole_physchem_chaos3d_graph"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/dipole_molftp_pdl"))
    parser.add_argument("--max-mols", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smiles-column", type=str, default="canonical_smiles")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--sim-thresh", type=float, default=0.5)
    parser.add_argument(
        "--skip-molftp",
        action="store_true",
        help="Disable supervised molFTP features and use only --extra-features for direct/PDL branches.",
    )
    parser.add_argument("--molftp-method", choices=["dummy_masking", "key_loo"], default="dummy_masking")
    parser.add_argument("--counting-method", choices=["counting", "binary_presence", "weighted_presence"], default="counting")
    parser.add_argument("--molftp-threads", type=int, default=1)
    parser.add_argument("--verbose-molftp", action="store_true")
    parser.add_argument(
        "--extra-features",
        choices=[
            "none",
            "sigma",
            "osmordred",
            "chemeleon",
            "sigma_chemeleon",
            "osmordred_sigma",
            "osmordred_chemeleon",
            "osmordred_sigma_chemeleon",
        ],
        default="none",
        help="Concatenate molFTP with existing molecule-level features before direct/PDL models.",
    )
    parser.add_argument("--pairs-per-fold", type=int, default=120_000)
    parser.add_argument("--min-abs-dy", type=float, default=0.0)
    parser.add_argument("--predict-anchors", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--pair-target", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--include-abs-delta", action="store_true")
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--max-features", type=float, default=0.65)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--direct-model", choices=["etr", "xgb", "ridge"], default="etr")
    parser.add_argument("--xgb-trees", type=int, default=700)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample", type=float, default=0.75)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="Compress molFTP molecule vectors inside each outer fold before direct/PDL models. 0 disables PCA.",
    )
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument(
        "--direct-raw-features",
        action="store_true",
        help="Let the direct ETR use the raw combined feature block while PDL uses the PCA-compressed block.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary, predictions, extra = run_cv(args)
    predictions.to_csv(args.out_dir / "dipole_molftp_pdl_predictions.csv", index=False)
    extra["folds"].to_csv(args.out_dir / "dipole_molftp_pdl_folds.csv", index=False)
    summary.to_csv(args.out_dir / "dipole_molftp_pdl_summary.csv", index=False)
    with (args.out_dir / "dipole_molftp_pdl_results.json").open("w") as f:
        payload = {
            "summary": summary.to_dict(orient="records"),
            "folds": extra["folds"].to_dict(orient="records"),
            "config": extra["config"],
        }
        json.dump(payload, f, indent=2)
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
