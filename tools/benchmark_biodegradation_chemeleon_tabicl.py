#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in [
    REPO_ROOT,
    Path("/Users/guillaume-osmo/Github/ChemeleonSMD"),
    Path("/Users/guillaume-osmo/Github/TabICL-MLX"),
    Path("/Users/guillaume-osmo/Github/mlx-addons/src"),
]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from chemeleon_smd import fingerprint
from tabicl_mlx import TabICLRegressorMLX
from train_biodegradation_chemprop_torch import DEFAULT_TARGET, protocol_features


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true[mask], dtype=np.float64)
    y_pred = np.asarray(y_pred[mask], dtype=np.float64)
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }
    for threshold in (60, 70):
        yy = (y_true >= float(threshold)).astype(int)
        pp = (y_pred >= float(threshold)).astype(int)
        out[f"bacc{threshold}"] = float(balanced_accuracy_score(yy, pp)) if len(np.unique(yy)) > 1 else np.nan
        try:
            out[f"roc_auc{threshold}"] = float(roc_auc_score(yy, y_pred)) if len(np.unique(yy)) > 1 else np.nan
        except Exception:
            out[f"roc_auc{threshold}"] = np.nan
    return out


def load_target(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.target_csv)
    df = df[df["canonical_smiles"].notna() & df[args.target_col].notna()].copy()
    df = df.reset_index(drop=True)
    if args.max_rows and args.max_rows < len(df):
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(np.arange(len(df)), size=int(args.max_rows), replace=False))
        df = df.iloc[keep].reset_index(drop=True)
    return df


def load_or_build_fingerprints(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"n{len(df)}_seed{args.seed}_v1"
    fp_path = args.cache_dir / f"chemeleon_fps_{key}.npz"
    smiles_path = args.cache_dir / f"chemeleon_fps_{key}.smiles.txt"
    smiles = df["canonical_smiles"].astype(str).tolist()
    if fp_path.exists() and smiles_path.exists():
        cached_smiles = smiles_path.read_text(encoding="utf-8").splitlines()
        if cached_smiles == smiles:
            return np.load(fp_path)["fps"].astype(np.float32)
    chunks: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(smiles), args.fp_batch_size):
        batch = smiles[start : start + args.fp_batch_size]
        fp = fingerprint(batch, batch_size=args.fp_batch_size)
        arr = np.asarray(fp, dtype=np.float32)
        if arr.shape[0] != len(batch):
            raise RuntimeError(f"ChemeleonSMD skipped molecules in batch {start}:{start + len(batch)}")
        chunks.append(arr)
        print(f"[chemeleon] fps {start + len(batch)}/{len(smiles)} elapsed={time.time() - t0:.1f}s", flush=True)
    fps = np.concatenate(chunks, axis=0).astype(np.float32)
    np.savez_compressed(fp_path, fps=fps)
    smiles_path.write_text("\n".join(smiles) + "\n", encoding="utf-8")
    return fps


def context_matrix(df: pd.DataFrame, include_context: bool) -> np.ndarray:
    if not include_context:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.stack([protocol_features(row) for _, row in df.iterrows()], axis=0).astype(np.float32)


def rpcholesky_feature_select(x_train: np.ndarray, k_max: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_train, dtype=np.float64)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1.0e-12, 1.0, std)
    x = (x - mean) / std
    sigma = (x.T @ x) / max(1, x.shape[0])
    rng = np.random.default_rng(seed)
    d = sigma.shape[0]
    k = min(int(k_max), d)
    res_diag = sigma.diagonal().copy().astype(np.float64)
    factor = np.zeros((d, k), dtype=np.float64)
    idx = np.zeros(k, dtype=np.int64)
    trace_res = np.zeros(k, dtype=np.float64)
    chosen = set()
    used = 0
    for j in range(k):
        probs = np.maximum(res_diag, 0.0)
        if chosen:
            probs[list(chosen)] = 0.0
        total = float(probs.sum())
        if total <= 1.0e-12:
            break
        pivot = int(rng.choice(d, p=probs / total))
        chosen.add(pivot)
        idx[j] = pivot
        c = sigma[:, pivot] - factor[:, :j] @ factor[pivot, :j]
        denom = np.sqrt(max(float(c[pivot]), 1.0e-12))
        factor[:, j] = c / denom
        res_diag = np.maximum(res_diag - factor[:, j] ** 2, 0.0)
        trace_res[j] = float(res_diag.sum())
        used = j + 1
    return idx[:used], trace_res[:used]


def fit_reduce_transform(kind: str, x_train: np.ndarray, x_val: np.ndarray, seed: int, n_components: int):
    if kind == "none":
        return x_train.astype(np.float32), x_val.astype(np.float32), {"reducer": "none", "members": 1}
    if kind in {"rpc", "rpcholesky"}:
        idx, trace_res = rpcholesky_feature_select(x_train, n_components, seed)
        return x_train[:, idx].astype(np.float32), x_val[:, idx].astype(np.float32), {
            "reducer": "rpc",
            "members": 1,
            "rpc_features": int(len(idx)),
            "rpc_trace_residual": float(trace_res[-1]) if len(trace_res) else np.nan,
        }
    if kind == "pca":
        from mlx_addons.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=seed).fit(x_train)
        return reducer.transform(x_train).astype(np.float32), reducer.transform(x_val).astype(np.float32), {
            "reducer": "pca",
            "members": 1,
        }
    if kind == "ensemble_rp":
        from mlx_addons.decomposition import EnsembleRandomProjection

        reducer = EnsembleRandomProjection(
            n_components=n_components,
            n_pca=1,
            n_sparse=2,
            n_gaussian=2,
            random_state=seed,
        ).fit(x_train)
        return reducer.transform(x_train), reducer.transform(x_val), {
            "reducer": "ensemble_rp",
            "members": int(reducer.n_members_),
            "member_kinds": reducer.member_kinds(),
        }
    raise ValueError(f"unknown reducer {kind!r}")


def tabicl_fit_predict(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, *, seed: int, n_estimators: int) -> np.ndarray:
    xs = StandardScaler()
    ys = StandardScaler()
    xtr_s = np.clip(xs.fit_transform(xtr), -6.0, 6.0).astype(np.float32)
    xva_s = np.clip(xs.transform(xva), -6.0, 6.0).astype(np.float32)
    ytr_s = ys.fit_transform(ytr.reshape(-1, 1)).ravel().astype(np.float32)
    model = TabICLRegressorMLX(n_estimators=n_estimators, batch_size=1, random_state=seed, low_memory=True)
    model.fit(xtr_s, ytr_s)
    pred_s = np.asarray(model.predict(xva_s), dtype=np.float32).reshape(-1)
    pred = ys.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    del model
    gc.collect()
    return pred.astype(np.float32)


def run_variant(
    df: pd.DataFrame,
    fps: np.ndarray,
    ctx: np.ndarray,
    *,
    reducer: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = df[args.target_col].to_numpy(dtype=np.float32)
    groups = df["canonical_smiles"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=args.folds).split(np.arange(len(df)), groups=groups))
    pred = np.full(len(df), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for fold_id, (tr, va) in enumerate(folds, start=1):
        ztr, zva, reduce_meta = fit_reduce_transform(reducer, fps[tr], fps[va], args.seed + fold_id, args.n_components)
        member_preds = []
        if ztr.ndim == 3:
            for member in range(ztr.shape[0]):
                xtr = np.concatenate([ztr[member], ctx[tr]], axis=1)
                xva = np.concatenate([zva[member], ctx[va]], axis=1)
                member_preds.append(tabicl_fit_predict(xtr, y[tr], xva, seed=args.seed + fold_id + member, n_estimators=args.n_estimators))
            pred_va = np.mean(np.stack(member_preds, axis=0), axis=0)
        else:
            xtr = np.concatenate([ztr, ctx[tr]], axis=1)
            xva = np.concatenate([zva, ctx[va]], axis=1)
            pred_va = tabicl_fit_predict(xtr, y[tr], xva, seed=args.seed + fold_id, n_estimators=args.n_estimators)
        pred[va] = pred_va
        row = {"fold": fold_id, **metrics(y[va], pred_va), **reduce_meta}
        fold_rows.append(row)
        print(
            f"[tabicl] reducer={reducer} fold={fold_id}/{args.folds} "
            f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} R2={row['r2']:.4f}",
            flush=True,
        )
    summary = {
        "model": f"chemeleon_tabicl_{reducer}",
        "reducer": reducer,
        "n_rows": int(len(df)),
        "n_molecules": int(df["canonical_smiles"].nunique()),
        "n_components": int(args.n_components),
        "n_estimators": int(args.n_estimators),
        "include_context": bool(args.include_context),
        "seconds": float(time.time() - t0),
        **metrics(y, pred),
    }
    pred_df = df[["canonical_smiles", "duration_round", "best_protocol_group", "best_guideline_norm"]].copy()
    pred_df["y_true"] = y
    pred_df["y_pred"] = pred
    pred_df["reducer"] = reducer
    return summary, pred_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChemeleonSMD fingerprints + TabICL-MLX CV for biodegradation.")
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--target-col", default="upper_consensus_y_percent")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/biodegradation_protocol_annotation/chemeleon_tabicl_upper_consensus_v1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmarks/biodegradation_protocol_annotation/chemeleon_cache"))
    parser.add_argument("--reducers", default="pca,ensemble_rp,rpc")
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp-batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-context", dest="include_context", action="store_false")
    parser.set_defaults(include_context=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_target(args)
    fps = load_or_build_fingerprints(df, args)
    ctx = context_matrix(df, args.include_context)
    summaries = []
    for reducer in [x.strip() for x in args.reducers.split(",") if x.strip()]:
        summary, pred_df = run_variant(df, fps, ctx, reducer=reducer, args=args)
        summaries.append(summary)
        pred_df.to_csv(args.out_dir / f"predictions_{reducer}.csv", index=False)
    summary_df = pd.DataFrame(summaries).sort_values("mae")
    summary_df.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "run_report.json").write_text(
        json.dumps({"args": vars(args), "summaries": summaries}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
