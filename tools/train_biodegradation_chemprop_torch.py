#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import json
import random
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mlxmolkit.dipole_features import dipole_atom_feature_tensors
from train_dipole_sigma_gnn import GraphRecord, train_gnn_cv


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


DEFAULT_TARGET = Path(
    "benchmarks/biodegradation_protocol_annotation/"
    "best_protocol_same_duration_v2_consensus/best_same_molecule_duration_target.csv"
)
CHAOS_PROFILE_GRID = np.round(np.arange(-0.025, 0.0251, 0.001), 6)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
        y_bin = (y_true >= float(threshold)).astype(int)
        p_bin = (y_pred >= float(threshold)).astype(int)
        out[f"bacc{threshold}"] = (
            float(balanced_accuracy_score(y_bin, p_bin)) if len(np.unique(y_bin)) > 1 else np.nan
        )
        try:
            out[f"roc_auc{threshold}"] = (
                float(roc_auc_score(y_bin, y_pred)) if len(np.unique(y_bin)) > 1 else np.nan
            )
        except Exception:
            out[f"roc_auc{threshold}"] = np.nan
    return out


def protocol_features(row: pd.Series) -> np.ndarray:
    duration = float(row.get("duration_round", np.nan))
    if not np.isfinite(duration):
        duration = 28.0
    protocol = str(row.get("best_protocol_group", "")).lower()
    guideline = str(row.get("best_guideline_norm", "")).upper()
    y_range = float(row.get("y_range", 0.0))
    conflict_weight = float(row.get("protocol_conflict_weight", 1.0))
    used_median = bool(row.get("upper_consensus_used_median", False))

    def has(text: str) -> float:
        return float(text in guideline)

    return np.asarray(
        [
            duration,
            np.log1p(duration),
            duration / 28.0,
            min(max(duration / 28.0, 0.0), 1.0),
            max(duration - 28.0, 0.0) / 28.0,
            float(protocol == "ready"),
            float(protocol == "inherent"),
            has("OECD301"),
            has("OECD302"),
            has("OECD310"),
            has("301A"),
            has("301B"),
            has("301C"),
            has("301D"),
            has("301F"),
            has("302A"),
            has("302B"),
            has("302C"),
            has("BODIS"),
            y_range / 100.0,
            conflict_weight,
            float(used_median),
            float(row.get("n_protocol_candidates", 1.0)),
            float(row.get("n_total_obs", 1.0)),
        ],
        dtype=np.float32,
    )


def canonical_smiles(smiles: object, *, isomeric: bool = True) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def load_chaos_molecular_sigma(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    data = np.load(args.chaos_npz, allow_pickle=False)
    chaos_ids = np.asarray(data["chaos_ids"])
    smiles = np.asarray(data["canonical_smiles"])
    sigma_grid = np.asarray(data["sigma_grid_e_per_A2"], dtype=np.float64)
    mu = np.asarray(data["mu_J_per_mol"], dtype=np.float64)
    iso_index = {str(s): i for i, s in enumerate(smiles)}
    no_stereo_index: dict[str, int] = {}
    for i, smi in enumerate(smiles):
        key = canonical_smiles(str(smi), isomeric=False)
        if key and key not in no_stereo_index:
            no_stereo_index[key] = i

    zf = zipfile.ZipFile(args.chaos_zip) if args.chaos_zip and Path(args.chaos_zip).exists() else None
    rows: list[np.ndarray] = []
    matched = 0
    for smi in df["canonical_smiles"].astype(str):
        iso = canonical_smiles(smi, isomeric=True)
        no = canonical_smiles(smi, isomeric=False)
        match_mode = 0.0
        idx = -1
        if iso in iso_index:
            idx = int(iso_index[iso])
            match_mode = 1.0
        elif no in no_stereo_index:
            idx = int(no_stereo_index[no])
            match_mode = 0.5
        if idx >= 0:
            matched += 1
            mu_i = np.asarray(mu[idx], dtype=np.float64)
            if zf is not None:
                try:
                    data_i = json.loads(zf.read(f"{chaos_ids[idx]}.json"))
                    sig_total = np.asarray(data_i["solvation"]["Sigma_total"], dtype=np.float64)
                    profile_i = np.interp(sigma_grid, CHAOS_PROFILE_GRID, sig_total, left=0.0, right=0.0)
                except Exception:
                    profile_i = np.zeros_like(sigma_grid)
            else:
                profile_i = np.zeros_like(sigma_grid)
            extra = np.asarray(
                [
                    1.0,
                    match_mode,
                    float(np.sum(profile_i)),
                    float(np.sum(np.abs(mu_i))),
                    float(np.max(np.abs(mu_i))) if mu_i.size else 0.0,
                ],
                dtype=np.float64,
            )
            rows.append(np.concatenate([mu_i, profile_i, extra], axis=0).astype(np.float32))
        else:
            rows.append(np.zeros((sigma_grid.size * 2 + 5,), dtype=np.float32))
    if zf is not None:
        zf.close()
    print(
        f"[biodeg-chemprop] CHAOS sigma matched {matched}/{len(df)} rows "
        f"({df.loc[[bool(x[0]) for x in rows], 'canonical_smiles'].nunique() if rows else 0} molecules)",
        flush=True,
    )
    return np.stack(rows, axis=0).astype(np.float32)


def load_records(args: argparse.Namespace) -> tuple[list[GraphRecord], pd.DataFrame]:
    df = pd.read_csv(args.target_csv)
    df = df[df["canonical_smiles"].notna() & df[args.target_col].notna()].copy().reset_index(drop=True)
    if args.max_rows and args.max_rows < len(df):
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(np.arange(len(df)), size=int(args.max_rows), replace=False))
        df = df.iloc[keep].reset_index(drop=True)

    sigma_mol = None
    if getattr(args, "sigma_mol_source", "none") == "chaos":
        sigma_mol = load_chaos_molecular_sigma(df, args)

    records: list[GraphRecord] = []
    failures: list[dict[str, Any]] = []
    t0 = time.time()
    for i, row in df.iterrows():
        smiles = str(row["canonical_smiles"])
        try:
            tensors = dipole_atom_feature_tensors(smiles, seed=args.seed, include_h=args.include_h)
        except Exception as exc:
            failures.append({"row": int(i), "smiles": smiles, "error": str(exc)})
            if args.strict:
                raise
            continue
        mol_features = tensors.molecule_features.astype(np.float32)
        if args.mol_feature_mode == "none":
            mol_features = np.zeros((0,), dtype=np.float32)
        elif args.mol_feature_mode == "safe":
            mol_features = mol_features[[0, 1, 2, 3, 4, 5, 8, 9]]
        elif args.mol_feature_mode != "all":
            raise ValueError(f"unknown --mol-feature-mode {args.mol_feature_mode!r}")
        if args.include_protocol_features:
            mol_features = np.concatenate([mol_features, protocol_features(row)], axis=0).astype(np.float32)
        if sigma_mol is not None:
            mol_features = np.concatenate([mol_features, sigma_mol[i]], axis=0).astype(np.float32)
        records.append(
            GraphRecord(
                index=int(i),
                smiles=smiles,
                chaos_id=int(i),
                node_features=tensors.atom_features.astype(np.float32),
                edge_index=tensors.edge_index.astype(np.int64),
                edge_features=tensors.edge_attr.astype(np.float32),
                mol_features=mol_features,
                y=float(row[args.target_col]),
            )
        )
        if args.progress_every and (i + 1) % args.progress_every == 0:
            elapsed = max(time.time() - t0, 1.0e-9)
            print(
                f"[biodeg-chemprop] built {i + 1}/{len(df)} records "
                f"kept={len(records)} failed={len(failures)} rate={(i + 1) / elapsed:.1f}/s",
                flush=True,
            )

    if failures:
        fail_path = Path(args.out_dir) / "graph_failures.json"
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    kept = df.iloc[[r.index for r in records]].copy().reset_index(drop=True)
    return records, kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chemprop-like Torch DMPNN CV for biodegradation consensus target.")
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--target-col", default="upper_consensus_y_percent")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/biodegradation_protocol_annotation/chemprop_torch_upper_consensus_v1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--score-dt", type=float, default=0.5)
    parser.add_argument("--model-type", choices=["score_dmpnn", "residual_mpnn"], default="score_dmpnn")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--mol-feature-mode", choices=["none", "safe", "all"], default="safe")
    parser.add_argument("--sigma-mol-source", choices=["none", "chaos"], default="none")
    parser.add_argument("--chaos-npz", type=Path, default=Path("data/chaos_25a_mu_matrix.npz"))
    parser.add_argument("--chaos-zip", type=Path, default=Path("/Users/guillaume-osmo/Github/data/CHAOS.zip"))
    parser.add_argument("--no-protocol-features", dest="include_protocol_features", action="store_false")
    parser.add_argument("--include-h", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(include_protocol_features=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    records, df = load_records(args)
    groups = df["canonical_smiles"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=args.folds).split(np.arange(len(records)), groups=groups))
    summary, pred_df = train_gnn_cv(records, folds, args)
    metrics = metric_row(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
    summary["classification_metrics"] = metrics
    summary["overall"].update(metrics)
    summary["n_rows"] = int(len(records))
    summary["n_molecules"] = int(df["canonical_smiles"].nunique())
    summary["target_col"] = args.target_col
    summary["include_protocol_features"] = bool(args.include_protocol_features)
    pred_df.to_csv(args.out_dir / "predictions.csv", index=False)
    pd.DataFrame([{**summary["overall"], **{k: v for k, v in summary.items() if not isinstance(v, (dict, list))}}]).to_csv(
        args.out_dir / "summary.csv",
        index=False,
    )
    (args.out_dir / "run_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(pd.read_csv(args.out_dir / "summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
