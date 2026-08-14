#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from train_biodegradation_chemprop_torch import DEFAULT_TARGET, load_records
from train_dipole_sigma_gnn import train_gnn_cv


def make_args(base: argparse.Namespace, params: dict[str, Any], out_dir: Path) -> SimpleNamespace:
    values = vars(base).copy()
    values.update(params)
    values["out_dir"] = out_dir
    values["model_type"] = "score_dmpnn"
    values["include_protocol_features"] = True
    values["target_col"] = base.target_col
    values["verbose"] = bool(base.verbose_trials)
    values["log_every"] = int(base.log_every)
    values["progress_every"] = 0
    values["compile"] = False
    values["compile_backend"] = "inductor"
    values["compile_mode"] = "default"
    values["compile_fullgraph"] = False
    values["strict"] = False
    values["include_h"] = False
    return SimpleNamespace(**values)


def sample_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hidden": trial.suggest_categorical("hidden", [64, 96, 128, 160, 192, 256]),
        "layers": trial.suggest_int("layers", 2, 6),
        "head_layers": trial.suggest_int("head_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.22),
        "lr": trial.suggest_float("lr", 4.0e-4, 4.0e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1.0e-6, 3.0e-3, log=True),
        "score_dt": trial.suggest_float("score_dt", 0.25, 0.85),
        "batch_size": trial.suggest_categorical("batch_size", [64, 96, 128, 192]),
        "epochs": int(args.epochs_per_trial),
        "patience": int(args.patience_per_trial),
    }


def objective_factory(records, folds, args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, args)
        trial_dir = Path(args.out_dir) / "trials" / f"trial_{trial.number:04d}"
        trial_args = make_args(args, params, trial_dir)
        t0 = time.time()
        summary, pred_df = train_gnn_cv(records, folds[: args.folds_per_trial], trial_args)
        score = float(summary["overall"]["mae"])
        metric = {
            "trial": int(trial.number),
            "mae": score,
            "rmse": float(summary["overall"]["rmse"]),
            "r2": float(summary["overall"]["r2"]),
            "seconds": float(time.time() - t0),
            **params,
        }
        trial.set_user_attr("metrics", metric)
        trial_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(trial_dir / "predictions.csv", index=False)
        (trial_dir / "summary.json").write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pd.DataFrame([metric]).to_csv(trial_dir / "summary.csv", index=False)
        print(
            "[optuna-dmpnn] "
            f"trial={trial.number} mae={metric['mae']:.4f} rmse={metric['rmse']:.4f} "
            f"r2={metric['r2']:.4f} params={params}",
            flush=True,
        )
        return score if math.isfinite(score) else 1.0e9

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for biodegradation SCORE-DMPNN.")
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--target-col", default="upper_consensus_y_percent")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/biodegradation_protocol_annotation/optuna_score_dmpnn_upper_consensus_v1"))
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--folds-per-trial", type=int, default=3)
    parser.add_argument("--final-folds", type=int, default=5)
    parser.add_argument("--epochs-per-trial", type=int, default=45)
    parser.add_argument("--patience-per-trial", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=90)
    parser.add_argument("--final-patience", type=int, default=18)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--mol-feature-mode", choices=["none", "safe", "all"], default="safe")
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--verbose-trials", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_args = SimpleNamespace(
        target_csv=args.target_csv,
        target_col=args.target_col,
        max_rows=args.max_rows,
        seed=args.seed,
        include_h=False,
        strict=False,
        mol_feature_mode=args.mol_feature_mode,
        include_protocol_features=True,
        out_dir=args.out_dir,
        progress_every=250,
    )
    records, df = load_records(load_args)
    groups = df["canonical_smiles"].astype(str).to_numpy()
    all_folds = list(GroupKFold(n_splits=max(args.final_folds, args.folds_per_trial)).split(np.arange(len(records)), groups=groups))
    study = optuna.create_study(
        study_name="biodeg_score_dmpnn_upper_consensus",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective_factory(records, all_folds, args), n_trials=args.n_trials, gc_after_trial=True)
    trial_rows = []
    for trial in study.trials:
        row = dict(trial.user_attrs.get("metrics", {}))
        row["state"] = str(trial.state)
        row["value"] = trial.value
        trial_rows.append(row)
    trials_df = pd.DataFrame(trial_rows).sort_values("mae", na_position="last")
    trials_df.to_csv(args.out_dir / "trials_summary.csv", index=False)
    best_params = dict(study.best_trial.params)
    best_report = {"best_trial": int(study.best_trial.number), "best_value": float(study.best_value), "best_params": best_params}
    (args.out_dir / "best_params.json").write_text(json.dumps(best_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[optuna-dmpnn] best", json.dumps(best_report, sort_keys=True), flush=True)

    if not args.skip_final:
        final_params = best_params.copy()
        final_params.update({"epochs": int(args.final_epochs), "patience": int(args.final_patience)})
        final_args = make_args(args, final_params, args.out_dir / "best_cv5")
        final_args.verbose = True
        final_args.log_every = 10
        summary, pred_df = train_gnn_cv(records, all_folds[: args.final_folds], final_args)
        pred_df.to_csv(args.out_dir / "best_cv5_predictions.csv", index=False)
        flat = {
            **summary["overall"],
            "n_rows": int(len(records)),
            "n_molecules": int(df["canonical_smiles"].nunique()),
            "best_trial": int(study.best_trial.number),
            **best_params,
        }
        pd.DataFrame([flat]).to_csv(args.out_dir / "best_cv5_summary.csv", index=False)
        (args.out_dir / "best_cv5_run_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(pd.DataFrame([flat]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
