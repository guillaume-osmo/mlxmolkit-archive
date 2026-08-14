#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from molftp.pdl import common_unique_pair_features


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


DEFAULT_RIFM_2026 = Path("/Users/guillaume-osmo/Downloads/BioDegradationData2026.xlsx")
OSMO_REPO = Path("/Users/guillaume-osmo/Github/osmo")


@dataclass
class FeaturePrep:
    finite_mask: np.ndarray
    keep_mask: np.ndarray
    medians: np.ndarray
    arcsinh_mask: np.ndarray
    threshold: float


@dataclass
class FeatureSelector:
    indices: np.ndarray
    method: str
    requested_k: int
    scores: np.ndarray


_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_parent_mol(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest organic fragment and neutralize when chemically possible."""

    try:
        mol = rdMolStandardize.Cleanup(mol)
    except Exception:
        pass
    try:
        mol = _FRAGMENT_CHOOSER.choose(mol)
    except Exception:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if fragments:
            mol = max(fragments, key=lambda m: (m.GetNumHeavyAtoms(), Descriptors.MolWt(m)))
    try:
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def canonical_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    mol = standardize_parent_mol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def parse_days(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group(0)) if match else np.nan


def normalize_guideline(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    return text if text and text != "NAN" else "UNKNOWN"


def experiment_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Protocol-aware features for biodegradation observations.

    The workbook mixes ready-biodegradation tests, inherent-biodegradation tests,
    and older EC method names.  These features make the experimental context
    explicit instead of leaving the model to infer everything from a sparse
    one-hot guideline code.
    """

    g = df["guideline_norm"].astype(str)
    duration = df["duration_days"].to_numpy(dtype=np.float32)
    days = np.nan_to_num(duration, nan=28.0)
    upper = g.str.upper()

    def has(pattern: str) -> np.ndarray:
        return upper.str.contains(pattern, regex=True).to_numpy(dtype=np.float32)

    is_301 = has(r"OECD301|C\.4[ABCDEFG]")
    is_302 = has(r"OECD302")
    is_310 = has(r"OECD310")
    is_ready = np.maximum(is_301, is_310)
    is_inherent = is_302

    features = [
        days,
        np.log1p(days),
        days / 28.0,
        np.clip(days / 28.0, 0.0, 1.0),
        np.maximum(days - 28.0, 0.0) / 28.0,
        np.isnan(duration).astype(np.float32),
        is_ready,
        is_inherent,
        is_301,
        is_302,
        is_310,
        has(r"301A|C\.4A"),
        has(r"301B|C\.4C"),
        has(r"301C|C\.4F"),
        has(r"301D|C\.4E"),
        has(r"301F"),
        has(r"302A|C\.4B"),
        has(r"302B|C\.4D"),
        has(r"302C"),
        has(r"BODIS"),
        has(r"MITI|301C|302C"),
        has(r"CO2|301B|310|C\.4C"),
        has(r"BOD|301D|301F|BODIS|C\.4E"),
        has(r"DOC|301A|302B|C\.4A|C\.4D"),
    ]
    names = [
        "exp_duration_days",
        "exp_log1p_duration_days",
        "exp_duration_over_28",
        "exp_duration_progress_0_28",
        "exp_duration_after_28",
        "exp_duration_missing",
        "exp_ready_test",
        "exp_inherent_test",
        "exp_oecd301_family",
        "exp_oecd302_family",
        "exp_oecd310_family",
        "exp_301a_doc_dieaway",
        "exp_301b_co2_evolution",
        "exp_301c_miti",
        "exp_301d_closed_bottle",
        "exp_301f_respirometry",
        "exp_302a_scasa",
        "exp_302b_zahn_wellens",
        "exp_302c_miti_inherent",
        "exp_bodis",
        "exp_miti_like",
        "exp_co2_endpoint",
        "exp_bod_endpoint",
        "exp_doc_endpoint",
    ]
    return np.stack(features, axis=1).astype(np.float32), names


def descriptor_matrix(mols: list[Chem.Mol], names: list[str]) -> np.ndarray:
    desc_fns = dict(Descriptors.descList)
    out = np.empty((len(mols), len(names)), dtype=np.float32)
    for i, mol in enumerate(mols):
        for j, name in enumerate(names):
            try:
                value = desc_fns[name](mol)
            except Exception:
                value = np.nan
            out[i, j] = value if np.isfinite(value) else np.nan
    return out


def morgan_count_matrix(mols: list[Chem.Mol], n_bits: int, radius: int) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = gen.GetCountFingerprint(mol)
        for bit, count in fp.GetNonzeroElements().items():
            if 0 <= bit < n_bits:
                out[i, bit] = float(count)
    return out


def _load_osmo_feature_extractor():
    repo = str(OSMO_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return importlib.import_module("src.runway.mol2elointensity.features")


def osmo_feature_matrix(
    smiles: list[str],
    *,
    blocks: list[str],
    n_jobs: int,
    use_cache: bool,
) -> tuple[np.ndarray, list[str]]:
    features_mod = _load_osmo_feature_extractor()
    cache_dir = Path("benchmarks/biodegradation_pdl_ordered/feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_text = "\n".join(smiles) + "\n" + ",".join(blocks) + f"\nn_jobs={n_jobs}"
    cache_key = hashlib.md5(key_text.encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"osmo_stack_raw_{cache_key}.npz"
    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return z["features"].astype(np.float32), [str(n) for n in z["names"].tolist()]

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_idx = [i for i, mol in enumerate(mols) if mol is not None]
    valid_mols = [mols[i] for i in valid_idx]
    out_blocks: list[np.ndarray] = []
    out_names: list[str] = []

    for block in blocks:
        block = block.upper()
        if block == "RDKIT217":
            if not hasattr(rdMolDescriptors, "ExtractRDKitDescriptorsFromMolsBatch"):
                raise RuntimeError("RDKit217 C++ batch API is missing")
            names = list(rdMolDescriptors.GetRDKit217DescriptorNames())
            arr = np.full((len(smiles), len(names)), np.nan, dtype=np.float32)
            if valid_mols:
                batch = np.asarray(
                    rdMolDescriptors.ExtractRDKitDescriptorsFromMolsBatch(valid_mols, n_jobs),
                    dtype=np.float32,
                )
                if batch.ndim == 1:
                    batch = batch.reshape(1, -1)
                for k, i in enumerate(valid_idx):
                    if k < batch.shape[0]:
                        arr[i] = batch[k]
            out_blocks.append(arr)
            out_names.extend([f"osmo_stack_rdkit_{name}" for name in names])
        elif block == "OSMO":
            arr, names = features_mod.compute_osmordred_features(smiles, n_jobs=n_jobs)
            out_blocks.append(np.asarray(arr, dtype=np.float32))
            out_names.extend([f"osmo_stack_osm_{name}" for name in names])
        elif block == "ABRAHAM":
            arr, names = features_mod.compute_abraham_smarts_features(smiles, n_jobs=n_jobs)
            if arr is None or names is None:
                continue
            out_blocks.append(np.asarray(arr, dtype=np.float32))
            out_names.extend([f"osmo_stack_abraham_{name}" for name in names])
        elif block == "FUNCGROUPS":
            arr, names = features_mod.funcgroups.compute_funcgroup_features(smiles)
            out_blocks.append(np.asarray(arr, dtype=np.float32))
            out_names.extend([f"osmo_stack_fg_{name}" for name in names])
        elif block == "GOLD":
            arr, names = features_mod.compute_golden_features(smiles)
            out_blocks.append(np.asarray(arr, dtype=np.float32))
            out_names.extend([f"osmo_stack_golden_{name}" for name in names])
        else:
            raise ValueError(f"unknown OSMO feature block {block!r}")

    x = np.concatenate(out_blocks, axis=1).astype(np.float32)
    x = np.where(np.isinf(x), np.nan, x)
    if use_cache:
        np.savez_compressed(cache_path, features=x, names=np.asarray(out_names, dtype=object))
    return x, out_names


def load_rifm(path: Path, *, clip_target: bool) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "SMILES" not in df.columns or "Reviewed Data/Results" not in df.columns:
        raise ValueError(f"{path} does not look like the RIFM biodegradation workbook")

    df = df.copy()
    df["rifm_source_row"] = np.arange(len(df), dtype=np.int64)
    df["canonical_smiles"] = df["SMILES"].map(canonical_smiles)
    df["y_percent_raw"] = pd.to_numeric(df["Reviewed Data/Results"], errors="coerce")
    df["duration_days"] = df["Duration"].map(parse_days)
    df["guideline_norm"] = df["Test guideline"].map(normalize_guideline)
    df["unit_norm"] = df["Unit"].astype(str).str.strip()

    keep = df["canonical_smiles"].notna() & df["y_percent_raw"].notna()
    keep &= df["unit_norm"].eq("%") | df["Unit"].isna()
    df = df.loc[keep].reset_index(drop=True)
    df["y_percent"] = df["y_percent_raw"].clip(0.0, 100.0) if clip_target else df["y_percent_raw"]
    return df


def build_features(
    df: pd.DataFrame,
    *,
    n_bits: int,
    radius: int,
    include_biowin: bool,
    include_epi_physchem: bool,
    guideline_categories: list[str] | None = None,
    molecular_feature_stack: str = "osmo",
    feature_blocks: list[str] | None = None,
    n_jobs: int = 4,
    use_osmo_cache: bool = True,
) -> tuple[np.ndarray, list[str]]:
    mols = [Chem.MolFromSmiles(s) for s in df["canonical_smiles"].astype(str)]
    if any(m is None for m in mols):
        raise RuntimeError("canonical SMILES unexpectedly failed RDKit parsing")

    blocks: list[np.ndarray] = []
    names: list[str] = []
    if molecular_feature_stack == "osmo":
        stack_blocks = feature_blocks or ["RDKIT217", "OSMO", "GOLD", "ABRAHAM", "FUNCGROUPS"]
        x_osmo, osmo_names = osmo_feature_matrix(
            df["canonical_smiles"].astype(str).tolist(),
            blocks=stack_blocks,
            n_jobs=n_jobs,
            use_cache=use_osmo_cache,
        )
        blocks.append(x_osmo)
        names.extend(osmo_names)
    elif molecular_feature_stack == "rdkit":
        desc_names = [name for name, _ in Descriptors.descList]
        blocks.append(descriptor_matrix(mols, desc_names))
        names.extend([f"rdkit_desc_{n}" for n in desc_names])
    else:
        raise ValueError(f"unknown molecular_feature_stack={molecular_feature_stack!r}")

    blocks.append(morgan_count_matrix(mols, n_bits=n_bits, radius=radius))
    names.extend([f"morgan_count_r{radius}_{i:04d}" for i in range(n_bits)])

    exp_features, exp_names = experiment_feature_matrix(df)
    blocks.append(exp_features)
    names.extend(exp_names)

    guideline_values = df["guideline_norm"].astype(str)
    if guideline_categories is None:
        guideline = pd.get_dummies(guideline_values, prefix="guideline", dtype=np.float32)
        guideline_values_matrix = guideline.to_numpy(dtype=np.float32)
        guideline_names = list(guideline.columns)
    else:
        guideline_values_matrix = np.stack(
            [(guideline_values == cat).to_numpy(dtype=np.float32) for cat in guideline_categories],
            axis=1,
        )
        guideline_names = [f"guideline_{cat}" for cat in guideline_categories]
    blocks.append(guideline_values_matrix)
    names.extend(guideline_names)

    if include_epi_physchem:
        phys_cols = [
            "MW",
            "Predicted Log Kow",
            "Predicted Water Solubility, WSKow (mg/L)",
            "Predicted Vapor Pressure (mmHg at 25 deg C)",
            "Predicted HLC, VP/WSOL Method (atm-m3/mol at 25 deg C)",
            "Predicted HLC, VP/WSOL Method (atm-m3/mol at 25 deg C).1",
            "Predicted HLC, VP/WSOL Method (atm-m3/mol at 25 deg C).2",
        ]
        present = [c for c in phys_cols if c in df.columns]
        if present:
            blocks.append(df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
            names.extend([f"epi_{c}" for c in present])

    if include_biowin:
        biowin_cols = [
            "BioWin5 (MITI Linear Model Prediction)",
            "BioWin6 (MITI Non-Linear Model Prediction)",
        ]
        present = [c for c in biowin_cols if c in df.columns]
        if present:
            blocks.append(df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
            names.extend([f"biowin_{c}" for c in present])

    x = np.concatenate(blocks, axis=1).astype(np.float32)
    return x, names


def fit_feature_prep(x: np.ndarray, threshold: float) -> tuple[np.ndarray, FeaturePrep]:
    arr = np.asarray(x, dtype=np.float64)
    # The tabular stack intentionally drops any train column containing NaN/inf.
    # This is stricter than imputation and matches the PDL-ETR hygiene rule.
    finite_mask = np.all(np.isfinite(arr), axis=0)
    arr = arr[:, finite_mask].copy()
    med = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        rows, cols = np.where(bad)
        arr[rows, cols] = med[cols]
    std = arr.std(axis=0)
    keep = std > 1.0e-10
    arr = arr[:, keep].copy()
    max_abs = np.max(np.abs(arr), axis=0) if arr.size else np.zeros((0,), dtype=np.float64)
    arcsinh = max_abs > threshold
    if np.any(arcsinh):
        arr[:, arcsinh] = np.arcsinh(arr[:, arcsinh] / threshold)
    prep = FeaturePrep(finite_mask, keep, med, arcsinh, float(threshold))
    return arr.astype(np.float32), prep


def apply_feature_prep(x: np.ndarray, prep: FeaturePrep) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)[:, prep.finite_mask].copy()
    bad = ~np.isfinite(arr)
    if np.any(bad):
        rows, cols = np.where(bad)
        arr[rows, cols] = prep.medians[cols]
    arr = arr[:, prep.keep_mask].copy()
    if np.any(prep.arcsinh_mask):
        arr[:, prep.arcsinh_mask] = np.arcsinh(arr[:, prep.arcsinh_mask] / prep.threshold)
    return arr.astype(np.float32)


def fit_feature_selector(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    k: int,
    seed: int,
    jobs: int,
    trees: int,
) -> tuple[np.ndarray, FeatureSelector]:
    n_features = int(x.shape[1])
    if method == "none" or k <= 0 or k >= n_features:
        idx = np.arange(n_features, dtype=int)
        return x, FeatureSelector(idx, "none", int(k), np.ones(n_features, dtype=np.float32))

    if method == "rpcholesky":
        idx, scores = rpcholesky_feature_select(x, k=min(k, n_features), seed=seed)
    elif method == "variance":
        scores = np.var(x, axis=0, dtype=np.float64)
    elif method == "f_regression":
        scores, _ = f_regression(x, y, center=True, force_finite=True)
    elif method == "mutual_info":
        scores = mutual_info_regression(x, y, random_state=seed, n_neighbors=5)
    elif method == "etr_importance":
        model = ExtraTreesRegressor(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=jobs,
        )
        model.fit(x, y)
        scores = model.feature_importances_
    else:
        raise ValueError(f"unknown feature selection method: {method}")

    if method != "rpcholesky":
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        if not np.any(np.isfinite(scores)):
            idx = np.arange(min(k, n_features), dtype=int)
        else:
            idx = np.argsort(scores)[::-1][: min(k, n_features)]
    idx = np.sort(idx.astype(int))
    return x[:, idx], FeatureSelector(idx, method, int(k), scores.astype(np.float32))


def apply_feature_selector(x: np.ndarray, selector: FeatureSelector) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)[:, selector.indices]


def rpcholesky_feature_select(x: np.ndarray, *, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Unsupervised feature selection by random pivoted Cholesky on X.T @ X.

    Columns are z-scored using the current training fold only. The pivots cover
    feature-feature covariance residual variance, so selected columns are
    informative about the feature manifold without looking at the target.
    """

    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0)
    std = x64.std(axis=0)
    std = np.where(std < 1.0e-9, 1.0, std)
    z = (x64 - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    n, d = z.shape
    if d == 0 or k <= 0:
        return np.zeros((0,), dtype=int), np.zeros((d,), dtype=np.float32)
    k = min(k, d)
    sigma = (z.T @ z) / max(n, 1)
    sigma = np.asarray((sigma + sigma.T) * 0.5, dtype=np.float64)

    idx, _, trace_res = rpcholesky_columns(sigma, k_max=k, seed=seed)
    scores = np.zeros(d, dtype=np.float64)
    if len(idx):
        # Higher score for earlier pivots; trace is diagnostic, not a target score.
        scores[idx] = np.arange(len(idx), 0, -1, dtype=np.float64)
    return idx.astype(int), scores.astype(np.float32)


def rpcholesky_columns(
    kernel: np.ndarray,
    k_max: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical RPCholesky from papers/rpc_feature_selection/scripts/run_rpc_decay.py."""

    rng = np.random.default_rng(seed)
    d = kernel.shape[0]
    res_diag = kernel.diagonal().copy().astype(np.float64)
    factor = np.zeros((d, k_max), dtype=np.float64)
    idx_chosen = np.zeros(k_max, dtype=np.int64)
    trace_res = np.zeros(k_max, dtype=np.float64)

    for j in range(k_max):
        probs = np.maximum(res_diag, 0.0)
        total = probs.sum()
        if total <= 1.0e-12:
            idx_chosen = idx_chosen[:j]
            factor = factor[:, :j]
            trace_res = trace_res[:j]
            break
        pivot = int(rng.choice(d, p=probs / total))
        idx_chosen[j] = pivot
        col = kernel[:, pivot] - factor[:, :j] @ factor[pivot, :j]
        denom = np.sqrt(max(col[pivot], 1.0e-12))
        factor[:, j] = col / denom
        res_diag = np.maximum(res_diag - factor[:, j] ** 2, 0.0)
        trace_res[j] = float(res_diag.sum())

    return idx_chosen, factor, trace_res


MODEL_ALIASES = {
    "cat": "catboost",
    "cb": "catboost",
    "lightboost": "lgbm",
    "lightgbm": "lgbm",
    "linear_svr": "svm",
    "svr": "svm",
}


def normalize_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(key, key)


def parse_model_names(text: str) -> list[str]:
    names: list[str] = []
    for raw in str(text).split(","):
        name = normalize_model_name(raw)
        if name and name not in names:
            names.append(name)
    return names


class AveragingRegressor:
    def __init__(self, estimators: list[tuple[str, object]]):
        self.estimators = estimators
        self.model_names = [name for name, _ in estimators]

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.fitted_estimators_: list[tuple[str, object]] = []
        for name, estimator in self.estimators:
            t0 = time.time()
            estimator.fit(x, y)
            self.fitted_estimators_.append((name, estimator))
            print(f"[ensemble] fitted {name} in {time.time() - t0:.1f}s", flush=True)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = [np.asarray(est.predict(x), dtype=np.float32) for _, est in self.fitted_estimators_]
        return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


def make_regressor(name: str, args: argparse.Namespace, seed: int):
    name = normalize_model_name(name)
    if name == "ensemble":
        estimators: list[tuple[str, object]] = []
        base_names = parse_model_names(getattr(args, "ensemble_models", "etr,rf,xgb,catboost,svm,lgbm"))
        for offset, base_name in enumerate(base_names):
            if base_name == "ensemble":
                continue
            estimators.append((base_name, make_regressor(base_name, args, seed + 101 * (offset + 1))))
        if not estimators:
            raise RuntimeError("ensemble has no available base regressors")
        return AveragingRegressor(estimators)
    if name == "etr":
        return ExtraTreesRegressor(
            n_estimators=args.trees,
            max_features=args.max_features,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=args.jobs,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=args.trees,
            max_features=args.max_features,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=args.jobs,
        )
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=args.xgb_estimators,
            max_depth=args.xgb_depth,
            learning_rate=args.xgb_lr,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=args.jobs,
            tree_method="hist",
        )
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("catboost is not installed") from exc

        return CatBoostRegressor(
            iterations=int(getattr(args, "catboost_iterations", 300)),
            depth=int(getattr(args, "catboost_depth", 6)),
            learning_rate=float(getattr(args, "catboost_lr", 0.04)),
            loss_function="RMSE",
            random_seed=seed,
            thread_count=args.jobs,
            verbose=False,
            allow_writing_files=False,
        )
    if name == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is not installed") from exc

        return LGBMRegressor(
            n_estimators=int(getattr(args, "lgbm_estimators", 300)),
            learning_rate=float(getattr(args, "lgbm_lr", 0.04)),
            num_leaves=int(getattr(args, "lgbm_num_leaves", 31)),
            max_depth=int(getattr(args, "lgbm_depth", -1)),
            subsample=0.9,
            colsample_bytree=0.8,
            objective="regression",
            random_state=seed,
            n_jobs=args.jobs,
            verbose=-1,
        )
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVR(
                C=float(getattr(args, "svm_c", 1.0)),
                epsilon=float(getattr(args, "svm_epsilon", 0.1)),
                loss="squared_epsilon_insensitive",
                dual=False,
                random_state=seed,
                max_iter=int(getattr(args, "svm_max_iter", 5000)),
                tol=float(getattr(args, "svm_tol", 1.0e-4)),
            ),
        )
    raise ValueError(f"unknown model: {name}")


def sample_pairs(
    y: np.ndarray,
    n_pairs: int,
    min_abs_dy: float,
    seed: int,
    row_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    prob = None
    if row_weights is not None:
        weights = sanitize_anchor_weights(row_weights, n)
        if weights is not None:
            prob = weights.astype(np.float64)
            prob = prob / prob.sum()
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    have = 0
    attempts = 0
    while have < n_pairs and attempts < 20:
        budget = max(n_pairs - have, 1) * 4
        if prob is None:
            a = rng.integers(0, n, size=budget)
            b = rng.integers(0, n, size=budget)
        else:
            a = rng.choice(n, size=budget, replace=True, p=prob)
            b = rng.choice(n, size=budget, replace=True, p=prob)
        keep = a != b
        if min_abs_dy > 0:
            keep &= np.abs(y[b] - y[a]) >= min_abs_dy
        a = a[keep]
        b = b[keep]
        if len(a):
            need = n_pairs - have
            a_parts.append(a[:need])
            b_parts.append(b[:need])
            have += min(len(a), need)
        attempts += 1
    if not a_parts:
        raise RuntimeError("no PDL training pairs survived filtering")
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    return a, b, (y[b] - y[a]).astype(np.float32)


def build_pair_features(x_a: np.ndarray, x_b: np.ndarray, y_a: np.ndarray, include_abs_delta: bool) -> np.ndarray:
    pair = common_unique_pair_features(x_a, x_b, include_delta=True, include_abs_delta=include_abs_delta)
    return np.concatenate([pair, np.asarray(y_a, dtype=np.float32).reshape(-1, 1)], axis=1)


def standardize_for_neighbors(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def sanitize_anchor_weights(anchor_weights: np.ndarray | None, n_train: int) -> np.ndarray | None:
    if anchor_weights is None:
        return None
    weights = np.asarray(anchor_weights, dtype=np.float32).reshape(-1)
    if len(weights) != n_train:
        raise ValueError(f"anchor_weights has length {len(weights)} but x_train has {n_train} rows")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0).astype(np.float32)
    if not np.any(weights > 0.0):
        return None
    return weights


def weighted_row_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0).astype(np.float32)
    empty = weights.sum(axis=1) <= 0.0
    if np.any(empty):
        weights = weights.copy()
        weights[empty] = 1.0
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cutoff = 0.5 * sorted_weights.sum(axis=1, keepdims=True)
    cumsum = np.cumsum(sorted_weights, axis=1)
    pos = np.argmax(cumsum >= cutoff, axis=1)
    return sorted_values[np.arange(sorted_values.shape[0]), pos].astype(np.float32)


def predict_pdl(
    model,
    pair_prep: FeaturePrep,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    anchors: int,
    include_abs_delta: bool,
    aggregate: str,
    batch_size: int,
    anchor_weights: np.ndarray | None = None,
    anchor_candidate_factor: int = 1,
    anchor_quality_power: float = 1.0,
) -> np.ndarray:
    k = min(max(1, anchors), len(x_train))
    weights = sanitize_anchor_weights(anchor_weights, len(x_train))
    x_train_z, x_test_z = standardize_for_neighbors(x_train, x_test)
    candidate_factor = max(1, int(anchor_candidate_factor))
    candidate_k = k if weights is None else min(len(x_train), max(k, k * candidate_factor))
    nn = NearestNeighbors(n_neighbors=candidate_k, metric="euclidean")
    nn.fit(x_train_z)
    if weights is None:
        anchor_idx = nn.kneighbors(x_test_z, return_distance=False)
        selected_weights = None
    else:
        dist, candidate_idx = nn.kneighbors(x_test_z, return_distance=True)
        candidate_weights = weights[candidate_idx]
        quality_power = max(0.0, float(anchor_quality_power))
        quality = np.maximum(candidate_weights, 1.0e-6) ** quality_power
        effective_dist = dist.astype(np.float32) / quality
        take = np.argsort(effective_dist, axis=1)[:, :k]
        anchor_idx = np.take_along_axis(candidate_idx, take, axis=1)
        selected_weights = np.take_along_axis(candidate_weights, take, axis=1)

    preds = np.empty((len(x_test),), dtype=np.float32)
    for start in range(0, len(x_test), batch_size):
        stop = min(start + batch_size, len(x_test))
        local = anchor_idx[start:stop]
        local_weights = selected_weights[start:stop] if selected_weights is not None else None
        flat_anchor = local.reshape(-1)
        x_a = x_train[flat_anchor]
        x_b = np.repeat(x_test[start:stop], k, axis=0)
        y_a = y_train[flat_anchor]
        pair_raw = build_pair_features(x_a, x_b, y_a, include_abs_delta)
        pair = apply_feature_prep(pair_raw, pair_prep)
        dy = model.predict(pair).reshape(stop - start, k)
        candidates = y_train[local] + dy
        if aggregate == "mean":
            if local_weights is None:
                preds[start:stop] = candidates.mean(axis=1)
            else:
                denom = np.maximum(local_weights.sum(axis=1), 1.0e-12)
                preds[start:stop] = (candidates * local_weights).sum(axis=1) / denom
        else:
            if local_weights is None:
                preds[start:stop] = np.median(candidates, axis=1)
            else:
                preds[start:stop] = weighted_row_median(candidates, local_weights)
    return preds


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def order_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, n_pairs: int, min_abs_dy: float, seed: int) -> float:
    a, b, dy = sample_pairs(y_true, n_pairs=n_pairs, min_abs_dy=min_abs_dy, seed=seed)
    dp = y_pred[b] - y_pred[a]
    return float(np.mean(np.sign(dy) == np.sign(dp)))


def make_folds(df: pd.DataFrame, n_splits: int, seed: int, split: str):
    idx = np.arange(len(df))
    if split == "group":
        groups = df["canonical_smiles"].astype(str).to_numpy()
        n = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n)
        return list(splitter.split(idx, groups=groups))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(idx))


def write_2024_2026_diff(path_2024: Path, path_2026: Path, out_dir: Path) -> dict:
    if not path_2024.exists() or not path_2026.exists():
        return {"available": False}
    frames = {}
    for label, path in [("2024", path_2024), ("2026", path_2026)]:
        df = load_rifm(path, clip_target=False)
        frames[label] = df
    f24 = frames["2024"]
    f26 = frames["2026"]
    key_cols = ["canonical_smiles", "guideline_norm", "Duration", "y_percent_raw", "Reference"]
    a = set(map(tuple, f24[key_cols].astype(str).values.tolist()))
    b = set(map(tuple, f26[key_cols].astype(str).values.tolist()))
    added = f26[[tuple(row) in (b - a) for row in f26[key_cols].astype(str).values.tolist()]]
    removed = f24[[tuple(row) in (a - b) for row in f24[key_cols].astype(str).values.tolist()]]
    added.to_csv(out_dir / "rifm_2026_added_vs_2024.csv", index=False)
    removed.to_csv(out_dir / "rifm_2024_removed_vs_2026.csv", index=False)
    return {
        "available": True,
        "rows_2024": int(len(f24)),
        "rows_2026": int(len(f26)),
        "unique_molecules_2024": int(f24["canonical_smiles"].nunique()),
        "unique_molecules_2026": int(f26["canonical_smiles"].nunique()),
        "added_obs": int(len(added)),
        "removed_obs": int(len(removed)),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = load_rifm(Path(args.rifm_xlsx), clip_target=args.clip_target)
    x_raw, feature_names = build_features(
        df,
        n_bits=args.n_bits,
        radius=args.radius,
        include_biowin=args.include_biowin,
        include_epi_physchem=args.include_epi_physchem,
        molecular_feature_stack=args.molecular_feature_stack,
        feature_blocks=[b.strip().upper() for b in args.feature_blocks.split(",") if b.strip()],
        n_jobs=args.jobs,
        use_osmo_cache=not args.no_osmo_cache,
    )
    y = df["y_percent"].to_numpy(dtype=np.float32)
    folds = make_folds(df, args.folds, args.seed, args.split)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for model_name in models:
        direct_pred = np.full(len(df), np.nan, dtype=np.float32)
        pdl_pred = np.full(len(df), np.nan, dtype=np.float32)

        for fold_id, (tr, va) in enumerate(folds, start=1):
            x_train, prep = fit_feature_prep(x_raw[tr], args.arcsinh_threshold)
            x_val = apply_feature_prep(x_raw[va], prep)
            y_train = y[tr]
            x_train, selector = fit_feature_selector(
                x_train,
                y_train,
                method=args.select_method,
                k=args.select_k,
                seed=args.seed + 3000 + fold_id,
                jobs=args.jobs,
                trees=args.select_trees,
            )
            x_val = apply_feature_selector(x_val, selector)

            direct = make_regressor(model_name, args, args.seed + fold_id)
            direct.fit(x_train, y_train)
            direct_pred[va] = direct.predict(x_val).astype(np.float32)

            pair_a, pair_b, dy = sample_pairs(
                y_train,
                n_pairs=args.pairs_per_fold,
                min_abs_dy=args.min_abs_dy,
                seed=args.seed + 1000 + fold_id,
            )
            pair_raw = build_pair_features(
                x_train[pair_a],
                x_train[pair_b],
                y_train[pair_a],
                include_abs_delta=args.include_abs_delta,
            )
            pair_train, pair_prep = fit_feature_prep(pair_raw, args.arcsinh_threshold)
            pdl = make_regressor(model_name, args, args.seed + 2000 + fold_id)
            pdl.fit(pair_train, dy)
            pdl_pred[va] = predict_pdl(
                pdl,
                pair_prep,
                x_train,
                y_train,
                x_val,
                anchors=args.anchors,
                include_abs_delta=args.include_abs_delta,
                aggregate=args.aggregate,
                batch_size=args.predict_batch_size,
            )

            print(
                f"[biodeg-pdl] model={model_name} fold={fold_id}/{len(folds)} "
                f"train={len(tr)} val={len(va)} features={x_train.shape[1]} pairs={len(dy)} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

        for method, pred in [(f"{model_name}_direct", direct_pred), (f"{model_name}_pdl_delta", pdl_pred)]:
            row = {
                "method": method,
                "split": args.split,
                "n": int(np.isfinite(pred).sum()),
                "clip_target": bool(args.clip_target),
                "include_epi_physchem": bool(args.include_epi_physchem),
                "include_biowin": bool(args.include_biowin),
                "n_raw_features": int(x_raw.shape[1]),
                "select_method": args.select_method,
                "select_k": int(args.select_k),
            }
            row.update(metric_row(y, pred))
            row["order_acc_5pct"] = order_accuracy(y, pred, n_pairs=args.eval_pairs, min_abs_dy=5.0, seed=args.seed + 9)
            row["order_acc_10pct"] = order_accuracy(y, pred, n_pairs=args.eval_pairs, min_abs_dy=10.0, seed=args.seed + 10)
            summary_rows.append(row)

        pred_df = df[
            [
                "canonical_smiles",
                "SMILES",
                "CAS",
                "Chemical_name",
                "guideline_norm",
                "Test guideline",
                "Duration",
                "duration_days",
                "y_percent_raw",
                "y_percent",
            ]
        ].copy()
        pred_df[f"{model_name}_direct_pred"] = direct_pred
        pred_df[f"{model_name}_pdl_delta_pred"] = pdl_pred
        prediction_frames.append(pred_df)

    summary = pd.DataFrame(summary_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

    merged_pred = prediction_frames[0]
    for frame in prediction_frames[1:]:
        cols = [c for c in frame.columns if c.endswith("_pred")]
        merged_pred = pd.concat([merged_pred, frame[cols]], axis=1)
    merged_pred.to_csv(out_dir / "predictions.csv", index=False)

    meta = {
        "rifm_xlsx": str(Path(args.rifm_xlsx)),
        "rows": int(len(df)),
        "unique_molecules": int(df["canonical_smiles"].nunique()),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "target_mean": float(np.mean(y)),
        "feature_count": int(x_raw.shape[1]),
        "select_method": args.select_method,
        "select_k": int(args.select_k),
        "feature_names_path": "feature_names.txt",
        "args": vars(args),
        "elapsed_seconds": float(time.time() - t0),
        "rifm_2024_2026_diff": write_2024_2026_diff(Path(args.rifm_2024_xlsx), Path(args.rifm_xlsx), out_dir),
    }
    (out_dir / "run_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[biodeg-pdl] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise PDL regression for RIFM biodegradation percent data.")
    parser.add_argument("--rifm-xlsx", default=str(DEFAULT_RIFM_2026))
    parser.add_argument("--rifm-2024-xlsx", default="/Users/guillaume-osmo/Downloads/BioDegradationData2024 (2).xlsx")
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_pdl_ordered/rifm2026_percent_groupkfold")
    parser.add_argument("--split", choices=["group", "row"], default="group")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", default="etr")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--molecular-feature-stack", choices=["osmo", "rdkit"], default="osmo")
    parser.add_argument(
        "--feature-blocks",
        default="RDKIT217,OSMO,GOLD,ABRAHAM,FUNCGROUPS",
        help="OSMO stack blocks; GOLD is the calcPhysChem/v34 cascade feature block.",
    )
    parser.add_argument("--no-osmo-cache", action="store_true")
    parser.add_argument("--clip-target", action="store_true", help="Clip percent biodegradation target to [0, 100].")
    parser.add_argument("--include-epi-physchem", action="store_true")
    parser.add_argument("--include-biowin", action="store_true", help="Diagnostic only; BioWin is a biodegradation predictor.")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--select-method",
        choices=["none", "rpcholesky", "variance", "f_regression", "mutual_info", "etr_importance"],
        default="rpcholesky",
    )
    parser.add_argument("--select-k", type=int, default=1024)
    parser.add_argument("--select-trees", type=int, default=50)
    parser.add_argument("--xgb-estimators", type=int, default=600)
    parser.add_argument("--xgb-depth", type=int, default=4)
    parser.add_argument("--xgb-lr", type=float, default=0.035)
    parser.add_argument("--ensemble-models", default="etr,rf,xgb,catboost,svm,lgbm")
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-lr", type=float, default=0.04)
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--lgbm-depth", type=int, default=-1)
    parser.add_argument("--lgbm-lr", type=float, default=0.04)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-epsilon", type=float, default=0.1)
    parser.add_argument("--svm-max-iter", type=int, default=5000)
    parser.add_argument("--svm-tol", type=float, default=1.0e-4)
    parser.add_argument("--pairs-per-fold", type=int, default=80_000)
    parser.add_argument("--min-abs-dy", type=float, default=3.0)
    parser.add_argument("--include-abs-delta", action="store_true")
    parser.add_argument("--anchors", type=int, default=96)
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--predict-batch-size", type=int, default=64)
    parser.add_argument("--arcsinh-threshold", type=float, default=100.0)
    parser.add_argument("--eval-pairs", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
