#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class GraphRecord:
    index: int
    smiles: str
    chaos_id: int
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    mol_features: np.ndarray
    y: float


@dataclass
class Scalers:
    node_kept_mask: np.ndarray
    node_arcsinh_mask: np.ndarray
    node_mean: np.ndarray
    node_std: np.ndarray
    edge_kept_mask: np.ndarray
    edge_arcsinh_mask: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray
    mol_kept_mask: np.ndarray
    mol_arcsinh_mask: np.ndarray
    mol_mean: np.ndarray
    mol_std: np.ndarray
    y_mean: float
    y_std: float
    arcsinh_scale: float


SAFE_MOL_FEATURE_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 9, 10, 11], dtype=np.int64)
DIPOLE_PROXY_MOL_FEATURE_INDICES = np.asarray([6, 7, 8], dtype=np.int64)


def _as_float_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_std(x: np.ndarray, axis: int = 0) -> np.ndarray:
    std = np.asarray(np.std(x, axis=axis), dtype=np.float32)
    return np.where(std < 1.0e-8, 1.0, std).astype(np.float32)


def fit_numeric_feature_transform(x: np.ndarray, arcsinh_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x64)
    finite &= np.abs(x64) < 1.0e300
    kept_mask = np.all(finite, axis=0)
    x_keep = x64[:, kept_mask].copy()
    if x_keep.size == 0:
        arcsinh_mask = np.zeros((0,), dtype=bool)
    else:
        max_abs = np.max(np.abs(x_keep), axis=0)
        arcsinh_mask = max_abs > arcsinh_scale
        x_keep[:, arcsinh_mask] = np.arcsinh(x_keep[:, arcsinh_mask] / arcsinh_scale)
    return x_keep.astype(np.float32), kept_mask, arcsinh_mask


def apply_numeric_feature_transform(
    x: np.ndarray,
    kept_mask: np.ndarray,
    arcsinh_mask: np.ndarray,
    arcsinh_scale: float,
) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)[:, kept_mask].copy()
    x64[~np.isfinite(x64)] = 0.0
    too_large = np.abs(x64) >= 1.0e300
    x64[too_large] = 0.0
    if np.any(arcsinh_mask):
        x64[:, arcsinh_mask] = np.arcsinh(x64[:, arcsinh_mask] / arcsinh_scale)
    return x64.astype(np.float32)


def _select_molecule_features(mol_base: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sigma_only":
        return np.zeros((0,), dtype=np.float32)
    if mode == "safe":
        return mol_base[SAFE_MOL_FEATURE_INDICES]
    if mode == "all":
        return mol_base
    raise ValueError(f"Unknown molecule feature mode: {mode}")


def load_sigma_graph_dataset(
    data_dir: Path,
    max_mols: int | None = None,
    seed: int = 42,
    mol_feature_mode: str = "safe",
) -> tuple[list[GraphRecord], pd.DataFrame]:
    df = pd.read_csv(data_dir / "dipole_sigma_chemprop.csv")
    atom_npz = np.load(data_dir / "atom_features.npz", allow_pickle=False)
    mol_npz = np.load(data_dir / "molecule_features.npz", allow_pickle=False)
    graph_npz = np.load(data_dir / "graph_tensors.npz", allow_pickle=False)
    sigma_npz = np.load(data_dir / "sigma_features.npz", allow_pickle=False)

    sigma_mu = _as_float_matrix(sigma_npz["mu_J_per_mol"])
    sigma_profile = _as_float_matrix(sigma_npz["profile_area_A2"])

    n_total = len(df)
    order = np.arange(n_total)
    if max_mols is not None and max_mols < n_total:
        rng = np.random.default_rng(seed)
        order = np.sort(rng.choice(order, size=max_mols, replace=False))
        df = df.iloc[order].reset_index(drop=True)

    records: list[GraphRecord] = []
    for local_i, source_i in enumerate(order):
        node = _as_float_matrix(atom_npz[f"arr_{source_i}"])
        edge_index = np.asarray(graph_npz[f"edge_index_{source_i}"], dtype=np.int64)
        edge_features = _as_float_matrix(graph_npz[f"edge_attr_{source_i}"])
        mol_base_full = _as_float_matrix(mol_npz[f"arr_{source_i}"]).reshape(-1)
        mol_base = _select_molecule_features(mol_base_full, mol_feature_mode)
        mol_features = np.concatenate([mol_base, sigma_mu[source_i], sigma_profile[source_i]], axis=0).astype(np.float32)
        row = df.iloc[local_i]
        records.append(
            GraphRecord(
                index=int(source_i),
                smiles=str(row["smiles"]),
                chaos_id=int(row["chaos_id"]),
                node_features=node,
                edge_index=edge_index,
                edge_features=edge_features,
                mol_features=mol_features,
                y=float(row["dipole_debye"]),
            )
        )
    return records, df


def fit_scalers(records: list[GraphRecord], train_idx: np.ndarray, arcsinh_scale: float) -> Scalers:
    train_records = [records[int(i)] for i in train_idx]
    nodes = np.concatenate([r.node_features for r in train_records], axis=0).astype(np.float32)
    edges_list = [r.edge_features for r in train_records if r.edge_features.size]
    if edges_list:
        edges = np.concatenate(edges_list, axis=0).astype(np.float32)
    else:
        edge_dim = train_records[0].edge_features.shape[1]
        edges = np.zeros((1, edge_dim), dtype=np.float32)
    mol = np.stack([r.mol_features for r in train_records], axis=0).astype(np.float32)
    nodes_t, node_kept_mask, node_arcsinh_mask = fit_numeric_feature_transform(nodes, arcsinh_scale)
    edges_t, edge_kept_mask, edge_arcsinh_mask = fit_numeric_feature_transform(edges, arcsinh_scale)
    mol_t, mol_kept_mask, mol_arcsinh_mask = fit_numeric_feature_transform(mol, arcsinh_scale)
    y = np.asarray([r.y for r in train_records], dtype=np.float32)
    y_std = float(np.std(y))
    if y_std < 1.0e-8:
        y_std = 1.0
    return Scalers(
        node_kept_mask=node_kept_mask,
        node_arcsinh_mask=node_arcsinh_mask,
        node_mean=np.mean(nodes_t, axis=0).astype(np.float32),
        node_std=_safe_std(nodes_t, axis=0),
        edge_kept_mask=edge_kept_mask,
        edge_arcsinh_mask=edge_arcsinh_mask,
        edge_mean=np.mean(edges_t, axis=0).astype(np.float32),
        edge_std=_safe_std(edges_t, axis=0),
        mol_kept_mask=mol_kept_mask,
        mol_arcsinh_mask=mol_arcsinh_mask,
        mol_mean=np.mean(mol_t, axis=0).astype(np.float32),
        mol_std=_safe_std(mol_t, axis=0),
        y_mean=float(np.mean(y)),
        y_std=y_std,
        arcsinh_scale=float(arcsinh_scale),
    )


def make_batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator) -> Iterable[np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, idx.size, batch_size):
        yield idx[start : start + batch_size]


def collate_graphs(records: list[GraphRecord], idx: np.ndarray, scalers: Scalers, device: str):
    import torch

    xs: list[np.ndarray] = []
    eis: list[np.ndarray] = []
    eas: list[np.ndarray] = []
    batch_ids: list[np.ndarray] = []
    mols: list[np.ndarray] = []
    ys: list[float] = []
    offset = 0
    for b, i in enumerate(idx):
        r = records[int(i)]
        node_t = apply_numeric_feature_transform(
            r.node_features,
            scalers.node_kept_mask,
            scalers.node_arcsinh_mask,
            scalers.arcsinh_scale,
        )
        edge_t = apply_numeric_feature_transform(
            r.edge_features,
            scalers.edge_kept_mask,
            scalers.edge_arcsinh_mask,
            scalers.arcsinh_scale,
        )
        mol_t = apply_numeric_feature_transform(
            r.mol_features[None, :],
            scalers.mol_kept_mask,
            scalers.mol_arcsinh_mask,
            scalers.arcsinh_scale,
        ).reshape(-1)
        x = (node_t - scalers.node_mean) / scalers.node_std
        e = (edge_t - scalers.edge_mean) / scalers.edge_std
        m = (mol_t - scalers.mol_mean) / scalers.mol_std
        xs.append(x.astype(np.float32))
        if r.edge_index.size:
            eis.append((r.edge_index + offset).astype(np.int64))
            eas.append(e.astype(np.float32))
        batch_ids.append(np.full(r.node_features.shape[0], b, dtype=np.int64))
        mols.append(m.astype(np.float32))
        ys.append((r.y - scalers.y_mean) / scalers.y_std)
        offset += r.node_features.shape[0]

    edge_dim = scalers.edge_mean.shape[0]
    edge_index = np.concatenate(eis, axis=1) if eis else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.concatenate(eas, axis=0) if eas else np.zeros((0, edge_dim), dtype=np.float32)
    edge_reverse = np.full(edge_index.shape[1], -1, dtype=np.int64)
    if edge_index.shape[1]:
        edge_pos = {(int(s), int(d)): e for e, (s, d) in enumerate(edge_index.T)}
        for e, (s, d) in enumerate(edge_index.T):
            edge_reverse[e] = edge_pos.get((int(d), int(s)), -1)
    return {
        "x": torch.as_tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(edge_attr, dtype=torch.float32, device=device),
        "edge_reverse": torch.as_tensor(edge_reverse, dtype=torch.long, device=device),
        "batch": torch.as_tensor(np.concatenate(batch_ids, axis=0), dtype=torch.long, device=device),
        "mol": torch.as_tensor(np.stack(mols, axis=0), dtype=torch.float32, device=device),
        "y": torch.as_tensor(np.asarray(ys, dtype=np.float32), dtype=torch.float32, device=device),
    }


def _make_mlp(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int, dropout: float):
    from torch import nn

    modules: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, n_hidden_layers)):
        modules.extend([nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout)])
        dim = hidden
    modules.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*modules)


def build_torch_model(
    node_dim: int,
    edge_dim: int,
    mol_dim: int,
    hidden: int,
    layers: int,
    dropout: float,
    *,
    model_type: str = "score_dmpnn",
    head_layers: int = 2,
    score_dt: float = 0.5,
):
    import torch
    from torch import nn

    class ResidualMessageLayer(nn.Module):
        def __init__(self, hidden_dim: int, edge_hidden_dim: int, dropout_p: float):
            super().__init__()
            self.message = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.update = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, h, edge_index, edge_h):
            if edge_index.numel() == 0:
                agg = torch.zeros_like(h)
            else:
                src = edge_index[0]
                dst = edge_index[1]
                msg = self.message(torch.cat([h[src], h[dst], edge_h], dim=-1))
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, msg)
            return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))

    class ScoreDMPNNBlock(nn.Module):
        """Directed D-MPNN update with SCORE recurrent residual depth.

        A single shared block is iterated `layers` times:
            m_{t+1} = (1 - dt) m_t + dt F(m_t)
        where F uses incoming directed messages and excludes the immediate reverse edge.
        """

        def __init__(self, hidden_dim: int, dropout_p: float, dt: float):
            super().__init__()
            self.dt = float(dt)
            self.update = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, m, edge_index, edge_reverse, n_nodes: int):
            if edge_index.numel() == 0:
                return m
            src = edge_index[0]
            dst = edge_index[1]
            incoming = m.new_zeros((n_nodes, m.shape[-1]))
            incoming.index_add_(0, dst, m)
            rev_idx = edge_reverse.clamp_min(0)
            reverse = m[rev_idx]
            reverse = reverse * (edge_reverse >= 0).to(m.dtype).unsqueeze(-1)
            context = incoming[src] - reverse
            proposal = self.update(torch.cat([m, context], dim=-1))
            return self.norm((1.0 - self.dt) * m + self.dt * proposal)

    class SigmaGraphRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_type = model_type
            self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
            self.edge_proj = nn.Sequential(
                nn.Linear(hidden + edge_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            if model_type == "residual_mpnn":
                self.layers = nn.ModuleList([ResidualMessageLayer(hidden, hidden, dropout) for _ in range(layers)])
                self.score_block = None
                self.readout = nn.Identity()
            elif model_type == "score_dmpnn":
                self.layers = nn.ModuleList()
                self.score_block = ScoreDMPNNBlock(hidden, dropout, score_dt)
                self.readout = nn.Sequential(
                    nn.Linear(2 * hidden, hidden),
                    nn.SiLU(),
                    nn.LayerNorm(hidden),
                )
            else:
                raise ValueError(f"Unknown model_type={model_type!r}")
            self.head = _make_mlp(2 * hidden + mol_dim, hidden, 1, head_layers, dropout)

        def forward(self, x, edge_index, edge_attr, edge_reverse, batch, mol):
            h = self.node_proj(x)
            if edge_attr.numel():
                edge_h = self.edge_proj(torch.cat([h[edge_index[0]], edge_attr], dim=-1))
            else:
                edge_h = edge_attr.new_zeros((0, h.shape[-1]))
            if self.model_type == "residual_mpnn":
                for layer in self.layers:
                    h = layer(h, edge_index, edge_h)
            else:
                for _ in range(layers):
                    edge_h = self.score_block(edge_h, edge_index, edge_reverse, h.shape[0])
                incoming = h.new_zeros((h.shape[0], h.shape[-1]))
                if edge_index.numel():
                    incoming.index_add_(0, edge_index[1], edge_h)
                h = self.readout(torch.cat([h, incoming], dim=-1))

            n_graphs = int(mol.shape[0])
            sum_pool = h.new_zeros((n_graphs, h.shape[-1]))
            sum_pool.index_add_(0, batch, h)
            count = h.new_zeros((n_graphs, 1))
            count.index_add_(0, batch, torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device))
            mean_pool = sum_pool / count.clamp_min(1.0)
            sqrt_pool = sum_pool / torch.sqrt(count.clamp_min(1.0))
            return self.head(torch.cat([mean_pool, sqrt_pool, mol], dim=-1)).squeeze(-1)

    return SigmaGraphRegressor()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / denom) if denom > 0.0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


@dataclass
class TabularPreprocessResult:
    x: np.ndarray
    names: list[str]
    kept_mask: np.ndarray
    arcsinh_mask: np.ndarray
    n_dropped_nonfinite: int
    n_arcsinh: int


def preprocess_tabular_features(
    x: np.ndarray,
    names: list[str],
    *,
    high_abs_threshold: float = 100.0,
) -> TabularPreprocessResult:
    """Drop bad descriptor columns, then compress high-dynamic-range columns.

    The two steps are intentionally simple and dataset-level:
    1. remove any column containing NaN/inf/overflow-scale values;
    2. apply arcsinh(A / threshold) to columns whose remaining max(|A|) exceeds the threshold.
    """

    x64 = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x64)
    finite &= np.abs(x64) < 1.0e300
    kept_mask = np.all(finite, axis=0)
    x_clean = x64[:, kept_mask].copy()
    kept_names = [name for name, keep in zip(names, kept_mask) if bool(keep)]
    if x_clean.size == 0:
        arcsinh_mask = np.zeros((0,), dtype=bool)
    else:
        max_abs = np.max(np.abs(x_clean), axis=0)
        arcsinh_mask = max_abs > high_abs_threshold
        x_clean[:, arcsinh_mask] = np.arcsinh(x_clean[:, arcsinh_mask] / high_abs_threshold)
    return TabularPreprocessResult(
        x=x_clean.astype(np.float32),
        names=kept_names,
        kept_mask=kept_mask,
        arcsinh_mask=arcsinh_mask,
        n_dropped_nonfinite=int(np.size(kept_mask) - np.count_nonzero(kept_mask)),
        n_arcsinh=int(np.count_nonzero(arcsinh_mask)),
    )


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_gnn_cv(
    records: list[GraphRecord],
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    import torch

    if args.torch_threads is not None and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    rng = np.random.default_rng(args.seed)
    raw_node_dim = records[0].node_features.shape[1]
    raw_edge_dim = records[0].edge_features.shape[1]
    raw_mol_dim = records[0].mol_features.shape[0]
    node_dim = raw_node_dim
    edge_dim = raw_edge_dim
    mol_dim = raw_mol_dim
    rows: list[dict] = []
    fold_metrics: list[dict] = []
    t0 = time.time()

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        torch.manual_seed(args.seed + fold_id)
        random.seed(args.seed + fold_id)
        scalers = fit_scalers(records, train_idx, arcsinh_scale=args.arcsinh_threshold)
        node_dim = int(scalers.node_mean.shape[0])
        edge_dim = int(scalers.edge_mean.shape[0])
        mol_dim = int(scalers.mol_mean.shape[0])
        if args.verbose:
            print(
                f"[gnn] fold={fold_id + 1}/{len(folds)} preprocess "
                f"node {raw_node_dim}->{node_dim} arcsinh={int(np.count_nonzero(scalers.node_arcsinh_mask))}; "
                f"edge {raw_edge_dim}->{edge_dim} arcsinh={int(np.count_nonzero(scalers.edge_arcsinh_mask))}; "
                f"mol {raw_mol_dim}->{mol_dim} arcsinh={int(np.count_nonzero(scalers.mol_arcsinh_mask))}",
                flush=True,
            )
        model = build_torch_model(
            node_dim,
            edge_dim,
            mol_dim,
            args.hidden,
            args.layers,
            args.dropout,
            model_type=args.model_type,
            head_layers=args.head_layers,
            score_dt=args.score_dt,
        ).to(device)
        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("--compile requested, but this torch build does not provide torch.compile")
            model = torch.compile(
                model,
                backend=args.compile_backend,
                mode=args.compile_mode,
                fullgraph=args.compile_fullgraph,
            )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_state = None
        best_val = math.inf
        bad_epochs = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            for batch_idx in make_batches(train_idx, args.batch_size, True, rng):
                batch = collate_graphs(records, batch_idx, scalers, device)
                opt.zero_grad(set_to_none=True)
                pred = model(
                    batch["x"],
                    batch["edge_index"],
                    batch["edge_attr"],
                    batch["edge_reverse"],
                    batch["batch"],
                    batch["mol"],
                )
                loss = torch.nn.functional.mse_loss(pred, batch["y"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            preds_std: list[np.ndarray] = []
            y_std_true: list[np.ndarray] = []
            with torch.no_grad():
                for batch_idx in make_batches(val_idx, args.batch_size, False, rng):
                    batch = collate_graphs(records, batch_idx, scalers, device)
                    pred = model(
                        batch["x"],
                        batch["edge_index"],
                        batch["edge_attr"],
                        batch["edge_reverse"],
                        batch["batch"],
                        batch["mol"],
                    )
                    preds_std.append(pred.detach().cpu().numpy())
                    y_std_true.append(batch["y"].detach().cpu().numpy())
            pred_val = np.concatenate(preds_std) * scalers.y_std + scalers.y_mean
            true_val = np.concatenate(y_std_true) * scalers.y_std + scalers.y_mean
            val_metrics = _metrics(true_val, pred_val)
            val_mae = val_metrics["mae"]
            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if args.verbose and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
                print(
                    f"[gnn] fold={fold_id + 1}/{len(folds)} epoch={epoch:03d} "
                    f"loss={np.mean(train_losses):.4f} "
                    f"val_mae={val_metrics['mae']:.4f} "
                    f"val_rmse={val_metrics['rmse']:.4f} "
                    f"val_r2={val_metrics['r2']:.4f} "
                    f"best={best_val:.4f}",
                    flush=True,
                )
            if bad_epochs >= args.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        preds: list[float] = []
        with torch.no_grad():
            for batch_idx in make_batches(val_idx, args.batch_size, False, rng):
                batch = collate_graphs(records, batch_idx, scalers, device)
                pred = model(
                    batch["x"],
                    batch["edge_index"],
                    batch["edge_attr"],
                    batch["edge_reverse"],
                    batch["batch"],
                    batch["mol"],
                )
                vals = pred.detach().cpu().numpy() * scalers.y_std + scalers.y_mean
                preds.extend([float(v) for v in vals])

        true = np.asarray([records[int(i)].y for i in val_idx], dtype=np.float64)
        metric = _metrics(true, np.asarray(preds))
        metric.update({"fold": fold_id, "best_val_mae": float(best_val)})
        fold_metrics.append(metric)
        for i, pred in zip(val_idx, preds):
            r = records[int(i)]
            rows.append(
                {
                    "model": "sigma_gnn_torch",
                    "fold": fold_id,
                    "source_index": r.index,
                    "smiles": r.smiles,
                    "chaos_id": r.chaos_id,
                    "y_true": r.y,
                    "y_pred": pred,
                }
            )
        print(
            f"[gnn] fold={fold_id + 1}/{len(folds)} done "
            f"MAE={metric['mae']:.4f} RMSE={metric['rmse']:.4f} R2={metric['r2']:.4f}",
            flush=True,
        )

    pred_df = pd.DataFrame(rows).sort_values(["source_index", "model"]).reset_index(drop=True)
    overall = _metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
    return {
        "model": "sigma_gnn_torch",
        "device": device,
        "raw_node_dim": raw_node_dim,
        "raw_edge_dim": raw_edge_dim,
        "raw_mol_dim": raw_mol_dim,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "mol_dim": mol_dim,
        "n_features": int(node_dim + edge_dim + mol_dim),
        "mol_feature_mode": args.mol_feature_mode,
        "model_type": args.model_type,
        "hidden": int(args.hidden),
        "conv_steps": int(args.layers),
        "head_layers": int(args.head_layers),
        "score_dt": float(args.score_dt),
        "arcsinh_scale": float(args.arcsinh_threshold),
        "folds": fold_metrics,
        "overall": overall,
        "seconds": float(time.time() - t0),
    }, pred_df


def osmordred_cache_path(data_dir: Path, out_dir: Path, n_rows: int, max_mols: int | None) -> Path:
    if max_mols is None:
        return data_dir / "osmordred_features.npz"
    return out_dir / f"osmordred_features_n{n_rows}.npz"


def load_or_compute_osmordred(df: pd.DataFrame, cache_path: Path, force: bool = False) -> tuple[np.ndarray, list[str]]:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    smiles = df["smiles"].astype(str).to_numpy()
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=True)
        cached_smiles = cached["smiles"].astype(str)
        if cached_smiles.shape == smiles.shape and np.all(cached_smiles == smiles):
            return np.asarray(cached["X"], dtype=np.float64), [str(x) for x in cached["names"]]

    names = [str(x) for x in rdMolDescriptors.GetOsmordredDescriptorNames()]
    X = np.full((len(smiles), len(names)), np.nan, dtype=np.float32)
    t0 = time.time()
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        vals = np.asarray(list(rdMolDescriptors.CalcOsmordred(mol)), dtype=np.float64)
        X[i, :] = vals.astype(np.float32)
        if (i + 1) % 250 == 0:
            print(f"[osmordred] computed {i + 1}/{len(smiles)} descriptors", flush=True)
    finite = np.isfinite(X)
    print(
        f"[osmordred] done shape={X.shape} finite={100.0 * finite.mean():.2f}% "
        f"seconds={time.time() - t0:.1f}",
        flush=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, names=np.asarray(names), smiles=smiles)
    return X, names


def run_xgboost_cv(
    df: pd.DataFrame,
    records: list[GraphRecord],
    data_dir: Path,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[list[dict], pd.DataFrame]:
    from xgboost import XGBRegressor

    cache_path = osmordred_cache_path(data_dir, args.out_dir, len(df), args.max_mols)
    X_osmo_raw, osmo_names = load_or_compute_osmordred(df, cache_path, force=args.recompute_osmordred)
    sigma_npz = np.load(data_dir / "sigma_features.npz", allow_pickle=False)
    if args.max_mols is None:
        source_indices = np.asarray([r.index for r in records], dtype=np.int64)
    else:
        source_indices = np.asarray([r.index for r in records], dtype=np.int64)
    X_sigma_all = np.concatenate(
        [
            _as_float_matrix(sigma_npz["mu_J_per_mol"])[source_indices],
            _as_float_matrix(sigma_npz["profile_area_A2"])[source_indices],
        ],
        axis=1,
    )
    sigma_names = [f"sigma_mu_{i:02d}" for i in range(61)] + [f"sigma_profile_{i:02d}" for i in range(61)]
    y = np.asarray([r.y for r in records], dtype=np.float64)
    raw_feature_sets = {
        "xgb_osmordred": (X_osmo_raw, osmo_names),
        "xgb_osmordred_sigma": (
            np.concatenate([X_osmo_raw, X_sigma_all], axis=1),
            osmo_names + sigma_names,
        ),
    }

    results: list[dict] = []
    pred_rows: list[dict] = []
    for model_name, (X_raw, feature_names) in raw_feature_sets.items():
        prep = preprocess_tabular_features(
            X_raw,
            feature_names,
            high_abs_threshold=args.arcsinh_threshold,
        )
        X = prep.x
        prep_path = args.out_dir / f"{model_name}_preprocess.json"
        with prep_path.open("w") as f:
            json.dump(
                {
                    "model": model_name,
                    "n_raw_features": int(np.asarray(X_raw).shape[1]),
                    "n_features": int(X.shape[1]),
                    "n_dropped_nonfinite": prep.n_dropped_nonfinite,
                    "n_arcsinh": prep.n_arcsinh,
                    "arcsinh_threshold": float(args.arcsinh_threshold),
                    "kept_feature_names": prep.names,
                    "arcsinh_feature_names": [
                        name for name, use_arcsinh in zip(prep.names, prep.arcsinh_mask) if bool(use_arcsinh)
                    ],
                },
                f,
                indent=2,
            )
        print(
            f"[{model_name}] preprocess raw={np.asarray(X_raw).shape[1]} kept={X.shape[1]} "
            f"dropped_nonfinite={prep.n_dropped_nonfinite} arcsinh={prep.n_arcsinh}",
            flush=True,
        )
        fold_metrics = []
        t0 = time.time()
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            model = XGBRegressor(
                n_estimators=args.xgb_trees,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample,
                min_child_weight=args.xgb_min_child_weight,
                reg_lambda=args.xgb_reg_lambda,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=args.seed + fold_id,
                n_jobs=args.xgb_jobs,
                verbosity=0,
            )
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx]).astype(np.float64)
            metric = _metrics(y[val_idx], pred)
            metric.update({"fold": fold_id})
            fold_metrics.append(metric)
            for i, p in zip(val_idx, pred):
                r = records[int(i)]
                pred_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_id,
                        "source_index": r.index,
                        "smiles": r.smiles,
                        "chaos_id": r.chaos_id,
                        "y_true": r.y,
                        "y_pred": float(p),
                    }
                )
            print(
                f"[{model_name}] fold={fold_id + 1}/{len(folds)} "
                f"MAE={metric['mae']:.4f} RMSE={metric['rmse']:.4f} R2={metric['r2']:.4f}",
                flush=True,
            )
        pred_model = pd.DataFrame([r for r in pred_rows if r["model"] == model_name])
        overall = _metrics(pred_model["y_true"].to_numpy(), pred_model["y_pred"].to_numpy())
        results.append(
            {
                "model": model_name,
                "n_raw_features": int(np.asarray(X_raw).shape[1]),
                "n_features": int(X.shape[1]),
                "n_dropped_nonfinite": prep.n_dropped_nonfinite,
                "n_arcsinh": prep.n_arcsinh,
                "folds": fold_metrics,
                "overall": overall,
                "seconds": float(time.time() - t0),
            }
        )
    return results, pd.DataFrame(pred_rows)


def calcphyschemprop_baseline(df: pd.DataFrame) -> dict | None:
    if "calcphyschemprop_pred_debye" not in df.columns:
        return None
    y_true = df["dipole_debye"].to_numpy(dtype=np.float64)
    y_pred = df["calcphyschemprop_pred_debye"].to_numpy(dtype=np.float64)
    return {
        "model": "calcphyschemprop_pred_debye",
        "overall": _metrics(y_true, y_pred),
        "n_features": 0,
    }


def make_cv_folds(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    n_splits = min(n_splits, n)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train.astype(np.int64), val.astype(np.int64)) for train, val in splitter.split(np.arange(n))]


def _clone_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def _suggest_gnn_params(trial, args: argparse.Namespace) -> dict:
    hidden = trial.suggest_categorical("hidden", [64, 96, 128, 192, 256, 384])
    return {
        "model_type": "score_dmpnn",
        "hidden": hidden,
        "layers": trial.suggest_int("conv_steps", 2, 8),
        "head_layers": trial.suggest_int("head_layers", 1, 4),
        "score_dt": trial.suggest_float("score_dt", 0.10, 1.00),
        "dropout": trial.suggest_float("dropout", 0.0, 0.20),
        "lr": trial.suggest_float("lr", 3.0e-4, 5.0e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1.0e-7, 3.0e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [96, 128, 256, 512]),
    }


def run_optuna_study(
    records: list[GraphRecord],
    args: argparse.Namespace,
) -> tuple[dict, "object"]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO if args.verbose else optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, args.optuna_warmup_trials)),
    )
    folds = make_cv_folds(len(records), args.optuna_folds, args.seed)

    def objective(trial):
        params = _suggest_gnn_params(trial, args)
        trial_args = _clone_args(
            args,
            **params,
            folds=args.optuna_folds,
            epochs=args.optuna_epochs,
            patience=args.optuna_patience,
            skip_xgb=True,
            compile=args.optuna_compile,
            verbose=False,
        )
        result, _pred = train_gnn_cv(records, folds, trial_args)
        overall = result["overall"]
        trial.set_user_attr("rmse", float(overall["rmse"]))
        trial.set_user_attr("r2", float(overall["r2"]))
        trial.set_user_attr("seconds", float(result["seconds"]))
        trial.report(float(overall["mae"]), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(overall["mae"])

    study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(args.out_dir / "optuna_trials.csv", index=False)
    best = dict(study.best_trial.params)
    best["layers"] = int(best.pop("conv_steps"))
    best["model_type"] = "score_dmpnn"
    best_summary = {
        "best_value_mae": float(study.best_value),
        "best_params": best,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "optuna_epochs": args.optuna_epochs,
        "optuna_folds": args.optuna_folds,
    }
    with (args.out_dir / "optuna_best_params.json").open("w") as f:
        json.dump(best_summary, f, indent=2)
    print(
        f"[optuna] best MAE={study.best_value:.4f} "
        f"RMSE={study.best_trial.user_attrs.get('rmse', float('nan')):.4f} "
        f"R2={study.best_trial.user_attrs.get('r2', float('nan')):.4f} "
        f"params={best}",
        flush=True,
    )
    return best, study


def write_summary(results: list[dict], out_dir: Path) -> pd.DataFrame:
    rows = []
    for r in results:
        overall = r.get("overall", {})
        rows.append(
            {
                "model": r["model"],
                "mae": overall.get("mae", float("nan")),
                "rmse": overall.get("rmse", float("nan")),
                "r2": overall.get("r2", float("nan")),
                "seconds": r.get("seconds", float("nan")),
                "n_features": r.get("n_features", float("nan")),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    summary.to_csv(out_dir / "dipole_sigma_benchmark_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dipole moment benchmark on the CHAOS 3D sigma graph dataset: "
            "pure Torch sigma graph model vs XGBoost OSMORDRED+sigma."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/dipole_physchem_chaos3d_graph"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/dipole_sigma_gnn"))
    parser.add_argument("--max-mols", type=int, default=None, help="Random subset for smoke tests.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads for the pure-Torch GNN. 1 avoids macOS/Accelerate Linear segfaults seen in this env.",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the GNN model.")
    parser.add_argument("--compile-backend", type=str, default="inductor", help="torch.compile backend.")
    parser.add_argument("--compile-mode", type=str, default="default", help="torch.compile mode.")
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Request fullgraph=True in torch.compile.",
    )
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-timeout", type=int, default=None)
    parser.add_argument("--optuna-study-name", type=str, default="dipole_sigma_score_dmpnn")
    parser.add_argument("--optuna-storage", type=str, default=None)
    parser.add_argument("--optuna-folds", type=int, default=2)
    parser.add_argument("--optuna-epochs", type=int, default=10)
    parser.add_argument("--optuna-patience", type=int, default=5)
    parser.add_argument("--optuna-warmup-trials", type=int, default=4)
    parser.add_argument("--optuna-compile", action="store_true")
    parser.add_argument(
        "--optuna-run-best",
        action="store_true",
        help="After tuning, run the normal CV path using the best Optuna parameters.",
    )
    parser.add_argument(
        "--mol-feature-mode",
        choices=["safe", "sigma_only", "all"],
        default="safe",
        help=(
            "Molecular features passed to the graph model before sigma profiles. "
            "'safe' excludes direct dipole/proxy columns; 'all' intentionally allows leakage probes."
        ),
    )
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument(
        "--include-calcphyschemprop-baseline",
        action="store_true",
        help="Report the existing calcphyschemprop dipole prediction column as a teacher/reference baseline.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--model-type", choices=["score_dmpnn", "residual_mpnn"], default="score_dmpnn")
    parser.add_argument("--score-dt", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recompute-osmordred", action="store_true")
    parser.add_argument("--xgb-trees", type=int, default=700)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample", type=float, default=0.75)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    parser.add_argument("--xgb-jobs", type=int, default=-1)
    parser.add_argument(
        "--arcsinh-threshold",
        type=float,
        default=100.0,
        help="Apply arcsinh to cleaned tabular columns with max absolute value above this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, df = load_sigma_graph_dataset(
        args.data_dir,
        max_mols=args.max_mols,
        seed=args.seed,
        mol_feature_mode=args.mol_feature_mode,
    )
    folds = make_cv_folds(len(records), args.folds, args.seed)
    print(
        f"Loaded {len(records)} CHAOS 3D sigma graphs from {args.data_dir} "
        f"(node_dim={records[0].node_features.shape[1]}, "
        f"edge_dim={records[0].edge_features.shape[1]}, mol_dim={records[0].mol_features.shape[0]}, "
        f"mol_feature_mode={args.mol_feature_mode})",
        flush=True,
    )

    if args.optuna_trials > 0:
        best_params, _study = run_optuna_study(records, args)
        if not args.optuna_run_best:
            return
        for key, value in best_params.items():
            setattr(args, key, value)
        args.skip_xgb = True
        print(f"[optuna] running final CV with best params: {best_params}", flush=True)

    results: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    baseline = calcphyschemprop_baseline(df) if args.include_calcphyschemprop_baseline else None
    if baseline is not None:
        results.append(baseline)
        print(
            f"[calcphyschemprop] MAE={baseline['overall']['mae']:.4f} "
            f"RMSE={baseline['overall']['rmse']:.4f} R2={baseline['overall']['r2']:.4f}",
            flush=True,
        )

    if not args.skip_gnn:
        gnn_result, gnn_pred = train_gnn_cv(records, folds, args)
        results.append(gnn_result)
        pred_frames.append(gnn_pred)

    if not args.skip_xgb:
        xgb_results, xgb_pred = run_xgboost_cv(df, records, args.data_dir, folds, args)
        results.extend(xgb_results)
        pred_frames.append(xgb_pred)

    if pred_frames:
        predictions = pd.concat(pred_frames, axis=0, ignore_index=True)
        predictions.to_csv(args.out_dir / "dipole_sigma_benchmark_predictions.csv", index=False)

    with (args.out_dir / "dipole_sigma_benchmark_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    summary = write_summary(results, args.out_dir)
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
