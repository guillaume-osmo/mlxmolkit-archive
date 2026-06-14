#!/usr/bin/env python
"""Analyze openCHEESE teacher reproducibility and cheap descriptor probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from tools.benchmark_opencheese_shape_descriptors import (
    _best_owner_scores,
    load_descriptor_cache,
)
from tools.train_cheese_projection import _retrieval_metrics_np, size_residual_teacher_to_unit


DEFAULT_TEACHER = Path("outputs/cheese_projection/opencheese_old1k_plus_medchem4k_teacher_k10_bestpair_principal_carbo.npz")
DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/opencheese_old1k_plus_medchem4k_ensembles_k10_q_resp_cached.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--channel", choices=["shape", "combined", "electrostatic"], default="shape")
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--compare-teacher", type=Path, action="append", default=[])
    parser.add_argument("--descriptor-benchmark", type=Path, action="append", default=[])
    parser.add_argument("--concat-descriptor-cache", type=Path, nargs=2, default=None)
    parser.add_argument("--embedding", type=Path, action="append", default=[])
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_teacher(path: Path, channel: str) -> dict[str, np.ndarray | dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"][0])) if "metadata_json" in data.files else {}
    return {
        "path": str(path),
        "metadata": metadata,
        "source_indices": data["source_indices"].astype(np.int64),
        "ids": data["ids"].astype(str),
        "smiles": data["smiles"].astype(str) if "smiles" in data.files else np.asarray([], dtype=str),
        "matrix": data[channel].astype(np.float32),
    }


def source_index_positions(source: np.ndarray) -> dict[int, int]:
    return {int(src): i for i, src in enumerate(np.asarray(source, dtype=np.int64))}


def align_matrix_to_source(
    matrix: np.ndarray,
    source: np.ndarray,
    target_source: np.ndarray,
) -> np.ndarray:
    pos = source_index_positions(source)
    indices = np.asarray([pos[int(src)] for src in target_source], dtype=np.int64)
    return matrix[np.ix_(indices, indices)].astype(np.float32, copy=False)


def topk_jaccard(reference: np.ndarray, candidate: np.ndarray, *, k: int) -> float:
    n = int(reference.shape[0])
    if n <= 1:
        return 0.0
    kk = min(int(k), n - 1)
    values = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        population = np.flatnonzero(mask)
        ref_top = set(int(x) for x in population[np.argsort(reference[i, mask])[::-1][:kk]])
        cand_top = set(int(x) for x in population[np.argsort(candidate[i, mask])[::-1][:kk]])
        union = ref_top | cand_top
        values.append(len(ref_top & cand_top) / float(len(union)) if union else 0.0)
    return float(np.mean(values))


def topk_overlap(reference: np.ndarray, candidate: np.ndarray, *, k: int) -> float:
    n = int(reference.shape[0])
    if n <= 1:
        return 0.0
    kk = min(int(k), n - 1)
    values = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        population = np.flatnonzero(mask)
        ref_top = set(int(x) for x in population[np.argsort(reference[i, mask])[::-1][:kk]])
        cand_top = set(int(x) for x in population[np.argsort(candidate[i, mask])[::-1][:kk]])
        values.append(len(ref_top & cand_top) / float(kk))
    return float(np.mean(values))


def matrix_probe_metrics(similarity: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    metrics = _retrieval_metrics_np(1.0 - similarity.astype(np.float32), teacher.astype(np.float32))
    return {key: float(value) for key, value in metrics.items()}


def embedding_probe_metrics(path: Path, teacher_source: np.ndarray, teacher_matrix: np.ndarray) -> dict[str, float]:
    data = np.load(path, allow_pickle=False)
    source = data["source_indices"].astype(np.int64)
    pos = source_index_positions(source)
    indices = np.asarray([pos[int(src)] for src in teacher_source], dtype=np.int64)
    emb = data["embeddings"][indices].astype(np.float32)
    dist = np.sqrt(np.maximum(np.sum((emb[:, None, :] - emb[None, :, :]) ** 2, axis=-1), 0.0)).astype(np.float32)
    return {key: float(value) for key, value in _retrieval_metrics_np(dist, teacher_matrix).items()}


def atom_counts_for_teacher(ensembles_path: Path, teacher_source: np.ndarray) -> np.ndarray:
    data = np.load(ensembles_path, allow_pickle=False)
    source = data["source_indices"].astype(np.int64)
    counts = (data["atom_offsets"][1:] - data["atom_offsets"][:-1]).astype(np.float32)
    pos = source_index_positions(source)
    return np.asarray([counts[pos[int(src)]] for src in teacher_source], dtype=np.float32)


def _concat_score_matrix(row_desc: np.ndarray, col_desc: np.ndarray) -> np.ndarray:
    distance = np.mean(np.abs(row_desc[:, None, :] - col_desc[None, :, :]), axis=2, dtype=np.float32)
    return (1.0 / (1.0 + distance)).astype(np.float32)


def concat_descriptor_similarity(cache_paths: Sequence[Path], *, tile_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    caches = [load_descriptor_cache(path) for path in cache_paths]
    source = np.asarray(caches[0]["source_indices"], dtype=np.int64)
    ids = np.asarray(caches[0]["ids"], dtype=str)
    offsets = np.asarray(caches[0]["descriptor_offsets"], dtype=np.int64)
    for cache in caches[1:]:
        if not np.array_equal(source, np.asarray(cache["source_indices"], dtype=np.int64)):
            raise ValueError("concat descriptor caches must have identical source_indices")
        if not np.array_equal(offsets, np.asarray(cache["descriptor_offsets"], dtype=np.int64)):
            raise ValueError("concat descriptor caches must have identical descriptor_offsets")
    normalized_parts = []
    for cache in caches:
        desc = np.asarray(cache["descriptors"], dtype=np.float32)
        mean = desc.mean(axis=0, keepdims=True)
        std = np.maximum(desc.std(axis=0, keepdims=True), 1.0e-6)
        normalized_parts.append((desc - mean) / std)
    descriptors = np.concatenate(normalized_parts, axis=1).astype(np.float32)

    n = int(len(offsets) - 1)
    out = np.zeros((n, n), dtype=np.float32)
    for row0 in range(0, n, tile_size):
        row1 = min(n, row0 + tile_size)
        row_desc = descriptors[int(offsets[row0]) : int(offsets[row1])]
        row_owner = np.concatenate(
            [np.full((int(offsets[i + 1] - offsets[i]),), i - row0, dtype=np.int32) for i in range(row0, row1)]
        )
        for col0 in range(row0, n, tile_size):
            col1 = min(n, col0 + tile_size)
            col_desc = descriptors[int(offsets[col0]) : int(offsets[col1])]
            col_owner = np.concatenate(
                [np.full((int(offsets[j + 1] - offsets[j]),), j - col0, dtype=np.int32) for j in range(col0, col1)]
            )
            score = _concat_score_matrix(row_desc, col_desc)
            block = _best_owner_scores(score, row_owner, col_owner, row1 - row0, col1 - col0)
            out[row0:row1, col0:col1] = block
            if col0 != row0:
                out[col0:col1, row0:row1] = block.T
    return out, source, ids


def summarize_teacher_pair(reference: np.ndarray, candidate: np.ndarray, top_k: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    metrics = matrix_probe_metrics(candidate, reference)
    out.update({f"retrieval_{k}": v for k, v in metrics.items()})
    for k in top_k:
        out[f"top{k}_jaccard"] = topk_jaccard(reference, candidate, k=int(k))
        out[f"top{k}_overlap"] = topk_overlap(reference, candidate, k=int(k))
    mask = ~np.eye(reference.shape[0], dtype=bool)
    if np.std(reference[mask]) > 1.0e-12 and np.std(candidate[mask]) > 1.0e-12:
        out["pearson_offdiag"] = float(np.corrcoef(reference[mask], candidate[mask])[0, 1])
    else:
        out["pearson_offdiag"] = 0.0
    return out


def main() -> None:
    args = parse_args()
    base = load_teacher(args.teacher, args.channel)
    source = np.asarray(base["source_indices"], dtype=np.int64)
    raw_teacher = np.asarray(base["matrix"], dtype=np.float32)
    atom_counts = atom_counts_for_teacher(args.ensembles, source)
    residual_teacher, residual_metadata = size_residual_teacher_to_unit(raw_teacher, atom_counts)

    report: dict[str, object] = {
        "teacher": str(args.teacher),
        "channel": args.channel,
        "n_molecules": int(len(source)),
        "top_k": [int(k) for k in args.top_k],
        "size_residual_metadata": residual_metadata,
        "compare_teachers": {},
        "descriptor_benchmarks": {},
        "concat_descriptor_probe": {},
        "embeddings": {},
    }

    for path in args.compare_teacher:
        candidate = load_teacher(path, args.channel)
        aligned = align_matrix_to_source(
            np.asarray(candidate["matrix"], dtype=np.float32),
            np.asarray(candidate["source_indices"], dtype=np.int64),
            source,
        )
        report["compare_teachers"][str(path)] = {
            "raw": summarize_teacher_pair(raw_teacher, aligned, args.top_k),
            "size_residual": summarize_teacher_pair(residual_teacher, aligned, args.top_k),
        }

    for path in args.descriptor_benchmark:
        data = np.load(path, allow_pickle=False)
        bench_source = data["source_indices"].astype(np.int64)
        sim = align_matrix_to_source(data["descriptor_similarity"].astype(np.float32), bench_source, source)
        report["descriptor_benchmarks"][str(path)] = {
            "raw": matrix_probe_metrics(sim, raw_teacher),
            "size_residual": matrix_probe_metrics(sim, residual_teacher),
        }

    if args.concat_descriptor_cache is not None:
        sim, concat_source, _ = concat_descriptor_similarity(args.concat_descriptor_cache)
        sim = align_matrix_to_source(sim, concat_source, source)
        report["concat_descriptor_probe"] = {
            "caches": [str(path) for path in args.concat_descriptor_cache],
            "raw": matrix_probe_metrics(sim, raw_teacher),
            "size_residual": matrix_probe_metrics(sim, residual_teacher),
        }

    for path in args.embedding:
        report["embeddings"][str(path)] = {
            "raw": embedding_probe_metrics(path, source, raw_teacher),
            "size_residual": embedding_probe_metrics(path, source, residual_teacher),
        }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"Wrote {args.out}", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
