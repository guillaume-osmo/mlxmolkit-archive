#!/usr/bin/env python
"""Refine selected openCHEESE teacher pairs with full conformer-pair overlay.

The fast ensemble teacher scores every conformer pair in a canonical frame. This
script improves the labels that matter most for retrieval by rerunning selected
pairs through the expensive Roshambo-style alignment path and replacing the
teacher entries with the best aligned conformer-pair score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from opencheese.descriptors import CheeseAlignmentConfig, align_cheese_pair
from tools.compute_cheese_ensemble_teacher import EnsembleCache


DEFAULT_TEACHER = Path("outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz")
DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_refined_toppairs.npz")


def selected_pairs(
    score: np.ndarray,
    *,
    top_k_per_row: int,
    min_score: float,
    max_pairs: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen = set()
    n = int(score.shape[0])
    for i in range(n):
        order = np.argsort(score[i])[::-1]
        added = 0
        for j in order:
            j = int(j)
            if i == j or float(score[i, j]) < min_score:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            added += 1
            if added >= top_k_per_row:
                break
            if max_pairs > 0 and len(pairs) >= max_pairs:
                return pairs
    return pairs[:max_pairs] if max_pairs > 0 else pairs


def refine_pair(
    cache: EnsembleCache,
    source_i: int,
    source_j: int,
    *,
    config: CheeseAlignmentConfig,
    max_conformers: int,
) -> tuple[float, float, float]:
    atoms_i, coords_i, charges_i = cache.molecule(int(source_i))
    atoms_j, coords_j, charges_j = cache.molecule(int(source_j))
    if max_conformers > 0:
        coords_i = coords_i[:max_conformers]
        coords_j = coords_j[:max_conformers]

    best_shape = -np.inf
    best_electrostatic = -np.inf
    best_combined = -np.inf
    for conf_i in coords_i:
        for conf_j in coords_j:
            result = align_cheese_pair(
                atoms_i,
                conf_i,
                atoms_j,
                conf_j,
                probe_charges=charges_i,
                reference_charges=charges_j,
                config=config,
            )
            if float(result.combined) > best_combined:
                best_shape = float(result.shape)
                best_electrostatic = float(result.electrostatic)
                best_combined = float(result.combined)
    return best_shape, best_electrostatic, best_combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--select-channel", choices=["shape", "electrostatic", "combined"], default="shape")
    parser.add_argument("--top-k-per-row", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--max-conformers", type=int, default=10)
    parser.add_argument("--refine-top-k", type=int, default=8)
    parser.add_argument("--max-refine-steps", type=int, default=40)
    parser.add_argument("--cocluster-starts", type=int, default=16)
    parser.add_argument("--random-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher = np.load(args.teacher, allow_pickle=False)
    cache = EnsembleCache(args.ensembles)

    source_indices = teacher["source_indices"].astype(np.int64)
    shape = teacher["shape"].astype(np.float32).copy()
    electrostatic = teacher["electrostatic"].astype(np.float32).copy()
    combined = teacher["combined"].astype(np.float32).copy()
    channel = {"shape": shape, "electrostatic": electrostatic, "combined": combined}[args.select_channel]
    pairs = selected_pairs(
        channel,
        top_k_per_row=max(1, args.top_k_per_row),
        min_score=float(args.min_score),
        max_pairs=max(0, args.max_pairs),
    )
    config = CheeseAlignmentConfig(
        start_mode="roshambo",
        refine_top_k=max(1, args.refine_top_k),
        max_refine_steps=max(1, args.max_refine_steps),
        cocluster_starts=max(0, args.cocluster_starts),
        random_starts=max(0, args.random_starts),
        random_seed=int(args.seed),
        electrostatic_metric="carbo",
        optimize=args.select_channel if args.select_channel != "combined" else "shape",
    )

    start_time = time.perf_counter()
    print(f"Refining {len(pairs)} teacher pairs with full conformer-pair overlay", flush=True)
    refined_pair_indices = []
    refined_pair_scores = []
    for pair_index, (i, j) in enumerate(pairs, start=1):
        s, e, c = refine_pair(
            cache,
            int(source_indices[i]),
            int(source_indices[j]),
            config=config,
            max_conformers=args.max_conformers,
        )
        shape[i, j] = shape[j, i] = s
        electrostatic[i, j] = electrostatic[j, i] = e
        combined[i, j] = combined[j, i] = c
        refined_pair_indices.append((i, j))
        refined_pair_scores.append((s, e, c))
        if pair_index == 1 or pair_index == len(pairs) or pair_index % 10 == 0:
            print(f"  refined {pair_index}/{len(pairs)} pair=({i},{j}) shape={s:.4f} combined={c:.4f}", flush=True)

    elapsed = time.perf_counter() - start_time
    metadata = json.loads(str(teacher["metadata_json"][0])) if "metadata_json" in teacher.files else {}
    metadata.update(
        {
            "format": "opencheese.refined_pairwise_teacher",
            "base_teacher": str(args.teacher),
            "refinement": "full_conformer_pair_overlay",
            "select_channel": args.select_channel,
            "top_k_per_row": args.top_k_per_row,
            "min_score": args.min_score,
            "max_pairs": args.max_pairs,
            "max_conformers": args.max_conformers,
            "n_refined_pairs": len(refined_pair_indices),
            "refine_seconds": elapsed,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        format_version=np.array([1], dtype=np.int64),
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)], dtype=str),
        source_indices=source_indices,
        ids=teacher["ids"].astype(str),
        smiles=teacher["smiles"].astype(str) if "smiles" in teacher.files else np.array([], dtype=str),
        shape=shape,
        electrostatic=electrostatic,
        combined=combined,
        refined_pair_indices=np.asarray(refined_pair_indices, dtype=np.int64).reshape((-1, 2)),
        refined_pair_scores=np.asarray(refined_pair_scores, dtype=np.float32).reshape((-1, 3)),
    )
    print(f"Wrote {args.out} in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
