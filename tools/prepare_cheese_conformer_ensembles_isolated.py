#!/usr/bin/env python
"""Generate openCHEESE conformer ensembles with per-molecule process isolation."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np

from tools.prepare_cheese_conformer_ensembles import (
    DEFAULT_DATASET,
    atom_order_mapping_from_dataset_to_rdkit,
    mol_from_graph_with_conformers,
    mol_from_smiles_with_conformers,
    select_conformer_ensemble,
)
from tools.train_charge_model import ChargeTrainingDataset


DEFAULT_OUT = Path("outputs/cheese_projection/cheese_ensembles_500_k10_diverse_isolated_q_resp.npz")


def _worker(result_queue, payload: dict[str, object]) -> None:
    try:
        dataset = ChargeTrainingDataset(payload["data"])
        dataset_index = int(payload["dataset_index"])
        target = str(payload["target"])
        smiles = str(dataset.smiles[dataset_index])
        z0, _, bond0, total_charge, q = dataset.molecule_arrays(dataset_index, target)
        try:
            mol, conf_ids, energies, converged = mol_from_smiles_with_conformers(
                smiles,
                n_conformers=int(payload["n_candidates"]),
                min_conformers=int(payload["n_conformers"]),
                seed=int(payload["seed"]) + dataset_index,
                prune_rms_thresh=float(payload["prune_rms_thresh"]),
                max_embed_attempts=int(payload["max_embed_attempts"]),
                optimize=bool(payload["optimize"]),
                mmff_variant=str(payload["mmff_variant"]),
                max_opt_iters=int(payload["max_opt_iters"]),
            )
            atom_map = atom_order_mapping_from_dataset_to_rdkit(z0, bond0, mol)
        except Exception:
            mol, conf_ids, energies, converged = mol_from_graph_with_conformers(
                z0,
                bond0,
                formal_charge=int(round(float(total_charge))),
                n_conformers=int(payload["n_candidates"]),
                min_conformers=int(payload["n_conformers"]),
                seed=int(payload["seed"]) + dataset_index,
                prune_rms_thresh=float(payload["prune_rms_thresh"]),
                max_embed_attempts=int(payload["max_embed_attempts"]),
                optimize=bool(payload["optimize"]),
                mmff_variant=str(payload["mmff_variant"]),
                max_opt_iters=int(payload["max_opt_iters"]),
            )
            atom_map = np.arange(len(z0), dtype=np.int32)

        conf_ids, energies, converged, selection_stats = select_conformer_ensemble(
            mol,
            conf_ids,
            energies,
            converged,
            n_keep=int(payload["n_conformers"]),
            selection_mode=str(payload["selection_mode"]),
            rms_thresh=float(payload["selection_rms_thresh"]),
            energy_window=float(payload["selection_energy_window"]),
            atom_mode=str(payload["selection_rmsd_atoms"]),
            fill_to_n=bool(payload["fill_to_n_conformers"]),
        )
        atoms = np.asarray(z0, dtype=np.int32)
        xyz = np.stack(
            [
                np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), dtype=np.float32)[atom_map]
                for conf_id in conf_ids
            ],
            axis=0,
        )
        bonds = np.asarray(bond0, dtype=np.int32)
        rows, cols = np.nonzero(bonds)
        result_queue.put(
            {
                "ok": True,
                "error": "",
                "dataset_index": dataset_index,
                "id": str(dataset.ids[dataset_index]),
                "smiles": smiles,
                "total_charge": float(total_charge),
                "atoms": atoms,
                "coords": xyz.reshape(-1, 3).astype(np.float32),
                "charges": q.astype(np.float32),
                "bond_i": rows.astype(np.int32),
                "bond_j": cols.astype(np.int32),
                "bond_state": bonds[rows, cols].astype(np.int32),
                "conformer_energy": energies.astype(np.float32),
                "conformer_converged": converged.astype(bool),
                "conformer_source_id": np.asarray(conf_ids, dtype=np.int32),
                "selection_stats": selection_stats,
            }
        )
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "dataset_index": int(payload["dataset_index"]),
            }
        )


def _run_isolated(payload: dict[str, object], *, timeout: float, context) -> dict[str, object]:
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(result_queue, payload))
    process.start()
    process.join(None if timeout <= 0 else float(timeout))
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join(2.0)
        return {
            "ok": False,
            "error": f"TimeoutError: process exceeded {timeout:.1f}s",
            "dataset_index": int(payload["dataset_index"]),
        }
    try:
        return result_queue.get(timeout=1.0)
    except queue.Empty:
        return {
            "ok": False,
            "error": f"WorkerError: no result, exitcode={process.exitcode}",
            "dataset_index": int(payload["dataset_index"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", choices=["q_reference", "q_esp", "q_resp"], default="q_resp")
    parser.add_argument("--n-conformers", type=int, default=10)
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--selection-mode", choices=["energy_diverse", "energy", "first"], default="energy_diverse")
    parser.add_argument("--selection-rms-thresh", type=float, default=0.75)
    parser.add_argument("--selection-energy-window", type=float, default=15.0)
    parser.add_argument("--selection-rmsd-atoms", choices=["heavy", "all"], default="heavy")
    parser.add_argument("--fill-to-n-conformers", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--max-embed-attempts", type=int, default=100)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--mmff-variant", default="MMFF94")
    parser.add_argument("--max-opt-iters", type=int, default=100)
    parser.add_argument("--process-timeout", type=float, default=45.0)
    parser.add_argument("--start-method", choices=["spawn", "fork"], default="spawn")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_conformers <= 0:
        raise ValueError("n-conformers must be positive")
    if args.candidate_multiplier < 1:
        raise ValueError("candidate-multiplier must be >= 1")
    if args.max_candidates < args.n_conformers:
        raise ValueError("max-candidates must be >= n-conformers")
    n_candidates = min(
        int(args.max_candidates),
        max(int(args.n_conformers), int(np.ceil(args.n_conformers * float(args.candidate_multiplier)))),
    )

    dataset = ChargeTrainingDataset(args.data)
    source_indices = dataset.finite_label_indices(args.target)
    if args.limit > 0:
        source_indices = source_indices[: int(args.limit)]
    context = mp.get_context(args.start_method)

    atom_offsets = [0]
    conformer_offsets = [0]
    coord_offsets = [0]
    bond_offsets = [0]
    atomic_numbers_parts = []
    coords_parts = []
    charge_parts = []
    bond_i_parts = []
    bond_j_parts = []
    bond_state_parts = []
    conformer_energies = []
    conformer_converged = []
    conformer_source_ids = []
    ok = []
    errors = []
    n_conformers_out = []
    selection_min_pair_rmsd = []
    selection_mean_pair_rmsd = []
    selection_max_relative_energy = []
    selection_n_candidates = []
    selection_n_energy_eligible = []
    manifest_rows = []
    start_time = time.perf_counter()

    print(
        f"Generating {len(source_indices)} isolated x up-to-{args.n_conformers} diverse conformers "
        f"from {n_candidates} candidates each timeout={args.process_timeout}s",
        flush=True,
    )
    base_payload = {
        "data": str(args.data),
        "target": args.target,
        "n_conformers": args.n_conformers,
        "n_candidates": n_candidates,
        "seed": args.seed,
        "prune_rms_thresh": args.prune_rms_thresh,
        "max_embed_attempts": args.max_embed_attempts,
        "optimize": not args.no_optimize,
        "mmff_variant": args.mmff_variant,
        "max_opt_iters": args.max_opt_iters,
        "selection_mode": args.selection_mode,
        "selection_rms_thresh": args.selection_rms_thresh,
        "selection_energy_window": args.selection_energy_window,
        "selection_rmsd_atoms": args.selection_rmsd_atoms,
        "fill_to_n_conformers": bool(args.fill_to_n_conformers),
    }
    for out_index, dataset_index in enumerate(source_indices):
        payload = dict(base_payload)
        payload["dataset_index"] = int(dataset_index)
        result = _run_isolated(payload, timeout=float(args.process_timeout), context=context)
        if bool(result.get("ok")):
            atoms = result["atoms"]
            coords = result["coords"]
            charges = result["charges"]
            bonds_i = result["bond_i"]
            bonds_j = result["bond_j"]
            bonds_state = result["bond_state"]
            energies = result["conformer_energy"]
            converged = result["conformer_converged"]
            source_ids = result["conformer_source_id"]
            stats = result["selection_stats"]
            n_atoms = int(len(atoms))
            n_conf = int(len(source_ids))

            atomic_numbers_parts.append(atoms)
            coords_parts.append(coords)
            charge_parts.append(charges)
            bond_i_parts.append(bonds_i)
            bond_j_parts.append(bonds_j)
            bond_state_parts.append(bonds_state)
            conformer_energies.append(energies)
            conformer_converged.append(converged)
            conformer_source_ids.append(source_ids)
            atom_offsets.append(atom_offsets[-1] + n_atoms)
            conformer_offsets.append(conformer_offsets[-1] + n_conf)
            coord_offsets.append(coord_offsets[-1] + n_conf * n_atoms)
            bond_offsets.append(bond_offsets[-1] + len(bonds_i))
            ok.append(True)
            errors.append("")
            n_conformers_out.append(n_conf)
            selection_min_pair_rmsd.append(float(stats["min_pair_rmsd"]))
            selection_mean_pair_rmsd.append(float(stats["mean_pair_rmsd"]))
            selection_max_relative_energy.append(float(stats["max_selected_relative_energy"]))
            selection_n_candidates.append(int(stats["n_candidates"]))
            selection_n_energy_eligible.append(int(stats["n_energy_eligible"]))
            manifest_rows.append(
                {
                    "row": out_index,
                    "source_index": int(dataset_index),
                    "id": str(result["id"]),
                    "n_atoms": n_atoms,
                    "n_conformers": n_conf,
                    "n_candidates": int(stats["n_candidates"]),
                    "n_energy_eligible": int(stats["n_energy_eligible"]),
                    "selection_min_pair_rmsd": float(stats["min_pair_rmsd"]),
                    "selection_mean_pair_rmsd": float(stats["mean_pair_rmsd"]),
                    "selection_max_relative_energy": float(stats["max_selected_relative_energy"]),
                    "selection_rejected_for_rms": int(stats["rejected_for_rms"]),
                    "total_charge": float(result["total_charge"]),
                    "ok": True,
                    "error": "",
                }
            )
        else:
            ok.append(False)
            errors.append(str(result.get("error", "unknown error")))
            n_conformers_out.append(0)
            selection_min_pair_rmsd.append(np.nan)
            selection_mean_pair_rmsd.append(np.nan)
            selection_max_relative_energy.append(np.nan)
            selection_n_candidates.append(0)
            selection_n_energy_eligible.append(0)
            atom_offsets.append(atom_offsets[-1])
            conformer_offsets.append(conformer_offsets[-1])
            coord_offsets.append(coord_offsets[-1])
            bond_offsets.append(bond_offsets[-1])
            manifest_rows.append(
                {
                    "row": out_index,
                    "source_index": int(dataset_index),
                    "id": str(dataset.ids[int(dataset_index)]),
                    "n_atoms": 0,
                    "n_conformers": 0,
                    "n_candidates": 0,
                    "n_energy_eligible": 0,
                    "selection_min_pair_rmsd": np.nan,
                    "selection_mean_pair_rmsd": np.nan,
                    "selection_max_relative_energy": np.nan,
                    "selection_rejected_for_rms": 0,
                    "total_charge": np.nan,
                    "ok": False,
                    "error": errors[-1],
                }
            )
        if (out_index + 1) % 25 == 0 or out_index + 1 == len(source_indices):
            print(f"  {out_index + 1}/{len(source_indices)} ok={int(np.sum(ok))}", flush=True)

    metadata = {
        "format": "opencheese.conformer_ensembles",
        "format_version": 1,
        "dataset": str(dataset.path),
        "target": args.target,
        "requested_n_conformers": args.n_conformers,
        "candidate_conformers": n_candidates,
        "candidate_multiplier": args.candidate_multiplier,
        "max_candidates": args.max_candidates,
        "selection_mode": args.selection_mode,
        "selection_rms_thresh": args.selection_rms_thresh,
        "selection_energy_window": args.selection_energy_window,
        "selection_rmsd_atoms": args.selection_rmsd_atoms,
        "fill_to_n_conformers": bool(args.fill_to_n_conformers),
        "seed": args.seed,
        "prune_rms_thresh": args.prune_rms_thresh,
        "optimize": not args.no_optimize,
        "mmff_variant": args.mmff_variant,
        "max_opt_iters": args.max_opt_iters,
        "process_timeout": args.process_timeout,
        "start_method": args.start_method,
        "seconds": time.perf_counter() - start_time,
        "n_ok": int(np.sum(ok)),
        "n_failed": int(np.sum(~np.asarray(ok, dtype=bool))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        format_version=np.array([1], dtype=np.int64),
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)], dtype=str),
        source_indices=source_indices.astype(np.int64),
        ids=dataset.ids[source_indices].astype(str),
        smiles=dataset.smiles[source_indices].astype(str),
        ok=np.asarray(ok, dtype=bool),
        errors=np.asarray(errors, dtype=str),
        n_conformers=np.asarray(n_conformers_out, dtype=np.int32),
        selection_min_pair_rmsd=np.asarray(selection_min_pair_rmsd, dtype=np.float32),
        selection_mean_pair_rmsd=np.asarray(selection_mean_pair_rmsd, dtype=np.float32),
        selection_max_relative_energy=np.asarray(selection_max_relative_energy, dtype=np.float32),
        selection_n_candidates=np.asarray(selection_n_candidates, dtype=np.int32),
        selection_n_energy_eligible=np.asarray(selection_n_energy_eligible, dtype=np.int32),
        atom_offsets=np.asarray(atom_offsets, dtype=np.int64),
        conformer_offsets=np.asarray(conformer_offsets, dtype=np.int64),
        coord_offsets=np.asarray(coord_offsets, dtype=np.int64),
        bond_offsets=np.asarray(bond_offsets, dtype=np.int64),
        atomic_numbers=np.concatenate(atomic_numbers_parts) if atomic_numbers_parts else np.empty((0,), dtype=np.int32),
        coords=np.concatenate(coords_parts, axis=0) if coords_parts else np.empty((0, 3), dtype=np.float32),
        charges=np.concatenate(charge_parts) if charge_parts else np.empty((0,), dtype=np.float32),
        bond_i=np.concatenate(bond_i_parts) if bond_i_parts else np.empty((0,), dtype=np.int32),
        bond_j=np.concatenate(bond_j_parts) if bond_j_parts else np.empty((0,), dtype=np.int32),
        bond_state=np.concatenate(bond_state_parts) if bond_state_parts else np.empty((0,), dtype=np.int32),
        conformer_energy=np.concatenate(conformer_energies) if conformer_energies else np.empty((0,), dtype=np.float32),
        conformer_source_id=(
            np.concatenate(conformer_source_ids) if conformer_source_ids else np.empty((0,), dtype=np.int32)
        ),
        conformer_converged=(
            np.concatenate(conformer_converged) if conformer_converged else np.empty((0,), dtype=bool)
        ),
    )
    manifest = args.manifest or args.out.with_suffix(".manifest.csv")
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
