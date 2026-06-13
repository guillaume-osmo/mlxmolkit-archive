#!/usr/bin/env python
"""Generate and cache conformer ensembles for openCHEESE pairwise teachers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import signal
import time

import numpy as np

from mlxmolkit.charge_model import bond_matrix_from_rdkit_mol, bond_state_from_rdkit_bond
from tools.train_charge_model import ChargeTrainingDataset


DEFAULT_DATASET = Path(
    "data/espaloma_charge_zenodo_17308526/recalculated_charges/"
    "test_random1000_both_symmetrized_partial_bcc_fill/"
    "cheese_charge_training_am1bcc_resp.npz"
)
DEFAULT_OUT = Path("outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz")


class MoleculeTimeoutError(TimeoutError):
    """Raised when a single conformer-generation row exceeds its time budget."""


def _raise_molecule_timeout(signum, frame):  # noqa: ARG001
    raise MoleculeTimeoutError("per-molecule conformer generation timeout")


def mol_from_smiles_with_conformers(
    smiles: str,
    *,
    n_conformers: int,
    min_conformers: int,
    seed: int,
    prune_rms_thresh: float,
    max_embed_attempts: int,
    optimize: bool,
    mmff_variant: str,
    max_opt_iters: int,
):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid SMILES")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = float(prune_rms_thresh)
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = int(max_embed_attempts)
    params.useRandomCoords = False
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))
    if len(conf_ids) < int(min_conformers):
        params.pruneRmsThresh = -1.0
        params.useRandomCoords = True
        params.randomSeed = int(seed) + 7919
        mol.RemoveAllConformers()
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))
    if not conf_ids:
        raise ValueError("RDKit could not generate conformers")

    energies = np.full((len(conf_ids),), np.nan, dtype=np.float32)
    converged = np.zeros((len(conf_ids),), dtype=bool)
    if optimize:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol,
                numThreads=0,
                maxIters=int(max_opt_iters),
                mmffVariant=mmff_variant,
            )
        else:
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol,
                numThreads=0,
                maxIters=int(max_opt_iters),
            )
        for i, (status, energy) in enumerate(results[: len(conf_ids)]):
            converged[i] = int(status) == 0
            energies[i] = float(energy)
    return mol, conf_ids, energies, converged


def mol_from_graph_with_conformers(
    atoms: np.ndarray,
    bond_matrix: np.ndarray,
    *,
    formal_charge: int,
    n_conformers: int,
    min_conformers: int,
    seed: int,
    prune_rms_thresh: float,
    max_embed_attempts: int,
    optimize: bool,
    mmff_variant: str,
    max_opt_iters: int,
):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rw = Chem.RWMol()
    heavy_indices = []
    for atom_index, atomic_number in enumerate(np.asarray(atoms, dtype=np.int32)):
        atom = Chem.Atom(int(atomic_number))
        # The total charge is enough for geometry/teacher generation; exact
        # per-atom formal-charge placement is recovered from the original label
        # charges during scoring, and MMFF can still optimize neutral topology.
        if int(atomic_number) != 1:
            heavy_indices.append(atom_index)
        rw.AddAtom(atom)

    bonds = np.asarray(bond_matrix, dtype=np.int32)
    for i in range(bonds.shape[0]):
        for j in range(i + 1, bonds.shape[1]):
            state = int(bonds[i, j])
            if state <= 0:
                continue
            rw.AddBond(i, j, _rdkit_bond_type_from_state(state))
            if state == 4:
                rw.GetAtomWithIdx(i).SetIsAromatic(True)
                rw.GetAtomWithIdx(j).SetIsAromatic(True)
                rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)

    if int(round(formal_charge)) != 0 and heavy_indices:
        # Minimal valence-friendly placeholder so RDKit tracks the molecular
        # charge. This is not used for charge labels, which come from the NPZ.
        rw.GetAtomWithIdx(heavy_indices[0]).SetFormalCharge(int(round(formal_charge)))

    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        catchErrors=True,
    )

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = float(prune_rms_thresh)
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = int(max_embed_attempts)
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))
    if len(conf_ids) < int(min_conformers):
        params.pruneRmsThresh = -1.0
        params.useRandomCoords = True
        params.randomSeed = int(seed) + 7919
        mol.RemoveAllConformers()
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))
    if not conf_ids:
        raise ValueError("RDKit could not generate conformers from graph")

    energies = np.full((len(conf_ids),), np.nan, dtype=np.float32)
    converged = np.zeros((len(conf_ids),), dtype=bool)
    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                results = AllChem.MMFFOptimizeMoleculeConfs(
                    mol,
                    numThreads=0,
                    maxIters=int(max_opt_iters),
                    mmffVariant=mmff_variant,
                )
            else:
                results = AllChem.UFFOptimizeMoleculeConfs(
                    mol,
                    numThreads=0,
                    maxIters=int(max_opt_iters),
                )
            for i, (status, energy) in enumerate(results[: len(conf_ids)]):
                converged[i] = int(status) == 0
                energies[i] = float(energy)
        except Exception:
            pass
    return mol, conf_ids, energies, converged


def heavy_atom_indices(mol) -> np.ndarray:
    indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(indices) < 2:
        indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    return np.asarray(indices, dtype=np.int32)


def conformer_kabsch_rmsd(
    mol,
    conf_a: int,
    conf_b: int,
    atom_indices: np.ndarray,
    *,
    eps: float = 1.0e-8,
) -> float:
    coords_a = np.asarray(mol.GetConformer(int(conf_a)).GetPositions(), dtype=np.float64)[atom_indices]
    coords_b = np.asarray(mol.GetConformer(int(conf_b)).GetPositions(), dtype=np.float64)[atom_indices]
    if len(coords_a) == 0:
        return 0.0
    a = coords_a - coords_a.mean(axis=0, keepdims=True)
    b = coords_b - coords_b.mean(axis=0, keepdims=True)
    covariance = a.T @ b
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    diff = a @ rotation - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)) + eps))


def conformer_pairwise_rmsd(
    mol,
    conf_ids: list[int],
    *,
    atom_mode: str,
) -> np.ndarray:
    if atom_mode == "heavy":
        atom_indices = heavy_atom_indices(mol)
    elif atom_mode == "all":
        atom_indices = np.arange(mol.GetNumAtoms(), dtype=np.int32)
    else:
        raise ValueError("rmsd atom mode must be 'heavy' or 'all'")
    n = len(conf_ids)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = conformer_kabsch_rmsd(mol, int(conf_ids[i]), int(conf_ids[j]), atom_indices)
            out[i, j] = out[j, i] = rmsd
    return out


def _conformer_energy_order(energies: np.ndarray, converged: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies, dtype=np.float64)
    converged = np.asarray(converged, dtype=bool)
    original = np.arange(len(energies), dtype=np.int64)
    finite_energy = np.where(np.isfinite(energies), energies, np.inf)
    convergence_rank = np.where(converged, 0, 1)
    return np.lexsort((original, finite_energy, convergence_rank))


def select_conformer_ensemble(
    mol,
    conf_ids: list[int],
    energies: np.ndarray,
    converged: np.ndarray,
    *,
    n_keep: int,
    selection_mode: str,
    rms_thresh: float,
    energy_window: float,
    atom_mode: str,
    fill_to_n: bool,
) -> tuple[list[int], np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Select low-energy, spatially diverse conformers from an overgenerated pool."""

    conf_ids = [int(conf_id) for conf_id in conf_ids]
    energies = np.asarray(energies, dtype=np.float32)
    converged = np.asarray(converged, dtype=bool)
    if len(conf_ids) == 0:
        raise ValueError("cannot select from an empty conformer pool")
    if len(conf_ids) != len(energies) or len(conf_ids) != len(converged):
        raise ValueError("conf_ids, energies, and converged must have matching lengths")
    n_keep = min(max(1, int(n_keep)), len(conf_ids))

    order = _conformer_energy_order(energies, converged)
    finite = np.isfinite(energies)
    if np.any(finite) and energy_window >= 0:
        min_energy = float(np.min(energies[finite]))
        relative_energy = energies - min_energy
        eligible_mask = finite & (relative_energy <= float(energy_window))
    else:
        min_energy = float(np.min(energies[finite])) if np.any(finite) else float("nan")
        relative_energy = np.full_like(energies, np.nan, dtype=np.float32)
        eligible_mask = np.ones((len(conf_ids),), dtype=bool)
    eligible_order = np.asarray([idx for idx in order if eligible_mask[int(idx)]], dtype=np.int64)
    if eligible_order.size == 0:
        eligible_order = order

    if selection_mode == "first":
        selected_positions = np.arange(n_keep, dtype=np.int64)
    elif selection_mode == "energy":
        selected_positions = order[:n_keep]
    elif selection_mode == "energy_diverse":
        rmsd = conformer_pairwise_rmsd(mol, conf_ids, atom_mode=atom_mode)
        selected: list[int] = []
        rejected_for_rms = 0
        for position in eligible_order:
            position = int(position)
            if not selected:
                selected.append(position)
            else:
                min_rmsd = float(np.min(rmsd[position, selected]))
                if min_rmsd >= float(rms_thresh):
                    selected.append(position)
                else:
                    rejected_for_rms += 1
            if len(selected) >= n_keep:
                break
        if fill_to_n and len(selected) < n_keep:
            selected_set = set(selected)
            for position in order:
                position = int(position)
                if position not in selected_set:
                    selected.append(position)
                    selected_set.add(position)
                if len(selected) >= n_keep:
                    break
        selected_positions = np.asarray(selected, dtype=np.int64)
    else:
        raise ValueError("selection_mode must be 'energy_diverse', 'energy', or 'first'")

    selected_positions = selected_positions[:n_keep]
    selected_conf_ids = [conf_ids[int(position)] for position in selected_positions]
    selected_energies = energies[selected_positions].astype(np.float32)
    selected_converged = converged[selected_positions].astype(bool)
    selected_rel = relative_energy[selected_positions] if len(selected_positions) else np.asarray([], dtype=np.float32)

    if len(selected_conf_ids) > 1:
        selected_rmsd = conformer_pairwise_rmsd(mol, selected_conf_ids, atom_mode=atom_mode)
        pair_values = selected_rmsd[np.triu_indices(len(selected_conf_ids), k=1)]
        min_pair_rmsd = float(np.min(pair_values)) if pair_values.size else 0.0
        mean_pair_rmsd = float(np.mean(pair_values)) if pair_values.size else 0.0
    else:
        min_pair_rmsd = 0.0
        mean_pair_rmsd = 0.0

    stats = {
        "selection_mode": selection_mode,
        "n_candidates": int(len(conf_ids)),
        "n_energy_eligible": int(np.sum(eligible_mask)),
        "n_selected": int(len(selected_conf_ids)),
        "min_energy": min_energy,
        "max_selected_relative_energy": float(np.nanmax(selected_rel)) if selected_rel.size else float("nan"),
        "min_pair_rmsd": min_pair_rmsd,
        "mean_pair_rmsd": mean_pair_rmsd,
        "rms_thresh": float(rms_thresh),
        "energy_window": float(energy_window),
        "fill_to_n": int(bool(fill_to_n)),
        "rejected_for_rms": int(rejected_for_rms) if selection_mode == "energy_diverse" else 0,
    }
    return selected_conf_ids, selected_energies, selected_converged, stats


def _rdkit_bond_type_from_state(state: int):
    from rdkit import Chem

    if state == 1:
        return Chem.BondType.SINGLE
    if state == 2:
        return Chem.BondType.DOUBLE
    if state == 3:
        return Chem.BondType.TRIPLE
    if state == 4:
        return Chem.BondType.AROMATIC
    if state == 5:
        return Chem.BondType.DATIVE
    return Chem.BondType.SINGLE


def atom_order_mapping_from_dataset_to_rdkit(
    dataset_atoms: np.ndarray,
    dataset_bonds: np.ndarray,
    mol,
) -> np.ndarray:
    """Return RDKit atom indices ordered like the dataset graph."""

    import networkx as nx

    def coarse_state(state: int) -> int:
        if state in {1, 2, 3}:
            return int(state)
        if state == 4:
            return 4
        return 1

    g_dataset = nx.Graph()
    for i, atomic_number in enumerate(np.asarray(dataset_atoms, dtype=np.int32)):
        g_dataset.add_node(i, z=int(atomic_number))
    bonds = np.asarray(dataset_bonds, dtype=np.int32)
    for i in range(bonds.shape[0]):
        for j in range(i + 1, bonds.shape[1]):
            state = int(bonds[i, j])
            if state > 0:
                g_dataset.add_edge(i, j, state=coarse_state(state))

    g_rdkit = nx.Graph()
    for atom in mol.GetAtoms():
        g_rdkit.add_node(atom.GetIdx(), z=int(atom.GetAtomicNum()))
    for bond in mol.GetBonds():
        g_rdkit.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            state=coarse_state(bond_state_from_rdkit_bond(bond)),
        )

    def node_match(left, right):
        return int(left["z"]) == int(right["z"])

    def edge_match(left, right):
        a = int(left["state"])
        b = int(right["state"])
        return a == b or a == 4 or b == 4

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        g_dataset,
        g_rdkit,
        node_match=node_match,
        edge_match=edge_match,
    )
    try:
        mapping = next(matcher.isomorphisms_iter())
    except StopIteration as exc:
        raise ValueError("could not map parsed RDKit atom order to charge dataset graph") from exc
    return np.asarray([mapping[i] for i in range(len(dataset_atoms))], dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", choices=["q_reference", "q_esp", "q_resp"], default="q_resp")
    parser.add_argument("--n-conformers", type=int, default=10)
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=4.0,
        help="Generate this many candidates per requested conformer before diversity selection.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="Cap the overgenerated conformer candidate pool.",
    )
    parser.add_argument("--selection-mode", choices=["energy_diverse", "energy", "first"], default="energy_diverse")
    parser.add_argument(
        "--selection-rms-thresh",
        type=float,
        default=0.75,
        help="Post-MMFF aligned RMSD threshold for accepting another conformer.",
    )
    parser.add_argument(
        "--selection-energy-window",
        type=float,
        default=15.0,
        help="Only select conformers within this kcal/mol window when energies are available; negative disables.",
    )
    parser.add_argument("--selection-rmsd-atoms", choices=["heavy", "all"], default="heavy")
    parser.add_argument(
        "--fill-to-n-conformers",
        action="store_true",
        help="Fill with lower-diversity conformers if strict RMSD selection finds fewer than n.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--max-embed-attempts", type=int, default=1000)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--mmff-variant", default="MMFF94")
    parser.add_argument("--max-opt-iters", type=int, default=300)
    parser.add_argument(
        "--per-molecule-timeout",
        type=int,
        default=0,
        help="Seconds before skipping one molecule; 0 disables the timeout.",
    )
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

    atom_offsets = [0]
    conformer_offsets = [0]
    coord_offsets = [0]
    atomic_numbers_parts = []
    coords_parts = []
    charge_parts = []
    bond_i_parts = []
    bond_j_parts = []
    bond_state_parts = []
    bond_offsets = [0]
    conformer_energies = []
    conformer_converged = []
    ok = []
    errors = []
    n_conformers_out = []
    selection_min_pair_rmsd = []
    selection_mean_pair_rmsd = []
    selection_max_relative_energy = []
    selection_n_candidates = []
    selection_n_energy_eligible = []
    conformer_source_ids = []
    start_time = time.perf_counter()
    manifest_rows = []

    print(
        f"Generating {len(source_indices)} x up-to-{args.n_conformers} diverse conformer ensemble cache "
        f"from {n_candidates} candidates each",
        flush=True,
    )
    for out_index, dataset_index in enumerate(source_indices):
        dataset_index = int(dataset_index)
        smiles = str(dataset.smiles[dataset_index])
        if args.per_molecule_timeout > 0:
            old_alarm_handler = signal.signal(signal.SIGALRM, _raise_molecule_timeout)
            signal.alarm(int(args.per_molecule_timeout))
        else:
            old_alarm_handler = None
        try:
            z0, _, bond0, total_charge, q = dataset.molecule_arrays(dataset_index, args.target)
            try:
                mol, conf_ids, energies, converged = mol_from_smiles_with_conformers(
                    smiles,
                    n_conformers=n_candidates,
                    min_conformers=args.n_conformers,
                    seed=args.seed + dataset_index,
                    prune_rms_thresh=args.prune_rms_thresh,
                    max_embed_attempts=args.max_embed_attempts,
                    optimize=not args.no_optimize,
                    mmff_variant=args.mmff_variant,
                    max_opt_iters=args.max_opt_iters,
                )
                atom_map = atom_order_mapping_from_dataset_to_rdkit(z0, bond0, mol)
            except Exception:
                mol, conf_ids, energies, converged = mol_from_graph_with_conformers(
                    z0,
                    bond0,
                    formal_charge=int(round(total_charge)),
                    n_conformers=n_candidates,
                    min_conformers=args.n_conformers,
                    seed=args.seed + dataset_index,
                    prune_rms_thresh=args.prune_rms_thresh,
                    max_embed_attempts=args.max_embed_attempts,
                    optimize=not args.no_optimize,
                    mmff_variant=args.mmff_variant,
                    max_opt_iters=args.max_opt_iters,
                )
                atom_map = np.arange(len(z0), dtype=np.int32)
            conf_ids, energies, converged, selection_stats = select_conformer_ensemble(
                mol,
                conf_ids,
                energies,
                converged,
                n_keep=args.n_conformers,
                selection_mode=args.selection_mode,
                rms_thresh=args.selection_rms_thresh,
                energy_window=args.selection_energy_window,
                atom_mode=args.selection_rmsd_atoms,
                fill_to_n=bool(args.fill_to_n_conformers),
            )
            atoms = np.asarray(z0, dtype=np.int32)
            n_atoms = len(atoms)
            xyz = np.stack(
                [
                    np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), dtype=np.float32)[atom_map]
                    for conf_id in conf_ids
                ],
                axis=0,
            )
            bonds = np.asarray(bond0, dtype=np.int32)
            rows, cols = np.nonzero(bonds)

            atomic_numbers_parts.append(atoms)
            coords_parts.append(xyz.reshape(-1, 3).astype(np.float32))
            charge_parts.append(q.astype(np.float32))
            bond_i_parts.append(rows.astype(np.int32))
            bond_j_parts.append(cols.astype(np.int32))
            bond_state_parts.append(bonds[rows, cols].astype(np.int32))
            conformer_energies.append(energies.astype(np.float32))
            conformer_converged.append(converged.astype(bool))
            conformer_source_ids.append(np.asarray(conf_ids, dtype=np.int32))
            selection_min_pair_rmsd.append(float(selection_stats["min_pair_rmsd"]))
            selection_mean_pair_rmsd.append(float(selection_stats["mean_pair_rmsd"]))
            selection_max_relative_energy.append(float(selection_stats["max_selected_relative_energy"]))
            selection_n_candidates.append(int(selection_stats["n_candidates"]))
            selection_n_energy_eligible.append(int(selection_stats["n_energy_eligible"]))
            atom_offsets.append(atom_offsets[-1] + n_atoms)
            conformer_offsets.append(conformer_offsets[-1] + len(conf_ids))
            coord_offsets.append(coord_offsets[-1] + len(conf_ids) * n_atoms)
            bond_offsets.append(bond_offsets[-1] + len(rows))
            ok.append(True)
            errors.append("")
            n_conformers_out.append(len(conf_ids))
            manifest_rows.append(
                {
                    "row": out_index,
                    "source_index": dataset_index,
                    "id": str(dataset.ids[dataset_index]),
                    "n_atoms": n_atoms,
                    "n_conformers": len(conf_ids),
                    "n_candidates": int(selection_stats["n_candidates"]),
                    "n_energy_eligible": int(selection_stats["n_energy_eligible"]),
                    "selection_min_pair_rmsd": float(selection_stats["min_pair_rmsd"]),
                    "selection_mean_pair_rmsd": float(selection_stats["mean_pair_rmsd"]),
                    "selection_max_relative_energy": float(selection_stats["max_selected_relative_energy"]),
                    "selection_rejected_for_rms": int(selection_stats["rejected_for_rms"]),
                    "total_charge": total_charge,
                    "ok": True,
                    "error": "",
                }
            )
        except Exception as exc:
            ok.append(False)
            errors.append(f"{type(exc).__name__}: {exc}")
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
                    "source_index": dataset_index,
                    "id": str(dataset.ids[dataset_index]),
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
        finally:
            if args.per_molecule_timeout > 0:
                signal.signal(signal.SIGALRM, signal.SIG_IGN)
                signal.alarm(0)
                if old_alarm_handler is not None:
                    signal.signal(signal.SIGALRM, old_alarm_handler)
        if (out_index + 1) % 50 == 0 or out_index + 1 == len(source_indices):
            print(f"  {out_index + 1}/{len(source_indices)}", flush=True)

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
        "per_molecule_timeout": args.per_molecule_timeout,
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
