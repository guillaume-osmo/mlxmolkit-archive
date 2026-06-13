#!/usr/bin/env python
"""Generate and cache conformer ensembles for openCHEESE pairwise teachers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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


def mol_from_smiles_with_conformers(
    smiles: str,
    *,
    n_conformers: int,
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
    if len(conf_ids) < int(n_conformers):
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
    if len(conf_ids) < int(n_conformers):
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--max-embed-attempts", type=int, default=1000)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--mmff-variant", default="MMFF94")
    parser.add_argument("--max-opt-iters", type=int, default=300)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_conformers <= 0:
        raise ValueError("n-conformers must be positive")

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
    start_time = time.perf_counter()
    manifest_rows = []

    print(f"Generating {len(source_indices)} x {args.n_conformers} conformer ensemble cache", flush=True)
    for out_index, dataset_index in enumerate(source_indices):
        dataset_index = int(dataset_index)
        smiles = str(dataset.smiles[dataset_index])
        try:
            z0, _, bond0, total_charge, q = dataset.molecule_arrays(dataset_index, args.target)
            try:
                mol, conf_ids, energies, converged = mol_from_smiles_with_conformers(
                    smiles,
                    n_conformers=args.n_conformers,
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
                    n_conformers=args.n_conformers,
                    seed=args.seed + dataset_index,
                    prune_rms_thresh=args.prune_rms_thresh,
                    max_embed_attempts=args.max_embed_attempts,
                    optimize=not args.no_optimize,
                    mmff_variant=args.mmff_variant,
                    max_opt_iters=args.max_opt_iters,
                )
                atom_map = np.arange(len(z0), dtype=np.int32)
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
                    "total_charge": total_charge,
                    "ok": True,
                    "error": "",
                }
            )
        except Exception as exc:
            ok.append(False)
            errors.append(f"{type(exc).__name__}: {exc}")
            n_conformers_out.append(0)
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
                    "total_charge": np.nan,
                    "ok": False,
                    "error": errors[-1],
                }
            )
        if (out_index + 1) % 50 == 0 or out_index + 1 == len(source_indices):
            print(f"  {out_index + 1}/{len(source_indices)}", flush=True)

    metadata = {
        "format": "opencheese.conformer_ensembles",
        "format_version": 1,
        "dataset": str(dataset.path),
        "target": args.target,
        "requested_n_conformers": args.n_conformers,
        "seed": args.seed,
        "prune_rms_thresh": args.prune_rms_thresh,
        "optimize": not args.no_optimize,
        "mmff_variant": args.mmff_variant,
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
