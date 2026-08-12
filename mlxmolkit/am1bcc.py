from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

import mlx.core as mx

from mlxmolkit.nddo import nddo_energy, nddo_energy_batch
from mlxmolkit.nddo.methods import METHOD_PARAMS


@dataclass(frozen=True)
class BCCParameter:
    """A single AM1-BCC bond charge correction."""

    smirks: str
    value: float
    code: str | None = None


@dataclass(frozen=True)
class AM1BCCResult:
    """AM1-BCC charge assignment result."""

    charges: np.ndarray
    am1_charges: np.ndarray
    bcc_corrections: np.ndarray
    total_charge: float
    n_bonds_assigned: int
    metadata: dict[str, Any]


def load_original_am1_bcc_parameters(path: str | Path | None = None) -> list[BCCParameter]:
    """Load the original AM1-BCC correction table ported by OpenFF Recharge."""

    if path is None:
        path = resources.files("mlxmolkit").joinpath("data/bcc/original-am1-bcc.json")

    with open(path) as file:
        raw_parameters = json.load(file)

    return [
        BCCParameter(
            smirks=entry["smirks"],
            value=float(entry["value"]),
            code=(entry.get("provenance") or {}).get("code"),
        )
        for entry in raw_parameters
    ]


def _compile_bcc_parameters(parameters: Sequence[BCCParameter]):
    from rdkit import Chem

    compiled = []
    for index, parameter in enumerate(parameters):
        query = Chem.MolFromSmarts(parameter.smirks)
        if query is None:
            raise ValueError(f"could not parse AM1-BCC SMIRKS: {parameter.smirks}")

        mapped_atoms = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in query.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }
        if set(mapped_atoms) != {1, 2}:
            raise ValueError(f"AM1-BCC SMIRKS must contain map indices 1 and 2: {parameter.smirks}")

        compiled.append((index, parameter, query, mapped_atoms[1], mapped_atoms[2]))
    return compiled


def _bond_key(atom_i: int, atom_j: int) -> tuple[int, int]:
    return tuple(sorted((int(atom_i), int(atom_j))))


def _copy_mol_for_matching(mol: Any, *, set_aromaticity: bool = False, kekulize: bool = False):
    from rdkit import Chem

    match_mol = Chem.Mol(mol)
    sanitize_ops = Chem.SANITIZE_ALL
    if not set_aromaticity:
        sanitize_ops ^= Chem.SANITIZE_SETAROMATICITY
    Chem.SanitizeMol(match_mol, sanitize_ops)
    if kekulize:
        Chem.Kekulize(match_mol)
    return match_mol


def _match_smirks_rdkit(
    smirks: str,
    mol: Any,
    is_atom_aromatic: dict[int, bool],
    is_bond_aromatic: dict[tuple[int, int], bool],
    *,
    unique: bool,
    kekulize: bool = False,
) -> list[dict[int, int]]:
    from rdkit import Chem

    match_mol = _copy_mol_for_matching(mol, kekulize=kekulize)
    atoms = {atom.GetIdx(): atom for atom in match_mol.GetAtoms()}
    bonds = {_bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): bond for bond in match_mol.GetBonds()}

    for atom_index, is_aromatic in is_atom_aromatic.items():
        atoms[atom_index].SetIsAromatic(bool(is_aromatic))
    for (atom_i, atom_j), is_aromatic in is_bond_aromatic.items():
        bonds[_bond_key(atom_i, atom_j)].SetIsAromatic(bool(is_aromatic))

    query = Chem.MolFromSmarts(smirks)
    if query is None:
        raise ValueError(f"could not parse AM1-BCC SMIRKS: {smirks}")

    matches = match_mol.GetSubstructMatches(
        query,
        uniquify=unique,
        maxMatches=np.iinfo(np.uintc).max,
        useChirality=True,
    )

    mapped_matches: list[dict[int, int]] = []
    for match in matches:
        mapped_matches.append(
            {
                query_atom.GetAtomMapNum() - 1: int(atom_index)
                for atom_index, query_atom in zip(match, query.GetAtoms())
                if query_atom.GetAtomMapNum() != 0
            }
        )
    return mapped_matches


def _find_ring_bonds_rdkit(mol: Any) -> dict[tuple[int, int], bool]:
    match_mol = _copy_mol_for_matching(mol)
    return {
        _bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): bool(bond.IsInRing())
        for bond in match_mol.GetBonds()
    }


def _set_aromatic_matches(
    ring_matches: Sequence[dict[int, int]],
    is_bond_in_ring: dict[tuple[int, int], bool],
    is_atom_aromatic: dict[int, bool],
    is_bond_aromatic: dict[tuple[int, int], bool],
) -> None:
    for ring_match in ring_matches:
        ring_atom_indices = set(ring_match.values())

        for atom_index in ring_atom_indices:
            is_atom_aromatic[atom_index] = True

        for atom_i, atom_j in list(is_bond_aromatic):
            if atom_i not in ring_atom_indices or atom_j not in ring_atom_indices:
                continue
            if not is_bond_in_ring[_bond_key(atom_i, atom_j)]:
                continue
            is_bond_aromatic[(atom_i, atom_j)] = True


def _am1bcc_aromaticity_flags_rdkit(mol: Any) -> tuple[dict[int, bool], dict[tuple[int, int], bool]]:
    """Assign the original AM1-BCC aromaticity model using RDKit matching."""

    match_mol = _copy_mol_for_matching(mol)
    is_atom_aromatic = {atom.GetIdx(): False for atom in match_mol.GetAtoms()}
    is_bond_aromatic = {
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): False
        for bond in match_mol.GetBonds()
    }
    is_bond_in_ring = _find_ring_bonds_rdkit(match_mol)

    x_type = "[#6X3,#7X2,#15X2,#7X3+1,#15X3+1,#8X2+1,#16X2+1:N]"
    y_type = "[#6X2-1,#7X2-1,#8X2,#16X2,#7X3,#15X3:N]"
    z_type = x_type

    case_1_smirks = (
        f"{x_type.replace('N', '1')}1"
        f"=@{x_type.replace('N', '2')}"
        f"-@{x_type.replace('N', '3')}"
        f"=@{x_type.replace('N', '4')}"
        f"-@{x_type.replace('N', '5')}"
        f"=@{x_type.replace('N', '6')}-@1"
    )
    case_1_matches = _match_smirks_rdkit(
        case_1_smirks,
        match_mol,
        is_atom_aromatic,
        is_bond_aromatic,
        unique=True,
        kekulize=True,
    )
    case_1_atoms = {atom for match in case_1_matches for atom in match.values()}
    _set_aromatic_matches(case_1_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic)

    ar6_assignments = set(case_1_atoms)

    case_2_smirks = (
        f"{x_type.replace('N', '1')}1"
        f"=@{x_type.replace('N', '2')}"
        f"-@{x_type.replace('N', '3')}"
        f"=@{x_type.replace('N', '4')}"
        f"-@{x_type.replace('N', '5')}"
        f"-,:@{x_type.replace('N', '6')}-@1"
    )
    previous_case_2_atoms = None
    case_2_atoms: set[int] = set()
    while previous_case_2_atoms != case_2_atoms:
        case_2_matches = _match_smirks_rdkit(
            case_2_smirks,
            match_mol,
            is_atom_aromatic,
            is_bond_aromatic,
            unique=True,
            kekulize=True,
        )
        case_2_matches = [
            match
            for match in case_2_matches
            if match[4] in ar6_assignments and match[5] in ar6_assignments
        ]
        previous_case_2_atoms = case_2_atoms
        case_2_atoms = {atom for match in case_2_matches for atom in match.values()}
        ar6_assignments.update(case_2_atoms)
        _set_aromatic_matches(case_2_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic)

    case_3_smirks = (
        f"{x_type.replace('N', '1')}1"
        f"=@{x_type.replace('N', '2')}"
        f"-@{x_type.replace('N', '3')}"
        f"-,:@{x_type.replace('N', '4')}"
        f"~@{x_type.replace('N', '5')}"
        f"-,:@{x_type.replace('N', '6')}-@1"
    )
    previous_case_3_atoms = None
    case_3_atoms: set[int] = set()
    while previous_case_3_atoms != case_3_atoms:
        case_3_matches = _match_smirks_rdkit(
            case_3_smirks,
            match_mol,
            is_atom_aromatic,
            is_bond_aromatic,
            unique=True,
            kekulize=True,
        )
        case_3_matches = [
            match
            for match in case_3_matches
            if match[2] in ar6_assignments
            and match[3] in ar6_assignments
            and match[4] in ar6_assignments
            and match[5] in ar6_assignments
        ]
        previous_case_3_atoms = case_3_atoms
        case_3_atoms = {atom for match in case_3_matches for atom in match.values()}
        ar6_assignments.update(case_3_atoms)
        _set_aromatic_matches(case_3_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic)

    case_4_smirks = (
        "[#6+1:1]1"
        f"-@{x_type.replace('N', '2')}"
        f"=@{x_type.replace('N', '3')}"
        f"-@{x_type.replace('N', '4')}"
        f"=@{x_type.replace('N', '5')}"
        f"-@{x_type.replace('N', '6')}"
        f"=@{x_type.replace('N', '7')}-@1"
    )
    case_4_matches = _match_smirks_rdkit(
        case_4_smirks,
        match_mol,
        is_atom_aromatic,
        is_bond_aromatic,
        unique=True,
        kekulize=True,
    )
    case_4_atoms = {atom for match in case_4_matches for atom in match.values()}
    _set_aromatic_matches(case_4_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic)

    case_5_smirks = (
        f"{y_type.replace('N', '1')}1"
        f"-@{z_type.replace('N', '2')}"
        f"=@{z_type.replace('N', '3')}"
        f"-@{x_type.replace('N', '4')}"
        f"=@{x_type.replace('N', '5')}-@1"
    )
    ar_6_ar_7_matches = {*case_1_atoms, *case_2_atoms, *case_3_atoms, *case_4_atoms}
    case_5_matches = _match_smirks_rdkit(
        case_5_smirks,
        match_mol,
        is_atom_aromatic,
        is_bond_aromatic,
        unique=True,
        kekulize=True,
    )
    case_5_matches = [
        match
        for match in case_5_matches
        if match[1] not in ar_6_ar_7_matches and match[2] not in ar_6_ar_7_matches
    ]
    _set_aromatic_matches(case_5_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic)

    return is_atom_aromatic, is_bond_aromatic


def bcc_corrections_from_rdkit_mol(
    mol: Any,
    parameters: Sequence[BCCParameter] | None = None,
    *,
    validate: bool = True,
) -> np.ndarray:
    """Compute AM1-BCC charge increments for an explicit-H RDKit molecule.

    This is a lightweight RDKit SMARTS matcher over OpenFF Recharge's
    hand-converted original AM1-BCC parameter table. It preserves total charge
    by applying ``+value`` to mapped atom ``:1`` and ``-value`` to mapped atom
    ``:2`` for each assigned bond.
    """

    parameters = list(parameters or load_original_am1_bcc_parameters())
    compiled_parameters = _compile_bcc_parameters(parameters)
    match_mol = _copy_mol_for_matching(mol)
    is_atom_aromatic, is_bond_aromatic = _am1bcc_aromaticity_flags_rdkit(match_mol)

    n_atoms = match_mol.GetNumAtoms()
    corrections = np.zeros(n_atoms, dtype=np.float64)
    matched_bonds: set[tuple[int, int]] = set()
    assignments_per_bond: dict[tuple[int, int], BCCParameter] = {}

    for _, parameter, _query, _map_1_idx, _map_2_idx in compiled_parameters:
        for match in _match_smirks_rdkit(
            parameter.smirks,
            match_mol,
            is_atom_aromatic,
            is_bond_aromatic,
            unique=False,
        ):
            atom_i = match[0]
            atom_j = match[1]

            bond = match_mol.GetBondBetweenAtoms(atom_i, atom_j)
            if bond is None:
                continue

            bond_key = _bond_key(atom_i, atom_j)
            if bond_key in matched_bonds:
                continue

            corrections[atom_i] += parameter.value
            corrections[atom_j] -= parameter.value
            matched_bonds.add(bond_key)
            assignments_per_bond[bond_key] = parameter

    if not np.isclose(corrections.sum(), 0.0, atol=1.0e-10):
        raise ValueError("AM1-BCC corrections changed the total molecular charge")

    if validate:
        all_bonds = {
            _bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in match_mol.GetBonds()
        }
        missing_bonds = sorted(all_bonds - matched_bonds)
        if missing_bonds:
            missing_text = ", ".join(f"{i}-{j}" for i, j in missing_bonds[:12])
            suffix = "" if len(missing_bonds) <= 12 else f", ... ({len(missing_bonds)} total)"
            raise ValueError(f"AM1-BCC parameters did not cover bonds: {missing_text}{suffix}")

    return corrections


def apply_bcc_corrections_mlx(am1_charges: Any, bcc_corrections: Any, total_charge: float | None = None) -> mx.array:
    """Apply BCC increments to AM1 charges with an optional sum projection."""

    charges = mx.array(am1_charges, dtype=mx.float32) + mx.array(bcc_corrections, dtype=mx.float32)
    if total_charge is not None:
        charge_error = (mx.sum(charges) - mx.array(float(total_charge), dtype=mx.float32)) / charges.shape[0]
        charges = charges - charge_error
    return charges


def charge_symmetry_classes_from_rdkit_mol(
    mol: Any,
    *,
    resonance: bool = True,
    max_resonance_forms: int = 128,
) -> list[list[int]]:
    """Return atom-index groups that should share fixed force-field charges."""

    from rdkit import Chem

    n_atoms = mol.GetNumAtoms()
    rank_signatures: list[list[int]] = [[] for _ in range(n_atoms)]

    if resonance:
        forms = Chem.ResonanceMolSupplier(mol)
    else:
        forms = [mol]

    n_forms = 0
    for form in forms:
        ranks = list(Chem.CanonicalRankAtoms(form, breakTies=False))
        if len(ranks) != n_atoms:
            continue
        for atom_index, rank in enumerate(ranks):
            rank_signatures[atom_index].append(int(rank))
        n_forms += 1
        if n_forms >= max_resonance_forms:
            break

    if n_forms == 0:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        for atom_index, rank in enumerate(ranks):
            rank_signatures[atom_index].append(int(rank))

    keyed_atoms: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for atom_index, atom in enumerate(mol.GetAtoms()):
        key = (int(atom.GetAtomicNum()), tuple(sorted(rank_signatures[atom_index])))
        keyed_atoms.setdefault(key, []).append(atom_index)

    return [atoms for atoms in keyed_atoms.values() if len(atoms) > 1]


def symmetrize_charges_by_topology(
    mol: Any,
    charges: Any,
    *,
    total_charge: float | None = None,
    resonance: bool = True,
    max_resonance_forms: int = 128,
) -> tuple[np.ndarray, list[list[int]]]:
    """Average charges over topologically/resonance-equivalent atoms."""

    out = np.asarray(charges, dtype=np.float64).copy()
    if out.shape != (mol.GetNumAtoms(),):
        raise ValueError(f"charges must have shape ({mol.GetNumAtoms()},), got {out.shape}")

    classes = charge_symmetry_classes_from_rdkit_mol(
        mol,
        resonance=resonance,
        max_resonance_forms=max_resonance_forms,
    )
    for atom_indices in classes:
        out[atom_indices] = float(np.mean(out[atom_indices]))

    if total_charge is not None:
        out -= (float(np.sum(out)) - float(total_charge)) / out.shape[0]
    return out, classes


def _mol_with_3d_coordinates(mol: Any, add_hs: bool = True, random_seed: int = 0xA11BCC):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    work_mol = Chem.Mol(mol)
    if add_hs:
        work_mol = Chem.AddHs(work_mol, addCoords=True)

    needs_embedding = work_mol.GetNumConformers() == 0
    if not needs_embedding:
        conf = work_mol.GetConformer()
        needs_embedding = not conf.Is3D()

    if needs_embedding:
        work_mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)
        status = AllChem.EmbedMolecule(work_mol, params)
        if status != 0:
            status = AllChem.EmbedMolecule(work_mol, randomSeed=int(random_seed), useRandomCoords=True)
        if status != 0:
            stereo_relaxed = Chem.Mol(work_mol)
            Chem.RemoveStereochemistry(stereo_relaxed)
            stereo_relaxed.RemoveAllConformers()
            params = AllChem.ETKDGv3()
            params.randomSeed = int(random_seed)
            params.useRandomCoords = True
            params.ignoreSmoothingFailures = True
            params.maxIterations = 500
            status = AllChem.EmbedMolecule(stereo_relaxed, params)
            if status == 0:
                stereo_relaxed.SetBoolProp("_mlxmolkit_embedding_stereo_stripped", True)
                work_mol = stereo_relaxed
        if status != 0:
            raise ValueError("could not generate a 3D conformer for AM1-BCC charges")
        try:
            AllChem.MMFFOptimizeMolecule(work_mol, maxIters=200)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(work_mol, maxIters=200)
            except Exception:
                pass

    return work_mol


def _atoms_and_coords_from_rdkit_mol(mol: Any, conf_id: int = -1) -> tuple[list[int], np.ndarray]:
    conf = mol.GetConformer(int(conf_id))
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords = np.asarray(conf.GetPositions(), dtype=np.float64)
    return atoms, coords


def _mulliken_charges_from_density(
    atoms: Sequence[int],
    density: Any,
    am1_method: str,
) -> np.ndarray:
    """Recover atom-centered Mulliken charges from an NDDO density matrix."""

    params = METHOD_PARAMS[am1_method]
    density = np.asarray(density, dtype=np.float64)
    charges = np.zeros(len(atoms), dtype=np.float64)
    basis_index = 0
    for atom_index, atomic_number in enumerate(atoms):
        atom_params = params[int(atomic_number)]
        population = 0.0
        for _ in range(atom_params.n_basis):
            population += density[basis_index, basis_index]
            basis_index += 1
        charges[atom_index] = atom_params.n_valence - population
    return charges


def am1_bcc_charges_from_rdkit_mol(
    mol: Any,
    *,
    total_charge: float | None = None,
    parameters: Sequence[BCCParameter] | None = None,
    am1_method: str = "AM1",
    add_hs: bool = True,
    conf_id: int = -1,
    random_seed: int = 0xA11BCC,
    validate_bcc_coverage: bool = True,
    project_total_charge: bool = True,
    require_scf_convergence: bool = True,
    symmetrize: bool = True,
    symmetrize_resonance: bool = True,
) -> AM1BCCResult:
    """Assign local AM1-BCC charges using MLXMolKit AM1 plus BCC increments."""

    from rdkit import Chem

    work_mol = _mol_with_3d_coordinates(mol, add_hs=add_hs, random_seed=random_seed)
    if total_charge is None:
        total_charge = float(Chem.GetFormalCharge(work_mol))

    atoms, coords = _atoms_and_coords_from_rdkit_mol(work_mol, conf_id=conf_id)
    supported_elements = set(METHOD_PARAMS[am1_method])
    unsupported_elements = sorted(set(atoms) - supported_elements)
    if unsupported_elements:
        raise ValueError(
            f"{am1_method} parameters are unavailable for atomic numbers "
            f"{unsupported_elements}; current exact AM1-BCC baseline supports "
            "molecules whose AM1 semiempirical parameters are present locally"
        )

    am1 = nddo_energy(atoms, coords, method=am1_method, molecular_charge=total_charge)
    if require_scf_convergence and not bool(am1.get("converged", False)):
        raise ValueError(
            f"{am1_method} SCF did not converge after "
            f"{int(am1.get('n_iter', -1))} iterations"
        )
    am1_charges = np.asarray(am1["charges"], dtype=np.float64)
    bcc_corrections = bcc_corrections_from_rdkit_mol(
        work_mol,
        parameters=parameters,
        validate=validate_bcc_coverage,
    )
    charges_mx = apply_bcc_corrections_mlx(
        am1_charges,
        bcc_corrections,
        total_charge=total_charge if project_total_charge else None,
    )
    mx.eval(charges_mx)
    charges = np.asarray(charges_mx, dtype=np.float64)
    symmetry_classes: list[list[int]] = []
    if symmetrize:
        charges, symmetry_classes = symmetrize_charges_by_topology(
            work_mol,
            charges,
            total_charge=total_charge if project_total_charge else None,
            resonance=symmetrize_resonance,
        )

    return AM1BCCResult(
        charges=charges,
        am1_charges=am1_charges,
        bcc_corrections=bcc_corrections,
        total_charge=float(total_charge),
        n_bonds_assigned=work_mol.GetNumBonds(),
        metadata={
            "method": "AM1-BCC",
            "am1_backend": f"mlxmolkit.nddo.nddo_energy(method='{am1_method}')",
            "bcc_source": "OpenFF Recharge original-am1-bcc.json",
            "bcc_aromaticity": "OpenFF Recharge AM1BCC aromaticity model ported to RDKit",
            "scf_converged": bool(am1.get("converged", False)),
            "scf_n_iter": int(am1.get("n_iter", -1)),
            "scf_eigh_backend": am1.get("eigh_backend", "numpy.linalg.eigh"),
            "embedding_stereo_stripped": bool(work_mol.HasProp("_mlxmolkit_embedding_stereo_stripped")),
            "symmetrized": bool(symmetrize),
            "symmetry_class_count": len(symmetry_classes),
            "symmetrized_atom_count": sum(len(group) for group in symmetry_classes),
            "rdkit_smiles": Chem.MolToSmiles(work_mol, isomericSmiles=True),
        },
    )


def am1_bcc_charges_from_rdkit_mols(
    mols: Sequence[Any],
    *,
    total_charges: Sequence[float] | None = None,
    parameters: Sequence[BCCParameter] | None = None,
    am1_method: str = "AM1",
    add_hs: bool = True,
    conf_id: int = -1,
    random_seed: int = 0xA11BCC,
    validate_bcc_coverage: bool = True,
    project_total_charge: bool = True,
    max_iter: int = 100,
    conv_tol: float = 1.0e-6,
    use_metal: bool = True,
    verbose: bool = False,
    require_scf_convergence: bool = True,
    symmetrize: bool = True,
    symmetrize_resonance: bool = True,
) -> list[AM1BCCResult]:
    """Assign AM1-BCC charges to a batch of RDKit molecules.

    RDKit molecule preparation and BCC SMARTS assignment remain per molecule,
    while the AM1 Mulliken baseline is evaluated through the batched NDDO SCF
    path. This keeps the public AM1-BCC semantics identical to
    :func:`am1_bcc_charges_from_rdkit_mol` and amortizes the expensive SCF
    Fock construction across the batch.
    """

    from rdkit import Chem

    mols = list(mols)
    if not mols:
        return []

    if total_charges is not None and len(total_charges) != len(mols):
        raise ValueError("total_charges must match the number of molecules")

    parameters = list(parameters or load_original_am1_bcc_parameters())
    work_mols = [
        _mol_with_3d_coordinates(mol, add_hs=add_hs, random_seed=random_seed)
        for mol in mols
    ]
    totals = (
        [float(Chem.GetFormalCharge(mol)) for mol in work_mols]
        if total_charges is None
        else [float(charge) for charge in total_charges]
    )

    atom_coord_batch: list[tuple[list[int], np.ndarray]] = []
    supported_elements = set(METHOD_PARAMS[am1_method])
    for mol_index, work_mol in enumerate(work_mols):
        atoms, coords = _atoms_and_coords_from_rdkit_mol(work_mol, conf_id=conf_id)
        unsupported_elements = sorted(set(atoms) - supported_elements)
        if unsupported_elements:
            raise ValueError(
                f"{am1_method} parameters are unavailable for molecule {mol_index} "
                f"atomic numbers {unsupported_elements}; current exact AM1-BCC "
                "baseline supports molecules whose AM1 semiempirical parameters "
                "are present locally"
            )
        atom_coord_batch.append((atoms, coords))

    am1_results = nddo_energy_batch(
        atom_coord_batch,
        max_iter=max_iter,
        conv_tol=conv_tol,
        use_metal=use_metal,
        verbose=verbose,
        method=am1_method,
        molecular_charges=totals,
    )
    if require_scf_convergence:
        failed = [
            index
            for index, am1 in enumerate(am1_results)
            if not bool(am1.get("converged", False))
        ]
        if failed:
            preview = ", ".join(str(index) for index in failed[:12])
            suffix = "" if len(failed) <= 12 else f", ... ({len(failed)} total)"
            raise ValueError(f"{am1_method} batch SCF did not converge for molecule indices: {preview}{suffix}")

    results: list[AM1BCCResult] = []
    for mol_index, (work_mol, total_charge, am1) in enumerate(zip(work_mols, totals, am1_results)):
        atoms = atom_coord_batch[mol_index][0]
        am1_charges = (
            np.asarray(am1["charges"], dtype=np.float64)
            if "charges" in am1
            else _mulliken_charges_from_density(atoms, am1["density"], am1_method)
        )
        bcc_corrections = bcc_corrections_from_rdkit_mol(
            work_mol,
            parameters=parameters,
            validate=validate_bcc_coverage,
        )
        charges_mx = apply_bcc_corrections_mlx(
            am1_charges,
            bcc_corrections,
            total_charge=total_charge if project_total_charge else None,
        )
        mx.eval(charges_mx)
        charges = np.asarray(charges_mx, dtype=np.float64)
        symmetry_classes: list[list[int]] = []
        if symmetrize:
            charges, symmetry_classes = symmetrize_charges_by_topology(
                work_mol,
                charges,
                total_charge=total_charge if project_total_charge else None,
                resonance=symmetrize_resonance,
            )

        results.append(
            AM1BCCResult(
                charges=charges,
                am1_charges=am1_charges,
                bcc_corrections=bcc_corrections,
                total_charge=float(total_charge),
                n_bonds_assigned=work_mol.GetNumBonds(),
                metadata={
                    "method": "AM1-BCC",
                    "am1_backend": (
                        "mlxmolkit.nddo.nddo_energy_batch"
                        f"(method='{am1_method}', use_metal={use_metal})"
                    ),
                    "bcc_source": "OpenFF Recharge original-am1-bcc.json",
                    "bcc_aromaticity": "OpenFF Recharge AM1BCC aromaticity model ported to RDKit",
                    "batch_index": mol_index,
                    "batch_size": len(work_mols),
                    "scf_converged": bool(am1.get("converged", False)),
                    "scf_n_iter": int(am1.get("n_iter", -1)),
                    "scf_eigh_backend": am1.get("eigh_backend", "numpy.linalg.eigh"),
                    "scf_density_solver": am1.get("density_solver"),
                    "scf_batch_bucket_index": am1.get("batch_bucket_index"),
                    "scf_batch_bucket_count": am1.get("batch_bucket_count"),
                    "scf_batch_bucket_size": am1.get("batch_bucket_size"),
                    "scf_batch_bucket_max_basis": am1.get("batch_bucket_max_basis"),
                    "embedding_stereo_stripped": bool(
                        work_mol.HasProp("_mlxmolkit_embedding_stereo_stripped")
                    ),
                    "symmetrized": bool(symmetrize),
                    "symmetry_class_count": len(symmetry_classes),
                    "symmetrized_atom_count": sum(len(group) for group in symmetry_classes),
                    "rdkit_smiles": Chem.MolToSmiles(work_mol, isomericSmiles=True),
                },
            )
        )

    return results
