from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Sequence

import numpy as np

from mlxmolkit.esp_resp import (
    EspRespFitResult,
    connolly_surface_grid,
    coulomb_potential_from_charges_mlx,
    fit_esp_charges_mlx,
    fit_resp_charges_mlx,
)


@dataclass(frozen=True)
class CheeseChargeTrainingRecord:
    """One molecule with graph/3D inputs and ESP/RESP charge labels."""

    identifier: str
    smiles: str
    atomic_numbers: np.ndarray
    coords: np.ndarray
    bond_matrix: np.ndarray
    total_charge: float
    q_reference: np.ndarray | None = None
    q_esp: np.ndarray | None = None
    q_resp: np.ndarray | None = None
    esp_rmse: float = np.nan
    resp_rmse: float = np.nan
    resp_converged: bool = False
    n_grid: int = 0
    ok: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def esp_resp_labels_from_reference_charges(
    atoms: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
    reference_charges: Sequence[float] | np.ndarray,
    *,
    total_charge: float = 0.0,
    shell_factors: Sequence[float] = (1.4, 1.6, 1.8, 2.0),
    point_density: float = 1.0,
    restraint_a: float = 5.0e-4,
    restraint_b: float = 0.1,
    restrain_hydrogens: bool = False,
    max_iter: int = 100,
    conv_tol: float = 1.0e-6,
    equivalence_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[EspRespFitResult, EspRespFitResult]:
    """Fit ESP and RESP labels from a reference point-charge potential.

    This is a practical open bootstrap for CHEESE-style charge training:
    AM1-BCC, PM6, xTB, or another charge source supplies a molecular
    electrostatic potential on a Connolly/MK surface; the same constrained
    linear and RESP fits then produce paired labels for two charge models.
    """

    atom_array = np.asarray(atoms, dtype=np.int64)
    coord_array = np.asarray(coords, dtype=np.float64)
    q_ref = np.asarray(reference_charges, dtype=np.float64)
    if atom_array.shape != (coord_array.shape[0],):
        raise ValueError("atoms and coords must have the same number of atoms")
    if q_ref.shape != atom_array.shape:
        raise ValueError("reference_charges must have one value per atom")

    grid = connolly_surface_grid(
        atom_array,
        coord_array,
        shell_factors=shell_factors,
        point_density=point_density,
    )
    esp_values = coulomb_potential_from_charges_mlx(coord_array, q_ref, grid)
    esp = fit_esp_charges_mlx(
        coord_array,
        grid,
        esp_values,
        total_charge=total_charge,
        equivalence_groups=equivalence_groups,
    )
    resp = fit_resp_charges_mlx(
        coord_array,
        grid,
        esp_values,
        total_charge=total_charge,
        equivalence_groups=equivalence_groups,
        restraint_a=restraint_a,
        restraint_b=restraint_b,
        restrain_hydrogens=restrain_hydrogens,
        atoms=atom_array,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    esp.metadata.update({"source": "reference_charge_connolly_surface"})
    resp.metadata.update({"source": "reference_charge_connolly_surface"})
    return esp, resp


def failed_charge_training_record(
    identifier: str,
    smiles: str,
    atomic_numbers: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
    bond_matrix: Sequence[Sequence[int]] | np.ndarray,
    total_charge: float,
    error: Exception | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> CheeseChargeTrainingRecord:
    """Create a record that preserves graph inputs while marking labels failed."""

    error_text = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return CheeseChargeTrainingRecord(
        identifier=str(identifier),
        smiles=str(smiles),
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
        coords=np.asarray(coords, dtype=np.float32),
        bond_matrix=np.asarray(bond_matrix, dtype=np.int32),
        total_charge=float(total_charge),
        ok=False,
        error=error_text,
        metadata=dict(metadata or {}),
    )


def write_cheese_charge_training_npz(
    records: Sequence[CheeseChargeTrainingRecord],
    path: str | Path,
    *,
    label_source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write variable-size CHEESE charge-training records to one NPZ file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        raise ValueError("at least one record is required")

    atom_offsets = np.zeros(len(records) + 1, dtype=np.int64)
    bond_offsets = np.zeros(len(records) + 1, dtype=np.int64)
    bond_i_parts: list[np.ndarray] = []
    bond_j_parts: list[np.ndarray] = []
    bond_state_parts: list[np.ndarray] = []

    for index, record in enumerate(records):
        _validate_record(record)
        n_atoms = int(record.atomic_numbers.shape[0])
        atom_offsets[index + 1] = atom_offsets[index] + n_atoms
        rows, cols = np.nonzero(record.bond_matrix)
        bond_offsets[index + 1] = bond_offsets[index] + len(rows)
        bond_i_parts.append(rows.astype(np.int32, copy=False))
        bond_j_parts.append(cols.astype(np.int32, copy=False))
        bond_state_parts.append(record.bond_matrix[rows, cols].astype(np.int32, copy=False))

    n_total_atoms = int(atom_offsets[-1])
    atomic_numbers = np.concatenate([r.atomic_numbers.astype(np.int32, copy=False) for r in records])
    coords = np.concatenate([r.coords.astype(np.float32, copy=False) for r in records], axis=0)
    q_reference = _flatten_optional_charge(records, "q_reference", n_total_atoms)
    q_esp = _flatten_optional_charge(records, "q_esp", n_total_atoms)
    q_resp = _flatten_optional_charge(records, "q_resp", n_total_atoms)

    out_metadata = {
        "format": "mlxmolkit.cheese_charge_training",
        "format_version": 1,
        "label_source": str(label_source),
        **dict(metadata or {}),
    }
    np.savez_compressed(
        path,
        format_version=np.array([1], dtype=np.int64),
        label_source=np.array([label_source], dtype=str),
        metadata_json=np.array([json.dumps(out_metadata, sort_keys=True)], dtype=str),
        ids=np.array([r.identifier for r in records], dtype=str),
        smiles=np.array([r.smiles for r in records], dtype=str),
        n_atoms=np.array([r.atomic_numbers.shape[0] for r in records], dtype=np.int32),
        total_charge=np.array([r.total_charge for r in records], dtype=np.float32),
        atom_offsets=atom_offsets,
        atomic_numbers=atomic_numbers,
        coords=coords,
        bond_offsets=bond_offsets,
        bond_i=np.concatenate(bond_i_parts) if bond_i_parts else np.empty((0,), dtype=np.int32),
        bond_j=np.concatenate(bond_j_parts) if bond_j_parts else np.empty((0,), dtype=np.int32),
        bond_state=np.concatenate(bond_state_parts) if bond_state_parts else np.empty((0,), dtype=np.int32),
        q_reference=q_reference,
        q_esp=q_esp,
        q_resp=q_resp,
        esp_rmse=np.array([r.esp_rmse for r in records], dtype=np.float32),
        resp_rmse=np.array([r.resp_rmse for r in records], dtype=np.float32),
        resp_converged=np.array([r.resp_converged for r in records], dtype=bool),
        n_grid=np.array([r.n_grid for r in records], dtype=np.int32),
        ok=np.array([r.ok for r in records], dtype=bool),
        errors=np.array([r.error for r in records], dtype=str),
        record_metadata_json=np.array([json.dumps(r.metadata, sort_keys=True) for r in records], dtype=str),
    )
    return {
        **out_metadata,
        "path": str(path),
        "n_molecules": len(records),
        "n_atoms": n_total_atoms,
        "n_ok": int(sum(record.ok for record in records)),
        "n_failed": int(sum(not record.ok for record in records)),
    }


def _flatten_optional_charge(
    records: Sequence[CheeseChargeTrainingRecord],
    field_name: str,
    n_total_atoms: int,
) -> np.ndarray:
    out = np.full((n_total_atoms,), np.nan, dtype=np.float32)
    cursor = 0
    for record in records:
        n_atoms = int(record.atomic_numbers.shape[0])
        value = getattr(record, field_name)
        if value is not None:
            q = np.asarray(value, dtype=np.float32)
            if q.shape != (n_atoms,):
                raise ValueError(f"{field_name} for {record.identifier!r} must have shape ({n_atoms},)")
            out[cursor : cursor + n_atoms] = q
        cursor += n_atoms
    return out


def _validate_record(record: CheeseChargeTrainingRecord) -> None:
    atoms = np.asarray(record.atomic_numbers)
    coords = np.asarray(record.coords)
    bonds = np.asarray(record.bond_matrix)
    if atoms.ndim != 1:
        raise ValueError(f"atomic_numbers for {record.identifier!r} must be 1D")
    if coords.shape != (atoms.shape[0], 3):
        raise ValueError(f"coords for {record.identifier!r} must have shape (n_atoms, 3)")
    if bonds.shape != (atoms.shape[0], atoms.shape[0]):
        raise ValueError(f"bond_matrix for {record.identifier!r} must have shape (n_atoms, n_atoms)")
