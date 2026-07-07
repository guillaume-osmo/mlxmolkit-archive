"""CHEESE-style 3D shape and electrostatic similarity descriptors.

The exact CHEESE workflow is a conformer/alignment problem. This module
implements the fast, differentiable scoring core once conformers are in a
common frame: Gaussian molecular shape overlap, charge-weighted electrostatic
field overlap, and active-template aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from mlxmolkit.esp_resp import VDW_RADII


@dataclass(frozen=True)
class CheeseBatch:
    """Padded molecular batch used by CHEESE-style descriptors."""

    atomic_numbers: Any
    coords: Any
    charges: Any
    mask: Any
    ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CheeseSimilarityResult:
    """Pairwise shape, electrostatic, and combined similarity matrices."""

    shape: Any
    electrostatic: Any
    combined: Any


@dataclass(frozen=True)
class CheeseAlignmentConfig:
    """Options for ROCS/Roshambo-style rigid shape overlay."""

    start_mode: str = "roshambo"
    max_initial_starts: int | None = 256
    refine_top_k: int = 12
    local_refine: bool = True
    max_refine_steps: int = 80
    rotation_step_degrees: float = 12.0
    translation_step: float = 0.25
    min_rotation_step_degrees: float = 0.1
    min_translation_step: float = 0.002
    wiggle_degrees: float = 25.0
    random_starts: int = 0
    random_seed: int = 0
    gaussian_alpha: float = 2.7
    vdw_scale: float = 1.0
    default_radius: float = 1.80
    shape_weight: float = 1.0
    electrostatic_weight: float | None = None
    map_electrostatic_to_unit: bool = True
    electrostatic_metric: str = "carbo"
    optimize: str = "shape"
    start_score_backend: str = "auto"
    cocluster_starts: int = 0
    cocluster_pair_pool: int = 24
    cocluster_neighbors: int = 8
    cocluster_distance_sigma: float = 0.35
    cocluster_offdiag_weight: float = 0.5


@dataclass(frozen=True)
class CheeseAlignmentResult:
    """Best rigid overlay and CHEESE scores for one probe/reference pair."""

    shape: float
    electrostatic: float
    combined: float
    overlap: float
    probe_self_overlap: float
    reference_self_overlap: float
    rotation: np.ndarray
    translation: np.ndarray
    aligned_coords: np.ndarray
    probe_index: int = -1
    reference_index: int = -1
    n_starts: int = 0
    n_refined: int = 0
    converged: bool = True


def cheese_batch(
    atom_numbers: Sequence[Sequence[int]],
    coords: Sequence[Any],
    charges: Sequence[Any] | None = None,
    *,
    ids: Sequence[str] | None = None,
    pad_to: int | None = None,
) -> CheeseBatch:
    """Pack variable-size molecules into padded MLX arrays.

    ``charges`` may be omitted for shape-only work; the electrostatic channel
    will then be all zeros.
    """

    if len(atom_numbers) != len(coords):
        raise ValueError("atom_numbers and coords must contain the same number of molecules")
    if not atom_numbers:
        raise ValueError("at least one molecule is required")
    if charges is not None and len(charges) != len(atom_numbers):
        raise ValueError("charges must match the number of molecules")
    if ids is not None and len(ids) != len(atom_numbers):
        raise ValueError("ids must match the number of molecules")

    atom_arrays = [np.asarray(atoms, dtype=np.int32) for atoms in atom_numbers]
    coord_arrays = [np.asarray(xyz, dtype=np.float32) for xyz in coords]
    charge_arrays = (
        [np.asarray(q, dtype=np.float32) for q in charges]
        if charges is not None
        else [np.zeros((len(atoms),), dtype=np.float32) for atoms in atom_arrays]
    )

    max_atoms = max(len(atoms) for atoms in atom_arrays)
    if pad_to is not None:
        max_atoms = max(max_atoms, int(pad_to))

    z_pad = np.zeros((len(atom_arrays), max_atoms), dtype=np.int32)
    xyz_pad = np.zeros((len(atom_arrays), max_atoms, 3), dtype=np.float32)
    q_pad = np.zeros((len(atom_arrays), max_atoms), dtype=np.float32)
    mask = np.zeros((len(atom_arrays), max_atoms), dtype=np.float32)

    for i, (atoms, xyz, q) in enumerate(zip(atom_arrays, coord_arrays, charge_arrays, strict=True)):
        n_atoms = len(atoms)
        if n_atoms == 0:
            raise ValueError("molecules must contain at least one atom")
        if xyz.shape != (n_atoms, 3):
            raise ValueError(f"coords[{i}] must have shape ({n_atoms}, 3)")
        if q.shape != (n_atoms,):
            raise ValueError(f"charges[{i}] must have shape ({n_atoms},)")
        if np.any(atoms < 0):
            raise ValueError("atomic numbers must be non-negative")
        z_pad[i, :n_atoms] = atoms
        xyz_pad[i, :n_atoms] = xyz
        q_pad[i, :n_atoms] = q
        mask[i, :n_atoms] = 1.0

    return CheeseBatch(
        atomic_numbers=mx.array(z_pad),
        coords=mx.array(xyz_pad),
        charges=mx.array(q_pad),
        mask=mx.array(mask),
        ids=None if ids is None else tuple(ids),
    )


def cheese_batch_from_rdkit_mols(
    mols: Sequence[Any],
    charges: Sequence[Any] | None = None,
    *,
    conf_ids: Sequence[int] | None = None,
    ids: Sequence[str] | None = None,
    pad_to: int | None = None,
) -> CheeseBatch:
    """Build a CHEESE batch from conformer-bearing RDKit molecules."""

    if conf_ids is None:
        conf_ids = [-1] * len(mols)
    if len(conf_ids) != len(mols):
        raise ValueError("conf_ids must match the number of molecules")

    atom_numbers: list[list[int]] = []
    xyz: list[np.ndarray] = []
    for mol_index, (mol, conf_id) in enumerate(zip(mols, conf_ids, strict=True)):
        if mol.GetNumConformers() == 0:
            raise ValueError(f"molecule {mol_index} has no conformer")
        atom_numbers.append([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        xyz.append(np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), dtype=np.float32))

    return cheese_batch(atom_numbers, xyz, charges, ids=ids, pad_to=pad_to)


def atom_gaussian_parameters_mlx(
    atomic_numbers: Any,
    *,
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
) -> tuple[mx.array, mx.array]:
    """Return Gaussian exponents and amplitudes for atom-centered volumes.

    The exponent is ``gaussian_alpha / radius**2``. The amplitude is normalized
    so that each isolated atom integrates to its vdW sphere volume. This keeps
    shape scores stable while preserving atom-size differences.
    """

    z = _as_mx_int(atomic_numbers)
    radii = mx.take(_vdw_radius_table(default_radius=default_radius), z)
    radii = mx.maximum(radii * float(vdw_scale), mx.array(1.0e-4, dtype=mx.float32))
    exponent = mx.array(float(gaussian_alpha), dtype=mx.float32) / (radii * radii)
    volume = (4.0 / 3.0) * _PI * radii**3
    amplitude = volume * (exponent / _PI) ** 1.5
    return exponent, amplitude


def gaussian_shape_overlap_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
) -> mx.array:
    """Return analytic Gaussian volume overlaps for all probe/reference pairs."""

    reference = probe if reference is None else reference
    probe_exp, probe_amp = atom_gaussian_parameters_mlx(
        probe.atomic_numbers,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    ref_exp, ref_amp = atom_gaussian_parameters_mlx(
        reference.atomic_numbers,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    return _gaussian_overlap_matrix(
        probe.coords,
        probe_exp,
        probe_amp,
        probe.mask,
        reference.coords,
        ref_exp,
        ref_amp,
        reference.mask,
    )


def gaussian_shape_self_overlap_mlx(
    batch: CheeseBatch,
    *,
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
) -> mx.array:
    """Return per-molecule Gaussian self-overlaps."""

    exponent, amplitude = atom_gaussian_parameters_mlx(
        batch.atomic_numbers,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    return _gaussian_self_overlap(batch.coords, exponent, amplitude, batch.mask)


def shape_tanimoto_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
    eps: float = 1.0e-8,
) -> mx.array:
    """Return Gaussian shape Tanimoto similarities in ``[0, 1]``."""

    reference = probe if reference is None else reference
    overlap = gaussian_shape_overlap_matrix_mlx(
        probe,
        reference,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    self_probe = gaussian_shape_self_overlap_mlx(
        probe,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    self_ref = gaussian_shape_self_overlap_mlx(
        reference,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    denom = self_probe[:, None] + self_ref[None, :] - overlap
    score = overlap / mx.maximum(denom, mx.array(float(eps), dtype=overlap.dtype))
    return mx.clip(score, 0.0, 1.0)


def shape_carbo_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
    eps: float = 1.0e-8,
) -> mx.array:
    """Return CHEESE/paper-style shape Carbo similarities in ``[0, 1]``.

    The CHEESE preprint defines shapesim analogously to the electrostatic
    Carbo index: cross-overlap normalized by the geometric mean of self
    overlaps. This is distinct from ROCS/RDKit-style shape Tanimoto.
    """

    reference = probe if reference is None else reference
    overlap = gaussian_shape_overlap_matrix_mlx(
        probe,
        reference,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    self_probe = gaussian_shape_self_overlap_mlx(
        probe,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    self_ref = gaussian_shape_self_overlap_mlx(
        reference,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    denom = mx.sqrt(
        mx.maximum(self_probe[:, None], mx.array(float(eps), dtype=overlap.dtype))
        * mx.maximum(self_ref[None, :], mx.array(float(eps), dtype=overlap.dtype))
    )
    return mx.clip(overlap / mx.maximum(denom, mx.array(float(eps), dtype=overlap.dtype)), 0.0, 1.0)


def rdkit_shape_tanimoto_from_mols(
    probe_mol: Any,
    reference_mol: Any,
    *,
    probe_conf_id: int = -1,
    reference_conf_id: int = -1,
    grid_spacing: float = 0.5,
    bits_per_point: Any | None = None,
    vdw_scale: float = 0.8,
    step_size: float = 0.25,
    max_layers: int = -1,
    ignore_hs: bool = True,
) -> float:
    """Return exact RDKit/ESP-Sim shape similarity for aligned RDKit mols.

    This is the parity path for ``espsim.GetShapeSim()``, which is
    ``1 - AllChem.ShapeTanimotoDist(...)`` with RDKit's voxel occupancy grid.
    """

    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - import guard for non-RDKit installs.
        raise ImportError("RDKit is required for rdkit_shape_tanimoto_from_mols()") from exc

    if bits_per_point is None:
        bits_per_point = DataStructs.DiscreteValueType.TWOBITVALUE
    distance = AllChem.ShapeTanimotoDist(
        probe_mol,
        reference_mol,
        int(probe_conf_id),
        int(reference_conf_id),
        float(grid_spacing),
        bits_per_point,
        float(vdw_scale),
        float(step_size),
        int(max_layers),
        bool(ignore_hs),
    )
    return float(1.0 - distance)


def rdkit_shape_tanimoto_matrix_from_mols(
    probe_mols: Sequence[Any],
    reference_mols: Sequence[Any] | None = None,
    *,
    probe_conf_ids: Sequence[int] | None = None,
    reference_conf_ids: Sequence[int] | None = None,
    grid_spacing: float = 0.5,
    bits_per_point: Any | None = None,
    vdw_scale: float = 0.8,
    step_size: float = 0.25,
    max_layers: int = -1,
    ignore_hs: bool = True,
) -> mx.array:
    """Return exact RDKit shape similarities for RDKit molecule lists."""

    reference_mols = probe_mols if reference_mols is None else reference_mols
    if probe_conf_ids is None:
        probe_conf_ids = [-1] * len(probe_mols)
    if reference_conf_ids is None:
        reference_conf_ids = [-1] * len(reference_mols)
    if len(probe_conf_ids) != len(probe_mols):
        raise ValueError("probe_conf_ids must match probe_mols")
    if len(reference_conf_ids) != len(reference_mols):
        raise ValueError("reference_conf_ids must match reference_mols")

    out = np.zeros((len(probe_mols), len(reference_mols)), dtype=np.float32)
    for i, mol_a in enumerate(probe_mols):
        for j, mol_b in enumerate(reference_mols):
            out[i, j] = rdkit_shape_tanimoto_from_mols(
                mol_a,
                mol_b,
                probe_conf_id=int(probe_conf_ids[i]),
                reference_conf_id=int(reference_conf_ids[j]),
                grid_spacing=grid_spacing,
                bits_per_point=bits_per_point,
                vdw_scale=vdw_scale,
                step_size=step_size,
                max_layers=max_layers,
                ignore_hs=ignore_hs,
            )
    return mx.array(out)


def rdkit_shape_occupancy_mlx(
    batch: CheeseBatch,
    grid_coords: Any,
    *,
    bits_per_point: int = 2,
    vdw_scale: float = 0.8,
    step_size: float = 0.25,
    max_layers: int = -1,
    ignore_hs: bool = True,
    default_radius: float = 1.80,
) -> mx.array:
    """Encode RDKit-style discrete shape occupancy on a shared grid.

    The output is a ``(batch, n_grid)`` integer-valued MLX array stored as
    ``float32`` for fast reductions. It mirrors RDKit's ``setSphereOccupancy``
    layer rules for a caller-supplied common grid.
    """

    if bits_per_point <= 0 or bits_per_point > 16:
        raise ValueError("bits_per_point must be in the range [1, 16]")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    max_val = (1 << int(bits_per_point)) - 1
    n_layers = max_val
    val_step = 1
    if 0 < int(max_layers) <= n_layers:
        n_layers = int(max_layers)
        val_step = (max_val + 1) // n_layers

    xyz = _as_mx_float(batch.coords)
    atom_mask = _as_mx_float(batch.mask)
    z = _as_mx_int(batch.atomic_numbers)
    if ignore_hs:
        atom_mask = atom_mask * (z != 1)
    radii = mx.take(_rdkit_vdw_radius_table(default_radius=default_radius), z) * float(vdw_scale)
    grid = _as_mx_float(grid_coords)
    if grid.ndim != 2 or grid.shape[-1] != 3:
        raise ValueError(f"grid_coords must have shape (n_grid, 3), got {grid.shape}")

    dist = mx.sqrt(mx.sum((grid[None, :, None, :] - xyz[:, None, :, :]) ** 2, axis=-1))
    base = radii[:, None, :]
    outer = base + float(n_layers) * float(step_size)
    layer = mx.floor((dist - base) / float(step_size) + 1.0)
    val_change = layer * float(val_step)
    shell_val = mx.where(val_change < float(max_val), float(max_val) - val_change, 0.0)
    atom_values = mx.where(dist < base, float(max_val), mx.where(dist < outer, shell_val, 0.0))
    atom_values = atom_values * atom_mask[:, None, :]
    return mx.max(atom_values, axis=-1)


def rdkit_grid_shape_tanimoto_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    grid_coords: Any,
    bits_per_point: int = 2,
    vdw_scale: float = 0.8,
    step_size: float = 0.25,
    max_layers: int = -1,
    ignore_hs: bool = True,
    default_radius: float = 1.80,
    eps: float = 1.0e-8,
) -> mx.array:
    """Return RDKit-style shape Tanimoto on a shared MLX grid.

    This is the fast screening form of RDKit's occupancy-grid shape metric. Use
    ``rdkit_shape_tanimoto_from_mols()`` when exact pairwise RDKit parity is
    required, because RDKit builds a pair-specific canonical union grid.
    """

    reference = probe if reference is None else reference
    probe_occ = rdkit_shape_occupancy_mlx(
        probe,
        grid_coords,
        bits_per_point=bits_per_point,
        vdw_scale=vdw_scale,
        step_size=step_size,
        max_layers=max_layers,
        ignore_hs=ignore_hs,
        default_radius=default_radius,
    )
    ref_occ = rdkit_shape_occupancy_mlx(
        reference,
        grid_coords,
        bits_per_point=bits_per_point,
        vdw_scale=vdw_scale,
        step_size=step_size,
        max_layers=max_layers,
        ignore_hs=ignore_hs,
        default_radius=default_radius,
    )
    total_probe = mx.sum(probe_occ, axis=-1)
    total_ref = mx.sum(ref_occ, axis=-1)
    l1 = mx.sum(mx.abs(probe_occ[:, None, :] - ref_occ[None, :, :]), axis=-1)
    intersection = 0.5 * (total_probe[:, None] + total_ref[None, :] - l1)
    union = l1 + intersection
    score = intersection / mx.maximum(union, mx.array(float(eps), dtype=intersection.dtype))
    return mx.clip(score, 0.0, 1.0)


def electrostatic_carbo_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
    eps: float = 1.0e-8,
) -> mx.array:
    """Return signed ESP-Sim Gaussian Coulomb Carbo similarities.

    Identical charge fields score ``1`` and inverted charge fields score ``-1``.
    The shape-Gaussian keyword arguments are accepted for API compatibility and
    ignored by the ESP-Sim electrostatic kernel.
    """

    del gaussian_alpha, vdw_scale, default_radius
    return electrostatic_similarity_matrix_mlx(probe, reference, metric="carbo", eps=eps)


def electrostatic_overlap_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
) -> mx.array:
    """Return ESP-Sim analytic Gaussian Coulomb overlap integrals."""

    reference = probe if reference is None else reference
    return _espsim_charge_overlap_matrix(
        probe.coords,
        probe.charges,
        probe.mask,
        reference.coords,
        reference.charges,
        reference.mask,
    )


def electrostatic_self_overlap_mlx(batch: CheeseBatch) -> mx.array:
    """Return per-molecule ESP-Sim analytic self-overlap integrals."""

    return _espsim_charge_self_overlap(batch.coords, batch.charges, batch.mask)


def electrostatic_similarity_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    metric: str = "carbo",
    renormalize: bool = False,
    eps: float = 1.0e-8,
) -> mx.array:
    """Return ESP-Sim-style electrostatic similarities.

    ``metric="carbo"`` returns the signed Carbo index in ``[-1, 1]``.
    ``metric="tanimoto"`` returns the ESP-Sim electrostatic Tanimoto value,
    whose theoretical lower bound is ``-1/3``. Set ``renormalize=True`` to map
    the chosen metric to ``[0, 1]`` using ESP-Sim's published ranges.
    """

    metric = metric.lower()
    reference = probe if reference is None else reference
    cross = electrostatic_overlap_matrix_mlx(probe, reference)
    self_probe = electrostatic_self_overlap_mlx(probe)
    self_ref = electrostatic_self_overlap_mlx(reference)
    score = _esp_similarity_from_sums(
        cross,
        self_probe[:, None],
        self_ref[None, :],
        metric=metric,
        eps=eps,
    )
    if renormalize:
        score = _renormalize_electrostatic_mlx(score, metric)
    return score


def cheese_similarity_matrix_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    shape_weight: float = 1.0,
    electrostatic_weight: float = 1.0,
    map_electrostatic_to_unit: bool = True,
    electrostatic_metric: str = "carbo",
    shape_metric: str = "tanimoto",
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
) -> CheeseSimilarityResult:
    """Return pairwise CHEESE-style shape, ESP, and combined similarities."""

    reference = probe if reference is None else reference
    shape_metric = shape_metric.lower()
    if shape_metric == "tanimoto":
        shape = shape_tanimoto_matrix_mlx(
            probe,
            reference,
            gaussian_alpha=gaussian_alpha,
            vdw_scale=vdw_scale,
            default_radius=default_radius,
        )
    elif shape_metric == "carbo":
        shape = shape_carbo_matrix_mlx(
            probe,
            reference,
            gaussian_alpha=gaussian_alpha,
            vdw_scale=vdw_scale,
            default_radius=default_radius,
        )
    else:
        raise ValueError("shape_metric must be 'tanimoto' or 'carbo'")
    electrostatic = electrostatic_similarity_matrix_mlx(
        probe,
        reference,
        metric=electrostatic_metric,
    )
    electrostatic_channel = (
        _renormalize_electrostatic_mlx(electrostatic, electrostatic_metric)
        if map_electrostatic_to_unit
        else electrostatic
    )
    total_weight = float(shape_weight) + float(electrostatic_weight)
    if total_weight <= 0:
        raise ValueError("at least one CHEESE similarity weight must be positive")
    combined = (float(shape_weight) * shape + float(electrostatic_weight) * electrostatic_channel) / total_weight
    return CheeseSimilarityResult(shape=shape, electrostatic=electrostatic, combined=combined)


def gaussian_shape_paired_overlap_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch,
    *,
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
) -> mx.array:
    """Return analytic Gaussian volume overlaps for paired batches only."""

    if probe.coords.shape[0] != reference.coords.shape[0]:
        raise ValueError("probe and reference batches must have the same length")
    probe_exp, probe_amp = atom_gaussian_parameters_mlx(
        probe.atomic_numbers,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    ref_exp, ref_amp = atom_gaussian_parameters_mlx(
        reference.atomic_numbers,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    return _gaussian_paired_overlap(
        probe.coords,
        probe_exp,
        probe_amp,
        probe.mask,
        reference.coords,
        ref_exp,
        ref_amp,
        reference.mask,
    )


def electrostatic_paired_overlap_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch,
) -> mx.array:
    """Return ESP-Sim analytic Gaussian Coulomb overlaps for paired batches."""

    if probe.coords.shape[0] != reference.coords.shape[0]:
        raise ValueError("probe and reference batches must have the same length")
    return _espsim_charge_paired_overlap(
        probe.coords,
        probe.charges,
        probe.mask,
        reference.coords,
        reference.charges,
        reference.mask,
    )


def cheese_similarity_pairs_mlx(
    probe: CheeseBatch,
    reference: CheeseBatch,
    *,
    shape_weight: float = 1.0,
    electrostatic_weight: float = 1.0,
    map_electrostatic_to_unit: bool = True,
    electrostatic_metric: str = "carbo",
    shape_metric: str = "tanimoto",
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
    eps: float = 1.0e-8,
) -> CheeseSimilarityResult:
    """Return CHEESE shape, ESP, and combined similarities for paired batches.

    Unlike :func:`cheese_similarity_matrix_mlx`, this computes only pair
    ``i`` vs pair ``i``. It is the right primitive for conformer-pair scoring
    after a batched alignment step.
    """

    if probe.coords.shape[0] != reference.coords.shape[0]:
        raise ValueError("probe and reference batches must have the same length")

    shape_metric = shape_metric.lower()
    overlap = gaussian_shape_paired_overlap_mlx(
        probe,
        reference,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    self_probe = gaussian_shape_self_overlap_mlx(
        probe,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    self_ref = gaussian_shape_self_overlap_mlx(
        reference,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    if shape_metric == "tanimoto":
        denom = self_probe + self_ref - overlap
        shape = mx.clip(overlap / mx.maximum(denom, mx.array(float(eps), dtype=overlap.dtype)), 0.0, 1.0)
    elif shape_metric == "carbo":
        denom = mx.sqrt(
            mx.maximum(self_probe, mx.array(float(eps), dtype=overlap.dtype))
            * mx.maximum(self_ref, mx.array(float(eps), dtype=overlap.dtype))
        )
        shape = mx.clip(overlap / mx.maximum(denom, mx.array(float(eps), dtype=overlap.dtype)), 0.0, 1.0)
    else:
        raise ValueError("shape_metric must be 'tanimoto' or 'carbo'")

    electrostatic_cross = electrostatic_paired_overlap_mlx(probe, reference)
    electrostatic = _esp_similarity_from_sums(
        electrostatic_cross,
        electrostatic_self_overlap_mlx(probe),
        electrostatic_self_overlap_mlx(reference),
        metric=electrostatic_metric.lower(),
        eps=eps,
    )
    electrostatic_channel = (
        _renormalize_electrostatic_mlx(electrostatic, electrostatic_metric)
        if map_electrostatic_to_unit
        else electrostatic
    )
    total_weight = float(shape_weight) + float(electrostatic_weight)
    if total_weight <= 0:
        raise ValueError("at least one CHEESE similarity weight must be positive")
    combined = (float(shape_weight) * shape + float(electrostatic_weight) * electrostatic_channel) / total_weight
    return CheeseSimilarityResult(shape=shape, electrostatic=electrostatic, combined=combined)


def electrostatic_potential_on_grid_mlx(
    coords: Any,
    charges: Any,
    grid_coords: Any,
    *,
    mask: Any | None = None,
    min_distance: float = 1.0e-4,
) -> mx.array:
    """Evaluate batched point-charge ESP values on a shared grid.

    ``coords`` is ``(B, N, 3)`` or ``(N, 3)``; ``grid_coords`` is ``(G, 3)``.
    The result is ``(B, G)``.
    """

    xyz, q, atom_mask = _as_batched_coords_charges(coords, charges, mask)
    grid = _as_mx_float(grid_coords)
    if grid.ndim != 2 or grid.shape[-1] != 3:
        raise ValueError(f"grid_coords must have shape (n_grid, 3), got {grid.shape}")

    dist = _batched_distance_from_grid_to_atoms(grid[None, :, :], xyz, min_distance=min_distance)
    return mx.sum(q[:, None, :] * atom_mask[:, None, :] / dist, axis=-1)


def electrostatic_potential_on_grid_metal(
    coords: Any,
    charges: Any,
    grid_coords: Any,
    *,
    mask: Any | None = None,
    min_distance: float = 1.0e-4,
) -> mx.array:
    """Evaluate batched point-charge ESP on a shared grid with one Metal kernel.

    This path avoids materializing the ``batch * n_grid * n_atoms`` distance
    tensor. Each GPU thread owns one ``(molecule, grid point)`` output and loops
    over atoms in registers.
    """

    if not hasattr(mx.fast, "metal_kernel"):
        return electrostatic_potential_on_grid_mlx(
            coords,
            charges,
            grid_coords,
            mask=mask,
            min_distance=min_distance,
        )

    xyz, q, atom_mask = _as_batched_coords_charges(coords, charges, mask)
    grid = _as_mx_float(grid_coords)
    if grid.ndim != 2 or grid.shape[-1] != 3:
        raise ValueError(f"grid_coords must have shape (n_grid, 3), got {grid.shape}")

    n_batch = int(xyz.shape[0])
    n_grid = int(grid.shape[0])
    total = n_batch * n_grid
    if total == 0:
        return mx.zeros((n_batch, n_grid), dtype=mx.float32)
    kernel = _get_esp_grid_kernel()
    config = mx.array([float(min_distance)], dtype=mx.float32)
    return kernel(
        inputs=[xyz, q, atom_mask, grid, config],
        output_shapes=[(n_batch, n_grid)],
        output_dtypes=[mx.float32],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
    )[0]


def electrostatic_similarity_to_template_grids_tiled_mlx(
    probe: CheeseBatch,
    template_grid_coords: Any,
    template_esp_values: Any,
    *,
    template_grid_mask: Any | None = None,
    metric: str = "carbo",
    probe_tile_size: int = 128,
    template_tile_size: int = 8,
    grid_tile_size: int = 2048,
    min_distance: float = 1.0e-4,
    eps: float = 1.0e-8,
) -> mx.array:
    """Memory-bounded template-grid ESP similarity.

    The naive tensor is ``n_probe * n_template * n_grid * n_atoms``. This
    tiled path keeps only a small ``probe_tile * template_tile * grid_tile *
    n_atoms`` block live and evaluates distances via batched matmul.
    """

    grid = _as_mx_float(template_grid_coords)
    ref = _as_mx_float(template_esp_values)
    if grid.ndim == 2:
        grid = grid[None, :, :]
    if ref.ndim == 1:
        ref = ref[None, :]
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError(f"template_grid_coords must have shape (n_templates, n_grid, 3), got {grid.shape}")
    if ref.shape != grid.shape[:2]:
        raise ValueError(f"template_esp_values shape {ref.shape} does not match grid shape {grid.shape[:2]}")

    if template_grid_mask is None:
        grid_mask = mx.ones(ref.shape, dtype=mx.float32)
    else:
        grid_mask = _as_mx_float(template_grid_mask)
        if grid_mask.ndim == 1:
            grid_mask = grid_mask[None, :]
        if grid_mask.shape != ref.shape:
            raise ValueError("template_grid_mask must match template_esp_values")

    coords = _as_mx_float(probe.coords)
    charges = _as_mx_float(probe.charges)
    atom_mask = _as_mx_float(probe.mask)

    n_probe = int(coords.shape[0])
    n_templates = int(grid.shape[0])
    n_grid = int(grid.shape[1])
    if probe_tile_size <= 0 or template_tile_size <= 0 or grid_tile_size <= 0:
        raise ValueError("tile sizes must be positive")

    row_blocks = []
    for p0 in range(0, n_probe, int(probe_tile_size)):
        p1 = min(n_probe, p0 + int(probe_tile_size))
        coord_tile = coords[p0:p1]
        charge_tile = charges[p0:p1]
        atom_mask_tile = atom_mask[p0:p1]
        col_blocks = []
        for t0 in range(0, n_templates, int(template_tile_size)):
            t1 = min(n_templates, t0 + int(template_tile_size))
            dot = mx.zeros((p1 - p0, t1 - t0), dtype=mx.float32)
            probe_norm2 = mx.zeros_like(dot)
            ref_norm2 = mx.zeros_like(dot)
            for g0 in range(0, n_grid, int(grid_tile_size)):
                g1 = min(n_grid, g0 + int(grid_tile_size))
                grid_tile = grid[t0:t1, g0:g1, :]
                ref_tile = ref[t0:t1, g0:g1]
                weight_tile = grid_mask[t0:t1, g0:g1]
                dist = _batched_distance_from_template_grid_to_atoms(
                    grid_tile,
                    coord_tile,
                    min_distance=min_distance,
                )
                probe_esp = mx.sum(
                    charge_tile[:, None, None, :] * atom_mask_tile[:, None, None, :] / dist,
                    axis=-1,
                )
                weights = weight_tile[None, :, :]
                dot = dot + mx.sum(probe_esp * ref_tile[None, :, :] * weights, axis=-1)
                probe_norm2 = probe_norm2 + mx.sum(probe_esp * probe_esp * weights, axis=-1)
                ref_norm2 = ref_norm2 + mx.sum(ref_tile[None, :, :] * ref_tile[None, :, :] * weights, axis=-1)
            col_blocks.append(_esp_similarity_from_sums(dot, probe_norm2, ref_norm2, metric=metric, eps=eps))
        row_blocks.append(mx.concatenate(col_blocks, axis=1))
    return mx.concatenate(row_blocks, axis=0)


def electrostatic_similarity_to_template_grids_mlx(
    probe: CheeseBatch,
    template_grid_coords: Any,
    template_esp_values: Any,
    *,
    template_grid_mask: Any | None = None,
    metric: str = "carbo",
    min_distance: float = 1.0e-4,
    eps: float = 1.0e-8,
) -> mx.array:
    """Compare probe ESPs to template ESP values on template-specific grids.

    ``template_grid_coords`` is ``(T, G, 3)`` or ``(G, 3)`` and
    ``template_esp_values`` is ``(T, G)`` or ``(G,)``. The result is a
    ``(n_probe, n_templates)`` matrix. This is the stricter CHEESE path when
    candidate conformers have already been aligned to each active/template.
    """

    grid = _as_mx_float(template_grid_coords)
    ref = _as_mx_float(template_esp_values)
    if grid.ndim == 2:
        grid = grid[None, :, :]
    if ref.ndim == 1:
        ref = ref[None, :]
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError(f"template_grid_coords must have shape (n_templates, n_grid, 3), got {grid.shape}")
    if ref.shape != grid.shape[:2]:
        raise ValueError(f"template_esp_values shape {ref.shape} does not match grid shape {grid.shape[:2]}")

    if template_grid_mask is None:
        grid_mask = mx.ones(ref.shape, dtype=mx.float32)
    else:
        grid_mask = _as_mx_float(template_grid_mask)
        if grid_mask.ndim == 1:
            grid_mask = grid_mask[None, :]
        if grid_mask.shape != ref.shape:
            raise ValueError("template_grid_mask must match template_esp_values")

    coords = _as_mx_float(probe.coords)
    charges = _as_mx_float(probe.charges)
    atom_mask = _as_mx_float(probe.mask)
    dist = _batched_distance_from_template_grid_to_atoms(grid, coords, min_distance=min_distance)
    probe_esp = mx.sum(charges[:, None, None, :] * atom_mask[:, None, None, :] / dist, axis=-1)

    weights = grid_mask[None, :, :]
    dot = mx.sum(probe_esp * ref[None, :, :] * weights, axis=-1)
    probe_norm2 = mx.sum(probe_esp * probe_esp * weights, axis=-1)
    ref_norm2 = mx.sum(ref[None, :, :] * ref[None, :, :] * weights, axis=-1)
    return _esp_similarity_from_sums(dot, probe_norm2, ref_norm2, metric=metric, eps=eps)


def mean_similarity_to_actives_mlx(
    similarity_matrix: Any,
    *,
    active_mask: Any | None = None,
    top_k: int | None = None,
) -> mx.array:
    """Aggregate candidate-to-active similarities like the CHEESE slide equation.

    With ``top_k=None`` this returns ``mean_j S(i, active_j)``. With ``top_k``
    it averages only the best active/template similarities per candidate.
    """

    sim = _as_mx_float(similarity_matrix)
    if sim.ndim != 2:
        raise ValueError("similarity_matrix must be two-dimensional")

    if active_mask is None:
        weights = mx.ones((sim.shape[1],), dtype=mx.float32)
    else:
        weights = _as_mx_float(active_mask)
        if weights.shape != (sim.shape[1],):
            raise ValueError("active_mask must have one value per similarity column")
        weights = mx.where(weights > 0, mx.ones_like(weights), mx.zeros_like(weights))

    if top_k is None:
        denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=sim.dtype))
        return mx.sum(sim * weights[None, :], axis=1) / denom

    k = min(int(top_k), sim.shape[1])
    if k <= 0:
        raise ValueError("top_k must be positive")
    masked = mx.where(weights[None, :] > 0, sim, mx.array(-1.0e9, dtype=sim.dtype))
    values = mx.topk(masked, k, axis=1)
    valid = values > -1.0e8
    denom = mx.maximum(mx.sum(valid.astype(mx.float32), axis=1), mx.array(1.0, dtype=sim.dtype))
    return mx.sum(mx.where(valid, values, mx.zeros_like(values)), axis=1) / denom


def align_cheese_pair(
    probe_atoms: Sequence[int],
    probe_coords: Any,
    reference_atoms: Sequence[int],
    reference_coords: Any,
    *,
    probe_charges: Sequence[float] | np.ndarray | None = None,
    reference_charges: Sequence[float] | np.ndarray | None = None,
    config: CheeseAlignmentConfig | None = None,
    probe_index: int = -1,
    reference_index: int = -1,
) -> CheeseAlignmentResult:
    """Overlay one probe conformer onto one reference conformer.

    This is an open ROCS/Roshambo-style search: principal-axis starts, proper
    signed/permuted axis rotations, optional wiggles/random rotations, and
    local rigid-body refinement against Gaussian shape overlap.
    """

    cfg = config or CheeseAlignmentConfig()
    _validate_alignment_config(cfg)

    probe_atom_array = np.asarray(probe_atoms, dtype=np.int64)
    ref_atom_array = np.asarray(reference_atoms, dtype=np.int64)
    probe_xyz = _as_coords_np(probe_coords, "probe_coords")
    ref_xyz = _as_coords_np(reference_coords, "reference_coords")
    if probe_atom_array.shape != (probe_xyz.shape[0],):
        raise ValueError("probe_atoms and probe_coords must have the same length")
    if ref_atom_array.shape != (ref_xyz.shape[0],):
        raise ValueError("reference_atoms and reference_coords must have the same length")

    probe_q = None if probe_charges is None else np.asarray(probe_charges, dtype=np.float64)
    ref_q = None if reference_charges is None else np.asarray(reference_charges, dtype=np.float64)
    if probe_q is not None and probe_q.shape != (probe_xyz.shape[0],):
        raise ValueError("probe_charges must have one value per probe atom")
    if ref_q is not None and ref_q.shape != (ref_xyz.shape[0],):
        raise ValueError("reference_charges must have one value per reference atom")

    probe_exp, probe_amp = _gaussian_parameters_np(
        probe_atom_array,
        gaussian_alpha=cfg.gaussian_alpha,
        vdw_scale=cfg.vdw_scale,
        default_radius=cfg.default_radius,
    )
    ref_exp, ref_amp = _gaussian_parameters_np(
        ref_atom_array,
        gaussian_alpha=cfg.gaussian_alpha,
        vdw_scale=cfg.vdw_scale,
        default_radius=cfg.default_radius,
    )
    probe_self = _gaussian_cross_overlap_np(probe_xyz, probe_exp, probe_amp, probe_xyz, probe_exp, probe_amp)
    ref_self = _gaussian_cross_overlap_np(ref_xyz, ref_exp, ref_amp, ref_xyz, ref_exp, ref_amp)

    probe_centroid, probe_axes = _weighted_principal_axes_np(probe_xyz, probe_amp)
    ref_centroid, ref_axes = _weighted_principal_axes_np(ref_xyz, ref_amp)
    centered_probe = probe_xyz - probe_centroid[None, :]

    rotations = _alignment_start_rotations_np(probe_axes, ref_axes, cfg)
    deltas = np.zeros((len(rotations), 3), dtype=np.float64)
    if cfg.cocluster_starts > 0:
        cocluster_rot, cocluster_delta = _cocluster_start_transforms_np(
            probe_xyz,
            ref_xyz,
            probe_centroid,
            ref_centroid,
            probe_exp,
            probe_amp,
            ref_exp,
            ref_amp,
            cfg,
        )
        if len(cocluster_rot):
            rotations = np.concatenate([rotations, cocluster_rot], axis=0)
            deltas = np.concatenate([deltas, cocluster_delta], axis=0)
    start_scores = _shape_scores_for_rotations(
        centered_probe,
        ref_xyz,
        rotations,
        deltas,
        ref_centroid,
        probe_exp,
        probe_amp,
        ref_exp,
        ref_amp,
        probe_self,
        ref_self,
        cfg,
    )
    if cfg.max_initial_starts is not None and len(rotations) > int(cfg.max_initial_starts):
        keep = np.argsort(start_scores)[::-1][: int(cfg.max_initial_starts)]
        rotations = rotations[keep]
        deltas = deltas[keep]
        start_scores = start_scores[keep]

    if cfg.local_refine:
        refine_order = np.argsort(start_scores)[::-1][: min(int(cfg.refine_top_k), len(rotations))]
    else:
        refine_order = np.argsort(start_scores)[::-1][:1]

    charges_available = probe_q is not None and ref_q is not None
    probe_q_self = (
        _espsim_charge_overlap_np(probe_xyz, probe_q, probe_xyz, probe_q)
        if charges_available
        else 0.0
    )
    ref_q_self = (
        _espsim_charge_overlap_np(ref_xyz, ref_q, ref_xyz, ref_q)
        if charges_available
        else 0.0
    )

    best_payload = None
    converged = True
    for idx in refine_order:
        rotation = rotations[int(idx)]
        delta = deltas[int(idx)].copy()
        if cfg.local_refine:
            rotation, delta, _, this_converged = _refine_alignment_np(
                centered_probe,
                ref_xyz,
                rotation,
                delta,
                ref_centroid,
                probe_exp,
                probe_amp,
                ref_exp,
                ref_amp,
                probe_self,
                ref_self,
                cfg,
                probe_q=probe_q,
                ref_q=ref_q,
                probe_q_self=probe_q_self,
                ref_q_self=ref_q_self,
            )
            converged = converged and this_converged

        aligned = centered_probe @ rotation + ref_centroid[None, :] + delta[None, :]
        shape_overlap = _gaussian_cross_overlap_np(aligned, probe_exp, probe_amp, ref_xyz, ref_exp, ref_amp)
        shape = _shape_tanimoto_from_overlap_np(shape_overlap, probe_self, ref_self)
        electrostatic = (
            _electrostatic_similarity_np(
                aligned,
                probe_q,
                ref_xyz,
                ref_q,
                probe_q_self,
                ref_q_self,
                metric=cfg.electrostatic_metric,
            )
            if charges_available
            else 0.0
        )
        combined = _combined_score_np(
            shape,
            electrostatic,
            shape_weight=cfg.shape_weight,
            electrostatic_weight=_effective_electrostatic_weight(cfg, charges_available),
            map_electrostatic_to_unit=cfg.map_electrostatic_to_unit,
            electrostatic_metric=cfg.electrostatic_metric,
        )
        objective = _alignment_objective_np(shape, electrostatic, cfg, charges_available)
        payload = (objective, combined, shape, electrostatic, shape_overlap, rotation, delta, aligned)
        if best_payload is None or payload[0] > best_payload[0]:
            best_payload = payload

    assert best_payload is not None
    _, combined, shape, electrostatic, shape_overlap, rotation, delta, aligned = best_payload
    translation = ref_centroid + delta - probe_centroid @ rotation
    return CheeseAlignmentResult(
        shape=float(shape),
        electrostatic=float(electrostatic),
        combined=float(combined),
        overlap=float(shape_overlap),
        probe_self_overlap=float(probe_self),
        reference_self_overlap=float(ref_self),
        rotation=np.asarray(rotation, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64),
        aligned_coords=np.asarray(probe_xyz @ rotation + translation[None, :], dtype=np.float64),
        probe_index=int(probe_index),
        reference_index=int(reference_index),
        n_starts=int(len(rotations)),
        n_refined=int(len(refine_order)),
        converged=bool(converged),
    )


def cheese_alignment_matrix(
    probe: CheeseBatch,
    reference: CheeseBatch,
    *,
    config: CheeseAlignmentConfig | None = None,
    return_alignments: bool = False,
) -> CheeseSimilarityResult | tuple[CheeseSimilarityResult, list[list[CheeseAlignmentResult]]]:
    """Align every probe conformer/molecule to every reference and score it."""

    probe_mols = [_unpack_cheese_batch_molecule(probe, i) for i in range(int(probe.coords.shape[0]))]
    ref_mols = [_unpack_cheese_batch_molecule(reference, i) for i in range(int(reference.coords.shape[0]))]
    shape = np.zeros((len(probe_mols), len(ref_mols)), dtype=np.float32)
    electrostatic = np.zeros_like(shape)
    combined = np.zeros_like(shape)
    alignments: list[list[CheeseAlignmentResult]] = []
    for i, (probe_atoms, probe_xyz, probe_q) in enumerate(probe_mols):
        row: list[CheeseAlignmentResult] = []
        for j, (ref_atoms, ref_xyz, ref_q) in enumerate(ref_mols):
            result = align_cheese_pair(
                probe_atoms,
                probe_xyz,
                ref_atoms,
                ref_xyz,
                probe_charges=probe_q,
                reference_charges=ref_q,
                config=config,
                probe_index=i,
                reference_index=j,
            )
            shape[i, j] = result.shape
            electrostatic[i, j] = result.electrostatic
            combined[i, j] = result.combined
            row.append(result)
        alignments.append(row)

    scores = CheeseSimilarityResult(
        shape=mx.array(shape),
        electrostatic=mx.array(electrostatic),
        combined=mx.array(combined),
    )
    return (scores, alignments) if return_alignments else scores


def kabsch_align(
    probe_coords: Any,
    reference_coords: Any,
    *,
    weights: Any | None = None,
) -> np.ndarray:
    """Return probe coordinates rigidly aligned to reference coordinates.

    This CPU helper is for known atom mappings, such as same-molecule conformer
    comparisons. Full shape-based alignment remains a separate search problem.
    """

    probe = np.asarray(probe_coords, dtype=np.float64)
    ref = np.asarray(reference_coords, dtype=np.float64)
    if probe.shape != ref.shape or probe.ndim != 2 or probe.shape[1] != 3:
        raise ValueError("probe_coords and reference_coords must both have shape (n_atoms, 3)")
    if weights is None:
        w = np.ones((probe.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (probe.shape[0],):
            raise ValueError("weights must have one value per atom")
    w = w / max(float(w.sum()), 1.0e-12)
    probe_centroid = np.sum(probe * w[:, None], axis=0)
    ref_centroid = np.sum(ref * w[:, None], axis=0)
    probe_centered = probe - probe_centroid[None, :]
    ref_centered = ref - ref_centroid[None, :]
    covariance = (probe_centered * w[:, None]).T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    return (probe_centered @ rotation) + ref_centroid[None, :]


def _horn_key_matrix_mlx(covariance: mx.array) -> mx.array:
    sxx = covariance[..., 0, 0]
    sxy = covariance[..., 0, 1]
    sxz = covariance[..., 0, 2]
    syx = covariance[..., 1, 0]
    syy = covariance[..., 1, 1]
    syz = covariance[..., 1, 2]
    szx = covariance[..., 2, 0]
    szy = covariance[..., 2, 1]
    szz = covariance[..., 2, 2]

    row0 = mx.stack([sxx + syy + szz, syz - szy, szx - sxz, sxy - syx], axis=-1)
    row1 = mx.stack([syz - szy, sxx - syy - szz, sxy + syx, szx + sxz], axis=-1)
    row2 = mx.stack([szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy], axis=-1)
    row3 = mx.stack([sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz], axis=-1)
    return mx.stack([row0, row1, row2, row3], axis=-2)


def _symmetric4_top_eigenvector_power_mlx(
    matrix: mx.array,
    *,
    n_iters: int = 32,
    eps: float = 1.0e-12,
) -> mx.array:
    """Return the top eigenvector of batched symmetric 4x4 matrices.

    This intentionally avoids MLX ``eigh``/``svd`` because those fall back to
    CPU on current Apple Silicon builds. A positive diagonal shift makes power
    iteration target the largest algebraic eigenvalue, and multiple fixed
    starts avoid the rare orthogonal-start failure mode.
    """

    mat = _as_mx_float(matrix)
    starts = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=mat.dtype,
    )
    starts = starts / mx.sqrt(mx.sum(starts * starts, axis=-1, keepdims=True))
    q = mx.broadcast_to(starts, mat.shape[:-2] + starts.shape)

    eye = mx.eye(4, dtype=mat.dtype)
    shift = mx.max(mx.sum(mx.abs(mat), axis=-1), axis=-1) + mx.array(1.0, dtype=mat.dtype)
    shifted = mat + shift[..., None, None] * eye

    for _ in range(int(n_iters)):
        q = mx.matmul(shifted[..., None, :, :], q[..., :, None])[..., 0]
        q = q / mx.sqrt(mx.maximum(mx.sum(q * q, axis=-1, keepdims=True), mx.array(float(eps), dtype=mat.dtype)))

    kq = mx.matmul(mat[..., None, :, :], q[..., :, None])[..., 0]
    rayleigh = mx.sum(q * kq, axis=-1)
    best = mx.argmax(rayleigh, axis=-1)
    selector = (mx.arange(starts.shape[0]) == best[..., None]).astype(mat.dtype)
    out = mx.sum(q * selector[..., None], axis=-2)
    return out / mx.sqrt(mx.maximum(mx.sum(out * out, axis=-1, keepdims=True), mx.array(float(eps), dtype=mat.dtype)))


def _quaternion_to_row_rotation_mlx(quaternion: mx.array) -> mx.array:
    q = _as_mx_float(quaternion)
    q = q / mx.sqrt(mx.maximum(mx.sum(q * q, axis=-1, keepdims=True), mx.array(1.0e-12, dtype=q.dtype)))
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    row0 = mx.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        axis=-1,
    )
    row1 = mx.stack(
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        axis=-1,
    )
    row2 = mx.stack(
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    )
    return mx.stack([row0, row1, row2], axis=-2)


def horn_align_pairwise_mlx(
    probe_coords: Any,
    reference_coords: Any,
    *,
    weights: Any | None = None,
    power_iters: int = 32,
    eps: float = 1.0e-8,
) -> tuple[mx.array, mx.array, mx.array]:
    """Align every probe conformer to every reference conformer on GPU.

    Parameters
    ----------
    probe_coords
        ``(P, N, 3)`` coordinates.
    reference_coords
        ``(R, N, 3)`` coordinates with the same atom ordering.
    weights
        Optional ``(N,)`` atom weights. Passing heavy-atom weights reproduces
        the RMSD convention used by the RDKit/MLX conformer comparison script.

    Returns
    -------
    aligned, rmsd, rotations
        ``aligned`` has shape ``(P, R, N, 3)``, ``rmsd`` has shape ``(P, R)``,
        and ``rotations`` has shape ``(P, R, 3, 3)``.
    """

    probe = _as_mx_float(probe_coords)
    reference = _as_mx_float(reference_coords)
    if probe.ndim != 3 or reference.ndim != 3 or probe.shape[-1] != 3 or reference.shape[-1] != 3:
        raise ValueError("probe_coords and reference_coords must have shape (n_confs, n_atoms, 3)")
    if probe.shape[1] != reference.shape[1]:
        raise ValueError("probe and reference conformers must have the same atom count")
    n_atoms = probe.shape[1]
    if weights is None:
        atom_weights = mx.ones((n_atoms,), dtype=probe.dtype)
    else:
        atom_weights = _as_mx_float(weights)
        if atom_weights.shape != (n_atoms,):
            raise ValueError("weights must have shape (n_atoms,)")
    atom_weights = atom_weights / mx.maximum(mx.sum(atom_weights), mx.array(float(eps), dtype=probe.dtype))

    probe_centroid = mx.sum(probe * atom_weights[None, :, None], axis=1)
    ref_centroid = mx.sum(reference * atom_weights[None, :, None], axis=1)
    probe_centered = probe - probe_centroid[:, None, :]
    ref_centered = reference - ref_centroid[:, None, :]

    weighted_probe = probe_centered * atom_weights[None, :, None]
    covariance_rp = mx.matmul(mx.transpose(ref_centered, (0, 2, 1))[:, None, :, :], weighted_probe[None, :, :, :])
    covariance = mx.transpose(covariance_rp, (1, 0, 2, 3))
    horn_key = _horn_key_matrix_mlx(covariance)
    quat = _symmetric4_top_eigenvector_power_mlx(horn_key, n_iters=power_iters, eps=eps)
    rotations = _quaternion_to_row_rotation_mlx(quat)

    aligned = mx.matmul(probe_centered[:, None, :, :], rotations) + ref_centroid[None, :, None, :]
    diff = aligned - reference[None, :, :, :]
    rmsd = mx.sqrt(mx.sum(mx.sum(diff * diff, axis=-1) * atom_weights[None, None, :], axis=-1))
    return aligned, rmsd, rotations


_PI = mx.array(np.pi, dtype=mx.float32)
_ESPSIM_GAUSS_A_NP = np.asarray(
    [
        [15.90600036, 3.95348310, 17.61453176],
        [3.95348310, 5.21580206, 1.91045387],
        [17.61453176, 1.91045387, 238.75820253],
    ],
    dtype=np.float64,
)
_ESPSIM_GAUSS_B_NP = np.asarray(
    [
        [-0.02495000, -0.04539319, -0.00247124],
        [-0.04539319, -0.25130000, -0.00258662],
        [-0.00247124, -0.00258662, -0.00130000],
    ],
    dtype=np.float64,
)


def _vdw_radius_table(default_radius: float = 1.80) -> mx.array:
    max_z = max(118, max(VDW_RADII))
    table = np.full((max_z + 1,), float(default_radius), dtype=np.float32)
    table[0] = 0.0
    for z, radius in VDW_RADII.items():
        if z <= max_z:
            table[int(z)] = float(radius)
    return mx.array(table)


def _rdkit_vdw_radius_table(default_radius: float = 1.80) -> mx.array:
    max_z = 118
    table = np.full((max_z + 1,), float(default_radius), dtype=np.float32)
    table[0] = 0.0
    try:
        from rdkit.Chem import GetPeriodicTable

        periodic_table = GetPeriodicTable()
        for z in range(1, max_z + 1):
            table[z] = float(periodic_table.GetRvdw(z))
    except ImportError:  # pragma: no cover - RDKit is present in the normal package env.
        for z, radius in VDW_RADII.items():
            if z <= max_z:
                table[int(z)] = float(radius)
    return mx.array(table)


def _gaussian_overlap_matrix(
    coords_a: Any,
    exp_a: Any,
    amp_a: Any,
    mask_a: Any,
    coords_b: Any,
    exp_b: Any,
    amp_b: Any,
    mask_b: Any,
) -> mx.array:
    xyz_a = _as_mx_float(coords_a)
    xyz_b = _as_mx_float(coords_b)
    ea = _as_mx_float(exp_a)
    eb = _as_mx_float(exp_b)
    aa = _as_mx_float(amp_a)
    ab = _as_mx_float(amp_b)
    ma = _as_mx_float(mask_a)
    mb = _as_mx_float(mask_b)

    exp_sum = ea[:, None, :, None] + eb[None, :, None, :]
    mixed = ea[:, None, :, None] * eb[None, :, None, :] / mx.maximum(exp_sum, 1.0e-12)
    delta = xyz_a[:, None, :, None, :] - xyz_b[None, :, None, :, :]
    dist2 = mx.sum(delta * delta, axis=-1)
    prefactor = (_PI / mx.maximum(exp_sum, 1.0e-12)) ** 1.5
    pair = aa[:, None, :, None] * ab[None, :, None, :] * prefactor * mx.exp(-mixed * dist2)
    pair = pair * ma[:, None, :, None] * mb[None, :, None, :]
    return mx.sum(mx.sum(pair, axis=-1), axis=-1)


def _gaussian_paired_overlap(
    coords_a: Any,
    exp_a: Any,
    amp_a: Any,
    mask_a: Any,
    coords_b: Any,
    exp_b: Any,
    amp_b: Any,
    mask_b: Any,
) -> mx.array:
    xyz_a = _as_mx_float(coords_a)
    xyz_b = _as_mx_float(coords_b)
    ea = _as_mx_float(exp_a)
    eb = _as_mx_float(exp_b)
    aa = _as_mx_float(amp_a)
    ab = _as_mx_float(amp_b)
    ma = _as_mx_float(mask_a)
    mb = _as_mx_float(mask_b)

    exp_sum = ea[:, :, None] + eb[:, None, :]
    mixed = ea[:, :, None] * eb[:, None, :] / mx.maximum(exp_sum, 1.0e-12)
    delta = xyz_a[:, :, None, :] - xyz_b[:, None, :, :]
    dist2 = mx.sum(delta * delta, axis=-1)
    prefactor = (_PI / mx.maximum(exp_sum, 1.0e-12)) ** 1.5
    pair = aa[:, :, None] * ab[:, None, :] * prefactor * mx.exp(-mixed * dist2)
    pair = pair * ma[:, :, None] * mb[:, None, :]
    return mx.sum(mx.sum(pair, axis=-1), axis=-1)


def _gaussian_self_overlap(coords: Any, exponent: Any, amplitude: Any, mask: Any) -> mx.array:
    xyz = _as_mx_float(coords)
    expn = _as_mx_float(exponent)
    amp = _as_mx_float(amplitude)
    atom_mask = _as_mx_float(mask)

    exp_sum = expn[:, :, None] + expn[:, None, :]
    mixed = expn[:, :, None] * expn[:, None, :] / mx.maximum(exp_sum, 1.0e-12)
    delta = xyz[:, :, None, :] - xyz[:, None, :, :]
    dist2 = mx.sum(delta * delta, axis=-1)
    prefactor = (_PI / mx.maximum(exp_sum, 1.0e-12)) ** 1.5
    pair = amp[:, :, None] * amp[:, None, :] * prefactor * mx.exp(-mixed * dist2)
    pair = pair * atom_mask[:, :, None] * atom_mask[:, None, :]
    return mx.sum(mx.sum(pair, axis=-1), axis=-1)


def _espsim_coefficients_mlx(dtype: Any = mx.float32) -> tuple[mx.array, mx.array]:
    return mx.array(_ESPSIM_GAUSS_A_NP.reshape(-1), dtype=dtype), mx.array(
        _ESPSIM_GAUSS_B_NP.reshape(-1), dtype=dtype
    )


def _espsim_charge_overlap_matrix(
    coords_a: Any,
    charges_a: Any,
    mask_a: Any,
    coords_b: Any,
    charges_b: Any,
    mask_b: Any,
) -> mx.array:
    xyz_a = _as_mx_float(coords_a)
    xyz_b = _as_mx_float(coords_b)
    qa = _as_mx_float(charges_a)
    qb = _as_mx_float(charges_b)
    ma = _as_mx_float(mask_a)
    mb = _as_mx_float(mask_b)
    coeff_a, coeff_b = _espsim_coefficients_mlx(xyz_a.dtype)

    delta = xyz_a[:, None, :, None, :] - xyz_b[None, :, None, :, :]
    dist2 = mx.sum(delta * delta, axis=-1)
    kernel = mx.sum(coeff_a[None, None, None, None, :] * mx.exp(dist2[..., None] * coeff_b), axis=-1)
    pair = qa[:, None, :, None] * qb[None, :, None, :] * kernel
    pair = pair * ma[:, None, :, None] * mb[None, :, None, :]
    return mx.sum(mx.sum(pair, axis=-1), axis=-1)


def _espsim_charge_paired_overlap(
    coords_a: Any,
    charges_a: Any,
    mask_a: Any,
    coords_b: Any,
    charges_b: Any,
    mask_b: Any,
) -> mx.array:
    xyz_a = _as_mx_float(coords_a)
    xyz_b = _as_mx_float(coords_b)
    qa = _as_mx_float(charges_a)
    qb = _as_mx_float(charges_b)
    ma = _as_mx_float(mask_a)
    mb = _as_mx_float(mask_b)
    coeff_a, coeff_b = _espsim_coefficients_mlx(xyz_a.dtype)

    delta = xyz_a[:, :, None, :] - xyz_b[:, None, :, :]
    dist2 = mx.sum(delta * delta, axis=-1)
    kernel = mx.sum(coeff_a[None, None, None, :] * mx.exp(dist2[..., None] * coeff_b), axis=-1)
    pair = qa[:, :, None] * qb[:, None, :] * kernel
    pair = pair * ma[:, :, None] * mb[:, None, :]
    return mx.sum(mx.sum(pair, axis=-1), axis=-1)


def _espsim_charge_self_overlap(coords: Any, charges: Any, mask: Any) -> mx.array:
    xyz = _as_mx_float(coords)
    q = _as_mx_float(charges)
    atom_mask = _as_mx_float(mask)
    coeff_a, coeff_b = _espsim_coefficients_mlx(xyz.dtype)

    delta = xyz[:, :, None, :] - xyz[:, None, :, :]
    dist2 = mx.sum(delta * delta, axis=-1)
    kernel = mx.sum(coeff_a[None, None, None, :] * mx.exp(dist2[..., None] * coeff_b), axis=-1)
    pair = q[:, :, None] * q[:, None, :] * kernel
    pair = pair * atom_mask[:, :, None] * atom_mask[:, None, :]
    return mx.sum(mx.sum(pair, axis=-1), axis=-1)


def _batched_distance_from_grid_to_atoms(grid: mx.array, coords: mx.array, *, min_distance: float) -> mx.array:
    """Return distances for shared grid ``(1, G, 3)`` and coords ``(B, N, 3)``."""

    grid_norm = mx.sum(grid * grid, axis=-1)[:, :, None]
    atom_norm = mx.sum(coords * coords, axis=-1)[:, None, :]
    dot = mx.matmul(grid, mx.transpose(coords, (0, 2, 1)))
    dist2 = mx.maximum(grid_norm + atom_norm - 2.0 * dot, 0.0)
    return mx.sqrt(dist2 + float(min_distance) ** 2)


def _batched_distance_from_template_grid_to_atoms(
    grid: mx.array,
    coords: mx.array,
    *,
    min_distance: float,
) -> mx.array:
    """Return distances for template grids ``(T, G, 3)`` and coords ``(B, N, 3)``."""

    grid_norm = mx.sum(grid * grid, axis=-1)[None, :, :, None]
    atom_norm = mx.sum(coords * coords, axis=-1)[:, None, None, :]
    dot = mx.matmul(grid[None, :, :, :], mx.transpose(coords[:, None, :, :], (0, 1, 3, 2)))
    dist2 = mx.maximum(grid_norm + atom_norm - 2.0 * dot, 0.0)
    return mx.sqrt(dist2 + float(min_distance) ** 2)


def _esp_similarity_from_sums(
    dot: mx.array,
    probe_norm2: mx.array,
    ref_norm2: mx.array,
    *,
    metric: str,
    eps: float,
) -> mx.array:
    if metric == "carbo":
        denom = mx.sqrt(
            mx.maximum(probe_norm2, mx.array(float(eps), dtype=dot.dtype))
            * mx.maximum(ref_norm2, mx.array(float(eps), dtype=dot.dtype))
        )
        return mx.clip(mx.where(denom > float(eps), dot / denom, mx.zeros_like(dot)), -1.0, 1.0)
    if metric == "tanimoto":
        denom = probe_norm2 + ref_norm2 - dot
        return dot / mx.maximum(denom, mx.array(float(eps), dtype=dot.dtype))
    raise ValueError("metric must be 'carbo' or 'tanimoto'")


def _renormalize_electrostatic_mlx(score: mx.array, metric: str) -> mx.array:
    metric = metric.lower()
    if metric == "carbo":
        return (score + 1.0) * 0.5
    if metric == "tanimoto":
        return (score + (1.0 / 3.0)) * 0.75
    raise ValueError("metric must be 'carbo' or 'tanimoto'")


def _validate_alignment_config(config: CheeseAlignmentConfig) -> None:
    if config.start_mode not in {"principal", "roshambo", "rotate_45"}:
        raise ValueError("start_mode must be 'principal', 'roshambo', or 'rotate_45'")
    if config.refine_top_k <= 0:
        raise ValueError("refine_top_k must be positive")
    if config.max_initial_starts is not None and config.max_initial_starts <= 0:
        raise ValueError("max_initial_starts must be positive or None")
    if config.max_refine_steps < 0:
        raise ValueError("max_refine_steps must be non-negative")
    if config.rotation_step_degrees <= 0 or config.translation_step <= 0:
        raise ValueError("initial refinement steps must be positive")
    if config.min_rotation_step_degrees <= 0 or config.min_translation_step <= 0:
        raise ValueError("minimum refinement steps must be positive")
    if config.shape_weight < 0:
        raise ValueError("shape_weight must be non-negative")
    if config.electrostatic_weight is not None and config.electrostatic_weight < 0:
        raise ValueError("electrostatic_weight must be non-negative")
    if config.optimize not in {"shape", "combined"}:
        raise ValueError("optimize must be 'shape' or 'combined'")
    if config.electrostatic_metric.lower() not in {"carbo", "tanimoto"}:
        raise ValueError("electrostatic_metric must be 'carbo' or 'tanimoto'")
    if config.start_score_backend not in {"auto", "metal", "numpy"}:
        raise ValueError("start_score_backend must be 'auto', 'metal', or 'numpy'")
    if config.cocluster_starts < 0:
        raise ValueError("cocluster_starts must be non-negative")
    if config.cocluster_pair_pool <= 0 or config.cocluster_neighbors <= 0:
        raise ValueError("cocluster_pair_pool and cocluster_neighbors must be positive")
    if config.cocluster_distance_sigma <= 0:
        raise ValueError("cocluster_distance_sigma must be positive")
    if config.cocluster_offdiag_weight < 0:
        raise ValueError("cocluster_offdiag_weight must be non-negative")


def _unpack_cheese_batch_molecule(batch: CheeseBatch, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(batch.mask, dtype=np.float32)[index] > 0
    atoms = np.asarray(batch.atomic_numbers, dtype=np.int64)[index, mask]
    coords = np.asarray(batch.coords, dtype=np.float64)[index, mask]
    charges = np.asarray(batch.charges, dtype=np.float64)[index, mask]
    return atoms, coords, charges


def _as_coords_np(coords: Any, name: str) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_atoms, 3), got {out.shape}")
    return out


def _gaussian_parameters_np(
    atoms: np.ndarray,
    *,
    gaussian_alpha: float,
    vdw_scale: float,
    default_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.asarray([VDW_RADII.get(int(z), float(default_radius)) for z in atoms], dtype=np.float64)
    radii = np.maximum(radii * float(vdw_scale), 1.0e-4)
    exponent = float(gaussian_alpha) / (radii * radii)
    volume = (4.0 / 3.0) * np.pi * radii**3
    amplitude = volume * (exponent / np.pi) ** 1.5
    return exponent, amplitude


def _gaussian_cross_overlap_np(
    coords_a: np.ndarray,
    exp_a: np.ndarray,
    amp_a: np.ndarray,
    coords_b: np.ndarray,
    exp_b: np.ndarray,
    amp_b: np.ndarray,
) -> float:
    exp_sum = exp_a[:, None] + exp_b[None, :]
    mixed = exp_a[:, None] * exp_b[None, :] / np.maximum(exp_sum, 1.0e-12)
    delta = coords_a[:, None, :] - coords_b[None, :, :]
    dist2 = np.sum(delta * delta, axis=-1)
    prefactor = (np.pi / np.maximum(exp_sum, 1.0e-12)) ** 1.5
    pair = amp_a[:, None] * amp_b[None, :] * prefactor * np.exp(-mixed * dist2)
    return float(np.sum(pair))


def _shape_tanimoto_from_overlap_np(overlap: float, self_a: float, self_b: float) -> float:
    denom = max(float(self_a) + float(self_b) - float(overlap), 1.0e-12)
    return float(np.clip(float(overlap) / denom, 0.0, 1.0))


def _electrostatic_carbo_np(
    probe_coords: np.ndarray,
    probe_charges: np.ndarray | None,
    ref_coords: np.ndarray,
    ref_charges: np.ndarray | None,
    probe_q_self: float,
    ref_q_self: float,
) -> float:
    return _electrostatic_similarity_np(
        probe_coords,
        probe_charges,
        ref_coords,
        ref_charges,
        probe_q_self,
        ref_q_self,
        metric="carbo",
    )


def _espsim_charge_overlap_np(
    coords_a: np.ndarray,
    charges_a: np.ndarray,
    coords_b: np.ndarray,
    charges_b: np.ndarray,
) -> float:
    dist2 = np.sum((coords_a[:, None, :] - coords_b[None, :, :]) ** 2, axis=-1)
    kernel = np.sum(
        _ESPSIM_GAUSS_A_NP.reshape(-1)[:, None, None]
        * np.exp(_ESPSIM_GAUSS_B_NP.reshape(-1)[:, None, None] * dist2[None, :, :]),
        axis=0,
    )
    return float(np.sum(charges_a[:, None] * charges_b[None, :] * kernel))


def _electrostatic_similarity_np(
    probe_coords: np.ndarray,
    probe_charges: np.ndarray | None,
    ref_coords: np.ndarray,
    ref_charges: np.ndarray | None,
    probe_q_self: float,
    ref_q_self: float,
    *,
    metric: str,
) -> float:
    if probe_charges is None or ref_charges is None:
        return 0.0
    cross = _espsim_charge_overlap_np(probe_coords, probe_charges, ref_coords, ref_charges)
    metric = metric.lower()
    if metric == "carbo":
        denom = np.sqrt(max(float(probe_q_self), 1.0e-12) * max(float(ref_q_self), 1.0e-12))
        if denom <= 1.0e-12:
            return 0.0
        return float(np.clip(cross / denom, -1.0, 1.0))
    if metric == "tanimoto":
        denom = max(float(probe_q_self) + float(ref_q_self) - float(cross), 1.0e-12)
        return float(cross / denom)
    raise ValueError("metric must be 'carbo' or 'tanimoto'")


def _renormalize_electrostatic_np(score: float, metric: str) -> float:
    metric = metric.lower()
    if metric == "carbo":
        return 0.5 * (float(score) + 1.0)
    if metric == "tanimoto":
        return 0.75 * (float(score) + (1.0 / 3.0))
    raise ValueError("metric must be 'carbo' or 'tanimoto'")


def _combined_score_np(
    shape: float,
    electrostatic: float,
    *,
    shape_weight: float,
    electrostatic_weight: float,
    map_electrostatic_to_unit: bool,
    electrostatic_metric: str,
) -> float:
    total_weight = float(shape_weight) + float(electrostatic_weight)
    if total_weight <= 0:
        raise ValueError("at least one alignment score weight must be positive")
    esp_channel = (
        _renormalize_electrostatic_np(electrostatic, electrostatic_metric)
        if map_electrostatic_to_unit
        else float(electrostatic)
    )
    return float((float(shape_weight) * float(shape) + float(electrostatic_weight) * esp_channel) / total_weight)


def _effective_electrostatic_weight(config: CheeseAlignmentConfig, charges_available: bool) -> float:
    if config.electrostatic_weight is not None:
        return float(config.electrostatic_weight)
    return 1.0 if charges_available else 0.0


def _alignment_objective_np(
    shape: float,
    electrostatic: float,
    config: CheeseAlignmentConfig,
    charges_available: bool,
) -> float:
    if config.optimize == "combined":
        return _combined_score_np(
            shape,
            electrostatic,
            shape_weight=config.shape_weight,
            electrostatic_weight=_effective_electrostatic_weight(config, charges_available),
            map_electrostatic_to_unit=config.map_electrostatic_to_unit,
            electrostatic_metric=config.electrostatic_metric,
        )
    return float(shape)


def _weighted_principal_axes_np(coords: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    total = max(float(np.sum(weights)), 1.0e-12)
    centroid = np.sum(coords * weights[:, None], axis=0) / total
    centered = coords - centroid[None, :]
    cov = (centered * weights[:, None]).T @ centered / total
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return centroid, axes


def _alignment_start_rotations_np(
    probe_axes: np.ndarray,
    ref_axes: np.ndarray,
    config: CheeseAlignmentConfig,
) -> np.ndarray:
    starts = []
    for symmetry in _proper_signed_permutation_matrices_np():
        base = probe_axes @ symmetry @ ref_axes.T
        starts.append(base)
        if config.start_mode in {"roshambo", "rotate_45"}:
            starts.extend(base @ wiggle for wiggle in _wiggle_rotations_np(config.wiggle_degrees))
        if config.start_mode == "rotate_45":
            starts.extend(base @ wiggle for wiggle in _rotate_45_pair_wiggles_np())

    if config.random_starts > 0:
        rng = np.random.default_rng(int(config.random_seed))
        starts.extend(_random_rotation_np(rng) for _ in range(int(config.random_starts)))

    unique = []
    seen = set()
    for rot in starts:
        if np.linalg.det(rot) < 0:
            continue
        key = tuple(np.round(rot.reshape(-1), 7))
        if key not in seen:
            seen.add(key)
            unique.append(rot.astype(np.float64, copy=False))
    return np.stack(unique, axis=0)


def _proper_signed_permutation_matrices_np() -> list[np.ndarray]:
    out = []
    for perm in _permutations3():
        p = np.zeros((3, 3), dtype=np.float64)
        for row, col in enumerate(perm):
            p[row, col] = 1.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    m = np.diag([sx, sy, sz]) @ p
                    if np.linalg.det(m) > 0:
                        out.append(m)
    return out


def _permutations3() -> tuple[tuple[int, int, int], ...]:
    return (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )


def _wiggle_rotations_np(degrees: float) -> list[np.ndarray]:
    angle = np.deg2rad(float(degrees))
    out = []
    for axis in range(3):
        out.append(_axis_rotation_np(axis, angle))
        out.append(_axis_rotation_np(axis, -angle))
    return out


def _rotate_45_pair_wiggles_np() -> list[np.ndarray]:
    angle = np.deg2rad(45.0)
    out = []
    for axis_a in range(3):
        for axis_b in range(axis_a + 1, 3):
            for sign_a in (-1.0, 1.0):
                for sign_b in (-1.0, 1.0):
                    out.append(_axis_rotation_np(axis_a, sign_a * angle) @ _axis_rotation_np(axis_b, sign_b * angle))
    return out


def _axis_rotation_np(axis: int, angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    if axis == 0:
        col = np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    elif axis == 1:
        col = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    elif axis == 2:
        col = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return col.T


def _random_rotation_np(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    col = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return col.T


def _cocluster_start_transforms_np(
    probe_coords: np.ndarray,
    ref_coords: np.ndarray,
    probe_centroid: np.ndarray,
    ref_centroid: np.ndarray,
    probe_exp: np.ndarray,
    probe_amp: np.ndarray,
    ref_exp: np.ndarray,
    ref_amp: np.ndarray,
    config: CheeseAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if probe_coords.shape[0] < 3 or ref_coords.shape[0] < 3:
        return np.empty((0, 3, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    pair_weight = _atom_pair_similarity_np(probe_exp, probe_amp, ref_exp, ref_amp)
    flat_order = np.argsort(pair_weight.reshape(-1))[::-1]
    n_pairs = min(int(config.cocluster_pair_pool), int(flat_order.size))
    if n_pairs < 3:
        return np.empty((0, 3, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    probe_idx = flat_order[:n_pairs] // ref_coords.shape[0]
    ref_idx = flat_order[:n_pairs] % ref_coords.shape[0]
    pair_scores = pair_weight[probe_idx, ref_idx]

    probe_dist = _pairwise_distances_np(probe_coords)
    ref_dist = _pairwise_distances_np(ref_coords)
    candidates = []
    sigma2 = max(float(config.cocluster_distance_sigma) ** 2, 1.0e-8)
    for anchor in range(n_pairs):
        i0 = int(probe_idx[anchor])
        j0 = int(ref_idx[anchor])
        compat = np.zeros((n_pairs,), dtype=np.float64)
        for other in range(n_pairs):
            i1 = int(probe_idx[other])
            j1 = int(ref_idx[other])
            if other == anchor or i1 == i0 or j1 == j0:
                compat[other] = -1.0
                continue
            dd = probe_dist[i0, i1] - ref_dist[j0, j1]
            compat[other] = pair_scores[other] * np.exp(-0.5 * dd * dd / sigma2)
        neighbors = np.argsort(compat)[::-1][: min(int(config.cocluster_neighbors), n_pairs)]
        neighbors = [int(x) for x in neighbors if compat[int(x)] > 0]
        for a_pos, first in enumerate(neighbors):
            for second in neighbors[a_pos + 1 :]:
                probe_triplet = np.asarray([i0, probe_idx[first], probe_idx[second]], dtype=np.int64)
                ref_triplet = np.asarray([j0, ref_idx[first], ref_idx[second]], dtype=np.int64)
                if len(set(probe_triplet.tolist())) < 3 or len(set(ref_triplet.tolist())) < 3:
                    continue
                if _triangle_area_np(probe_coords[probe_triplet]) < 1.0e-4:
                    continue
                if _triangle_area_np(ref_coords[ref_triplet]) < 1.0e-4:
                    continue
                score = (
                    _cocluster_diagonal_contrast_np(probe_triplet, ref_triplet, pair_weight, config.cocluster_offdiag_weight)
                    * _distance_consistency_np(probe_triplet, ref_triplet, probe_dist, ref_dist, sigma2)
                )
                if score <= 0:
                    continue
                weights = np.asarray([pair_scores[anchor], pair_scores[first], pair_scores[second]], dtype=np.float64)
                rotation, translation = _kabsch_transform_np(
                    probe_coords[probe_triplet],
                    ref_coords[ref_triplet],
                    weights=weights,
                )
                delta = translation - ref_centroid + probe_centroid @ rotation
                candidates.append((float(score), rotation, delta))

    if not candidates:
        return np.empty((0, 3, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    candidates.sort(key=lambda x: x[0], reverse=True)
    rotations = []
    deltas = []
    seen = set()
    for _, rotation, delta in candidates:
        key = tuple(np.round(np.concatenate([rotation.reshape(-1), delta]), 6))
        if key in seen:
            continue
        seen.add(key)
        rotations.append(rotation)
        deltas.append(delta)
        if len(rotations) >= int(config.cocluster_starts):
            break
    if not rotations:
        return np.empty((0, 3, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.stack(rotations, axis=0), np.stack(deltas, axis=0)


def _atom_pair_similarity_np(
    probe_exp: np.ndarray,
    probe_amp: np.ndarray,
    ref_exp: np.ndarray,
    ref_amp: np.ndarray,
) -> np.ndarray:
    probe_radius_like = 1.0 / np.sqrt(np.maximum(probe_exp, 1.0e-12))
    ref_radius_like = 1.0 / np.sqrt(np.maximum(ref_exp, 1.0e-12))
    radius_delta = probe_radius_like[:, None] - ref_radius_like[None, :]
    radius_sim = np.exp(-0.5 * (radius_delta / 0.20) ** 2)
    amp_sim = np.sqrt(np.maximum(probe_amp[:, None], 0.0) * np.maximum(ref_amp[None, :], 0.0))
    amp_sim = amp_sim / max(float(np.max(amp_sim)), 1.0e-12)
    return radius_sim * (0.25 + 0.75 * amp_sim)


def _cocluster_diagonal_contrast_np(
    probe_indices: np.ndarray,
    ref_indices: np.ndarray,
    pair_weight: np.ndarray,
    offdiag_weight: float,
) -> float:
    block = pair_weight[np.asarray(probe_indices, dtype=np.int64)[:, None], np.asarray(ref_indices, dtype=np.int64)[None, :]]
    diag = np.diag(block)
    if block.shape[0] <= 1:
        return float(np.mean(diag))
    offdiag_sum = float(np.sum(block) - np.sum(diag))
    offdiag_mean = offdiag_sum / float(block.size - len(diag))
    diag_mean = float(np.mean(diag))
    contrast = diag_mean - float(offdiag_weight) * offdiag_mean
    return float(max(contrast, 0.0) ** len(diag))


def _pairwise_distances_np(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))


def _triangle_area_np(coords: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0])))


def _distance_consistency_np(
    probe_triplet: np.ndarray,
    ref_triplet: np.ndarray,
    probe_dist: np.ndarray,
    ref_dist: np.ndarray,
    sigma2: float,
) -> float:
    score = 1.0
    for a in range(3):
        for b in range(a + 1, 3):
            dd = probe_dist[int(probe_triplet[a]), int(probe_triplet[b])] - ref_dist[int(ref_triplet[a]), int(ref_triplet[b])]
            score *= float(np.exp(-0.5 * dd * dd / sigma2))
    return score


def _kabsch_transform_np(
    probe_coords: np.ndarray,
    ref_coords: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        w = np.ones((probe_coords.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(np.sum(w)), 1.0e-12)
    probe_center = np.sum(probe_coords * w[:, None], axis=0)
    ref_center = np.sum(ref_coords * w[:, None], axis=0)
    probe_centered = probe_coords - probe_center[None, :]
    ref_centered = ref_coords - ref_center[None, :]
    covariance = (probe_centered * w[:, None]).T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    translation = ref_center - probe_center @ rotation
    return rotation, translation


def _shape_scores_for_rotations_np(
    centered_probe: np.ndarray,
    ref_coords: np.ndarray,
    rotations: np.ndarray,
    deltas: np.ndarray,
    ref_centroid: np.ndarray,
    probe_exp: np.ndarray,
    probe_amp: np.ndarray,
    ref_exp: np.ndarray,
    ref_amp: np.ndarray,
    probe_self: float,
    ref_self: float,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    scores = []
    for start in range(0, len(rotations), int(chunk_size)):
        rot = rotations[start : start + int(chunk_size)]
        shift = deltas[start : start + int(chunk_size)]
        aligned = np.einsum("ni,cij->cnj", centered_probe, rot) + ref_centroid[None, None, :] + shift[:, None, :]
        overlap = _gaussian_cross_overlap_many_np(aligned, probe_exp, probe_amp, ref_coords, ref_exp, ref_amp)
        denom = np.maximum(float(probe_self) + float(ref_self) - overlap, 1.0e-12)
        scores.append(np.clip(overlap / denom, 0.0, 1.0))
    return np.concatenate(scores).astype(np.float64, copy=False)


def _shape_scores_for_rotations(
    centered_probe: np.ndarray,
    ref_coords: np.ndarray,
    rotations: np.ndarray,
    deltas: np.ndarray,
    ref_centroid: np.ndarray,
    probe_exp: np.ndarray,
    probe_amp: np.ndarray,
    ref_exp: np.ndarray,
    ref_amp: np.ndarray,
    probe_self: float,
    ref_self: float,
    config: CheeseAlignmentConfig,
) -> np.ndarray:
    backend = config.start_score_backend
    if backend in {"auto", "metal"} and hasattr(mx.fast, "metal_kernel"):
        try:
            return _shape_scores_for_rotations_metal(
                centered_probe,
                ref_coords,
                rotations,
                deltas,
                ref_centroid,
                probe_exp,
                probe_amp,
                ref_exp,
                ref_amp,
                probe_self,
                ref_self,
            )
        except Exception:
            if backend == "metal":
                raise
    return _shape_scores_for_rotations_np(
        centered_probe,
        ref_coords,
        rotations,
        deltas,
        ref_centroid,
        probe_exp,
        probe_amp,
        ref_exp,
        ref_amp,
        probe_self,
        ref_self,
    )


def _shape_scores_for_rotations_metal(
    centered_probe: np.ndarray,
    ref_coords: np.ndarray,
    rotations: np.ndarray,
    deltas: np.ndarray,
    ref_centroid: np.ndarray,
    probe_exp: np.ndarray,
    probe_amp: np.ndarray,
    ref_exp: np.ndarray,
    ref_amp: np.ndarray,
    probe_self: float,
    ref_self: float,
) -> np.ndarray:
    n_starts = int(rotations.shape[0])
    if n_starts == 0:
        return np.empty((0,), dtype=np.float64)
    kernel = _get_shape_start_score_kernel()
    config = mx.array([float(probe_self), float(ref_self)], dtype=mx.float32)
    scores = kernel(
        inputs=[
            mx.array(np.asarray(centered_probe, dtype=np.float32)),
            mx.array(np.asarray(ref_coords, dtype=np.float32)),
            mx.array(np.asarray(rotations, dtype=np.float32)),
            mx.array(np.asarray(deltas, dtype=np.float32)),
            mx.array(np.asarray(ref_centroid, dtype=np.float32)),
            mx.array(np.asarray(probe_exp, dtype=np.float32)),
            mx.array(np.asarray(probe_amp, dtype=np.float32)),
            mx.array(np.asarray(ref_exp, dtype=np.float32)),
            mx.array(np.asarray(ref_amp, dtype=np.float32)),
            config,
        ],
        output_shapes=[(n_starts,)],
        output_dtypes=[mx.float32],
        grid=(n_starts, 1, 1),
        threadgroup=(min(256, n_starts), 1, 1),
    )[0]
    mx.eval(scores)
    return np.asarray(scores, dtype=np.float64)


def _gaussian_cross_overlap_many_np(
    coords_a: np.ndarray,
    exp_a: np.ndarray,
    amp_a: np.ndarray,
    coords_b: np.ndarray,
    exp_b: np.ndarray,
    amp_b: np.ndarray,
) -> np.ndarray:
    exp_sum = exp_a[None, :, None] + exp_b[None, None, :]
    mixed = exp_a[None, :, None] * exp_b[None, None, :] / np.maximum(exp_sum, 1.0e-12)
    delta = coords_a[:, :, None, :] - coords_b[None, None, :, :]
    dist2 = np.sum(delta * delta, axis=-1)
    prefactor = (np.pi / np.maximum(exp_sum, 1.0e-12)) ** 1.5
    pair = amp_a[None, :, None] * amp_b[None, None, :] * prefactor * np.exp(-mixed * dist2)
    return np.sum(pair, axis=(1, 2))


def _refine_alignment_np(
    centered_probe: np.ndarray,
    ref_coords: np.ndarray,
    rotation: np.ndarray,
    delta: np.ndarray,
    ref_centroid: np.ndarray,
    probe_exp: np.ndarray,
    probe_amp: np.ndarray,
    ref_exp: np.ndarray,
    ref_amp: np.ndarray,
    probe_self: float,
    ref_self: float,
    config: CheeseAlignmentConfig,
    *,
    probe_q: np.ndarray | None,
    ref_q: np.ndarray | None,
    probe_q_self: float,
    ref_q_self: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    charges_available = probe_q is not None and ref_q is not None
    rot = np.asarray(rotation, dtype=np.float64).copy()
    shift = np.asarray(delta, dtype=np.float64).copy()
    rot_step = np.deg2rad(float(config.rotation_step_degrees))
    trans_step = float(config.translation_step)
    min_rot = np.deg2rad(float(config.min_rotation_step_degrees))
    min_trans = float(config.min_translation_step)
    score = _alignment_score_for_transform_np(
        centered_probe,
        ref_coords,
        rot,
        shift,
        ref_centroid,
        probe_exp,
        probe_amp,
        ref_exp,
        ref_amp,
        probe_self,
        ref_self,
        config,
        probe_q=probe_q,
        ref_q=ref_q,
        probe_q_self=probe_q_self,
        ref_q_self=ref_q_self,
        charges_available=charges_available,
    )
    converged = False
    for _ in range(int(config.max_refine_steps)):
        candidates = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                candidates.append((rot @ _axis_rotation_np(axis, sign * rot_step), shift))
        for axis in range(3):
            for sign in (-1.0, 1.0):
                d = shift.copy()
                d[axis] += sign * trans_step
                candidates.append((rot, d))

        best_candidate = None
        best_score = score
        for cand_rot, cand_shift in candidates:
            cand_score = _alignment_score_for_transform_np(
                centered_probe,
                ref_coords,
                cand_rot,
                cand_shift,
                ref_centroid,
                probe_exp,
                probe_amp,
                ref_exp,
                ref_amp,
                probe_self,
                ref_self,
                config,
                probe_q=probe_q,
                ref_q=ref_q,
                probe_q_self=probe_q_self,
                ref_q_self=ref_q_self,
                charges_available=charges_available,
            )
            if cand_score > best_score + 1.0e-8:
                best_score = cand_score
                best_candidate = (cand_rot, cand_shift)
        if best_candidate is not None:
            rot, shift = best_candidate
            score = best_score
            continue
        rot_step *= 0.5
        trans_step *= 0.5
        if rot_step <= min_rot and trans_step <= min_trans:
            converged = True
            break
    return rot, shift, float(score), converged


def _alignment_score_for_transform_np(
    centered_probe: np.ndarray,
    ref_coords: np.ndarray,
    rotation: np.ndarray,
    delta: np.ndarray,
    ref_centroid: np.ndarray,
    probe_exp: np.ndarray,
    probe_amp: np.ndarray,
    ref_exp: np.ndarray,
    ref_amp: np.ndarray,
    probe_self: float,
    ref_self: float,
    config: CheeseAlignmentConfig,
    *,
    probe_q: np.ndarray | None,
    ref_q: np.ndarray | None,
    probe_q_self: float,
    ref_q_self: float,
    charges_available: bool,
) -> float:
    aligned = centered_probe @ rotation + ref_centroid[None, :] + delta[None, :]
    overlap = _gaussian_cross_overlap_np(aligned, probe_exp, probe_amp, ref_coords, ref_exp, ref_amp)
    shape = _shape_tanimoto_from_overlap_np(overlap, probe_self, ref_self)
    electrostatic = (
        _electrostatic_similarity_np(
            aligned,
            probe_q,
            ref_coords,
            ref_q,
            probe_q_self,
            ref_q_self,
            metric=config.electrostatic_metric,
        )
        if charges_available
        else 0.0
    )
    return _alignment_objective_np(shape, electrostatic, config, charges_available)


_SHAPE_START_SCORE_METAL_SOURCE = """
uint tid = thread_position_in_grid.x;
uint C = rotations_shape[0];
uint N = centered_probe_shape[0];
uint M = ref_coords_shape[0];
if (tid >= C) { return; }

uint r0 = tid * 9;
float r00 = rotations[r0 + 0];
float r01 = rotations[r0 + 1];
float r02 = rotations[r0 + 2];
float r10 = rotations[r0 + 3];
float r11 = rotations[r0 + 4];
float r12 = rotations[r0 + 5];
float r20 = rotations[r0 + 6];
float r21 = rotations[r0 + 7];
float r22 = rotations[r0 + 8];
uint d0 = tid * 3;
float tx = deltas[d0 + 0];
float ty = deltas[d0 + 1];
float tz = deltas[d0 + 2];
float cx = ref_centroid[0];
float cy = ref_centroid[1];
float cz = ref_centroid[2];
const float PI = 3.14159265358979323846f;
float overlap = 0.0f;

for (uint i = 0; i < N; i++) {
  uint pi = i * 3;
  float px = centered_probe[pi + 0];
  float py = centered_probe[pi + 1];
  float pz = centered_probe[pi + 2];
  // Row-vector convention: aligned = centered_probe @ rotation + ref_centroid + delta.
  float ax = px * r00 + py * r10 + pz * r20 + cx + tx;
  float ay = px * r01 + py * r11 + pz * r21 + cy + ty;
  float az = px * r02 + py * r12 + pz * r22 + cz + tz;
  float ea = probe_exp[i];
  float aa = probe_amp[i];
  for (uint j = 0; j < M; j++) {
    uint rj = j * 3;
    float dx = ax - ref_coords[rj + 0];
    float dy = ay - ref_coords[rj + 1];
    float dz = az - ref_coords[rj + 2];
    float eb = ref_exp[j];
    float exp_sum = max(ea + eb, 1.0e-12f);
    float mixed = ea * eb / exp_sum;
    float dist2 = dx * dx + dy * dy + dz * dz;
    float prefactor = pow(PI / exp_sum, 1.5f);
    overlap += aa * ref_amp[j] * prefactor * exp(-mixed * dist2);
  }
}

float denom = max(config[0] + config[1] - overlap, 1.0e-12f);
float score = overlap / denom;
out[tid] = clamp(score, 0.0f, 1.0f);
"""

_shape_start_score_kernel = None


def _get_shape_start_score_kernel():
    global _shape_start_score_kernel
    if _shape_start_score_kernel is None:
        _shape_start_score_kernel = mx.fast.metal_kernel(
            name="cheese_shape_start_scores",
            input_names=[
                "centered_probe",
                "ref_coords",
                "rotations",
                "deltas",
                "ref_centroid",
                "probe_exp",
                "probe_amp",
                "ref_exp",
                "ref_amp",
                "config",
            ],
            output_names=["out"],
            source=_SHAPE_START_SCORE_METAL_SOURCE,
            ensure_row_contiguous=True,
        )
    return _shape_start_score_kernel


_ESP_GRID_METAL_SOURCE = """
uint tid = thread_position_in_grid.x;
uint B = coords_shape[0];
uint N = coords_shape[1];
uint G = grid_shape[0];
if (tid >= B * G) { return; }

uint b = tid / G;
uint g = tid - b * G;
float gx = grid[g * 3 + 0];
float gy = grid[g * 3 + 1];
float gz = grid[g * 3 + 2];
float min_d = config[0];
float min_d2 = min_d * min_d;
float value = 0.0f;

for (uint a = 0; a < N; a++) {
  uint atom_idx = b * N + a;
  float m = mask[atom_idx];
  if (m > 0.0f) {
    uint coord_idx = atom_idx * 3;
    float dx = gx - coords[coord_idx + 0];
    float dy = gy - coords[coord_idx + 1];
    float dz = gz - coords[coord_idx + 2];
    float r2 = dx * dx + dy * dy + dz * dz + min_d2;
    value += charges[atom_idx] * m * rsqrt(r2);
  }
}
out[tid] = value;
"""

_esp_grid_kernel = None


def _get_esp_grid_kernel():
    global _esp_grid_kernel
    if _esp_grid_kernel is None:
        _esp_grid_kernel = mx.fast.metal_kernel(
            name="cheese_esp_grid_shared",
            input_names=["coords", "charges", "mask", "grid", "config"],
            output_names=["out"],
            source=_ESP_GRID_METAL_SOURCE,
            ensure_row_contiguous=True,
        )
    return _esp_grid_kernel


def _as_batched_coords_charges(coords: Any, charges: Any, mask: Any | None) -> tuple[mx.array, mx.array, mx.array]:
    xyz = _as_mx_float(coords)
    q = _as_mx_float(charges)
    if xyz.ndim == 2:
        xyz = xyz[None, :, :]
    if q.ndim == 1:
        q = q[None, :]
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"coords must have shape (batch, n_atoms, 3), got {xyz.shape}")
    if q.shape != xyz.shape[:2]:
        raise ValueError(f"charges shape {q.shape} does not match coords shape {xyz.shape[:2]}")
    if mask is None:
        atom_mask = mx.ones(q.shape, dtype=mx.float32)
    else:
        atom_mask = _as_mx_float(mask)
        if atom_mask.ndim == 1:
            atom_mask = atom_mask[None, :]
        if atom_mask.shape != q.shape:
            raise ValueError("mask must match charges")
    return xyz, q, atom_mask


def _as_mx_float(value: Any) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    return mx.array(value, dtype=mx.float32)


def _as_mx_int(value: Any) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.int32)
    return mx.array(value, dtype=mx.int32)
