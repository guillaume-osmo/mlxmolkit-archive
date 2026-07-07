from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ChargeModelConfig:
    """Configuration for the CHEESE-style learned charge predictor."""

    max_atomic_number: int = 118
    n_bond_states: int = 13
    hidden_dim: int = 128
    bond_dim: int = 16
    n_layers: int = 4
    n_heads: int = 4
    n_rbf: int = 32
    rbf_min: float = 0.0
    rbf_max: float = 8.0
    ffn_multiplier: int = 4
    readout: str = "direct"
    hardness_floor: float = 1.0e-4


@dataclass(frozen=True)
class ChargeModelBatch:
    """Padded molecular batch consumed by :class:`GeometricChargePredictor`."""

    atomic_numbers: Any
    coords: Any
    bond_matrix: Any
    mask: Any
    total_charge: Any
    labels: Any | None = None


class _GeometryAwareChargeBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, pair_dim: int, ffn_multiplier: int):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pair_bias = nn.Linear(pair_dim, n_heads, bias=False)

        ffn_dim = hidden_dim * ffn_multiplier
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def __call__(self, x: mx.array, pair_features: mx.array, mask: mx.array) -> mx.array:
        batch, n_atoms, _ = x.shape
        h = self.norm_attn(x)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = mx.reshape(q, (batch, n_atoms, self.n_heads, self.head_dim))
        k = mx.reshape(k, (batch, n_atoms, self.n_heads, self.head_dim))
        v = mx.reshape(v, (batch, n_atoms, self.n_heads, self.head_dim))
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        logits = (q @ mx.transpose(k, (0, 1, 3, 2))) / np.sqrt(float(self.head_dim))
        bias = self.pair_bias(pair_features)
        bias = mx.transpose(bias, (0, 3, 1, 2))
        logits = logits + bias

        key_mask = mask[:, None, None, :]
        logits = mx.where(key_mask > 0, logits, mx.array(-1.0e9, dtype=logits.dtype))
        attn = mx.softmax(logits, axis=-1) * key_mask

        message = attn @ v
        message = mx.transpose(message, (0, 2, 1, 3))
        message = mx.reshape(message, (batch, n_atoms, self.hidden_dim))

        x = (x + self.out_proj(message)) * mask[:, :, None]
        x = (x + self.ffn(self.norm_ffn(x))) * mask[:, :, None]
        return x


class GeometricChargePredictor(nn.Module):
    """Geometry-aware MLX model for AI-predicted ESP/RESP-like partial charges.

    The model consumes atom types, 3D Cartesian coordinates, and an integer bond
    matrix. It outputs one scalar charge per atom and projects the result onto
    the requested molecular total charge.
    """

    def __init__(self, config: ChargeModelConfig | None = None):
        super().__init__()
        self.config = config or ChargeModelConfig()
        if self.config.hidden_dim % self.config.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.config.n_rbf < 1:
            raise ValueError("n_rbf must be positive")
        if self.config.n_bond_states < 1:
            raise ValueError("n_bond_states must be positive")
        if self.config.readout not in {"direct", "qeq"}:
            raise ValueError("readout must be 'direct' or 'qeq'")

        self.atom_embedding = nn.Embedding(self.config.max_atomic_number + 1, self.config.hidden_dim)
        self.bond_embedding = nn.Embedding(self.config.n_bond_states, self.config.bond_dim)
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)

        pair_dim = self.config.bond_dim + self.config.n_rbf
        self.blocks = [
            _GeometryAwareChargeBlock(
                hidden_dim=self.config.hidden_dim,
                n_heads=self.config.n_heads,
                pair_dim=pair_dim,
                ffn_multiplier=self.config.ffn_multiplier,
            )
            for _ in range(self.config.n_layers)
        ]
        self.final_norm = nn.LayerNorm(self.config.hidden_dim)
        readout_dim = 2 if self.config.readout == "qeq" else 1
        self.charge_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, readout_dim),
        )

        centers = np.linspace(self.config.rbf_min, self.config.rbf_max, self.config.n_rbf).astype(np.float32)
        spacing = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
        self._rbf_centers_np = centers
        self._rbf_gamma = 1.0 / max(spacing, 1.0e-3) ** 2

    def __call__(
        self,
        atomic_numbers: mx.array,
        coords: mx.array,
        bond_matrix: mx.array,
        mask: mx.array,
        total_charge: mx.array | float | int = 0.0,
    ) -> mx.array:
        atomic_numbers = atomic_numbers.astype(mx.int32)
        bond_matrix = bond_matrix.astype(mx.int32)
        coords = coords.astype(mx.float32)
        mask = mask.astype(mx.float32)
        if not isinstance(total_charge, mx.array):
            total_charge = mx.array(total_charge, dtype=mx.float32)
        total_charge = total_charge.astype(mx.float32)
        if total_charge.ndim == 0:
            total_charge = mx.broadcast_to(total_charge[None], (atomic_numbers.shape[0],))

        x = self.atom_embedding(atomic_numbers) * mask[:, :, None]
        x = self.input_norm(x) * mask[:, :, None]
        pair_features = self._pair_features(coords, bond_matrix)

        for block in self.blocks:
            x = block(x, pair_features, mask)

        raw = self.charge_head(self.final_norm(x))

        if self.config.readout == "qeq":
            electronegativity = raw[:, :, 0] * mask
            hardness = nn.softplus(raw[:, :, 1]) + mx.array(self.config.hardness_floor, dtype=raw.dtype)
            return qeq_charges_mlx(
                electronegativity,
                hardness,
                mask,
                total_charge=total_charge,
                hardness_floor=self.config.hardness_floor,
            )

        raw = mx.squeeze(raw, axis=-1) * mask

        n_atoms = mx.maximum(mx.sum(mask, axis=1), mx.array(1.0, dtype=mask.dtype))
        charge_error = (mx.sum(raw, axis=1) - total_charge) / n_atoms
        charges = (raw - charge_error[:, None]) * mask
        return charges

    def _pair_features(self, coords: mx.array, bond_matrix: mx.array) -> mx.array:
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = mx.sqrt(mx.sum(diff * diff, axis=-1) + mx.array(1.0e-8, dtype=coords.dtype))
        centers = mx.array(self._rbf_centers_np, dtype=coords.dtype)
        rbf = mx.exp(-self._rbf_gamma * (dist[:, :, :, None] - centers) ** 2)
        bond_features = self.bond_embedding(bond_matrix)
        return mx.concatenate([bond_features, rbf], axis=-1)


def qeq_charges_mlx(
    electronegativity: mx.array,
    hardness: mx.array,
    mask: mx.array,
    total_charge: mx.array | float | int = 0.0,
    hardness_floor: float = 1.0e-4,
) -> mx.array:
    """Analytical charge-equilibration readout used by EspalomaCharge.

    Given per-atom electronegativity ``e`` and hardness ``s``, this minimizes
    ``sum_i e_i q_i + 0.5 s_i q_i^2`` subject to ``sum_i q_i = Q``.
    """

    electronegativity = electronegativity.astype(mx.float32)
    hardness = hardness.astype(mx.float32)
    mask = mask.astype(mx.float32)
    if not isinstance(total_charge, mx.array):
        total_charge = mx.array(total_charge, dtype=mx.float32)
    total_charge = total_charge.astype(mx.float32)
    if total_charge.ndim == 0:
        total_charge = mx.broadcast_to(total_charge[None], (electronegativity.shape[0],))

    hardness = mx.maximum(hardness, mx.array(hardness_floor, dtype=hardness.dtype))
    s_inv = mask / hardness
    e_s_inv = electronegativity * s_inv
    sum_s_inv = mx.maximum(mx.sum(s_inv, axis=1), mx.array(1.0e-8, dtype=hardness.dtype))
    sum_e_s_inv = mx.sum(e_s_inv, axis=1)
    lagrange = (total_charge + sum_e_s_inv) / sum_s_inv
    return (-electronegativity * s_inv + s_inv * lagrange[:, None]) * mask


def charge_model_batch(
    atom_numbers: Sequence[Sequence[int]],
    coords: Sequence[Any],
    bond_matrices: Sequence[Any] | None = None,
    total_charges: Sequence[float] | None = None,
    labels: Sequence[Any] | None = None,
    pad_to: int | None = None,
    n_bond_states: int = 13,
) -> ChargeModelBatch:
    """Pack variable-size molecules into padded MLX arrays."""

    if len(atom_numbers) != len(coords):
        raise ValueError("atom_numbers and coords must contain the same number of molecules")
    if not atom_numbers:
        raise ValueError("at least one molecule is required")

    n_mols = len(atom_numbers)
    atom_arrays = [np.asarray(atoms, dtype=np.int32) for atoms in atom_numbers]
    coord_arrays = [np.asarray(xyz, dtype=np.float32) for xyz in coords]
    max_atoms = max(len(atoms) for atoms in atom_arrays)
    if pad_to is not None:
        max_atoms = max(max_atoms, int(pad_to))

    if bond_matrices is None:
        bond_arrays = [np.zeros((len(atoms), len(atoms)), dtype=np.int32) for atoms in atom_arrays]
    else:
        if len(bond_matrices) != n_mols:
            raise ValueError("bond_matrices must match the number of molecules")
        bond_arrays = [np.asarray(bonds, dtype=np.int32) for bonds in bond_matrices]

    label_arrays = None
    if labels is not None:
        if len(labels) != n_mols:
            raise ValueError("labels must match the number of molecules")
        label_arrays = [np.asarray(label, dtype=np.float32) for label in labels]

    z_pad = np.zeros((n_mols, max_atoms), dtype=np.int32)
    xyz_pad = np.zeros((n_mols, max_atoms, 3), dtype=np.float32)
    bonds_pad = np.zeros((n_mols, max_atoms, max_atoms), dtype=np.int32)
    mask = np.zeros((n_mols, max_atoms), dtype=np.float32)
    labels_pad = None if label_arrays is None else np.zeros((n_mols, max_atoms), dtype=np.float32)

    for i, (atoms, xyz, bonds) in enumerate(zip(atom_arrays, coord_arrays, bond_arrays)):
        n_atoms = len(atoms)
        if n_atoms == 0:
            raise ValueError("molecules must contain at least one atom")
        if xyz.shape != (n_atoms, 3):
            raise ValueError(f"coords[{i}] must have shape ({n_atoms}, 3)")
        if bonds.shape != (n_atoms, n_atoms):
            raise ValueError(f"bond_matrices[{i}] must have shape ({n_atoms}, {n_atoms})")
        if np.any(atoms < 0):
            raise ValueError("atomic numbers must be non-negative")
        if np.any((bonds < 0) | (bonds >= n_bond_states)):
            raise ValueError(f"bond states must be in [0, {n_bond_states})")

        z_pad[i, :n_atoms] = atoms
        xyz_pad[i, :n_atoms] = xyz
        bonds_pad[i, :n_atoms, :n_atoms] = bonds
        mask[i, :n_atoms] = 1.0
        if labels_pad is not None:
            label = label_arrays[i]
            if label.shape != (n_atoms,):
                raise ValueError(f"labels[{i}] must have shape ({n_atoms},)")
            labels_pad[i, :n_atoms] = label

    if total_charges is None:
        total_charge_array = np.zeros((n_mols,), dtype=np.float32)
    else:
        if len(total_charges) != n_mols:
            raise ValueError("total_charges must match the number of molecules")
        total_charge_array = np.asarray(total_charges, dtype=np.float32)

    return ChargeModelBatch(
        atomic_numbers=mx.array(z_pad),
        coords=mx.array(xyz_pad),
        bond_matrix=mx.array(bonds_pad),
        mask=mx.array(mask),
        total_charge=mx.array(total_charge_array),
        labels=None if labels_pad is None else mx.array(labels_pad),
    )


def charge_prediction_loss(
    predicted: mx.array,
    target: mx.array,
    mask: mx.array,
    atom_weights: mx.array | None = None,
) -> mx.array:
    """Masked mean-squared error for per-atom charge targets."""

    mask = mask.astype(mx.float32)
    err = (predicted - target) ** 2 * mask
    denom = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mask.dtype))
    if atom_weights is not None:
        weights = atom_weights.astype(mx.float32) * mask
        err = err * weights
        denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mask.dtype))
    return mx.sum(err) / denom


def predict_partial_charges_mlx(
    model: GeometricChargePredictor,
    atoms: Sequence[int],
    coords: Any,
    bond_matrix: Any | None = None,
    total_charge: float = 0.0,
) -> np.ndarray:
    """Predict charges for one molecule and return an unpadded NumPy vector."""

    batch = charge_model_batch(
        [list(atoms)],
        [coords],
        None if bond_matrix is None else [bond_matrix],
        total_charges=[total_charge],
        n_bond_states=model.config.n_bond_states,
    )
    charges = model(
        batch.atomic_numbers,
        batch.coords,
        batch.bond_matrix,
        batch.mask,
        batch.total_charge,
    )
    mx.eval(charges)
    return np.asarray(charges[0, : len(atoms)])


def load_charge_model(
    path: str | Path,
    config: ChargeModelConfig | None = None,
) -> GeometricChargePredictor:
    """Create a predictor and load MLX weights from ``path``."""

    model = GeometricChargePredictor(config)
    model.load_weights(str(path))
    return model


def rdkit_mol_from_smiles_3d(
    smiles: str,
    *,
    add_hs: bool = True,
    random_seed: int = 0xC0FFEE,
    optimize: bool = True,
    mmff_variant: str = "MMFF94",
    max_iters: int = 200,
):
    """Parse a SMILES string and generate one 3D RDKit conformer.

    EspalomaCharge itself is graph-only at inference time. The geometry-aware
    MLX charge model in this module is different: it consumes atomic numbers,
    3D coordinates, and bond states, so SMILES inputs must pass through an
    explicit conformer generation step before batched inference.
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    if add_hs:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(
            mol,
            randomSeed=int(random_seed),
            useRandomCoords=True,
        )
    if status != 0:
        raise ValueError(f"could not generate a 3D conformer for SMILES: {smiles!r}")

    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(
                    mol,
                    mmffVariant=mmff_variant,
                    maxIters=int(max_iters),
                )
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
        except Exception:
            # Keep the ETKDG conformer if force-field cleanup is unavailable.
            pass

    return mol


def rdkit_mols_from_smiles_3d(
    smiles_list: Sequence[str],
    *,
    add_hs: bool = True,
    random_seed: int = 0xC0FFEE,
    optimize: bool = True,
    mmff_variant: str = "MMFF94",
    max_iters: int = 200,
) -> list[Any]:
    """Generate one 3D RDKit molecule for each SMILES string."""

    return [
        rdkit_mol_from_smiles_3d(
            smiles,
            add_hs=add_hs,
            random_seed=int(random_seed) + i,
            optimize=optimize,
            mmff_variant=mmff_variant,
            max_iters=max_iters,
        )
        for i, smiles in enumerate(smiles_list)
    ]


def bond_state_from_rdkit_bond(bond: Any) -> int:
    """Map an RDKit bond object to the 13-state integer encoding."""

    from rdkit import Chem

    bond_type = bond.GetBondType()
    if bond.GetIsAromatic() or bond_type == Chem.BondType.AROMATIC:
        return 4
    if bond_type == Chem.BondType.SINGLE:
        return 1
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.BondType.DATIVE:
        return 5
    if bond_type == Chem.BondType.ONEANDAHALF:
        return 6
    if bond_type == Chem.BondType.TWOANDAHALF:
        return 7
    if bond_type == Chem.BondType.THREEANDAHALF:
        return 8
    if bond_type == Chem.BondType.QUADRUPLE:
        return 9
    if bond_type == Chem.BondType.QUINTUPLE:
        return 10
    if bond_type == Chem.BondType.HEXTUPLE:
        return 11
    return 12


def bond_matrix_from_rdkit_mol(mol: Any, n_bond_states: int = 13) -> np.ndarray:
    """Return a symmetric integer bond-state matrix for an RDKit molecule."""

    n_atoms = mol.GetNumAtoms()
    bonds = np.zeros((n_atoms, n_atoms), dtype=np.int32)
    for bond in mol.GetBonds():
        state = bond_state_from_rdkit_bond(bond)
        if state >= n_bond_states:
            state = n_bond_states - 1
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds[i, j] = state
        bonds[j, i] = state
    return bonds


def charge_model_batch_from_rdkit_mols(
    mols: Sequence[Any],
    conf_ids: Sequence[int] | None = None,
    total_charges: Sequence[float] | None = None,
    labels: Sequence[Any] | None = None,
    pad_to: int | None = None,
    n_bond_states: int = 13,
) -> ChargeModelBatch:
    """Build a charge-model batch directly from RDKit molecules with conformers."""

    from rdkit import Chem

    if conf_ids is None:
        conf_ids = [-1] * len(mols)
    if len(conf_ids) != len(mols):
        raise ValueError("conf_ids must match the number of molecules")
    if total_charges is None:
        total_charges = [float(Chem.GetFormalCharge(mol)) for mol in mols]

    atom_numbers: list[list[int]] = []
    coords: list[np.ndarray] = []
    bond_matrices: list[np.ndarray] = []

    for mol_index, (mol, conf_id) in enumerate(zip(mols, conf_ids)):
        if mol.GetNumConformers() == 0:
            raise ValueError(
                f"molecule {mol_index} has no conformer; use "
                "rdkit_mol_from_smiles_3d(), rdkit_mols_from_smiles_3d(), "
                "or charge_model_batch_from_smiles() first"
            )
        atom_numbers.append([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        conf = mol.GetConformer(int(conf_id))
        coords.append(np.asarray(conf.GetPositions(), dtype=np.float32))
        bond_matrices.append(bond_matrix_from_rdkit_mol(mol, n_bond_states=n_bond_states))

    return charge_model_batch(
        atom_numbers,
        coords,
        bond_matrices=bond_matrices,
        total_charges=total_charges,
        labels=labels,
        pad_to=pad_to,
        n_bond_states=n_bond_states,
    )


def charge_model_batch_from_smiles(
    smiles_list: Sequence[str],
    *,
    total_charges: Sequence[float] | None = None,
    add_hs: bool = True,
    random_seed: int = 0xC0FFEE,
    optimize: bool = True,
    mmff_variant: str = "MMFF94",
    max_iters: int = 200,
    labels: Sequence[Any] | None = None,
    pad_to: int | None = None,
    n_bond_states: int = 13,
) -> ChargeModelBatch:
    """Build a padded charge-model batch directly from SMILES.

    Pipeline: SMILES -> RDKit Mol -> AddHs -> ETKDGv3 3D conformer ->
    optional MMFF/UFF cleanup -> padded MLX tensors.
    """

    mols = rdkit_mols_from_smiles_3d(
        smiles_list,
        add_hs=add_hs,
        random_seed=random_seed,
        optimize=optimize,
        mmff_variant=mmff_variant,
        max_iters=max_iters,
    )
    return charge_model_batch_from_rdkit_mols(
        mols,
        total_charges=total_charges,
        labels=labels,
        pad_to=pad_to,
        n_bond_states=n_bond_states,
    )


def predict_partial_charges_batch_mlx(
    model: GeometricChargePredictor,
    batch: ChargeModelBatch,
) -> list[np.ndarray]:
    """Run one batched MLX forward pass and return unpadded charge vectors."""

    charges = model(
        batch.atomic_numbers,
        batch.coords,
        batch.bond_matrix,
        batch.mask,
        batch.total_charge,
    )
    mx.eval(charges)
    charge_array = np.asarray(charges, dtype=np.float64)
    mask_array = np.asarray(batch.mask, dtype=np.float32)
    n_atoms = mask_array.sum(axis=1).astype(np.int64)
    return [charge_array[i, : int(n)] for i, n in enumerate(n_atoms)]


def predict_partial_charges_from_rdkit_mols_mlx(
    model: GeometricChargePredictor,
    mols: Sequence[Any],
    *,
    conf_ids: Sequence[int] | None = None,
    total_charges: Sequence[float] | None = None,
    pad_to: int | None = None,
) -> list[np.ndarray]:
    """Predict charges for many conformer-bearing RDKit molecules."""

    batch = charge_model_batch_from_rdkit_mols(
        mols,
        conf_ids=conf_ids,
        total_charges=total_charges,
        pad_to=pad_to,
        n_bond_states=model.config.n_bond_states,
    )
    return predict_partial_charges_batch_mlx(model, batch)


def predict_partial_charges_from_smiles_mlx(
    model: GeometricChargePredictor,
    smiles_list: Sequence[str],
    *,
    total_charges: Sequence[float] | None = None,
    add_hs: bool = True,
    random_seed: int = 0xC0FFEE,
    optimize: bool = True,
    mmff_variant: str = "MMFF94",
    max_iters: int = 200,
    pad_to: int | None = None,
) -> list[np.ndarray]:
    """Predict charges for many SMILES strings through a shared MLX batch."""

    batch = charge_model_batch_from_smiles(
        smiles_list,
        total_charges=total_charges,
        add_hs=add_hs,
        random_seed=random_seed,
        optimize=optimize,
        mmff_variant=mmff_variant,
        max_iters=max_iters,
        pad_to=pad_to,
        n_bond_states=model.config.n_bond_states,
    )
    return predict_partial_charges_batch_mlx(model, batch)


def iter_charge_training_batches(
    atom_numbers: Sequence[Sequence[int]],
    coords: Sequence[Any],
    labels: Sequence[Any],
    bond_matrices: Sequence[Any] | None = None,
    total_charges: Sequence[float] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int | None = None,
    n_bond_states: int = 13,
) -> Iterable[ChargeModelBatch]:
    """Yield padded batches for charge-model training."""

    n_mols = len(atom_numbers)
    if len(coords) != n_mols or len(labels) != n_mols:
        raise ValueError("atom_numbers, coords, and labels must have the same length")
    indices = np.arange(n_mols)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n_mols, batch_size):
        idx = indices[start : start + batch_size]
        batch_bonds = None if bond_matrices is None else [bond_matrices[i] for i in idx]
        batch_charges = None if total_charges is None else [total_charges[i] for i in idx]
        yield charge_model_batch(
            [atom_numbers[i] for i in idx],
            [coords[i] for i in idx],
            bond_matrices=batch_bonds,
            total_charges=batch_charges,
            labels=[labels[i] for i in idx],
            n_bond_states=n_bond_states,
        )
