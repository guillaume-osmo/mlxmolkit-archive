from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlxmolkit.charge_model import bond_matrix_from_rdkit_mol


@dataclass(frozen=True)
class CheeseEmbeddingConfig:
    """Configuration for a CHEESE-style graph/3D transformer embedding model."""

    max_atomic_number: int = 118
    n_bond_states: int = 13
    hidden_dim: int = 128
    embedding_dim: int = 128
    bond_dim: int = 16
    n_layers: int = 4
    n_heads: int = 4
    n_rbf: int = 32
    rbf_min: float = 0.0
    rbf_max: float = 8.0
    ffn_multiplier: int = 4
    use_charges: bool = True
    use_chiral_features: bool = True
    normalize_embeddings: bool = True
    eps: float = 1.0e-8


@dataclass(frozen=True)
class CheeseEmbeddingBatch:
    """Padded graph/geometry batch consumed by :class:`CheeseGraphTransformer`."""

    atomic_numbers: Any
    coords: Any
    bond_matrix: Any
    mask: Any
    charges: Any | None = None
    chiral_features: Any | None = None
    ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CheeseEmbeddingOutput:
    """Outputs from :class:`CheeseGraphTransformer`."""

    embedding: Any
    atom_embeddings: Any
    atom_logits: Any
    reconstructed_coords: Any


@dataclass(frozen=True)
class CheeseEmbeddingLoss:
    """Scalar loss and named components for CHEESE embedding training."""

    total: Any
    metric: Any
    atom: Any
    distance: Any


class _CheeseEmbeddingBlock(nn.Module):
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
        bias = mx.transpose(self.pair_bias(pair_features), (0, 3, 1, 2))
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


class CheeseGraphTransformer(nn.Module):
    """CHEESE-style graph/3D transformer for vector-space similarity search.

    The model produces a normalized molecular embedding suitable for ANN search.
    It also exposes lightweight autoencoder heads for atom identity and pairwise
    distance reconstruction, which regularize the latent space while the main
    teacher loss learns shape/ESP/combined CHEESE similarities.
    """

    def __init__(self, config: CheeseEmbeddingConfig | None = None):
        super().__init__()
        self.config = config or CheeseEmbeddingConfig()
        if self.config.hidden_dim % self.config.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.config.n_rbf < 1:
            raise ValueError("n_rbf must be positive")
        if self.config.n_bond_states < 1:
            raise ValueError("n_bond_states must be positive")

        self.atom_embedding = nn.Embedding(self.config.max_atomic_number + 1, self.config.hidden_dim)
        self.bond_embedding = nn.Embedding(self.config.n_bond_states, self.config.bond_dim)
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)
        self.charge_projection = nn.Linear(1, self.config.hidden_dim) if self.config.use_charges else None
        self.chiral_projection = nn.Linear(1, self.config.hidden_dim) if self.config.use_chiral_features else None

        pair_dim = self.config.bond_dim + self.config.n_rbf
        self.blocks = [
            _CheeseEmbeddingBlock(
                hidden_dim=self.config.hidden_dim,
                n_heads=self.config.n_heads,
                pair_dim=pair_dim,
                ffn_multiplier=self.config.ffn_multiplier,
            )
            for _ in range(self.config.n_layers)
        ]
        self.final_norm = nn.LayerNorm(self.config.hidden_dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.embedding_dim),
        )
        self.atom_decoder = nn.Linear(self.config.hidden_dim, self.config.max_atomic_number + 1)
        self.coord_decoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 3),
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
        charges: mx.array | None = None,
        chiral_features: mx.array | None = None,
    ) -> CheeseEmbeddingOutput:
        atomic_numbers = atomic_numbers.astype(mx.int32)
        bond_matrix = bond_matrix.astype(mx.int32)
        coords = coords.astype(mx.float32)
        mask = mask.astype(mx.float32)

        x = self.atom_embedding(atomic_numbers)
        if self.config.use_charges and self.charge_projection is not None:
            if charges is None:
                charges = mx.zeros(atomic_numbers.shape, dtype=mx.float32)
            charge_feature = charges.astype(mx.float32)[:, :, None]
            x = x + self.charge_projection(charge_feature)
        if self.config.use_chiral_features and self.chiral_projection is not None:
            if chiral_features is None:
                chiral_features = mx.zeros(atomic_numbers.shape, dtype=mx.float32)
            x = x + self.chiral_projection(chiral_features.astype(mx.float32)[:, :, None])
        x = self.input_norm(x) * mask[:, :, None]

        pair_features = self._pair_features(coords, bond_matrix)
        for block in self.blocks:
            x = block(x, pair_features, mask)

        atom_embeddings = self.final_norm(x) * mask[:, :, None]
        pooled = masked_mean_mlx(atom_embeddings, mask, axis=1)
        embedding = self.embedding_head(pooled)
        if self.config.normalize_embeddings:
            embedding = l2_normalize_mlx(embedding, eps=self.config.eps)

        atom_logits = self.atom_decoder(atom_embeddings)
        reconstructed_coords = self.coord_decoder(atom_embeddings) * mask[:, :, None]
        return CheeseEmbeddingOutput(
            embedding=embedding,
            atom_embeddings=atom_embeddings,
            atom_logits=atom_logits,
            reconstructed_coords=reconstructed_coords,
        )

    def encode_batch(self, batch: CheeseEmbeddingBatch) -> mx.array:
        """Return molecular embeddings for a padded batch."""

        return self(
            batch.atomic_numbers,
            batch.coords,
            batch.bond_matrix,
            batch.mask,
            batch.charges,
            batch.chiral_features,
        ).embedding

    def _pair_features(self, coords: mx.array, bond_matrix: mx.array) -> mx.array:
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = mx.sqrt(mx.sum(diff * diff, axis=-1) + mx.array(1.0e-8, dtype=coords.dtype))
        centers = mx.array(self._rbf_centers_np, dtype=coords.dtype)
        rbf = mx.exp(-self._rbf_gamma * (dist[:, :, :, None] - centers) ** 2)
        bond_features = self.bond_embedding(bond_matrix)
        return mx.concatenate([bond_features, rbf], axis=-1)


def l2_normalize_mlx(x: mx.array, *, axis: int = -1, eps: float = 1.0e-8) -> mx.array:
    """L2-normalize an MLX array along ``axis``."""

    norm = mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True) + mx.array(float(eps), dtype=x.dtype))
    return x / norm


def masked_mean_mlx(x: mx.array, mask: mx.array, *, axis: int = 1, eps: float = 1.0e-8) -> mx.array:
    """Masked mean pooling for padded atom tensors."""

    mask = mask.astype(mx.float32)
    while mask.ndim < x.ndim:
        mask = mask[..., None]
    denom = mx.maximum(mx.sum(mask, axis=axis), mx.array(float(eps), dtype=x.dtype))
    return mx.sum(x * mask, axis=axis) / denom


def cheese_embedding_batch(
    atom_numbers: Sequence[Sequence[int]],
    coords: Sequence[Any],
    bond_matrices: Sequence[Any] | None = None,
    charges: Sequence[Any] | None = None,
    chiral_features: Sequence[Any] | None = None,
    *,
    ids: Sequence[str] | None = None,
    pad_to: int | None = None,
    n_bond_states: int = 13,
    compute_chiral_features: bool = True,
) -> CheeseEmbeddingBatch:
    """Pack variable-size molecules into a graph/3D transformer batch."""

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

    charge_arrays = None
    if charges is not None:
        if len(charges) != n_mols:
            raise ValueError("charges must match the number of molecules")
        charge_arrays = [np.asarray(q, dtype=np.float32) for q in charges]

    if chiral_features is not None:
        if len(chiral_features) != n_mols:
            raise ValueError("chiral_features must match the number of molecules")
        chiral_arrays = [np.asarray(v, dtype=np.float32) for v in chiral_features]
    elif compute_chiral_features:
        chiral_arrays = [
            local_chiral_volume_features_np(atoms, xyz, bonds)
            for atoms, xyz, bonds in zip(atom_arrays, coord_arrays, bond_arrays, strict=True)
        ]
    else:
        chiral_arrays = None

    z_pad = np.zeros((n_mols, max_atoms), dtype=np.int32)
    xyz_pad = np.zeros((n_mols, max_atoms, 3), dtype=np.float32)
    bonds_pad = np.zeros((n_mols, max_atoms, max_atoms), dtype=np.int32)
    mask = np.zeros((n_mols, max_atoms), dtype=np.float32)
    charges_pad = None if charge_arrays is None else np.zeros((n_mols, max_atoms), dtype=np.float32)
    chiral_pad = None if chiral_arrays is None else np.zeros((n_mols, max_atoms), dtype=np.float32)

    for i, (atoms, xyz, bonds) in enumerate(zip(atom_arrays, coord_arrays, bond_arrays, strict=True)):
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
        if charges_pad is not None:
            q = charge_arrays[i]
            if q.shape != (n_atoms,):
                raise ValueError(f"charges[{i}] must have shape ({n_atoms},)")
            charges_pad[i, :n_atoms] = q
        if chiral_pad is not None:
            v = chiral_arrays[i]
            if v.shape != (n_atoms,):
                raise ValueError(f"chiral_features[{i}] must have shape ({n_atoms},)")
            chiral_pad[i, :n_atoms] = v

    return CheeseEmbeddingBatch(
        atomic_numbers=mx.array(z_pad),
        coords=mx.array(xyz_pad),
        bond_matrix=mx.array(bonds_pad),
        mask=mx.array(mask),
        charges=None if charges_pad is None else mx.array(charges_pad),
        chiral_features=None if chiral_pad is None else mx.array(chiral_pad),
        ids=None if ids is None else tuple(ids),
    )


def cheese_embedding_batch_from_rdkit_mols(
    mols: Sequence[Any],
    charges: Sequence[Any] | None = None,
    *,
    conf_ids: Sequence[int] | None = None,
    ids: Sequence[str] | None = None,
    pad_to: int | None = None,
    n_bond_states: int = 13,
    compute_chiral_features: bool = True,
) -> CheeseEmbeddingBatch:
    """Build an embedding batch directly from conformer-bearing RDKit molecules."""

    if conf_ids is None:
        conf_ids = [-1] * len(mols)
    if len(conf_ids) != len(mols):
        raise ValueError("conf_ids must match the number of molecules")

    atom_numbers: list[list[int]] = []
    coords: list[np.ndarray] = []
    bonds: list[np.ndarray] = []
    for mol_index, (mol, conf_id) in enumerate(zip(mols, conf_ids, strict=True)):
        if mol.GetNumConformers() == 0:
            raise ValueError(f"molecule {mol_index} has no conformer")
        atom_numbers.append([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        coords.append(np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), dtype=np.float32))
        bonds.append(bond_matrix_from_rdkit_mol(mol, n_bond_states=n_bond_states))

    return cheese_embedding_batch(
        atom_numbers,
        coords,
        bonds,
        charges,
        ids=ids,
        pad_to=pad_to,
        n_bond_states=n_bond_states,
        compute_chiral_features=compute_chiral_features,
    )


def local_chiral_volume_features_np(
    atom_numbers: Sequence[int],
    coords: Any,
    bond_matrix: Any,
    *,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Return per-atom signed local-volume features for stereo sensitivity.

    Pair distances are mirror-invariant. This scalar changes sign under a mirror
    transform for atoms with at least three bonded neighbors, giving the
    transformer a compact chirality cue without using absolute XYZ orientation.
    """

    atoms = np.asarray(atom_numbers, dtype=np.int32)
    xyz = np.asarray(coords, dtype=np.float32)
    bonds = np.asarray(bond_matrix, dtype=np.int32)
    if xyz.shape != (len(atoms), 3):
        raise ValueError("coords must have shape (n_atoms, 3)")
    if bonds.shape != (len(atoms), len(atoms)):
        raise ValueError("bond_matrix must have shape (n_atoms, n_atoms)")

    features = np.zeros((len(atoms),), dtype=np.float32)
    for center in range(len(atoms)):
        neighbors = np.flatnonzero(bonds[center] > 0)
        if len(neighbors) < 3:
            continue
        order = np.lexsort((neighbors, atoms[neighbors], bonds[center, neighbors]))
        chosen = neighbors[order[:3]]
        vecs = xyz[chosen] - xyz[center]
        denom = np.prod(np.maximum(np.linalg.norm(vecs, axis=1), eps))
        features[center] = float(np.dot(np.cross(vecs[0], vecs[1]), vecs[2]) / denom)
    return features


def embedding_cosine_similarity_matrix_mlx(
    embeddings: mx.array,
    reference_embeddings: mx.array | None = None,
    *,
    normalize: bool = True,
    map_to_unit: bool = False,
    eps: float = 1.0e-8,
) -> mx.array:
    """Return pairwise embedding cosine similarities."""

    reference_embeddings = embeddings if reference_embeddings is None else reference_embeddings
    left = embeddings.astype(mx.float32)
    right = reference_embeddings.astype(mx.float32)
    if normalize:
        left = l2_normalize_mlx(left, eps=eps)
        right = l2_normalize_mlx(right, eps=eps)
    sim = left @ mx.transpose(right, (1, 0))
    if map_to_unit:
        sim = 0.5 * (sim + mx.array(1.0, dtype=sim.dtype))
    return sim


def embedding_euclidean_distance_matrix_mlx(
    embeddings: mx.array,
    reference_embeddings: mx.array | None = None,
    *,
    normalize: bool = True,
    eps: float = 1.0e-8,
) -> mx.array:
    """Return pairwise Euclidean distances between embeddings."""

    reference_embeddings = embeddings if reference_embeddings is None else reference_embeddings
    left = embeddings.astype(mx.float32)
    right = reference_embeddings.astype(mx.float32)
    if normalize:
        left = l2_normalize_mlx(left, eps=eps)
        right = l2_normalize_mlx(right, eps=eps)
    diff = left[:, None, :] - right[None, :, :]
    return mx.sqrt(mx.sum(diff * diff, axis=-1) + mx.array(float(eps), dtype=left.dtype))


def cheese_embedding_metric_loss(
    embeddings: mx.array,
    target_similarity: mx.array,
    reference_embeddings: mx.array | None = None,
    *,
    pair_mask: mx.array | None = None,
    target_is_unit: bool = True,
    include_diagonal: bool = False,
    eps: float = 1.0e-8,
) -> mx.array:
    """MSE loss that makes embedding cosine reproduce CHEESE teacher scores.

    CHEESE teacher similarities are normally in ``[0, 1]``. With normalized
    embeddings, cosine and Euclidean distance are monotonic, so the target is
    mapped to ``[-1, 1]`` before fitting cosine.
    """

    pred = embedding_cosine_similarity_matrix_mlx(
        embeddings,
        reference_embeddings,
        normalize=True,
        map_to_unit=False,
        eps=eps,
    )
    target = target_similarity.astype(mx.float32)
    if target_is_unit:
        target = 2.0 * target - 1.0

    if pair_mask is None:
        weights = mx.ones(target.shape, dtype=target.dtype)
    else:
        weights = pair_mask.astype(mx.float32)

    if not include_diagonal and reference_embeddings is None and target.shape[0] == target.shape[1]:
        offdiag = mx.ones(target.shape, dtype=target.dtype) - mx.eye(target.shape[0], dtype=target.dtype)
        weights = weights * offdiag

    err = (pred - target) ** 2 * weights
    denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=target.dtype))
    return mx.sum(err) / denom


def atom_reconstruction_loss_mlx(
    atom_logits: mx.array,
    atomic_numbers: mx.array,
    mask: mx.array,
) -> mx.array:
    """Masked atom-type reconstruction cross entropy."""

    per_atom = nn.losses.cross_entropy(atom_logits, atomic_numbers.astype(mx.int32), axis=-1, reduction="none")
    weights = mask.astype(mx.float32)
    denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=weights.dtype))
    return mx.sum(per_atom * weights) / denom


def pair_distance_reconstruction_loss_mlx(
    reconstructed_coords: mx.array,
    target_coords: mx.array,
    mask: mx.array,
    *,
    include_diagonal: bool = False,
) -> mx.array:
    """Reconstruct pairwise distances, avoiding any absolute-orientation target."""

    pred = _pairwise_distances(reconstructed_coords.astype(mx.float32))
    target = _pairwise_distances(target_coords.astype(mx.float32))
    weights = mask[:, :, None].astype(mx.float32) * mask[:, None, :].astype(mx.float32)
    if not include_diagonal:
        n_atoms = target.shape[1]
        offdiag = mx.ones((n_atoms, n_atoms), dtype=weights.dtype) - mx.eye(n_atoms, dtype=weights.dtype)
        weights = weights * offdiag[None, :, :]
    err = (pred - target) ** 2 * weights
    denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=weights.dtype))
    return mx.sum(err) / denom


def cheese_autoencoder_loss_mlx(
    output: CheeseEmbeddingOutput,
    batch: CheeseEmbeddingBatch,
    *,
    atom_weight: float = 1.0,
    distance_weight: float = 1.0,
) -> CheeseEmbeddingLoss:
    """Return autoencoder reconstruction components for a CHEESE embedding."""

    atom = atom_reconstruction_loss_mlx(output.atom_logits, batch.atomic_numbers, batch.mask)
    distance = pair_distance_reconstruction_loss_mlx(output.reconstructed_coords, batch.coords, batch.mask)
    total = float(atom_weight) * atom + float(distance_weight) * distance
    zero = mx.array(0.0, dtype=total.dtype)
    return CheeseEmbeddingLoss(total=total, metric=zero, atom=atom, distance=distance)


def cheese_embedding_loss_mlx(
    output: CheeseEmbeddingOutput,
    batch: CheeseEmbeddingBatch,
    target_similarity: mx.array,
    *,
    metric_weight: float = 1.0,
    atom_weight: float = 0.05,
    distance_weight: float = 0.05,
    pair_mask: mx.array | None = None,
    target_is_unit: bool = True,
    include_diagonal: bool = False,
) -> CheeseEmbeddingLoss:
    """Combined teacher-similarity and autoencoder training loss."""

    metric = cheese_embedding_metric_loss(
        output.embedding,
        target_similarity,
        pair_mask=pair_mask,
        target_is_unit=target_is_unit,
        include_diagonal=include_diagonal,
    )
    atom = atom_reconstruction_loss_mlx(output.atom_logits, batch.atomic_numbers, batch.mask)
    distance = pair_distance_reconstruction_loss_mlx(output.reconstructed_coords, batch.coords, batch.mask)
    total = float(metric_weight) * metric + float(atom_weight) * atom + float(distance_weight) * distance
    return CheeseEmbeddingLoss(total=total, metric=metric, atom=atom, distance=distance)


def _pairwise_distances(coords: mx.array) -> mx.array:
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return mx.sqrt(mx.sum(diff * diff, axis=-1) + mx.array(1.0e-8, dtype=coords.dtype))
