# mlxmolkit — GPU-accelerated molecular toolkit on Apple Silicon

Port of [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (CUDA) to Apple Metal via [MLX](https://github.com/ml-explore/mlx). Two pipelines:

1. **Molecular Clustering** — Morgan FP → Tanimoto similarity → Butina clustering
2. **3D Conformer Generation** — DG (4D) → ETK (3D) → MMFF94 optimization

## Installation

```bash
pip install mlxmolkit-rdkit
```

Requires macOS with Apple Silicon (M1/M2/M3/M4). RDKit is needed for molecular input:

```bash
conda install -c conda-forge rdkit
pip install mlxmolkit-rdkit
```

## Quick Start

```python
from mlxmolkit import generate_conformers_nk, butina_tanimoto_mlx

# 3D conformers: 100 molecules x 10 conformers each
result = generate_conformers_nk(
    smiles_list=["c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"],
    n_confs_per_mol=10,
    run_mmff=True,
)

# Clustering: 150k+ molecules
import mlx.core as mx
result = butina_tanimoto_mlx(mx.array(fp_bytes), cutoff=0.4)
```

## Features

- **Conformer Generation** — Drop-in replacement for RDKit's ETKDG (`EmbedMolecules`). Supports ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG, ETDG, and pure DG.
- **MMFF94 Optimization** — GPU-accelerated force field optimization (`MMFFOptimizeMoleculesConfs`). All 7 MMFF energy terms with fused Metal kernel. Full BFGS or L-BFGS in-kernel (zero CPU round-trips).
- **Molecular Clustering** — Butina clustering at 150k+ molecules with divide-and-conquer memory management.
- **N x k Parallel** — Generate k conformers for N molecules simultaneously. Constraints shared across conformers (`conf_to_mol` indirection, 50% memory savings).

## Performance

### Conformer Generation (1000 distinct SPICE molecules, Apple M3 Max)

Benchmark uses 1000 distinct drug-like molecules from SPICE-2.0.1 (see `data/benchmark_1000_smiles.csv`).

| Scale | Pipeline | Time | Throughput |
|-------|----------|------|-----------|
| N=100 × k=50 | DG only | 8.6s | **580 conf/s** |
| N=100 × k=50 | DG + ETKDGv2 | 9.3s | **536 conf/s** |
| N=100 × k=50 | DG + ETK + MMFF | 9.5s | **525 conf/s** |

### Conformer Memory Scaling (DG + ETK + MMFF, batch=500)

| Conformers | Batch | GPU (BFGS) | GPU (L-BFGS) | Time | Throughput |
|-----------|-------|-----------|-------------|------|-----------|
| 1,000 | 1000 | 5.1 MB | 2.9 MB | 0.43s | 2,342/s |
| 2,000 | 500 | 2.6 MB | 1.5 MB | 1.43s | 1,402/s |
| 4,000 | 500 | 2.6 MB | 1.5 MB | 1.91s | 2,094/s |
| 10,000 | 500 | **2.6 MB** | **1.5 MB** | 4.82s | 2,075/s |

GPU memory stays constant regardless of total conformers thanks to divide-and-conquer batching.

### Scale Tests (1000 distinct molecules, Apple M3 Max)

| Scale | Pipeline | Time | Throughput |
|-------|----------|------|-----------|
| N=1000, k=10 | DG only | 17.5s | **573 conf/s** |
| N=1000, k=10 | DG + ETKDGv2 | 21.1s | **474 conf/s** |
| N=1000, k=10 | DG + ETK + MMFF | 21.4s | **467 conf/s** |

All stages on Metal (including MMFF94 — zero RDKit post-processing).

### Batch Size Impact

Larger batches = fewer kernel launches = higher throughput. Auto-sizing (default) picks the largest batch that fits in free memory.

### GPU Memory per Conformer

| Atoms | DG (4D) | ETK (3D) | MMFF (BFGS) | MMFF (L-BFGS) |
|------:|--------:|---------:|------------:|--------------:|
| 5 | 1.9 KB | 1.4 KB | 1.3 KB | 1.4 KB |
| 12 | 4.4 KB | 3.3 KB | 6.0 KB | 3.3 KB |
| 21 | 7.6 KB | 5.7 KB | 17.2 KB | 5.7 KB |
| 30 | 10.8 KB | 8.1 KB | 34.1 KB | 8.1 KB |
| 50 | 18.0 KB | 13.5 KB | 92.0 KB | 13.5 KB |
| 64 | 23.1 KB | 17.3 KB | **149.2 KB** | 17.3 KB |

MMFF BFGS memory grows as O(n^2) due to the dense Hessian (n_atoms x 3)^2. **BFGS is faster than L-BFGS at all typical drug-like sizes** (up to 74 atoms with H) because the better curvature information requires fewer iterations. L-BFGS is only needed for very large molecules (>150 atoms) where the Hessian exceeds ~1 MB per conformer.

| Molecule | Atoms (with H) | BFGS | L-BFGS | Winner |
|----------|---------------|------|--------|--------|
| Methane | 5 | 0.255s | 0.215s | BFGS |
| Benzene | 12 | 0.213s | 0.222s | ~tie |
| Aspirin | 21 | 0.241s | 0.230s | ~tie |
| Testosterone | 49 | 0.364s | 0.335s | BFGS |
| Cholesterol | 74 | 0.590s | 0.486s | BFGS |

Recommendation: use BFGS (default) for all molecules <150 atoms with H. The pipeline auto-switches to L-BFGS at 150+ atoms (`mmff_use_lbfgs=None`, the default).

**Important:** Always add explicit hydrogens (`Chem.AddHs`) before conformer generation. Convergence is significantly better with explicit H because the distance geometry constraints are more complete and the force field terms (bond/angle/torsion) are fully defined. The pipeline calls `AddHs` automatically.

With 64 GB unified memory, a single batch can hold:

| Molecule size | DG/ETK | MMFF (BFGS) | MMFF (L-BFGS) |
|--------------|-------:|------------:|--------------:|
| 12 atoms | ~9.8M conformers | ~7.4M | ~9.8M |
| 30 atoms | ~3.9M conformers | ~1.3M | ~3.9M |
| 64 atoms | ~1.8M conformers | **~300K** | ~1.8M |

The divide-and-conquer queue automatically splits into multiple batches when total exceeds free memory.

### Clustering (Enamine REAL subset, Apple M3 Max)

| N | Fused sim→CSR | Butina | **Total** | vs RDKit | Memory |
|---|---|---|---|---|---|
| 20k | 0.26s | 0.09s | **0.35s** | **152x** | 0.1 MB |
| 50k | 1.26s | 0.36s | **1.62s** | — | 0.5 MB |
| 100k | 4.87s | 0.97s | **5.84s** | — | 1.3 MB |
| 150k+ | blockwise | — | scales | — | bounded |

### ETKDG Variant Comparison (N=20, k=50, same molecule)

Throughput for homogeneous batches (all conformers of the same molecule, best case):

| Variant | conf/s | Convergence |
|---------|--------|-------------|
| DG | 7,549 | 96.6% |
| KDG | 7,243 | 96.6% |
| ETDG | 1,844 | 96.6% |
| ETKDG | 6,064 | 96.6% |
| ETKDGv2 | 6,228 | 96.6% |
| ETKDGv3 | 6,636 | 96.6% |
| srETKDGv3 | 6,678 | 96.6% |

Note: throughput is lower for diverse molecule batches due to variable atom counts and padding overhead. The Scale Tests section above shows realistic numbers for heterogeneous batches.

## Architecture

### Conformer Generation (N x k parallel)

```
SMILES x N
    |
[RDKit CPU] Extract params ONCE per molecule
    |
[Pack] SharedConstraintBatch (conf_to_mol indirection)
    |
+-- Stage 1: DG minimize (4D, Metal TPM=32) --------+
|   One threadgroup per conformer                    |
|   L-BFGS in-kernel, GPU-parallel line search       |
|   Shared constraints via conf_to_mol               |
+----------------------------------------------------+
    |
[Extract 3D] Drop 4th coordinate
    |
+-- Stage 2: ETK minimize (3D, Metal TPM=32) --------+
|   CSD torsion + improper + 1-4 distance             |
|   Optional parallel_grad for large molecules        |
+-----------------------------------------------------+
    |
+-- Stage 3: MMFF94 optimize (Metal, in-kernel) ------+
|   7 energy terms: bond, angle, stretch-bend,         |
|   OOP, torsion, vdW, electrostatic                   |
|   BFGS (default) or L-BFGS option                    |
+------------------------------------------------------+
    |
Optimized 3D conformers
```

### Clustering (divide-and-conquer for 150k+)

```
Morgan FP (RDKit CPU)
        |
   uint8 -> uint32 packing
        |
+-- N <= 100k: Fused Metal Kernel ------+
|   Single dispatch, no NxN matrix       |
+-- N > 100k: Blockwise D&C ------------+
|   Tile both dimensions (auto-sized)    |
|   mx.eval() between tiles (free GPU)  |
+----------------------------------------+
        |
   Butina greedy (CPU, numpy CSR)
        |
   Clusters
```

## Adaptive Iteration Scaling

Iterations auto-scale by molecule complexity (default). Small molecules converge early via in-kernel TOLX/gradient checks — no wasted GPU compute.

Formula: `max_iters = base + scale * max(n_atoms, sqrt(n_constraints))`

| Molecule | Atoms | Constraints | DG iters | ETK iters | MMFF iters |
|----------|-------|-------------|----------|-----------|------------|
| Methane | 5 | 10 | 400 | 200 | 275 |
| Benzene | 12 | 66 | 540 | 270 | 380 |
| Aspirin | 21 | 210 | 720 | 360 | 515 |
| Testosterone | 49 | 1176 | 1280 | 640 | 935 |
| 64-atom | 64 | 2016 | 1580 | 790 | 1160 |

Override with explicit values when needed:

```python
# Auto (default) — scales with molecule size
result = generate_conformers_nk(smiles_list, n_confs_per_mol=10)

# Fixed iterations for fine control
result = generate_conformers_nk(smiles_list, n_confs_per_mol=10,
    dg_max_iters=1000, etk_max_iters=500, mmff_max_iters=400)
```

## Optimization Options

| Option | Flag | Effect | When to use |
|--------|------|--------|-------------|
| Auto iterations (default) | `dg_max_iters=0` | Scales with molecule size | Always (default) |
| Warm-start retry | automatic | Re-runs non-converged with 2x iters | Always (automatic) |
| ETK parallel gradient | `parallel_grad=True` | 1.18x ETK speedup | Many distance constraints |
| DG parallel gradient | `parallel_grad=True` | Parallelizes dist gradient | >500 distance constraints |
| MMFF94s variant | `mmff_variant="MMFF94s"` | Softer torsion barriers | Conjugated/aromatic molecules |
| MMFF L-BFGS | `mmff_use_lbfgs=True` | 5x less memory | Molecules >50 atoms |
| MMFF BFGS (default) | `mmff_use_lbfgs=False` | 2x faster for small mols | Molecules <50 atoms |
| ETKDG variant | `variant="ETKDGv3"` | 7 variants supported | Choose per use case |

### MMFF94 Force Field Variants

| Variant | Flag | Torsion barriers | Best for |
|---------|------|-----------------|----------|
| MMFF94 (default) | `mmff_variant="MMFF94"` | Standard | General molecules |
| MMFF94s | `mmff_variant="MMFF94s"` | Softer for conjugated systems | Aromatic, planar, conjugated |

## Usage

### 3D Conformer Generation

```python
from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk

# Basic: 10 conformers per molecule, ETKDGv2
result = generate_conformers_nk(
    smiles_list=["c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"],
    n_confs_per_mol=10,
)
for mol in result.molecules:
    print(f"{mol.n_atoms} atoms, {len(mol.positions_3d)} conformers")

# Full pipeline with MMFF94
result = generate_conformers_nk(
    smiles_list=["c1ccccc1", "CC(=O)O"],
    n_confs_per_mol=50,
    variant="ETKDGv3",
    run_mmff=True,
    mmff_use_lbfgs=False,       # BFGS (default, fast for <50 atoms)
    max_confs_per_batch=500,    # divide-and-conquer batch size
)

# Pure distance geometry (no torsion refinement)
result = generate_conformers_nk(
    smiles_list=["c1ccccc1"],
    n_confs_per_mol=100,
    variant="DG",
)

# Large molecules: use L-BFGS for MMFF
result = generate_conformers_nk(
    smiles_list=[large_smiles],
    n_confs_per_mol=20,
    run_mmff=True,
    mmff_use_lbfgs=True,        # L-BFGS for >50 atoms
)
```

### Example Script

```bash
# Basic: 20 molecules x 10 conformers
python examples/conf3d_example.py

# Scale test: 1000 molecules x 10 conformers with MMFF
python examples/conf3d_example.py --n-mols 1000 --n-confs 10 --mmff

# All options
python examples/conf3d_example.py --n-mols 100 --n-confs 20 --variant ETKDGv3 \
    --mmff --mmff-variant MMFF94s --batch-size 200

# Custom SMILES
python examples/conf3d_example.py --smiles "c1ccccc1" "CC(=O)O" --n-confs 50 --mmff
```

### Molecular Clustering

```python
from mlxmolkit import butina_tanimoto_mlx
import mlx.core as mx

# Automatic: fused kernel for N<=100k, blockwise for N>100k
result = butina_tanimoto_mlx(mx.array(fp_bytes), cutoff=0.4)
print(f"{len(result.clusters)} clusters")
```

### Low-level API

```python
from mlxmolkit import (
    fp_uint8_to_uint32,
    fused_neighbor_list_metal,
    tanimoto_neighbors_blockwise,
    butina_from_neighbor_list_csr,
)

fp_u32 = fp_uint8_to_uint32(mx.array(fp_bytes))

# Small N: fused single-dispatch
offsets, indices = fused_neighbor_list_metal(fp_u32, cutoff=0.4)

# Large N (150k+): divide-and-conquer blockwise
offsets, indices = tanimoto_neighbors_blockwise(fp_u32, cutoff=0.4)

result = butina_from_neighbor_list_csr(offsets, indices, N, cutoff=0.4)
```

## Tests

```bash
pip install -e .
pytest tests/ -v
```

## Conformer Quality vs RDKit

mlxmolkit conformers rescored by RDKit's MMFF94 for fair comparison (k=20, ETKDGv2):

After MMFF optimization, mlxmolkit conformers converge to the **same energy basins** as RDKit:

| Molecule | Atoms | RMSD (pre-MMFF) | RMSD (post-MMFF) | E gap |
|----------|------:|----------------:|-----------------:|------:|
| Benzene | 12 | 0.12 A | **0.00 A** | **0.0** |
| Aspirin | 21 | 0.98 A | **0.00 A** | **0.0** |
| Ibuprofen | 33 | 1.79 A | **0.96 A** | **0.6** |
| Acetaminophen | 20 | 0.99 A | **0.00 A** | **0.0** |

Full nvMolKit pipeline: DG (4D) → 4D→3D collapse → setReferenceValues → stereo checks → ETK (3D) → MMFF94. Bond/angle geometry matches RDKit within 0.03 A. After MMFF, energy gap is <1 kcal/mol.

## References

- [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) — NVIDIA's CUDA implementation (Apache 2.0)
- [shivampatel10/mlxmolkit](https://github.com/shivampatel10/mlxmolkit) — TPM threadgroup kernels and MMFF Metal implementation
- [RDKit blog: Butina clustering with nvMolKit](https://greglandrum.github.io/rdkit-blog/posts/2026-02-28-nvmolkit-clustering.html)
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework with Metal kernel support
- [MMFF94](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P) — Halgren, J. Comput. Chem. 1996
- [Butina, D. (1999)](https://doi.org/10.1021/ci9803381) — Performance of Kier-Hall and molecular connectivity indices
