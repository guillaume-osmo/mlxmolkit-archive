# mlxmolkit — GPU-accelerated molecular clustering on Apple Silicon

Port of the [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (CUDA) molecular clustering pipeline to Apple Metal via [MLX](https://github.com/ml-explore/mlx).

Implements the same 3-step workflow as the [RDKit blog post](https://greglandrum.github.io/rdkit-blog/posts/2026-02-28-nvmolkit-clustering.html):

1. **Morgan Fingerprinting** — RDKit `GetMorganGenerator` (CPU, multi-threaded)
2. **Pairwise Tanimoto Similarity** — Custom Metal kernel (GPU)
3. **Butina Clustering** — Greedy algorithm on CSR neighbor list (CPU)

## What's new in v0.2.0

- **Memory fix for large arrays**: Divide-and-conquer blockwise tiling so the full N x N similarity matrix is never materialised. Block size is auto-computed from available system memory. Handles 150k+ molecules without OOM.
- **Faster Metal kernels**: Fused tiled kernel (Tanimoto + threshold in one pass) with threadgroup shared memory, ~3x faster than the pure-MLX path.
- **Auto-selection**: `butina_tanimoto_mlx()` automatically picks the fused single-dispatch kernel for N <= 100k and the blockwise divide-and-conquer path for larger datasets.

## Key results

Tested on Enamine REAL 10.4M subset (same dataset as the blog), Apple M3 Max:

| N | Fused sim→CSR | Butina | **Total** | vs RDKit | Memory |
|---|---|---|---|---|---|
| 20k | 0.26s | 0.09s | **0.35s** | **152x** | 0.1 MB |
| 50k | 1.26s | 0.36s | **1.62s** | — | 0.5 MB |
| 100k | 4.87s | 0.97s | **5.84s** | — | 1.3 MB |
| 150k+ | blockwise | — | scales | — | bounded |

Cluster count parity with RDKit: delta of 5 clusters at 20k (0.04%), caused by float32 vs float64 precision — comparable to nvMolKit's delta of 2.

## Architecture

```
Morgan FP (RDKit CPU)
        ↓
   uint8 → uint32 packing
        ↓
┌─ N <= 100k: Fused Metal Kernel ──────┐
│  Single dispatch, no N×N matrix       │
│  Each thread: Tanimoto → threshold    │
│  → CSR neighbor list                  │
├─ N > 100k: Blockwise Divide & Conquer┤
│  Tile both dimensions (auto-sized)    │
│  Per tile: tiled Metal kernel → mask  │
│  → sparse neighbors → merge CSR      │
│  mx.eval() between tiles (free GPU)  │
└───────────────────────────────────────┘
        ↓
   Butina greedy (CPU, numpy CSR)
        ↓
   Clusters
```

No N x N similarity matrix is ever materialised. At N=50k that's 10 GB saved; at N=100k it's 40 GB; at N=150k it's 90 GB — impossible without fusion/tiling on a laptop.

### Metal kernels

- **Tanimoto tiled** (`tanimoto_metal_u32.py`) — 16×16 threadgroup tiles with shared memory, uint32-packed fingerprints, bit-parallel popcount. `union = cnt_i + cnt_j - inter` (no OR+popcount).
- **Fused Tanimoto→CSR** (`fused_tanimoto_nlist.py`) — Two-pass (count then fill), cutoff embedded in kernel source. Pre-computed per-row popcounts.
- **Neighbor list** (`butina_metal.py`) — Count + fill kernels from a pre-computed similarity matrix (used for testing/validation).

### MLX `metal_kernel` notes

- `grid` parameter = **total threads**, not number of threadgroups. For a 2D tiled kernel with TILE=16: `grid=(ceil(N/16)*16, ceil(N/16)*16, 1)`.
- 1D buffer inputs work correctly when read with simple indexing from global memory. Issues were observed when combining 1D buffers with threadgroup memory in tiled kernels — solved by embedding scalar constants (like cutoff) directly in the kernel source.

## Usage

```python
import mlx.core as mx
from mlxmolkit import (
    fp_uint8_to_uint32,
    fused_neighbor_list_metal,
    butina_from_neighbor_list_csr,
)

# fp_bytes: (N, 128) uint8 from RDKit Morgan FP (1024 bits)
fp_u32 = fp_uint8_to_uint32(mx.array(fp_bytes))
offsets, indices = fused_neighbor_list_metal(fp_u32, cutoff=0.4)
result = butina_from_neighbor_list_csr(offsets, indices, N, cutoff=0.4)
print(f"{len(result.clusters)} clusters")
```

Or the all-in-one wrapper:

```python
from mlxmolkit import butina_tanimoto_mlx

result = butina_tanimoto_mlx(mx.array(fp_bytes), cutoff=0.4)
```

## Benchmark

```bash
python compare_rdkit_mlx.py --n-max 20000
```

Compares the full pipeline (same fingerprints) against RDKit's `BulkTanimotoSimilarity` + `ClusterData`.

## Tests

```bash
pip install -e .
pytest tests/ -v
```

## References

- [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) — NVIDIA's CUDA implementation (Apache 2.0)
- [RDKit blog: Butina clustering with nvMolKit](https://greglandrum.github.io/rdkit-blog/posts/2026-02-28-nvmolkit-clustering.html) — Greg Landrum's benchmark
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework with Metal kernel support
- [Butina, D. (1999)](https://doi.org/10.1021/ci9803381) — Performance of Kier-Hall and molecular connectivity indices in clustering
