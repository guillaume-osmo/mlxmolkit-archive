# Benchmarks

Measured tables for [mlxmolkit](../README.md). The README's
["Read this before you benchmark it"](../README.md#read-this-before-you-benchmark-it)
section holds the conclusions that change what you should *do*; this file is the
data behind them.

All measurements on Apple M3 Max unless stated.

---

## MMFF on Metal vs RDKit — the honest comparison

50 molecules × 50 conformers = **2500 conformers**, atoms 9–49 (mean 21), quiet
box (load 3.0), fresh `Chem.Mol` copies per method. Measured on `main` @
`0a1f2b3`.

| Method | total | ms/conf | vs 1T | vs 14T | mean abs ΔE | within 0.1 |
|---|---:|---:|---:|---:|---:|---:|
| RDKit 1T | 3.98 s | 1.59 | ref | — | — | — |
| **RDKit 14T** | **0.50 s** | 0.20 | 7.99× | 1.00× | 0.0000 | — |
| FUSED 200 it | **0.41 s** | **0.16** | 9.78× | **1.22×** | 1.742 | 800/2500 |
| FUSED 500 it | 0.84 s | 0.34 | 4.73× | 0.59× | 0.520 | 1431 |
| FUSED 1000 it | 1.59 s | 0.64 | 2.49× | 0.31× | 0.203 | 2019 |
| FUSED 2000 it | 3.10 s | 1.24 | 1.28× | 0.16× | 0.073 | 2326 |
| MULTI 1000 it | 3.39 s | 1.36 | 1.17× | 0.15× | **0.015** | **2470/2500** |
| MULTI 2000 it | 3.69 s | 1.48 | 1.08× | 0.13× | 0.015 | 2470 |

**Read the two right-hand columns together with the timing.** FUSED at 200
iterations is the only row that beats RDKit-14T, and it converges 800 of 2500
conformers. Matching RDKit's energies takes ~2000 iterations, and FUSED cost is
linear in iterations (0.41 / 0.84 / 1.59 / 3.10 s for 200 / 500 / 1000 / 2000).
At matched accuracy the Metal path is **~6× behind RDKit-14T** and roughly at
parity with RDKit single-threaded.

`MULTI` early-exit works — 1000 and 2000 iterations give identical accuracy for
+0.3 s. `FUSED` scales linearly, so its per-conformer early exit is not firing.
That is a small, concrete target.

---

## Conformer generation throughput

### N = 20 molecules, k = 50 conformers (1000 total)

| Pipeline | Time | Throughput | GPU memory |
|---|---:|---:|---:|
| DG only | 1.00 s | 1,002 conf/s | 2.6 MB |
| DG + ETK | 1.11 s | 900 conf/s | 2.6 MB |
| DG + ETK + MMFF | 1.63 s | 614 conf/s | 5.1 MB |

### Scale

| Scale | Pipeline | Time | Throughput | Convergence |
|---|---|---:|---:|---:|
| N=1000, k=10 | DG + ETK | 5.0 s | **2,017 conf/s** | 99.7% |
| N=1000, k=10 | DG + ETK + MMFF | 8.0 s | **1,250 conf/s** | 99.7% |
| N=10000, k=1 | DG + ETK | 17.7 s | 565 conf/s | 99.6% |
| N=10000, k=1 | DG + ETK + MMFF | 37.9 s | 264 conf/s | 99.6% |
| **N=10000, k=10** | DG + ETK | **38.1 s** | **2,625 conf/s** | 99.7% |
| **N=10000, k=10** | DG + ETK + MMFF | **67.9 s** | **1,473 conf/s** | 99.7% |

100,000 conformers in a single GPU batch, every stage on Metal — no RDKit
post-processing.

### Batch size (N=20, k=50, C=1000)

| Batch | Batches | Time | conf/s |
|---:|---:|---:|---:|
| 100 | 10 | 0.62 s | 1,610 |
| 500 | 2 | 0.29 s | 3,394 |
| 1000+ | 1 | 0.22 s | **4,442** |

Larger batches mean fewer kernel launches. Auto-sizing (the default) picks the
largest batch that fits in free memory.

### ETKDG variants (N=20, k=50)

| Variant | conf/s | Convergence |
|---|---:|---:|
| DG | 1,005 | 100.0% |
| KDG | 910 | 99.1% |
| ETDG | 1,009 | 100.0% |
| ETDGv2 | 1,008 | 100.0% |
| ETKDG | 907 | 99.1% |
| ETKDGv2 | 904 | 99.1% |
| ETKDGv3 | 895 | 99.1% |
| srETKDGv3 | 914 | 100.0% |

---

## Memory

### Per conformer, by molecule size

| Atoms | DG (4D) | ETK (3D) | MMFF (BFGS) | MMFF (L-BFGS) |
|---:|---:|---:|---:|---:|
| 5 | 1.9 KB | 1.4 KB | 1.3 KB | 1.4 KB |
| 12 | 4.4 KB | 3.3 KB | 6.0 KB | 3.3 KB |
| 21 | 7.6 KB | 5.7 KB | 17.2 KB | 5.7 KB |
| 30 | 10.8 KB | 8.1 KB | 34.1 KB | 8.1 KB |
| 50 | 18.0 KB | 13.5 KB | 92.0 KB | 13.5 KB |
| 64 | 23.1 KB | 17.3 KB | **149.2 KB** | 17.3 KB |

MMFF BFGS memory grows as O(n²) — the dense Hessian is `(n_atoms × 3)²`.

### Scaling with total conformers (DG + ETK + MMFF, batch = 500)

| Conformers | Batch | GPU (BFGS) | GPU (L-BFGS) | Time | Throughput |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1000 | 5.1 MB | 2.9 MB | 0.43 s | 2,342/s |
| 2,000 | 500 | 2.6 MB | 1.5 MB | 1.43 s | 1,402/s |
| 4,000 | 500 | 2.6 MB | 1.5 MB | 1.91 s | 2,094/s |
| 10,000 | 500 | **2.6 MB** | **1.5 MB** | 4.82 s | 2,075/s |

GPU memory stays constant regardless of total conformers — divide-and-conquer
batching. With 64 GB of unified memory a single batch holds ~9.8M conformers at
12 atoms, ~3.9M at 30, and ~1.8M at 64 (~300K under BFGS, where the Hessian
dominates).

### BFGS vs L-BFGS

| Molecule | Atoms (with H) | BFGS | L-BFGS | Winner |
|---|---:|---:|---:|---|
| Methane | 5 | 0.255 s | 0.215 s | BFGS |
| Benzene | 12 | 0.213 s | 0.222 s | ~tie |
| Aspirin | 21 | 0.241 s | 0.230 s | ~tie |
| Testosterone | 49 | 0.364 s | 0.335 s | BFGS |
| Cholesterol | 74 | 0.590 s | 0.486 s | BFGS |

**BFGS is faster at every drug-like size.** Better curvature information needs
fewer iterations, and that beats the memory saving until the Hessian exceeds
~1 MB/conformer — roughly 150 atoms with H, where the pipeline auto-switches.

### Adaptive iteration scaling

`max_iters = base + scale · max(n_atoms, √n_constraints)`

| Molecule | Atoms | Constraints | DG | ETK | MMFF |
|---|---:|---:|---:|---:|---:|
| Methane | 5 | 10 | 400 | 200 | 275 |
| Benzene | 12 | 66 | 540 | 270 | 380 |
| Aspirin | 21 | 210 | 720 | 360 | 515 |
| Testosterone | 49 | 1176 | 1280 | 640 | 935 |
| 64-atom | 64 | 2016 | 1580 | 790 | 1160 |

---

## Clustering

Enamine REAL subset:

| N | Fused sim→CSR | Butina | **Total** | vs RDKit | Memory |
|---|---:|---:|---:|---:|---:|
| 20k | 0.26 s | 0.09 s | **0.35 s** | **152×** | 0.1 MB |
| 50k | 1.26 s | 0.36 s | **1.62 s** | — | 0.5 MB |
| 100k | 4.87 s | 0.97 s | **5.84 s** | — | 1.3 MB |
| 150k+ | blockwise | — | scales | — | bounded |

---

## Conformer quality vs RDKit

### Rescored by RDKit's MMFF94 (k=20, ETKDGv2)

After MMFF optimization, mlxmolkit conformers reach the **same energy basins**:

| Molecule | Atoms | RMSD pre-MMFF | RMSD post-MMFF | ΔE |
|---|---:|---:|---:|---:|
| Benzene | 12 | 0.12 Å | **0.00 Å** | **0.0** |
| Aspirin | 21 | 0.98 Å | **0.00 Å** | **0.0** |
| Ibuprofen | 33 | 1.79 Å | **0.96 Å** | **0.6** |
| Acetaminophen | 20 | 0.99 Å | **0.00 Å** | **0.0** |

Bond and angle geometry match RDKit within 0.03 Å; the post-MMFF energy gap is
under 1 kcal/mol.

### Shape and electrostatic overlap (N=20, k=10, CHEESE charge-training subset)

Both sides ETKDGv3 + MMFF94:

| Metric | Value |
|---|---:|
| Mean best shape Carbo | 0.934633 |
| Mean best electrostatic Carbo | 0.996936 |
| Mean best combined score | 0.966551 |
| Median best heavy-atom RMSD | 0.449579 Å |

Scoring aligns all mlxmolkit conformers to all RDKit conformers with batched MLX
Horn quaternion alignment, then evaluates paired shape and electrostatic Carbo
in one MLX tensor pass per molecule — avoiding per-pair CPU Kabsch/SVD loops and
repeated tiny CHEESE kernel launches.

```bash
python tools/compare_rdkit_mlx_conformers.py \
  --limit 20 --n-conformers 10 \
  --out outputs/cheese_projection/rdkit_vs_mlx_conformers_20_k10_batched_scoring_after_fixes.csv
```

---

## NDDO

- Per-pair W tensor matches PYSEQM to **2.66e-15** (machine epsilon).
- 27/27 SCF charge tests pass against frozen PYSEQM/MOPAC references.
- numba JIT + einsum on the hot kernels (`w_withquaternion`,
  `GenerateRotationMatrix`, `Rotate2Center2Electron`) gives ~2× end-to-end per
  pair — 3.3 → 1.6 ms per S–C pair on M3 Max.
- Gradient and SCF performance: 4.1× and 2.9× respectively on cholesterol (#74).
