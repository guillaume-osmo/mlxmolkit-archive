# mlxmolkit performance audit — M3 Max

Two questions per subsystem: **where are we now**, and **what is the single highest-value
next change**. Workloads are the ones the existing test suite already uses, not invented
benchmarks. Every number below is either measured on this machine or quoted with its source.

Machine: Apple M3 Max, macOS 15, Python 3.11, conda env `rdkit_build_fb`,
numpy 1.26.4 (OpenBLAS64), numba 0.65.1, mlx 0.31.2.
Commit under audit: `0a1f2b3` (#74).

---

## Method, and one thing I got wrong first

Geometries are frozen once into `geom.npz` (cholesterol 74 atoms, testosterone 49,
aspirin 21, thioanisole 16 — the d-orbital path) so every commit in the series measures
byte-identical input.

The first pass was **contaminated and thrown away**. I ran the commit-series benchmark
concurrently with eight audit agents, reasoning they were read-only. Read-only is still
CPU-heavy, load average hit 10+, and the NDDO SCF is CPU-bound NumPy. The tell was in the
data: adjacent commits `#73` and `#74` read 387 ms and 709 ms, where `#74` should be the
faster of the two. Timings for anything CPU-bound must be taken on an otherwise idle
machine.

The harness now reports the **minimum** of N repetitions rather than the mean — for
CPU-bound work interference only ever adds time, so the minimum is the least contaminated
estimate — and carries the median and spread so a noisy point announces itself instead of
being averaged into a plausible-looking trend. It also records the SCF heat of formation,
iteration count and gradient norm next to each timing, so the series is a **correctness**
regression as well as a performance one.

---

## 1. Regression: what the last two weeks actually bought

### 1.1 Warm (min of 5 calls in one process), ms

| milestone | SCF chol | grad chol | SCF thio | grad thio | MMFF BFGS | MMFF L-BFGS |
|---|---:|---:|---:|---:|---:|---:|
| `pre-mmff` | 924.1 | 3858.1 | 205.5 | 518.6 | 139.1 | 144.7 |
| `pr55-armijo` | 841.2 | 3827.6 | 208.9 | 518.3 | 125.4 | 128.9 |
| `pr56-iters500` | 894.2 | 3724.4 | 206.3 | 515.8 | 125.6 | 129.0 |
| `pr61-batch1000` | 835.1 | 3490.0 | 205.7 | 513.6 | 125.5 | 128.8 |
| `pr64-gxtb-merge` | 833.3 | 3493.9 | 206.5 | 516.3 | 125.5 | 128.9 |
| `pr65-conv-fix` | 901.4 | 3573.4 | 206.7 | 516.3 | 125.5 | 129.1 |
| `pr66-eigvec-follow` | 849.0 | 3508.5 | 207.7 | 519.8 | 125.5 | 128.9 |
| `pr67-overlap-norm` | 812.8 | 4012.9 | 206.0 | 516.9 | 125.4 | 129.0 |
| `pr70-grad3x` | 675.2 | 1844.7 | 197.9 | 491.6 | 125.6 | 129.0 |
| `pr71-fock-shape` | 272.9 | 1199.1 | 187.2 | 468.4 | 125.4 | 129.1 |
| `pr72-kernel-off` | 294.3 | 1230.7 | 188.0 | 466.6 | 125.5 | 128.8 |
| `pr73-perpair-grad` | 288.4 | 824.7 | 185.5 | 379.7 | 125.8 | 128.9 |
| `pr74-head` | 116.7 | 325.9 | 90.0 | 195.3 | 125.5 | 129.2 |

### 1.2 Cold vs warm — and why the difference is itself a finding

One call per fresh interpreter, median of 3, cholesterol, PM6:

| milestone | cold SCF | warm SCF | cold/warm | cold grad | warm grad | cold/warm |
|---|---:|---:|---:|---:|---:|---:|
| `pre-mmff` | 787.9 | 924.1 | 0.85 | 4511.2 | 3858.1 | 1.17 |
| `pr70-grad3x` | 672.6 | 675.2 | 1.00 | 2010.7 | 1844.7 | 1.09 |
| `pr71-fock-shape` | 330.9 | 272.9 | 1.21 | 1395.5 | 1199.1 | 1.16 |
| `pr73-perpair-grad` | 347.6 | 288.4 | 1.21 | 943.9 | 824.7 | 1.14 |
| `pr74-head` | 194.5 | 116.7 | **1.67** | 472.6 | 325.9 | **1.45** |

The ratio climbs monotonically toward HEAD. That is the signature of accumulating
memoisation — `pair_cache`, `precompute_pair_w`, cached parameter loads — and it means a
warm benchmark measures *how much a commit caches* as much as how much work it does.

**Headline speedups, `pre-mmff` → HEAD:**

| | cold (single-shot) | warm (repeated) |
|---|---:|---:|
| SCF cholesterol | **4.05x** | 7.92x |
| gradient cholesterol | **9.55x** | 11.84x |

Both are real and they answer different questions. A user computing one molecule pays the
cold number; a batch or an optimiser trajectory pays the warm one. Quote the cold figure
unless the context is explicitly repeated calls — my first draft of this report quoted 7.92x
and would have overstated single-shot SCF by nearly 2x.

### 1.3 Correctness across the series

The gradient norm on cholesterol is **constant across all 13 commits** — bit-stable through
a 9.55x speedup. That is the strongest single result here.

Two MMFF energies move, both expected and both mine: L-BFGS 185.3357 -> 185.3648 and BFGS
185.2696 -> 185.2697 at `pr55-armijo`, which changed the summation order inside the Armijo
test by design.


---

## 2. Per-subsystem audit

### NDDO SCF

**Current state.** 100% CPU/NumPy float64 for the sequential path. Per iteration:
`_build_fock` → `np.linalg.eigh` → density → adaptive mixing, with Pulay DIIS inline.
The batched path (`rm1_energy_batch_mlx`) is MLX float32 with the Fock build on Metal;
its `conv_tol` is floored at 1e-5 because float32 cannot reach 1e-6.

Worth stating plainly: **"10 methods" is 6 distinct SCFs.** PM6, PM6_D, PM6_D3, PM6_D3H4
and PM6_D3H4X all share one `PM6_FULL_PARAMS` object (`methods.py:632-647`); the D3/H4/X
variants differ only by a post-SCF correction on the heat of formation.

**Dominant cost — d-bearing molecules.** `precompute_pair_w` (`scf.py:581-582`) filters to
`n_basis <= 4`, so d-bearing pairs never enter the hoisted cache. `_build_fock` therefore
passes `w=None` and `rotate_integrals_to_molecular_frame` re-runs every iteration — for a
quantity that depends only on the geometry and is invariant across the whole SCF. On
thioanisole, **255 of 270 rotation calls (94%) recompute 15 distinct answers**, costing
0.092 s of a 0.183 s single point — half the SCF.

**Next improvement.** Stop filtering d pairs out of `precompute_pair_w`: add one scalar
`rotate_integrals_to_molecular_frame` per d-bearing pair, stored under the same `(i, j)`
key. `_pair_fock_twocentre` then receives `w` and skips the recomputation. This is exactly
the hoist #74 already did for sp pairs — the d case was simply missed.

**Dominant cost — sp molecules.** `np.linalg.eigh`, 20% of menthol's 44 ms and the single
largest profile line. O(n³), so it grows with basis size. DIIS is *not* the bottleneck
(~1.5%).

**Regression risk.** The GPU-Jacobi thresholds are stale: `_addons_batched_eigh_max_n()`
reads `JACOBI_MAX_N` dynamically and the installed `mlx_addons` reports **96**, but
`_scf_sign_min_basis()` hardcodes **33**. Also `build_fock_batch_cpu` — the
`use_metal=False` path every parity test uses — is a 6-deep Python loop measured at
**4× to 27× slower than the single-molecule path** on the same molecule and method.

### NDDO gradient and optimiser

**The finding that reframes the whole perf effort: every PR body benchmarks PM6, while
every entry point defaults to RM1.** For the default method the scalar core-core loop is
now the largest line in the gradient — `anal_grad.py:365-369` is a Python comprehension
over all N(N−1)/2 pairs calling `pair_repulsion_for_method` one pair at a time, six times
per gradient.

**Next improvement.** Add a vectorised `am1_pair_repulsion_batch` mirroring the existing
`pwcct.pm6_pair_repulsion_batch`, and call it from `anal_grad.py:357-369`. #74 already made
exactly this edit on the PM6 side of the same function; the template exists.
**Expected ~2.0× on a 74-atom RM1/AM1/PM3 gradient**, growing as N².

**Correctness risk, flagged ahead of performance.** PM6-ORG's gradient differentiates a
different energy than its SCF minimises — `nuclear_repulsion_for_method` special-cases
`PM6_ORG`, but `pair_repulsion_for_method` does not.

### NDDO integrals (issue #21)

**Current state.** `prepare_batch` allocates **0 bytes of MLX device memory** — re-verified
on `0a1f2b3`. The integrals are entirely CPU/NumPy. Issue #21's headline is now *more* true
than when filed: the surrounding code got 4.9× faster while the integrals did not move.

**Dominant cost.** The per-pair Python loop at `batch.py:454-504` — 30.1% (499 ms of
1656 ms), the single largest line item, larger than any integral primitive. It is pure
bookkeeping: every value it writes was already computed in bulk above it.

**Next improvement.** The work is **already done and unmerged.** Commits `adcf7d1` and
`a31b386` on `origin/guillaume/scf-iterations-and-metal` descend from this exact HEAD and
delete that loop. Measured on the branch, 200-molecule batch: `prepare_batch` 694 → 550 →
532 ms. This is a merge, not a port.

### MMFF94

**Dominant cost.** The thread-0 serial gradient inside the threadgroup kernels. Per
iteration the threadgroup runs one `SEQ_COMPUTE_EG` with all N_terms evaluated by a single
thread while 31 stall on the barrier, against line-search energies that are already striped
32-wide.

**Next improvement.** Parallelise the gradient the way `PAR_COMPUTE_E` already is: give
each thread a gradient replica, stripe every term loop by `tid`, then tree-reduce. Amdahl
with f = 0.91–0.96 at 32-wide predicts 8–14× on kernel time before barrier overhead.

**Dead code.** `mmff_minimize.py:26` reads `mmff_bfgs_source.metal` (262 lines) at import;
neither `_mmff_kernel` nor `MAX_ATOMS_METAL` is ever used.

### Conformer generation

**Dominant cost.** The same thread-0 pattern: `conformer_metal.py:313-326` and `:487-499`
run every distance/chiral/fourth term serially on `tid==0`, with 8 device loads and 8
read-modify-writes per term on a single active lane — latency fully exposed.

**Next improvement.** Replace it with a per-atom CSR stripe built once in
`pack_shared_dg_batch`, so each thread loops only over its own atom's terms. Estimated
2–3× on the DG and ETK kernels, 1.5–2× end-to-end with `run_mmff=False`.

**Regression risk.** A measured 1.4× was silently reverted and never re-measured — commit
`51f094b` gated the Stage-1b DG retry on `0 < fail_frac <= 0.05` and dropped it to 1×
iterations.

### Fingerprints and clustering

**Dominant cost.** Every MLX kernel emulates popcount with a 12-operation SWAR bit-hack
(`fused_tanimoto_nlist.py:39-43`, `:68-72`, and two more files) instead of Metal's
`popcount()` intrinsic. At N=100k the README splits 5.84 s as 4.87 s (83%) in sim→CSR.

**Next improvement.** Use `popcount()`. The inner loop collapses from a dozen shifts and
masks to one intrinsic. **Expected 2–4× on the sim→CSR stage**, N=100k 5.84 s → ~2.3–3.4 s.

**The fast path already exists and is unreachable.** `metal/fused_tanimoto.metal` *does*
use real `popcount()`, and the tracked `libfused_metal.dylib` / `fused_tanimoto.metallib`
are the fastest-by-construction path — zero JIT, one ctypes call — but nothing in
`butina_tanimoto_mlx` can reach them.

### xtb / g-xTB

**Dominant cost.** Python-level primitive-integral loops. For water, 936
`_primitive_overlap` calls and as many `_multipole_primitive`, each entering
`_augmented_overlap_axis` three times and allocating a fresh small array per axis — roughly
5800 `np.zeros` and 1900 `np.sum` per single point. Interpreter and allocator overhead; the
FLOPs are negligible.

**Next improvement, and the largest single number in this audit.** The C++ kernel is
already in the tree and unused. Replacing two call sites — `gxtb_basis.py:264` and
`gxtb_aes.py:68` — with `multipole_matrices_cpp`, guarded by its existing `CPP_AVAILABLE`,
is **measured at 282× on water** (13.78 ms → 0.049 ms) and 259× on H2S, with
max|diff| = 4.44e-16.

**Orphaned work.** `gxtb_overlap_batched.py` advertises itself as replacing "the dominant
g-xTB cost (~50% of SCF wall time)" and the profile confirms the diagnosis — but nothing in
`mlxmolkit/` imports it.

### COSMO / solvation and charges

**Dominant cost.** `surface._jfa_edt` (`surface.py:155-200`) — 26 neighbour offsets ×
(log2(max_dim)+2) passes, each offset costing ~31 MLX ops because `_shift` is a `mx.pad`
plus slice, i.e. a materialised copy per axis. Almost entirely dispatch overhead.

**Next improvement.** Replace the jump-flooding EDT with a probe-truncated separable
squared EDT: three 1-D lower-envelope passes over a window of
`ceil(probe_radius/grid) + 1` voxels. Truncation is safe because `phi` is only ever used
through the sign test `phi < 0`. Roughly **150–200× fewer MLX dispatches**.

**Duplication.** Two SES implementations compute the same quantity from the same inputs
with different radii tables, and the GPU one may be the slower.

---

## 3. The pattern worth naming

Four of the eight subsystems have the same shape: **the fast path already exists and is
not wired up.**

| subsystem | the fast thing | why it is not used |
|---|---|---|
| xtb / g-xTB | `multipole_matrices_cpp`, measured 282× | two call sites still point at the Python loop |
| NDDO integrals | `adcf7d1` + `a31b386`, measured 1.30× | sitting unmerged on a branch off this HEAD |
| clustering | `fused_tanimoto.metal` with real `popcount()` | unreachable from `butina_tanimoto_mlx` |
| g-xTB overlap | `gxtb_overlap_batched.py` | imported by nothing in `mlxmolkit/` |

None of these needs new algorithm work. They need wiring, a merge, and a four-line kernel
edit. That is where the cheapest wall-clock in the project currently sits.

---

## 4. Ranked next actions

1. **Wire the g-xTB C++ multipole kernel** — 282× measured, two call sites, existing
   fallback pattern.
2. **Merge `adcf7d1` + `a31b386`** — 1.30× on `prepare_batch`, already measured, descends
   from HEAD.
3. **Hoist the d-pair rotation in `precompute_pair_w`** — removes 94% redundant work worth
   half the SCF on d molecules; ~5 lines, mirrors the existing sp hoist.
4. **Vectorise the non-PM6 core-core in the gradient** — ~2× for RM1/AM1/PM3, which is what
   the default entry points actually run.
5. **`popcount()` in the Tanimoto kernels** — 2–4× on the dominant clustering stage.
6. **Parallelise the thread-0 gradients** in the MMFF and conformer kernels — largest
   theoretical win (8–14×) but the most work and the most risk.

Two correctness items outrank all of the above: **PM6-ORG's gradient differentiates a
different energy than its SCF minimises**, and the stale GPU-Jacobi threshold (33 hardcoded
against an installed capability of 96).
