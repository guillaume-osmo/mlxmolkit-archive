# Negative results

Things that were tried, measured, and did **not** pay. Each one is here so it is
not re-attempted from first principles, and each has a runnable script so the
verdict can be re-checked when the hardware or MLX changes.

Every number below was measured on this machine (Apple Silicon, MLX 0.30.6,
numpy 2.4.6). Run with the project on `PYTHONPATH`:

```bash
PYTHONPATH=. python experiments/<script>.py
```

The failed branches are kept on the remote rather than deleted:
`guillaume/nddo-per-orbital` carries the per-orbital vectorisation and the
float32 MLX rotation gating.

---

## 1. float64 on Metal is impossible; pseudo-float48 emulates it, and still loses

`experiments/float48_double_single.py`, `experiments/float48_at_scale.py`

MLX cannot do float64 on the GPU — the operation raises
`ValueError: float64 is not supported on the GPU`. float64 works only on MLX's
**CPU** stream, and there it is **4× slower than numpy** (scatter into
`(P,4,4,4,4)`: numpy 84.7 ms, mlx-cpu 350.5 ms, mlx-gpu 2.96 ms).

So double-single arithmetic was tried: carry `x` as `hi + lo`, both float32, via
Dekker/Knuth two-sum and two-product. **It works** — the real risk was that
Metal's compiler would contract or reassociate the error terms away, since the
algorithms are valid only under exact IEEE rounding, and a fast-math compiler
would silently degrade them to float32 while still appearing to run. It doesn't:

| | float32 | df32 |
|---|---|---|
| add | 1.11e-07 | 7.02e-15 |
| mul | 1.61e-07 | 1.44e-14 |

**46 effective mantissa bits** against float32's 24 and float64's 53.

But the emulation costs 5–25× the float32 time, which is more than the GPU's
entire margin over CPU float64:

| P | ops/elem | numpy f64 | mlx f32 | mlx df32 | df32 vs numpy |
|---|---|---|---|---|---|
| 102k | 5 | 0.18 ms | 0.30 ms | 1.77 ms | 0.10× |
| 102k | 60 | 1.86 ms | 0.68 ms | 7.38 ms | 0.25× |
| 1M | 20 | 16.84 ms | 1.24 ms | 27.78 ms | 0.61× |
| 1M | 60 | 49.65 ms | 3.27 ms | 81.13 ms | 0.61× |

Never above 0.68×.

**Reconsider if:** a future MLX exposes float64 on Metal, or a fused
`metal_kernel` implementation of df32 (one kernel, values kept in registers)
brings the overhead under ~3× — the arithmetic here goes through separate MLX
ops, each round-tripping to memory.

## 2. The GPU is not slow, our batches are small

`experiments/mlx_vs_numpy_shapes.py`

Same data, read the `mlx f32` column across sizes:

| P | ops | mlx f32 vs numpy f64 |
|---|---|---|
| 102k | 60 | **2.7×** |
| 1M | 60 | **15.2×** |

An 800-molecule batch is ~102k pairs, where the GPU is barely ahead. This is why
porting the integral block to MLX returned nothing end to end — the work was
real, there was just nowhere near enough of it to amortise dispatch.

Separately, elementwise arithmetic — most of what `overlaps` and `tetci` do — is
only **1.32×** on GPU. It is memory-bound; the CPU is already at bandwidth. The
one operation that wins big is **scatter, 28.9×** (84.7 → 2.93 ms).

Precision is *not* the obstacle: end to end, a float32 rotation left the energy
error against the float64 sequential solver identical to four digits
(1.168e-03 eV), because the Metal Fock kernel's float32 dominates regardless.

**Reconsider if:** batches reach ~1M pairs (~8000 molecules), or the per-shape
groups are fused so one dispatch covers all of them.

## 3. Routing a gradient's 6N displacements through `prepare_batch` is size-dependent

`experiments/batched_vs_incremental_gradient.py`

The plan in #12 was that the 6N displaced geometries share one topology and one
frozen density, making them the ideal `prepare_batch` input. But the incremental
path refreshes only the N−1 pairs that moved — O(N) per displacement — while
`prepare_batch` rebuilds all N(N−1)/2:

| | incremental | prepare_batch (6N geoms) | |
|---|---|---|---|
| ethanol (9) | 38.5 ms | 21.3 ms | batch wins 1.8× |
| benzaldehyde (14) | 116.1 ms | 89.7 ms | batch wins 1.3× |
| thioanisole (16) | 421.0 ms | 188.9 ms | batch wins 2.2× |
| menthol (31) | 422.2 ms | 932.9 ms | **incremental wins 2.2×** |

Crossover ~20 atoms, and the `prepare_batch` column excludes the energy
evaluation, so it flatters the batch.

## 4. Vectorising `prepare_batch`'s per-orbital work — 0.2%

Branch `guillaume/nddo-per-orbital`.

Building `basis_to_atom`/`basis_type`, the five-parameter-per-atom copy, and the
branching `H` diagonal fill were replaced by one flat atom axis and one flat
orbital axis across the batch. Measured: mixed 1.649 → 1.621 s, inside a 1.15×
run-to-run spread; `prepare_batch` tottime 0.696 → 0.694 s under cProfile.

Wall-clock section timers say why — the per-orbital work is **2.6 ms of 1656 ms,
0.2%**:

```
MAIN MOLECULE LOOP        499 ms  30.1%
overlaps                  301 ms  18.2%
rotate_pairs sp           282 ms  17.0%
tetci d                   220 ms  13.3%
packing                   102 ms   6.1%
pair table + spec lists    89 ms   5.4%
attraction                 61 ms   3.7%
resonance                  47 ms   2.8%
core-core                  27 ms   1.6%
padding + assembly         26 ms   1.6%
per-orbital precompute    2.6 ms   0.2%
```

## 5. Iteration count *is* predictable from 2D descriptors — and sorting on it still loses

`experiments/iteration_count_vs_flexibility.py`

MMFF L-BFGS iteration counts span **65×** over 40 drug-like molecules (20 for
benzene, 1300 for hexadecane), and the spread is not noise — it tracks
**flexibility**, not size:

| descriptor | Pearson r |
|---|---|
| rotatable bonds | **0.807** |
| atoms (with H) | 0.791 |
| MW | 0.682 |
| **rings** | **−0.327** |
| TPSA | 0.079 |

Rings correlate *negatively*: rigid aromatics are the cheapest things in the set
(benzene 20, chlorobenzene 24, naphthalene 36), floppy chains the most expensive.
Physically sensible — soft torsional modes condition the Hessian badly.

So difficulty is predictable a priori, which suggests sorting molecules into
difficulty-matched batches so one straggler cannot hold up a dispatch. **It does
not work.** On `mmff_optimize_metal_multi_mol`, 40 molecules:

| | ms | vs one dispatch |
|---|---|---|
| **one dispatch of 40** | **2535** | **1.00×** |
| 4 buckets, by atom count | 5669 | 0.45× |
| 4 buckets, by difficulty | 7023 | 0.36× |
| 4 buckets, random | 7310 | 0.35× |

Kernel-launch overhead dwarfs any straggler saving, so splitting a batch costs
2–3× no matter how it is split. Atom-count sorting being the least-bad split
(0.45× vs 0.35×) does confirm padding-to-`max_dim` is real — just an order of
magnitude smaller than a launch. **The lever points the other way: make batches
bigger.**

On `mmff_optimize_molecules_batch` sorting cannot help even in principle — it is
a Python `for` loop over molecules (10 benzenes cost 10.8× one benzene), so cost
is additive and there is no straggler to eliminate.

What the flexibility finding *did* buy: it exposed that the Metal multi-mol path
ran all `max_iters` unconditionally. Fixing that (per-conformer `grad_tol` check
at the existing sync points) made `max_iters` a cap rather than a cost there —
2.65× at `max_iters=2000`, energies unchanged to 6.7e-06 kcal/mol.

**Reconsider if:** dispatch overhead falls far enough that a launch is cheap
relative to the iteration work — then straggler-matching becomes the dominant
term and the ranking above should flip.

## 6. A strong-Wolfe line search is *not* what separates us from MOPAC

`mlxmolkit/nddo/gradient.py` (`_strong_wolfe`), landed anyway — see below.

MOPAC's L-BFGS uses a Wolfe line search (`lnsrlb` → `dcsrch`, `ftol=1e-3`,
`gtol=0.9`), verified against the source; ours enforced Armijo only. On menthol
MOPAC's L-BFGS converges in 80 cycles where ours took 94, so the missing
curvature condition looked like the obvious explanation for the 18% gap.

It isn't. Total gradient calls, same molecules, same start:

| molecule | Armijo | strong Wolfe |
|---|---|---|
| ethanol | 52 | 52 |
| benzaldehyde | 26 | 26 |
| chlorobenzene | 19 | 18 |
| thioanisole | 51 | 51 |
| menthol | 97 | 94 |
| **total** | **245** | **241** (−1.6%) |

Instrumenting the search says why: **the unit step already satisfies both
Wolfe conditions in ~94% of line searches** (44/47 on ethanol, 84/88 on
menthol), and the `sy <= 0` count that a curvature condition exists to prevent
was already **zero**. There was no violation to fix.

It was landed regardless, because it costs nothing (−1.6% gradients), it
replaces a `step = 1e-4` stall-fallback with the best decreasing step found,
and it makes the `sy > 0` guard structural rather than lucky. But it is
insurance, not a speedup, and **the gap to MOPAC is elsewhere** — most likely
EF's explicit Hessian (`gethes` + Powell/BFGS updates + P-RFO) against our
limited-memory approximation. That is the next thing to try, and unlike here
the cycle counts justify it: MOPAC EF takes 75 cycles on menthol against its
own L-BFGS's 80 and our 94.

**Reconsider if:** a molecule class shows up where `sy <= 0` is common — the
guard then starts earning its keep.

---

## Method note

**Rebuild the inputs inside every timed run.** The MMFF optimizers mutate their
RDKit molecules in place, so a parameter sweep that builds the molecules once
and reuses them feeds each run the previous run's optimised geometry. The tell
is an impossible result — iteration counts that *fall* as the budget rises
(103/277/401 at `max_iters=500`, then 3/3/63 at 2000). Treat non-monotonicity in
a monotonic-by-construction sweep as harness contamination before believing it.

**Do not size an optimisation target with cProfile in this codebase.** Its
per-call overhead inflates anything doing many small Python statements: it put
0.7 s of `tottime` on `prepare_batch`'s body where wall-clock section timers
found the section in question was 2.6 ms. cProfile is good for *finding* which
function is hot; use wall-clock timers around explicit sections to decide
whether it is worth attacking, and always A/B in one session — this machine
drifts several percent between invocations, which is larger than several of the
wins recorded above.
