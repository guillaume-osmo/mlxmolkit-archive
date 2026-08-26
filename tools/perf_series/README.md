# perf_series — reproduce the performance audit

Replays one frozen workload across a series of commits so a speedup claim can be checked,
and so a change that quietly altered an answer shows up next to the timing that motivated it.

Report: [`benchmarks/PERFORMANCE_AUDIT.md`](../../benchmarks/PERFORMANCE_AUDIT.md).

## Files

| file | what it does |
|---|---|
| `make_geom.py` | writes `geom.npz` — cholesterol, testosterone, aspirin, thioanisole, ETKDG seed 20260814 then MMFF-relaxed |
| `geom.npz` | the frozen geometries, committed so every run measures byte-identical input |
| `run_bench.py` | NDDO SCF + gradient (PM6) and the MMFF GPU optimiser at whatever commit is on `PYTHONPATH` |
| `run_allmethods.py` | the same, crossed over every entry in `METHOD_PARAMS` × 4 molecules |
| `cold.py` | exactly one call per interpreter — what a single-point user actually pays |
| `drive.sh` | checks out each milestone into `mlxmolkit-bench` and runs `run_bench.py` |
| `cold.sh` | same series, driving `cold.py` three times per point |
| `results_warm.jsonl`, `results_cold.jsonl` | the measurements behind the report |

## Running it

```bash
git worktree add ../mlxmolkit-bench <sha>
conda activate rdkit_build_fb
python tools/perf_series/make_geom.py     # only if regenerating geom.npz
zsh tools/perf_series/drive.sh
zsh tools/perf_series/cold.sh
```

## Two traps this harness exists to avoid

**Run it on an idle machine.** The NDDO SCF is CPU-bound NumPy. The first pass of this audit
ran concurrently with other work at load average 10+, and produced a table in which adjacent
commits #73 and #74 read 387 ms and 709 ms — a fake 1.8x regression between neighbours. Every
CPU-bound number here is worthless if something else is running.

**Cold and warm measure different things, and the gap grows.** `min`-of-N inside one process
rewards memoisation (`pair_cache`, `precompute_pair_w`, cached parameter loads), so it partly
measures how much a commit caches rather than how much work it does. Measured cold/warm ratio
climbs monotonically along the series — 0.85 at `pre-mmff` to 1.67 at HEAD — and the headline
SCF speedup is 4.05x cold against 7.92x warm. Quote the cold figure unless the context is
explicitly repeated calls.

`run_bench.py` reports the minimum of N as its headline, since interference only ever adds
time, and carries the median and spread so a noisy point announces itself. It also records the
heat of formation, iteration count and gradient norm beside each timing, which is what makes
the series a correctness regression as well as a performance one.
