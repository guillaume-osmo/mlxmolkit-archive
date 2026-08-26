# Baseline: concurrent `import mlx.core` on macOS 26.2

Recorded before upgrading to macOS Tahoe 26.6.1, so the upgrade can be judged
against something. Machine: Apple M4 Pro, 24 GB, macOS 26.2 (25C56), Python
3.11.15.

Context: ml-explore/mlx#4268. A maintainer could not reproduce this on macOS
26.5.1 / 26.6 across five configurations including this exact chip and this
exact MLX version, which leaves the OS as the one variable not yet held fixed.

## Wall clock, N processes each doing one `import` (mlx_repro.py)

| workers | numpy | mlx 0.30.6 | mlx 0.32.0 |
|--------:|------:|-----------:|-----------:|
| 1  | 0.10 s | 0.15 s | 0.15 s |
| 2  | 0.09 s | 0.19 s | 0.22 s |
| 4  | 0.10 s | 0.48 s | 0.56 s |
| 8  | 0.12 s | 1.30 s | ~1.3-1.5 s |
| 14 | 0.19 s | **2.77 s** | **0.42 / 0.90 / 1.12 s** (erratic) |

mlx 0.30.6 is consistent run to run; 0.32.0 is not, and is oddly non-monotonic
(8 workers slower than 14).

## Shape of it (mlx_timeline.py, 14 workers, 0.30.6)

All fourteen enter at t=0.00 and each takes 2.42-2.69 s. Contended, not
serialised — a lock would give a staircase.

## Where the time goes (mlx_sample2.py, 0.30.6)

1705 samples on the main thread, one stack:

    mlx::core::random::key -> MetalAllocator -> metal::device -> Device::Device
      -> load_colocated_library -> newLibraryWithFile -> parseArchive
        -> LibraryWithFile::setPosition -> fseek -> read

Leaves: `__lseek` 1025, `__read_nocancel` 636 — 97% in seek/read on the
122 MB `mlx.metallib` (155 MB on 0.32.0).

On 0.32.0 the trigger differs — `compile_clear_cache()` ->
`CompilerCache::CompilerCache()` -> `metal::allocator()` -> same terminus —
but this was NOT independently confirmed: the check used `vmmap -summary`,
which does not list image names, so it proves nothing either way. Redo with
plain `vmmap` before claiming the upstream fix (PR #3278, v0.31.2) is
incomplete.

## After the upgrade

    python experiments/mlx_import_contention/mlx_repro.py

If 14 workers lands near numpy's ~0.2 s, the answer is an OS fix and #4268
should be closed. If it still shows 2.7 s on 0.30.6, the OS is not the
variable and the page-cache theory (24 GB here against the maintainer's 48 and
128 GB) is the next thing to test.
