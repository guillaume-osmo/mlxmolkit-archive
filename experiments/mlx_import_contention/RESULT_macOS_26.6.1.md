# Result: macOS 26.6.1 (25G76), Apple M4 Pro, 24 GB

Upgraded from 26.2 specifically to test ml-explore/mlx#4268, where a maintainer
could not reproduce on 26.5.1 / 26.6 and the OS was the one variable left.

Measured on a settled machine: disk idle (0.00 MB/s), no `mdworker`, `backupd`
or `photoanalysisd`. An earlier post-upgrade attempt was discarded — Spotlight
was reindexing at 3582 tps, and this defect is disk-bound.

## Wall clock, 14 processes each doing one import

| mlx | macOS 26.2 | macOS 26.6.1 |
|---|---|---|
| 0.30.6 | 2.77 s | **3.48 / 3.49 / 3.48 / 3.49 s** |
| 0.32.0 | 0.42-1.12 s | **1.19 / 0.77 / 0.69 s** |
| numpy (control) | 0.19 s | 0.15-0.19 s |

## Is the GPU driver loaded at import? (`vmmap`, images mapped)

| stage | mlx 0.30.6 | mlx 0.32.0 |
|---|---|---|
| before import | none | none |
| after `import mlx.core` | **AGXMetal**, Metal, MetalTools, MPS | Metal, MetalTools, MPS — **no AGXMetal** |
| after touching an array | AGXMetal, Metal, MetalTools, MPS | AGXMetal, Metal, MetalTools, MPS |

## Conclusions

1. **The OS is not the variable.** 26.6.1 reproduces 0.30.6's behaviour and is
   slightly worse, on a quieter machine than the 26.2 baseline.
2. **PR #3278 is the fix.** On 0.32.0 `import mlx.core` no longer maps the GPU
   driver — AGXMetal appears only once an array is touched — and the import
   storm drops 3.48 s -> ~0.7-1.2 s.
3. **Retracted:** the earlier claim that 0.32.0 still reaches `Device::Device()`
   at import, based on one `sample` trace showing `compile_clear_cache`. `vmmap`
   is the direct check and contradicts it. The trace was read with Python frames
   filtered out, so its context was not established.
4. **Residual, minor:** 0.32.0 is still 4-7x the numpy control and does map
   Metal.framework/MetalTools at import. Smaller and separate; not the 122 MB
   metallib parse.

Fix for a consumer: require mlx >= 0.31.2. Deferring the import (as
mlxmolkit does) avoids it on any version.
