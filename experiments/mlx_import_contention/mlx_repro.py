"""Concurrent `import mlx.core` across processes serialises. Minimal repro."""
import time
from concurrent.futures import ProcessPoolExecutor

def import_numpy(_):
    import numpy
    return 1

def import_mlx(_):
    import mlx.core
    return 1

if __name__ == "__main__":
    import mlx.core as mx, platform, sys
    print(f"mlx {mx.__version__}  python {sys.version.split()[0]}  "
          f"macOS {platform.mac_ver()[0]}  {platform.machine()}")
    for fn, label in ((import_numpy, "import numpy "), (import_mlx, "import mlx.core")):
        for w in (1, 2, 4, 8, 14):
            t = time.perf_counter()
            with ProcessPoolExecutor(max_workers=w) as ex:
                list(ex.map(fn, range(w)))
            print(f"  {label}  {w:2d} worker(s): {time.perf_counter() - t:5.2f} s")
