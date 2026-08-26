"""Per-worker timings: is `import mlx.core` serialised, or just slow for all?"""
import os, time
from concurrent.futures import ProcessPoolExecutor

def noop(_):
    return os.getpid()

def worker(_):
    start = time.time()
    import mlx.core            # noqa: F401
    return os.getpid(), start, time.time()

if __name__ == "__main__":
    W = 14
    with ProcessPoolExecutor(max_workers=W) as ex:
        list(ex.map(noop, range(W * 3)))          # start every worker first
        rows = sorted(ex.map(worker, range(W)), key=lambda r: r[1])
    base = min(r[1] for r in rows)
    print(f"{'pid':>8s} {'enter (s)':>10s} {'exit (s)':>9s} {'duration':>9s}")
    for pid, s, e in rows:
        print(f"{pid:8d} {s-base:10.2f} {e-base:9.2f} {e-s:9.2f}")
    span = max(r[2] for r in rows) - base
    dur = [r[2] - r[1] for r in rows]
    print(f"\ntotal span {span:.2f} s; per-worker import {min(dur):.2f}-{max(dur):.2f} s")
    print("SERIALISED: each waits its turn" if max(dur) < 0.6 * span
          else "CONTENDED: all slow together")
