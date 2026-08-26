"""One measurement per process: what a user actually pays for a single-point call.

min-of-N inside one process rewards implementations that memoise across calls
(pair_cache, lru_cache on parameter loads). That flatters whichever commit added
the most caching rather than measuring the work itself. This does exactly one
call per interpreter and is driven N times from outside.
"""
import json, os, sys, time
import numpy as np

G = np.load("/Users/tgg/Github/_mlxmolkit_safety/bench/geom.npz")
name = os.environ.get("COLD_MOL", "cholesterol")
what = os.environ.get("COLD_WHAT", "scf")
Z, R = G[f"{name}_Z"].tolist(), G[f"{name}_R"]

from mlxmolkit.nddo import nddo_energy, nddo_gradient

t0 = time.perf_counter()
if what == "scf":
    nddo_energy(Z, R, method="PM6")
else:
    nddo_gradient(Z, R, method="PM6")
print(json.dumps({"label": os.environ.get("BENCH_LABEL", "?"), "mol": name,
                  "what": what, "ms": round((time.perf_counter() - t0) * 1e3, 1)}))
