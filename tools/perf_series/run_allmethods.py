"""Per-method audit of the NDDO stack at whatever commit is on PYTHONPATH.

Covers every method in METHOD_PARAMS, not just PM6, and reports correctness next to
cost so a speedup that broke a method cannot hide. Workload molecules are the ones the
existing tests already use (thioanisole is the d-orbital path; cholesterol is what the
upstream perf PRs quote).

    BENCH_LABEL=head PYTHONPATH=<worktree> python run_allmethods.py
"""
import json, os, time
import numpy as np

B = "/Users/tgg/Github/_mlxmolkit_safety/bench"
G = np.load(f"{B}/geom.npz")
LABEL = os.environ.get("BENCH_LABEL", "?")
MOLS = os.environ.get("BENCH_MOLS", "thioanisole,aspirin,testosterone,cholesterol").split(",")

from mlxmolkit.nddo import nddo_energy, nddo_gradient
from mlxmolkit.nddo.methods import METHOD_PARAMS

rows = []
for method in sorted(METHOD_PARAMS):
    for name in MOLS:
        Z, R = G[f"{name}_Z"].tolist(), G[f"{name}_R"]
        rec = {"label": LABEL, "method": method, "mol": name, "n_atoms": len(Z)}
        try:
            t0 = time.perf_counter()
            out = nddo_energy(Z, R, method=method)
            rec["scf_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            # nddo_energy returns a dict; pull whatever energy key exists
            if isinstance(out, dict):
                for k in ("heat_of_formation", "E_total", "energy", "Etot"):
                    if k in out:
                        rec["E"] = round(float(out[k]), 6)
                        rec["E_key"] = k
                        break
                rec["scf_iters"] = out.get("n_iter") or out.get("iterations")
                rec["converged"] = bool(out.get("converged", True))
        except Exception as e:
            rec["scf_err"] = f"{type(e).__name__}: {str(e)[:60]}"
        try:
            t0 = time.perf_counter()
            e, g = nddo_gradient(Z, R, method=method)
            rec["grad_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            rec["grad_norm"] = round(float(np.linalg.norm(np.asarray(g))), 6)
        except Exception as e:
            rec["grad_err"] = f"{type(e).__name__}: {str(e)[:60]}"
        rows.append(rec)
        print("ROWJSON " + json.dumps(rec), flush=True)

print(f"# {len(rows)} rows over {len(METHOD_PARAMS)} methods x {len(MOLS)} molecules")
