"""Measure NDDO SCF / gradient and the MMFF GPU optimizer at whatever commit is on PYTHONPATH.

Frozen geometries from geom.npz so every commit sees identical input.
Emits one JSON line so results can be collected across a commit series.
"""
import json, os, sys, time
import numpy as np

B = "/Users/tgg/Github/_mlxmolkit_safety/bench"
G = np.load(f"{B}/geom.npz")
LABEL = os.environ.get("BENCH_LABEL", "?")
res = {"label": LABEL}


def timed(fn, reps, warmup=1, budget=240.0):
    """Return (min_ms, median_ms, spread_pct) over independent reps.

    Reports the MINIMUM as the headline: for CPU-bound work the minimum is the
    least contaminated estimate, since interference only ever adds time. The
    spread is carried so a noisy measurement announces itself instead of being
    quietly averaged into a fake trend.
    """
    t0 = time.perf_counter()
    for _ in range(warmup):
        fn()
    solo = time.perf_counter() - t0
    if solo * reps > budget:
        reps = max(1, int(budget / max(solo, 1e-9)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    lo = ts[0]
    med = ts[len(ts) // 2]
    spread = (ts[-1] - lo) / lo * 100.0 if lo > 0 else 0.0
    return lo, med, spread, len(ts)


def bench_nddo():
    from mlxmolkit.nddo import nddo_energy, nddo_gradient
    for name in ("cholesterol", "thioanisole"):
        Z, R = G[f"{name}_Z"].tolist(), G[f"{name}_R"]
        try:
            lo, med, sp, n = timed(lambda: nddo_energy(Z, R, method="PM6"), 5)
            out = nddo_energy(Z, R, method="PM6")
            if isinstance(out, dict):
                for k in ("heat_of_formation", "E_total", "energy", "Etot"):
                    if k in out:
                        res[f"scf_{name}_E"] = round(float(out[k]), 6)
                        break
                res[f"scf_{name}_iters"] = out.get("n_iter") or out.get("iterations")
            res[f"scf_{name}_ms"] = round(lo, 1)
            res[f"scf_{name}_med"] = round(med, 1)
            res[f"scf_{name}_spread"] = round(sp, 1)
        except Exception as e:
            res[f"scf_{name}_ms"] = None
            res[f"scf_{name}_err"] = f"{type(e).__name__}: {str(e)[:70]}"
        try:
            lo, med, sp, n = timed(lambda: nddo_gradient(Z, R, method="PM6"), 5)
            _e, _g = nddo_gradient(Z, R, method="PM6")
            res[f"grad_{name}_norm"] = round(float(np.linalg.norm(np.asarray(_g))), 8)
            res[f"grad_{name}_ms"] = round(lo, 1)
            res[f"grad_{name}_med"] = round(med, 1)
            res[f"grad_{name}_spread"] = round(sp, 1)
        except Exception as e:
            res[f"grad_{name}_ms"] = None
            res[f"grad_{name}_err"] = f"{type(e).__name__}: {str(e)[:70]}"


def bench_mmff():
    try:
        from mlxmolkit.mmff_minimize import mmff_minimize_nk
        from mlxmolkit.mmff_params import extract_mmff_params
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
    except Exception as e:
        res["mmff_err"] = f"{type(e).__name__}: {str(e)[:70]}"
        return
    smis = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O", "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"]
    params, pos = [], []
    for s in smis:
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        p = AllChem.ETKDGv3(); p.randomSeed = 20260814
        if AllChem.EmbedMolecule(m, p) != 0:
            continue
        params.append(extract_mmff_params(m))
        pos.append(np.asarray(m.GetConformer().GetPositions(), dtype=np.float32).reshape(-1))
    flat = np.concatenate(pos)
    for tag, lb in (("bfgs", False), ("lbfgs", True)):
        try:
            fn = lambda: mmff_minimize_nk(params, [1]*len(params), flat,
                                          max_iters=200, grad_tol=1e-4, use_lbfgs=lb)
            lo, med, sp, n = timed(fn, 7, warmup=2)
            _, e, c = fn()
            res[f"mmff_{tag}_ms"] = round(lo, 1)
            res[f"mmff_{tag}_spread"] = round(sp, 1)
            res[f"mmff_{tag}_conv"] = int(np.sum(c))
            res[f"mmff_{tag}_E"] = round(float(np.sum(e)), 4)
        except Exception as ex:
            res[f"mmff_{tag}_ms"] = None
            res[f"mmff_{tag}_err"] = f"{type(ex).__name__}: {str(ex)[:70]}"


bench_nddo()
bench_mmff()
print("BENCHJSON " + json.dumps(res))
