"""#12 proposes routing the 6N displacements through prepare_batch.
#11 made the incremental path O(N) per displacement. Which actually wins?"""
import time, numpy as np
from mlxmolkit.nddo.pipeline import _smiles_to_3d
from mlxmolkit.nddo.anal_grad import analytical_gradient
from mlxmolkit.nddo.batch import prepare_batch
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo import anal_grad as AG

P6 = get_params("PM6")
for name, smi in [("ethanol","CCO"), ("benzaldehyde","O=Cc1ccccc1"),
                  ("thioanisole","CSc1ccccc1"), ("menthol","CC(C)C1CCC(C)CC1O")]:
    atoms, coords = _smiles_to_3d(smi, seed=42)[:2]
    n = len(atoms)
    analytical_gradient(atoms, coords, method="PM6")
    ts = []
    for _ in range(3):
        t = time.perf_counter(); analytical_gradient(atoms, coords, method="PM6")
        ts.append(time.perf_counter()-t)
    t_inc = min(ts)

    # what #12 proposes: the 6N displaced geometries as one uniform batch
    disp = []
    for a in range(n):
        for d in range(3):
            for s in (1.0, -1.0):
                c = coords.copy(); c[a, d] += s * 1e-5
                disp.append((atoms, c))
    prepare_batch(disp[:4], P6, method="PM6")
    ts = []
    for _ in range(3):
        t = time.perf_counter(); prepare_batch(disp, P6, method="PM6")
        ts.append(time.perf_counter()-t)
    t_batch = min(ts)
    verdict = "incremental wins" if t_inc < t_batch else "BATCH wins"
    print(f"  {name:14s} {n:3d} atoms  incremental {t_inc*1e3:8.1f} ms   "
          f"prepare_batch({len(disp)} geoms) {t_batch*1e3:8.1f} ms   -> {verdict}")
