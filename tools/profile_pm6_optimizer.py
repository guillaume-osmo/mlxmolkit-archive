"""PM6 geometry optimisation: mlxmolkit on GPU vs MOPAC, and where the time goes.

Three questions:
  1. Is the frozen-density analytical gradient actually right? Checked against
     central differences on the same geometry.
  2. How does a full PM6 optimisation compare with MOPAC, in wall clock and in
     final energy?
  3. What dominates mlxmolkit's time — the SCF, or the gradient on top of it?
"""
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

from mlxmolkit.nddo import nddo_energy, nddo_gradient, nddo_optimize

MOPAC = Path.home() / "miniconda3/bin/mopac"
WORK = Path(__file__).resolve().parent.parent / "build" / "mopac_opt"
WORK.mkdir(parents=True, exist_ok=True)
EV_PER_KCAL = 0.0433641

MOLS = [   # small on purpose: the CPU gradient makes anything larger impractical

    ("ethanol",      "CCO"),
    ("benzaldehyde", "O=Cc1ccccc1"),
    ("linalool",     "CC(C)=CCCC(C)(O)C=C"),
    ("menthol",      "CC1CCC(C(C)C)CC1O"),
    ("epoxycarvone", "C=C(C)C1CC(=O)C2(C)OC2C1"),
]


def geometry(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return atoms, np.asarray(conf.GetPositions(), dtype=float)


def run_mopac(atoms, coords, name, keywords):
    path = WORK / f"{name}.mop"
    sym = Chem.GetPeriodicTable()
    lines = [keywords, name, ""]
    for z, (x, y, zc) in zip(atoms, coords):
        lines.append(f"{sym.GetElementSymbol(int(z)):<3} "
                     f"{x:12.6f} 1 {y:12.6f} 1 {zc:12.6f} 1")
    path.write_text("\n".join(lines) + "\n")
    t0 = time.perf_counter()
    subprocess.run([str(MOPAC), path.name], cwd=WORK,
                   capture_output=True, text=True, timeout=1800)
    dt = time.perf_counter() - t0
    text = path.with_suffix(".out").read_text()
    m = re.search(r"FINAL HEAT OF FORMATION =\s+(-?\d+\.\d+)\s*KCAL", text)
    cycles = len(re.findall(r"CYCLE:", text))
    return (float(m.group(1)) if m else float("nan")), dt, cycles


print("=== 1. is the analytical (frozen-density) gradient correct? ===")
atoms, coords = geometry("CCO")
_, g_ana = nddo_gradient(atoms, coords, method="PM6", analytical=True)
_, g_num = nddo_gradient(atoms, coords, method="PM6", analytical=False)
err = np.abs(g_ana - g_num)
print(f"  ethanol, PM6: max |analytic - numeric| = {err.max():.4f} eV/A")
print(f"                RMS                      = {np.sqrt((err**2).mean()):.4f} eV/A")
print(f"                numeric gradient norm    = {np.linalg.norm(g_num):.4f} eV/A")
cos = (g_ana.ravel() @ g_num.ravel()) / (np.linalg.norm(g_ana) * np.linalg.norm(g_num))
print(f"                direction agreement      = {cos:.4f}  (1.0 = identical)")

print("\n=== 2. where does mlxmolkit spend its time? (PM6, per call) ===")
for name, smi in MOLS[:4]:
    atoms, coords = geometry(smi)
    t0 = time.perf_counter(); nddo_energy(atoms, coords, method="PM6")
    t_scf = time.perf_counter() - t0
    t0 = time.perf_counter(); nddo_gradient(atoms, coords, method="PM6", analytical=True)
    t_grad = time.perf_counter() - t0
    print(f"  {name:<14} {len(atoms):>3} atoms   SCF {t_scf*1000:8.1f} ms   "
          f"energy+gradient {t_grad*1000:8.1f} ms   "
          f"gradient overhead {t_grad/max(t_scf,1e-9):5.2f}x")

print("\n=== 3. full PM6 geometry optimisation, mlxmolkit vs MOPAC ===")
print(f"{'molecule':<14}{'atoms':>6}{'mlx s':>9}{'mopac s':>9}{'ratio':>8}"
      f"{'mlx eV':>12}{'mopac eV':>12}{'d kcal':>9}")
for name, smi in MOLS:
    atoms, coords = geometry(smi)
    t0 = time.perf_counter()
    res = nddo_optimize(atoms, coords, method="PM6", max_iter=200, grad_tol=0.01)
    t_mlx = time.perf_counter() - t0
    hof, t_mop, cycles = run_mopac(atoms, coords, name, "PM6 PRECISE GNORM=0.1")

    e_mlx = res.get("energy_eV", float("nan"))
    # MOPAC reports a heat of formation, mlxmolkit a total electronic energy;
    # only the *difference between geometries* is comparable, so compare the
    # optimised MOPAC HoF against mlxmolkit's own geometry scored by MOPAC.
    hof_of_mlx, _, _ = run_mopac(atoms, res["coords"], name + "_mlxgeom",
                                 "PM6 PRECISE 1SCF")
    print(f"{name:<14}{len(atoms):>6}{t_mlx:>9.2f}{t_mop:>9.2f}"
          f"{t_mop/max(t_mlx,1e-9):>8.2f}{e_mlx:>12.3f}{hof:>12.3f}"
          f"{hof_of_mlx - hof:>9.3f}")
print("\nd kcal = mlxmolkit's optimised geometry scored by MOPAC, minus MOPAC's own")
print("optimum. Positive means mlxmolkit stopped short of MOPAC's minimum.")
