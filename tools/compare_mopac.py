"""Compare mlxmolkit.rm1 PM6 against OpenMOPAC (ground truth) on fixed geometries.

For each molecule: one fixed RDKit geometry (ETKDGv3 seed=1 + MMFF) ->
  * MOPAC `PM6 1SCF`            -> FINAL HEAT OF FORMATION + NET ATOMIC CHARGES
  * mlxmolkit nddo_energy PM6_D -> heat_of_formation_kcal + charges
Prints a parity table and emits the reference dict to paste into tests/test_mopac_parity.py.

Run:  PYTHONPATH=. python tools/compare_mopac.py
"""
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from mlxmolkit.rm1 import nddo_energy

RDLogger.DisableLog("rdApp.*")
MOPAC = "/Users/guillaume-osmo/miniconda3/envs/osmo/bin/mopac"

MOLS = [
    ("ethanol", "CCO"),
    ("methanol", "CO"),
    ("acetone", "CC(C)=O"),
    ("dimethyl_sulfide", "CSC"),
    ("methanethiol", "CS"),
    ("dimethyl_disulfide", "CSSC"),
    ("thiophene", "c1ccsc1"),
    ("chlorobenzene", "c1ccccc1Cl"),
    ("trimethyl_phosphate", "COP(=O)(OC)OC"),
]


def fixed_geometry(smi: str):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m)
    conf = m.GetConformer()
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    return Z, xyz, sym


def run_mopac_1scf(name: str, sym, xyz, workdir: Path):
    mop = workdir / f"{name}.mop"
    lines = ["PM6 1SCF", name, ""]
    for s, (x, y, z) in zip(sym, xyz):
        lines.append(f"{s:2s} {x:.6f} 1 {y:.6f} 1 {z:.6f} 1")
    mop.write_text("\n".join(lines) + "\n")
    subprocess.run([MOPAC, mop.name], cwd=workdir, capture_output=True, timeout=120)
    out = (workdir / f"{name}.out").read_text()

    m = re.search(r"FINAL HEAT OF FORMATION\s*=\s*(-?\d+\.\d+)\s*KCAL/MOL", out)
    hf = float(m.group(1))
    qm = re.search(r"NET ATOMIC CHARGES.*?\n(.*?)\n\s*DIPOLE", out, re.S)
    charges = [float(ln.split()[2]) for ln in qm.group(1).splitlines()
               if len(ln.split()) >= 3 and ln.split()[0].isdigit()]
    return hf, np.array(charges)


def main():
    print(f"{'molecule':20s} {'Hf MOPAC':>10s} {'Hf PM6_D':>10s} {'ΔHf':>7s}   "
          f"{'max|Δq|':>8s} {'q(hetero) MOPAC vs ours'}")
    ref = {}
    with tempfile.TemporaryDirectory() as td:
        for name, smi in MOLS:
            Z, xyz, sym = fixed_geometry(smi)
            hf_m, q_m = run_mopac_1scf(name, sym, xyz, Path(td))
            r = nddo_energy(Z, xyz, method="PM6_D")
            hf_o, q_o = r["heat_of_formation_kcal"], r["charges"]
            dq = np.abs(q_m - q_o).max()
            het = [(s, i) for i, s in enumerate(sym) if s not in ("C", "H")]
            hetstr = "  ".join(f"{s}{i}:{q_m[i]:+.3f}/{q_o[i]:+.3f}" for s, i in het[:3])
            print(f"{name:20s} {hf_m:+10.2f} {hf_o:+10.2f} {hf_o-hf_m:+7.2f}   {dq:8.4f}   {hetstr}")
            ref[name] = dict(smiles=smi, hf_mopac=hf_m, charges_mopac=q_m.round(4).tolist())
    print("\n# --- paste into tests/test_mopac_parity.py ---")
    print("MOPAC_REF =", repr(ref))


if __name__ == "__main__":
    main()
