"""Freeze the benchmark geometries once so every commit measures identical input."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

MOLS = {
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "thioanisole": "CSc1ccccc1",          # d path
    "aspirin":     "CC(=O)OC1=CC=CC=C1C(=O)O",
    "testosterone":"CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
}
out = {}
for name, smi in MOLS.items():
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3(); p.randomSeed = 20260814
    assert AllChem.EmbedMolecule(m, p) == 0, name
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    out[f"{name}_Z"] = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=np.int32)
    out[f"{name}_R"] = np.asarray(m.GetConformer().GetPositions(), dtype=np.float64)
    print(f"  {name:<13} {m.GetNumAtoms():3d} atoms ({sum(1 for a in m.GetAtoms() if a.GetAtomicNum()>1)} heavy)")
np.savez("/Users/tgg/Github/_mlxmolkit_safety/bench/geom.npz", **out)
print("wrote geom.npz")
