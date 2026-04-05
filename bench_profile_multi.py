"""Profile time breakdown: param extraction, kernel, Python L-BFGS overhead."""
import time

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from mlxmolkit.mmff_metal_kernel import (
    mmff_energy_grad_metal_mega,
    pack_multi_mol_for_metal,
)
from mlxmolkit.mmff_metal_optimizer import (
    mmff_optimize_metal_batch,
    mmff_optimize_metal_multi_mol,
)
from mlxmolkit.mmff_params import extract_mmff_params

SMILES_10 = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "OC(=O)CC1=CC=CC=C1NC1=C(Cl)C=CC=C1Cl",
    "CC(=O)NC1=CC=C(O)C=C1",
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
    "OC(=O)C1=CC=CC=C1O",
    "COC1=CC(=CC(=C1O)OC)C=O",
    "CC(=O)OCC=C",
    "OC1=CC=CC2=CC=CC=C12",
]


def prepare_mols(smiles_list, n_conf):
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        m = Chem.AddHs(m)
        cids = AllChem.EmbedMultipleConfs(m, numConfs=n_conf, params=params)
        if len(cids) == 0:
            continue
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m)
        if props is None:
            continue
        mols.append(m)
    return mols


def profile_breakdown(mols, n_conf):
    total_confs = sum(m.GetNumConformers() for m in mols)
    print(f"\n{'='*70}")
    print(f"  {len(mols)} mols x {n_conf} confs = {total_confs} total conformers")
    print(f"{'='*70}")

    # 1. Param extraction
    t0 = time.perf_counter()
    params_list = []
    conf_ids_list = []
    conf_counts = []
    for mol in mols:
        cids = [c.GetId() for c in mol.GetConformers()]
        conf_ids_list.append(cids)
        params_list.append(extract_mmff_params(mol, conf_id=cids[0]))
        conf_counts.append(len(cids))
    t_extract = time.perf_counter() - t0
    print(f"  Param extraction:  {t_extract*1000:8.1f} ms")

    # 2. Packing for mega-kernel
    t0 = time.perf_counter()
    dims_per_mol = [p.n_atoms * 3 for p in params_list]
    max_dim = max(dims_per_mol)
    (idx_buf, param_buf, all_meta, dispatch, _, _) = pack_multi_mol_for_metal(
        params_list, conf_counts,
    )
    pos_offsets = mx.array(
        np.arange(total_confs, dtype=np.uint32) * max_dim
    )
    total_padded = total_confs * max_dim
    t_pack = time.perf_counter() - t0
    print(f"  Parameter packing: {t_pack*1000:8.1f} ms")

    # 3. Position filling
    t0 = time.perf_counter()
    pos_np = np.zeros((total_confs, max_dim), dtype=np.float32)
    ci = 0
    for m_idx, mol in enumerate(mols):
        n_atoms = mol.GetNumAtoms()
        for cid in conf_ids_list[m_idx]:
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                p = conf.GetAtomPosition(a)
                pos_np[ci, a * 3] = p.x
                pos_np[ci, a * 3 + 1] = p.y
                pos_np[ci, a * 3 + 2] = p.z
            ci += 1
    x = mx.array(pos_np.ravel())
    mx.eval(x)
    t_pos = time.perf_counter() - t0
    print(f"  Position init:     {t_pos*1000:8.1f} ms")

    # 4. Pure kernel time (single call, no optimizer)
    mx.eval(x)
    t0 = time.perf_counter()
    for _ in range(200):
        e, g = mmff_energy_grad_metal_mega(
            idx_buf, param_buf, all_meta, dispatch, pos_offsets,
            x, total_confs, total_padded,
        )
    mx.eval(e, g)
    t_kernel_200 = time.perf_counter() - t0
    print(f"  200x kernel calls: {t_kernel_200*1000:8.1f} ms  "
          f"({t_kernel_200/200*1000:.2f} ms/call)")

    # 5. Full optimizer
    copies = [Chem.RWMol(m) for m in mols]
    specs = [(m, None) for m in copies]
    # warm up
    mmff_optimize_metal_multi_mol(specs, max_iters=10)
    copies = [Chem.RWMol(m) for m in mols]
    specs = [(m, None) for m in copies]
    t0 = time.perf_counter()
    mmff_optimize_metal_multi_mol(specs, max_iters=200)
    t_opt = time.perf_counter() - t0
    print(f"  Full optimizer:    {t_opt*1000:8.1f} ms")
    overhead = t_opt - t_kernel_200
    print(f"  Python overhead:   {overhead*1000:8.1f} ms  "
          f"({overhead/t_opt*100:.0f}% of total)")

    # 6. RDKit comparison
    copies = [Chem.RWMol(m) for m in mols]
    t0 = time.perf_counter()
    for m in copies:
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(m, maxIters=200, numThreads=1)
    t_rdk1 = time.perf_counter() - t0

    copies = [Chem.RWMol(m) for m in mols]
    n_cpu = 14
    t0 = time.perf_counter()
    for m in copies:
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(m, maxIters=200, numThreads=n_cpu)
    t_rdkN = time.perf_counter() - t0

    print(f"\n  RDKit 1T:          {t_rdk1*1000:8.1f} ms")
    print(f"  RDKit 14T:         {t_rdkN*1000:8.1f} ms")
    print(f"  Metal optimizer:   {t_opt*1000:8.1f} ms")
    print(f"  Metal kernels only:{t_kernel_200*1000:8.1f} ms")
    print(f"\n  Speedup vs 1T:     {t_rdk1/t_opt:.2f}x (optimizer)  "
          f"{t_rdk1/t_kernel_200:.1f}x (kernel-only)")
    print(f"  Speedup vs 14T:    {t_rdkN/t_opt:.2f}x (optimizer)  "
          f"{t_rdkN/t_kernel_200:.1f}x (kernel-only)")


def main():
    for n_conf in [10, 50, 100, 500]:
        mols = prepare_mols(SMILES_10, n_conf)
        profile_breakdown(mols, n_conf)


if __name__ == "__main__":
    main()
