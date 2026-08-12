"""Is the macrocycle / fused-ring gap systematic, or three unlucky molecules?

The 100-molecule perfumery set had only 3 macrocycles. This pulls a
ring-enriched sample straight from the 12k ePOM pool so the question can be
answered with numbers instead of anecdote.

Established before writing this, and worth keeping in mind when reading the
output: mlxmolkit's MMFF94 reproduces RDKit's MMFF94 to 0.00 kcal/mol on
identical coordinates, for every topology including macrocycles. So any gap
here is a *search* difference — which local minimum each optimiser lands in —
not a missing force-field term.

    python tools/bench_ring_systems_vs_rdkit.py --n-per-class 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPOM = Path(os.path.expanduser("~/epom_data/data/consensus_12k.csv"))
SUPPORTED_Z = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
MW_MIN, MW_MAX = 80.0, 400.0     # wider than the 100-set: macrocycles are big


def topology(mol) -> str:
    ri = mol.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    if any(s >= 8 for s in sizes):
        return "macro"
    if any(ri.NumAtomRings(a) > 1 for a in range(mol.GetNumAtoms())):
        return "fused"
    if sizes:
        return "ring"
    return "acyclic"


def build_pool(n_per_class: int, seed: int = 0):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    df = pd.read_csv(EPOM)
    buckets: dict[str, list[str]] = {k: [] for k in ("macro", "fused", "ring", "acyclic")}
    seen = set()
    for smi in df["smiles"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or "." in smi:
            continue
        if any(a.GetAtomicNum() not in SUPPORTED_Z for a in mol.GetAtoms()):
            continue
        if not (MW_MIN <= Descriptors.MolWt(mol) <= MW_MAX):
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        buckets[topology(mol)].append(can)

    rng = np.random.default_rng(seed)
    picked = {}
    for k, v in buckets.items():
        take = min(n_per_class, len(v))
        idx = rng.choice(len(v), size=take, replace=False) if take else []
        picked[k] = [v[i] for i in idx]
        print(f"  {k:<8} pool {len(v):5d}   sampled {take}")
    return picked


def rdkit_ensemble(smiles: str, n_confs: int, seed: int, threads: int):
    """RDKit ETKDGv3 + MMFF94, threaded — both stages take numThreads."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.useSmallRingTorsions = True
    p.numThreads = threads
    if not len(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=p)):
        return None
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=800, numThreads=threads)
    return mol


def mmff_min(mol) -> float | None:
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    es = [AllChem.MMFFGetMoleculeForceField(mol, props, confId=c.GetId()).CalcEnergy()
          for c in mol.GetConformers()]
    return min(es) if es else None


def score_positions(template, positions):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    mol = Chem.Mol(template)
    mol.RemoveAllConformers()
    for pos in positions:
        pos = np.asarray(pos, dtype=float)
        if pos.shape[0] != mol.GetNumAtoms():
            continue
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(pos):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
    if not mol.GetNumConformers():
        return None
    return mmff_min(mol)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tests" / "data" / "bench_ring_systems.csv")
    args = ap.parse_args()

    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from mlxmolkit import generate_conformers_nk

    print("sampling a ring-enriched pool from the 12k ePOM subset:")
    picked = build_pool(args.n_per_class, args.seed)
    smiles, kinds = [], []
    for k, v in picked.items():
        smiles += v
        kinds += [k] * len(v)
    print(f"\n{len(smiles)} molecules x {args.n_confs} conformers "
          f"(RDKit threads: {'all cores' if args.threads == 0 else args.threads})\n")

    t0 = time.perf_counter()
    refs = [rdkit_ensemble(s, args.n_confs, args.seed, args.threads) for s in smiles]
    t_rdkit = time.perf_counter() - t0
    print(f"RDKit      {t_rdkit:7.2f} s   {sum(r is not None for r in refs)}/{len(smiles)}")

    t0 = time.perf_counter()
    mlx = generate_conformers_nk(smiles, args.n_confs, run_mmff=True)
    t_mlx = time.perf_counter() - t0
    print(f"mlxmolkit  {t_mlx:7.2f} s   {mlx.total_conformers} conformers "
          f"({t_rdkit / max(t_mlx, 1e-9):.2f}x)\n")

    rows = []
    for i, (smi, kind, ref) in enumerate(zip(smiles, kinds, refs)):
        rec = dict(smiles=smi, topology=kind)
        got = mlx.molecules[i] if i < len(mlx.molecules) else None
        if ref is None or got is None or not got.positions_3d:
            rows.append(rec)
            continue
        e_ref = mmff_min(ref)
        e_mlx = score_positions(ref, got.positions_3d)
        if e_ref is not None and e_mlx is not None:
            rec["rdkit_mmff"] = e_ref
            rec["mlx_mmff"] = e_mlx
            rec["d_mmff"] = e_mlx - e_ref
        rows.append(rec)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df.dropna(subset=["d_mmff"])
    print(f"compared {len(ok)}/{len(df)}\n")
    print(f"{'topology':<10}{'n':>5}{'median':>9}{'mean|d|':>9}{'p90|d|':>9}"
          f"{'worst':>9}{'<=1kcal':>9}{'mlx wins':>10}")
    for kind in ("acyclic", "ring", "fused", "macro"):
        g = ok[ok.topology == kind]
        if not len(g):
            continue
        d = g.d_mmff
        print(f"{kind:<10}{len(g):>5}{d.median():>9.2f}{d.abs().mean():>9.2f}"
              f"{d.abs().quantile(.9):>9.2f}{d.abs().max():>9.2f}"
              f"{(d.abs() <= 1).mean() * 100:>8.0f}%{(d < -0.1).mean() * 100:>9.0f}%")

    hard = ok[ok.topology.isin(["macro", "fused"])].d_mmff.abs()
    easy = ok[~ok.topology.isin(["macro", "fused"])].d_mmff.abs()
    if len(hard) and len(easy):
        from scipy import stats
        u = stats.mannwhitneyu(hard, easy, alternative="greater")
        print(f"\nmacro+fused (n={len(hard)}) mean|d| {hard.mean():.3f}  vs  "
              f"rest (n={len(easy)}) {easy.mean():.3f}   p = {u.pvalue:.2e}")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
