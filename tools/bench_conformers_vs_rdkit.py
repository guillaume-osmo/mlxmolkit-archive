"""Baseline mlxmolkit's GPU conformer pipeline against RDKit.

The comparison is only meaningful if both ensembles are judged by the *same*
ruler, so every geometry from both sides is scored with **RDKit's** MMFF94.
Comparing mlxmolkit's own MMFF energy against RDKit's would compare two
energy functions, not two sets of geometries.

Four things are measured per molecule:

  success      did each side produce the requested conformers at all
  energy       RDKit-MMFF94 energy of the best conformer from each side
  geometry     best symmetry-aware RMSD from each mlxmolkit conformer to the
               closest RDKit conformer — do they find the same basins
  throughput   wall clock for the whole set

    python tools/bench_conformers_vs_rdkit.py --n-confs 10
    python tools/bench_conformers_vs_rdkit.py --limit 20      # quick pass
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SET = ROOT / "tests" / "data" / "perfumery_benchmark_100.csv"


def _rdkit_reference(smiles: str, n_confs: int, seed: int):
    """RDKit ETKDGv3 + MMFF94, the baseline everyone compares to."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not len(cids):
        return None
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    return mol


def _score_with_rdkit_mmff(mol) -> list[float]:
    """MMFF94 energy of every conformer on `mol`, in kcal/mol."""
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return []
    out = []
    for conf in mol.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
        out.append(float(ff.CalcEnergy()) if ff is not None else float("nan"))
    return out


def _mol_from_positions(template, positions_list):
    """Put mlxmolkit geometries onto a copy of the RDKit molecule.

    mlxmolkit builds its graph through RDKit from the same SMILES, so atom
    order matches; the length check below is what actually guarantees it.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.Mol(template)
    mol.RemoveAllConformers()
    kept = 0
    for pos in positions_list:
        pos = np.asarray(pos, dtype=float)
        if pos.shape[0] != mol.GetNumAtoms():
            continue
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(pos):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
        kept += 1
    return mol if kept else None


def _best_rmsd_matrix(probe, ref) -> float:
    """Mean over probe conformers of the best RMSD to any ref conformer."""
    from rdkit.Chem import rdMolAlign

    best = []
    for i in range(probe.GetNumConformers()):
        d = []
        for j in range(ref.GetNumConformers()):
            try:
                d.append(rdMolAlign.GetBestRMS(probe, ref, prbId=i, refId=j))
            except Exception:
                continue
        if d:
            best.append(min(d))
    return float(np.mean(best)) if best else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tests" / "data" / "bench_conformers_vs_rdkit.csv")
    args = ap.parse_args()

    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from mlxmolkit import generate_conformers_nk

    bench = pd.read_csv(args.set)
    if args.limit:
        bench = bench.head(args.limit)
    smiles = bench["smiles"].tolist()
    print(f"{len(smiles)} molecules x {args.n_confs} conformers\n")

    # --- RDKit baseline -------------------------------------------------
    t0 = time.perf_counter()
    ref_mols = [_rdkit_reference(s, args.n_confs, args.seed) for s in smiles]
    t_rdkit = time.perf_counter() - t0
    n_ref_ok = sum(m is not None for m in ref_mols)
    print(f"RDKit      {t_rdkit:7.2f} s   {n_ref_ok}/{len(smiles)} embedded")

    # --- mlxmolkit, batched (all N molecules in one dispatch) ------------
    t0 = time.perf_counter()
    mlx = generate_conformers_nk(smiles, args.n_confs, run_mmff=True)
    t_mlx = time.perf_counter() - t0
    print(f"mlxmolkit  {t_mlx:7.2f} s   {mlx.total_conformers} conformers "
          f"({t_rdkit / max(t_mlx, 1e-9):.2f}x vs RDKit)")

    # Anything the batch dropped is retried one molecule at a time on the same
    # GPU path. If a molecule only works singly, the batch has a bug — that is
    # the point of the retry, so the fallbacks are reported, never silent.
    batch_missed = [i for i, s_ in enumerate(smiles)
                    if i >= len(mlx.molecules)
                    or not mlx.molecules[i].positions_3d]
    single_rescued = []
    if batch_missed:
        print(f"           batch produced nothing for {len(batch_missed)} "
              f"molecule(s); retrying each singly on the GPU")
        for i in batch_missed:
            try:
                one = generate_conformers_nk([smiles[i]], args.n_confs,
                                             run_mmff=True)
            except Exception as exc:
                print(f"             [{i}] {smiles[i][:40]}: single also failed "
                      f"({type(exc).__name__})")
                continue
            if one.molecules and one.molecules[0].positions_3d:
                while len(mlx.molecules) <= i:
                    mlx.molecules.append(one.molecules[0])
                mlx.molecules[i] = one.molecules[0]
                single_rescued.append(i)
        if single_rescued:
            print(f"           {len(single_rescued)} rescued by the single path "
                  f"-> BATCH BUG on: "
                  + ", ".join(smiles[i][:28] for i in single_rescued[:5]))
        else:
            print("           none rescued — those molecules fail on both paths")
    else:
        print("           batch produced conformers for every molecule")
    print()

    rows = []
    for idx, (smi, ref) in enumerate(zip(smiles, ref_mols)):
        rec = dict(smiles=smi, classes=bench["classes"].iloc[idx],
                   n_heavy=int(bench["n_heavy"].iloc[idx]))
        got = mlx.molecules[idx] if idx < len(mlx.molecules) else None
        rec["rdkit_confs"] = ref.GetNumConformers() if ref is not None else 0
        rec["mlx_confs"] = len(got.positions_3d) if got is not None else 0

        if ref is None or got is None or not got.positions_3d:
            rows.append(rec)
            continue

        probe = _mol_from_positions(ref, got.positions_3d)
        if probe is None:
            rec["atom_order_mismatch"] = True
            rows.append(rec)
            continue

        e_ref = _score_with_rdkit_mmff(ref)
        e_mlx = _score_with_rdkit_mmff(probe)
        if e_ref:
            rec["rdkit_best_mmff"] = min(e_ref)
        if e_mlx:
            rec["mlx_best_mmff"] = min(e_mlx)
        if e_ref and e_mlx:
            rec["d_best_mmff"] = min(e_mlx) - min(e_ref)
        rec["mean_best_rmsd"] = _best_rmsd_matrix(probe, ref)
        rows.append(rec)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    if batch_missed:
        df["batch_failed"] = df.index.isin(batch_missed)
        df["single_rescued"] = df.index.isin(single_rescued)

    ok = df.dropna(subset=["d_best_mmff"])
    print(f"compared {len(ok)}/{len(df)} molecules "
          f"(both sides produced scorable conformers)\n")
    if len(ok):
        d = ok["d_best_mmff"]
        print("Best-conformer MMFF94 energy, mlxmolkit - RDKit (kcal/mol)")
        print(f"  median {d.median():+7.2f}   mean {d.mean():+7.2f}   "
              f"p10 {d.quantile(.1):+7.2f}   p90 {d.quantile(.9):+7.2f}")
        print(f"  mlxmolkit finds a LOWER minimum for "
              f"{(d < -0.1).sum()}/{len(d)} molecules; "
              f"within 1 kcal/mol for {(d.abs() <= 1).sum()}/{len(d)}")
        r = ok["mean_best_rmsd"].dropna()
        if len(r):
            print(f"\nGeometry agreement (mean best RMSD to nearest RDKit conformer, A)")
            print(f"  median {r.median():.3f}   p90 {r.quantile(.9):.3f}   "
                  f"max {r.max():.3f}")
        worst = ok.reindex(ok["d_best_mmff"].abs().sort_values(ascending=False).index)
        print("\nLargest energy disagreements:")
        for _, row in worst.head(5).iterrows():
            print(f"  {row.d_best_mmff:+8.2f} kcal/mol  {row.smiles[:44]:44s} "
                  f"{str(row.classes)[:28]}")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
