"""Which ETKDG setting should you actually use — in vacuum, and in liquid?

RDKit names eight variants, but the names do not span the flag space. The
interesting omission: ``srETKDGv3`` turns the macrocycle terms *off*, so no
named variant combines small-ring torsions with macrocycle torsions and the
macrocycle 1-4 bounds. This sweeps the named eight *and* the unnamed
combinations, so the gap gets measured instead of assumed.

Two rulers, both RDKit's own MMFF94, differing only in how electrostatics are
screened:

  vacuum   constant dielectric, eps = 1        the usual gas-phase convention
  liquid   distance-dependent, eps = 80        fully screened

The liquid ruler is screened electrostatics, **not** a solvation free energy —
there is no cavitation or dispersion term, so it is not a dG_solv. What it does
capture is the dominant conformational effect of a polar solvent: it removes the
intramolecular electrostatic collapse (the folded, self-hydrogen-bonded
conformers) that a gas-phase force field over-rewards. That is the effect that
decides which conformer a variant should be finding.

Geometries are relaxed under the same ruler they are scored with, so a variant
is never judged on a minimum belonging to a different energy function.

Scoring is by **regret**: for each molecule the best energy any competitor found
is the reference, and a variant's regret is how far above it that variant landed.
Zero regret means the variant found the best conformer known for that molecule.
This is comparable across molecules in a way that raw energies are not.

    python tools/bench_etkdg_variants.py --limit 40      # quick pass
    python tools/bench_etkdg_variants.py                 # the full set
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SET = ROOT / "tests" / "data" / "bench_ring_systems.csv"

# The eight RDKit names, in increasing order of knowledge applied.
NAMED = ["DG", "KDG", "ETDG", "ETDGv2", "ETKDG", "ETKDGv2", "ETKDGv3", "srETKDGv3"]

# Combinations RDKit does not name. Flags are
# (exp_torsion, basic_knowledge, small_ring, macrocycle, macrocycle14, et_version).
# All keep experimental torsions and basic knowledge on — turning those off is
# already covered by DG/KDG/ETDG above.
UNNAMED = {
    "sr+macro":     (True, True, True,  True,  True,  2),   # the gap in the table
    "sr+macro14":   (True, True, True,  False, True,  2),   # bounds without torsions
    "macro-tors":   (True, True, False, True,  False, 2),   # torsions without bounds
    "macro14-only": (True, True, False, False, True,  2),   # bounds alone
}

RULERS = {
    # label:      (dielectric model, dielectric constant)
    "vacuum": (1, 1.0),
    "liquid": (2, 80.0),
}


def register_unnamed() -> None:
    """Teach the extractor the combinations RDKit has no factory for.

    ``extract_etk_params`` looks a variant up in this table and only calls
    ``getattr(rdDistGeom, name)`` when such a factory exists; for these names it
    does not, so the explicit-flag path runs with exactly the tuple below.
    """
    from mlxmolkit.etk_extract import ETKDG_VARIANTS
    ETKDG_VARIANTS.update(UNNAMED)


def props_for(mol, ruler: str):
    from rdkit.Chem import AllChem

    model, eps = RULERS[ruler]
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    props.SetMMFFDielectricModel(model)
    props.SetMMFFDielectricConstant(eps)
    return props


def relax_and_score(template, positions, ruler: str, max_iters: int) -> float | None:
    """Relax every geometry under `ruler` and return the best energy found."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    mol = Chem.Mol(template)
    props = props_for(mol, ruler)
    if props is None:
        return None
    conf = mol.GetConformer()
    best = None
    for pos in positions:
        pos = np.asarray(pos, dtype=float)
        if pos.shape[0] != mol.GetNumAtoms():
            continue
        for i, (x, y, z) in enumerate(pos):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            continue
        ff.Minimize(maxIts=max_iters)
        e = ff.CalcEnergy()
        best = e if best is None else min(best, e)
    return best


def rdkit_positions(smiles: str, n_confs: int, variant: str, seed: int, threads: int):
    """RDKit's own generator, as the external competitor."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    factory = getattr(rdDistGeom, variant, None)
    params = factory() if factory is not None else AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = threads
    if not len(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)):
        return None
    return [np.asarray(c.GetPositions(), dtype=float) for c in mol.GetConformers()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--n-confs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="molecules per topology")
    ap.add_argument("--relax-iters", type=int, default=400)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tests" / "data" / "bench_etkdg_variants.csv")
    args = ap.parse_args()

    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    register_unnamed()
    from mlxmolkit import generate_conformers_nk

    bench = pd.read_csv(args.set).dropna(subset=["topology"])
    if args.limit:
        bench = bench.groupby("topology", group_keys=False).head(args.limit)
    smiles = bench["smiles"].tolist()
    topo = dict(zip(bench["smiles"], bench["topology"]))
    # An empty conformer, not an embedded one: every position gets overwritten
    # below, and a molecule RDKit cannot embed would otherwise drop out of the
    # comparison entirely.
    templates = {}
    for s in smiles:
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        m.AddConformer(Chem.Conformer(m.GetNumAtoms()), assignId=True)
        templates[s] = m

    competitors = [(f"mlx:{v}", v) for v in NAMED + list(UNNAMED)]
    competitors += [("rdkit:ETKDGv3", "ETKDGv3"), ("rdkit:ETKDGv2", "ETKDGv2")]
    print(f"{len(smiles)} molecules x {args.n_confs} conformers "
          f"x {len(competitors)} settings x {len(RULERS)} rulers\n")

    # --- generate every ensemble once -----------------------------------
    ensembles: dict[str, dict[str, list]] = {}
    for label, variant in competitors:
        t0 = time.perf_counter()
        if label.startswith("rdkit:"):
            got = {s: rdkit_positions(s, args.n_confs, variant, args.seed, args.threads)
                   for s in smiles}
        else:
            res = generate_conformers_nk(smiles, args.n_confs, run_mmff=True,
                                         variant=variant)
            got = {s: (m.positions_3d or None)
                   for s, m in zip(smiles, res.molecules)}
        ensembles[label] = got
        n_ok = sum(v is not None for v in got.values())
        print(f"  {label:<22} {time.perf_counter() - t0:6.1f}s   {n_ok}/{len(smiles)}")

    # --- relax and score under each ruler -------------------------------
    print("\nrelaxing and scoring under each ruler...")
    rows = []
    for ruler in RULERS:
        t0 = time.perf_counter()
        for label, _ in competitors:
            for s in smiles:
                pos = ensembles[label].get(s)
                if not pos:
                    continue
                e = relax_and_score(templates[s], pos, ruler, args.relax_iters)
                if e is not None:
                    rows.append(dict(smiles=s, topology=topo[s], setting=label,
                                     ruler=ruler, energy=e))
        print(f"  {ruler:<8} {time.perf_counter() - t0:6.1f}s")

    df = pd.DataFrame(rows)
    # regret = how far above the best any competitor found for that molecule
    df["regret"] = df.groupby(["smiles", "ruler"])["energy"].transform(
        lambda g: g - g.min())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    order = [c for c, _ in competitors]
    for ruler in RULERS:
        sub = df[df.ruler == ruler]
        print(f"\n=== {ruler.upper()}  "
              f"(mean regret in kcal/mol; 0.00 = found the best known conformer) ===")
        piv = sub.pivot_table(index="setting", columns="topology",
                              values="regret", aggfunc="mean")
        cols = [c for c in ("acyclic", "ring", "fused", "macro") if c in piv.columns]
        overall = sub.groupby("setting")["regret"].mean()
        wins = sub.assign(w=sub.regret < 1e-6).groupby("setting")["w"].mean() * 100
        print(f"{'setting':<22}" + "".join(f"{c:>9}" for c in cols)
              + f"{'ALL':>9}{'best%':>8}")
        for label in sorted(order, key=lambda x: overall.get(x, 9e9)):
            if label not in piv.index:
                continue
            print(f"{label:<22}"
                  + "".join(f"{piv.loc[label, c]:>9.2f}" for c in cols)
                  + f"{overall[label]:>9.2f}{wins[label]:>7.0f}%")

    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
