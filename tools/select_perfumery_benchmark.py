"""Pick a 100-molecule perfumery benchmark set from the 12k ePOM subset.

The set is the basis for the external-baseline comparisons (RDKit conformers,
MOPAC energies, DFT sigma profiles) and for regression tests, so it has to be
both *representative* of the ePOM distribution and *cover the chemistry* that
breaks things — sulfur and halogens especially, since those are the elements
PM6_D treats with d orbitals.

Two-stage selection:

  1. cluster the filtered pool and take medoids, for representativeness;
  2. top up so every named chemistry class is present, for coverage.

Stage 1 is run with two different molecular representations and the better one
is chosen on a measured criterion rather than by preference — see
`--compare-representations`.

    python tools/select_perfumery_benchmark.py --compare-representations
    python tools/select_perfumery_benchmark.py --out tests/data/perfumery_benchmark_100.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPOM = Path(os.path.expanduser("~/epom_data/data/consensus_12k.csv"))
DEFAULT_OUT = ROOT / "tests" / "data" / "perfumery_benchmark_100.csv"

# Elements the NDDO methods are parameterised for. Anything else cannot be a
# benchmark case because there is no reference to compare against.
SUPPORTED_Z = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

MW_MIN, MW_MAX = 80.0, 300.0

# ePOM is an *odour* dataset, which is broader than perfumery: it contains
# solvents, agrochemicals and pharmaceuticals that happen to smell. Selecting
# medoids over the raw set returns chloroform, carbon tetrachloride and a
# sulfonamide diuretic. Filter on the descriptor columns that ship with the
# file instead of guessing from structure.
PERFUMERY_DESCRIPTORS = [
    "aldehydic", "amber", "anisic", "balsamic", "camphoreous", "citrus",
    "creamy", "earthy", "fatty", "floral", "fresh", "fruity", "green",
    "herbal", "jasmin", "leathery", "musk", "powdery", "rose", "spicy",
    "sweet", "vanilla", "waxy", "woody",
]
# A molecule described only by these is an industrial smell, not a material.
NON_PERFUMERY_DESCRIPTORS = ["odorless", "solvent", "medicinal"]

# Chemistry classes, in the order they are topped up. Each is a SMARTS plus a
# minimum count in the final set. Sulfur and the halogens are over-weighted
# relative to their ePOM frequency on purpose: they are the d-orbital elements
# under PM6_D and the ones that exposed the batch crash.
CLASSES: list[tuple[str, str, int]] = [
    ("thiol",       "[#16;H1]",                          3),
    ("thioether",   "[#16;X2]([#6])[#6]",                4),
    ("chloro",      "[Cl]",                              4),
    ("bromo_iodo",  "[Br,I]",                            2),
    ("aldehyde",    "[CX3H1](=O)[#6]",                   8),
    ("ketone",      "[#6][CX3](=O)[#6]",                 8),
    ("ester",       "[CX3](=O)[OX2H0][#6]",             10),
    ("carboxylic",  "[CX3](=O)[OX2H1]",                  5),
    ("alcohol",     "[OX2H][CX4]",                      10),
    ("phenol",      "[OX2H]c",                           3),
    ("ether",       "[OD2]([#6])[#6]",                   6),
    ("lactone",     "[CX3](=O)[OX2H0;R]",                3),
    ("enol",        "[OX2H][CX3]=[CX3]",                 2),
    ("aromatic",    "c1ccccc1",                         12),
    ("heteroarom",  "[n,o,s;a]",                         3),
    ("terpene",     "[CX4](C)(C)[CX4,CX3]",              8),
    ("nitrogen",    "[#7]",                              5),
]


def _load_pool():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd

    if not EPOM.exists():
        raise SystemExit(f"ePOM subset not found at {EPOM}")
    df = pd.read_csv(EPOM)

    have = [d for d in PERFUMERY_DESCRIPTORS if d in df.columns]
    block = [d for d in NON_PERFUMERY_DESCRIPTORS if d in df.columns]
    perfumery_score = df[have].sum(axis=1) if have else None
    blocked = df[block].sum(axis=1) > 0 if block else None

    rows = []
    for pos, smi in enumerate(df["smiles"].astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or "." in smi:                    # no salts / mixtures
            continue
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            continue
        if any(a.GetAtomicNum() not in SUPPORTED_Z for a in mol.GetAtoms()):
            continue
        mw = Descriptors.MolWt(mol)
        if not (MW_MIN <= mw <= MW_MAX):
            continue
        if perfumery_score is not None:
            score = float(perfumery_score.iloc[pos])
            if score < 1.0:                       # no perfumery descriptor at all
                continue
            # solvent/medicinal/odorless only counts against a molecule that
            # is not also solidly perfumery
            if blocked is not None and bool(blocked.iloc[pos]) and score < 2.0:
                continue
        else:
            score = 0.0
        rows.append((Chem.MolToSmiles(mol), mol, mw, score))

    # canonical SMILES dedupe
    seen, pool = set(), []
    for smi, mol, mw, score in rows:
        if smi in seen:
            continue
        seen.add(smi)
        pool.append((smi, mol, mw, score))
    return pool


def _features(mols, kind: str):
    """Binary ECFP4 bits, or the count-based ('continuous') form."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    if kind == "binary":
        arr = np.array([gen.GetFingerprintAsNumPy(m) for m in mols],
                       dtype=np.float32)
        return arr
    counts = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols],
                      dtype=np.float32)
    # log1p compresses the heavy tail of substructure counts; without it a few
    # very repetitive scaffolds dominate every distance.
    return np.log1p(counts)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return an @ bn.T


def _kmedoids(feats: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """k-medoids by farthest-point init then Lloyd swaps on cosine distance."""
    rng = np.random.default_rng(seed)
    n = len(feats)
    first = int(rng.integers(n))
    medoids = [first]
    sim_to_set = _cosine(feats, feats[[first]]).ravel()
    while len(medoids) < k:
        nxt = int(np.argmin(sim_to_set))
        medoids.append(nxt)
        sim_to_set = np.maximum(sim_to_set, _cosine(feats, feats[[nxt]]).ravel())

    for _ in range(8):
        sims = _cosine(feats, feats[medoids])
        assign = np.argmax(sims, axis=1)
        moved = False
        for j in range(k):
            members = np.flatnonzero(assign == j)
            if len(members) < 2:
                continue
            within = _cosine(feats[members], feats[members]).sum(axis=1)
            best = int(members[int(np.argmax(within))])
            if best != medoids[j]:
                medoids[j] = best
                moved = True
        if not moved:
            break
    return medoids


def _coverage(feats: np.ndarray, chosen: list[int]) -> float:
    """Mean similarity from every pool molecule to its nearest chosen one.

    This is the criterion: a representative subset is one nothing is far from.
    """
    return float(_cosine(feats, feats[chosen]).max(axis=1).mean())


def compare_representations(pool, k: int = 100) -> str:
    mols = [m for _, m, _, _ in pool]
    results = {}
    for kind in ("binary", "count_log1p"):
        feats = _features(mols, kind)
        chosen = _kmedoids(feats, k)
        # score every candidate on BOTH representations, so the comparison is
        # not just each scoring itself on its home turf
        results[kind] = {
            other: _coverage(_features(mols, other), chosen)
            for other in ("binary", "count_log1p")
        }
    print(f"\nCoverage of {len(pool)} molecules by {k} medoids")
    print("(mean nearest-neighbour cosine to the selected set; higher = more representative)\n")
    print(f"  {'selected using':<16} {'scored: binary':>16} {'scored: count+log1p':>21}")
    for kind, scores in results.items():
        print(f"  {kind:<16} {scores['binary']:>16.4f} {scores['count_log1p']:>21.4f}")
    # decide on the neutral criterion: mean of the two scorings
    means = {k_: np.mean(list(v.values())) for k_, v in results.items()}
    winner = max(means, key=means.get)
    print(f"\n  -> {winner} wins (mean {means[winner]:.4f} vs "
          f"{means[min(means, key=means.get)]:.4f})")
    return winner


def select(pool, k: int, representation: str, seed: int = 0):
    from rdkit import Chem

    mols = [m for _, m, _, _ in pool]
    scores = np.array([s for _, _, _, s in pool])
    feats = _features(mols, representation)

    patterns = {name: Chem.MolFromSmarts(sma) for name, sma, _ in CLASSES}
    tags = []
    for mol in mols:
        tags.append({n for n, p in patterns.items()
                     if p is not None and mol.HasSubstructMatch(p)})

    # stage 1 — medoids for representativeness
    chosen = list(_kmedoids(feats, k, seed=seed))

    # stage 2 — top up under-represented classes, replacing the pool members
    # that are most redundant (highest similarity to another chosen molecule)
    for name, _sma, need in CLASSES:
        have = sum(1 for i in chosen if name in tags[i])
        if have >= need:
            continue
        candidates = [i for i in range(len(pool))
                      if name in tags[i] and i not in chosen]
        if not candidates:
            continue
        # Rank by "strongly perfumery, and not already covered". Ranking by
        # dissimilarity alone picks the weirdest member of each class, which
        # is how chloroform got in.
        cand_sim = _cosine(feats[candidates], feats[chosen]).max(axis=1)
        cand_score = scores[candidates]
        rank = (cand_score / max(cand_score.max(), 1.0)) - 0.5 * cand_sim
        order = np.argsort(-rank)
        for pick in order[: need - have]:
            idx = candidates[int(pick)]
            inner = _cosine(feats[chosen], feats[chosen])
            np.fill_diagonal(inner, -1.0)
            # drop the most redundant chosen molecule that is not itself the
            # last representative of some class
            for drop_pos in np.argsort(-inner.max(axis=1)):
                cand_drop = chosen[int(drop_pos)]
                if any(sum(1 for j in chosen if c in tags[j]) <= n
                       for c, _s, n in CLASSES if c in tags[cand_drop]):
                    continue
                chosen[int(drop_pos)] = idx
                break
            else:
                chosen.append(idx)
    return chosen, tags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--compare-representations", action="store_true")
    ap.add_argument("--representation", default=None,
                    choices=["binary", "count_log1p"])
    args = ap.parse_args()

    pool = _load_pool()
    print(f"pool after filtering: {len(pool)} molecules "
          f"({MW_MIN:.0f}-{MW_MAX:.0f} Da, supported elements, no salts)")

    representation = args.representation
    if args.compare_representations or representation is None:
        representation = compare_representations(pool, args.k)

    chosen, tags = select(pool, args.k, representation)

    import pandas as pd
    from rdkit.Chem import Descriptors
    rows = []
    for i in chosen:
        smi, mol, mw, score = pool[i]
        rows.append({
            "smiles": smi,
            "mw": round(mw, 2),
            "n_atoms": mol.GetNumAtoms(),
            "n_heavy": mol.GetNumHeavyAtoms(),
            "n_rot_bonds": Descriptors.NumRotatableBonds(mol),
            "n_odour_terms": int(score),
            "classes": "|".join(sorted(tags[i])) or "other",
        })
    out = pd.DataFrame(rows).sort_values(["classes", "mw"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nwrote {len(out)} molecules -> {args.out.relative_to(ROOT)}")
    print(f"representation: {representation}\n")
    print(f"  {'class':<14} {'target':>7} {'selected':>9}")
    for name, _s, need in CLASSES:
        got = sum(1 for i in chosen if name in tags[i])
        flag = "" if got >= need else "   <-- short"
        print(f"  {name:<14} {need:>7} {got:>9}{flag}")
    print(f"\n  heavy atoms {out.n_heavy.min()}-{out.n_heavy.max()}, "
          f"MW {out.mw.min():.0f}-{out.mw.max():.0f}, "
          f"rot bonds {out.n_rot_bonds.min()}-{out.n_rot_bonds.max()}")


if __name__ == "__main__":
    main()
