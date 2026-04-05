"""
Head-to-head benchmark: Guillaume's mlxmolkit_phase1 vs Shivam's mlxmolkit.

Guillaume's code:  RDKit ETKDG embedding (CPU) → Metal L-BFGS MMFF94 optimization (GPU)
Shivam's code:     Full ETKDG embedding pipeline on Metal/MLX (GPU), no MMFF optimization

This benchmark tests 5 pipelines on the SAME molecules:
  A) RDKit baseline:       RDKit EmbedMultipleConfs + RDKit MMFFOptimize
  B) Guillaume (yours):    RDKit EmbedMultipleConfs + Metal MMFF optimize
  C) Shivam:               Metal ETKDG EmbedMolecules (no MMFF)
  D) Shivam + RDKit MMFF:  Metal ETKDG EmbedMolecules + RDKit MMFFOptimize
  E) Shivam + Guillaume:   Metal ETKDG EmbedMolecules + Metal MMFF optimize

Metrics: wall-clock time, conformers/sec, success rate, MMFF energy quality.
"""

import sys
import os
import time
import warnings
import importlib

import numpy as np

# ── Import RDKit ──
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ── Path setup ──
PHASE1_DIR = os.path.dirname(os.path.abspath(__file__))
SHIVAM_DIR = os.path.join(os.path.dirname(PHASE1_DIR), "mlxmolkit_shivam")

# ── Import Guillaume's code ──
sys.path.insert(0, PHASE1_DIR)
from mlxmolkit.mmff_optimizer import mmff_optimize
from mlxmolkit.mmff_batch_optimizer import mmff_optimize_batch


def _load_shivam_embed():
    saved = {}
    for k in list(sys.modules):
        if k == "mlxmolkit" or k.startswith("mlxmolkit."):
            saved[k] = sys.modules.pop(k)
    sys.path.insert(0, SHIVAM_DIR)
    try:
        mod = importlib.import_module("mlxmolkit.embed_molecules")
        fn = mod.EmbedMolecules
    finally:
        sys.path.remove(SHIVAM_DIR)
        for k in list(sys.modules):
            if k == "mlxmolkit" or k.startswith("mlxmolkit."):
                del sys.modules[k]
        sys.modules.update(saved)
    return fn


ShivamEmbedMolecules = _load_shivam_embed()


# ═══════════════════════════════════════════════════════════════════════
# Molecules
# ═══════════════════════════════════════════════════════════════════════
SMILES_SET = [
    ("ethanol",       "CCO"),
    ("benzene",       "c1ccccc1"),
    ("acetic_acid",   "CC(=O)O"),
    ("cyclohexane",   "C1CCCCC1"),
    ("phenol",        "C1=CC=CC=C1O"),
    ("aspirin",       "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("ibuprofen",     "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("caffeine",      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("acetaminophen", "CC(=O)NC1=CC=C(O)C=C1"),
    ("diclofenac",    "OC(=O)CC1=CC=CC=C1NC1=C(Cl)C=CC=C1Cl"),
]


def prepare_mols(smiles_list):
    mols = []
    for name, smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.AddHs(mol)
            mols.append((name, mol))
    return mols


def deep_copy_mols(mol_list):
    return [(n, Chem.RWMol(m)) for n, m in mol_list]


def get_mmff_energies(mol):
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is None:
        return [float("nan")] * mol.GetNumConformers()
    energies = []
    for conf in mol.GetConformers():
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
        if ff is not None:
            energies.append(ff.CalcEnergy())
        else:
            energies.append(float("nan"))
    return energies


def fmt(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


# ═══════════════════════════════════════════════════════════════════════
# Pipeline A: RDKit baseline (CPU embed + CPU MMFF)
# ═══════════════════════════════════════════════════════════════════════
def pipeline_a(mol_list, n_confs, params):
    copies = deep_copy_mols(mol_list)

    t0 = time.perf_counter()
    for _, mol in copies:
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _, mol in copies:
        if mol.GetNumConformers() > 0:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    t_opt = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    n_emb = sum(m.GetNumConformers() for _, m in copies)
    energies = []
    for _, mol in copies:
        energies.extend(get_mmff_energies(mol))

    return dict(name="A) RDKit (CPU embed + CPU MMFF)", t_embed=t_embed, t_opt=t_opt,
                t_total=t_total, n_embedded=n_emb, energies=energies)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline B: Guillaume (CPU embed + Metal MMFF)
# ═══════════════════════════════════════════════════════════════════════
def pipeline_b(mol_list, n_confs, params):
    copies = deep_copy_mols(mol_list)

    t0 = time.perf_counter()
    for _, mol in copies:
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _, mol in copies:
        if mol.GetNumConformers() == 0:
            continue
        try:
            mmff_optimize_batch(mol, method="lbfgs", max_iters=200, scale_grads=True)
        except Exception:
            for conf in mol.GetConformers():
                try:
                    mmff_optimize(mol, conf_id=conf.GetId(), method="lbfgs",
                                  max_iters=200, scale_grads=True)
                except Exception:
                    pass
    t_opt = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    n_emb = sum(m.GetNumConformers() for _, m in copies)
    energies = []
    for _, mol in copies:
        energies.extend(get_mmff_energies(mol))

    return dict(name="B) Guillaume (CPU embed + Metal MMFF)", t_embed=t_embed, t_opt=t_opt,
                t_total=t_total, n_embedded=n_emb, energies=energies)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline C: Shivam (Metal ETKDG, no MMFF)
# ═══════════════════════════════════════════════════════════════════════
def pipeline_c(mol_list, n_confs, params):
    copies = deep_copy_mols(mol_list)
    just_mols = [m for _, m in copies]

    t0 = time.perf_counter()
    try:
        ShivamEmbedMolecules(just_mols, params, confsPerMolecule=n_confs)
    except Exception as e:
        print(f"    [Shivam error: {e}]")
    t_embed = time.perf_counter() - t0

    n_emb = sum(m.GetNumConformers() for m in just_mols)
    energies = []
    for m in just_mols:
        energies.extend(get_mmff_energies(m))

    return dict(name="C) Shivam (Metal ETKDG, no MMFF)", t_embed=t_embed, t_opt=0.0,
                t_total=t_embed, n_embedded=n_emb, energies=energies)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline D: Shivam embed + RDKit MMFF
# ═══════════════════════════════════════════════════════════════════════
def pipeline_d(mol_list, n_confs, params):
    copies = deep_copy_mols(mol_list)
    just_mols = [m for _, m in copies]

    t0 = time.perf_counter()
    try:
        ShivamEmbedMolecules(just_mols, params, confsPerMolecule=n_confs)
    except Exception as e:
        print(f"    [Shivam error: {e}]")
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for m in just_mols:
        if m.GetNumConformers() > 0:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    t_opt = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    n_emb = sum(m.GetNumConformers() for m in just_mols)
    energies = []
    for m in just_mols:
        energies.extend(get_mmff_energies(m))

    return dict(name="D) Shivam embed + RDKit MMFF", t_embed=t_embed, t_opt=t_opt,
                t_total=t_total, n_embedded=n_emb, energies=energies)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline E: Shivam embed + Guillaume Metal MMFF
# ═══════════════════════════════════════════════════════════════════════
def pipeline_e(mol_list, n_confs, params):
    copies = deep_copy_mols(mol_list)
    just_mols = [m for _, m in copies]

    t0 = time.perf_counter()
    try:
        ShivamEmbedMolecules(just_mols, params, confsPerMolecule=n_confs)
    except Exception as e:
        print(f"    [Shivam error: {e}]")
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for m in just_mols:
        if m.GetNumConformers() == 0:
            continue
        try:
            mmff_optimize_batch(m, method="lbfgs", max_iters=200, scale_grads=True)
        except Exception:
            for conf in m.GetConformers():
                try:
                    mmff_optimize(m, conf_id=conf.GetId(), method="lbfgs",
                                  max_iters=200, scale_grads=True)
                except Exception:
                    pass
    t_opt = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    n_emb = sum(m.GetNumConformers() for m in just_mols)
    energies = []
    for m in just_mols:
        energies.extend(get_mmff_energies(m))

    return dict(name="E) Shivam embed + Guillaume Metal MMFF", t_embed=t_embed, t_opt=t_opt,
                t_total=t_total, n_embedded=n_emb, energies=energies)


def warmup(params):
    """JIT warmup for all Metal codepaths."""
    print("  Warming up Metal kernels...")
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, numConfs=2, params=params)
    try:
        mmff_optimize_batch(mol, method="lbfgs", max_iters=10, scale_grads=True)
    except Exception:
        pass

    mol2 = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    try:
        ShivamEmbedMolecules([mol2], params, confsPerMolecule=1)
    except Exception:
        pass
    print("  Warmup done.\n")


def main():
    print()
    print("=" * 80)
    print("  HEAD-TO-HEAD BENCHMARK")
    print("  Guillaume mlxmolkit (Metal MMFF) vs Shivam mlxmolkit (Metal ETKDG)")
    print("=" * 80)
    print()
    print("  Guillaume: RDKit ETKDG embed (CPU) + Metal L-BFGS MMFF94 (GPU)")
    print("  Shivam:    Full ETKDG pipeline on Metal/MLX (GPU), no MMFF")
    print("  RDKit:     CPU ETKDG embed + CPU MMFF optimize (baseline)")
    print()

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True

    warmup(params)

    mol_list = prepare_mols(SMILES_SET)
    n_mols = len(mol_list)
    total_atoms = sum(m.GetNumAtoms() for _, m in mol_list)

    # ── Two test configs ──
    for n_confs in [1, 5]:
        target = n_mols * n_confs

        print("=" * 80)
        print(f"  {n_mols} molecules × {n_confs} conformers = {target} total")
        print(f"  Total atoms: {total_atoms} (avg {total_atoms / n_mols:.0f}/mol)")
        print("  Molecules:", ", ".join(n for n, _ in mol_list))
        print("=" * 80)

        results = {}

        # A) RDKit baseline — fast, run first
        print("\n  Running A) RDKit baseline...")
        r = pipeline_a(mol_list, n_confs, params)
        results["A"] = r
        print(f"    Embed: {fmt(r['t_embed'])}  Opt: {fmt(r['t_opt'])}  "
              f"Total: {fmt(r['t_total'])}  Confs: {r['n_embedded']}/{target}")

        # B) Guillaume — fast
        print("\n  Running B) Guillaume (CPU embed + Metal MMFF)...")
        r = pipeline_b(mol_list, n_confs, params)
        results["B"] = r
        print(f"    Embed: {fmt(r['t_embed'])}  Opt: {fmt(r['t_opt'])}  "
              f"Total: {fmt(r['t_total'])}  Confs: {r['n_embedded']}/{target}")

        # C) Shivam — slow, only 1 repeat
        print("\n  Running C) Shivam (Metal ETKDG, no MMFF)...")
        r = pipeline_c(mol_list, n_confs, params)
        results["C"] = r
        print(f"    Embed: {fmt(r['t_embed'])}  "
              f"Total: {fmt(r['t_total'])}  Confs: {r['n_embedded']}/{target}")

        # D) Shivam + RDKit MMFF
        print("\n  Running D) Shivam embed + RDKit MMFF...")
        r = pipeline_d(mol_list, n_confs, params)
        results["D"] = r
        print(f"    Embed: {fmt(r['t_embed'])}  Opt: {fmt(r['t_opt'])}  "
              f"Total: {fmt(r['t_total'])}  Confs: {r['n_embedded']}/{target}")

        # E) Shivam + Guillaume Metal MMFF
        print("\n  Running E) Shivam embed + Guillaume Metal MMFF...")
        r = pipeline_e(mol_list, n_confs, params)
        results["E"] = r
        print(f"    Embed: {fmt(r['t_embed'])}  Opt: {fmt(r['t_opt'])}  "
              f"Total: {fmt(r['t_total'])}  Confs: {r['n_embedded']}/{target}")

        # ── Summary table ──
        print("\n" + "-" * 80)
        print(f"  TIMING SUMMARY — {n_mols} mols × {n_confs} confs")
        print("-" * 80)
        hdr = f"  {'Pipeline':<42s} {'Embed':>9s} {'MMFF':>9s} {'Total':>9s} {'Confs':>6s} {'c/s':>8s} {'vs A':>7s}"
        print(hdr)
        print("  " + "-" * 77)

        t_baseline = results["A"]["t_total"]
        for key in ["A", "B", "C", "D", "E"]:
            r = results[key]
            t = r["t_total"]
            n = r["n_embedded"]
            cps = n / t if t > 0 else 0
            ratio = t_baseline / t if t > 0 else float("inf")
            tag = r["name"]
            opt_str = fmt(r["t_opt"]) if r["t_opt"] > 0 else "—"
            print(f"  {tag:<42s} {fmt(r['t_embed']):>9s} {opt_str:>9s} "
                  f"{fmt(t):>9s} {n:>6d} {cps:>8.1f} {ratio:>6.2f}x")

        # ── Energy quality ──
        ref_e = results["A"]["energies"]
        ref_valid = [e for e in ref_e if not np.isnan(e)]
        if ref_valid:
            ref_mean = np.mean(ref_valid)
            print(f"\n  CONFORMER QUALITY (MMFF94 energy, kcal/mol — lower = better)")
            print(f"  {'Pipeline':<42s} {'N':>5s} {'Mean':>10s} {'Median':>10s} {'Min':>10s} {'dE vs A':>10s}")
            print("  " + "-" * 77)
            for key in ["A", "B", "C", "D", "E"]:
                r = results[key]
                tag = r["name"]
                valid = [e for e in r["energies"] if not np.isnan(e)]
                if valid:
                    m = np.mean(valid)
                    med = np.median(valid)
                    mn = min(valid)
                    delta = m - ref_mean
                    print(f"  {tag:<42s} {len(valid):>5d} {m:>10.2f} {med:>10.2f} {mn:>10.2f} {delta:>+10.2f}")
                else:
                    print(f"  {tag:<42s} {'0':>5s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")

        print()

    # ── Per-molecule breakdown for 1 conf ──
    print("=" * 80)
    print("  PER-MOLECULE EMBEDDING SPEED (1 conformer each)")
    print("=" * 80)
    params1 = rdDistGeom.ETKDGv3()
    params1.randomSeed = 42
    params1.useRandomCoords = True

    print(f"\n  {'Molecule':<16s} {'Atoms':>6s}  {'RDKit':>10s}  {'Shivam':>10s}  {'Ratio':>8s}")
    print("  " + "-" * 56)

    for name, mol_orig in mol_list:
        n_atoms = mol_orig.GetNumAtoms()

        # RDKit
        m1 = Chem.RWMol(mol_orig)
        t0 = time.perf_counter()
        AllChem.EmbedMultipleConfs(m1, numConfs=1, params=params1)
        t_rdkit = time.perf_counter() - t0

        # Shivam
        m2 = Chem.RWMol(mol_orig)
        t0 = time.perf_counter()
        try:
            ShivamEmbedMolecules([m2], params1, confsPerMolecule=1)
        except Exception:
            pass
        t_shivam = time.perf_counter() - t0

        ratio = t_shivam / t_rdkit if t_rdkit > 0 else float("inf")
        winner = "RDKit" if ratio > 1 else "Shivam"
        print(f"  {name:<16s} {n_atoms:>6d}  {fmt(t_rdkit):>10s}  {fmt(t_shivam):>10s}  "
              f"{ratio:>7.1f}x  ({winner} faster)")

    print()
    print("=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
