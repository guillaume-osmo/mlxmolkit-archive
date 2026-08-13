"""openMOPAC vs mlxmolkit PM6/PM6_D on 200 molecules — task #18.

Ground truth is MOPAC's own PM6 1SCF at a frozen geometry, so the only thing
compared is the electronic structure: same nuclei, same positions, same method.

The 100-molecule perfumery set is all sp chemistry (alcohol, terpene, ester,
ether, ketone), which would leave the d-orbital path — the part that carries
sulfur, phosphorus and the halogens, and most of the recent work — untested. So
it is extended with 100 molecules chosen for element coverage rather than for
odour, and the report breaks the two groups out separately.

Only structures and SMARTS-derived chemical classes are recorded. No odour
descriptors appear here or in the output.

Run:  PYTHONPATH=. python tools/bench_mopac_parity_200.py
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

def _find_mopac() -> str:
    """Locate the MOPAC binary.

    It is installed in the conda *base* prefix, not in the `osmo` env, so
    neither `shutil.which` under the env nor a hardcoded env path finds it —
    which is exactly the wrong conclusion ("MOPAC is not installed") to draw
    from a benchmark that silently produces nothing.
    """
    import os
    import shutil

    found = shutil.which("mopac")
    if found:
        return found
    candidates = [
        os.path.expanduser("~/miniconda3/bin/mopac"),
        os.path.expanduser("~/miniconda3/envs/osmo/bin/mopac"),
        "/opt/homebrew/bin/mopac",
        "/usr/local/bin/mopac",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "MOPAC not found. Looked on PATH and in: " + ", ".join(candidates)
    )


MOPAC = _find_mopac()
PERFUMERY = Path("tests/data/perfumery_benchmark_100.csv")

# Chosen for element and motif coverage, not for smell: every one carries at
# least one atom that routes through the d-orbital path (S, P, Cl, Br, I) or
# exercises a ring/functional motif the perfumery set lacks.
D_ELEMENT_SET = [
    "CSC", "CCSC", "CSCC=O", "CSSC", "CSc1ccccc1", "c1ccsc1", "Cc1ccsc1",
    "CS(=O)C", "CS(=O)(=O)C", "CCS", "CC(=O)SC", "CSCCC=O", "CSCCCO",
    "c1ccc2sccc2c1", "CSCC(=O)C", "CCSSCC", "CSCCCC=O", "c1csc(c1)C=O",
    "CC1=CC(=O)CC(C)(C)S1", "CSCCO",
    "Clc1ccccc1", "ClCCl", "ClC(Cl)Cl", "ClC(Cl)(Cl)Cl", "CCCl", "ClCCCl",
    "Clc1ccc(Cl)cc1", "ClCc1ccccc1", "Clc1cccc(Cl)c1", "CC(Cl)Cl",
    "Brc1ccccc1", "BrCC", "BrCCBr", "Brc1ccc(Br)cc1", "BrCc1ccccc1",
    "Ic1ccccc1", "ICI", "CCI",
    "COP(=O)(OC)OC", "CCOP(=O)(OCC)OCC", "CP(C)C", "c1ccc(cc1)P(c1ccccc1)c1ccccc1",
    "FC(F)(F)c1ccccc1", "FCc1ccccc1", "Fc1ccccc1", "FC(F)(F)S",
    "c1ccc(cc1)S(=O)(=O)N", "CS(=O)(=O)c1ccccc1", "c1ccc(cc1)SC",
    "N#Cc1ccccc1", "CC#N", "c1ccncc1", "c1cc[nH]c1", "c1ccoc1", "C1CCOC1",
    "O=C1CCCN1", "c1ccc2[nH]ccc2c1", "O=C1CCCCC1", "C1CCNCC1",
    "CC(=O)N(C)C", "CN(C)C=O", "O=C(N)c1ccccc1", "CC(=O)Nc1ccccc1",
    "OCC(O)CO", "OCCO", "OCCOCCO", "COCCOC", "CCOCC",
    "c1ccc(cc1)C(=O)O", "CC(=O)O", "OC(=O)CCC(=O)O", "OC(=O)c1ccccc1O",
    "c1ccc(cc1)N", "CN(C)c1ccccc1", "Nc1ccc(N)cc1", "NCCN",
    "O=[N+]([O-])c1ccccc1", "CC(=O)c1ccc(cc1)OC", "COc1ccc(cc1)C=O",
    "c1ccc2c(c1)cccc2O", "Oc1ccccc1", "Oc1ccc(cc1)C(C)(C)C",
    "C1CC2CCC1C2", "C1CCCCC1", "C1CCCCCC1", "c1ccc2ccccc2c1",
    "CC1(C)C2CCC1(C)C(=O)C2", "CC1=CCC2CC1C2(C)C", "CC(C)C1CCC(C)CC1=O",
    "C=CCSSCC=C", "C=CCSC", "CC(C)=CCO", "CC(C)=CCCC(C)=CCO",
    "O=Cc1ccccc1", "CC(=O)c1ccccc1", "c1ccc(cc1)CC=O", "OCc1ccccc1",
    "CCCCCCCC=O", "CCCCCCCCO", "CCCCCC(=O)OCC", "CC(C)COC(=O)C",
]


def classes_of(mol) -> str:
    """SMARTS-derived chemical classes. Deliberately structural, never odour."""
    pats = [
        ("thioether", "[#16X2]([#6])[#6]"), ("thiol", "[#16X2H]"),
        ("disulfide", "[#16X2][#16X2]"), ("sulfoxide", "[#16X3](=O)"),
        ("sulfone", "[#16X4](=O)(=O)"), ("thiophene", "c1ccsc1"),
        ("chloro", "[Cl]"), ("bromo", "[Br]"), ("iodo", "[I]"), ("fluoro", "[F]"),
        ("phosphate", "[#15](=O)"), ("phosphine", "[#15X3]"),
        ("nitrile", "C#N"), ("nitro", "[N+](=O)[O-]"), ("amide", "C(=O)N"),
        ("amine", "[NX3;H2,H1,H0;!$(NC=O)]"), ("carboxylic", "C(=O)[OH]"),
        ("ester", "C(=O)O[#6]"), ("ketone", "[#6]C(=O)[#6]"), ("aldehyde", "[CX3H1]=O"),
        ("alcohol", "[OX2H]"), ("ether", "[OD2]([#6])[#6]"),
        ("phenol", "c[OX2H]"), ("aromatic", "c1ccccc1"),
    ]
    hits = [n for n, s in pats
            if mol.HasSubstructMatch(Chem.MolFromSmarts(s))]
    return "|".join(hits) if hits else "hydrocarbon"


def geometry(smi: str, seed: int = 1):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        return None
    conf = m.GetConformer()
    Z = [a.GetAtomicNum() for a in m.GetAtoms()]
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    return Z, sym, xyz, m


def run_mopac(name, sym, xyz, workdir: Path):
    """PM6 1SCF at the given geometry. Returns (heat of formation, charges)."""
    mop = workdir / f"{name}.mop"
    lines = ["PM6 1SCF", name, ""]
    for s, (x, y, z) in zip(sym, xyz):
        lines.append(f"{s:2s} {x:.6f} 1 {y:.6f} 1 {z:.6f} 1")
    mop.write_text("\n".join(lines) + "\n")
    try:
        subprocess.run([MOPAC, mop.name], cwd=workdir,
                       capture_output=True, timeout=300)
        out = (workdir / f"{name}.out").read_text()
    except Exception:
        return None
    m = re.search(r"FINAL HEAT OF FORMATION\s*=\s*(-?\d+\.\d+)\s*KCAL/MOL", out)
    if m is None:
        return None
    hf = float(m.group(1))
    qm = re.search(r"NET ATOMIC CHARGES.*?\n(.*?)\n\s*DIPOLE", out, re.S)
    charges = None
    if qm:
        try:
            charges = [float(ln.split()[2]) for ln in qm.group(1).splitlines()
                       if len(ln.split()) >= 3 and ln.split()[0].isdigit()]
        except (ValueError, IndexError):
            charges = None
    return hf, charges


def main():
    from mlxmolkit.nddo.scf import nddo_energy

    rows = []
    for r in csv.DictReader(open(PERFUMERY)):
        rows.append((r["smiles"], "perfumery"))
    seen = {s for s, _ in rows}
    for s in D_ELEMENT_SET:
        if s not in seen:
            rows.append((s, "coverage"))
            seen.add(s)
    print(f"{len(rows)} molecules: "
          f"{sum(1 for _, g in rows if g == 'perfumery')} perfumery, "
          f"{sum(1 for _, g in rows if g == 'coverage')} coverage")

    workdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mopac_parity")
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (smi, group) in enumerate(rows):
        g = geometry(smi)
        if g is None:
            print(f"  skip (embed) {smi}")
            continue
        Z, sym, xyz, mol = g
        ref = run_mopac(f"m{i:04d}", sym, xyz, workdir)
        if ref is None:
            print(f"  skip (mopac) {smi}")
            continue
        hf_mopac, q_mopac = ref

        rec = dict(smiles=smi, group=group, n_atoms=len(Z),
                   classes=classes_of(mol),
                   has_d=bool({15, 16, 17, 35, 53} & set(Z)),
                   hf_mopac=hf_mopac)
        for method in ("PM6", "PM6_D"):
            t = time.perf_counter()
            try:
                r = nddo_energy(Z, xyz, method=method, max_iter=300, conv_tol=1e-8)
            except Exception as exc:
                rec[f"err_{method}"] = f"{type(exc).__name__}: {exc}"
                continue
            rec[f"t_{method}"] = time.perf_counter() - t
            rec[f"hf_{method}"] = r["heat_of_formation_kcal"]
            rec[f"conv_{method}"] = bool(r["converged"])
            if q_mopac is not None and len(q_mopac) == len(Z):
                dq = np.abs(np.asarray(r["charges"]) - np.asarray(q_mopac)).max()
                rec[f"dq_{method}"] = float(dq)
        results.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    out = Path("tests/data/mopac_parity_200.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {len(results)} records to {out}")
    return results


if __name__ == "__main__":
    main()
