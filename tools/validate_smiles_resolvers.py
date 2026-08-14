#!/usr/bin/env python3
"""Cross-validate OPSIN vs PubChem (CAS) vs CIR (CAS) on a sample of
paper-Database names.

For each (name, cas) tuple, query all three resolvers and report:
  - OPSIN(name)        : deterministic IUPAC parser
  - CIR(clean_cas)     : NIH CIR resolved via the sanitized CAS
  - PubChem(clean_cas) : PubChem CAS lookup
Canonicalize each result via RDKit. Report when ≥2 of the 3 agree (the
"correct" answer is the consensus); flag entries where they disagree.

Uses CAS as the cross-check key because OPSIN parses NAMES, but the
authoritative truth is the CAS number — CIR + PubChem both honour CAS
properly.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

_DATE_CAS = re.compile(r"^(\d+)-(\d+)-(\d+)\s+\d+:\d+:\d+$")
_CAS_OK = re.compile(r"^\d+-\d+-\d+$")


def sanitize_cas(raw: str) -> str:
    raw = (raw or "").strip()
    if _CAS_OK.match(raw):
        return raw
    m = _DATE_CAS.match(raw)
    if m:
        return f"{m.group(1)}-{m.group(2).lstrip('0') or '0'}-{m.group(3).lstrip('0') or '0'}"
    return raw


def cir_resolve(query: str) -> str:
    if not query:
        return ""
    url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(query)}/smiles"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            txt = r.read().decode("ascii", errors="replace").strip()
            if txt.startswith("Page not found"):
                return ""
            return txt
    except Exception:
        return ""


def pubchem_resolve_cas(query: str) -> str:
    if not query:
        return ""
    try:
        import pubchempy as pcp
        cs = pcp.get_compounds(query, "name")
        if cs:
            return cs[0].isomeric_smiles or cs[0].canonical_smiles or ""
    except Exception:
        pass
    return ""


def canon(smi: str) -> str:
    from rdkit import Chem
    if not smi:
        return ""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else ""


def main() -> None:
    from py2opsin import py2opsin

    paper = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx", sheet_name="Database", header=0)
    # Sample 80 entries that have BOTH name AND cas
    rows = []
    for _, r in paper.iterrows():
        name = str(r["Name"]).strip() if pd.notna(r["Name"]) else ""
        cas = sanitize_cas(str(r["CAS"]).strip() if pd.notna(r["CAS"]) else "")
        if name and cas and _CAS_OK.match(cas):
            rows.append((name, cas))
    print(f"Pool: {len(rows)} (name, valid-CAS) pairs")
    import random
    random.seed(42)
    sample = random.sample(rows, min(80, len(rows)))

    # OPSIN: batched (Java) call — single subprocess, thread-safe internally.
    t0 = time.perf_counter()
    opsin_out = py2opsin([n for n, _ in sample])
    if isinstance(opsin_out, str):
        opsin_out = [opsin_out]
    opsin_canon = [canon(s or "") for s in opsin_out]
    print(f"OPSIN batch: {time.perf_counter()-t0:.1f}s")

    def probe_net(args: tuple[int, tuple[str, str], str]) -> dict:
        i, (name, cas), op = args
        cir = canon(cir_resolve(cas))
        pc = canon(pubchem_resolve_cas(cas))
        consensus = [s for s in (op, cir, pc) if s]
        votes: dict[str, int] = {}
        for s in consensus:
            votes[s] = votes.get(s, 0) + 1
        winner, n_votes = (max(votes.items(), key=lambda x: x[1]) if votes else ("", 0))
        return {"name": name, "cas": cas, "opsin": op, "cir": cir, "pubchem": pc,
                "consensus": winner, "n_votes": n_votes,
                "opsin_correct": op == winner if winner else False,
                "pubchem_correct": pc == winner if winner else False,
                "cir_correct": cir == winner if winner else False}

    t0 = time.perf_counter()
    tasks = [(i, sample[i], opsin_canon[i]) for i in range(len(sample))]
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r in pool.map(probe_net, tasks):
            results.append(r)
    print(f"Network resolvers: {time.perf_counter()-t0:.1f}s\n")

    # Stats
    n = len(results)
    n_op_ok = sum(1 for r in results if r["opsin_correct"])
    n_pc_ok = sum(1 for r in results if r["pubchem_correct"])
    n_cir_ok = sum(1 for r in results if r["cir_correct"])
    n_no_consensus = sum(1 for r in results if r["n_votes"] < 2)
    print(f"Of {n} sampled mols (consensus = ≥2 resolvers agreeing on canonical SMILES):")
    print(f"  OPSIN matches consensus  : {n_op_ok}/{n}  ({100*n_op_ok/n:.0f}%)")
    print(f"  CIR matches consensus    : {n_cir_ok}/{n}  ({100*n_cir_ok/n:.0f}%)")
    print(f"  PubChem matches consensus: {n_pc_ok}/{n}  ({100*n_pc_ok/n:.0f}%)")
    print(f"  No consensus (all disagree or only 1 resolved): {n_no_consensus}/{n}")

    # Show disagreements
    print(f"\n=== Sample disagreements (OPSIN ≠ PubChem) ===")
    n_shown = 0
    for r in results:
        if r["opsin"] and r["pubchem"] and r["opsin"] != r["pubchem"]:
            verdict = "OPSIN=consensus" if r["opsin_correct"] else ("PubChem=consensus" if r["pubchem_correct"] else "no-consensus")
            print(f"  {r['name']:<35}  cas={r['cas']:<14}  {verdict}")
            print(f"    OPSIN  : {r['opsin']}")
            print(f"    PubChem: {r['pubchem']}")
            print(f"    CIR    : {r['cir']}")
            n_shown += 1
            if n_shown >= 10:
                print(f"    ...({sum(1 for r in results if r['opsin'] and r['pubchem'] and r['opsin'] != r['pubchem'])-10} more)")
                break


if __name__ == "__main__":
    main()
