#!/usr/bin/env python3
"""Bulk fetch SMILES via PubChem for the RSC Adv 2026 paper Database.

The benchmark's PubChem lookups happen serially during runs. To match
the full 1588-mol Database against CHAOS we need SMILES up front. This
runs concurrent PubChem queries (CAS first, name fallback) and writes
back to data/paper_database_smiles_cache.json. Resumable.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cache_path = REPO_ROOT / "data" / "paper_database_smiles_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    lock = threading.Lock()

    paper = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx", sheet_name="Database", header=0)

    todo: list[tuple[str, str]] = []
    for _, r in paper.iterrows():
        name = str(r["Name"]).strip() if pd.notna(r["Name"]) else ""
        cas = str(r["CAS"]).strip() if pd.notna(r["CAS"]) else ""
        if not name:
            continue
        key = f"{name}||{cas}"
        if key in cache:  # includes empty-string entries (known-failed)
            continue
        todo.append((name, cas))
    print(f"Cache: {len(cache)} entries.  To fetch: {len(todo)}.")

    import pubchempy as pcp

    def fetch_one(item: tuple[str, str]) -> tuple[str, str]:
        name, cas = item
        key = f"{name}||{cas}"
        for q in (cas, name):
            if not q:
                continue
            try:
                cs = pcp.get_compounds(q, "name")
                if cs:
                    smi = cs[0].isomeric_smiles or cs[0].canonical_smiles
                    if smi:
                        return key, smi
            except Exception:
                pass
        return key, ""

    n_done = 0
    t0 = time.perf_counter()
    save_every = 50
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key, smi in pool.map(fetch_one, todo):
            with lock:
                cache[key] = smi
                n_done += 1
                if n_done % save_every == 0:
                    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
                    rate = n_done / max(time.perf_counter() - t0, 1e-6)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    print(f"  {n_done:>5d}/{len(todo)}  rate={rate:.1f}/s  ETA {eta:.0f}s  ({(cache_path.stat().st_size/1024):.0f} KB)", flush=True)

    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    resolved = sum(1 for v in cache.values() if v)
    print(f"\nDone. Cache: {len(cache)} entries, {resolved} with SMILES, {len(cache)-resolved} known-failed.")


if __name__ == "__main__":
    main()
