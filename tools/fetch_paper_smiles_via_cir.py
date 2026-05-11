#!/usr/bin/env python3
"""Second-pass SMILES resolver via NIH CIR for paper-Database names that
PubChem couldn't resolve.

CIR (Chemical Identifier Resolver) at https://cactus.nci.nih.gov has
broader chemistry-name coverage than PubChem — handles common names,
abbreviations, and CAS more robustly. Runs in parallel against the
"known-failed" entries in the cache.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def cir_resolve(query: str) -> str:
    """Query NIH CIR for canonical SMILES; empty string on failure."""
    if not query:
        return ""
    url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(query)}/smiles"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read().decode("ascii", errors="replace").strip()
    except Exception:
        return ""


def main() -> None:
    cache_path = REPO_ROOT / "data" / "paper_database_smiles_cache.json"
    cache = json.loads(cache_path.read_text())
    paper = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx", sheet_name="Database", header=0)
    lock = threading.Lock()

    todo: list[tuple[str, str]] = []
    for _, r in paper.iterrows():
        name = str(r["Name"]).strip() if pd.notna(r["Name"]) else ""
        cas = str(r["CAS"]).strip() if pd.notna(r["CAS"]) else ""
        if not name:
            continue
        key = f"{name}||{cas}"
        if cache.get(key):  # already resolved
            continue
        todo.append((name, cas))
    print(f"Trying CIR on {len(todo)} entries that PubChem failed to resolve.")

    def fetch_one(item: tuple[str, str]) -> tuple[str, str]:
        name, cas = item
        key = f"{name}||{cas}"
        for q in (cas, name):
            if not q:
                continue
            smi = cir_resolve(q)
            if smi and not smi.startswith("Page not found"):
                return key, smi
        return key, ""

    n_done = 0
    n_resolved_now = 0
    t0 = time.perf_counter()
    save_every = 30
    with ThreadPoolExecutor(max_workers=6) as pool:
        for key, smi in pool.map(fetch_one, todo):
            with lock:
                if smi and not cache.get(key):
                    cache[key] = smi
                    n_resolved_now += 1
                n_done += 1
                if n_done % save_every == 0:
                    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
                    rate = n_done / max(time.perf_counter() - t0, 1e-6)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    print(f"  {n_done:>5d}/{len(todo)}  +{n_resolved_now} resolved  rate={rate:.1f}/s  ETA {eta:.0f}s", flush=True)

    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    resolved_total = sum(1 for v in cache.values() if v)
    print(f"\nDone. CIR added {n_resolved_now} new SMILES.")
    print(f"Cache now: {len(cache)} entries, {resolved_total} with SMILES "
          f"({100*resolved_total/len(cache):.1f}% coverage).")


if __name__ == "__main__":
    main()
