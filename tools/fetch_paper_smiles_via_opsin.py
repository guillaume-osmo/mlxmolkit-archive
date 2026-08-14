#!/usr/bin/env python3
"""Third-pass SMILES resolver for the paper Database: OPSIN + CAS-sanitized
CIR fallback.

After PubChem (22%) and NIH CIR (46%), ~859 entries remain unresolved.
Two attack vectors:

1. **OPSIN**: Open-source IUPAC name parser (py2opsin). 72% of failed
   entries are IUPAC-style (`2-methyl-3-hexyne` etc.) — exactly OPSIN's
   sweet spot.

2. **CAS sanitization**: ~840/859 failed entries have a CAS field, but
   many are mangled by Excel auto-formatting:
       "2459-10-1"  →  "2459-10-01 00:00:00"   (treated as date)
   We strip the time portion and try CIR again with the cleaned CAS.
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
    raw = raw.strip()
    if not raw:
        return ""
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


def main() -> None:
    from py2opsin import py2opsin

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
        if cache.get(key):
            continue
        todo.append((name, cas))
    print(f"Re-trying {len(todo)} unresolved entries with OPSIN + sanitized-CAS CIR.")

    def fetch_one(item: tuple[str, str]) -> tuple[str, str, str]:
        name, cas = item
        key = f"{name}||{cas}"
        # 1) OPSIN on the name (no network)
        try:
            smi = py2opsin(name)
            if smi and "," not in smi:  # ignore comma-separated multi-output
                return key, smi, "opsin"
        except Exception:
            pass
        # 2) Sanitized CAS via CIR
        clean_cas = sanitize_cas(cas)
        if clean_cas and clean_cas != cas:
            smi = cir_resolve(clean_cas)
            if smi:
                return key, smi, "cir_cas_clean"
        # 3) Try CIR on the name once more (in case CIR was throttled before)
        smi = cir_resolve(name)
        if smi:
            return key, smi, "cir_name_retry"
        return key, "", "fail"

    n_done, n_added = 0, 0
    by_route: dict[str, int] = {}
    t0 = time.perf_counter()
    save_every = 50
    with ThreadPoolExecutor(max_workers=12) as pool:
        for key, smi, route in pool.map(fetch_one, todo):
            with lock:
                if smi:
                    cache[key] = smi
                    n_added += 1
                by_route[route] = by_route.get(route, 0) + 1
                n_done += 1
                if n_done % save_every == 0:
                    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
                    rate = n_done / max(time.perf_counter() - t0, 1e-6)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    print(f"  {n_done:>5d}/{len(todo)}  +{n_added} new  rate={rate:.1f}/s  ETA {eta:.0f}s", flush=True)

    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    resolved_total = sum(1 for v in cache.values() if v)
    print(f"\nDone. +{n_added} new SMILES this pass.")
    print(f"By route: {by_route}")
    print(f"Cache now: {len(cache)} entries, {resolved_total} with SMILES "
          f"({100*resolved_total/len(cache):.1f}% coverage).")


if __name__ == "__main__":
    main()
