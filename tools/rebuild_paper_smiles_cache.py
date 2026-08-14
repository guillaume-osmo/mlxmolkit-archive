#!/usr/bin/env python3
"""Rebuild the paper-Database SMILES cache with OPSIN-first ordering.

The original cache had ~700 wrong SMILES written by PubChem's name-search
endpoint, which returns the first compound whose name matches a SUBSTRING.
For paper-style chemistry names (`propylhexanoate`, `cyclohexanethiol`,
`oxiranemethanol`...) PubChem returned pyrrole, cyclohexyl peroxide,
1,4-difluorobenzene respectively — totally unrelated compounds.

Fix: OPSIN is a deterministic IUPAC parser. When it can parse a name,
the answer is correct by construction. We try it first; only fall
through to CIR (with CAS-sanitization) and PubChem when OPSIN fails.

Existing cache is renamed *.backup.json. New cache is written fresh.
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


def pubchem_resolve(query: str) -> str:
    """Last-resort PubChem name lookup. Returns "" on any failure."""
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


def main() -> None:
    from py2opsin import py2opsin

    cache_path = REPO_ROOT / "data" / "paper_database_smiles_cache.json"
    backup_path = cache_path.with_suffix(".backup.json")

    if cache_path.exists():
        if not backup_path.exists():
            backup_path.write_bytes(cache_path.read_bytes())
            print(f"Backed up existing cache to {backup_path}")
        old_cache = json.loads(cache_path.read_text())
    else:
        old_cache = {}

    paper = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx", sheet_name="Database", header=0)
    new_cache: dict[str, str] = {}
    lock = threading.Lock()

    targets: list[tuple[str, str]] = []
    for _, r in paper.iterrows():
        name = str(r["Name"]).strip() if pd.notna(r["Name"]) else ""
        cas = str(r["CAS"]).strip() if pd.notna(r["CAS"]) else ""
        if not name:
            continue
        targets.append((name, cas))
    print(f"Re-resolving {len(targets)} paper Database entries.")

    # 1) OPSIN — BATCHED single Java call (py2opsin's temp-file is NOT
    #    thread-safe; concurrent calls shuffle outputs across inputs).
    t0 = time.perf_counter()
    names = [n for n, _ in targets]
    opsin_out = py2opsin(names)
    if isinstance(opsin_out, str):
        opsin_out = [opsin_out]
    print(f"OPSIN batch over {len(names)} names: {time.perf_counter()-t0:.1f}s")

    # Stash OPSIN results; identify residual to query via network resolvers
    residual: list[tuple[int, str, str]] = []
    by_route: dict[str, int] = {}
    for i, ((name, cas), smi) in enumerate(zip(targets, opsin_out)):
        key = f"{name}||{cas}"
        if smi and "," not in smi:
            new_cache[key] = smi
            by_route["opsin"] = by_route.get("opsin", 0) + 1
        else:
            residual.append((i, name, cas))
    print(f"OPSIN resolved {len(targets) - len(residual)} entries; "
          f"{len(residual)} fall through to network resolvers.")

    # 2-5) Network resolvers in parallel ONLY for OPSIN failures.
    def net_resolve(item: tuple[int, str, str]) -> tuple[str, str, str]:
        _, name, cas = item
        key = f"{name}||{cas}"
        clean_cas = sanitize_cas(cas)
        if clean_cas:
            smi = cir_resolve(clean_cas)
            if smi:
                return key, smi, "cir_cas"
        smi = cir_resolve(name)
        if smi:
            return key, smi, "cir_name"
        smi = pubchem_resolve(clean_cas) if clean_cas else ""
        if smi:
            return key, smi, "pubchem_cas"
        smi = pubchem_resolve(name)
        if smi:
            return key, smi, "pubchem_name"
        return key, "", "fail"

    n_done = 0
    t0 = time.perf_counter()
    if residual:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for key, smi, route in pool.map(net_resolve, residual):
                with lock:
                    new_cache[key] = smi
                    by_route[route] = by_route.get(route, 0) + 1
                    n_done += 1
                    if n_done % 50 == 0:
                        cache_path.write_text(json.dumps(new_cache, indent=2, sort_keys=True))
                        rate = n_done / max(time.perf_counter() - t0, 1e-6)
                        eta = (len(residual) - n_done) / max(rate, 1e-6)
                        print(f"  net {n_done:>5d}/{len(residual)}  rate={rate:.1f}/s  ETA {eta:.0f}s", flush=True)

    # Diff vs old
    n_changed = sum(1 for k, v in new_cache.items() if old_cache.get(k, "") != v)

    cache_path.write_text(json.dumps(new_cache, indent=2, sort_keys=True))
    resolved = sum(1 for v in new_cache.values() if v)
    print(f"\nDone. {len(new_cache)} entries, {resolved} resolved ({100*resolved/len(new_cache):.1f}%).")
    print(f"Changed vs old cache: {n_changed} entries.")
    print(f"By route: {by_route}")


if __name__ == "__main__":
    main()
