#!/usr/bin/env python3
"""Build an index of CHAOS dataset SMILES without extracting the zip.

CHAOS.zip contains ~53k gaussian_CHAOS JSON files. We only need
``general.CanonicalSMILES`` + ``general.MolecularFormula`` per entry,
which lives in the first ~500 bytes of each file. Parsing all 53k
JSON files in parallel produces a compact CSV index.

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/chaos_index_smiles.py \\
        --zip /Users/guillaume-osmo/Github/data/CHAOS.zip \\
        --out data/chaos_index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _extract_general(chunk: str) -> tuple[str, str] | None:
    idx_g = chunk.find('"general"')
    if idx_g < 0:
        return None
    brace_start = chunk.find("{", idx_g)
    depth = 0
    end = None
    for i, c in enumerate(chunk[brace_start:], start=brace_start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        obj = json.loads(chunk[brace_start:end])
    except json.JSONDecodeError:
        return None
    return obj.get("CanonicalSMILES", "") or "", obj.get("MolecularFormula", "") or ""


def _process_batch(args: tuple[str, list[str]]) -> list[tuple[str, str, str]]:
    """Worker process: open zip once, stream entries, return triples."""
    zip_path, names = args
    out: list[tuple[str, str, str]] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in names:
                try:
                    with zf.open(name) as f:
                        chunk = f.read(4096).decode("utf-8", errors="replace")
                except Exception:
                    continue
                got = _extract_general(chunk)
                if got is None:
                    continue
                smi, formula = got
                out.append((Path(name).stem, smi, formula))
    except Exception:
        pass
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=Path("/Users/guillaume-osmo/Github/data/CHAOS.zip"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    p.add_argument("--limit", type=int, default=0, help="0 = all entries")
    args = p.parse_args()

    print(f"Opening {args.zip} ({args.zip.stat().st_size/1024**3:.1f} GB)...")
    with zipfile.ZipFile(args.zip, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
    if args.limit:
        names = names[: args.limit]
    print(f"  {len(names)} JSON entries to index. Using {args.workers} workers.")

    # Split into batches so each worker opens the zip ONCE per batch.
    batch_size = max(200, len(names) // (args.workers * 4))
    batches: list[list[str]] = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    print(f"  {len(batches)} batches of ≤{batch_size} entries each.")

    rows: list[tuple[str, str, str]] = []
    t0 = time.perf_counter()
    done_entries = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_batch, (str(args.zip), b)): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            res = fut.result()
            rows.extend(res)
            done_entries += len(batch)
            elapsed = time.perf_counter() - t0
            rate = done_entries / elapsed
            eta = max(0.0, (len(names) - done_entries) / rate)
            print(f"  {done_entries:>6d} / {len(names)}  ({rate:.0f} files/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Indexed {len(rows)}/{len(names)} entries in {elapsed:.1f}s ({len(rows)/elapsed:.0f} files/s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chaos_id", "canonical_smiles", "molecular_formula"])
        rows.sort(key=lambda r: int(r[0]) if r[0].isdigit() else 0)
        w.writerows(rows)
    print(f"Wrote {args.out}  ({args.out.stat().st_size/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
