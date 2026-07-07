#!/usr/bin/env python
"""Pre-build a shared 3D-conformer cache for LIT-PCBA targets.

The fusion eval regenerates ETKDG+MMFF conformers for every student it embeds,
even though `embed3d_ready` is deterministic (fixed seed) so the conformers are
identical across students. This builds them ONCE, in PARALLEL, and pickles the
3D RDKit mols per target. The eval then just loads + GPU-encodes (~30s) instead
of regenerating (~15 min) each time.

No MLX import here -> safe to fork a multiprocessing Pool.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Shared CPU conformer source of truth. Imports only rdkit + numpy (no MLX/Metal),
# so importing it stays safe inside the multiprocessing workers below.
from tools.conformer_source import embed_molecule_3d


def read_smi(path, limit):
    out = []
    for line in open(path):
        parts = line.split()
        if parts:
            out.append(parts[0])
        if len(out) >= limit:
            break
    return out


def embed3d_ready(smi):
    # Same ETKDGv3 + MMFF source of truth as the eval and teacher caches, so this
    # prebuilt conformer cache is bit-compatible with cheese_litpcba_setsim_eval.
    try:
        return embed_molecule_3d(smi, seed=0xC0FFEE, max_iters=200)
    except Exception:
        return None


def _embed_one(item):
    smi, lab = item
    m = embed3d_ready(smi)
    if m is None:
        return None
    return (Chem.MolToMolBlock(m), lab, smi)  # molblock is picklable + compact


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+",
                    default=["ALDH1", "VDR", "PKM2", "MAPK1", "FEN1", "KAT2A", "GBA"])
    ap.add_argument("--litpcba-dir", type=Path, default=Path("data/litpcba"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cheese_projection/litpcba_conformers"))
    ap.add_argument("--n-actives", type=int, default=150)
    ap.add_argument("--n-inactives", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=max(1, cpu_count() - 2))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in a.targets:
        out = a.out_dir / f"{tgt}_a{a.n_actives}_i{a.n_inactives}_conformers.pkl"
        if out.exists():
            print(f"[{tgt}] cached already", flush=True)
            continue
        act = read_smi(a.litpcba_dir / tgt / "active_V.smi", a.n_actives)
        dec = read_smi(a.litpcba_dir / tgt / "inactive_V.smi", a.n_inactives)
        items = [(s, 1) for s in act] + [(s, 0) for s in dec]
        import time
        t0 = time.perf_counter()
        with Pool(a.procs) as pool:
            results = [r for r in pool.map(_embed_one, items, chunksize=16) if r is not None]
        molblocks = [r[0] for r in results]
        labels = [r[1] for r in results]
        smis = [r[2] for r in results]
        with open(out, "wb") as f:
            pickle.dump({"molblocks": molblocks, "labels": labels, "smis": smis}, f)
        print(f"[{tgt}] {len(results)} conformers in {time.perf_counter()-t0:.0f}s "
              f"(procs={a.procs}) -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
