#!/usr/bin/env python3
"""Convert all 53k CHAOS entries to 25a-quality σ-potential matrix.

For each entry:
  1. Read CHAOS SegmentList (DFT-COSMO raw segments) from the zip stream.
  2. Build a CosmoSegments object.
  3. Apply sigma_potential() with σ-orth correction (openCOSMORS25a kernel).
  4. Emit (chaos_id, canonical_smiles, μ(σ) on 61-bin grid).

No ORCA, no g-xTB — pure data processing on CHAOS's pre-computed
DFT-COSMO output. Wall ≈ minutes for 53k mols on 12 workers.

Output: NPZ with chaos_ids, smiles, sigma_grid (61), mu_matrix (N, 61).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


PAPER_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)
_GENERAL_RE = re.compile(r'"CanonicalSMILES"\s*:\s*"([^"]*)"')


def _extract_smiles(blob: str) -> str:
    m = _GENERAL_RE.search(blob[:2048])  # always in the first 1-2 KB
    return m.group(1) if m else ""


def _process_batch(args: tuple[str, list[str]]) -> list[tuple[str, str, list[float]]]:
    """Worker: open zip once, process N entries, return (id, smi, μ tuples)."""
    from mlxmolkit.xtb import sigma_potential
    from mlxmolkit.xtb.cosmo_sigma import CosmoSegments

    zip_path, names = args
    out: list[tuple[str, str, list[float]]] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in names:
                try:
                    with zf.open(name) as f:
                        blob = f.read().decode("utf-8", errors="replace")
                    smi = _extract_smiles(blob)
                    data = json.loads(blob)
                    sl = np.asarray(data["solvation"]["SegmentList"], dtype=np.float64)
                    if sl.size == 0 or sl.shape[1] < 8:
                        continue
                    cs = CosmoSegments(
                        epsilon=float("inf"), fepsi=1.0,
                        area=float(data["solvation"]["CavArea"]),
                        volume=float(data["solvation"]["CavVolume"]),
                        total_screening_charge=0.0,
                        total_energy_hartree=float("nan"),
                        dielectric_energy_hartree=float("nan"),
                        atom_radii=np.zeros(1),
                        atom_coords_bohr=np.zeros((1, 3)),
                        atom_z=[0],
                        segments_atom=sl[:, 1].astype(np.intp),
                        segments_xyz_bohr=sl[:, 2:5].copy(),
                        segments_charge=sl[:, 5].copy(),
                        segments_area=sl[:, 6].copy(),
                        segments_sigma=sl[:, 7].copy(),
                        segments_potential=sl[:, 8].copy() if sl.shape[1] > 8 else np.zeros(sl.shape[0]),
                        cosmo_text="",
                    )
                    _, mu = sigma_potential(cs, sigma_grid_e_per_A2=PAPER_GRID)
                    chaos_id = Path(name).stem
                    out.append((chaos_id, smi, mu.tolist()))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=Path("/Users/guillaume-osmo/Github/data/CHAOS.zip"))
    p.add_argument("--out-npz", type=Path, default=REPO_ROOT / "data" / "chaos_25a_mu_matrix.npz")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    args = p.parse_args()

    print(f"Opening {args.zip} ({args.zip.stat().st_size/1024**3:.1f} GB)…")
    with zipfile.ZipFile(args.zip, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
    if args.limit:
        names = names[: args.limit]
    print(f"  {len(names)} CHAOS entries to process. Using {args.workers} workers.")

    batch_size = max(200, len(names) // (args.workers * 4))
    batches: list[list[str]] = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    print(f"  {len(batches)} batches × ≤{batch_size} entries.")

    chaos_ids: list[str] = []
    smiles: list[str] = []
    mu_rows: list[list[float]] = []

    t0 = time.perf_counter()
    done_entries = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_batch, (str(args.zip), b)): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            res = fut.result()
            for cid, smi, mu in res:
                chaos_ids.append(cid)
                smiles.append(smi)
                mu_rows.append(mu)
            done_entries += len(batch)
            elapsed = time.perf_counter() - t0
            rate = done_entries / max(elapsed, 1e-6)
            eta = max(0.0, (len(names) - done_entries) / rate)
            print(f"  {done_entries:>6d} / {len(names)}  "
                  f"(processed {len(mu_rows)} successful, {rate:.0f} files/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {len(mu_rows)} σ-potentials in {elapsed:.1f}s "
          f"({len(mu_rows)/elapsed:.0f} mols/s, errors: {len(names)-len(mu_rows)})")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    mu_mat = np.asarray(mu_rows, dtype=np.float64)
    np.savez_compressed(
        args.out_npz,
        chaos_ids=np.asarray(chaos_ids),
        canonical_smiles=np.asarray(smiles),
        sigma_grid_e_per_A2=PAPER_GRID,
        mu_J_per_mol=mu_mat,
    )
    print(f"Wrote {args.out_npz}  ({args.out_npz.stat().st_size/1024**2:.1f} MB, shape={mu_mat.shape})")


if __name__ == "__main__":
    main()
