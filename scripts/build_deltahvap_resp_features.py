#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume — SPDX: MIT
"""New-pipeline feature block for the deltaHvapv3 4491-mol set (redo of the
g-xTB sigma-profile block with RESP/CAMM/GB features). Checkpointed/resumable.

Per molecule: g-xTB SCF -> RESP charges -> a RESP "sigma profile"
(sigma_a = q_a/area_a, area-weighted histogram, the analog of the COSMO sigma
profile) + CAMM/GB/global scalars. Row-aligned to the union CSV.

Usage: conda activate rdkit_build_fb && cd /Users/tgg/Github/mlxmolkit
       python3 scripts/build_deltahvap_resp_features.py [--max N]
"""
from __future__ import annotations
import sys, os, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA, Descriptors
os.environ.setdefault("OMP_NUM_THREADS", "1")    # 1 BLAS thread/worker for the pool
from mlxmolkit.xtb.scf_gxtb import gxtb_energy
from mlxmolkit.xtb.multipole_integrals import multipole_matrices   # numpy (CPU, fork-safe)
from mlxmolkit.xtb.gxtb_aes_assembly import mmompop_fast
from gxtb_resp_solvation import embed, GXTB_KW, mk_grid, camm_esp, resp_fit, hydration, ANG2BOHR
from concurrent.futures import ProcessPoolExecutor

ALLOWED = {1, 6, 7, 8, 9}
CSV = "data/delta_hvap_v2/deltaHvapv3_full4491_organic_singlefrag_gxtbvalid_broad_train.csv"
OUT = f"{HERE}/deltahvap_resp_features.npz"
SIG_BINS = np.linspace(-0.03, 0.03, 22)          # RESP sigma-profile bins (e/Ang^2)


def featurize(smi):
    m = embed(Chem.AddHs(Chem.MolFromSmiles(smi)))
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    C = m.GetConformer().GetPositions().astype(float)
    res = gxtb_energy(Z, C, **GXTB_KW)
    cb = res["basis"].cao_basis
    if max(b.l_total for b in cb) > 1:
        raise ValueError("d-shell")
    aoat = np.array([x.atom_idx for x in cb])
    P = np.asarray(res["density"]); S = np.asarray(res["S"]); q = np.asarray(res["atom_charges"])
    _, dp, qpi = multipole_matrices(cb)          # numpy CAO (fork-safe, no GPU)
    dipm, qpm = mmompop_fast(P, S, dp, qpi, aoat, C * ANG2BOHR)
    grid = mk_grid(Z, C)
    qr = resp_fit(grid, C, camm_esp(grid, C, q, dipm, qpm), float(round(q.sum())))
    # per-atom SASA -> RESP sigma profile
    rad = rdFreeSASA.classifyAtoms(m); tot = rdFreeSASA.CalcSASA(m, rad)
    area = np.array([float(m.GetAtomWithIdx(i).GetProp("SASA")) for i in range(len(Z))])
    sig = qr / np.maximum(area, 1e-6)             # surface charge density e/Ang^2
    prof, _ = np.histogram(sig, bins=SIG_BINS, weights=area)   # area-weighted (21 bins)
    solv = hydration(Z, C, qr, m)
    dipole = np.linalg.norm((qr[:, None] * (C * ANG2BOHR)).sum(0))
    scal = [solv["dG_pol"], solv["dG_np"], solv["dG_hyd"], tot, dipole,
            float(np.abs(qr).sum()), float(qr.max()), float(qr.min()), float(qr.std()),
            float(np.linalg.norm(dipm)), float(np.linalg.norm(qpm)),
            int((Z == 8).sum()), int((Z == 7).sum()), int((Z == 9).sum()),
            Descriptors.MolWt(m), len(Z)]
    return np.concatenate([prof, scal])


def _worker(args):
    i, smi = args
    m = Chem.MolFromSmiles(smi)
    if not (m and set(a.GetAtomicNum() for a in m.GetAtoms()) <= ALLOWED):
        return i, None
    try:
        return i, featurize(smi)
    except Exception:
        return i, None


def main(max_n, workers):
    df = pd.read_csv(CSV)
    smiles = df["canonical_smiles"].astype(str).tolist()
    if max_n: smiles = smiles[:max_n]
    N = len(smiles)
    nfeat = len(SIG_BINS) - 1 + 16
    feats = np.zeros((N, nfeat)); valid = np.zeros(N, bool)
    if os.path.exists(OUT) and max_n is None:
        z = np.load(OUT)
        if z["feats"].shape == (N, nfeat):            # resume only if same run
            feats = z["feats"]; valid = z["valid"]
    todo = [(i, smiles[i]) for i in range(N) if not valid[i]]
    print(f"{N} mols, {len(todo)} to do, {workers} workers", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, f in ex.map(_worker, todo, chunksize=4):
            if f is not None:
                feats[i] = f; valid[i] = True
            done += 1
            if done % 200 == 0:
                np.savez(OUT, feats=feats, valid=valid, done=int(valid.sum()), nfeat=nfeat)
                print(f"  {done}/{len(todo)}  valid={int(valid.sum())}", flush=True)
    np.savez(OUT, feats=feats, valid=valid, done=int(valid.sum()), nfeat=nfeat)
    print(f"done: {int(valid.sum())}/{N} valid -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); main(a.max, a.workers)
