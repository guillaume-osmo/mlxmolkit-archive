#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume — SPDX: MIT
"""Full deltaHvapv3 4491-mol feature dataset from the NEW GPU int1e-RESP pipeline
(all elements; no COSMO sigma). Single-process (int1e is GPU/MLX -> no fork). Checkpointed.

  conda activate rdkit_build_fb && cd /Users/tgg/Github/mlxmolkit
  python3 scripts/build_resp_dataset.py [--max N]
"""
from __future__ import annotations
import sys, os, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFreeSASA, Descriptors
from mlxmolkit.xtb.scf_gxtb import gxtb_energy
from gxtb_resp_solvation import embed, GXTB_KW, resp_int1e, hydration, ANG2BOHR

CSV = "data/delta_hvap_v2/deltaHvapv3_full4491_organic_singlefrag_gxtbvalid_broad_train.csv"
OUT = f"{HERE}/deltahvap_resp_int1e_features.npz"
SIG_BINS = np.linspace(-0.03, 0.03, 22)          # RESP sigma-profile bins (21) e/Ang^2


def featurize(smi):
    m = embed(Chem.AddHs(Chem.MolFromSmiles(smi)))
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    C = m.GetConformer().GetPositions().astype(float)
    res = gxtb_energy(Z, C, **GXTB_KW)
    qr = resp_int1e(res, Z, C, qtot=float(round(np.asarray(res["atom_charges"]).sum())))  # GPU int1e RESP
    rad = rdFreeSASA.classifyAtoms(m); tot = rdFreeSASA.CalcSASA(m, rad)
    area = np.array([float(m.GetAtomWithIdx(i).GetProp("SASA")) for i in range(len(Z))])
    sig = qr / np.maximum(area, 1e-6)
    prof, _ = np.histogram(sig, bins=SIG_BINS, weights=area)        # RESP sigma-profile (21)
    solv = hydration(Z, C, qr, m)
    dip = np.linalg.norm((qr[:, None] * (C * ANG2BOHR)).sum(0))
    scal = [solv["dG_pol"], solv["dG_np"], solv["dG_hyd"], tot, dip,
            float(np.abs(qr).sum()), float(qr.max()), float(qr.min()), float(qr.std()),
            Descriptors.MolWt(m), len(Z)]
    return np.concatenate([prof, scal])


def main(max_n):
    df = pd.read_csv(CSV); smiles = df["canonical_smiles"].astype(str).tolist()
    if max_n: smiles = smiles[:max_n]
    N = len(smiles); nfeat = len(SIG_BINS) - 1 + 11
    feats = np.zeros((N, nfeat)); valid = np.zeros(N, bool)
    if os.path.exists(OUT) and not max_n:
        z = np.load(OUT)
        if z["feats"].shape == (N, nfeat): feats = z["feats"]; valid = z["valid"]
    import time; t0 = time.time()
    for i in range(N):
        if valid[i]: continue
        try:
            feats[i] = featurize(smiles[i]); valid[i] = True
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            np.savez(OUT, feats=feats, valid=valid, nfeat=nfeat)
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{N} valid={int(valid.sum())} {rate:.1f}/s eta {(N-i-1)/rate/60:.0f}min", flush=True)
    np.savez(OUT, feats=feats, valid=valid, nfeat=nfeat)
    print(f"done: {int(valid.sum())}/{N} -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=None)
    main(ap.parse_args().max)
