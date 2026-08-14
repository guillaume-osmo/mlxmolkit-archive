#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume — SPDX: MIT
"""Redo the hydration model on the NEW g-xTB->RESP->GB pipeline (FreeSolv).

Delta-learning: target = experimental dG_hyd; baseline = the pipeline's GB
estimate; a small model learns the residual from g-xTB/RESP physics features.
This calibrates the uncalibrated GB step. H/C/N/O/F only (CAMM s/p path).

Usage:
  conda activate rdkit_build_fb && cd /Users/tgg/Github/mlxmolkit
  python3 scripts/train_freesolv_gxtb.py [--max N]
"""
from __future__ import annotations
import sys, os, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from rdkit import Chem
from rdkit.Chem import Descriptors
from mlxmolkit.xtb.scf_gxtb import gxtb_energy
from mlxmolkit.xtb.gxtb_overlap_batched import prep_basis, batch_multipole
from mlxmolkit.xtb.gxtb_aes_assembly import mmompop_fast
from gxtb_resp_solvation import (embed, GXTB_KW, mk_grid, camm_esp, resp_fit,
                                 hydration, ANG2BOHR)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

ALLOWED = {1, 6, 7, 8, 9}
DB = "/Users/tgg/Github/FreeSolv/database.txt"


def load_freesolv(max_n=None):
    rows = []
    for ln in open(DB):
        if ln.startswith("#") or ";" not in ln:
            continue
        f = [x.strip() for x in ln.split(";")]
        cid, smi, _iupac, exp, _u, gaff = f[0], f[1], f[2], float(f[3]), f[4], float(f[5])
        m = Chem.MolFromSmiles(smi)
        if m is None or not set(a.GetAtomicNum() for a in m.GetAtoms()) <= ALLOWED:
            continue
        rows.append((cid, smi, exp, gaff))
    if max_n:
        rows = rows[:max_n]
    return rows


def featurize(smi):
    m = embed(Chem.AddHs(Chem.MolFromSmiles(smi)))
    conf = m.GetConformer()
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    C = conf.GetPositions().astype(float)
    res = gxtb_energy(Z, C, **GXTB_KW)
    if max(b.l_total for b in res["basis"].cao_basis) > 1:
        raise ValueError("d-shell")
    cb = res["basis"].cao_basis
    aoat = np.array([x.atom_idx for x in cb])
    P = np.asarray(res["density"]); S = np.asarray(res["S"]); q = np.asarray(res["atom_charges"])
    _, dps, qps = batch_multipole([prep_basis(cb)])
    dipm, qpm = mmompop_fast(P, S, dps[0], qps[0], aoat, C * ANG2BOHR)
    grid = mk_grid(Z, C)
    qr = resp_fit(grid, C, camm_esp(grid, C, q, dipm, qpm), float(round(q.sum())))
    solv = hydration(Z, C, qr, m)
    dipole = np.linalg.norm((qr[:, None] * (C * ANG2BOHR)).sum(0))   # RESP dipole (a.u.)
    return [solv["dG_pol"], solv["dG_np"], solv["dG_hyd"], solv["sasa"], dipole,
            float(np.abs(qr).sum()), float(qr.max()), float(qr.min()),
            int((Z == 8).sum()), int((Z == 7).sum()), int((Z == 9).sum()),
            Descriptors.MolWt(m), len(Z)], solv["dG_hyd"]


def main(max_n):
    rows = load_freesolv(max_n)
    print(f"FreeSolv H/C/N/O/F: {len(rows)} molecules")
    X, y, gb, gaff = [], [], [], []
    for i, (cid, smi, exp, gf) in enumerate(rows):
        try:
            feats, dG_gb = featurize(smi)
        except Exception as e:
            continue
        X.append(feats); y.append(exp); gb.append(dG_gb); gaff.append(gf)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}")
    X = np.array(X); y = np.array(y); gb = np.array(gb); gaff = np.array(gaff)
    print(f"featurized {len(y)} molecules")

    def mae(a, b): return float(np.mean(np.abs(a - b)))
    def r2(a, b): return float(1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2))
    # delta-learning: model the residual exp - gb
    cv = KFold(5, shuffle=True, random_state=0)
    mdl = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, random_state=0)
    pred_resid = cross_val_predict(mdl, X, y - gb, cv=cv)
    pred = gb + pred_resid
    print("\n--- hydration dG (kcal/mol), vs experiment ---")
    print(f"  raw GB pipeline   : MAE {mae(gb, y):.2f}  R2 {r2(gb, y):.2f}")
    print(f"  GAFF (FreeSolv)   : MAE {mae(gaff, y):.2f}  R2 {r2(gaff, y):.2f}")
    print(f"  delta-learned (CV): MAE {mae(pred, y):.2f}  R2 {r2(pred, y):.2f}  (5-fold)")
    mdl.fit(X, y - gb)
    import joblib; joblib.dump(mdl, f"{HERE}/freesolv_gxtb_delta.joblib")
    print(f"saved model -> scripts/freesolv_gxtb_delta.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=None)
    main(ap.parse_args().max)
