#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume — SPDX: MIT
"""Retrain deltaHvapv3 with the NEW g-xTB->RESP feature block; compare to the
old COSMO sigma-profile block on the SAME HCNOF subset (apples-to-apples).

Runs after scripts/build_deltahvap_resp_features.py finishes.
  conda activate rdkit_build_fb && cd /Users/tgg/Github/mlxmolkit
  python3 scripts/retrain_deltahvap_resp.py
"""
from __future__ import annotations
import sys, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from train_delta_hvap_v2_homoset import (compute_rdkit_features, attach_row_aligned_gxtb_sigma)
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict, KFold

CSV = Path("data/delta_hvap_v2/deltaHvapv3_full4491_organic_singlefrag_gxtbvalid_broad_train.csv")
SIG = Path("data/delta_hvap_v2/deltaHvapv3_full4491_organic_gxtb_tmcosmo_sigma.npz")
RESP = f"{ROOT}/scripts/deltahvap_resp_features.npz"
RD_CACHE = Path("data/delta_hvap_v2/_rdkit_cache_full4491.npz")
TRUSTED = {"autovap_trusted", "deltaHvapv3_experimental_new"}
XGB = dict(n_estimators=600, max_depth=4, learning_rate=0.035, subsample=0.9,
           colsample_bytree=0.8, random_state=47, n_jobs=4)


def metrics(p, y):
    return (float(np.mean(np.abs(p - y))),
            float(np.sqrt(np.mean((p - y) ** 2))),
            float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)))


def main():
    df = pd.read_csv(CSV)
    smiles = df["canonical_smiles"].astype(str).tolist()
    rdkit_x, _ = compute_rdkit_features(smiles, RD_CACHE)
    old_sig, _, _ = attach_row_aligned_gxtb_sigma(df, SIG, include_profile=True)   # 122
    z = np.load(RESP); resp_x = z["feats"]; valid = z["valid"]
    y = df["trusted_target_kJmol"].to_numpy(float)
    trusted = df["target_source"].astype(str).isin(TRUSTED).to_numpy()
    mask = trusted & valid & np.isfinite(y)
    print(f"eval rows (trusted & HCNOF-RESP-valid): {int(mask.sum())} / {len(df)} "
          f"(baseline used 3728 incl. halogens/S)")

    cv = KFold(5, shuffle=True, random_state=47)
    Y = y[mask]
    sets = {
        "rdkit only            ": rdkit_x[mask],
        "rdkit + OLD sigma(122)": np.hstack([rdkit_x, old_sig])[mask],
        "rdkit + NEW resp(37)  ": np.hstack([rdkit_x, resp_x])[mask],
        "rdkit + OLD + NEW     ": np.hstack([rdkit_x, old_sig, resp_x])[mask],
    }
    print(f"\n{'feature set':24s}  MAE    RMSE   R2   (kJ/mol, XGBoost 5-fold)")
    for name, X in sets.items():
        p = cross_val_predict(XGBRegressor(**XGB), np.nan_to_num(X, nan=0.0), Y, cv=cv)
        mae, rmse, r2 = metrics(p, Y)
        print(f"{name}  {mae:5.2f}  {rmse:5.2f}  {r2:.3f}")
    print("\n(baseline rdkit+OLD-sigma on full 3728: MAE 3.05 R2 0.970)")


if __name__ == "__main__":
    main()
