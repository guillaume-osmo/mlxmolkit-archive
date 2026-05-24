#!/usr/bin/env python3
"""Validate COSMO-RS activity coefficients against experimental VLE data."""
import sys; sys.path.insert(0, '.')
import numpy as np
import time

from mlxmolkit.cosmo import batch_smiles_to_cosmo, batch_activity_coefficients
from mlxmolkit.cosmo.cosmors import activity_coefficients

# ================================================================
# Step 1: Batch COSMO generation benchmark
# ================================================================
print("=" * 70)
print("  Batch COSMO-RS Benchmark")
print("=" * 70)

drug_smiles = ['O', 'CO', 'CCO', 'CC=O', 'CC(=O)O', 'c1ccccc1',
               'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC',
               'CN', 'CCN', 'C=O', 'N', 'CC(C)C',
               'c1ccc(O)cc1', 'CC(=O)C', 'CCCO', 'OC=O']

t0 = time.time()
cosmo_results = batch_smiles_to_cosmo(drug_smiles, method='PM3',
                                       use_metal=True, verbose=True)
t_batch = time.time() - t0

n_ok = sum(1 for r in cosmo_results if r is not None)
print(f"\n  {n_ok}/{len(drug_smiles)} molecules in {t_batch:.2f}s "
      f"({n_ok/t_batch:.0f} mol/s)")

# ================================================================
# Step 2: VLE validation for multiple systems
# ================================================================
print(f"\n{'='*70}")
print("  VLE Validation: Activity Coefficients at 298.15 K")
print(f"{'='*70}")

# Pre-compute COSMO for all unique molecules
all_smiles = list(set(['O', 'CCO', 'CC(=O)C', 'CCCCCC', 'c1ccccc1', 'CO', 'CC(=O)O']))
cosmo_db = {}
results = batch_smiles_to_cosmo(all_smiles, method='PM3', use_metal=True)
for smi, r in zip(all_smiles, results):
    if r is not None:
        cosmo_db[smi] = r

# Systems to validate
# (name, SMILES1, SMILES2, x1_values, exp_gamma1_at_x1, exp_gamma2_at_x1)
vle_systems = [
    ("Water-Ethanol", "O", "CCO",
     [0.1, 0.3, 0.5, 0.7, 0.9],
     [1.8, 1.4, 1.3, 1.1, 1.0],    # approx exp gamma_H2O
     [1.0, 1.1, 1.5, 2.0, 2.5]),   # approx exp gamma_EtOH

    ("Water-Acetone", "O", "CC(=O)C",
     [0.1, 0.3, 0.5, 0.7, 0.9],
     [3.5, 2.2, 1.6, 1.2, 1.0],
     [1.0, 1.0, 1.2, 2.0, 5.0]),

    ("Water-Methanol", "O", "CO",
     [0.1, 0.3, 0.5, 0.7, 0.9],
     [1.4, 1.2, 1.1, 1.05, 1.0],
     [1.0, 1.0, 1.1, 1.2, 1.5]),

    ("Hexane-Ethanol", "CCCCCC", "CCO",
     [0.1, 0.3, 0.5, 0.7, 0.9],
     [3.0, 2.0, 1.5, 1.2, 1.0],
     [1.0, 1.1, 1.5, 2.5, 6.0]),
]

for sys_name, smi1, smi2, x1_vals, exp_g1, exp_g2 in vle_systems:
    print(f"\n--- {sys_name} ({smi1} / {smi2}) ---")

    if smi1 not in cosmo_db or smi2 not in cosmo_db:
        print("  SKIPPED (COSMO failed)")
        continue

    cosmo1 = cosmo_db[smi1]
    cosmo2 = cosmo_db[smi2]

    print(f"  {'x1':>5s} {'g1_calc':>8s} {'g1_exp':>8s} {'g2_calc':>8s} {'g2_exp':>8s}")

    errs1, errs2 = [], []
    for k, x1 in enumerate(x1_vals):
        x = np.array([x1, 1.0 - x1])
        lng = activity_coefficients([cosmo1, cosmo2], x, T=298.15)
        g = np.exp(lng)

        e1 = abs(g[0] - exp_g1[k])
        e2 = abs(g[1] - exp_g2[k])
        errs1.append(e1)
        errs2.append(e2)

        print(f"  {x1:>5.1f} {g[0]:>8.3f} {exp_g1[k]:>8.1f} {g[1]:>8.3f} {exp_g2[k]:>8.1f}")

    mae1 = np.mean(errs1)
    mae2 = np.mean(errs2)
    print(f"  MAE:  {mae1:>8.3f} {'':>8s} {mae2:>8.3f}")
