#!/usr/bin/env python3
"""Quick comparison: single-point vs optimized, batch Metal GPU."""
import sys; sys.path.insert(0, '.')
import numpy as np
import time

from mlxmolkit.rm1.pipeline import _smiles_to_3d
from mlxmolkit.rm1.gradient import nddo_optimize_batch
from mlxmolkit.rm1.scf import rm1_energy_batch

# Small set first
test_mols = [
    ('O', -57.8, 'Water'),
    ('C', -17.9, 'Methane'),
    ('CC', -20.0, 'Ethane'),
    ('C=C', 12.4, 'Ethylene'),
    ('N', -11.0, 'Ammonia'),
    ('CO', -48.1, 'Methanol'),
    ('C=O', -25.9, 'Formaldehyde'),
    ('CC=O', -39.7, 'Acetaldehyde'),
    ('CCO', -56.2, 'Ethanol'),
    ('c1ccccc1', 19.8, 'Benzene'),
    ('CC(=O)O', -103.3, 'Acetic acid'),
    ('OC=O', -90.5, 'Formic acid'),
]

# Generate 3D
mol_data = []
for smi, ref_hf, name in test_mols:
    r = _smiles_to_3d(smi)
    if r:
        mol_data.append((smi, ref_hf, name, r[0], r[1]))

N = len(mol_data)
methods = ['RM1', 'AM1', 'AM1_STAR', 'RM1_STAR']

print(f"{'='*90}")
print(f"  SP vs OPT comparison: {N} molecules × {len(methods)} methods (Metal GPU batch)")
print(f"{'='*90}")

for method in methods:
    mols = [(d[3], d[4]) for d in mol_data]

    # Single-point
    t0 = time.time()
    sp_results = rm1_energy_batch(mols, method=method, use_metal=True)
    t_sp = time.time() - t0

    # Optimized (15 steps of L-BFGS)
    mols_opt = [(d[3], d[4].copy()) for d in mol_data]
    t0 = time.time()
    opt_results = nddo_optimize_batch(mols_opt, max_iter=15, grad_tol=0.005,
                                      method=method, verbose=False)
    t_opt = time.time() - t0

    # Stats
    sp_errs, opt_errs = [], []
    print(f"\n--- {method} ---  (SP: {t_sp:.1f}s, OPT: {t_opt:.1f}s)")
    print(f"  {'Molecule':>15s} {'Exp':>7s} {'SP':>7s} {'err':>6s} {'OPT':>7s} {'err':>6s} {'opt_conv':>9s}")

    for i, (smi, ref, name, atoms, coords) in enumerate(mol_data):
        sp_hf = sp_results[i]['heat_of_formation_kcal'] if sp_results[i]['converged'] else None
        opt_hf = opt_results[i]['heat_of_formation_kcal'] if opt_results[i].get('converged', False) else None

        sp_str = f"{sp_hf:7.1f}" if sp_hf else "    NC"
        opt_str = f"{opt_hf:7.1f}" if opt_hf else "    NC"
        sp_err = f"{sp_hf-ref:+6.1f}" if sp_hf else "    --"
        opt_err = f"{opt_hf-ref:+6.1f}" if opt_hf else "    --"
        oc = "Y" if opt_results[i].get('opt_converged', False) else "N"

        if sp_hf: sp_errs.append(abs(sp_hf - ref))
        if opt_hf: opt_errs.append(abs(opt_hf - ref))

        print(f"  {name:>15s} {ref:>7.1f} {sp_str} {sp_err} {opt_str} {opt_err} {oc:>9s}")

    sp_mae = np.mean(sp_errs) if sp_errs else 0
    opt_mae = np.mean(opt_errs) if opt_errs else 0
    print(f"  {'MAE':>15s} {'':>7s} {'':>7s} {sp_mae:>6.1f} {'':>7s} {opt_mae:>6.1f} {'ΔMAE='+f'{opt_mae-sp_mae:+.1f}':>9s}")
