from pathlib import Path
import sys

print("Python:", sys.version.split()[0])
print("CWD:", Path.cwd())

import numpy as np
import pandas as pd
import rdkit
import mlx
import mlx.core as mx
import mlxmolkit

print("core imports OK")
print("MLX sum:", float(mx.sum(mx.array([1.0, 2.0, 3.0]))))

paths = [
    "data/delta_hvap_v2",
    "data/orcacosmo",
    "benchmarks/biodegradation_protocol_annotation/best_protocol_same_duration_v2_consensus/best_same_molecule_duration_target.csv",
]
for p in paths:
    print(p, "OK" if Path(p).exists() else "MISSING")

try:
    from chemeleon_smd import fingerprint
    fp = fingerprint(["CCO"], batch_size=1)
    print("ChemeleonSMD fingerprint:", fp.shape)
except Exception as e:
    print("ChemeleonSMD FAIL:", repr(e))

try:
    from tabicl_mlx import TabICLRegressorMLX
    print("TabICL-MLX import OK")
except Exception as e:
    print("TabICL-MLX FAIL:", repr(e))

p = Path("benchmarks/biodegradation_protocol_annotation/best_protocol_same_duration_v2_consensus/best_same_molecule_duration_target.csv")
if p.exists():
    df = pd.read_csv(p)
    print("biodeg rows:", len(df), "mols:", df["canonical_smiles"].nunique())
    print("target mean:", round(float(df["upper_consensus_y_percent"].mean()), 3))

print("INSTALL TEST DONE")
