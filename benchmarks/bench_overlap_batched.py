"""Validate + benchmark the GPU-batched overlap vs the numpy primitive loop.

Run: python3 benchmarks/bench_overlap_batched.py
"""
import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlxmolkit.xtb.gxtb_basis import build_gxtb_qvszp_basis
from mlxmolkit.xtb.basis import overlap_matrix
from mlxmolkit.xtb.gxtb_overlap_batched import prep_basis, batch_overlap, overlap_matrix_mlx

SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
SET = os.environ.get("GXTB_SET", "/tmp/gxtb_exact/broadset")

def rx(p):
    L = open(p).read().splitlines(); n = int(L[0].split()[0]); Z = []; C = []
    for ln in L[2:2 + n]:
        q = ln.split(); Z.append(SYM2Z[q[0]]); C.append([float(x) for x in q[1:4]])
    return np.array(Z), np.array(C)

names = list(json.load(open(f"{SET}/ref_charges.json")))
bases = [build_gxtb_qvszp_basis(*rx(f"{SET}/mols/{nm}.xyz"), total_charge=0.0).cao_basis for nm in names]

maxerr = max(float(np.max(np.abs(overlap_matrix(b) - overlap_matrix_mlx(b)))) for b in bases)
print(f"max|S_mlx - S_numpy| over {len(bases)} mols = {maxerr:.2e}  (f32 GPU precision)")

t = time.time()
for b in bases: overlap_matrix(b)
t_np = time.time() - t
preps = [prep_basis(b) for b in bases]
batch_overlap(preps[:2])               # warm
t = time.time(); batch_overlap(preps); t_b = time.time() - t
print(f"numpy sequential : {t_np*1000:7.1f} ms ({len(bases)} mols)")
print(f"MLX batched      : {t_b*1000:7.1f} ms  -> {t_np/t_b:.1f}x")
