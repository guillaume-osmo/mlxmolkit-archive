"""Parse dftd4's references.json + xtb's param_ref.fh fragments to
vendor pure-MLX D4 reference data.

Outputs ``mlxmolkit/xtb/params/d4_data.npz`` with:
    refn[Z]               : number of references per element  (118 ints)
    refcn[Z, ref]         : reference CN values               (118, 7)
    refq_gfn2[Z, ref]     : reference charges (GFN2-xTB scheme) (118, 7)
    alphaiw[Z, ref, w]    : polarizability at imaginary freqs (118, 7, 23)
    omega_w[w]            : Gauss-Legendre frequency weights  (23,)
    r4r2[Z]               : sqrt(0.5·r4r2·sqrt(Z))             (118,)

The polarizability tables come straight from dftd4's references.json
(MIT-licensed, ships with the dftd4 Python package).

Reference CN/charge tables come from xtb's param_ref.fh — those entries
are not in references.json (which only has the molecular-system polariz-
abilities). Parsing those Fortran data statements is regex-driven.
"""
import json
import re
import os
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# 1. Per-reference polarizabilities from dftd4's JSON.
# ---------------------------------------------------------------------------
DFTD4_JSON = (
    "/Users/guillaume-osmo/miniconda3/envs/osmo/lib/python3.11/"
    "site-packages/dftd4/references.json"
)
with open(DFTD4_JSON) as f:
    db = json.load(f)

# Group by primary_reference (element index). Each element has
# multiple internal references (1..refn(Z)).
by_z: dict[int, list[tuple[int, str, list[float]]]] = defaultdict(list)
for k, e in db.items():
    # references.json includes metadata entries ("ids" → list, "nextid" → int)
    # alongside the actual reference systems. Skip non-dict values.
    if not isinstance(e, dict) or "key_value_pairs" not in e:
        continue
    z = int(e["key_value_pairs"]["primary_reference"])
    intr = int(e["key_value_pairs"]["internal_reference"])
    nm = str(e["key_value_pairs"]["name"])
    alpha = list(e["data"]["dynamic_polarizabilities"])
    by_z[z].append((intr, nm, alpha))

MAX_ELEM = 118
MAX_REF = 7   # dftd4's parameter
MAX_W = 23    # 23-point Casimir-Polder quadrature

refn = np.zeros(MAX_ELEM, dtype=np.int32)
alphaiw = np.zeros((MAX_ELEM, MAX_REF, MAX_W), dtype=np.float64)

for z in range(1, MAX_ELEM + 1):
    refs = sorted(by_z.get(z, []))
    refn[z - 1] = len(refs)
    for ref_index, (intr, nm, alpha) in enumerate(refs):
        if ref_index >= MAX_REF:
            break
        if len(alpha) != MAX_W:
            raise ValueError(
                f"Z={z} ref={intr} ({nm}): expected {MAX_W} alpha values, got {len(alpha)}"
            )
        alphaiw[z - 1, ref_index, :] = np.asarray(alpha, dtype=np.float64)

print(f"Parsed dftd4 references.json: total {sum(int(n) for n in refn)} reference systems")
print(f"  H:  refn = {refn[0]}")
print(f"  C:  refn = {refn[5]}")
print(f"  N:  refn = {refn[6]}")
print(f"  O:  refn = {refn[7]}")
print(f"  Cl: refn = {refn[16]}")
print(f"  Br: refn = {refn[34]}")


# ---------------------------------------------------------------------------
# 2. Reference CN and ref charges from xtb's param_ref.fh.
# ---------------------------------------------------------------------------
PARAM_REF = "/private/tmp/xtb-src/include/param_ref.fh"
with open(PARAM_REF) as f:
    src = f.read()

refcn = np.zeros((MAX_ELEM, MAX_REF), dtype=np.float64)
refq_gfn2 = np.zeros((MAX_ELEM, MAX_REF), dtype=np.float64)

# Per-element 2D scalar tables: refcn / refq / hcount / ascale (float)
# and refsys (int).
def _parse_2d_scalar(name, dtype):
    arr = np.zeros((MAX_ELEM, MAX_REF), dtype=dtype)
    pat = re.compile(
        rf"data\s+{name}\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*([-\d\.eE+]+)(?:_wp)?\s*/",
    )
    for m in pat.finditer(src):
        i = int(m.group(1))
        z = int(m.group(2))
        v = m.group(3)
        if 1 <= z <= MAX_ELEM and 1 <= i <= MAX_REF:
            arr[z - 1, i - 1] = dtype(float(v))
    return arr


refcn = _parse_2d_scalar("refcn", np.float64)
refq_gfn2 = _parse_2d_scalar("refq", np.float64)
refh = _parse_2d_scalar("refh", np.float64)        # tmp_hq under default refq mode
hcount = _parse_2d_scalar("hcount", np.float64)
ascale = _parse_2d_scalar("ascale", np.float64)
refsys = _parse_2d_scalar("refsys", np.int32)

print(f"Parsed param_ref.fh: refcn nonzero for "
      f"{int(np.count_nonzero(np.any(refcn != 0, axis=1)))} elements")

# Secondary references (17 of them — small molecule pivots used for the
# alpha subtraction). Match `data secq (i) / val_wp /` and similar.
N_SEC = 17
secq = np.zeros(N_SEC, dtype=np.float64)
sscale = np.zeros(N_SEC, dtype=np.float64)
secaiw = np.zeros((N_SEC, MAX_W), dtype=np.float64)

PAT_SEC_SCALAR = re.compile(
    r"data\s+(secq|sscale|seccn|seccnD3)\s*\(\s*(\d+)\s*\)\s*/\s*([-\d\.eE+]+)_wp\s*/",
)
for m in PAT_SEC_SCALAR.finditer(src):
    name = m.group(1)
    i = int(m.group(2))
    v = float(m.group(3))
    if 1 <= i <= N_SEC:
        if name == "secq":
            secq[i - 1] = v
        elif name == "sscale":
            sscale[i - 1] = v

# secaiw (:, i) literal — multi-line. Walk through and grab the 23
# floats following each `data secaiw (:, i) /`.
PAT_SECAIW = re.compile(
    r"data\s+secaiw\s*\(\s*:\s*,\s*(\d+)\s*\)\s*/(.*?)/",
    re.DOTALL,
)
for m in PAT_SECAIW.finditer(src):
    i = int(m.group(1))
    body = m.group(2)
    floats = re.findall(r"-?\d+\.\d+(?:[eE][+\-]?\d+)?", body)
    if 1 <= i <= N_SEC and len(floats) >= MAX_W:
        secaiw[i - 1, :MAX_W] = np.asarray([float(x) for x in floats[:MAX_W]])

print(f"Parsed secondary refs: secq nonzero for "
      f"{int(np.count_nonzero(secq))} entries; secaiw filled for "
      f"{int(np.count_nonzero(np.any(secaiw != 0, axis=1)))} entries")


# ---------------------------------------------------------------------------
# 3. r4r2 and Gauss-Legendre weights (from xtb's mctc r4r2 + dftd4 freq).
# ---------------------------------------------------------------------------
# r4r2 — same table as our D3 vendor.
import sys
sys.path.insert(0, "/Users/guillaume-osmo/Github/mlxmolkit")
from mlxmolkit.xtb.dispersion_d3 import _R4_OVER_R2, _R4R2
# _R4R2 is sqrt(0.5 * r4_over_r2 * sqrt(Z)) for Z = 1..118
r4r2 = _R4R2.copy()

# Frequency weights: D4 uses 23 imaginary frequencies sampled at the
# Gauss-Chebyshev nodes mapped via tan(πx/2). The weights for the
# Casimir-Polder integral are encoded in xtb's dispersion routines.
# Extract them from dftd4's Fortran sources:
DFTD4_DISP = "/private/tmp/xtb-src/src/disp/dftd4.F90"
with open(DFTD4_DISP) as f:
    disp_src = f.read()

# Find the "freq" or "weights" array literal. dftd4 uses
#   freq(23) = [...]  and  weights are tan(πk/2) derived
PAT_FREQ_BLOCK = re.compile(
    r"real\(wp\),\s+parameter\s*::\s*freq\s*\(\s*\d+\s*\)\s*=\s*\[\s*(.*?)\s*\]",
    re.DOTALL,
)
m = PAT_FREQ_BLOCK.search(disp_src)
if m:
    freq_text = m.group(1)
    floats = re.findall(r"-?\d+\.\d+(?:[eE][+\-]?\d+)?", freq_text)
    omega_w = np.asarray([float(x) for x in floats[:23]], dtype=np.float64)
    if len(omega_w) != 23:
        omega_w = None
else:
    omega_w = None

# xtb's D4 uses 23 specific frequencies (not Gauss-Chebyshev) with
# trapezoidal weights derived from the spacing — see
# xtb/src/disp/dftd4.F90:386-419.
freqs = np.array([
    1e-6, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
    1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00, 4.00, 5.00, 7.50, 10.00,
], dtype=np.float64)
assert len(freqs) == 23
# trapezoidal weights: w[1] = ½(f[2]-f[1]), w[i] = ½((f[i]-f[i-1]) + (f[i+1]-f[i])),
# w[23] = ½(f[23]-f[22]).
weights = np.zeros(23, dtype=np.float64)
weights[0] = 0.5 * (freqs[1] - freqs[0])
for i in range(1, 22):
    weights[i] = 0.5 * ((freqs[i] - freqs[i - 1]) + (freqs[i + 1] - freqs[i]))
weights[22] = 0.5 * (freqs[22] - freqs[21])
omega_w = freqs       # imaginary-frequency grid points (au)
omega_weights = weights   # quadrature weights for ∫ dω
print(f"  freq grid (vendored): {omega_w[:3]} ... {omega_w[-3:]}")
print(f"  weights:              {omega_weights[:3]} ... {omega_weights[-3:]}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/Users/guillaume-osmo/Github/mlxmolkit/mlxmolkit/xtb/params/d4_data.npz"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez(
    out_path,
    refn=refn,
    refcn=refcn,
    refq_gfn2=refq_gfn2,
    refh=refh,
    alphaiw=alphaiw,
    hcount=hcount,
    ascale=ascale,
    refsys=refsys,
    secq=secq,
    sscale=sscale,
    secaiw=secaiw,
    omega_w=omega_w,
    omega_weights=omega_weights,
    r4r2=r4r2,
)
print(f"\nWrote {out_path}")
print(f"  refn shape:        {refn.shape}")
print(f"  refcn shape:       {refcn.shape}")
print(f"  refq_gfn2 shape:   {refq_gfn2.shape}")
print(f"  alphaiw shape:     {alphaiw.shape}")
print(f"  omega_w shape:     {omega_w.shape}")
print(f"  r4r2 shape:        {r4r2.shape}")
