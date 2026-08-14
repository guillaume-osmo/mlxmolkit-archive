#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume — SPDX: MIT
"""SMILES/SDF -> GPU g-xTB -> RESP charges -> hydration free energy / Henry.

Pipeline (the deployable goal: fast near-DFT charges + solvation from SMILES):
  1. read molecules (.smi/.txt = one SMILES[ name] per line, or .sdf with 3D)
  2. 3D geometry (RDKit ETKDG+MMFF) if not already 3D
  3. g-xTB SCF (mlxmolkit, validated call) -> density, atomic charges
  4. CAMM distributed multipoles via the GPU batched path (batch_multipole +
     mmompop_fast) -> the g-xTB electrostatic model (monopole+dipole+quad)
  5. RESP: fit atom-centred point charges to the CAMM electrostatic potential
     on a Merz-Singh-Kollman grid (2-stage hyperbolic restraint, total-Q constraint)
  6. solvation: generalized-Born (Still) polar term from RESP charges + SASA
     non-polar term -> dG_hyd (kcal/mol) and log10 Kaw = dG_hyd/1.364

Outputs: <out>.csv (per-molecule energetics) and <out>.npz (per-atom RESP charges).

Usage:
  conda activate rdkit_build_fb
  cd /Users/tgg/Github/mlxmolkit
  python3 scripts/gxtb_resp_solvation.py mols.smi -o results
  python3 scripts/gxtb_resp_solvation.py mols.sdf -o results
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
from mlxmolkit.xtb.scf_gxtb import gxtb_energy
from mlxmolkit.xtb.gxtb_overlap_batched import prep_basis, batch_multipole
from mlxmolkit.xtb.gxtb_aes_assembly import mmompop_fast
sys.path.append("/Users/tgg/Github/mlxmolkit_phase1")          # append: real mlxmolkit wins
from int1e_esp_mlx import int1e_esp_mlx                        # GPU: density-folded ESP, ~300x
from scipy.spatial.distance import cdist


def _fib(n):
    k = np.arange(n); ph = np.pi * (3 - 5 ** .5) * k; z = 1 - 2 * (k + .5) / n; r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


def cosmo_sigma(res, Z, C_ang, eps=78.4, nb=61, smin=-0.03, smax=0.03, dens=3.0, scale=1.0):
    """Self-contained COSMO sigma-profile p(sigma) (e/Ang^2), GPU ESP (no pyscf).
    SAS surface -> solute potential via int1e_esp_mlx -> COSMO q=-f A^-1 V -> sigma=q/area."""
    b_ = res["basis"]; T = np.asarray(b_.T_cao_to_sao); Pcao = T.T @ np.asarray(res["density"]) @ T
    Zval = np.bincount(np.asarray(b_.shell_atom), weights=np.asarray(b_.shell_zref), minlength=len(Z))
    aos = [dict(center=np.asarray(bf.center, float),
                prims=[(float(bf.alphas[k]), float(bf.coeffs[k]), *bf.l_xyz) for k in range(len(bf.alphas))])
           for bf in b_.cao_basis]
    cb = C_ang * ANG2BOHR; pts = []; ar = []
    for R, z in zip(cb, Z):
        rb = VDW.get(int(z), 1.7) * scale * ANG2BOHR; npt = max(20, int(dens * 4 * np.pi * (VDW.get(int(z), 1.7) * scale) ** 2))
        ap = 4 * np.pi * rb * rb / npt
        for p in _fib(npt) * rb + R:
            if not any(np.linalg.norm(p - Rj) < VDW.get(int(zj), 1.7) * scale * ANG2BOHR - 1e-6 for Rj, zj in zip(cb, Z)):
                pts.append(p); ar.append(ap)
    pts = np.array(pts); ar = np.array(ar)
    V = np.array([(Zval / np.linalg.norm(cb - p, axis=1)).sum() for p in pts]) - int1e_esp_mlx(aos, Pcao, pts)
    D = cdist(pts, pts); A = np.where(D > 0, 1.0 / np.where(D > 0, D, 1), 0.0)
    np.fill_diagonal(A, 1.07 * np.sqrt(4 * np.pi / ar))
    q = -((eps - 1) / (eps + 0.5)) * np.linalg.solve(A, V)
    arA = ar * (1 / ANG2BOHR) ** 2; sig = q / arA
    e = np.linspace(smin, smax, nb + 1); prof, _ = np.histogram(sig, bins=e, weights=arA)
    return 0.5 * (e[:-1] + e[1:]), prof, float(arA.sum())


def _mk_grid_bohr(Z, C_ang, shells=(1.4, 1.6, 1.8, 2.0), dens=1.0):
    """Merz-Singh-Kollman points (bohr): shells x vdW, ~dens/Ang^2, outside vdW."""
    cb = C_ang * ANG2BOHR; pts = []
    for s in shells:
        for R, z in zip(cb, Z):
            rad = VDW.get(int(z), 1.7) * s; npt = max(8, int(dens * 4 * np.pi * rad * rad))
            k = np.arange(npt); ph = np.pi * (3 - 5 ** .5) * k
            zz = 1 - 2 * (k + .5) / npt; rr = np.sqrt(1 - zz * zz)
            sph = np.stack([rr * np.cos(ph), rr * np.sin(ph), zz], 1) * (rad * ANG2BOHR) + R
            for p in sph:
                if not any(np.linalg.norm(p - Rj) < VDW.get(int(zj), 1.7) * s * ANG2BOHR - 1e-6
                           for Rj, zj in zip(cb, Z)):
                    pts.append(p)
    return np.array(pts)


def resp_int1e(res, Z, C_ang, qtot=0.0, a=0.02, b=0.1, nit=10):
    """GPU RESP: true QM ESP (g-xTB density via int1e_grids_mlx) -> charges, with
    a Mulliken-anchored hyperbolic restraint (a=0.02) that stabilizes BURIED atoms.
    Valence nuclear charges (q-vSZP); CAO density via the CAO->SAO transform (d-shells)."""
    b_ = res["basis"]; T = np.asarray(b_.T_cao_to_sao)
    Pcao = T.T @ np.asarray(res["density"]) @ T
    Zval = np.bincount(np.asarray(b_.shell_atom), weights=np.asarray(b_.shell_zref), minlength=len(Z))
    aos = [dict(center=np.asarray(bf.center, float),
                prims=[(float(bf.alphas[k]), float(bf.coeffs[k]), *bf.l_xyz) for k in range(len(bf.alphas))])
           for bf in b_.cao_basis]
    cb = C_ang * ANG2BOHR; pts = _mk_grid_bohr(Z, C_ang)
    Ve = int1e_esp_mlx(aos, Pcao, pts)                                          # GPU, ~300x
    Vqm = np.array([(Zval / np.linalg.norm(cb - p, axis=1)).sum() for p in pts]) - Ve
    A = 1.0 / np.linalg.norm(pts[:, None, :] - cb[None, :, :], axis=2)
    n = len(Z); qref = np.asarray(res["atom_charges"]); AtA = A.T @ A; Atb = A.T @ Vqm; q = qref.copy()
    for _ in range(nit):
        R = a / np.sqrt((q - qref) ** 2 + b * b)
        M = np.zeros((n + 1, n + 1)); M[:n, :n] = AtA + np.diag(R); M[:n, n] = 1; M[n, :n] = 1
        q = np.linalg.solve(M, np.append(Atb + R * qref, qtot))[:n]
    return q


def camm_from_res_batch(reslist, C_list):
    """CAMM (dipm,qp) in SAO for a BATCH, all elements (d-shells via CAO->SAO T).
    Multipole integrals run on GPU (batch_multipole); transform+mmompop are numpy."""
    _, dps, qps = batch_multipole([prep_basis(r["basis"].cao_basis) for r in reslist])  # GPU
    out = []
    for r, C, dp, qp in zip(reslist, C_list, dps, qps):
        b = r["basis"]; T = np.asarray(b.T_cao_to_sao)
        dp = np.einsum("ip,kpq,jq->kij", T, dp, T)             # CAO -> SAO
        qp = np.einsum("ip,cpq,jq->cij", T, qp, T)
        aoat = np.asarray(b.shell_atom)[np.asarray(b.bf_to_shell)]
        out.append(mmompop_fast(np.asarray(r["density"]), np.asarray(r["S"]),
                                dp, qp, aoat, C * ANG2BOHR))
    return out


def camm_from_res(res, C_ang):
    return camm_from_res_batch([res], [C_ang])[0]

ANG2BOHR = 1.8897259886
KCAL = 627.5094740631      # Hartree -> kcal/mol (for GB; charges/ESP stay atomic)
# Bondi van der Waals radii (Angstrom)
VDW = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80,
       17: 1.75, 35: 1.85, 53: 1.98}
GXTB_KW = dict(use_mfx_exchange=True, use_aes=True, use_onecenter=True,
               use_acp_hamiltonian=True, use_fourth_order=True, use_d4srev=False,
               use_pacp=False, use_third_order=False, use_twobody_third_order=False,
               use_halide_increment_correction=False)


# ----------------------------- input -------------------------------------- #
def load_molecules(path: str):
    """Return list of (name, rdkit Mol with one 3D conformer, Z, coords_ang)."""
    out = []
    if path.lower().endswith(".sdf"):
        for i, m in enumerate(Chem.SDMolSupplier(path, removeHs=False)):
            if m is None:
                continue
            if m.GetNumConformers() == 0:
                m = embed(Chem.AddHs(m))
            elif any(a.GetAtomicNum() == 1 for a in m.GetAtoms()) is False:
                m = embed(Chem.AddHs(m, addCoords=True))
            out.append(_pack(m.GetProp("_Name") if m.HasProp("_Name") else f"mol{i}", m))
    else:
        for i, line in enumerate(open(path)):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi, *rest = line.split()
            m = Chem.MolFromSmiles(smi)
            if m is None:
                print(f"  skip (bad SMILES): {smi}"); continue
            m = embed(Chem.AddHs(m))
            out.append(_pack(rest[0] if rest else f"mol{i}", m, smi))
    return out


def embed(m):
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    try: AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    except Exception: AllChem.UFFOptimizeMolecule(m, maxIters=2000)
    return m


def _pack(name, m, smi=None):
    conf = m.GetConformer()
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    C = conf.GetPositions().astype(float)
    return dict(name=name, smiles=smi or Chem.MolToSmiles(Chem.RemoveHs(m)), mol=m, Z=Z, C=C)


# --------------------------- RESP ----------------------------------------- #
def mk_grid(Z, C):
    """Merz-Singh-Kollman shells (1.4-2.0x vdW), ~1 pt/Ang^2, outside vdW."""
    pts = []
    rv = np.array([VDW.get(int(z), 1.7) for z in Z])
    for scale in (1.4, 1.6, 1.8, 2.0):
        for a in range(len(Z)):
            r = scale * rv[a]
            n = max(8, int(4 * np.pi * r * r))
            g = _fib_sphere(n) * r + C[a]
            d2 = ((g[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (n,nat)
            keep = (d2 >= (scale * rv[None, :]) ** 2).all(1)       # outside every shell
            pts.append(g[keep])
    return np.concatenate(pts, 0)


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def camm_esp(grid_ang, C_ang, q, dipm, qp):
    """ESP (atomic units) at grid points from g-xTB distributed multipoles.

    q (nat) e; dipm (3,nat) e*bohr; qp (6,nat, mmompop xx,xy,yy,xz,yz,zz) traceless.
    """
    g = grid_ang * ANG2BOHR; Cb = C_ang * ANG2BOHR
    d = g[:, None, :] - Cb[None, :, :]            # (npt,nat,3) bohr
    r2 = (d * d).sum(-1); r = np.sqrt(r2); inv = 1.0 / r
    mono = q[None, :] * inv                                       # (npt,nat)
    dip = np.einsum("pak,ka->pa", d, dipm) * inv ** 3            # d.r/r^3 per (pt,atom)
    # quadrupole: 0.5 * d.Q.d / r^5  (traceless Q -> the -r^2*delta term drops)
    Q = np.zeros((qp.shape[1], 3, 3))
    Q[:, 0, 0] = qp[0]; Q[:, 1, 1] = qp[2]; Q[:, 2, 2] = qp[5]
    Q[:, 0, 1] = Q[:, 1, 0] = qp[1]; Q[:, 0, 2] = Q[:, 2, 0] = qp[3]; Q[:, 1, 2] = Q[:, 2, 1] = qp[4]
    quad = 0.5 * np.einsum("pak,akl,pal->pa", d, Q, d) * inv ** 5
    return (mono + dip + quad).sum(1)


def resp_fit(grid_ang, C_ang, V, Qtot, qref=None, a=1e-3, b=0.1, niter=6):
    """Restrained ESP fit -> atom point charges (e), sum = Qtot.

    Restraint anchors toward ``qref`` (the g-xTB Mulliken charges) rather than
    zero, which keeps BURIED atoms (e.g. methyl C in halomethanes) stable instead
    of blowing up to non-physical values.
    """
    A = 1.0 / (np.linalg.norm(grid_ang[:, None, :] - C_ang[None, :, :], axis=-1) * ANG2BOHR)
    AtA = A.T @ A; AtV = A.T @ V; nat = len(C_ang)
    qref = np.zeros(nat) if qref is None else np.asarray(qref, float)
    q = qref.copy()
    for _ in range(niter):
        R = a / np.sqrt((q - qref) ** 2 + b * b)            # hyperbolic, anchored to qref
        M = np.zeros((nat + 1, nat + 1)); M[:nat, :nat] = AtA + np.diag(R)
        M[:nat, nat] = 1.0; M[nat, :nat] = 1.0
        rhs = np.append(AtV + R * qref, Qtot)
        q = np.linalg.solve(M, rhs)[:nat]
    return q


# ------------------------- solvation -------------------------------------- #
def hydration(Z, C_ang, q, mol, eps=78.4, gamma=0.0072, beta=0.0):
    """Still generalized-Born polar + SASA non-polar -> dG_hyd (kcal/mol).

    NOTE fast estimate: gamma/eps are standard water defaults; for production swap
    in the FreeSolv/SMD-calibrated radii+gamma (see mlxmolkit-smd-solvation).
    """
    rv = np.array([VDW.get(int(z), 1.7) for z in Z])
    Rb = rv - 0.09                                           # crude effective Born radii (Ang)
    r = np.linalg.norm(C_ang[:, None, :] - C_ang[None, :, :], axis=-1)
    RR = np.outer(Rb, Rb)
    fgb = np.sqrt(r * r + RR * np.exp(-r * r / (4.0 * RR)))  # Still f_GB (Ang)
    Gpol = -166.0 * (1.0 - 1.0 / eps) * np.sum(np.outer(q, q) / fgb)   # kcal/mol
    # SASA (Ang^2) via RDKit FreeSASA
    radii = rdFreeSASA.classifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, radii)
    Gnp = gamma * sasa + beta
    dG = Gpol + Gnp
    return dict(dG_pol=Gpol, dG_np=Gnp, dG_hyd=dG, logKaw=dG / 1.364, sasa=sasa)


# ----------------------------- driver ------------------------------------- #
def run(path, out):
    mols = load_molecules(path)
    print(f"loaded {len(mols)} molecules")
    # g-xTB SCF (per mol) -> density/charges; collect bases for the batched multipole
    bases, ok = [], []
    for m in mols:
        try:
            res = gxtb_energy(m["Z"], m["C"], **GXTB_KW)
        except Exception as e:
            print(f"  g-xTB failed {m['name']}: {e}"); continue
        m["res"] = res; ok.append(m)
    if not ok:
        print("nothing to do"); return
    rows = []; per_atom = {}
    for m in ok:
        res = m["res"]; q = np.asarray(res["atom_charges"])
        qresp = resp_int1e(res, m["Z"], m["C"], qtot=float(round(q.sum())))  # GPU int1e QM-ESP RESP
        sg, sprof, sarea = cosmo_sigma(res, m["Z"], m["C"])                  # GPU COSMO sigma-profile
        solv = hydration(m["Z"], m["C"], qresp, m["mol"])
        per_atom[m["name"]] = dict(Z=m["Z"].tolist(), q_mulliken=q.tolist(), q_resp=qresp.tolist(),
                                   cosmo_sigma_grid=sg.tolist(), cosmo_sigma_profile=sprof.tolist(),
                                   cosmo_area=sarea)
        rows.append((m["name"], m["smiles"], len(m["Z"]),
                     solv["dG_pol"], solv["dG_np"], solv["dG_hyd"], solv["logKaw"]))
        print(f"  {m['name']:16s} dG_hyd={solv['dG_hyd']:8.2f} kcal/mol  logKaw={solv['logKaw']:6.2f}")
    with open(out + ".csv", "w") as f:
        f.write("name,smiles,natoms,dG_pol_kcal,dG_np_kcal,dG_hyd_kcal,logKaw\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    np.savez(out + ".npz", per_atom=json.dumps(per_atom))
    print(f"wrote {out}.csv  ({len(rows)} mols) and {out}.npz (per-atom RESP charges)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help=".smi/.txt (SMILES per line) or .sdf (3D)")
    ap.add_argument("-o", "--out", default="gxtb_resp_solvation")
    a = ap.parse_args()
    run(a.input, a.out)
