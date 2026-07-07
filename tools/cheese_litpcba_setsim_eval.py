#!/usr/bin/env python
"""Reproduce the CHEESE protocol on LIT-PCBA with the trained student, and add
set-to-set scoring (incl. Sinkhorn Optimal Transport) -- all in MLX/Metal.

Tasks #5 + #6 from docs/CHEESE_VS_ATTEMPTS_ASSESSMENT.md.

- #5 single-ligand: each active queries the deck; mean per-query EF1% (the paper
  reports LIT-PCBA shape median EF1% ~ 1.80).
- #6 multi-ligand (set-to-set): score every molecule by a set function over the
  active set; paper finds Optimal Transport best (shape median EF1 ~ 5.05),
  mean/centroid worst. We implement Table-4 scorers in MLX, OT via entropic
  Sinkhorn (point-to-set closed form == singleton Sinkhorn).

Embeddings come from the trained openCHEESE shape student (CheeseGraphTransformer,
MLX). One ETKDG conformer per molecule (CHEESE is conformer-agnostic at search).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from mlxmolkit.cheese_embedding import (
    CheeseGraphTransformer,
    CheeseEmbeddingConfig,
    cheese_embedding_batch_from_rdkit_mols,
)
from tools.conformer_source import embed_molecule_3d

DEFAULT_CKPT = ("outputs/cheese_projection/"
                "opencheese_all_moses_overlay_shape_nocharge_h128_emb128_l4_e3_steps192_"
                "muonv2_scale1_warm_principal/best_pair_mse.safetensors")


def read_smi(path, n, seed=0):
    smis = [ln.split()[0] for ln in open(path) if ln.split()]
    rng = np.random.default_rng(seed)
    if len(smis) > n:
        smis = [smis[i] for i in rng.permutation(len(smis))[:n]]
    return smis


def embed3d_ready(smi):
    # Shared CHEESE conformer source of truth (ETKDGv3 + MMFF), so eval geometries
    # match the teacher/charge caches. Fixed seed -> deterministic conformers.
    try:
        return embed_molecule_3d(smi, seed=0xC0FFEE, max_iters=200)
    except Exception:
        return None


def load_shape_student(ckpt, embedding_dim=128):
    cfg = CheeseEmbeddingConfig(hidden_dim=128, embedding_dim=embedding_dim, pooling="mean",
                                use_charges=False, use_chiral_features=True,
                                normalize_embeddings=True)
    student = CheeseGraphTransformer(cfg)
    ck = mx.load(str(ckpt))
    mk = dict(tree_flatten(student.parameters()))
    student.load_weights([(k, v) for k, v in ck.items() if k in mk and mk[k].shape == v.shape],
                         strict=False)
    return student


def encode(student, mols, batch=1000):
    out = []
    for s in range(0, len(mols), batch):
        b = cheese_embedding_batch_from_rdkit_mols(mols[s:s + batch], pad_to=None)
        e = student.encode_batch(b)
        mx.eval(e)
        out.append(np.asarray(e))
    return np.concatenate(out)


# ---- metrics ----
def ef(scores, labels, frac=0.01):
    k = max(1, int(len(scores) * frac))
    top = np.argsort(-scores)[:k]
    return float(labels[top].mean() / max(labels.mean(), 1e-9))


def bedroc(scores, labels, alpha=20.0):
    order = np.argsort(-scores); y = labels[order]
    N = len(y); n = int(y.sum())
    if n == 0 or n == N:
        return float("nan")
    Ra = n / N; ri = np.flatnonzero(y == 1) + 1
    rie = (np.sum(np.exp(-alpha * ri / N)) / n) / ((1.0 / N) * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1))
    return float(rie * (Ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra)))
                 + 1.0 / (1 - np.exp(alpha * (1 - Ra))))


# ---- set-to-set scorers (MLX) ----
def sinkhorn_ot_scores(S, P, *, eps=0.05, tau=0.1, iters=25):
    """True set-level entropic OT (Sinkhorn) in MLX, paper structure
    s_ot = sum_ij gamma_ij sim_active(i,j). Each candidate's attention over the
    actives a=softmax(sim/tau) is transported (Sinkhorn plan gamma) to uniform b
    over the active-active cost geometry C=1-sim(p_i,p_j); the score is the plan-
    weighted active-active similarity (coherence of the actives the candidate
    prefers). Distinct from the point-to-set ot_eps (which is singleton OT)."""
    Asim = P @ P.T                                   # (nA, nA) active-active cosine
    K = mx.exp(-(1.0 - Asim) / eps)                  # Gibbs kernel
    nA = P.shape[0]
    a = mx.softmax(mx.where(S > -1e8, S, -1e9) / tau, axis=1)   # (N, nA)
    b = mx.full((nA,), 1.0 / nA)
    U = mx.ones_like(a)
    V = mx.ones_like(a)
    for _ in range(iters):
        U = a / mx.maximum(V @ K.T, 1e-12)
        V = b[None, :] / mx.maximum(U @ K, 1e-12)
    M = K * Asim
    return np.asarray(mx.sum(U * (V @ M.T), axis=1))


def set_scores(E, labels, eps_list, knn_k=16):
    """Return dict name -> per-molecule score (np), scoring every molecule by the
    active set (leave-one-out for actives). E: (N,D) embeddings."""
    Emx = mx.array(E.astype(np.float32))
    Emx = Emx / mx.maximum(mx.linalg.norm(Emx, axis=1, keepdims=True), 1e-9)
    ai = np.flatnonzero(labels == 1)
    ni = np.flatnonzero(labels == 0)
    P = Emx[mx.array(ai)]            # (nA, D) active prototypes
    Ngt = Emx[mx.array(ni)]         # (nI, D) inactive prototypes
    S = Emx @ P.T                   # (N, nA) cosine sim to every active
    # leave-one-out: an active can't match itself
    self_col = -np.ones((E.shape[0],), dtype=np.int64)
    for c, q in enumerate(ai):
        self_col[q] = c
    sc = np.asarray(self_col)
    rows = mx.arange(E.shape[0])
    Snp = np.asarray(S)
    for r in range(E.shape[0]):
        if sc[r] >= 0:
            Snp[r, sc[r]] = -1e9
    S = mx.array(Snp)

    out = {}
    out["max"] = np.asarray(mx.max(S, axis=1))
    out["mean"] = np.asarray(mx.sum(mx.where(S > -1e8, S, 0.0), axis=1) /
                             mx.maximum(mx.sum((S > -1e8).astype(mx.float32), axis=1), 1.0))
    # batch k-NN: mean of top-k sims
    topk = mx.topk(S, min(knn_k, S.shape[1]), axis=1)
    out[f"knn{knn_k}"] = np.asarray(mx.mean(topk, axis=1))
    # centroid: sim to mean active embedding (renormalized)
    cen = mx.mean(P, axis=0, keepdims=True)
    cen = cen / mx.maximum(mx.linalg.norm(cen, axis=1, keepdims=True), 1e-9)
    out["centroid"] = np.asarray((Emx @ cen.T)[:, 0])
    # active-inactive contrast: max sim to actives - max sim to inactives
    SN = Emx @ Ngt.T
    SNnp = np.asarray(SN)
    ic = -np.ones((E.shape[0],), dtype=np.int64)
    for c, q in enumerate(ni):
        ic[q] = c
    for r in range(E.shape[0]):
        if ic[r] >= 0:
            SNnp[r, ic[r]] = -1e9
    out["contrast"] = out["max"] - np.asarray(np.max(SNnp, axis=1))
    # Optimal Transport (entropic Sinkhorn, singleton->set closed form): the
    # entropic-OT similarity = eps * logsumexp(sim/eps) over the active set.
    # eps->0 -> max (Maximum Similarity), eps->inf -> mean. Masked self (-1e9)
    # contributes exp(-huge)=0. Verified to reduce to max as eps->0.
    for eps in eps_list:
        m = mx.max(S, axis=1, keepdims=True)
        lse = m[:, 0] + eps * mx.log(mx.sum(mx.exp((S - m) / eps), axis=1))
        out[f"ot_eps{eps}"] = np.asarray(lse)
    # true set-level Sinkhorn OT (transport plan over active-active geometry)
    for se in (0.05, 0.1):
        out[f"ot_sink{se}"] = sinkhorn_ot_scores(S, P, eps=se, tau=0.1, iters=25)
    return out


def single_ligand_ef(E, labels, frac=0.01):
    """Mean over active queries of per-query EF (leave-one-out, cosine 1-NN)."""
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ai = np.flatnonzero(labels == 1)
    efs = []
    for q in ai:
        s = En @ En[q]
        s[q] = -np.inf
        lab = labels.copy().astype(float); lab[q] = np.nan  # exclude self from pool
        mask = ~np.isnan(lab)
        efs.append(ef(s[mask], labels[mask], frac))
    return float(np.mean(efs))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=["ALDH1", "VDR", "PKM2", "MAPK1"])
    p.add_argument("--litpcba-dir", type=Path, default=Path("data/litpcba"))
    p.add_argument("--ckpt", type=Path, default=Path(DEFAULT_CKPT))
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--n-actives", type=int, default=150)
    p.add_argument("--n-inactives", type=int, default=2000)
    p.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/cheese_projection/litpcba_emb"))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    import hashlib
    a = parse_args()
    student = load_shape_student(a.ckpt, a.embedding_dim)
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = hashlib.md5(str(a.ckpt).encode()).hexdigest()[:8]  # embeddings depend on the student
    per_target = {}
    for tgt in a.targets:
        cache = a.cache_dir / f"{tgt}_a{a.n_actives}_i{a.n_inactives}_{ckpt_tag}.npz"
        if cache.exists():
            z = np.load(cache); E = z["E"]; labels = z["labels"]
            print(f"[{tgt}] loaded {len(E)} embeddings from cache", flush=True)
        else:
            act = read_smi(a.litpcba_dir / tgt / "active_V.smi", a.n_actives)
            dec = read_smi(a.litpcba_dir / tgt / "inactive_V.smi", a.n_inactives)
            t0 = time.perf_counter()
            mols, labels = [], []
            for smi, lab in [(s, 1) for s in act] + [(s, 0) for s in dec]:
                m = embed3d_ready(smi)
                if m is not None:
                    mols.append(m); labels.append(lab)
            labels = np.array(labels)
            E = encode(student, mols)
            np.savez(cache, E=E.astype(np.float32), labels=labels)
            print(f"[{tgt}] embedded {labels.sum()} act / {(labels==0).sum()} inact "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        sl = single_ligand_ef(E, labels)
        scorers = set_scores(E, labels, a.eps)
        row = {"single_ligand_ef1": sl, "n_act": int(labels.sum()), "n_inact": int((labels == 0).sum())}
        for nm, sc in scorers.items():
            row[f"ml_ef1_{nm}"] = ef(sc, labels, 0.01)
            row[f"ml_bedroc_{nm}"] = bedroc(sc, labels)
        per_target[tgt] = row
        print(f"[{tgt}] single-ligand EF1%={sl:.2f} | multi-ligand EF1%: "
              + " ".join(f"{nm}={row[f'ml_ef1_{nm}']:.2f}" for nm in scorers), flush=True)

    # medians across targets (paper reports medians)
    names = [k for k in next(iter(per_target.values())) if k.startswith("ml_ef1_")]
    print(f"\n=== MEDIANS across {len(per_target)} targets (paper: shape single ~1.80, multi ~4.15, OT ~5.05) ===")
    print(f"  single_ligand_ef1   median = {np.median([r['single_ligand_ef1'] for r in per_target.values()]):.3f}")
    print(f"  {'set-to-set scorer':18s} {'median EF1%':>12s} {'median BEDROC':>14s}")
    summary = {"single_ligand_ef1_median": float(np.median([r['single_ligand_ef1'] for r in per_target.values()]))}
    for nm in names:
        base = nm[len("ml_ef1_"):]
        med_ef = float(np.median([r[nm] for r in per_target.values()]))
        med_be = float(np.median([r[f"ml_bedroc_{base}"] for r in per_target.values()]))
        summary[nm] = med_ef
        print(f"  {base:18s} {med_ef:12.3f} {med_be:14.3f}")

    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"per_target": per_target, "medians": summary}, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
