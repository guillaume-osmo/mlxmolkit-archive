#!/usr/bin/env python
"""Full multi-channel fusion exploration on LIT-PCBA (tasks #8/#9).

Channels: shape (learned), ESP (learned, charge-aware), ECFP4 (Morgan/Tanimoto),
optional Osmordred (descriptor npz). Each channel -> a candidate x active
similarity matrix S. Set-to-set scorers {max (Maximum-Similarity), OT (entropic
Sinkhorn)} reduce S to a per-molecule score. We explore fusion at the SCORE level
(EF1% is rank-based & NOT additive, so we fuse scores then re-rank):

  per-channel baselines (esp+OT tests the paper's 5.05)
  additive-equal: sum of z-scored channel scores
  additive leave-one-out: drop each channel -> marginal contribution
  MoE-lite: per-molecule hard gate (max z over channels)
  concat (learned channels): concat embeddings -> cosine
And the proper ADDITIVE metric -> complementarity: do channels recover DIFFERENT
actives in the top-1% (union vs overlap)?

All persistent paths; charge-aware ESP uses Gasteiger charges (as the paper does).
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import time
from itertools import combinations
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from mlxmolkit.cheese_embedding import (
    CheeseGraphTransformer, CheeseEmbeddingConfig, cheese_embedding_batch_from_rdkit_mols,
)
from mlxmolkit.charge_model import (
    MultiEndpointGeometricChargePredictor, ChargeModelConfig, charge_model_batch_from_rdkit_mols,
)
from tools.cheese_litpcba_setsim_eval import read_smi, embed3d_ready, ef, bedroc

_CHARGE_MODEL = None  # lazy singleton


def load_charge_model(ckpt: Path):
    """Learned q_resp charge model (MultiEndpoint, h384/l6/qeq) — trained on the
    same espaloma data the ESP teacher used; far better than Gasteiger."""
    cfg_json = json.loads((ckpt.parent / "config.json").read_text())
    cfg = cfg_json.get("config", cfg_json)
    endpoints = cfg_json.get("endpoints", ["q_esp", "q_resp"])
    model = MultiEndpointGeometricChargePredictor(ChargeModelConfig(**cfg), endpoints=endpoints)
    model.load_weights(str(ckpt))
    return model


def qresp_charges(mols, cm, batch=512):
    out = []
    for s in range(0, len(mols), batch):
        chunk = mols[s:s + batch]
        b = charge_model_batch_from_rdkit_mols(chunk, n_bond_states=cm.config.n_bond_states)
        q = cm(b.atomic_numbers, b.coords, b.bond_matrix, b.mask, b.total_charge, endpoint="q_resp")
        mx.eval(q); q = np.asarray(q, dtype=np.float32)
        nat = np.asarray(b.mask).sum(1).astype(int)
        out.extend([q[i, :int(n)] for i, n in enumerate(nat)])
    return out


_ESPALOMA = None
def espaloma_charges(mols):
    """The actual espaloma_charge model that generated the dataset q_resp (graph-based).
    Needs dgl<2.0 (no graphbolt on ARM) + torch.load weights_only=False patch."""
    global _ESPALOMA
    if _ESPALOMA is None:
        import torch
        _ol = torch.load
        torch.load = lambda *a, **k: _ol(*a, **{**k, "weights_only": False})
        from espaloma_charge import charge as ec
        _ESPALOMA = ec
    out = []
    for m in mols:
        try:
            out.append(np.asarray(_ESPALOMA(m), dtype=np.float32))
        except Exception:
            out.append(np.zeros(m.GetNumAtoms(), dtype=np.float32))
    return out


def load_student(ckpt, use_charges):
    cfg = CheeseEmbeddingConfig(hidden_dim=128, embedding_dim=128, pooling="mean",
                                use_charges=use_charges, use_chiral_features=True, normalize_embeddings=True)
    s = CheeseGraphTransformer(cfg); ck = mx.load(str(ckpt)); mk = dict(tree_flatten(s.parameters()))
    s.load_weights([(k, v) for k, v in ck.items() if k in mk and mk[k].shape == v.shape], strict=False)
    return s


def gasteiger(mol):
    AllChem.ComputeGasteigerCharges(mol)
    q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()], dtype=np.float32)
    return np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)


def encode(student, mols, use_charges, charge_fn=None):
    charges_all = charge_fn(mols) if (use_charges and charge_fn is not None) else None
    out = []
    for s in range(0, len(mols), 1000):
        chunk = mols[s:s + 1000]
        ch = charges_all[s:s + 1000] if charges_all is not None else None
        b = cheese_embedding_batch_from_rdkit_mols(chunk, ch, pad_to=None)
        e = student.encode_batch(b); mx.eval(e); out.append(np.asarray(e))
    return np.concatenate(out)


def cos_S(E, ai):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = (En @ En[ai].T).astype(np.float64)
    for c, q in enumerate(ai):
        S[q, c] = -1e9
    return S


def tanimoto_S(fps, ai):
    N = len(fps); S = np.empty((N, len(ai)))
    for c, q in enumerate(ai):
        S[:, c] = DataStructs.BulkTanimotoSimilarity(fps[q], fps)
    for c, q in enumerate(ai):
        S[q, c] = -1e9
    return S


def set_scores_from_S(S, eps):
    """max + entropic-OT (eps*logsumexp). S has masked self at -1e9."""
    real = S > -1e8
    smax = np.where(real, S, -1e9).max(1)
    m = smax[:, None]
    lse = smax + eps * np.log(np.sum(np.exp((np.where(real, S, -1e30) - m) / eps), axis=1))
    return {"max": smax, "ot": lse}


def _z(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-9)


def topk_actives(score, labels, frac=0.01):
    k = max(1, int(len(score) * frac)); top = np.argsort(-score)[:k]
    return set(np.flatnonzero(labels[top] == 1).tolist()) | set(t for t in top if labels[t] == 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=["ALDH1", "VDR", "PKM2", "MAPK1", "FEN1", "KAT2A", "GBA"])
    p.add_argument("--litpcba-dir", type=Path, default=Path("data/litpcba"))
    p.add_argument("--shape-ckpt", type=Path, default=Path("outputs/cheese_projection/student_contrastive_spread_h128_l4_cw08_e24/best_recall_at_5.safetensors"))
    p.add_argument("--esp-ckpt", type=Path, default=Path("outputs/cheese_projection/student_esp_charges_h128_l4_cw08_e40/best_recall_at_5.safetensors"))
    p.add_argument("--osmordred-dir", type=Path, default=None, help="dir with {target}_osmordred.npz (optional 4th channel)")
    p.add_argument("--charge-ckpt", type=Path,
                   default=Path("outputs/charge_models/cheese_allmoses_transformer41mb_multitask_qeq_linear_h384_l6_from_allmoses_e2_lr5e5_logmae_valid100k/best.safetensors"),
                   help="learned q_resp charge model for ESP inference (replaces Gasteiger)")
    p.add_argument("--charge-source", choices=["espaloma", "model"], default="model",
                   help="model = MLX espaloma-q_resp model (Metal-native, 0.004 MAE, no DGL); espaloma = DGL reference")
    p.add_argument("--n-actives", type=int, default=150)
    p.add_argument("--n-inactives", type=int, default=2000)
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/cheese_projection/litpcba_fusion_emb"))
    p.add_argument("--conformer-cache-dir", type=Path, default=Path("outputs/cheese_projection/litpcba_conformers"),
                   help="shared student-independent 3D conformer cache (build with build_litpcba_conformer_cache.py)")
    p.add_argument("--out", type=Path, default=Path("outputs/cheese_projection/cheese_fusion_setsim_eval.json"))
    return p.parse_args()


def main():
    a = parse_args()
    shape_student = load_student(a.shape_ckpt, use_charges=False)
    esp_student = load_student(a.esp_ckpt, use_charges=True)
    if a.charge_source == "espaloma":
        charge_fn = espaloma_charges
        print("ESP inference charges = espaloma_charge (the model that made the dataset q_resp)", flush=True)
    else:
        cm = load_charge_model(a.charge_ckpt)
        charge_fn = lambda mols: qresp_charges(mols, cm)
        print(f"ESP inference charges = MLX q_resp proxy ({a.charge_ckpt.parent.name})", flush=True)
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    per_target = {}
    for tgt in a.targets:
        cache = a.cache_dir / f"{tgt}_a{a.n_actives}_i{a.n_inactives}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            Esh, Eesp, labels, smis = z["Esh"], z["Eesp"], z["labels"], list(z["smis"])
            print(f"[{tgt}] loaded {len(Esh)} dual embeddings", flush=True)
        else:
            t0 = time.perf_counter()
            conf_cache = a.conformer_cache_dir / f"{tgt}_a{a.n_actives}_i{a.n_inactives}_conformers.pkl"
            if conf_cache.exists():
                # shared, student-independent 3D conformers (deterministic seed) -> skip regen
                with open(conf_cache, "rb") as f:
                    cc = pickle.load(f)
                mols, labels, smis = [], [], []
                for mb, lab, smi in zip(cc["molblocks"], cc["labels"], cc["smis"]):
                    m = Chem.MolFromMolBlock(mb, removeHs=False)
                    if m is not None:
                        mols.append(m); labels.append(lab); smis.append(smi)
                labels = np.array(labels)
                print(f"[{tgt}] loaded {len(mols)} cached conformers ({time.perf_counter()-t0:.0f}s)", flush=True)
            else:
                act = read_smi(a.litpcba_dir / tgt / "active_V.smi", a.n_actives)
                dec = read_smi(a.litpcba_dir / tgt / "inactive_V.smi", a.n_inactives)
                mols, labels, smis = [], [], []
                for smi, lab in [(s, 1) for s in act] + [(s, 0) for s in dec]:
                    m = embed3d_ready(smi)
                    if m is not None:
                        mols.append(m); labels.append(lab); smis.append(smi)
                labels = np.array(labels)
            Esh = encode(shape_student, mols, False); Eesp = encode(esp_student, mols, True, charge_fn)
            np.savez(cache, Esh=Esh.astype(np.float32), Eesp=Eesp.astype(np.float32),
                     labels=labels, smis=np.array(smis, dtype=object))
            print(f"[{tgt}] embedded {labels.sum()} act / {(labels==0).sum()} inact ({time.perf_counter()-t0:.0f}s)", flush=True)

        labels = np.asarray(labels); ai = np.flatnonzero(labels == 1)
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smis]
        chans = {"shape": cos_S(Esh, ai), "esp": cos_S(Eesp, ai), "ecfp": tanimoto_S(fps, ai)}
        if a.osmordred_dir is not None and (a.osmordred_dir / f"{tgt}_osmordred.npz").exists():
            D = np.load(a.osmordred_dir / f"{tgt}_osmordred.npz")["descriptors"].astype(np.float64)
            Dz = (D - D.mean(0)) / (D.std(0) + 1e-9)
            # Mahalanobis whitening (PCA-96): decorrelate the redundant 3588 descriptors
            Xc = Dz - Dz.mean(0); U, sng, Vt = np.linalg.svd(Xc, full_matrices=False)
            kk = min(96, len(sng)); W = (Xc @ Vt[:kk].T) / (sng[:kk] + 1e-9) * np.sqrt(len(Dz))
            Dn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
            So = (Dn @ Dn[ai].T)
            for c, q in enumerate(ai):
                So[q, c] = -1e9
            chans["osmordred"] = So

        names = list(chans)
        sc = {ch: set_scores_from_S(chans[ch], a.ot_eps) for ch in names}
        row = {"n_act": int(labels.sum()), "channels": names}
        # per-channel
        for ch in names:
            row[f"max__{ch}"] = ef(sc[ch]["max"], labels)
            row[f"ot__{ch}"] = ef(sc[ch]["ot"], labels)
        # fusion at score level, per scorer
        def rrf(score_dict, k=60):  # reciprocal rank fusion: rewards top-rank by ANY channel
            out = np.zeros(len(labels))
            for v in score_dict.values():
                ranks = np.argsort(np.argsort(-np.asarray(v)))  # 0 = best
                out += 1.0 / (k + ranks)
            return out
        for s in ("max", "ot"):
            Z = {ch: _z(sc[ch][s]) for ch in names}
            raw = {ch: sc[ch][s] for ch in names}
            row[f"{s}__add_all"] = ef(sum(Z.values()), labels)
            row[f"{s}__moe_max"] = ef(np.max(np.stack([Z[ch] for ch in names]), axis=0), labels)
            row[f"{s}__rrf_all"] = ef(rrf(raw), labels)                         # complementarity-exploiting
            for ch in names:  # leave-one-out -> marginal value of dropping ch
                rest = [Z[c] for c in names if c != ch]
                if rest:
                    row[f"{s}__drop_{ch}"] = ef(sum(rest), labels)
                rrest = {c: raw[c] for c in names if c != ch}
                if rrest:
                    row[f"{s}__rrf_drop_{ch}"] = ef(rrf(rrest), labels)
        # concat (learned channels only)
        En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        Scat = (np.concatenate([En(Esh), En(Eesp)], 1) @ np.concatenate([En(Esh), En(Eesp)], 1)[ai].T)
        for c, q in enumerate(ai):
            Scat[q, c] = -1e9
        scat = set_scores_from_S(Scat, a.ot_eps)
        row["max__concat_sh_esp"] = ef(scat["max"], labels)
        row["ot__concat_sh_esp"] = ef(scat["ot"], labels)
        # complementarity (additive coverage): top-1% recovered actives per channel (max scorer)
        rec = {ch: topk_actives(sc[ch]["max"], labels) for ch in names}
        union = set().union(*rec.values());
        row["complementarity"] = {"per_channel_recovered": {ch: len(rec[ch]) for ch in names},
                                  "union_recovered": len(union),
                                  "best_single": max(len(rec[ch]) for ch in names),
                                  "union_gain_over_best": len(union) - max(len(rec[ch]) for ch in names)}
        per_target[tgt] = row
        print(f"[{tgt}] max: " + " ".join(f"{ch}={row[f'max__{ch}']:.2f}" for ch in names)
              + f" | add_all={row['max__add_all']:.2f} moe={row['max__moe_max']:.2f}"
              + f" | recovered/channel={row['complementarity']['per_channel_recovered']} union={row['complementarity']['union_recovered']}", flush=True)

    keys = [k for k in next(iter(per_target.values())) if k.startswith(("max__", "ot__"))]
    med = {k: float(np.median([per_target[t][k] for t in per_target if k in per_target[t]])) for k in keys}
    a.out.write_text(json.dumps({"per_target": per_target, "medians": med}, indent=2, sort_keys=True) + "\n")
    print(f"\n=== MEDIAN EF1% across {len(per_target)} targets ===")
    for s in ("max", "ot"):
        chs = " ".join(f"{k[len(s)+2:]}={med[k]:.2f}" for k in keys if k.startswith(s + "__"))
        print(f"[{s}] {chs}")
    tot_union = sum(per_target[t]["complementarity"]["union_recovered"] for t in per_target)
    tot_best = sum(per_target[t]["complementarity"]["best_single"] for t in per_target)
    print(f"\nCOMPLEMENTARITY (top-1% actives recovered, summed): union={tot_union} vs best-single-channel={tot_best} "
          f"-> +{tot_union-tot_best} from combining channels")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
