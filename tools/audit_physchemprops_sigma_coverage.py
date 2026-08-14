#!/usr/bin/env python3
"""Audit sigma-profile coverage for calcphyschemprops cascade targets.

The audit uses the trained calcphyschemprops target prediction CSVs as the
source of molecules/labels, then crosses their canonical SMILES against:

1. the 53k CHAOS 25A sigma-potential matrix,
2. locally generated sigma profiles already present in this repo,
3. completed/projected ORCA/COSMORS high-precision profiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
OSMO_ROOT = Path("/Users/guillaume-osmo/Github/osmo")
MODEL_DIR = OSMO_ROOT / "src/runway/physchemprops/models"
OUT_DIR = REPO_ROOT / "benchmarks/physchemprops_sigma_coverage"
COMMON_VOLATILE_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}

CASCADE_ORDER = [
    "V",
    "Polarizability",
    "L",
    "E",
    "B",
    "Density",
    "RI",
    "S",
    "A",
    "Modularity",
    "HansenTotal",
    "dD",
    "dH",
    "dP",
    "BP",
    "deltaHvap",
    "logVP",
    "logPow",
    "logWS",
    "deltaHf",
    "deltaHc",
    "MP",
    "Flashpoint",
    "logHenrycc",
    "DipoleMoment",
    "logViscosity",
    "logODT",
]


@dataclass(frozen=True)
class CanonSet:
    isomeric: set[str]
    no_stereo: set[str]

    @classmethod
    def from_smiles(cls, smiles: Iterable[str]) -> "CanonSet":
        iso: set[str] = set()
        nostereo: set[str] = set()
        for smi in smiles:
            c_iso, c_no = canonicalize(smi)
            if c_iso:
                iso.add(c_iso)
            if c_no:
                nostereo.add(c_no)
        return cls(iso, nostereo)

    def contains_iso(self, c_iso: str | None) -> bool:
        return bool(c_iso and c_iso in self.isomeric)

    def contains_loose(self, c_iso: str | None, c_no: str | None) -> bool:
        return bool((c_iso and c_iso in self.isomeric) or (c_no and c_no in self.no_stereo))

    def union(self, *others: "CanonSet") -> "CanonSet":
        iso = set(self.isomeric)
        no = set(self.no_stereo)
        for other in others:
            iso.update(other.isomeric)
            no.update(other.no_stereo)
        return CanonSet(iso, no)


def canonicalize(smiles: object) -> tuple[str | None, str | None]:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None, None
    smi = str(smiles).strip()
    if not smi:
        return None, None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
    )


def element_symbols(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def summarize_missing_elements(smiles: Iterable[str]) -> tuple[str, int]:
    element_counts: dict[str, int] = {}
    unusual_mol_count = 0
    for smi in smiles:
        elems = element_symbols(smi)
        if elems - COMMON_VOLATILE_ELEMENTS:
            unusual_mol_count += 1
        for elem in elems:
            element_counts[elem] = element_counts.get(elem, 0) + 1
    top = sorted(element_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
    return ";".join(f"{elem}:{count}" for elem, count in top), unusual_mol_count


def load_npz_smiles(path: Path, key: str = "canonical_smiles", valid_key: str | None = None) -> list[str]:
    if not path.exists():
        return []
    z = np.load(path, allow_pickle=True)
    if key not in z.files:
        return []
    smiles = np.asarray(z[key]).astype(object)
    if valid_key and valid_key in z.files:
        mask = np.asarray(z[valid_key]).astype(bool)
        smiles = smiles[mask]
    return [str(x) for x in smiles if str(x).strip()]


def load_orca_done_set() -> CanonSet:
    smiles: list[str] = []

    calc_orca = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_orca_cosmors_calc_only_sigma.npz"
    if calc_orca.exists():
        z = np.load(calc_orca, allow_pickle=True)
        valid = np.asarray(z["valid_mask"]).astype(bool)
        selected = np.asarray(z["selected_mask"]).astype(bool)
        smiles.extend(np.asarray(z["canonical_smiles"]).astype(object)[valid & selected].tolist())

    v3_cache = REPO_ROOT / "data/delta_hvap_v2/orca_cosmors_v3_new1347_molcache"
    queue = REPO_ROOT / "benchmarks/delta_hvap_v3_conflict_web_review/deltaHvapv3_new1347_sigma_queue.csv"
    if v3_cache.exists() and queue.exists():
        done_inchikey = {p.parent.name for p in v3_cache.glob("*/*/*.npz")}
        q = pd.read_csv(queue, low_memory=False)
        if "autovap_inchikey" in q.columns:
            m = q["autovap_inchikey"].astype(str).isin(done_inchikey)
            smiles.extend(q.loc[m, "canonical_smiles"].dropna().astype(str).tolist())

    return CanonSet.from_smiles(smiles)


def load_orca_projected_set() -> CanonSet:
    done = load_orca_done_set()
    queue = REPO_ROOT / "benchmarks/delta_hvap_v3_conflict_web_review/deltaHvapv3_new1347_sigma_queue.csv"
    if not queue.exists():
        return done
    q = pd.read_csv(queue, low_memory=False)
    queued = CanonSet.from_smiles(q["canonical_smiles"].dropna().astype(str).tolist())
    return done.union(queued)


def load_generated_sigma_set() -> CanonSet:
    smiles: list[str] = []
    smiles.extend(load_npz_smiles(REPO_ROOT / "data/autovap/autovap_sigma_filled.npz", key="smiles", valid_key="valid_mask"))
    smiles.extend(
        load_npz_smiles(
            REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_gxtb_tmcosmo_all_sigma.npz",
            key="canonical_smiles",
            valid_key="valid_mask",
        )
    )
    smiles.extend(
        load_npz_smiles(
            REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_orca_cosmors_calc_only_sigma.npz",
            key="canonical_smiles",
            valid_key="valid_mask",
        )
    )
    return CanonSet.from_smiles(smiles)


def audit_target(target: str, sets: dict[str, CanonSet]) -> tuple[dict[str, object], pd.DataFrame]:
    path = MODEL_DIR / target / f"{target}_predictions.csv"
    if not path.exists():
        return {"target": target, "missing_predictions_csv": True}, pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    y_col = "y_true" if "y_true" in df.columns else None
    if y_col is None:
        raise ValueError(f"{path} has no y_true column")
    df = df[df[y_col].notna()].copy()
    df[["canonical_isomeric", "canonical_no_stereo"]] = df["smiles"].apply(lambda x: pd.Series(canonicalize(x)))
    invalid_smiles_rows = int(df["canonical_isomeric"].isna().sum())
    df = df[df["canonical_isomeric"].notna()].copy()
    dedup = df.drop_duplicates("canonical_isomeric").copy()

    for name, cset in sets.items():
        dedup[f"{name}_iso"] = [cset.contains_iso(c) for c in dedup["canonical_isomeric"]]
        dedup[f"{name}_loose"] = [
            cset.contains_loose(c_iso, c_no)
            for c_iso, c_no in zip(dedup["canonical_isomeric"], dedup["canonical_no_stereo"], strict=False)
        ]

    def count(col: str) -> int:
        return int(dedup[col].sum())

    n = len(dedup)
    row: dict[str, object] = {
        "target": target,
        "n_rows": int(len(df)),
        "n_unique_canonical": int(n),
        "invalid_smiles_rows": invalid_smiles_rows,
    }
    for name in sets:
        for mode in ["iso", "loose"]:
            c = count(f"{name}_{mode}")
            row[f"{name}_{mode}_n"] = c
            row[f"{name}_{mode}_frac"] = c / n if n else np.nan

    missing_done = dedup[~dedup["sigma_any_done_loose"]]
    missing_projected = dedup[~dedup["sigma_any_projected_loose"]]
    top_done, unusual_done = summarize_missing_elements(missing_done["canonical_isomeric"].astype(str))
    top_projected, unusual_projected = summarize_missing_elements(missing_projected["canonical_isomeric"].astype(str))
    row["missing_done_n"] = int(len(missing_done))
    row["missing_projected_n"] = int(len(missing_projected))
    row["missing_done_top_elements"] = top_done
    row["missing_projected_top_elements"] = top_projected
    row["missing_done_unusual_element_mols"] = unusual_done
    row["missing_projected_unusual_element_mols"] = unusual_projected

    missing = missing_done.copy()
    missing["target"] = target
    missing = missing[["target", "smiles", "canonical_isomeric", "canonical_no_stereo", y_col]].head(25)
    return row, missing


def load_all_target_set() -> CanonSet:
    smiles: list[str] = []
    for target in CASCADE_ORDER:
        path = MODEL_DIR / target / f"{target}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["smiles"], low_memory=False)
        smiles.extend(df["smiles"].dropna().astype(str).tolist())
    return CanonSet.from_smiles(smiles)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chaos_smiles = load_npz_smiles(REPO_ROOT / "data/chaos_25a_mu_matrix.npz", key="canonical_smiles")
    chaos = CanonSet.from_smiles(chaos_smiles)
    generated = load_generated_sigma_set()
    orca_done = load_orca_done_set()
    orca_projected = load_orca_projected_set()

    sigma_any_done = chaos.union(generated, orca_done)
    sigma_any_projected = sigma_any_done.union(orca_projected)
    all_targets = load_all_target_set()

    sets = {
        "chaos53k": chaos,
        "sigma_any_done": sigma_any_done,
        "sigma_any_projected": sigma_any_projected,
        "orca_hp_done": orca_done,
        "orca_hp_projected": orca_projected,
    }

    rows = []
    missing_frames = []
    for target in CASCADE_ORDER:
        row, missing = audit_target(target, sets)
        rows.append(row)
        if not missing.empty:
            missing_frames.append(missing)

    coverage = pd.DataFrame(rows)
    coverage.to_csv(OUT_DIR / "calcphyschemprops_27target_sigma_coverage.csv", index=False)
    if missing_frames:
        pd.concat(missing_frames, ignore_index=True).to_csv(
            OUT_DIR / "calcphyschemprops_missing_sigma_examples.csv",
            index=False,
        )

    summary = {
        "model_dir": str(MODEL_DIR),
        "n_targets": len(CASCADE_ORDER),
        "chaos53k_unique_isomeric": len(chaos.isomeric),
        "chaos53k_unique_no_stereo": len(chaos.no_stereo),
        "generated_sigma_unique_no_stereo": len(generated.no_stereo),
        "orca_hp_done_unique_no_stereo": len(orca_done.no_stereo),
        "orca_hp_projected_unique_no_stereo": len(orca_projected.no_stereo),
        "sigma_any_done_unique_no_stereo": len(sigma_any_done.no_stereo),
        "sigma_any_projected_unique_no_stereo": len(sigma_any_projected.no_stereo),
        "all_27_targets_unique_no_stereo": len(all_targets.no_stereo),
        "all_27_targets_chaos53k_no_stereo_n": len(all_targets.no_stereo & chaos.no_stereo),
        "all_27_targets_chaos53k_no_stereo_frac": len(all_targets.no_stereo & chaos.no_stereo) / len(all_targets.no_stereo),
        "all_27_targets_sigma_any_done_no_stereo_n": len(all_targets.no_stereo & sigma_any_done.no_stereo),
        "all_27_targets_sigma_any_done_no_stereo_frac": len(all_targets.no_stereo & sigma_any_done.no_stereo) / len(all_targets.no_stereo),
        "all_27_targets_sigma_any_projected_no_stereo_n": len(all_targets.no_stereo & sigma_any_projected.no_stereo),
        "all_27_targets_sigma_any_projected_no_stereo_frac": len(all_targets.no_stereo & sigma_any_projected.no_stereo) / len(all_targets.no_stereo),
        "coverage_csv": str(OUT_DIR / "calcphyschemprops_27target_sigma_coverage.csv"),
        "missing_examples_csv": str(OUT_DIR / "calcphyschemprops_missing_sigma_examples.csv"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    display_cols = [
        "target",
        "n_unique_canonical",
        "chaos53k_loose_n",
        "chaos53k_loose_frac",
        "sigma_any_done_loose_n",
        "sigma_any_done_loose_frac",
        "sigma_any_projected_loose_n",
        "sigma_any_projected_loose_frac",
        "orca_hp_done_loose_n",
        "orca_hp_projected_loose_n",
    ]
    print(coverage[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
