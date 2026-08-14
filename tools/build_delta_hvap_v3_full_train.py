#!/usr/bin/env python3
"""Build full v3 deltaHvap train CSVs from the curated source table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/delta_hvap_v2"
DEFAULT_CURATED = DATA_DIR / "deltaHvapv3_curated_sources.csv"
ALLOWED_ORGANIC_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}


def make_view(curated: pd.DataFrame, *, weight_col: str, include_pseudo: bool) -> pd.DataFrame:
    out = pd.DataFrame()
    out["canonical_smiles"] = curated["canonical_smiles"].astype(str)
    out["trusted_target_kJmol"] = pd.to_numeric(curated["curated_deltaHvap_kJmol"], errors="coerce")
    out["curated_target_kJmol"] = out["trusted_target_kJmol"]
    out["target_source"] = "excluded_by_v3_curation"
    out["sample_weight"] = pd.to_numeric(curated[weight_col], errors="coerce").fillna(0.0)
    out["pseudo_calibration"] = np.where(curated["confidence"].astype(str).eq("pseudo"), "calcphyschemprop", "")
    out["curation_action"] = curated["curation_action"].astype(str)
    out["curation_note"] = curated["source_detail"].astype(str)

    out["autovap_dvap_kJmol"] = pd.to_numeric(curated.get("autovap_trusted_value_kJmol", np.nan), errors="coerce")
    out["calc_deltaHvap_source_kJmol"] = pd.to_numeric(curated.get("calcphyschemprop_pseudo_value_kJmol", np.nan), errors="coerce")
    out["calc_deltaHvap_pred_kJmol"] = np.nan
    out["calc_deltaHvap_source_homoset_aligned_kJmol"] = out["calc_deltaHvap_source_kJmol"]
    out["calc_deltaHvap_pred_homoset_aligned_kJmol"] = np.nan
    out["calc_deltaHvap_source_curated_kJmol"] = out["calc_deltaHvap_source_kJmol"]
    out["calc_deltaHvap_pred_curated_kJmol"] = np.nan

    out["autovap_n_rows"] = pd.to_numeric(curated.get("autovap_trusted_n", 0), errors="coerce").fillna(0)
    out["autovap_std_kJmol"] = np.nan
    out["autovap_range_kJmol"] = np.nan
    out["autovap_n_fragments"] = 1
    out["autovap_cas"] = ""
    out["autovap_inchikey"] = ""
    out["autovap_family"] = ""
    out["autovap_smiles"] = ""
    out["calc_smiles"] = curated["canonical_smiles"].astype(str)

    out["v3_confidence"] = curated["confidence"].astype(str)
    out["v3_curation_action"] = curated["curation_action"].astype(str)
    out["v3_source_names"] = curated["source_names"].astype(str)
    out["v3_experimental_spread_kJmol"] = pd.to_numeric(curated["experimental_spread_kJmol"], errors="coerce")

    positive = out["sample_weight"].to_numpy(dtype=float) > 0
    pseudo = curated["confidence"].astype(str).eq("pseudo").to_numpy()
    experimental = (pd.to_numeric(curated["n_experimental_labels"], errors="coerce").fillna(0).to_numpy() > 0)

    out.loc[positive & experimental, "target_source"] = "deltaHvapv3_experimental_new"
    if include_pseudo:
        out.loc[positive & pseudo, "target_source"] = "calcphyschemprop_calibrated_pseudo"

    return out


def is_single_fragment_smiles(smiles: object) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    return bool(mol is not None and len(Chem.GetMolFrags(mol)) == 1)


def has_allowed_elements(smiles: object) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    return all(atom.GetSymbol() in ALLOWED_ORGANIC_ELEMENTS for atom in mol.GetAtoms())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    curated = pd.read_csv(args.curated, low_memory=False)
    strict = make_view(curated, weight_col="strict_sample_weight", include_pseudo=False)
    broad = make_view(curated, weight_col="broad_sample_weight", include_pseudo=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strict_path = args.out_dir / "deltaHvapv3_full4509_strict_train.csv"
    broad_path = args.out_dir / "deltaHvapv3_full4509_broad_train.csv"
    single_mask = broad["canonical_smiles"].map(is_single_fragment_smiles).to_numpy(dtype=bool)
    allowed_element_mask = broad["canonical_smiles"].map(has_allowed_elements).to_numpy(dtype=bool)
    clean_mask = single_mask & allowed_element_mask
    strict_clean = strict[clean_mask].reset_index(drop=True)
    broad_clean = broad[clean_mask].reset_index(drop=True)
    strict_clean_path = args.out_dir / "deltaHvapv3_full4493_organic_singlefrag_strict_train.csv"
    broad_clean_path = args.out_dir / "deltaHvapv3_full4493_organic_singlefrag_broad_train.csv"
    strict.to_csv(strict_path, index=False)
    broad.to_csv(broad_path, index=False)
    strict_clean.to_csv(strict_clean_path, index=False)
    broad_clean.to_csv(broad_clean_path, index=False)

    summary = {
        "curated_rows": int(len(curated)),
        "strict_path": str(strict_path),
        "broad_path": str(broad_path),
        "strict_singlefrag_path": str(strict_clean_path),
        "broad_singlefrag_path": str(broad_clean_path),
        "allowed_organic_elements": sorted(ALLOWED_ORGANIC_ELEMENTS),
        "dropped_non_single_fragment_rows": int((~single_mask).sum()),
        "dropped_disallowed_element_rows": int((single_mask & ~allowed_element_mask).sum()),
        "dropped_total_clean_filter_rows": int((~clean_mask).sum()),
        "strict_target_source_counts": {str(k): int(v) for k, v in strict["target_source"].value_counts().items()},
        "broad_target_source_counts": {str(k): int(v) for k, v in broad["target_source"].value_counts().items()},
        "strict_singlefrag_target_source_counts": {str(k): int(v) for k, v in strict_clean["target_source"].value_counts().items()},
        "broad_singlefrag_target_source_counts": {str(k): int(v) for k, v in broad_clean["target_source"].value_counts().items()},
        "strict_train_weight_positive": int((strict["sample_weight"] > 0).sum()),
        "broad_train_weight_positive": int((broad["sample_weight"] > 0).sum()),
        "strict_singlefrag_train_weight_positive": int((strict_clean["sample_weight"] > 0).sum()),
        "broad_singlefrag_train_weight_positive": int((broad_clean["sample_weight"] > 0).sum()),
    }
    summary_path = args.out_dir / "deltaHvapv3_full4509_train.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
