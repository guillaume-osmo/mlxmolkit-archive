#!/usr/bin/env python3
"""Audit MDPI Molecules 2021 deltaHvap source tables.

The supplement for Molecules 2021, 26, 1045 provides logVP, deltaG_vap,
deltaS_vap, and deltaH_vap tables. This script parses the local supplement,
reconstructs deltaH_vap from deltaG_vap + T * deltaS_vap, and joins the result
to the deltaHvapv2 homoset.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPLEMENT = (
    REPO_ROOT
    / "data/delta_hvap_v2/source_mdpi_molecules_2021_26_1045/supplementary/molecules-1089923-SM-proofed"
)
DEFAULT_UNION = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_homoset_union.csv"
DEFAULT_OUT = REPO_ROOT / "benchmarks/delta_hvap_v2_mdpi_source_check"


def norm_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def canonical_smiles(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def ensure_pdf_text(pdf: Path, out_txt: Path) -> None:
    if out_txt.exists() and out_txt.stat().st_mtime >= pdf.stat().st_mtime:
        return
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", "-layout", str(pdf), str(out_txt)], check=True)


def parse_pdf_table(path: Path, value_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    num = r"[-+]?\d+(?:\.\d+)?"
    if value_prefix == "logVP":
        pat_full = re.compile(
            rf"^(?P<name>.*?)(?P<exp>{num})\s+(?P<calc>{num})\s+"
            rf"(?P<test>{num})\s+(?P<dev>{num})\s*$"
        )
        pat_short = re.compile(
            rf"^(?P<name>.*?)(?P<exp>{num})\s+(?P<calc>{num})\s+(?P<dev>{num})\s*$"
        )
    else:
        pat_full = re.compile(
            rf"^(?P<name>.*?)(?P<exp>{num})\s+(?P<calc>{num})\s+"
            rf"(?P<dev>{num})\s+(?P<pct>{num})\s*$"
        )
        pat_short = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped == "\x0c" or stripped.isdigit():
                continue
            if re.search(r"Molecule name|Deviation|Dev in|^\s*(exp|calc)\s*$", stripped, re.I):
                continue
            m = pat_full.match(stripped)
            short = None if m is not None or pat_short is None else pat_short.match(stripped)
            if m:
                name = m.group("name").strip()
                if not name:
                    continue
                row = {
                    "mdpi_name": name,
                    f"{value_prefix}_exp": float(m.group("exp")),
                    f"{value_prefix}_calc": float(m.group("calc")),
                    f"{value_prefix}_deviation": float(m.group("dev")),
                }
                if value_prefix == "logVP":
                    row[f"{value_prefix}_test"] = float(m.group("test"))
                else:
                    row[f"{value_prefix}_deviation_pct"] = float(m.group("pct"))
                rows.append(row)
            elif short:
                name = short.group("name").strip()
                if not name:
                    continue
                rows.append(
                    {
                        "mdpi_name": name,
                        f"{value_prefix}_exp": float(short.group("exp")),
                        f"{value_prefix}_calc": float(short.group("calc")),
                        f"{value_prefix}_test": np.nan,
                        f"{value_prefix}_deviation": float(short.group("dev")),
                    }
                )
            elif rows:
                # Several molecule names wrap after the numeric columns.
                rows[-1]["mdpi_name"] = str(rows[-1]["mdpi_name"]) + stripped
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mdpi_name_key"] = df["mdpi_name"].map(norm_name)
    return df


def load_s01_name_index(sdf_path: Path) -> pd.DataFrame:
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
    rows: list[dict[str, object]] = []
    for mol in supplier:
        if mol is None or not mol.HasProp("Alias name"):
            continue
        name = mol.GetProp("Alias name")
        rows.append(
            {
                "mdpi_name": name,
                "mdpi_name_key": norm_name(name),
                "canonical_smiles": canonical_smiles(mol),
                "mdpi_logVP_sdf": float(mol.GetProp("logVP")) if mol.HasProp("logVP") else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.sort_values(["mdpi_name_key", "canonical_smiles"])
        .drop_duplicates("mdpi_name_key", keep="first")
        .reset_index(drop=True)
    )


def load_s06(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(
        columns={
            df.columns[0]: "mdpi_name",
            df.columns[1]: "deltaH_vap_exp_s06",
            df.columns[2]: "deltaH_vap_calc_s06",
        }
    )
    df = df.dropna(subset=["mdpi_name"]).copy()
    df["mdpi_name_key"] = df["mdpi_name"].map(norm_name)
    df["deltaH_vap_exp_s06"] = pd.to_numeric(df["deltaH_vap_exp_s06"], errors="coerce")
    df["deltaH_vap_calc_s06"] = pd.to_numeric(df["deltaH_vap_calc_s06"], errors="coerce")
    return df


def abs_stats(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().abs()
    if clean.empty:
        return {"n": 0}
    return {
        "n": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")

    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--temperature-k", type=float, default=298.15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    text_dir = args.out_dir / "pdf_text"
    pdfs = {
        "logVP": args.supplement_dir / "S02. Experimental vs. calculated logVP Data Table.pdf",
        "deltaG_vap": args.supplement_dir / "S03. Experimental vs. calculated deltaG_vap) Data Table.pdf",
        "deltaS_vap": args.supplement_dir / "S04. Experimental vs. calculated deltaS_vap) Data Table.pdf",
    }
    text_paths: dict[str, Path] = {}
    for key, pdf in pdfs.items():
        out_txt = text_dir / f"{key}.txt"
        ensure_pdf_text(pdf, out_txt)
        text_paths[key] = out_txt

    s01 = load_s01_name_index(args.supplement_dir / "S01. Compounds List for logVP Calculations.utf8.sdf")
    s02 = parse_pdf_table(text_paths["logVP"], "logVP")
    s03 = parse_pdf_table(text_paths["deltaG_vap"], "deltaG_vap")
    s04 = parse_pdf_table(text_paths["deltaS_vap"], "deltaS_vap")
    s06 = load_s06(args.supplement_dir / "S06. Exp. and calc. deltaH_vap) Data Table.xls")

    thermo = s01.merge(
        s02.drop(columns=["mdpi_name"], errors="ignore"), on="mdpi_name_key", how="outer"
    )
    thermo = thermo.merge(
        s03.drop(columns=["mdpi_name"], errors="ignore"), on="mdpi_name_key", how="outer"
    )
    thermo = thermo.merge(
        s04.drop(columns=["mdpi_name"], errors="ignore"), on="mdpi_name_key", how="outer"
    )
    thermo = thermo.merge(
        s06.drop(columns=["mdpi_name"], errors="ignore"), on="mdpi_name_key", how="outer"
    )
    thermo["deltaH_vap_exp_from_GS"] = (
        thermo["deltaG_vap_exp"] + args.temperature_k * thermo["deltaS_vap_exp"] / 1000.0
    )
    thermo["deltaH_vap_calc_from_GS"] = (
        thermo["deltaG_vap_calc"] + args.temperature_k * thermo["deltaS_vap_calc"] / 1000.0
    )
    thermo["deltaH_exp_reconstruction_abs_err"] = (
        thermo["deltaH_vap_exp_from_GS"] - thermo["deltaH_vap_exp_s06"]
    ).abs()
    thermo["deltaH_calc_reconstruction_abs_err"] = (
        thermo["deltaH_vap_calc_from_GS"] - thermo["deltaH_vap_calc_s06"]
    ).abs()

    thermo_out = args.out_dir / "mdpi_thermo_reconstructed.csv"
    thermo.to_csv(thermo_out, index=False)

    union_join_out = None
    union_summary: dict[str, object] = {}
    if args.union.exists():
        union = pd.read_csv(args.union)
        mdpi_by_smiles = (
            thermo[thermo["canonical_smiles"].fillna("") != ""]
            .sort_values(["canonical_smiles", "deltaH_exp_reconstruction_abs_err"], na_position="last")
            .drop_duplicates("canonical_smiles", keep="first")
        )
        joined = union.merge(mdpi_by_smiles, on="canonical_smiles", how="left", suffixes=("", "_mdpi"))
        joined["target_minus_mdpi_deltaH_exp_s06"] = (
            joined["trusted_target_kJmol"] - joined["deltaH_vap_exp_s06"]
        )
        joined["target_minus_mdpi_deltaH_exp_from_GS"] = (
            joined["trusted_target_kJmol"] - joined["deltaH_vap_exp_from_GS"]
        )
        joined["target_minus_mdpi_deltaH_calc_s06"] = (
            joined["trusted_target_kJmol"] - joined["deltaH_vap_calc_s06"]
        )
        union_join_out = args.out_dir / "deltaHvapv2_union_vs_mdpi_thermo.csv"
        joined.to_csv(union_join_out, index=False)
        calc_only = joined["target_source"].eq("calcphyschemprop_calibrated_pseudo")
        autovap = joined["target_source"].eq("autovap_trusted")
        union_summary = {
            "union_rows": int(len(joined)),
            "union_mdpi_smiles_hits": int(joined["deltaH_vap_exp_s06"].notna().sum()),
            "autovap_mdpi_hits": int((autovap & joined["deltaH_vap_exp_s06"].notna()).sum()),
            "calc_only_mdpi_hits": int((calc_only & joined["deltaH_vap_exp_s06"].notna()).sum()),
            "autovap_target_vs_mdpi_exp_s06_abs_kJmol": abs_stats(
                joined.loc[autovap, "target_minus_mdpi_deltaH_exp_s06"]
            ),
            "autovap_target_vs_mdpi_calc_s06_abs_kJmol": abs_stats(
                joined.loc[autovap, "target_minus_mdpi_deltaH_calc_s06"]
            ),
            "calc_only_target_vs_mdpi_exp_s06_abs_kJmol": abs_stats(
                joined.loc[calc_only, "target_minus_mdpi_deltaH_exp_s06"]
            ),
            "calc_only_target_vs_mdpi_calc_s06_abs_kJmol": abs_stats(
                joined.loc[calc_only, "target_minus_mdpi_deltaH_calc_s06"]
            ),
        }

    summary = {
        "temperature_k": args.temperature_k,
        "counts": {
            "S01_sdf": int(len(s01)),
            "S02_logVP_pdf": int(len(s02)),
            "S03_deltaG_pdf": int(len(s03)),
            "S04_deltaS_pdf": int(len(s04)),
            "S06_deltaH_xls": int(len(s06)),
            "thermo_union_by_name": int(len(thermo)),
            "deltaG_and_deltaS_common": int(
                thermo[["deltaG_vap_exp", "deltaS_vap_exp"]].notna().all(axis=1).sum()
            ),
            "deltaG_deltaS_and_S06_common": int(
                thermo[["deltaG_vap_exp", "deltaS_vap_exp", "deltaH_vap_exp_s06"]]
                .notna()
                .all(axis=1)
                .sum()
            ),
        },
        "deltaH_exp_from_GS_vs_S06_abs_kJmol": abs_stats(
            thermo["deltaH_vap_exp_from_GS"] - thermo["deltaH_vap_exp_s06"]
        ),
        "deltaH_calc_from_GS_vs_S06_abs_kJmol": abs_stats(
            thermo["deltaH_vap_calc_from_GS"] - thermo["deltaH_vap_calc_s06"]
        ),
        "outputs": {
            "thermo_reconstructed_csv": str(thermo_out),
            "union_join_csv": str(union_join_out) if union_join_out else None,
        },
        "union_join": union_summary,
    }
    summary_path = args.out_dir / "mdpi_thermo_reconstructed.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
