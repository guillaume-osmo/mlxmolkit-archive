#!/usr/bin/env python3
"""Prepare web/literature review packets for deltaHvap v3 curation conflicts.

The script deliberately separates:

* identity/element sanity checks,
* source-value clustering,
* external lookup links,
* and the final human review queue.

Online PubChem lookup is optional but enabled by default.  The NIST WebBook and
literature searches are emitted as URLs so the review can be done reproducibly
without hiding manual judgment inside a scraper.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFLICTS = REPO_ROOT / "benchmarks/delta_hvap_v3_curation/deltaHvapv3_conflicts_for_review.csv"
DEFAULT_NEW = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv3_experimental_new_missing_sigma.csv"
DEFAULT_OUT = REPO_ROOT / "benchmarks/delta_hvap_v3_conflict_web_review"
VOLATILE_ORGANIC_ELEMENTS = {"C", "H", "B", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"}
WATCH_ELEMENTS = {"B", "P", "Si"}
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


@dataclass
class MolAudit:
    ok: bool
    formula: str = ""
    inchikey: str = ""
    elements: str = ""
    unusual_elements: str = ""
    watch_elements: str = ""
    formal_charge: int = 0
    n_fragments: int = 0
    heavy_atoms: int = 0
    mol_weight: float = math.nan
    keep_for_sigma: bool = False
    element_action: str = ""


def canonical_mol(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return mol


def audit_smiles(smiles: str, allowed: set[str]) -> MolAudit:
    mol = canonical_mol(smiles)
    if mol is None:
        return MolAudit(ok=False, element_action="invalid_smiles")
    frags = Chem.GetMolFrags(mol, asMols=False)
    symbols = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    unusual = sorted(set(symbols) - allowed)
    watch = sorted(set(symbols) & WATCH_ELEMENTS)
    charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    formula = rdMolDescriptors.CalcMolFormula(mol)
    inchikey = Chem.MolToInchiKey(mol)
    keep = len(frags) == 1 and charge == 0 and not unusual
    if unusual:
        action = "exclude_unusual_element"
    elif len(frags) != 1:
        action = "exclude_multifragment_or_salt"
    elif charge != 0:
        action = "exclude_charged"
    elif watch:
        action = "keep_but_watch_element"
    else:
        action = "keep_clean_organic"
    return MolAudit(
        ok=True,
        formula=formula,
        inchikey=inchikey,
        elements="|".join(symbols),
        unusual_elements="|".join(unusual),
        watch_elements="|".join(watch),
        formal_charge=charge,
        n_fragments=len(frags),
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        mol_weight=float(Descriptors.MolWt(mol)),
        keep_for_sigma=keep,
        element_action=action,
    )


def source_values(row: pd.Series, *, include_pseudo: bool = False) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    pairs = [
        ("autovap", "autovap_trusted_value_kJmol"),
        ("zenodo_nist", "zenodo8132046_nist_value_kJmol"),
        ("naef2017", "naef2017_sdf_value_kJmol"),
        ("mdpi2021", "mdpi2021_s06_value_kJmol"),
    ]
    if include_pseudo:
        pairs.append(("calcphyschemprop", "calcphyschemprop_pseudo_value_kJmol"))
    for source, col in pairs:
        try:
            value = float(row.get(col, math.nan))
        except Exception:
            value = math.nan
        if math.isfinite(value):
            out.append((source, value))
    return out


def cluster_values(values: list[tuple[str, float]], threshold: float = 2.0) -> tuple[str, float, str]:
    if not values:
        return "no_values", math.nan, ""
    items = sorted(values, key=lambda item: item[1])
    clusters: list[list[tuple[str, float]]] = []
    for item in items:
        placed = False
        for cluster in clusters:
            cluster_vals = [x[1] for x in cluster]
            if abs(item[1] - float(np.median(cluster_vals))) <= threshold:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    clusters.sort(key=lambda c: (len(c), -float(np.std([x[1] for x in c]))), reverse=True)
    best = clusters[0]
    best_value = float(np.median([x[1] for x in best]))
    best_sources = "|".join(source for source, _value in best)
    if len(best) >= 2:
        action = "use_majority_cluster"
    elif len(values) == 1:
        action = "single_source_manual_check"
    else:
        action = "manual_literature_review_or_exclude"
    return action, best_value, best_sources


def url_quote(text: str) -> str:
    return urllib.parse.quote(str(text), safe="")


def pubchem_json(url: str, timeout: float = 8.0) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers={"User-Agent": "mlxmolkit-deltaHvap-curation/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def pubchem_lookup(smiles: str, inchikey: str, sleep_s: float = 0.15) -> dict[str, Any]:
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    cid = ""
    if inchikey:
        data = pubchem_json(f"{base}/compound/inchikey/{url_quote(inchikey)}/cids/JSON")
        try:
            cid = str(data["IdentifierList"]["CID"][0]) if data else ""
        except Exception:
            cid = ""
        time.sleep(sleep_s)
    if not cid:
        data = pubchem_json(f"{base}/compound/smiles/{url_quote(smiles)}/cids/JSON")
        try:
            cid = str(data["IdentifierList"]["CID"][0]) if data else ""
        except Exception:
            cid = ""
        time.sleep(sleep_s)

    out = {"pubchem_cid": cid, "pubchem_iupac": "", "pubchem_title": "", "pubchem_cas": "", "pubchem_synonyms": ""}
    if not cid:
        return out

    props = pubchem_json(
        f"{base}/compound/cid/{cid}/property/Title,IUPACName,MolecularFormula,InChIKey,CanonicalSMILES,IsomericSMILES/JSON"
    )
    try:
        prop = props["PropertyTable"]["Properties"][0] if props else {}
        out["pubchem_iupac"] = str(prop.get("IUPACName", ""))
        out["pubchem_title"] = str(prop.get("Title", ""))
    except Exception:
        pass
    time.sleep(sleep_s)

    syn = pubchem_json(f"{base}/compound/cid/{cid}/synonyms/JSON")
    try:
        synonyms = [str(x) for x in syn["InformationList"]["Information"][0].get("Synonym", [])] if syn else []
    except Exception:
        synonyms = []
    cas = [x for x in synonyms if CAS_RE.match(x)]
    out["pubchem_cas"] = "|".join(cas[:8])
    out["pubchem_synonyms"] = "|".join(synonyms[:12])
    time.sleep(sleep_s)
    return out


def extract_refs(source_detail: str) -> str:
    refs = []
    for match in re.finditer(r"refs=([^|]+?)(?:;|$)", str(source_detail)):
        text = match.group(1).strip()
        if text and text not in refs:
            refs.append(text)
    return " | ".join(refs)


def make_links(row: dict[str, Any]) -> dict[str, str]:
    names = [
        row.get("pubchem_title", ""),
        row.get("pubchem_iupac", ""),
        row.get("canonical_smiles", ""),
    ]
    name = next((str(x) for x in names if str(x).strip()), str(row.get("canonical_smiles", "")))
    query = f'"{name}" "enthalpy of vaporization" "kJ"'
    cas = str(row.get("pubchem_cas", "")).split("|")[0].strip()
    if cas:
        nist = f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas.replace('-', '')}&Units=SI&Mask=4#Thermo-Phase"
    else:
        nist = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={url_quote(name)}&Units=SI&Mask=4"
    return {
        "nist_webbook_url": nist,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{row.get('pubchem_cid')}" if row.get("pubchem_cid") else "",
        "google_search_url": f"https://www.google.com/search?q={url_quote(query)}",
        "google_scholar_url": f"https://scholar.google.com/scholar?q={url_quote(query)}",
        "crossref_search_url": f"https://search.crossref.org/?q={url_quote(query)}",
    }


def audit_dataframe(df: pd.DataFrame, allowed: set[str]) -> pd.DataFrame:
    audits = []
    for smi in df["canonical_smiles"].astype(str):
        audit = audit_smiles(smi, allowed)
        audits.append(audit.__dict__)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(audits)], axis=1)


def build_conflict_packet(conflicts: pd.DataFrame, allowed: set[str], *, online: bool) -> pd.DataFrame:
    severe = conflicts[conflicts["curation_action"].astype(str).eq("exclude_experimental_conflict_gt10")].copy()
    audited = audit_dataframe(severe, allowed)
    rows = []
    for _, row in audited.iterrows():
        rec = row.to_dict()
        values = source_values(row, include_pseudo=False)
        action, best_value, best_sources = cluster_values(values)
        rec["value_cluster_action"] = action
        rec["value_cluster_kJmol"] = best_value
        rec["value_cluster_sources"] = best_sources
        rec["source_refs"] = extract_refs(str(row.get("source_detail", "")))
        if online and bool(rec.get("ok", False)):
            rec.update(pubchem_lookup(str(row["canonical_smiles"]), str(row.get("inchikey", ""))))
        else:
            rec.update({"pubchem_cid": "", "pubchem_iupac": "", "pubchem_title": "", "pubchem_cas": "", "pubchem_synonyms": ""})
        rec.update(make_links(rec))
        if rec.get("element_action") != "keep_clean_organic" and rec.get("element_action") != "keep_but_watch_element":
            rec["review_recommendation"] = "exclude_element_or_charge"
        elif action == "use_majority_cluster":
            rec["review_recommendation"] = "accept_majority_after_nist_check"
        else:
            rec["review_recommendation"] = "manual_literature_review"
        rows.append(rec)
    return pd.DataFrame(rows)


def write_markdown(packet: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# deltaHvap v3 Severe Conflict Review",
        "",
        "Each case has >10 kJ/mol spread between experimental/provenance sources.",
        "Use the NIST link first, then the literature query and source references.",
        "",
    ]
    for i, row in packet.sort_values("experimental_spread_kJmol", ascending=False).iterrows():
        lines.extend(
            [
                f"## {i + 1}. {row.get('pubchem_title') or row['canonical_smiles']}",
                "",
                f"- SMILES: `{row['canonical_smiles']}`",
                f"- Formula/elements: `{row.get('formula', '')}` / `{row.get('elements', '')}`",
                f"- Spread: {float(row.get('experimental_spread_kJmol', math.nan)):.2f} kJ/mol",
                f"- Source values: AutoVap={row.get('autovap_trusted_value_kJmol', math.nan)}, "
                f"Zenodo/NIST={row.get('zenodo8132046_nist_value_kJmol', math.nan)}, "
                f"Naef2017={row.get('naef2017_sdf_value_kJmol', math.nan)}, "
                f"MDPI2021={row.get('mdpi2021_s06_value_kJmol', math.nan)}",
                f"- Cluster recommendation: `{row.get('value_cluster_action')}` -> "
                f"{row.get('value_cluster_kJmol')} from `{row.get('value_cluster_sources')}`",
                f"- Element action: `{row.get('element_action')}`",
                f"- Review recommendation: `{row.get('review_recommendation')}`",
                f"- Source refs: {row.get('source_refs', '')}",
                f"- NIST: {row.get('nist_webbook_url', '')}",
                f"- PubChem: {row.get('pubchem_url', '')}",
                f"- Literature: {row.get('google_scholar_url', '')}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--new-missing", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allowed-elements", default=",".join(sorted(VOLATILE_ORGANIC_ELEMENTS)))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    allowed = {x.strip() for x in args.allowed_elements.split(",") if x.strip()}
    conflicts = pd.read_csv(args.conflicts)
    new = pd.read_csv(args.new_missing)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    packet = build_conflict_packet(conflicts, allowed, online=not args.offline)
    packet_path = args.out_dir / "deltaHvapv3_severe22_web_review_packet.csv"
    packet.to_csv(packet_path, index=False)
    write_markdown(packet, args.out_dir / "deltaHvapv3_severe22_web_review_packet.md")

    new_audit = audit_dataframe(new, allowed)
    keep = new_audit[new_audit["keep_for_sigma"].astype(bool)].copy()
    drop = new_audit[~new_audit["keep_for_sigma"].astype(bool)].copy()
    new_audit.to_csv(args.out_dir / "deltaHvapv3_new1362_element_audit.csv", index=False)
    keep.to_csv(args.out_dir / "deltaHvapv3_new1362_keep_for_sigma.csv", index=False)
    drop.to_csv(args.out_dir / "deltaHvapv3_new1362_drop_or_review_elements.csv", index=False)
    sigma_queue = pd.DataFrame(
        {
            "canonical_smiles": keep["canonical_smiles"].astype(str),
            "trusted_target_kJmol": pd.to_numeric(keep["curated_deltaHvap_kJmol"], errors="coerce"),
            "curated_target_kJmol": pd.to_numeric(keep["curated_deltaHvap_kJmol"], errors="coerce"),
            "target_source": "deltaHvapv3_experimental_new",
            "sample_weight": pd.to_numeric(keep["strict_sample_weight"], errors="coerce").fillna(0.0),
            "pseudo_calibration": "",
            "curation_action": keep["curation_action"].astype(str),
            "curation_note": keep["source_detail"].astype(str),
            "autovap_dvap_kJmol": np.nan,
            "calc_deltaHvap_source_kJmol": np.nan,
            "calc_deltaHvap_pred_kJmol": np.nan,
            "calc_deltaHvap_source_homoset_aligned_kJmol": np.nan,
            "calc_deltaHvap_pred_homoset_aligned_kJmol": np.nan,
            "calc_deltaHvap_source_curated_kJmol": np.nan,
            "calc_deltaHvap_pred_curated_kJmol": np.nan,
            "autovap_n_rows": np.nan,
            "autovap_std_kJmol": np.nan,
            "autovap_range_kJmol": np.nan,
            "autovap_n_fragments": np.nan,
            "autovap_cas": "",
            "autovap_inchikey": keep["inchikey"].astype(str),
            "autovap_family": "",
            "autovap_smiles": "",
            "calc_smiles": keep["canonical_smiles"].astype(str),
            "v3_confidence": keep["confidence"].astype(str),
            "v3_elements": keep["elements"].astype(str),
            "v3_formula": keep["formula"].astype(str),
        }
    )
    sigma_queue_path = args.out_dir / "deltaHvapv3_new1347_sigma_queue.csv"
    sigma_queue.to_csv(sigma_queue_path, index=False)

    element_counts: dict[str, int] = {}
    for symbols in new_audit["elements"].fillna("").astype(str):
        for symbol in symbols.split("|"):
            if symbol:
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
    summary = {
        "severe_conflict_rows": int(len(packet)),
        "severe_conflict_recommendations": packet["review_recommendation"].value_counts().to_dict(),
        "severe_conflict_element_actions": packet["element_action"].value_counts().to_dict(),
        "new_rows": int(len(new_audit)),
        "new_keep_for_sigma": int(len(keep)),
        "new_drop_or_review": int(len(drop)),
        "new_element_actions": new_audit["element_action"].value_counts().to_dict(),
        "new_element_counts": dict(sorted(element_counts.items())),
        "allowed_elements": sorted(allowed),
        "outputs": {
            "severe_review_packet": str(packet_path),
            "severe_review_markdown": str(args.out_dir / "deltaHvapv3_severe22_web_review_packet.md"),
            "new_element_audit": str(args.out_dir / "deltaHvapv3_new1362_element_audit.csv"),
            "new_keep_for_sigma": str(args.out_dir / "deltaHvapv3_new1362_keep_for_sigma.csv"),
            "new_drop_or_review": str(args.out_dir / "deltaHvapv3_new1362_drop_or_review_elements.csv"),
            "new_sigma_queue": str(sigma_queue_path),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
