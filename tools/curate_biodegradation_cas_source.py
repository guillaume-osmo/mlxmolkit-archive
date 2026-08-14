#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from train_biodegradation_pdl_ordered import DEFAULT_RIFM_2026, canonical_smiles, normalize_guideline


DEFAULT_INPUT = Path("/Users/guillaume-osmo/Downloads/Biodegradation.csv")
DEFAULT_EXCELRA = Path("/Users/guillaume-osmo/Github/data/Biodegradation-cleaned-dataset_Excelra_240630.csv")
DEFAULT_RIFM_2024 = Path("/Users/guillaume-osmo/Downloads/BioDegradationData2024 (2).xlsx")


def norm_cas(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"\b\d{2,7}-\d{2}-\d\b", str(value).strip())
    return match.group(0) if match else None


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", ".")
    if not text:
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group(0)) if match else np.nan


def duration_to_days(value: object, unit: object) -> float:
    amount = parse_number(value)
    if not np.isfinite(amount):
        return np.nan
    text = "" if pd.isna(unit) else str(unit).strip().lower()
    if text.startswith("hour"):
        return amount / 24.0
    if text.startswith("week"):
        return amount * 7.0
    if text.startswith("month"):
        return amount * 30.4375
    if text.startswith("year"):
        return amount * 365.25
    return amount


def ready_label_from_result(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "readily biodegradable" in text:
        return 1.0
    if "no biodegradation" in text or "not biodegradable" in text:
        return 0.0
    return np.nan


def protocol_group(method: object) -> str:
    norm = normalize_guideline(method)
    text = norm.upper()
    if "301" in text or "C.4" in text or "C.5" in text or "C.6" in text:
        return "ready"
    if "302" in text:
        return "inherent"
    if "303" in text or "SIMULATION" in text:
        return "simulation"
    return "other" if norm not in {"UNKNOWN", ""} else "unknown"


def y_percent_from_row(row: pd.Series, prefix: str) -> tuple[float, str]:
    lower = parse_number(row.get(f"{prefix}|Deg Lower"))
    upper = parse_number(row.get(f"{prefix}|Deg Upper"))
    comparator = "" if pd.isna(row.get(f"{prefix}|Degradation")) else str(row.get(f"{prefix}|Degradation")).strip()
    if np.isfinite(lower) and np.isfinite(upper):
        return float(np.clip((lower + upper) / 2.0, 0.0, 100.0)), "range_midpoint"
    if np.isfinite(lower):
        source = {
            "<": "upper_bound_as_value",
            ">": "lower_bound_as_value",
            "equal": "direct",
            "ca.": "approx",
        }.get(comparator, "direct")
        return float(np.clip(lower, 0.0, 100.0)), source
    if np.isfinite(upper):
        return float(np.clip(upper, 0.0, 100.0)), "upper_only"
    return np.nan, "missing"


def load_local_cas_smiles(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []

    if Path(args.excelra_csv).exists():
        excelra = pd.read_csv(args.excelra_csv, usecols=["cas_number", "smiles"], dtype=str)
        for _, row in excelra.iterrows():
            cas = norm_cas(row["cas_number"])
            smi = canonical_smiles(row["smiles"])
            if cas and smi:
                rows.append({"cas": cas, "canonical_smiles": smi, "mapping_source": "excelra"})

    for label, path in [("rifm2026", args.rifm_xlsx), ("rifm2024", args.rifm_2024_xlsx)]:
        if not Path(path).exists():
            continue
        try:
            rifm = pd.read_excel(path, header=1, usecols=["CAS", "SMILES"], dtype=str)
        except Exception:
            continue
        for _, row in rifm.iterrows():
            cas = norm_cas(row["CAS"])
            smi = canonical_smiles(row["SMILES"])
            if cas and smi:
                rows.append({"cas": cas, "canonical_smiles": smi, "mapping_source": label})

    if not rows:
        return pd.DataFrame(columns=["cas", "canonical_smiles", "mapping_source", "n_smiles_for_cas"])

    raw = pd.DataFrame(rows).drop_duplicates()
    counts = raw.groupby("cas")["canonical_smiles"].nunique().rename("n_smiles_for_cas")
    source = raw.groupby("cas")["mapping_source"].agg(lambda x: "|".join(sorted(set(map(str, x))))).rename("mapping_source")
    first = raw.sort_values(["cas", "mapping_source", "canonical_smiles"]).drop_duplicates("cas")
    return first[["cas", "canonical_smiles"]].merge(source, on="cas").merge(counts, on="cas")


def fetch_pubchem_smiles(cas: str, cache_dir: Path, delay: float) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cas}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    quoted = urllib.parse.quote(cas, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quoted}/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    )
    result: dict[str, object] = {"cas": cas, "status": "missing"}
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        props = payload.get("PropertyTable", {}).get("Properties", [])
        if props:
            first = props[0]
            raw_smiles = (
                first.get("IsomericSMILES")
                or first.get("CanonicalSMILES")
                or first.get("SMILES")
                or first.get("ConnectivitySMILES")
            )
            smi = canonical_smiles(raw_smiles)
            if smi:
                result = {
                    "cas": cas,
                    "status": "ok",
                    "canonical_smiles": smi,
                    "pubchem_inchikey": first.get("InChIKey"),
                }
    except urllib.error.HTTPError as exc:
        result = {"cas": cas, "status": f"http_{exc.code}"}
    except Exception as exc:
        result = {"cas": cas, "status": f"error:{type(exc).__name__}"}

    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if delay > 0:
        time.sleep(delay)
    return result


def add_pubchem_mappings(mapping: pd.DataFrame, cas_values: list[str], args: argparse.Namespace) -> pd.DataFrame:
    existing = set(mapping["cas"].astype(str)) if len(mapping) else set()
    missing = [cas for cas in sorted(set(cas_values)) if cas and cas not in existing]
    rows: list[dict] = []
    cache_dir = Path(args.pubchem_cache_dir)
    for i, cas in enumerate(missing, start=1):
        result = fetch_pubchem_smiles(cas, cache_dir, args.pubchem_delay)
        if result.get("status") == "ok":
            rows.append(
                {
                    "cas": cas,
                    "canonical_smiles": result["canonical_smiles"],
                    "mapping_source": "pubchem",
                    "n_smiles_for_cas": 1,
                    "pubchem_inchikey": result.get("pubchem_inchikey"),
                }
            )
        if i % 50 == 0:
            print(f"[biodeg-cas] PubChem resolved {i}/{len(missing)} missing CAS", flush=True)
    if not rows:
        return mapping
    extra = pd.DataFrame(rows)
    for col in extra.columns:
        if col not in mapping.columns:
            mapping[col] = np.nan
    for col in mapping.columns:
        if col not in extra.columns:
            extra[col] = np.nan
    return pd.concat([mapping, extra[mapping.columns]], axis=0, ignore_index=True)


def extract_observations(df: pd.DataFrame) -> pd.DataFrame:
    block_ids: set[int] = set()
    for col in df.columns:
        match = re.match(r"Environmental Fate and Pathways_Biodegradation\|(\d+)\|", col)
        if match:
            block_ids.add(int(match.group(1)))

    rows: list[dict] = []
    for source_row, row in df.iterrows():
        cas = norm_cas(row.get("CAS Number"))
        for block_id in sorted(block_ids):
            prefix = f"Environmental Fate and Pathways_Biodegradation|{block_id}"
            keys = [
                "Result",
                "Method",
                "Type",
                "Degradation",
                "Deg Lower",
                "Deg Upper",
                "Deg Exp",
                "Deg Exp Unit",
                "Reliability",
                "GLP",
                "LUID",
            ]
            if all(pd.isna(row.get(f"{prefix}|{key}")) or str(row.get(f"{prefix}|{key}")).strip() == "" for key in keys):
                continue
            y_percent, y_source = y_percent_from_row(row, prefix)
            method = row.get(f"{prefix}|Method")
            result = row.get(f"{prefix}|Result")
            duration_days = duration_to_days(row.get(f"{prefix}|Deg Exp"), row.get(f"{prefix}|Deg Exp Unit"))
            rows.append(
                {
                    "source": "biodegradation_cas_csv",
                    "source_row": int(source_row),
                    "block_id": int(block_id),
                    "cas": cas,
                    "raw_cas": row.get("CAS Number"),
                    "name": row.get("Common Name"),
                    "molecular_formula": row.get("Molecular Formula"),
                    "molecular_weight": parse_number(row.get("Molecular Weight")),
                    "result": result,
                    "type": row.get(f"{prefix}|Type"),
                    "method": method,
                    "guideline_norm": normalize_guideline(method),
                    "protocol_group": protocol_group(method),
                    "year": parse_number(row.get(f"{prefix}|Year")),
                    "inoculum": row.get(f"{prefix}|Inoculum"),
                    "degradation_relation": row.get(f"{prefix}|Degradation"),
                    "y_percent": y_percent,
                    "y_percent_source": y_source,
                    "duration_days": duration_days,
                    "duration_unit": row.get(f"{prefix}|Deg Exp Unit"),
                    "ready_label_from_result": ready_label_from_result(result),
                    "reliability": row.get(f"{prefix}|Reliability"),
                    "glp": row.get(f"{prefix}|GLP"),
                    "luid": row.get(f"{prefix}|LUID"),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input_csv, low_memory=False)
    source_cas = pd.Series(raw["CAS Number"]).map(norm_cas).dropna().astype(str).tolist()
    observations = extract_observations(raw)
    mapping = load_local_cas_smiles(args)
    if args.pubchem:
        mapping = add_pubchem_mappings(mapping, source_cas, args)
    observations = observations.merge(mapping, on="cas", how="left")
    observations["has_smiles"] = observations["canonical_smiles"].notna()
    observations["is_mapping_conflict"] = observations["n_smiles_for_cas"].fillna(0).gt(1)

    mapped = observations[observations["has_smiles"] & ~observations["is_mapping_conflict"]].copy()
    usable = mapped[mapped["y_percent"].notna()].copy()

    observations.to_csv(out_dir / "biodegradation_cas_observations_long.csv", index=False)
    mapped.to_csv(out_dir / "biodegradation_cas_observations_mapped.csv", index=False)
    usable.to_csv(out_dir / "biodegradation_cas_observations_usable_percent.csv", index=False)
    mapping.to_csv(out_dir / "cas_to_smiles_local_mapping.csv", index=False)

    report = {
        "input_csv": str(Path(args.input_csv)),
        "raw_rows": int(len(raw)),
        "unique_cas_raw": int(pd.Series(raw["CAS Number"]).map(norm_cas).nunique()),
        "observation_rows": int(len(observations)),
        "observations_with_percent": int(observations["y_percent"].notna().sum()),
        "observations_with_smiles": int(observations["has_smiles"].sum()),
        "observations_usable_percent_nonconflict_mapping": int(len(usable)),
        "unique_cas_observations": int(observations["cas"].nunique()),
        "unique_source_cas_with_mapping": int(observations.loc[observations["has_smiles"], "cas"].nunique()),
        "unique_mapping_table_cas": int(mapping["cas"].nunique()),
        "mapping_conflict_cas": int(mapping["n_smiles_for_cas"].gt(1).sum()) if len(mapping) else 0,
        "pubchem_enabled": bool(args.pubchem),
        "protocol_group_counts": observations["protocol_group"].value_counts(dropna=False).to_dict(),
        "result_counts": observations["result"].value_counts(dropna=False).head(30).to_dict(),
        "y_percent_source_counts": observations["y_percent_source"].value_counts(dropna=False).to_dict(),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[biodeg-cas] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CAS-only biodegradation export into long observation tables.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default="benchmarks/biodegradation_homoset_audit/cas_source_biodegradation_csv_v1")
    parser.add_argument("--excelra-csv", default=str(DEFAULT_EXCELRA))
    parser.add_argument("--rifm-xlsx", default=str(DEFAULT_RIFM_2026))
    parser.add_argument("--rifm-2024-xlsx", default=str(DEFAULT_RIFM_2024))
    parser.add_argument("--pubchem", action="store_true")
    parser.add_argument("--pubchem-cache-dir", default="data/biodegradation/pubchem_cas_cache")
    parser.add_argument("--pubchem-delay", type=float, default=0.12)
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run(parse_args())
