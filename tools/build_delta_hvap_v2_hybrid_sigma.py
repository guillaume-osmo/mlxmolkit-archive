#!/usr/bin/env python3
"""Build a row-aligned hybrid sigma NPZ for deltaHvapv2.

The usual use is:

* base: complete degraded g-xTB/GFN2-tmCOSMO sigma for all homoset rows
* override: ORCA/COSMORS sigma generated for calcphyschemprop-only rows

The output keeps the base rows everywhere except the override selected rows,
where valid override sigma/profile arrays replace the base features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNION = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_homoset_union.csv"
DEFAULT_BASE = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_gxtb_tmcosmo_all_sigma.npz"
DEFAULT_OVERRIDE = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_orca_cosmors_calc_only_sigma.npz"
DEFAULT_OUT = REPO_ROOT / "data/delta_hvap_v2/deltaHvapv2_hybrid_gxtb_trusted_orca_calc_only_sigma.npz"


def scalar_text(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def assert_row_alignment(name: str, z: np.lib.npyio.NpzFile, expected: np.ndarray) -> None:
    canonical = np.asarray(z["canonical_smiles"]).astype(str)
    if canonical.shape[0] != expected.shape[0] or not np.array_equal(canonical, expected):
        raise ValueError(
            f"{name} is not row-aligned to the homoset union: "
            f"npz_rows={canonical.shape[0]} expected_rows={expected.shape[0]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--base-npz", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--override-npz", type=Path, default=DEFAULT_OVERRIDE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use base rows where the override selected rows are not yet valid.",
    )
    args = parser.parse_args()

    union = pd.read_csv(args.union)
    expected = union["canonical_smiles"].astype(str).to_numpy()
    base = np.load(args.base_npz, allow_pickle=True)
    override = np.load(args.override_npz, allow_pickle=True)
    assert_row_alignment("base", base, expected)
    assert_row_alignment("override", override, expected)

    base_valid = np.asarray(base["valid_mask"], dtype=bool)
    override_valid = np.asarray(override["valid_mask"], dtype=bool)
    override_selected = np.asarray(override["selected_mask"], dtype=bool)
    if not np.all(base_valid):
        raise ValueError(f"base has invalid rows: {int(base_valid.sum())}/{len(base_valid)}")
    missing_override = override_selected & ~override_valid
    if np.any(missing_override) and not args.allow_partial:
        raise ValueError(
            "override has invalid selected rows: "
            f"valid_selected={int(np.sum(override_valid & override_selected))}/"
            f"{int(np.sum(override_selected))}. Re-run with --allow-partial for a degraded smoke merge."
        )

    replace = override_selected & override_valid
    out: dict[str, np.ndarray] = {}
    for key in base.files:
        out[key] = np.asarray(base[key]).copy()

    for key in (
        "mu_J_per_mol",
        "profile_area_A2",
        "area_A2",
        "volume_A3",
        "n_atoms",
        "n_segments",
        "wall_s",
        "cache_path",
        "error",
    ):
        if key in out and key in override.files:
            arr = np.asarray(out[key]).copy()
            arr[replace] = np.asarray(override[key])[replace]
            out[key] = arr

    out["method"] = np.asarray("hybrid_gxtb_base_orca_override")
    out["subset"] = np.asarray("all")
    out["base_npz"] = np.asarray(str(args.base_npz))
    out["override_npz"] = np.asarray(str(args.override_npz))
    out["override_selected_mask"] = override_selected
    out["override_replaced_mask"] = replace
    out["source_method"] = np.where(replace, "orca_cosmors_water", "gxtb_tmcosmo_water").astype(object)
    out["valid_mask"] = base_valid.copy()
    out["valid_mask"][override_selected] = override_valid[override_selected] if not args.allow_partial else True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)

    summary = {
        "out": str(args.out),
        "base_npz": str(args.base_npz),
        "override_npz": str(args.override_npz),
        "rows": int(len(expected)),
        "override_selected_rows": int(np.sum(override_selected)),
        "override_valid_selected_rows": int(np.sum(override_valid & override_selected)),
        "override_replaced_rows": int(np.sum(replace)),
        "allow_partial": bool(args.allow_partial),
        "method": scalar_text(out["method"]),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
