"""Report on the openMOPAC vs mlxmolkit parity run — task #18.

Reads tests/data/mopac_parity_200.json and answers three questions the raw
numbers do not: does mlxmolkit reproduce MOPAC's PM6, does it do so on the
d-orbital elements specifically, and where it disagrees, is that a bug or the
documented PM6 vs PM6_D difference.

Deliberately reports the *distribution*, not just a mean. A benchmark that
quotes one average hides the handful of molecules that are actually wrong,
which are the only ones worth looking at.

Run:  PYTHONPATH=. python tools/report_mopac_parity.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("tests/data/mopac_parity_200.json")


def quantiles(x, qs=(50, 90, 95, 100)):
    x = np.asarray(x, dtype=float)
    return {q: float(np.percentile(x, q)) for q in qs}


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    records = json.loads(DATA.read_text())
    print(f"{len(records)} molecules with a MOPAC PM6 reference at a frozen geometry")

    for group in ("perfumery", "coverage", None):
        sel = [r for r in records if group is None or r["group"] == group]
        if not sel:
            continue
        label = group or "ALL"
        section(f"{label}  ({len(sel)} molecules, "
                f"{sum(r['has_d'] for r in sel)} with P/S/Cl/Br/I)")

        for method in ("PM6", "PM6_D"):
            key = f"hf_{method}"
            have = [r for r in sel if key in r]
            if not have:
                print(f"  {method}: no results")
                continue
            err = np.array([abs(r[key] - r["hf_mopac"]) for r in have])
            conv = sum(r.get(f"conv_{method}", False) for r in have)
            q = quantiles(err)
            print(f"  {method:6s} n={len(have):3d}  converged {conv}/{len(have)}  "
                  f"|dHf| kcal/mol  median {q[50]:7.3f}  p90 {q[90]:8.3f}  "
                  f"p95 {q[95]:8.3f}  max {q[100]:9.3f}")

            dq = [r[f"dq_{method}"] for r in have if f"dq_{method}" in r]
            if dq:
                qq = quantiles(dq)
                print(f"  {'':6s} charges n={len(dq):3d}                  "
                      f"|dq| e         median {qq[50]:7.4f}  p90 {qq[90]:8.4f}  "
                      f"p95 {qq[95]:8.4f}  max {qq[100]:9.4f}")

    # Where it disagrees, and whether d atoms are over-represented there
    section("Worst 12 by |dHf| (PM6)")
    have = [r for r in records if "hf_PM6" in r]
    have.sort(key=lambda r: -abs(r["hf_PM6"] - r["hf_mopac"]))
    print(f"  {'smiles':34s} {'d?':>3s} {'at':>3s} {'MOPAC':>10s} {'PM6':>10s} "
          f"{'dHf':>9s} {'dq':>7s}  classes")
    for r in have[:12]:
        d = abs(r["hf_PM6"] - r["hf_mopac"])
        print(f"  {r['smiles'][:34]:34s} {'y' if r['has_d'] else '.':>3s} "
              f"{r['n_atoms']:3d} {r['hf_mopac']:10.3f} {r['hf_PM6']:10.3f} "
              f"{d:9.3f} {r.get('dq_PM6', float('nan')):7.4f}  {r['classes'][:38]}")

    n_bad = sum(1 for r in have if abs(r["hf_PM6"] - r["hf_mopac"]) > 1.0)
    bad_d = sum(1 for r in have
                if abs(r["hf_PM6"] - r["hf_mopac"]) > 1.0 and r["has_d"])
    all_d = sum(1 for r in have if r["has_d"])
    section("Is the d path over-represented among the failures?")
    print(f"  |dHf| > 1 kcal/mol: {n_bad}/{len(have)} molecules "
          f"({100 * n_bad / len(have):.1f}%)")
    print(f"    of which carry P/S/Cl/Br/I: {bad_d}/{n_bad}"
          + (f"  ({100 * bad_d / n_bad:.0f}%)" if n_bad else ""))
    print(f"    base rate of d molecules:   {all_d}/{len(have)} "
          f"({100 * all_d / len(have):.0f}%)")

    section("Class breakdown of the failures")
    cls = Counter(c for r in have if abs(r["hf_PM6"] - r["hf_mopac"]) > 1.0
                  for c in r["classes"].split("|"))
    base = Counter(c for r in have for c in r["classes"].split("|"))
    print(f"  {'class':16s} {'failing':>8s} {'total':>6s} {'rate':>7s}")
    for c, n in cls.most_common(12):
        print(f"  {c:16s} {n:8d} {base[c]:6d} {100 * n / base[c]:6.0f}%")

    section("Speed")
    for method in ("PM6", "PM6_D"):
        t = [r[f"t_{method}"] for r in records if f"t_{method}" in r]
        if t:
            print(f"  {method:6s} median {np.median(t) * 1e3:7.1f} ms/molecule   "
                  f"total {sum(t):6.1f} s for {len(t)} molecules")

    errs = [(m, r) for r in records for m in ("PM6", "PM6_D")
            if f"err_{m}" in r]
    if errs:
        section(f"Errors ({len(errs)})")
        for m, r in errs[:10]:
            print(f"  {m:6s} {r['smiles'][:40]:40s} {r[f'err_{m}'][:60]}")


if __name__ == "__main__":
    main()
