#!/usr/bin/env python3
"""Compute openCOSMO-RS activity coefficients for liquid mixtures.

Examples
--------
Activity of a solute at infinite dilution in IPM:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \
    ~/miniconda3/envs/osmo/bin/python3 tools/cosmors_mixture_activity.py \
      --smiles-component solute='CC(=O)NC1=CC=C(C=C1)O' \
      --smiles-component IPM='CCCCCCCCCCCCCC(=O)OC(C)C' \
      --x 1e-8,0.99999999

Solubility in an IPM/ethanol 70/30 solvent mixture:

    ... tools/cosmors_mixture_activity.py \
      --smiles-component solute='CC(=O)NC1=CC=C(C=C1)O' \
      --smiles-component IPM='CCCCCCCCCCCCCC(=O)OC(C)C' \
      --smiles-component ethanol='CCO' \
      --solubility --solvent-x 0.7,0.3 \
      --delta-h-fusion 27000 --t-fusion 442.0
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from mlxmolkit.xtb.cosmors_activity import (
    WALDEN_DELTA_S_FUSION_J_MOL_K,
    activity_coefficients,
    estimate_delta_h_fusion_walden,
    solubility_in_solvent_mixture,
)


def _parse_component(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("component must be LABEL=PATH_OR_SMILES")
    label, value = spec.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise argparse.ArgumentTypeError("component must be LABEL=PATH_OR_SMILES")
    return label, value


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return vals


def _generate_cosmo(label: str, smiles: str, out_dir: Path, *, seed: int, solvent: str, acc: float) -> Path:
    from mlxmolkit.xtb import hybrid_gxtb_gfn2_cosmo_from_smiles, write_cosmo_file

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    path = out_dir / f"{safe}.cosmo"
    if path.exists():
        return path
    res = hybrid_gxtb_gfn2_cosmo_from_smiles(smiles, seed=seed, solvent=solvent, acc=acc)
    write_cosmo_file(res["cosmo"], path)
    return path


def _prepare_components(args: argparse.Namespace, work_dir: Path) -> tuple[list[str], list[Path]]:
    labels: list[str] = []
    paths: list[Path] = []

    for spec in args.component:
        label, value = _parse_component(spec)
        labels.append(label)
        paths.append(Path(value))

    for spec in args.smiles_component:
        label, smiles = _parse_component(spec)
        labels.append(label)
        paths.append(_generate_cosmo(label, smiles, work_dir, seed=args.seed, solvent=args.cosmo_solvent, acc=args.acc))

    if not paths:
        raise SystemExit("provide at least one --component or --smiles-component")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return labels, paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", action="append", default=[], help="LABEL=/path/to/component.cosmo or .orcacosmo")
    parser.add_argument("--smiles-component", action="append", default=[], help="LABEL=SMILES; generates a GFN2 tmCOSMO .cosmo")
    parser.add_argument("--x", type=_parse_float_list, help="comma-separated mole fractions for direct activity calculation")
    parser.add_argument("--temperature", "--T", type=float, default=298.15)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--cosmo-solvent", default="inf", help="solvent keyword for generated xtb tmCOSMO files")

    parser.add_argument("--solubility", action="store_true", help="solve solute solubility in a fixed solvent mixture")
    parser.add_argument("--solvent-x", type=_parse_float_list, help="solvent ratio, excluding component 0 solute")
    parser.add_argument("--delta-h-fusion", type=float, help="solute fusion enthalpy in J/mol")
    parser.add_argument(
        "--estimate-delta-h-fusion",
        choices=["walden"],
        help="estimate missing fusion enthalpy from T_fusion; explicit approximation for screening only",
    )
    parser.add_argument(
        "--delta-s-fusion",
        type=float,
        default=WALDEN_DELTA_S_FUSION_J_MOL_K,
        help="fusion entropy J/mol/K used by --estimate-delta-h-fusion walden",
    )
    parser.add_argument("--t-fusion", type=float, help="solute fusion temperature in K")
    parser.add_argument("--mp-c", type=float, help="solute melting point in degC; converted to T_fUSION K")
    parser.add_argument("--noniterative", action="store_true", help="use infinite-dilution solubility approximation")
    args = parser.parse_args()

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
        work_dir = args.work_dir
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="cosmors_mix_")
        work_dir = Path(tmp_ctx.name)

    try:
        t0 = time.perf_counter()
        labels, paths = _prepare_components(args, work_dir)

        if args.solubility:
            t_fusion = args.t_fusion
            if t_fusion is None and args.mp_c is not None:
                t_fusion = float(args.mp_c) + 273.15
            if t_fusion is None:
                raise SystemExit("--solubility requires --t-fusion K or --mp-c degC")
            delta_h_fusion = args.delta_h_fusion
            delta_h_source = "input"
            if delta_h_fusion is None:
                if args.estimate_delta_h_fusion != "walden":
                    raise SystemExit(
                        "--solubility requires --delta-h-fusion in J/mol, or "
                        "--estimate-delta-h-fusion walden with --t-fusion/--mp-c"
                    )
                delta_h_fusion = estimate_delta_h_fusion_walden(
                    t_fusion,
                    delta_s_fusion_J_mol_K=float(args.delta_s_fusion),
                )
                delta_h_source = "walden_estimate"
            solvent_x = args.solvent_x or [1.0] * (len(paths) - 1)
            result = solubility_in_solvent_mixture(
                paths,
                solvent_x,
                delta_h_fus_J_mol=float(delta_h_fusion),
                T_fus_K=float(t_fusion),
                T=float(args.temperature),
                labels=labels,
                iterative=not args.noniterative,
            )
            payload = {
                "mode": "solubility",
                "labels": list(result["labels"]),
                "component_paths": [str(p) for p in paths],
                "temperature_K": float(args.temperature),
                "delta_h_fusion_J_mol": float(delta_h_fusion),
                "delta_h_fusion_source": delta_h_source,
                "delta_s_fusion_J_mol_K": float(args.delta_s_fusion) if delta_h_source == "walden_estimate" else None,
                "t_fusion_K": float(t_fusion),
                "solvent_x": list(np.asarray(solvent_x, dtype=float) / np.sum(solvent_x)),
                "solubility_x": float(result["solubility_x"]),
                "composition": np.asarray(result["composition"], dtype=float).tolist(),
                "ln_gamma_solute": float(result["ln_gamma_solute"]),
                "ln_x_ideal": float(result["ln_x_ideal"]),
                "iterative": bool(result["iterative"]),
                "wall_s": time.perf_counter() - t0,
            }
        else:
            if args.x is None:
                raise SystemExit("direct activity calculation requires --x")
            result = activity_coefficients(paths, args.x, labels=labels, T=float(args.temperature))
            payload = {
                "mode": "activity",
                "labels": list(result.labels),
                "component_paths": [str(p) for p in paths],
                "temperature_K": float(result.T),
                "x": result.x.tolist(),
                "ln_gamma": result.ln_gamma.tolist(),
                "gamma": result.gamma.tolist(),
                "wall_s": time.perf_counter() - t0,
            }

        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(text + "\n")
        return 0
    finally:
        if tmp_ctx is not None and args.keep_work_dir:
            print(f"kept work dir: {work_dir}")
        elif tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
