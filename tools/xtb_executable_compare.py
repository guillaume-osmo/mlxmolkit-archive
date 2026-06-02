#!/usr/bin/env python3
"""Compare local mlxmolkit results against the public xtb executable.

This is an executable oracle harness: it runs the released ``xtb`` binary in a
temporary directory, parses the Turbomole-style ``*.engrad`` file, and compares
the result with the local Python implementation where available.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_ADDONS_SRC = Path.home() / "Github" / "mlx-addons" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if MLX_ADDONS_SRC.exists() and str(MLX_ADDONS_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_ADDONS_SRC))

from gxtb_oracle_probe import MOLECULES, molecule, write_xyz  # noqa: E402


ANG_TO_BOHR = 1.8897259886
KCAL_PER_HA = 627.5094740631


EXEC_METHOD_ARGS = {
    "gxtb": ("--gxtb",),
    "gfn2": ("--gfn", "2"),
}


@dataclass(frozen=True)
class GradientResult:
    name: str
    energy_hartree: float
    gradient_ha_per_ang: np.ndarray
    wall_time_s: float

    @property
    def grad_max_abs(self) -> float:
        return float(np.max(np.abs(self.gradient_ha_per_ang)))

    @property
    def grad_rms(self) -> float:
        g = self.gradient_ha_per_ang
        return float(np.sqrt(np.mean(g * g)))


def _first_data_line(lines: list[str], start: int) -> str:
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise ValueError("missing data line in engrad file")


def _find_marker(lines: list[str], marker: str) -> int:
    marker_lower = marker.lower()
    for i, line in enumerate(lines):
        if marker_lower in line.lower():
            return i
    raise ValueError(f"missing marker in engrad file: {marker!r}")


def parse_engrad(path: Path) -> tuple[float, np.ndarray]:
    """Parse xtb ``*.engrad`` and return energy plus gradient in Ha/Angstrom."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_idx = _find_marker(lines, "Number of atoms")
    n_atoms = int(_first_data_line(lines, n_idx).split()[0])

    e_idx = _find_marker(lines, "current total energy in Eh")
    energy = float(_first_data_line(lines, e_idx).split()[0])

    g_idx = _find_marker(lines, "current gradient in Eh/bohr")
    values: list[float] = []
    for line in lines[g_idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if values:
                break
            continue
        try:
            values.extend(float(x) for x in stripped.split())
        except ValueError:
            if values:
                break
        if len(values) >= 3 * n_atoms:
            break
    if len(values) < 3 * n_atoms:
        raise ValueError(f"expected {3 * n_atoms} gradient values, got {len(values)}")
    grad_bohr = np.array(values[: 3 * n_atoms], dtype=np.float64).reshape(n_atoms, 3)
    return energy, grad_bohr * ANG_TO_BOHR


def run_xtb_executable(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    xtb: Path,
    method: str,
    acc: float,
    charge: int,
    uhf: int,
) -> GradientResult:
    if method not in EXEC_METHOD_ARGS:
        raise ValueError(f"unsupported executable method: {method!r}")

    with tempfile.TemporaryDirectory(prefix=f"xtb-cmp-{method}-") as tmp:
        cwd = Path(tmp)
        xyz = cwd / "mol.xyz"
        namespace = f"cmp_{method}"
        write_xyz(xyz, atoms, coords_ang)

        env = os.environ.copy()
        libdir = str(xtb.parent.parent / "lib")
        bindir = str(xtb.parent)
        env["DYLD_LIBRARY_PATH"] = f"{libdir}:{bindir}:{env.get('DYLD_LIBRARY_PATH', '')}"

        cmd = [
            str(xtb),
            str(xyz.name),
            *EXEC_METHOD_ARGS[method],
            "--grad",
            "--acc",
            str(acc),
            "--chrg",
            str(charge),
            "--uhf",
            str(uhf),
            "--namespace",
            namespace,
        ]
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)
        wall = time.perf_counter() - start
        log = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"xtb {method} failed with code {proc.returncode}\n{log[-4000:]}")

        engrad_files = sorted(cwd.glob(f"{namespace}*.engrad"))
        if not engrad_files:
            raise RuntimeError(f"xtb {method} did not write an engrad file\n{log[-4000:]}")
        energy, gradient = parse_engrad(engrad_files[0])
        return GradientResult(
            name=f"xtb-{method}",
            energy_hartree=energy,
            gradient_ha_per_ang=gradient,
            wall_time_s=wall,
        )


def run_local_gfn2_fast(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int,
    conv_tol: float,
    max_iter: int,
    fd_workers: int | None,
) -> GradientResult:
    from mlxmolkit.xtb.gradient_gfn2_fast import gfn2_gradient_analytical_fast

    start = time.perf_counter()
    res = gfn2_gradient_analytical_fast(
        atoms,
        coords_ang,
        charge=charge,
        scf_kwargs={"conv_tol": conv_tol, "max_iter": max_iter},
        fd_workers=fd_workers,
    )
    wall = time.perf_counter() - start
    return GradientResult(
        name="mlxmolkit-gfn2-fast",
        energy_hartree=float(res["energy"]),
        gradient_ha_per_ang=np.asarray(res["gradient"], dtype=np.float64),
        wall_time_s=wall,
    )


def run_local_gxtb_reconstructed(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int,
    conv_tol: float,
    max_iter: int,
    use_first_order: bool,
    use_first_order_offsite: bool,
    use_mfx_exchange: bool,
    use_twobody_third_order: bool,
    use_acp_hamiltonian: bool,
    scc_scale: float,
) -> dict[str, object]:
    from mlxmolkit.xtb.scf_gxtb import gxtb_energy

    start = time.perf_counter()
    res = gxtb_energy(
        atoms,
        coords_ang,
        charge=charge,
        conv_tol=conv_tol,
        max_iter=max_iter,
        use_d4srev=False,
        use_pacp=False,
        use_first_order=use_first_order,
        use_first_order_offsite=use_first_order_offsite,
        use_mfx_exchange=use_mfx_exchange,
        use_twobody_third_order=use_twobody_third_order,
        use_acp_hamiltonian=use_acp_hamiltonian,
        scc_scale=scc_scale,
    )
    res["wall_time_s"] = time.perf_counter() - start
    return res


def _format_delta(a: GradientResult, b: GradientResult) -> str:
    de = a.energy_hartree - b.energy_hartree
    dg = a.gradient_ha_per_ang - b.gradient_ha_per_ang
    max_idx = np.unravel_index(np.argmax(np.abs(dg)), dg.shape)
    rms = float(np.sqrt(np.mean(dg * dg)))
    return (
        f"    Delta E ({a.name} - {b.name}): {de:+.12e} Ha"
        f" ({de * KCAL_PER_HA:+.6f} kcal/mol)\n"
        f"    max |Delta g|: {np.max(np.abs(dg)):.6e} Ha/A"
        f" at atom {max_idx[0] + 1} axis {'xyz'[max_idx[1]]}; RMS {rms:.6e}"
    )


def _print_result(result: GradientResult) -> None:
    print(
        f"  {result.name:<20} E={result.energy_hartree: .12f} Ha"
        f"  max|g|={result.grad_max_abs:.6e} Ha/A"
        f"  RMS(g)={result.grad_rms:.6e}"
        f"  wall={result.wall_time_s:.4f}s"
    )


def compare_molecule(
    name: str,
    *,
    xtb: Path,
    methods: list[str],
    local_gxtb: bool,
    local_gfn2: bool,
    acc: float,
    charge: int,
    uhf: int,
    conv_tol: float,
    max_iter: int,
    fd_workers: int | None,
    local_gxtb_first_order: bool,
    local_gxtb_offsite_xvec: bool,
    local_gxtb_mfx: bool,
    local_gxtb_tb3: bool,
    local_gxtb_acp: bool,
    local_gxtb_scc_scale: float,
) -> None:
    atoms, coords = molecule(name)
    print(f"\n{name} ({len(atoms)} atoms)")
    print("-" * (len(name) + 12))

    exec_results: dict[str, GradientResult] = {}
    for method in methods:
        result = run_xtb_executable(
            atoms,
            coords,
            xtb=xtb,
            method=method,
            acc=acc,
            charge=charge,
            uhf=uhf,
        )
        exec_results[method] = result
        _print_result(result)

    if "gxtb" in exec_results and "gfn2" in exec_results:
        print("  executable method difference (raw absolute zeroes are method-specific):")
        print(_format_delta(exec_results["gxtb"], exec_results["gfn2"]))

    if local_gxtb:
        local = run_local_gxtb_reconstructed(
            atoms,
            coords,
            charge=charge,
            conv_tol=conv_tol,
            max_iter=max_iter,
            use_first_order=local_gxtb_first_order,
            use_first_order_offsite=local_gxtb_offsite_xvec,
            use_mfx_exchange=local_gxtb_mfx,
            use_twobody_third_order=local_gxtb_tb3,
            use_acp_hamiltonian=local_gxtb_acp,
            scc_scale=local_gxtb_scc_scale,
        )
        e = float(local["energy_hartree"])
        e_inc = float(local["energy_plus_increment_hartree"])
        print(
            f"  mlxmolkit-gxtb-recon E={e: .12f} Ha"
            f"  E+increment={e_inc: .12f} Ha"
            f"  wall={float(local['wall_time_s']):.4f}s"
            f"  conv={bool(local['converged'])} iter={int(local['n_iter'])}"
            f"  tb1={'on' if local_gxtb_first_order else 'off'}"
            f"  xvec={'on' if local_gxtb_offsite_xvec else 'off'}"
            f"  mfx={'on' if local_gxtb_mfx else 'off'}"
            f"  tb3={'on' if local_gxtb_tb3 else 'off'}"
            f"  acp={'on' if local_gxtb_acp else 'off'}"
            f"  scc_scale={local_gxtb_scc_scale:g}"
        )
        if "gxtb" in exec_results:
            de = e_inc - exec_results["gxtb"].energy_hartree
            print(
                "  local reconstructed g-xTB energy deviation from executable "
                f"(using E+increment): {de:+.12e} Ha ({de * KCAL_PER_HA:+.6f} kcal/mol)"
            )

    if local_gfn2:
        local = run_local_gfn2_fast(
            atoms,
            coords,
            charge=charge,
            conv_tol=conv_tol,
            max_iter=max_iter,
            fd_workers=fd_workers,
        )
        _print_result(local)
        if "gfn2" in exec_results:
            print("  local implementation deviation from xtb GFN2:")
            print(_format_delta(local, exec_results["gfn2"]))
        else:
            print("  local GFN2 ran, but executable GFN2 was not requested.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xtb", type=Path, default=Path("/tmp/gxtb-v2-macos/bin/xtb"))
    parser.add_argument(
        "--molecule",
        choices=[*sorted(MOLECULES), "all"],
        default="water",
        help="molecule to test; 'all' runs water, vanillin, and hedione",
    )
    parser.add_argument(
        "--methods",
        default="gxtb,gfn2",
        help="comma-separated executable methods: gxtb,gfn2",
    )
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0)
    parser.add_argument(
        "--no-local-gfn2",
        action="store_true",
        help="skip the local mlxmolkit GFN2-fast comparison",
    )
    parser.add_argument(
        "--local-gxtb",
        action="store_true",
        help="run the experimental local reconstructed g-xTB energy path",
    )
    parser.add_argument(
        "--local-gxtb-first-order",
        action="store_true",
        help="enable the binary-observed onsite first-order TB term in the local g-xTB path",
    )
    parser.add_argument(
        "--local-gxtb-offsite-xvec",
        action="store_true",
        help="enable the experimental xvec/offsite effective-coulomb scaffold; requires --local-gxtb-first-order",
    )
    parser.add_argument(
        "--local-gxtb-mfx",
        action="store_true",
        help="enable the SI Eq. 153 range-separated Mulliken Fock exchange scaffold",
    )
    parser.add_argument(
        "--local-gxtb-tb3",
        action="store_true",
        help="enable the SI Eq. 129 two-body third-order TB scaffold",
    )
    parser.add_argument(
        "--local-gxtb-acp",
        action="store_true",
        help="enable the SI Eq. 78 reduced ACP projector Hamiltonian scaffold",
    )
    parser.add_argument(
        "--local-gxtb-scc-scale",
        type=float,
        default=1.0,
        help="diagnostic SCC scale for the local reconstructed g-xTB path",
    )
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--fd-workers", type=int, default=None)
    args = parser.parse_args()

    methods = [part.strip().lower() for part in args.methods.split(",") if part.strip()]
    invalid = [method for method in methods if method not in EXEC_METHOD_ARGS]
    if invalid:
        raise SystemExit(f"unsupported methods: {', '.join(invalid)}")

    names = sorted(MOLECULES) if args.molecule == "all" else [args.molecule]
    print(f"xtb executable: {args.xtb}")
    print(f"executable methods: {', '.join(methods)}")
    print(f"local native g-xTB: {'experimental recon energy' if args.local_gxtb else 'no'}")
    print(f"local GFN2-fast: {'no' if args.no_local_gfn2 else 'yes'}")
    for name in names:
        compare_molecule(
            name,
            xtb=args.xtb,
            methods=methods,
            local_gxtb=args.local_gxtb,
            local_gfn2=not args.no_local_gfn2,
            acc=args.acc,
            charge=args.charge,
            uhf=args.uhf,
            conv_tol=args.conv_tol,
            max_iter=args.max_iter,
            fd_workers=args.fd_workers,
            local_gxtb_first_order=args.local_gxtb_first_order,
            local_gxtb_offsite_xvec=args.local_gxtb_offsite_xvec,
            local_gxtb_mfx=args.local_gxtb_mfx,
            local_gxtb_tb3=args.local_gxtb_tb3,
            local_gxtb_acp=args.local_gxtb_acp,
            local_gxtb_scc_scale=args.local_gxtb_scc_scale,
        )


if __name__ == "__main__":
    main()
