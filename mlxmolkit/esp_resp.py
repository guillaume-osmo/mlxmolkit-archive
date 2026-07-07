"""ESP and RESP partial-charge fitting with an MLX linear algebra core.

This module is intentionally backend-neutral: it fits charges from any quantum
electrostatic-potential grid and provides convenience adapters for common open
routes such as xTB ``--esp`` grids and the local PM6/RM1 SCF stack.

Coordinates are treated consistently in whatever length unit the supplied grid
uses. The fitted charges are invariant to a global Coulomb-unit scale factor, so
the default Angstrom convention is suitable for ML labels and descriptor work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
from typing import Iterable, Literal, Sequence

import mlx.core as mx
import numpy as np


# Bondi-like van der Waals radii in Angstrom for RESP/MK surface generation.
VDW_RADII = {
    1: 1.20,
    2: 1.40,
    3: 1.82,
    4: 1.53,
    5: 1.92,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    10: 1.54,
    11: 2.27,
    12: 1.73,
    13: 1.84,
    14: 2.10,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    18: 1.88,
    35: 1.85,
    53: 1.98,
}


@dataclass(frozen=True)
class EspRespFitResult:
    """Result from an ESP/RESP charge fit."""

    charges: np.ndarray
    method: str
    rmse: float
    n_grid: int
    total_charge: float
    n_iter: int = 1
    converged: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


def connolly_surface_grid(
    atoms: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
    *,
    shell_factors: Sequence[float] = (1.4, 1.6, 1.8, 2.0),
    point_density: float = 1.0,
    min_points_per_shell: int = 32,
    max_points_per_shell: int = 256,
    prune_scale: float = 1.0,
) -> np.ndarray:
    """Generate a simple Connolly/MK-style molecular surface grid.

    Points are placed on Fibonacci spheres around each atom at multiple scaled
    vdW radii, then pruned if they fall inside another atom's vdW envelope.
    """

    atom_array = np.asarray(atoms, dtype=np.int64)
    coord_array = _as_coords(coords)
    if atom_array.shape != (coord_array.shape[0],):
        raise ValueError("atoms and coords must have the same length")
    if point_density <= 0:
        raise ValueError("point_density must be positive")

    radii = np.asarray([VDW_RADII.get(int(z), 1.80) for z in atom_array], dtype=np.float64)
    points = []
    for atom_idx, center in enumerate(coord_array):
        base_radius = radii[atom_idx]
        for factor in shell_factors:
            shell_radius = float(base_radius * factor)
            n_points = int(math.ceil(4.0 * math.pi * shell_radius * shell_radius * point_density))
            n_points = max(min_points_per_shell, min(max_points_per_shell, n_points))
            shell = center[None, :] + shell_radius * _fibonacci_sphere(n_points)
            keep = np.ones(n_points, dtype=bool)
            for other_idx, other_center in enumerate(coord_array):
                if other_idx == atom_idx:
                    continue
                cutoff = radii[other_idx] * prune_scale
                dist = np.linalg.norm(shell - other_center[None, :], axis=1)
                keep &= dist >= cutoff
            points.append(shell[keep])

    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(points).astype(np.float64, copy=False)


def coulomb_potential_from_charges_mlx(
    atom_coords: Sequence[Sequence[float]] | np.ndarray,
    charges: Sequence[float] | np.ndarray,
    grid_coords: Sequence[Sequence[float]] | np.ndarray,
    *,
    min_distance: float = 1.0e-8,
) -> np.ndarray:
    """Evaluate point-charge ESP values on a grid using MLX."""

    coords = mx.array(_as_coords(atom_coords), dtype=mx.float32)
    charge_array = mx.array(np.asarray(charges, dtype=np.float32))
    grid = mx.array(_as_coords(grid_coords), dtype=mx.float32)
    delta = grid[:, None, :] - coords[None, :, :]
    dist = mx.sqrt(mx.sum(delta * delta, axis=-1) + float(min_distance) ** 2)
    values = mx.sum(charge_array[None, :] / dist, axis=1)
    return np.asarray(values, dtype=np.float64)


def fit_esp_charges_mlx(
    atom_coords: Sequence[Sequence[float]] | np.ndarray,
    grid_coords: Sequence[Sequence[float]] | np.ndarray,
    esp_values: Sequence[float] | np.ndarray,
    *,
    total_charge: float = 0.0,
    equivalence_groups: Sequence[Sequence[int]] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    ridge: float = 0.0,
) -> EspRespFitResult:
    """Fit atom-centered ESP charges with a total-charge constraint."""

    coords = _as_coords(atom_coords)
    grid = _as_coords(grid_coords)
    values = np.asarray(esp_values, dtype=np.float64)
    if values.shape != (grid.shape[0],):
        raise ValueError(f"esp_values shape {values.shape} does not match grid {grid.shape}")
    if grid.shape[0] == 0:
        raise ValueError("ESP fit requires at least one grid point")

    design = _coulomb_design_matrix(coords, grid)
    charges = _solve_constrained_lstsq_mlx(
        design,
        values,
        total_charge=total_charge,
        equivalence_groups=equivalence_groups,
        weights=weights,
        ridge=ridge,
    )
    rmse = _esp_rmse(design, charges, values, weights)
    return EspRespFitResult(
        charges=charges,
        method="esp",
        rmse=rmse,
        n_grid=int(grid.shape[0]),
        total_charge=float(total_charge),
    )


def fit_resp_charges_mlx(
    atom_coords: Sequence[Sequence[float]] | np.ndarray,
    grid_coords: Sequence[Sequence[float]] | np.ndarray,
    esp_values: Sequence[float] | np.ndarray,
    *,
    total_charge: float = 0.0,
    equivalence_groups: Sequence[Sequence[int]] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    restraint_a: float = 5.0e-4,
    restraint_b: float = 0.1,
    restrain_hydrogens: bool = False,
    atoms: Sequence[int] | None = None,
    max_iter: int = 50,
    conv_tol: float = 1.0e-8,
) -> EspRespFitResult:
    """Fit RESP charges with the standard iterative hyperbolic restraint.

    By default hydrogens are not restrained, following the common RESP setup.
    """

    coords = _as_coords(atom_coords)
    grid = _as_coords(grid_coords)
    values = np.asarray(esp_values, dtype=np.float64)
    if values.shape != (grid.shape[0],):
        raise ValueError(f"esp_values shape {values.shape} does not match grid {grid.shape}")
    if restraint_a < 0 or restraint_b <= 0:
        raise ValueError("RESP restraint_a must be non-negative and restraint_b positive")

    design = _coulomb_design_matrix(coords, grid)
    if atoms is None:
        restrain_mask = np.ones(coords.shape[0], dtype=bool)
    else:
        atom_array = np.asarray(atoms, dtype=np.int64)
        if atom_array.shape != (coords.shape[0],):
            raise ValueError("atoms and atom_coords must have the same length")
        restrain_mask = atom_array != 1 if not restrain_hydrogens else np.ones_like(atom_array, dtype=bool)

    charges = _solve_constrained_lstsq_mlx(
        design,
        values,
        total_charge=total_charge,
        equivalence_groups=equivalence_groups,
        weights=weights,
    )
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        diagonal = np.zeros(coords.shape[0], dtype=np.float64)
        diagonal[restrain_mask] = restraint_a / np.sqrt(charges[restrain_mask] ** 2 + restraint_b ** 2)
        next_charges = _solve_constrained_lstsq_mlx(
            design,
            values,
            total_charge=total_charge,
            equivalence_groups=equivalence_groups,
            weights=weights,
            atom_diagonal=diagonal,
        )
        delta = float(np.max(np.abs(next_charges - charges)))
        charges = next_charges
        if delta < conv_tol:
            converged = True
            break

    rmse = _esp_rmse(design, charges, values, weights)
    return EspRespFitResult(
        charges=charges,
        method="resp",
        rmse=rmse,
        n_grid=int(grid.shape[0]),
        total_charge=float(total_charge),
        n_iter=n_iter,
        converged=converged,
        metadata={
            "restraint_a": float(restraint_a),
            "restraint_b": float(restraint_b),
            "restrain_hydrogens": bool(restrain_hydrogens),
        },
    )


def fit_esp_resp_charges_mlx(
    atoms: Sequence[int],
    atom_coords: Sequence[Sequence[float]] | np.ndarray,
    grid_coords: Sequence[Sequence[float]] | np.ndarray,
    esp_values: Sequence[float] | np.ndarray,
    *,
    total_charge: float = 0.0,
    equivalence_groups: Sequence[Sequence[int]] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    restraint_a: float = 5.0e-4,
    restraint_b: float = 0.1,
) -> tuple[EspRespFitResult, EspRespFitResult]:
    """Fit both ESP and RESP charges from a quantum ESP grid."""

    esp = fit_esp_charges_mlx(
        atom_coords,
        grid_coords,
        esp_values,
        total_charge=total_charge,
        equivalence_groups=equivalence_groups,
        weights=weights,
    )
    resp = fit_resp_charges_mlx(
        atom_coords,
        grid_coords,
        esp_values,
        total_charge=total_charge,
        equivalence_groups=equivalence_groups,
        weights=weights,
        restraint_a=restraint_a,
        restraint_b=restraint_b,
        atoms=atoms,
    )
    return esp, resp


def pm6_esp_resp_charge_labels(
    atoms: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
    *,
    total_charge: float = 0.0,
    method: str = "PM6",
    grid_coords: np.ndarray | None = None,
    esp_values: np.ndarray | None = None,
    shell_factors: Sequence[float] = (1.4, 1.6, 1.8, 2.0),
    point_density: float = 1.0,
) -> tuple[EspRespFitResult, EspRespFitResult]:
    """Generate PM6-backed ESP/RESP charge labels.

    If ``grid_coords`` and ``esp_values`` are provided, they are treated as the
    quantum ESP to fit. Without an explicit grid, the current PM6/RM1 SCF
    implementation supplies Mulliken-like atom charges; this function evaluates
    their point-charge potential on a Connolly grid and applies ESP/RESP fitting
    as a transparent PM6 proxy label.
    """

    atom_array = np.asarray(atoms, dtype=np.int64)
    coord_array = _as_coords(coords)
    if grid_coords is None or esp_values is None:
        from mlxmolkit.rm1 import nddo_energy

        scf = nddo_energy(
            atom_array.tolist(),
            coord_array,
            method=method,
            molecular_charge=total_charge,
        )
        if not scf.get("converged", False):
            raise RuntimeError(f"{method} SCF did not converge")
        pm6_charges = np.asarray(scf["charges"], dtype=np.float64)
        grid_coords = connolly_surface_grid(
            atom_array,
            coord_array,
            shell_factors=shell_factors,
            point_density=point_density,
        )
        esp_values = coulomb_potential_from_charges_mlx(coord_array, pm6_charges, grid_coords)
        source = f"{method.lower()}_mulliken_esp_proxy"
    else:
        source = f"{method.lower()}_external_esp_grid"

    esp, resp = fit_esp_resp_charges_mlx(
        atom_array,
        coord_array,
        grid_coords,
        esp_values,
        total_charge=total_charge,
    )
    esp.metadata.update({"source": source, "semiempirical_method": method})
    resp.metadata.update({"source": source, "semiempirical_method": method})
    return esp, resp


def parse_esp_grid_file(
    path: str | Path,
    *,
    layout: Literal["xyzv", "vxyz", "auto"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a whitespace ESP grid file into ``(grid_coords, esp_values)``.

    ``layout="vxyz"`` matches the xTB ``--esp`` DAT convention seen in public
    examples. ``layout="xyzv"`` matches many QM grid exports.
    """

    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        floats = []
        for part in parts:
            try:
                floats.append(float(part))
            except ValueError:
                pass
        if len(floats) >= 4:
            rows.append(floats[:4])
    if not rows:
        raise ValueError(f"no ESP grid rows with four numeric columns found in {path}")

    data = np.asarray(rows, dtype=np.float64)
    if layout == "vxyz":
        return data[:, 1:4], data[:, 0]
    if layout == "xyzv":
        return data[:, 0:3], data[:, 3]

    # xTB ESP values are usually small while coordinates span molecular size;
    # this heuristic picks vxyz when the first column has the smaller spread.
    first_span = float(np.ptp(data[:, 0]))
    last_span = float(np.ptp(data[:, 3]))
    if first_span < last_span:
        return data[:, 1:4], data[:, 0]
    return data[:, 0:3], data[:, 3]


def run_xtb_esp_grid(
    atoms: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
    *,
    total_charge: int = 0,
    uhf: int = 0,
    gfn: int = 2,
    xtb_binary: str = "xtb",
    workdir: str | Path | None = None,
    keep_workdir: bool = False,
    timeout: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run open-source xTB ``--esp`` and parse its ESP grid.

    The local environment must provide a working ``xtb`` binary. The command is
    intentionally thin so callers can replace it with PM6/MOPAC/SQM outputs and
    still reuse the same MLX ESP/RESP fitter.
    """

    binary = shutil.which(xtb_binary)
    if binary is None:
        raise FileNotFoundError(f"could not find xTB binary {xtb_binary!r}")

    tmp_ctx = None
    if workdir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="mlxmolkit_xtb_esp_")
        run_dir = Path(tmp_ctx.name)
    else:
        run_dir = Path(workdir)
        run_dir.mkdir(parents=True, exist_ok=True)

    try:
        xyz_path = run_dir / "mol.xyz"
        write_xyz(xyz_path, atoms, coords)
        cmd = [
            binary,
            str(xyz_path),
            "--gfn",
            str(int(gfn)),
            "--esp",
            "--chrg",
            str(int(total_charge)),
        ]
        if uhf:
            cmd.extend(["--uhf", str(int(uhf))])
        subprocess.run(cmd, cwd=run_dir, check=True, capture_output=True, text=True, timeout=timeout)

        candidates = [
            run_dir / "xtb_esp.dat",
            run_dir / "xtb_esp.grid",
            run_dir / "esp.dat",
        ]
        candidates.extend(sorted(run_dir.glob("*esp*.dat")))
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 0:
                return parse_esp_grid_file(candidate, layout="vxyz")
        raise FileNotFoundError(f"xTB finished but no ESP DAT file was found in {run_dir}")
    finally:
        if tmp_ctx is not None and not keep_workdir:
            tmp_ctx.cleanup()


def write_xyz(
    path: str | Path,
    atoms: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
) -> None:
    """Write an XYZ file from atomic numbers and coordinates."""

    coord_array = _as_coords(coords)
    atom_array = np.asarray(atoms, dtype=np.int64)
    if atom_array.shape != (coord_array.shape[0],):
        raise ValueError("atoms and coords must have the same length")
    lines = [str(len(atom_array)), "generated by mlxmolkit.esp_resp"]
    for z, xyz in zip(atom_array, coord_array, strict=True):
        lines.append(f"{_atomic_symbol(int(z))} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}")
    Path(path).write_text("\n".join(lines) + "\n")


def _solve_constrained_lstsq_mlx(
    design: np.ndarray,
    values: np.ndarray,
    *,
    total_charge: float,
    equivalence_groups: Sequence[Sequence[int]] | None,
    weights: Sequence[float] | np.ndarray | None,
    ridge: float = 0.0,
    atom_diagonal: np.ndarray | None = None,
) -> np.ndarray:
    n_atoms = design.shape[1]
    transform = _equivalence_transform(n_atoms, equivalence_groups)
    a = design @ transform
    y = values.astype(np.float64, copy=False)

    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))
        if w.shape != (design.shape[0],):
            raise ValueError("weights must have one value per grid point")
        a = a * w[:, None]
        y = y * w

    h = a.T @ a
    rhs = a.T @ y
    if ridge:
        h = h + float(ridge) * np.eye(h.shape[0], dtype=np.float64)
    if atom_diagonal is not None:
        diag = np.asarray(atom_diagonal, dtype=np.float64)
        if diag.shape != (n_atoms,):
            raise ValueError("atom_diagonal must have one value per atom")
        h = h + transform.T @ np.diag(diag) @ transform

    constraint = np.sum(transform, axis=0, keepdims=True)
    kkt = np.block(
        [
            [h, constraint.T],
            [constraint, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    rhs_kkt = np.concatenate([rhs, np.asarray([total_charge], dtype=np.float64)])
    solution = mx.linalg.solve(
        mx.array(kkt, dtype=mx.float32),
        mx.array(rhs_kkt, dtype=mx.float32),
        stream=mx.cpu,
    )
    variables = np.asarray(solution, dtype=np.float64)[: transform.shape[1]]
    charges = transform @ variables
    return charges.astype(np.float64, copy=False)


def _coulomb_design_matrix(atom_coords: np.ndarray, grid_coords: np.ndarray) -> np.ndarray:
    delta = grid_coords[:, None, :] - atom_coords[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    return 1.0 / np.maximum(dist, 1.0e-8)


def _esp_rmse(
    design: np.ndarray,
    charges: np.ndarray,
    values: np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
) -> float:
    residual = design @ charges - values
    if weights is not None:
        weight_array = np.asarray(weights, dtype=np.float64)
        return float(np.sqrt(np.average(residual * residual, weights=weight_array)))
    return float(np.sqrt(np.mean(residual * residual)))


def _equivalence_transform(
    n_atoms: int,
    equivalence_groups: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    labels = list(range(n_atoms))
    if equivalence_groups:
        for group in equivalence_groups:
            group = tuple(int(i) for i in group)
            if not group:
                continue
            root = group[0]
            for idx in group:
                if idx < 0 or idx >= n_atoms:
                    raise ValueError(f"equivalence atom index {idx} out of range")
                labels[idx] = root
    unique = {}
    for label in labels:
        unique.setdefault(label, len(unique))
    transform = np.zeros((n_atoms, len(unique)), dtype=np.float64)
    for atom_idx, label in enumerate(labels):
        transform[atom_idx, unique[label]] = 1.0
    return transform


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    idx = np.arange(n_points, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (idx + 0.5) / n_points
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * idx
    return np.column_stack((np.cos(theta) * radius, np.sin(theta) * radius, z))


def _as_coords(coords: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"expected coordinates with shape (n, 3), got {out.shape}")
    return out


def _atomic_symbol(atomic_number: int) -> str:
    symbols = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        35: "Br",
        53: "I",
    }
    return symbols.get(atomic_number, f"X{atomic_number}")
