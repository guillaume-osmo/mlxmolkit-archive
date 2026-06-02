#!/usr/bin/env python3
"""Archive ORCA ``.orcacosmo`` files safely.

Two archive forms are useful:

1. ``zip`` keeps the original files byte-for-byte.
2. ``npz`` stores parsed atom/segment arrays as ragged matrices with offsets,
   optionally embedding the raw file bytes for exact reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def iter_orcacosmo_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(Path(input_dir).glob("*.orcacosmo"))
    if limit is not None:
        files = files[: max(0, int(limit))]
    return files


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack_zip(files: list[Path], out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, arcname=path.name)


def pack_npz(files: list[Path], out_npz: Path, *, include_text: bool) -> dict[str, object]:
    from mlxmolkit.xtb.cosmo_sigma import cosmosegments_from_orcacosmo

    filenames: list[str] = []
    sha256: list[str] = []

    epsilon: list[float] = []
    fepsi: list[float] = []
    area: list[float] = []
    volume: list[float] = []
    total_screening_charge: list[float] = []
    total_energy_hartree: list[float] = []
    dielectric_energy_hartree: list[float] = []

    atom_offsets = [0]
    atom_radii: list[np.ndarray] = []
    atom_coords_bohr: list[np.ndarray] = []
    atom_z: list[np.ndarray] = []

    segment_offsets = [0]
    segments_atom: list[np.ndarray] = []
    segments_xyz_bohr: list[np.ndarray] = []
    segments_charge: list[np.ndarray] = []
    segments_area: list[np.ndarray] = []
    segments_sigma: list[np.ndarray] = []
    segments_potential: list[np.ndarray] = []

    text_offsets = [0]
    text_chunks: list[np.ndarray] = []

    t0 = time.perf_counter()
    for path in files:
        raw = path.read_bytes()
        cs = cosmosegments_from_orcacosmo(path)

        filenames.append(path.name)
        sha256.append(sha256_bytes(raw))
        epsilon.append(float(cs.epsilon))
        fepsi.append(float(cs.fepsi))
        area.append(float(cs.area))
        volume.append(float(cs.volume))
        total_screening_charge.append(float(cs.total_screening_charge))
        total_energy_hartree.append(float(cs.total_energy_hartree))
        dielectric_energy_hartree.append(float(cs.dielectric_energy_hartree))

        atom_radii.append(np.asarray(cs.atom_radii, dtype=np.float64))
        atom_coords_bohr.append(np.asarray(cs.atom_coords_bohr, dtype=np.float64))
        atom_z.append(np.asarray(cs.atom_z, dtype=np.int16))
        atom_offsets.append(atom_offsets[-1] + len(cs.atom_z))

        segments_atom.append(np.asarray(cs.segments_atom, dtype=np.int32))
        segments_xyz_bohr.append(np.asarray(cs.segments_xyz_bohr, dtype=np.float64))
        segments_charge.append(np.asarray(cs.segments_charge, dtype=np.float64))
        segments_area.append(np.asarray(cs.segments_area, dtype=np.float64))
        segments_sigma.append(np.asarray(cs.segments_sigma, dtype=np.float64))
        segments_potential.append(np.asarray(cs.segments_potential, dtype=np.float64))
        segment_offsets.append(segment_offsets[-1] + cs.segments_area.size)

        if include_text:
            chunk = np.frombuffer(raw, dtype=np.uint8).copy()
            text_chunks.append(chunk)
            text_offsets.append(text_offsets[-1] + chunk.size)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format": np.asarray(["mlxmolkit.orcacosmo.vectors.v1"]),
        "created_unix_s": np.asarray([time.time()], dtype=np.float64),
        "filenames": np.asarray(filenames),
        "sha256": np.asarray(sha256),
        "epsilon": np.asarray(epsilon, dtype=np.float64),
        "fepsi": np.asarray(fepsi, dtype=np.float64),
        "area_A2": np.asarray(area, dtype=np.float64),
        "volume_A3": np.asarray(volume, dtype=np.float64),
        "total_screening_charge_e": np.asarray(total_screening_charge, dtype=np.float64),
        "total_energy_hartree": np.asarray(total_energy_hartree, dtype=np.float64),
        "dielectric_energy_hartree": np.asarray(dielectric_energy_hartree, dtype=np.float64),
        "atom_offsets": np.asarray(atom_offsets, dtype=np.int64),
        "atom_radii_A": np.concatenate(atom_radii) if atom_radii else np.empty((0,), dtype=np.float64),
        "atom_coords_bohr": np.concatenate(atom_coords_bohr, axis=0) if atom_coords_bohr else np.empty((0, 3), dtype=np.float64),
        "atom_z": np.concatenate(atom_z) if atom_z else np.empty((0,), dtype=np.int16),
        "segment_offsets": np.asarray(segment_offsets, dtype=np.int64),
        "segments_atom": np.concatenate(segments_atom) if segments_atom else np.empty((0,), dtype=np.int32),
        "segments_xyz_bohr": np.concatenate(segments_xyz_bohr, axis=0) if segments_xyz_bohr else np.empty((0, 3), dtype=np.float64),
        "segments_charge_e": np.concatenate(segments_charge) if segments_charge else np.empty((0,), dtype=np.float64),
        "segments_area_A2": np.concatenate(segments_area) if segments_area else np.empty((0,), dtype=np.float64),
        "segments_sigma_e_per_A2": np.concatenate(segments_sigma) if segments_sigma else np.empty((0,), dtype=np.float64),
        "segments_potential": np.concatenate(segments_potential) if segments_potential else np.empty((0,), dtype=np.float64),
    }
    if include_text:
        payload["text_offsets"] = np.asarray(text_offsets, dtype=np.int64)
        payload["text_bytes"] = np.concatenate(text_chunks) if text_chunks else np.empty((0,), dtype=np.uint8)
    np.savez_compressed(out_npz, **payload)
    return {
        "files": len(files),
        "atoms": int(atom_offsets[-1]),
        "segments": int(segment_offsets[-1]),
        "include_text": bool(include_text),
        "seconds": time.perf_counter() - t0,
    }


def extract_text_from_npz(npz_path: Path, out_dir: Path) -> None:
    z = np.load(npz_path, allow_pickle=False)
    if "text_bytes" not in z.files or "text_offsets" not in z.files:
        raise ValueError(f"{npz_path} does not contain raw text bytes; repack with --include-text")
    names = z["filenames"].astype(str)
    text = np.asarray(z["text_bytes"], dtype=np.uint8)
    offsets = np.asarray(z["text_offsets"], dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        chunk = text[offsets[i] : offsets[i + 1]]
        (out_dir / name).write_bytes(chunk.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--input-dir", type=Path, default=Path.home() / ".cache" / "mlxmolkit-orcacosmo")
    p_pack.add_argument("--out-zip", type=Path, default=None)
    p_pack.add_argument("--out-npz", type=Path, default=Path("data/orcacosmo/orcacosmo_vectors.npz"))
    p_pack.add_argument("--include-text", action="store_true")
    p_pack.add_argument("--limit", type=int, default=None)

    p_extract = sub.add_parser("extract-text")
    p_extract.add_argument("--npz", type=Path, required=True)
    p_extract.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "pack":
        files = iter_orcacosmo_files(args.input_dir, args.limit)
        if not files:
            raise SystemExit(f"no .orcacosmo files found in {args.input_dir}")
        if args.out_zip is not None:
            pack_zip(files, args.out_zip)
        summary = pack_npz(files, args.out_npz, include_text=bool(args.include_text))
        summary["input_dir"] = str(args.input_dir)
        summary["out_npz"] = str(args.out_npz)
        summary["out_zip"] = str(args.out_zip) if args.out_zip else None
        print(json.dumps(summary, indent=2))
    elif args.cmd == "extract-text":
        extract_text_from_npz(args.npz, args.out_dir)


if __name__ == "__main__":
    main()
