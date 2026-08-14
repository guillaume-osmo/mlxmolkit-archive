#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_TMP_ROOT = Path("/private/var/folders/hl/mqfp4f3n7g3_z8vzl1b_nq6r0000gn/T")
DEFAULT_PREFIXES = (
    "tiered-cosmors-",
    "multi-cosmors-",
    "orca-cosmors-",
    "gxtb-opt-",
    "gfn2-tmcosmo-",
)


def directory_size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    return total


def active_temp_roots(tmp_root: Path, prefixes: tuple[str, ...]) -> set[Path]:
    active: set[Path] = set()
    prefix_re = "|".join(re.escape(p) for p in prefixes)
    pattern = re.compile(rf"({re.escape(str(tmp_root))}/(?:{prefix_re})[^/]+)")
    for proc in ("orca", "xtb"):
        try:
            out = subprocess.check_output(
                ["lsof", "-c", proc, "-d", "cwd", "-Fn"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            out = ""
        for line in out.splitlines():
            if not line.startswith("n"):
                continue
            match = pattern.search(line[1:])
            if match:
                active.add(Path(match.group(1)).resolve())
    return active


def iter_candidate_dirs(tmp_root: Path, prefixes: tuple[str, ...]) -> list[Path]:
    if not tmp_root.exists():
        return []
    return [p for p in tmp_root.iterdir() if p.is_dir() and p.name.startswith(prefixes)]


def cleanup_once(args: argparse.Namespace) -> dict:
    tmp_root = Path(args.tmp_root)
    prefixes = tuple(x.strip() for x in args.prefixes.split(",") if x.strip())
    active = active_temp_roots(tmp_root, prefixes)
    cutoff = time.time() - args.min_age_minutes * 60.0
    rows = []
    removed = 0
    freed = 0
    kept_active = 0
    kept_young = 0

    for path in iter_candidate_dirs(tmp_root, prefixes):
        try:
            stat = path.stat()
            resolved = path.resolve()
        except FileNotFoundError:
            continue
        age_minutes = (time.time() - stat.st_mtime) / 60.0
        size = directory_size(path) if args.compute_size or args.dry_run else 0
        status = "remove"
        reason = ""
        if resolved in active:
            status = "keep"
            reason = "active_lsof_cwd"
            kept_active += 1
        elif stat.st_mtime >= cutoff:
            status = "keep"
            reason = "younger_than_min_age"
            kept_young += 1

        if status == "remove":
            if args.dry_run:
                reason = "dry_run"
            else:
                if not args.compute_size:
                    size = directory_size(path)
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
                freed += size
                reason = "removed"

        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "status": status,
                "reason": reason,
                "age_minutes": age_minutes,
                "size_bytes": int(size),
            }
        )

    summary = {
        "tmp_root": str(tmp_root),
        "dry_run": bool(args.dry_run),
        "min_age_minutes": float(args.min_age_minutes),
        "active_dirs": sorted(p.name for p in active),
        "candidates": len(rows),
        "removed": removed,
        "kept_active": kept_active,
        "kept_young": kept_young,
        "freed_bytes": int(freed),
        "freed_gb": freed / 1.0e9,
        "rows": sorted(rows, key=lambda r: r["size_bytes"], reverse=True),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"[orca-temp-clean] candidates={summary['candidates']} removed={removed} "
            f"kept_active={kept_active} kept_young={kept_young} "
            f"freed_GB={summary['freed_gb']:.2f} dry_run={args.dry_run}"
        )
        for row in summary["rows"][: args.top]:
            print(
                f"{row['status']:>6} {row['size_bytes']/1e9:7.2f} GB "
                f"age_min={row['age_minutes']:7.1f} {row['reason']:>20} {row['name']}"
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely clean stale ORCA/xTB temporary work directories.")
    parser.add_argument("--tmp-root", default=str(DEFAULT_TMP_ROOT))
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    parser.add_argument("--min-age-minutes", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--compute-size", action="store_true", help="Compute sizes even for rows that will be kept.")
    parser.add_argument("--loop", action="store_true", help="Repeat cleanup until interrupted.")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        cleanup_once(args)
        if not args.loop:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
