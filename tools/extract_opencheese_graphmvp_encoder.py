#!/usr/bin/env python
"""Extract a naked openCHEESE encoder from a GraphMVP pretraining checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="GraphMVPPretrainer safetensors checkpoint.")
    parser.add_argument("out", type=Path, help="Output CheeseGraphTransformer safetensors checkpoint.")
    parser.add_argument("--encoder", choices=["model_3d", "model_2d"], default="model_3d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    weights = mx.load(str(args.checkpoint))
    prefix = f"{args.encoder}."
    extracted = {key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)}
    if not extracted:
        raise ValueError(f"checkpoint has no weights with prefix {prefix!r}")
    if not any(key.startswith("blocks.") for key in extracted):
        raise ValueError(f"extracted {args.encoder!r} weights do not look like a CheeseGraphTransformer")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out), extracted)
    print(f"wrote {args.out} with {len(extracted)} tensors from {args.encoder}")


if __name__ == "__main__":
    main()
