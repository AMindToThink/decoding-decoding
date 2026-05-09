"""Extend every topk trace from its current length to `--target-length`.

Usage:
    uv run python scripts/extend_topk_experiment.py --target-length 2048
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from decoding_decoding.generate import extend_topk_experiment
    from decoding_decoding.logging_utils import start_logging

    start_logging(args.data_dir / "logs", f"extend_topk_to_{args.target_length}")

    t0 = time.time()
    extend_topk_experiment(args.data_dir, target_length=args.target_length)
    elapsed = time.time() - t0
    print(f"Total wall clock: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
