"""Run the full generation experiment.

Generates 32 prompts × {1, 3, 5, inf} top-k conditions × 8 runs (1 for k=1) at
L=1024 tokens, on Qwen/Qwen2.5-3B. Saves run_metadata.parquet and per-sequence
.npz logprob files under data/.

Usage:
    uv run python scripts/run_experiment.py [--max-tokens N] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()

    # Pin to GPU 0 BEFORE importing anything that touches CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from decoding_decoding.generate import run_experiment

    t0 = time.time()
    run_experiment(args.out_dir, max_tokens=args.max_tokens)
    elapsed = time.time() - t0
    print(f"Total wall clock: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
