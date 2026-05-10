"""Shape-estimator sweep: rank-gap and Rényi-∞ β̂ vs baseline Laplace.

Tests the non-bootstrap shape-based β estimators ("B" and "C" from the
subagent brainstorm). The reference for both is computed by averaging
over real prompt positions during prefill — not from a single t=0 logit
(which would be too noisy because high-vs-low entropy of the first
generated token is prompt-dependent).

Usage (split across both GPUs):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_shape_estimator_sweep.py \
        --estimators rank_gap --cuda-device 0
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_shape_estimator_sweep.py \
        --estimators renyi_infty --cuda-device 1
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode_shape",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=1)
    parser.add_argument(
        "--beta-targets",
        type=str,
        default="0.25,0.5,1.0,2.0,4.0",
    )
    parser.add_argument(
        "--estimators",
        type=str,
        default="rank_gap,renyi_infty",
        help="Comma-separated estimators to run.",
    )
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    beta_targets = tuple(float(s.strip()) for s in args.beta_targets.split(","))
    estimators = tuple(s.strip() for s in args.estimators.split(",") if s.strip())

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

    t0 = time.time()
    for est in estimators:
        out = args.out_dir / f"est_{est}"
        print(f"\n[shape] === estimator={est} ===")
        run_counter_decode_experiment(
            out,
            beta_target_values=beta_targets,
            corrected_arms=(True,),
            n_runs_per_prompt=args.n_runs_per_prompt,
            max_tokens=args.max_tokens,
            sigma_0=0.5,  # unused by shape estimators but required by the harness
            sparse_positions=(0, args.max_tokens - 1),
            top_n=200,
            prior_kind=est,
        )
    elapsed = time.time() - t0
    print(f"\n[shape] total wall: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
