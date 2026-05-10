"""Run the full counter-decoding F0 sweep.

Generates trajectories for the headline experiment in
bayesian-sampler/counter_decoding.html (section 03 — "The plan"):

    β_target ∈ {0.5, 0.7, 1.0, 1.3, 1.7, 2.0} × {corrected, uncorrected}
    32 prompts × 3 runs each = 96 trajectories per condition
    T = 200 tokens per trajectory
    Qwen-2.5-3B, σ_0 = 0.5

Saves per-trajectory diagnostics under data/counter_decode/, including:
  - per-step Laplace state (β̂_pre, β_dec, J_pre, score, fisher)
  - per-step concentration stats on P_φ and P_sample (max-prob, entropy)
  - per-step top-N logprobs on P_φ (for archival / cross-checks)
  - sparse full-vocab logits at selected t for the SVD step (uncorrected only)

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_counter_decode.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _parse_betas(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=3)
    parser.add_argument(
        "--beta-targets",
        type=_parse_betas,
        default="0.5,0.7,1.0,1.3,1.7,2.0",
        help="Comma-separated β_target values to sweep.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="corrected,uncorrected",
        help="Comma-separated subset of {corrected, uncorrected}.",
    )
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument(
        "--sparse-positions",
        type=str,
        default="0,8,16,32,64,128,199",
        help=(
            "Comma-separated step indices at which to save full-vocab logits "
            "(uncorrected arm only)."
        ),
    )
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    arms_set = {a.strip().lower() for a in args.arms.split(",") if a.strip()}
    valid = {"corrected", "uncorrected"}
    if not arms_set.issubset(valid):
        raise ValueError(f"unknown arms {arms_set}; must be subset of {valid}")
    corrected_arms = []
    if "corrected" in arms_set:
        corrected_arms.append(True)
    if "uncorrected" in arms_set:
        corrected_arms.append(False)

    sparse_positions = tuple(int(s.strip()) for s in args.sparse_positions.split(",") if s.strip())

    print(
        f"[run-counter-decode] β_targets={args.beta_targets} arms={arms_set} "
        f"runs/prompt={args.n_runs_per_prompt} max_tokens={args.max_tokens}"
    )

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment
    from decoding_decoding.logging_utils import start_logging

    start_logging(args.out_dir / "logs", f"counter_decode_T{args.max_tokens}")

    t0 = time.time()
    run_counter_decode_experiment(
        args.out_dir,
        beta_target_values=tuple(args.beta_targets),
        corrected_arms=tuple(corrected_arms),
        n_runs_per_prompt=args.n_runs_per_prompt,
        max_tokens=args.max_tokens,
        sigma_0=args.sigma0,
        sparse_positions=sparse_positions,
        top_n=args.top_n,
    )
    elapsed = time.time() - t0
    print(f"[run-counter-decode] total wall clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
