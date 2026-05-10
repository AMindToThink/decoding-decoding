"""σ_0 × β_target sweep for the tight-prior hypothesis.

User asked: does anchoring the lognormal prior more tightly at β=1 (so the
filter ignores most of the bootstrap evidence) deliver stable, distinct
sample entropies at different β_target values?

Theoretically (see results/counter_decode/prior_sweep/findings.md):
  - σ_0 → ∞ : β̂ → β_target, β_dec → 1, no temperature applied.
  - σ_0 → 0 : β̂ pinned at 1, β_dec ≈ β_target = uncorrected sampling.

This script writes per-(σ_0, β_target) trajectories so we can plot the
operating-point map and check the prediction.

Usage (split across both GPUs):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_tight_prior_sweep.py \
        --sigma0-values 0.01,0.05,0.1,0.2 --cuda-device 0
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_tight_prior_sweep.py \
        --sigma0-values 0.5,1.0,2.0 --cuda-device 1
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
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode_tight_prior",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=1)
    parser.add_argument(
        "--beta-targets",
        type=str,
        default="0.25,1.0,4.0",
    )
    parser.add_argument(
        "--sigma0-values",
        type=str,
        default="0.01,0.05,0.1,0.2,0.5,1.0,2.0",
    )
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    sigma0_values = [float(s.strip()) for s in args.sigma0_values.split(",") if s.strip()]
    beta_targets = tuple(float(s.strip()) for s in args.beta_targets.split(",") if s.strip())

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

    t0 = time.time()
    for sigma_0 in sigma0_values:
        tag = f"sigma_{sigma_0}".replace(".", "p")
        out = args.out_dir / tag
        print(f"\n[tight-prior] === σ_0={sigma_0} ===")
        run_counter_decode_experiment(
            out,
            beta_target_values=beta_targets,
            corrected_arms=(True,),
            n_runs_per_prompt=args.n_runs_per_prompt,
            max_tokens=args.max_tokens,
            sigma_0=sigma_0,
            sparse_positions=(0, args.max_tokens - 1),
            top_n=200,
            prior_kind="lognormal_laplace",
            evidence_weight=1.0,
            memory_decay=1.0,
        )
    elapsed = time.time() - t0
    print(f"\n[tight-prior] total wall: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
