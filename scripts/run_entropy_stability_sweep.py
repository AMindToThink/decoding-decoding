"""(σ_0, γ) × β_target sweep for the "stable distinct entropy" target.

User goal: a single set of corrector hyperparameters such that, as β_target
ranges over a temperature ladder, the corrected sample entropy is

  (a) STABLE — not drifting up or down within a trajectory,
  (b) DISTINCT — monotonically different across β_target levels,
  (c) NON-DEGENERATE — not collapsed to greedy and not uniform.

The σ_0 sweep showed that σ_0 ≈ 0.2 already achieves (a) and roughly (b),
but at β_target=4 the sample is borderline degenerate (max P=0.89). This
script combines the prior anchoring knob (σ_0) with the memory-decay knob
(γ) to look for a sweet spot.

Usage (split across both GPUs):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_entropy_stability_sweep.py \
        --sigma0-values 0.1,0.15,0.2 --cuda-device 0
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_entropy_stability_sweep.py \
        --sigma0-values 0.3,0.5 --cuda-device 1
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
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode_entropy_stability",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=1)
    parser.add_argument(
        "--beta-targets",
        type=str,
        default="0.25,0.5,1.0,2.0,4.0",
    )
    parser.add_argument(
        "--sigma0-values",
        type=str,
        default="0.1,0.15,0.2,0.3,0.5",
    )
    parser.add_argument(
        "--gamma-values",
        type=str,
        default="1.0,0.97,0.9",
    )
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    sigma0_values = [float(s.strip()) for s in args.sigma0_values.split(",") if s.strip()]
    gamma_values = [float(s.strip()) for s in args.gamma_values.split(",") if s.strip()]
    beta_targets = tuple(float(s.strip()) for s in args.beta_targets.split(",") if s.strip())

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

    t0 = time.time()
    for sigma_0 in sigma0_values:
        for gamma in gamma_values:
            tag = f"s{sigma_0}_g{gamma}".replace(".", "p")
            out = args.out_dir / tag
            print(f"\n[ent-stab] === σ_0={sigma_0}, γ={gamma}, α={args.evidence_weight} ===")
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
                evidence_weight=args.evidence_weight,
                memory_decay=gamma,
            )
    elapsed = time.time() - t0
    print(f"\n[ent-stab] total wall: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
