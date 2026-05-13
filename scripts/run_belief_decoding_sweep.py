"""Joint sweep over (γ, α, p) to model the LLM's belief-update dynamics.

The streaming Laplace filter at parameters (γ, α) is our running model of
how the LLM updates its belief about its own decoder β over the sequence.
The new knob p is how much of that belief gets propagated to correct the
sampler:

    β_dec_t = β_target / β̂_t^p

  p = 0  → no correction (β_target alone, equivalent to the uncorrected arm)
  p = 1  → full correction (cancel the inferred imprint completely)
  p ∈ (0, 1) → partial: assume the LLM's belief is only partially in step
              with what the filter says.

We sweep all three at fixed σ_0 = 0.2 (the empirical best operating point
from the tight-prior sweep on 2026-05-10) across three β_target settings.

Success criterion: closed-loop stability, i.e. corrected sample-entropy
and max-P should both move monotonically and smoothly with β_target, with
no degenerate saturation at extreme β. Whichever (γ, α, p) achieves that
is the best running model of the LLM's belief dynamics.

Outputs: one subdirectory per (γ, α, p) under data/counter_decode_belief_sweep/.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_belief_decoding_sweep.py
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import product
from pathlib import Path


# Default sweep grid.
GAMMA_VALUES = (0.85, 0.95, 1.00)
ALPHA_VALUES = (0.5, 1.0)
EXPONENT_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode_belief_sweep",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=1)
    parser.add_argument("--beta-targets", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--sigma0", type=float, default=0.2,
                        help="best operating point per the tight-prior sweep")
    parser.add_argument("--gammas", type=str, default=",".join(str(g) for g in GAMMA_VALUES))
    parser.add_argument("--alphas", type=str, default=",".join(str(a) for a in ALPHA_VALUES))
    parser.add_argument("--exponents", type=str, default=",".join(str(p) for p in EXPONENT_VALUES))
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    beta_targets = tuple(float(x) for x in args.beta_targets.split(","))
    gammas = tuple(float(x) for x in args.gammas.split(","))
    alphas = tuple(float(x) for x in args.alphas.split(","))
    exponents = tuple(float(x) for x in args.exponents.split(","))

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

    grid = list(product(gammas, alphas, exponents))
    print(
        f"[belief-sweep] {len(grid)} configs × {len(beta_targets)} β_target × "
        f"32 prompts × {args.max_tokens} tokens"
    )

    t0 = time.time()
    for gamma, alpha, p in grid:
        tag = (
            f"g{gamma:.2f}_a{alpha:.2f}_p{p:.2f}"
            .replace(".", "p")
        )
        out = args.out_dir / tag
        if (out / "config.json").exists():
            print(f"[belief-sweep] skipping {tag} (already exists)")
            continue
        print(f"\n[belief-sweep] === γ={gamma}, α={alpha}, p={p} ===")
        run_counter_decode_experiment(
            out,
            beta_target_values=beta_targets,
            corrected_arms=(True,),
            n_runs_per_prompt=args.n_runs_per_prompt,
            max_tokens=args.max_tokens,
            sigma_0=args.sigma0,
            sparse_positions=(0, args.max_tokens - 1),
            top_n=200,
            prior_kind="lognormal_laplace",
            evidence_weight=alpha,
            memory_decay=gamma,
            correction_exponent=p,
        )
    elapsed = time.time() - t0
    print(f"\n[belief-sweep] total wall: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
