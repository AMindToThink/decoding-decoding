"""Sweep over the (evidence_weight, memory_decay) "fuzzy window" hyperparameters.

Tests the user's hypothesis that limiting how much evidence the filter
accumulates — and decaying old evidence exponentially — can prevent the
corrector from converging to its degenerate β̂ = β_target fixed point.

Configurations (α = evidence_weight, γ = memory_decay):
    - (1.00, 1.00) — baseline; full evidence, no decay (default).
    - (0.10, 1.00) — α down: each obs counts 10× less; J grows slower.
    - (1.00, 0.99) — γ down: exp decay τ ≈ 100 tokens.
    - (1.00, 0.95) — γ down: exp decay τ ≈ 20 tokens (sharper window).
    - (0.50, 0.99) — both: half-evidence + 100-token window.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_fuzzy_window_sweep.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


CONFIGS = (
    (1.00, 1.00),
    (0.10, 1.00),
    (1.00, 0.99),
    (1.00, 0.95),
    (0.50, 0.99),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode_fuzzy",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=1)
    parser.add_argument(
        "--beta-targets",
        type=str,
        default="0.25,4.0",
    )
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument(
        "--prior-kind",
        type=str,
        default="lognormal_laplace",
        help="Prior kind to test the fuzzy window with.",
    )
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    beta_targets = tuple(float(x) for x in args.beta_targets.split(","))

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

    t0 = time.time()
    for alpha, gamma in CONFIGS:
        tag = f"a{alpha:.2f}_g{gamma:.2f}".replace(".", "p")
        out = args.out_dir / tag
        print(f"\n[fuzzy] === α={alpha}, γ={gamma} ===")
        run_counter_decode_experiment(
            out,
            beta_target_values=beta_targets,
            corrected_arms=(True,),
            n_runs_per_prompt=args.n_runs_per_prompt,
            max_tokens=args.max_tokens,
            sigma_0=args.sigma0,
            sparse_positions=(0, args.max_tokens - 1),
            top_n=200,
            prior_kind=args.prior_kind,
            evidence_weight=alpha,
            memory_decay=gamma,
        )
    elapsed = time.time() - t0
    print(f"\n[fuzzy] total wall: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
