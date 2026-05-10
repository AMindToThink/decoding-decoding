"""Compare different priors on β under the counter-decoding corrector.

Triggered by: the streaming Laplace's β̂ has a stable fixed point at
β_target (where β_dec = 1 and the corrector neutralizes itself). The user
asked whether different priors — particularly the max-ent exponential prior
under the constraint E[β] = 1 — change the dynamics.

Implementation note: the discrete-grid filter accepts any prior pmf computed
from a scipy.stats distribution (per repo CLAUDE.md preference for library
implementations over hand-rolled ones). The streaming Laplace (current
default) uses the log-normal prior in closed form.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_prior_sweep.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


PRIORS = (
    "lognormal_laplace",  # baseline (current)
    "lognormal",          # discrete-grid version of the same prior
    "exponential",        # max-ent for E[β]=1
    "gamma",              # E[β]=1 with shape 2 (more concentrated than exp)
    "invgamma",           # heavy-tailed on the right
    "halfcauchy_logβ",    # heavy tail, only above β=1
    "cauchy_logβ",        # heavy two-sided tail on log β
    "uniform_logβ",       # flat in log β (improper, truncated)
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode_priors",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=1)
    parser.add_argument(
        "--beta-targets",
        type=str,
        default="0.25,1.0,4.0",
    )
    parser.add_argument(
        "--priors",
        type=str,
        default=",".join(PRIORS),
    )
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--memory-decay", type=float, default=1.0)
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    beta_targets = tuple(float(x) for x in args.beta_targets.split(","))
    priors = tuple(s.strip() for s in args.priors.split(","))

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

    t0 = time.time()
    for prior in priors:
        out = args.out_dir / f"prior_{prior.replace('_logβ', '_logbeta')}"
        print(f"\n[prior-sweep] === prior={prior} ===")
        run_counter_decode_experiment(
            out,
            beta_target_values=beta_targets,
            corrected_arms=(True,),  # corrected only — that's where the prior matters
            n_runs_per_prompt=args.n_runs_per_prompt,
            max_tokens=args.max_tokens,
            sigma_0=args.sigma0,
            sparse_positions=(0, args.max_tokens - 1),
            top_n=200,
            prior_kind=prior,
            evidence_weight=args.evidence_weight,
            memory_decay=args.memory_decay,
        )
    elapsed = time.time() - t0
    print(f"\n[prior-sweep] total wall: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
