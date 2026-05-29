"""σ_0 sensitivity sweep at the corrector's worst-case condition (β_target=0.5).

Tests whether retuning the log-normal prior std σ_0 fixes the corrected-arm
drift observed in the F0 sweep at β_target < 1 (the confusion regime).

Result (see results/counter_decode/summary.md): σ_0 retuning shifts the
*starting* point of the corrected curve (because β̂_0 = exp(−σ_0²/2)
varies) but does not fix the late-trajectory drift. The corrector failure
in the flattening regime is structural, not a prior-mismatch issue.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_counter_decode_sigma_sweep.py
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
        default=Path(__file__).resolve().parent.parent / "data" / "counter_decode_sigma_sweep",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-runs-per-prompt", type=int, default=2)
    parser.add_argument(
        "--sigma0-values",
        type=str,
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated σ_0 values to sweep.",
    )
    parser.add_argument("--beta-target", type=float, default=0.5)
    parser.add_argument("--cuda-device", type=str, default="0")
    args = parser.parse_args()

    sigma_0_values = [float(s.strip()) for s in args.sigma0_values.split(",") if s.strip()]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

    t0 = time.time()
    for sigma_0 in sigma_0_values:
        out = args.out_dir / f"sigma_{sigma_0}"
        run_counter_decode_experiment(
            out,
            beta_target_values=(args.beta_target,),
            corrected_arms=(True, False),
            n_runs_per_prompt=args.n_runs_per_prompt,
            max_tokens=args.max_tokens,
            sigma_0=sigma_0,
            sparse_positions=(0, args.max_tokens - 1),
            top_n=200,
        )
    elapsed = time.time() - t0
    print(f"[sigma-sweep] total wall: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
