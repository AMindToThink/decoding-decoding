"""Tiny end-to-end smoke test for counter-decoding generation.

Runs 2 prompts × 1 run × 12 tokens × {β_target=1.5, both arms}, on the real
Qwen2.5-3B model. Validates the full pipeline (model loading, batched
prefill with left-padding, KV-cached decode, Laplace state, save).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_counter_decode.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    out_dir = Path("/tmp/counter_decode_smoke")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # Limit prompts for the smoke test — patch PROMPTS into a 2-element tuple.
    import decoding_decoding.prompts as P
    saved_prompts = P.PROMPTS
    short_prompts = (
        "Marie Curie was a Polish-born physicist and chemist whose pioneering research on",
        "The old lighthouse stood at the edge of the cliff, and every night for the past forty years",
    )
    P.PROMPTS = short_prompts  # patch for downstream callers
    import decoding_decoding.counter_decode_generate as cdg
    # Also patch in the module that imported PROMPTS at top level.
    cdg.PROMPTS = short_prompts

    try:
        from decoding_decoding.counter_decode_generate import run_counter_decode_experiment

        run_counter_decode_experiment(
            out_dir,
            beta_target_values=(1.5,),
            corrected_arms=(True, False),
            n_runs_per_prompt=1,
            max_tokens=12,
            sigma_0=0.5,
            sparse_positions=(0, 4, 11),
            top_n=20,
        )
    finally:
        P.PROMPTS = saved_prompts
        cdg.PROMPTS = saved_prompts

    # Light validation.
    import polars as pl
    manifest = pl.read_parquet(out_dir / "manifest.parquet")
    print(manifest)
    assert len(manifest) == 4, f"expected 4 traces (2 prompts × 1 run × {{T,F}}); got {len(manifest)}"

    # Spot-check a trace.
    sample = manifest.row(0, named=True)
    npz = np.load(out_dir / sample["trace_path"])
    print("trace keys:", list(npz.keys()))
    print("beta_hat_pre:", npz["beta_hat_pre"])
    print("beta_dec:", npz["beta_dec"])
    print("J_pre:", npz["J_pre"])
    print("score:", npz["score"])
    print("fisher:", npz["fisher"])
    print("sampled token ids:", npz["sampled_token_ids"])
    print("max_p_phi:", npz["max_p_phi"])
    print("entropy_phi:", npz["entropy_phi"])

    # For the corrected arm, β_dec * β̂_pre should equal β_target = 1.5 (within fp).
    corr = manifest.filter(pl.col("condition_label").str.contains("corr"))
    for row in corr.iter_rows(named=True):
        npz = np.load(out_dir / row["trace_path"])
        eff = npz["beta_dec"] * npz["beta_hat_pre"]
        assert np.allclose(eff, 1.5, atol=1e-4), f"{row['trace_id']}: β_eff drift {eff}"
        # J_pre[t+1] == J_pre[t] + fisher[t]
        np.testing.assert_allclose(
            npz["J_pre"][1:], npz["J_pre"][:-1] + npz["fisher"][:-1], atol=1e-3, rtol=1e-3
        )

    # For uncorrected arm, β_dec is constant 1.5.
    uncorr = manifest.filter(pl.col("condition_label").str.contains("unco"))
    for row in uncorr.iter_rows(named=True):
        npz = np.load(out_dir / row["trace_path"])
        np.testing.assert_allclose(npz["beta_dec"], 1.5, atol=1e-6)
        # Sparse logits saved for uncorrected.
        sp = np.load(out_dir / "sparse_logits" / f"{row['trace_id']}.npz")
        print("sparse positions:", sp["positions"], "logits shape:", sp["logits"].shape)

    print("\n[smoke OK] all invariants hold.")


if __name__ == "__main__":
    main()
