"""Unit tests for counter_decode_analyze on synthetic on-disk data.

We synthesize a tiny manifest + traces directory with known per-step values,
then verify the loader/aggregator returns what we put in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from decoding_decoding.counter_decode_analyze import (
    f0_concentration,
    f3_beta_eff,
    load_counter_condition,
)
from decoding_decoding.data_layout import (
    MANIFEST_FILENAME,
    encode_params,
    upsert_manifest,
)


def _write_synthetic(
    out_dir: Path,
    *,
    n_prompts: int,
    n_runs: int,
    T: int,
    beta_target: float,
    corrected: bool,
    sigma_0: float = 0.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in range(n_prompts):
        for r in range(n_runs):
            arm = "corr" if corrected else "unco"
            bt_str = f"{beta_target:.2f}".replace(".", "p")
            tid = f"counter_p{pid:02d}_b{bt_str}_{arm}_r{r:02d}"
            ids = np.full((T, 5), -1, dtype=np.int32)
            lps = np.zeros((T, 5), dtype=np.float32)
            sampled = np.zeros((T,), dtype=np.int64)
            beta_hat_pre = np.linspace(1.0, 1.5, T, dtype=np.float32) + 0.01 * (pid + r)
            if corrected:
                beta_dec = beta_target / beta_hat_pre
            else:
                beta_dec = np.full((T,), beta_target, dtype=np.float32)
            J_pre = np.linspace(4.0, 30.0, T, dtype=np.float32)
            score = np.full((T,), 0.1, dtype=np.float32)
            fisher = np.full((T,), 0.5, dtype=np.float32)
            max_p_phi = np.linspace(0.5, 0.7, T, dtype=np.float32)
            entropy_phi = np.linspace(2.0, 1.5, T, dtype=np.float32)
            max_p_sample = np.linspace(0.6, 0.8, T, dtype=np.float32)
            entropy_sample = np.linspace(1.5, 1.0, T, dtype=np.float32)
            np.savez_compressed(
                out_dir / "traces" / f"{tid}.npz",
                top_token_ids=ids,
                top_logprobs=lps,
                sampled_token_ids=sampled,
                beta_hat_pre=beta_hat_pre,
                beta_dec=beta_dec,
                J_pre=J_pre,
                score=score,
                fisher=fisher,
                max_p_phi=max_p_phi,
                entropy_phi=entropy_phi,
                max_p_sample=max_p_sample,
                entropy_sample=entropy_sample,
                eta_hat_post_final=np.float32(0.5),
                J_post_final=np.float32(40.0),
            )
            params = {
                "beta_target": float(beta_target),
                "corrected": bool(corrected),
                "sigma_0": float(sigma_0),
            }
            arm = "corr" if corrected else "unco"
            rows.append(
                {
                    "trace_id": tid,
                    "family": "counter_decode",
                    "prompt_id": pid,
                    "prompt_text": f"prompt_{pid}",
                    "condition_label": f"btarget={beta_target:.2f}_arm={arm}",
                    "params_json": encode_params(params),
                    "run_idx": r,
                    "seed": 0,
                    "length": T,
                    "sampled_token_ids": list(map(int, sampled.tolist())),
                    "decoded_text": "",
                    "trace_path": f"traces/{tid}.npz",
                }
            )
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    upsert_manifest(out_dir, rows)


def test_load_counter_condition_basic_shapes(tmp_path: Path) -> None:
    _write_synthetic(
        tmp_path, n_prompts=4, n_runs=3, T=12, beta_target=1.7, corrected=True
    )
    cond = load_counter_condition(tmp_path, beta_target=1.7, corrected=True)
    assert cond.beta_hat_pre.shape == (4, 3, 12)
    assert cond.beta_dec.shape == (4, 3, 12)
    # corrected: β_dec * β̂ = β_target.
    np.testing.assert_allclose(cond.beta_dec * cond.beta_hat_pre, 1.7, atol=1e-5)


def test_f3_beta_eff_corrected_is_constant(tmp_path: Path) -> None:
    _write_synthetic(
        tmp_path, n_prompts=3, n_runs=4, T=20, beta_target=1.4, corrected=True
    )
    cond = load_counter_condition(tmp_path, beta_target=1.4, corrected=True)
    f3 = f3_beta_eff(cond)
    # By construction, β_eff is exactly β_target.
    np.testing.assert_allclose(f3.beta_eff, 1.4, atol=1e-5)
    np.testing.assert_allclose(f3.median_abs_log_err, 0.0, atol=1e-5)


def test_f3_beta_eff_uncorrected_drifts(tmp_path: Path) -> None:
    _write_synthetic(
        tmp_path, n_prompts=3, n_runs=4, T=20, beta_target=1.4, corrected=False
    )
    cond = load_counter_condition(tmp_path, beta_target=1.4, corrected=False)
    f3 = f3_beta_eff(cond)
    # β_eff = β_target * β̂_pre, with β̂_pre ramping from ~1 to ~1.5.
    # So β_eff[0] ≈ 1.4 and β_eff[-1] ≈ 2.1.
    assert f3.median_abs_log_err[0] < 0.05
    assert f3.median_abs_log_err[-1] > 0.2


def _write_synthetic_with_sparse(
    out_dir: Path,
    *,
    n_prompts: int,
    n_runs: int,
    T: int,
    beta_dec_values: list[float],
    sparse_positions: list[int],
    V: int = 64,
    sigma_0: float = 0.5,
    seed: int = 0,
) -> None:
    """Synthesize uncorrected traces + sparse_logits with EXACT multiplicativity:
        ℓ_{p,r}^(k,t) = β_k · ℓ*_p + small noise, where ℓ*_p is per-prompt.
    The SVD should recover R^(p)(t) ≈ 0 and β̄^(p)(k) ∝ β_k.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse_logits").mkdir(parents=True, exist_ok=True)
    rows = []
    ell_star_per_prompt = [rng.normal(size=V).astype(np.float32) for _ in range(n_prompts)]

    for pid in range(n_prompts):
        for bk in beta_dec_values:
            for r in range(n_runs):
                bt_str = f"{bk:.2f}".replace(".", "p")
                tid = f"counter_p{pid:02d}_b{bt_str}_unco_r{r:02d}"
                # Per-step Laplace state arrays — fill with constant placeholders.
                T_arr_f32 = np.zeros((T,), dtype=np.float32)
                np.savez_compressed(
                    out_dir / "traces" / f"{tid}.npz",
                    top_token_ids=np.full((T, 5), -1, dtype=np.int32),
                    top_logprobs=np.zeros((T, 5), dtype=np.float32),
                    sampled_token_ids=np.zeros((T,), dtype=np.int64),
                    beta_hat_pre=np.full((T,), float(bk), dtype=np.float32),
                    beta_dec=np.full((T,), float(bk), dtype=np.float32),
                    J_pre=T_arr_f32,
                    score=T_arr_f32,
                    fisher=T_arr_f32,
                    max_p_phi=T_arr_f32,
                    entropy_phi=T_arr_f32,
                    max_p_sample=T_arr_f32,
                    entropy_sample=T_arr_f32,
                    eta_hat_post_final=np.float32(0.0),
                    J_post_final=np.float32(0.0),
                )
                # Sparse logits: exact multiplicativity per (p, r).
                logits_p = np.stack(
                    [bk * ell_star_per_prompt[pid] + 0.01 * rng.normal(size=V)
                     for _ in sparse_positions],
                    axis=0,
                ).astype(np.float16)
                np.savez_compressed(
                    out_dir / "sparse_logits" / f"{tid}.npz",
                    positions=np.asarray(sparse_positions, dtype=np.int32),
                    logits=logits_p,
                )
                params = {
                    "beta_target": float(bk),
                    "corrected": False,
                    "sigma_0": float(sigma_0),
                }
                rows.append(
                    {
                        "trace_id": tid,
                        "family": "counter_decode",
                        "prompt_id": pid,
                        "prompt_text": f"prompt_{pid}",
                        "condition_label": f"btarget={bk:.2f}_arm=unco",
                        "params_json": encode_params(params),
                        "run_idx": r,
                        "seed": 0,
                        "length": T,
                        "sampled_token_ids": list(range(T)),
                        "decoded_text": "",
                        "trace_path": f"traces/{tid}.npz",
                    }
                )
    upsert_manifest(out_dir, rows)


def test_per_prompt_svd_recovers_multiplicative_imprint(tmp_path: Path) -> None:
    """When data is *exactly* multiplicative (ℓ_{p,r}^(k) = β_k · ℓ*_p) plus
    small noise, per-prompt SVD must:
      * give R^(p)(t) ≪ 1 (most variance on principal direction)
      * recover β̄^(p)(k) proportional to β_k

    With 8 runs/prompt and 0.01-scale noise, R^(p) should be ≪ 0.05.
    """
    from decoding_decoding.counter_decode_analyze import (
        f1_svd_per_prompt_at_position,
    )

    beta_dec_values = [0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    sparse_positions = [0, 8, 16]
    _write_synthetic_with_sparse(
        tmp_path,
        n_prompts=4,
        n_runs=8,
        T=20,
        beta_dec_values=beta_dec_values,
        sparse_positions=sparse_positions,
    )
    res = f1_svd_per_prompt_at_position(
        tmp_path, position=8, beta_dec_values=beta_dec_values
    )
    # R^(p)(t) should be small.
    assert res["R_median"] < 0.05, f"R_median too high: {res['R_median']}"

    # Recovered β̄^(p)(k) (per-prompt, geometric-mean-normalized) should
    # be proportional to β_k. The geometric mean of β_k is:
    #   gm(β) = exp(mean(log β_k))
    # so the expected β̄_norm(k) = β_k / gm(β).
    bk = np.asarray(beta_dec_values, dtype=np.float64)
    log_gm = np.log(bk).mean()
    expected_log = np.log(bk) - log_gm
    actual_log = res["log_beta_median_across_prompts"]
    np.testing.assert_allclose(actual_log, expected_log, atol=0.05)


def test_f0_concentration_returns_T_curves(tmp_path: Path) -> None:
    _write_synthetic(
        tmp_path, n_prompts=3, n_runs=2, T=10, beta_target=1.0, corrected=True
    )
    cond = load_counter_condition(tmp_path, beta_target=1.0, corrected=True)
    f0 = f0_concentration(cond, n_resamples=50)
    for s in (f0.max_p_phi, f0.entropy_phi, f0.max_p_sample, f0.entropy_sample):
        assert s.point.shape == (10,)
        assert s.ci_low.shape == (10,)
        assert s.ci_high.shape == (10,)
        # CI-low ≤ point ≤ CI-high.
        assert (s.ci_low <= s.point + 1e-6).all()
        assert (s.point <= s.ci_high + 1e-6).all()
