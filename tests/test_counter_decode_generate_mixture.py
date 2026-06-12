"""Integration tests for the grid-mixture (certificate) backend wired into the
generation loop (`_generate_batch`, prior_kind="grid_mixture").

The underlying certificate math (batched=serial, regret ≤ ln G, level-set bar,
Fixed-Share, bad-arg rejection) is covered directly in ``test_grid_mixture.py``.
These tests instead exercise the *wiring* in ``counter_decode_generate.py``:

  * the new branch reads β̂ from the grid MAP and the diagnostic slots carry the
    documented certificate quantities (J_pre = level-set width, score = realized
    regret, fisher = regret bound);
  * the per-sequence guarantee (realized regret ≤ regret bound) holds end to end;
  * β̂ tracks the *effective* temperature of the sampled stream (higher β_target
    ⇒ higher recovered β), confirming the loop feeds RAW logits + the actually
    sampled tokens to the filter;
  * evidence_weight / memory_decay / stray prior_kwargs are rejected up front.

A tiny deterministic fake model/tokenizer stands in for Qwen so the tests run on
CPU in milliseconds. The fake emits a FIXED logit direction at every position,
so the stream is stationary and its effective inverse-temperature is exactly the
arm's β_dec (= β_target in the uncorrected arm).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from decoding_decoding.counter_decode_generate import (
    GRID_MIXTURE_LOG_HI,
    GRID_MIXTURE_LOG_LO,
    GRID_MIXTURE_POINTS,
    _generate_batch,
)


# ---------------------------------------------------------------------------
# Tiny deterministic fake model + tokenizer (CPU, no transformers dependency).
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class _FakeOutput:
    def __init__(self, logits: torch.Tensor, past_key_values) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeModel:
    """Returns a fixed logit *direction* at every position, ignoring inputs.

    With identical logits each step, the stream a trajectory produces is
    stationary and its best-fit inverse temperature equals the β_dec it was
    sampled under — which is what the certificate filter should recover.
    """

    def __init__(self, direction: torch.Tensor) -> None:
        self.direction = direction                 # (V,) float32
        self.config = _FakeConfig(direction.shape[0])

    def __call__(self, *, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        B, seq = input_ids.shape
        V = self.direction.shape[0]
        logits = self.direction.view(1, 1, V).expand(B, seq, V).contiguous()
        return _FakeOutput(logits=logits, past_key_values=(past_key_values or 0) + 1)


class _FakeTokenizer:
    """Maps each prompt to a 1-token id; pads a batch to equal length."""

    def __call__(self, prompts, return_tensors="pt", padding=True):
        B = len(prompts)
        input_ids = torch.arange(1, B + 1, dtype=torch.long).unsqueeze(-1)   # (B, 1)
        attention_mask = torch.ones((B, 1), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


def _run(
    *,
    beta_target: float,
    corrected: bool = False,
    max_tokens: int = 120,
    vocab: int = 32,
    n_traj: int = 4,
    direction_span: float = 1.2,
    seed_base: int = 0,
    **batch_kwargs,
):
    torch.manual_seed(0)
    direction = torch.linspace(-direction_span, direction_span, vocab, dtype=torch.float32)
    model = _FakeModel(direction)
    tok = _FakeTokenizer()
    prompts = [f"p{i}" for i in range(n_traj)]
    return _generate_batch(
        model=model,
        tokenizer=tok,
        prompts=prompts,
        beta_target_per_traj=torch.full((n_traj,), float(beta_target)),
        corrected_per_traj=torch.full((n_traj,), bool(corrected)),
        sigma_0_per_traj=torch.full((n_traj,), 0.5),
        seeds_per_traj=list(range(seed_base, seed_base + n_traj)),
        max_tokens=max_tokens,
        top_n=5,
        sparse_positions=None,
        save_sparse_for_mask=torch.zeros(n_traj, dtype=torch.bool),
        device="cpu",
        prior_kind="grid_mixture",
        **batch_kwargs,
    )


# ---------------------------------------------------------------------------
# Wiring: readouts are finite, in-grid, right shape.
# ---------------------------------------------------------------------------


def test_grid_mixture_readouts_are_sane() -> None:
    res = _run(beta_target=1.0)
    beta_lo, beta_hi = math.exp(GRID_MIXTURE_LOG_LO), math.exp(GRID_MIXTURE_LOG_HI)

    # β̂ (grid MAP) is finite, positive, and strictly inside the grid range.
    assert np.isfinite(res.beta_hat_pre).all()
    assert (res.beta_hat_pre > 0).all()
    assert (res.beta_hat_pre >= beta_lo - 1e-6).all()
    assert (res.beta_hat_pre <= beta_hi + 1e-6).all()

    # J_pre carries the level-set WIDTH in log β: non-negative, finite, and at
    # the start (no evidence) it spans essentially the whole grid range.
    assert np.isfinite(res.J_pre).all()
    assert (res.J_pre >= -1e-6).all()
    full_range = GRID_MIXTURE_LOG_HI - GRID_MIXTURE_LOG_LO
    assert res.J_pre[:, 0].mean() > 0.9 * full_range  # step 0: bar ≈ whole grid

    # Final readouts well-formed.
    assert np.isfinite(res.eta_hat_post_final).all()
    assert (res.J_post_final >= -1e-6).all()
    assert res.beta_hat_pre.shape == (4, 120)


def test_grid_mixture_bar_narrows_with_evidence() -> None:
    """The level-set bar (J_pre slot) should shrink as the stream accumulates."""
    res = _run(beta_target=1.0)
    early = res.J_pre[:, 1].mean()      # after one token (step 0 is full-grid)
    late = res.J_pre[:, -1].mean()
    assert late < early


# ---------------------------------------------------------------------------
# The per-sequence guarantee holds end to end.
# ---------------------------------------------------------------------------


def test_grid_mixture_realized_regret_within_bound() -> None:
    """score slot = realized regret ≥ 0 and ≤ fisher slot = regret bound (ln G)."""
    res = _run(beta_target=1.5)
    realized = res.score      # realized regret
    bound = res.fisher        # regret bound = ln(1/π₀[best]); uniform ⇒ ln G

    assert (realized >= -1e-5).all()
    assert (realized <= bound + 1e-4).all()
    # Uniform prior over G grid points ⇒ bound ≈ ln G at every step.
    assert np.allclose(bound, math.log(GRID_MIXTURE_POINTS), atol=1e-4)


# ---------------------------------------------------------------------------
# β̂ tracks the stream's effective temperature (the loop feeds RAW logits +
# the actually sampled tokens to the filter).
# ---------------------------------------------------------------------------


def test_grid_mixture_recovers_effective_temperature_monotonic() -> None:
    """Higher β_target ⇒ sharper sampling ⇒ higher recovered grid-MAP β.

    Uncorrected arm, so β_dec = β_target and the stream's effective inverse
    temperature is exactly β_target. We check the final-token MAP both for
    monotonicity and for landing near log β_target (within a tolerance set by
    sampling noise + grid resolution, not tuned to the data).
    """
    targets = [0.5, 1.0, 2.0]
    final_log_map = []
    for bt in targets:
        res = _run(beta_target=bt, max_tokens=160)
        # mean over trajectories of the final-step β̂, in log β.
        final_log_map.append(float(np.log(res.beta_hat_pre[:, -1]).mean()))

    # Monotone increasing in β_target.
    assert final_log_map[0] < final_log_map[1] < final_log_map[2]
    # Each lands near the truth (≤ ~0.5 nats ≈ 10 grid steps of slack).
    for bt, lm in zip(targets, final_log_map):
        assert abs(lm - math.log(bt)) < 0.5, (bt, lm, math.log(bt))


# ---------------------------------------------------------------------------
# Incompatible knobs are rejected up front (fail loud, before any model call).
# ---------------------------------------------------------------------------


def test_grid_mixture_rejects_evidence_weight() -> None:
    with pytest.raises(ValueError, match="evidence_weight"):
        _run(beta_target=1.0, evidence_weight=2.0)


def test_grid_mixture_rejects_memory_decay() -> None:
    with pytest.raises(ValueError, match="memory_decay"):
        _run(beta_target=1.0, memory_decay=0.9)


def test_grid_mixture_rejects_unknown_prior_kwargs() -> None:
    with pytest.raises(ValueError, match="unexpected prior_kwargs"):
        _run(beta_target=1.0, prior_kwargs={"sigma_0": 0.3})


def test_grid_mixture_accepts_grid_prior_kwargs() -> None:
    """log_lo/log_hi/n_points reshape the grid without error."""
    res = _run(beta_target=1.0, prior_kwargs={"log_lo": -2.0, "log_hi": 2.0, "n_points": 81})
    # Regret bound now ln(81), confirming the smaller grid took effect.
    assert np.allclose(res.fisher, math.log(81), atol=1e-4)
    assert (res.beta_hat_pre >= math.exp(-2.0) - 1e-6).all()
    assert (res.beta_hat_pre <= math.exp(2.0) + 1e-6).all()


def test_grid_mixture_switch_rate_threads_through() -> None:
    """switch_rate > 0 (Fixed-Share) runs without error and stays in-grid."""
    res = _run(beta_target=1.0, switch_rate=0.02)
    assert np.isfinite(res.beta_hat_pre).all()
    assert (res.beta_hat_pre > 0).all()
