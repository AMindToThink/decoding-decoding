"""Tests for the grid-mixture (certificate) backend wired into the generation loop.

The math of ``grid_mixture_update_batched`` / ``grid_mixture_level_set`` is covered
by ``tests/test_grid_mixture.py``. This file tests the *wiring* added to
``counter_decode_generate._generate_batch`` for ``prior_kind="grid_mixture"``:

  (a) the bootstrap knobs (evidence_weight, memory_decay) and unknown prior_kwargs
      are rejected — they would void the per-sequence guarantee;
  (b) an end-to-end run recovers a known β on a well-specified synthetic stream;
  (c) the loop applies EXACTLY the canonical update (a step-by-step replay of
      ``grid_mixture_update_batched`` on the recorded tokens reproduces the final
      grid MAP bit-for-bit);
  (d) the realized-regret diagnostic obeys the Lemma-(i) bound 0 ≤ regret ≤ ln G,
      and the level-set width recorded in the J slot stays inside the grid span.

A tiny deterministic fake model (constant logits each step) stands in for Qwen so
these run on CPU in milliseconds without a network or GPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from decoding_decoding.counter_decode import (
    GridMixtureState,
    grid_mixture_update_batched,
    init_grid_mixture,
    make_beta_grid,
)
from decoding_decoding.counter_decode_generate import (
    _generate_batch,
    _init_grid_mixture_state,
)


# ---------------------------------------------------------------------------
# Minimal fake HF model / tokenizer (constant logits, CPU-only)
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, logits: torch.Tensor, past) -> None:
        self.logits = logits
        self.past_key_values = past


class _FakeConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class _FakeModel:
    """Returns the same base logit vector at every position/step (well-specified)."""

    def __init__(self, base_logits: torch.Tensor) -> None:
        self.base = base_logits                      # (V,)
        self.config = _FakeConfig(base_logits.shape[0])

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, input_ids, attention_mask=None, past_key_values=None,
                 use_cache=True):
        B, S = input_ids.shape
        V = self.base.shape[0]
        logits = self.base.view(1, 1, V).expand(B, S, V).to(input_ids.device)
        return _FakeOutput(logits, past_key_values)


class _FakeTokenizer:
    """Maps every prompt to a fixed 3-token sequence; full attention; trivial decode."""

    padding_side = "left"
    pad_token_id = 0
    eos_token = None
    pad_token = None

    def __call__(self, prompts, return_tensors="pt", padding=True):
        B = len(prompts)
        return {
            "input_ids": torch.zeros(B, 3, dtype=torch.long),
            "attention_mask": torch.ones(B, 3, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=False) -> str:
        return "x"


# ---------------------------------------------------------------------------
# (a) kwarg rejection — helper level
# ---------------------------------------------------------------------------


def test_init_grid_mixture_state_rejects_evidence_weight() -> None:
    with pytest.raises(ValueError, match="evidence_weight"):
        _init_grid_mixture_state(
            batch_size=1, prior_kwargs=None,
            evidence_weight=0.5, memory_decay=1.0, device="cpu",
        )


def test_init_grid_mixture_state_rejects_memory_decay() -> None:
    with pytest.raises(ValueError, match="memory_decay"):
        _init_grid_mixture_state(
            batch_size=1, prior_kwargs=None,
            evidence_weight=1.0, memory_decay=0.9, device="cpu",
        )


def test_init_grid_mixture_state_rejects_unknown_kwargs() -> None:
    with pytest.raises(ValueError, match="unknown prior_kwargs"):
        _init_grid_mixture_state(
            batch_size=1, prior_kwargs={"bogus": 1.0},
            evidence_weight=1.0, memory_decay=1.0, device="cpu",
        )


def test_init_grid_mixture_state_parses_switch_rate_and_grid() -> None:
    state, switch_rate = _init_grid_mixture_state(
        batch_size=2,
        prior_kwargs={"switch_rate": 0.05, "n_points": 81, "log_lo": -3.0, "log_hi": 3.0},
        evidence_weight=1.0, memory_decay=1.0, device="cpu",
    )
    assert switch_rate == pytest.approx(0.05)
    assert isinstance(state, GridMixtureState)
    assert state.beta_grid.shape == (81,)
    # Uniform prior over the grid.
    expected_log_prior = -math.log(81)
    assert torch.allclose(
        state.log_prior, torch.full((81,), expected_log_prior), atol=1e-5
    )
    # Grid endpoints honour the requested log range.
    assert state.beta_grid[0].item() == pytest.approx(math.exp(-3.0), rel=1e-5)
    assert state.beta_grid[-1].item() == pytest.approx(math.exp(3.0), rel=1e-5)


# ---------------------------------------------------------------------------
# (a) kwarg rejection — integration through _generate_batch
# ---------------------------------------------------------------------------


def test_generate_batch_grid_mixture_rejects_evidence_weight() -> None:
    torch.manual_seed(0)
    model = _FakeModel(torch.randn(20))
    tok = _FakeTokenizer()
    with pytest.raises(ValueError, match="evidence_weight"):
        _generate_batch(
            model=model, tokenizer=tok, prompts=["p"],
            beta_target_per_traj=torch.tensor([1.0]),
            corrected_per_traj=torch.tensor([False]),
            sigma_0_per_traj=torch.tensor([0.5]),
            seeds_per_traj=[1], max_tokens=4, top_n=5,
            sparse_positions=None, save_sparse_for_mask=torch.tensor([False]),
            device="cpu", prior_kind="grid_mixture", evidence_weight=0.5,
        )


# ---------------------------------------------------------------------------
# (b)-(d) end-to-end: recovery, exact replay, and the regret bound
# ---------------------------------------------------------------------------


def _run_grid_mixture_batch(beta_target: float, max_tokens: int, vocab: int = 40):
    torch.manual_seed(7)
    base = torch.randn(vocab)
    model = _FakeModel(base)
    tok = _FakeTokenizer()
    result = _generate_batch(
        model=model, tokenizer=tok, prompts=["p"],
        beta_target_per_traj=torch.tensor([beta_target]),
        corrected_per_traj=torch.tensor([False]),   # uncorrected: β̂ measured, not fed back
        sigma_0_per_traj=torch.tensor([0.5]),
        seeds_per_traj=[123], max_tokens=max_tokens, top_n=5,
        sparse_positions=None, save_sparse_for_mask=torch.tensor([False]),
        device="cpu", prior_kind="grid_mixture",
    )
    return base, result


def test_generate_batch_grid_mixture_recovers_known_beta() -> None:
    beta_target = 1.5
    _, result = _run_grid_mixture_batch(beta_target, max_tokens=400)
    # eta_hat_post_final = log(grid MAP). On a well-specified constant-logit stream
    # the grid MLE projects onto the true β (the family contains it exactly).
    recovered_log_beta = float(result.eta_hat_post_final[0])
    assert abs(recovered_log_beta - math.log(beta_target)) < 0.15


def test_generate_batch_grid_mixture_exact_replay() -> None:
    """The loop must apply exactly the canonical update on the recorded tokens."""
    base, result = _run_grid_mixture_batch(1.5, max_tokens=120)
    sampled = torch.from_numpy(result.sampled_token_ids[0]).long()   # (T,)
    T = sampled.shape[0]
    V = base.shape[0]

    state = init_grid_mixture(batch_size=1, beta_grid=make_beta_grid(n_points=161))
    for t in range(T):
        state = grid_mixture_update_batched(
            state, base.view(1, V), sampled[t : t + 1]
        )
    map_replay = state.beta_map()[0].item()
    # Final loop MAP (= exp(eta_hat_post_final)) lands on the same grid point.
    assert math.exp(float(result.eta_hat_post_final[0])) == pytest.approx(
        map_replay, rel=1e-5
    )


def test_generate_batch_grid_mixture_corrected_arm_no_spurious_t0_correction() -> None:
    """Corrected arm: the t=0 prior sentinel β̂=1 ⟹ β_dec = β_target at step 0,
    for any correction_exponent (1^p = 1). This guards the degenerate-MAP fix:
    without the sentinel the uniform-prior argmax ties to the smallest grid β and
    the first corrected step would over-correct by β_target / β_min."""
    torch.manual_seed(11)
    base = torch.randn(30)
    model = _FakeModel(base)
    tok = _FakeTokenizer()
    beta_target = 1.7
    for p in (1.0, 0.5):
        result = _generate_batch(
            model=model, tokenizer=tok, prompts=["p"],
            beta_target_per_traj=torch.tensor([beta_target]),
            corrected_per_traj=torch.tensor([True]),
            sigma_0_per_traj=torch.tensor([0.5]),
            seeds_per_traj=[5], max_tokens=6, top_n=5,
            sparse_positions=None, save_sparse_for_mask=torch.tensor([False]),
            device="cpu", prior_kind="grid_mixture", correction_exponent=p,
        )
        # Step 0: β̂=1 sentinel ⟹ β_dec = β_target / 1^p = β_target.
        assert result.beta_dec[0][0] == pytest.approx(beta_target, rel=1e-5)
        # Later steps generally correct away from β_target (β̂ ≠ 1 once tokens land);
        # at minimum β_dec stays finite and positive.
        assert np.all(result.beta_dec[0] > 0.0)
        assert np.all(np.isfinite(result.beta_dec[0]))


def test_generate_batch_grid_mixture_diagnostics_bounds() -> None:
    base, result = _run_grid_mixture_batch(1.5, max_tokens=200)
    G = 161
    ln_G = math.log(G)
    grid_span = 8.0  # log_hi - log_lo for the default [-4, 4] grid

    # fisher slot holds the realized mixture regret: 0 ≤ regret ≤ ln G (Lemma (i)).
    regret = result.fisher[0]
    assert np.all(regret >= -1e-5)
    assert np.all(regret <= ln_G + 1e-4)

    # J slot holds the level-set width in log β: non-negative, within the grid span.
    width = result.J_pre[0]
    assert np.all(width >= -1e-5)
    assert np.all(width <= grid_span + 1e-4)

    # β̂ stays inside the grid; the t=0 prior point estimate is the grid centre β=1.
    beta_hat = result.beta_hat_pre[0]
    assert beta_hat[0] == pytest.approx(1.0)
    assert np.all(beta_hat >= math.exp(-4.0) - 1e-4)
    assert np.all(beta_hat <= math.exp(4.0) + 1e-2)
