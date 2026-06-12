"""Tests for the guarantee-bearing grid mixture filter (`grid_mixture_*`).

The filter is the exponentially-weighted-average forecaster / Bayes mixture
over a grid of inverse-temperature experts P_{β_g}(x) = softmax(β_g · ℓ_t)[x],
scored on RAW logits (no bootstrap rescaling). Properties under test:

  (a) telescoping: posterior weights equal π₀ · exp(−L_n) renormalized,
      and the cumulative mixture loss equals −ln Σ_g π₀[g] · exp(−L_n[g]);
  (b) MAP = argmin over the grid of the cumulative loss (uniform prior);
  (c) per-sequence regret ≤ ln(1/π₀[best]) on an adversarially chosen
      token sequence (worst-case, not sampled);
  (d) the level-set error bar covers the continuum MLE on well-specified
      synthetic streams;
  (e) the bar is wider on peaked-logit streams than on natural-scale ones;
  (f) Fixed-Share (switch_rate > 0) tracks a mid-stream β switch better
      than the pure Bayes mixture, and keeps weights a valid pmf.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

from decoding_decoding.counter_decode import (
    grid_mixture_level_set,
    grid_mixture_update_batched,
    init_grid_mixture,
    make_beta_grid,
)


def _cumulative_loss_numpy(
    beta_grid: np.ndarray, logits_seq: np.ndarray, tokens: np.ndarray
) -> np.ndarray:
    """Reference L_n(β_g) = Σ_t [logsumexp(β_g ℓ_t) − β_g ℓ_t[x_t]], shape (G,)."""
    G = beta_grid.shape[0]
    total = np.zeros(G, dtype=np.float64)
    for t in range(logits_seq.shape[0]):
        scaled = beta_grid[:, None] * logits_seq[t][None, :]  # (G, V)
        m = scaled.max(axis=-1, keepdims=True)
        lse = np.log(np.exp(scaled - m).sum(axis=-1)) + m.squeeze(-1)
        total += lse - scaled[:, tokens[t]]
    return total


def _continuum_loss(beta: float, logits_seq: np.ndarray, tokens: np.ndarray) -> float:
    scaled = beta * logits_seq  # (T, V)
    m = scaled.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(scaled - m).sum(axis=-1)) + m.squeeze(-1)  # (T,)
    return float((lse - scaled[np.arange(len(tokens)), tokens]).sum())


def _run_filter(
    beta_grid: torch.Tensor,
    logits_seq: torch.Tensor,
    tokens_seq: torch.Tensor,
    *,
    prior_pmf: torch.Tensor | None = None,
    switch_rate: float = 0.0,
):
    """Stream a (T, B, V) logits / (T, B) tokens sequence through the filter."""
    B = logits_seq.shape[1]
    state = init_grid_mixture(batch_size=B, beta_grid=beta_grid, prior_pmf=prior_pmf)
    for t in range(logits_seq.shape[0]):
        state = grid_mixture_update_batched(
            state, logits_seq[t], tokens_seq[t], switch_rate=switch_rate
        )
    return state


def _synthetic_stream(
    rng: np.random.Generator,
    *,
    n_steps: int,
    vocab: int,
    beta_true: float,
    logit_scale: float,
    dtype: torch.dtype = torch.float64,
):
    """Well-specified stream: logits ~ N(0, scale²), tokens ~ softmax(β·ℓ)."""
    logits = rng.normal(0.0, logit_scale, size=(n_steps, 1, vocab))
    scaled = beta_true * logits[:, 0, :]
    p = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    tokens = np.array(
        [rng.choice(vocab, p=p[t]) for t in range(n_steps)], dtype=np.int64
    )
    return (
        torch.tensor(logits, dtype=dtype),
        torch.tensor(tokens[:, None], dtype=torch.int64),
    )


class TestTelescoping:
    def test_weights_match_prior_times_exp_neg_loss(self) -> None:
        rng = np.random.default_rng(0)
        beta_grid = make_beta_grid(log_lo=-2.0, log_hi=2.0, n_points=17, dtype=torch.float64)
        logits, tokens = _synthetic_stream(
            rng, n_steps=50, vocab=20, beta_true=1.3, logit_scale=2.0
        )
        state = _run_filter(beta_grid, logits, tokens)

        L = _cumulative_loss_numpy(
            beta_grid.numpy(), logits[:, 0, :].numpy(), tokens[:, 0].numpy()
        )
        np.testing.assert_allclose(
            state.cum_loss[0].numpy(), L, rtol=0, atol=1e-8
        )
        # log w ∝ log π₀ − L_n  (uniform prior ⇒ just −L_n), renormalized.
        expected_log_w = -L - np.log(np.exp(-L + L.min()).sum()) + L.min()
        np.testing.assert_allclose(
            state.log_w[0].numpy(), expected_log_w, rtol=0, atol=1e-8
        )

    def test_mixture_loss_is_log_marginal(self) -> None:
        rng = np.random.default_rng(1)
        beta_grid = make_beta_grid(log_lo=-2.0, log_hi=2.0, n_points=9, dtype=torch.float64)
        logits, tokens = _synthetic_stream(
            rng, n_steps=30, vocab=15, beta_true=0.8, logit_scale=1.5
        )
        state = _run_filter(beta_grid, logits, tokens)

        L = _cumulative_loss_numpy(
            beta_grid.numpy(), logits[:, 0, :].numpy(), tokens[:, 0].numpy()
        )
        G = beta_grid.shape[0]
        # −ln Σ_g π₀[g] e^{−L[g]} with uniform π₀ = 1/G.
        m = (-L).max()
        expected = -(m + np.log(np.exp(-L - m).sum() / G))
        assert state.mix_loss[0].item() == pytest.approx(expected, abs=1e-8)


class TestMapIdentity:
    def test_map_is_grid_argmin_of_cumulative_loss(self) -> None:
        rng = np.random.default_rng(2)
        beta_grid = make_beta_grid(log_lo=-3.0, log_hi=3.0, n_points=61, dtype=torch.float64)
        logits, tokens = _synthetic_stream(
            rng, n_steps=200, vocab=30, beta_true=2.0, logit_scale=1.0
        )
        state = _run_filter(beta_grid, logits, tokens)

        L = _cumulative_loss_numpy(
            beta_grid.numpy(), logits[:, 0, :].numpy(), tokens[:, 0].numpy()
        )
        assert state.beta_map()[0].item() == pytest.approx(
            beta_grid[int(np.argmin(L))].item()
        )


class TestAdversarialRegret:
    def test_regret_bounded_by_ln_K_on_adversarial_tokens(self) -> None:
        """Tokens chosen to MINIMIZE the mixture's predicted probability each step."""
        rng = np.random.default_rng(3)
        n_steps, vocab = 300, 12
        beta_grid = make_beta_grid(log_lo=-2.0, log_hi=2.0, n_points=33, dtype=torch.float64)
        G = beta_grid.shape[0]
        state = init_grid_mixture(batch_size=1, beta_grid=beta_grid)

        for _ in range(n_steps):
            logits = torch.tensor(
                rng.normal(0.0, 3.0, size=(1, vocab)), dtype=torch.float64
            )
            # Mixture predictive probability of each candidate token.
            scaled = beta_grid.view(G, 1) * logits[0].view(1, vocab)  # (G, V)
            log_pred = torch.logsumexp(
                state.log_w[0].view(G, 1) + torch.log_softmax(scaled, dim=-1), dim=0
            )  # (V,)
            worst = int(torch.argmin(log_pred).item())
            state = grid_mixture_update_batched(
                state, logits, torch.tensor([worst], dtype=torch.int64)
            )

        regret = state.realized_regret()[0].item()
        assert regret <= math.log(G) + 1e-6
        # The bound should be doing real work: adversarial play gets close to it.
        assert regret > 0.0

    def test_regret_bound_respects_nonuniform_prior(self) -> None:
        rng = np.random.default_rng(4)
        beta_grid = make_beta_grid(log_lo=-2.0, log_hi=2.0, n_points=17, dtype=torch.float64)
        prior = torch.tensor(
            rng.dirichlet(np.ones(17)), dtype=torch.float64
        )
        logits, tokens = _synthetic_stream(
            rng, n_steps=100, vocab=10, beta_true=1.0, logit_scale=2.0
        )
        state = _run_filter(beta_grid, logits, tokens, prior_pmf=prior)

        # mix_loss ≤ L_n[g] + ln(1/π₀[g]) for EVERY expert g.
        bound = state.cum_loss[0] - torch.log(prior)
        assert state.mix_loss[0].item() <= bound.min().item() + 1e-8


class TestLevelSetCertificate:
    def test_level_set_covers_continuum_mle_well_specified(self) -> None:
        rng = np.random.default_rng(5)
        beta_grid = make_beta_grid(log_lo=-3.0, log_hi=3.0, n_points=121, dtype=torch.float64)
        for seed in range(5):
            sub_rng = np.random.default_rng(100 + seed)
            logits, tokens = _synthetic_stream(
                sub_rng, n_steps=400, vocab=25, beta_true=1.0, logit_scale=1.5
            )
            state = _run_filter(beta_grid, logits, tokens)
            lo, hi = grid_mixture_level_set(state)

            res = minimize_scalar(
                lambda b: _continuum_loss(
                    b, logits[:, 0, :].numpy(), tokens[:, 0].numpy()
                ),
                bounds=(math.exp(-3.0), math.exp(3.0)),
                method="bounded",
            )
            log_mle = math.log(res.x)
            # Allow one grid-spacing of slack on each side (discretization).
            spacing = 6.0 / 120
            assert lo[0].item() - spacing <= log_mle <= hi[0].item() + spacing

    def test_bar_wider_on_peaked_logits(self) -> None:
        rng_a = np.random.default_rng(6)
        rng_b = np.random.default_rng(6)
        beta_grid = make_beta_grid(log_lo=-3.0, log_hi=3.0, n_points=121, dtype=torch.float64)

        logits_nat, tokens_nat = _synthetic_stream(
            rng_a, n_steps=300, vocab=25, beta_true=1.0, logit_scale=1.5
        )
        logits_peak, tokens_peak = _synthetic_stream(
            rng_b, n_steps=300, vocab=25, beta_true=1.0, logit_scale=8.0
        )
        state_nat = _run_filter(beta_grid, logits_nat, tokens_nat)
        state_peak = _run_filter(beta_grid, logits_peak, tokens_peak)

        lo_n, hi_n = grid_mixture_level_set(state_nat)
        lo_p, hi_p = grid_mixture_level_set(state_peak)
        width_nat = (hi_n - lo_n)[0].item()
        width_peak = (hi_p - lo_p)[0].item()
        assert width_peak > width_nat


class TestFixedShare:
    def test_weights_remain_valid_pmf(self) -> None:
        rng = np.random.default_rng(7)
        beta_grid = make_beta_grid(log_lo=-2.0, log_hi=2.0, n_points=17, dtype=torch.float64)
        logits, tokens = _synthetic_stream(
            rng, n_steps=50, vocab=10, beta_true=1.0, logit_scale=2.0
        )
        state = _run_filter(beta_grid, logits, tokens, switch_rate=0.05)
        total = torch.exp(state.log_w[0]).sum().item()
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_fixed_share_tracks_beta_switch_better_than_pure_bayes(self) -> None:
        """β jumps 0.5 → 2.0 mid-stream; Fixed-Share's second-half mixture loss
        must beat pure Bayes (whose posterior has ossified on 0.5)."""
        n_half, vocab = 250, 30
        beta_grid = make_beta_grid(log_lo=-2.0, log_hi=2.0, n_points=33, dtype=torch.float64)

        rng = np.random.default_rng(8)
        logits_1, tokens_1 = _synthetic_stream(
            rng, n_steps=n_half, vocab=vocab, beta_true=0.5, logit_scale=3.0
        )
        logits_2, tokens_2 = _synthetic_stream(
            rng, n_steps=n_half, vocab=vocab, beta_true=2.0, logit_scale=3.0
        )
        logits = torch.cat([logits_1, logits_2], dim=0)
        tokens = torch.cat([tokens_1, tokens_2], dim=0)

        second_half_loss: dict[float, float] = {}
        for rate in (0.0, 0.02):
            state = init_grid_mixture(batch_size=1, beta_grid=beta_grid)
            mid_loss = 0.0
            for t in range(2 * n_half):
                state = grid_mixture_update_batched(
                    state, logits[t], tokens[t], switch_rate=rate
                )
                if t == n_half - 1:
                    mid_loss = state.mix_loss[0].item()
            second_half_loss[rate] = state.mix_loss[0].item() - mid_loss

        assert second_half_loss[0.02] < second_half_loss[0.0]


class TestValidation:
    def test_bad_prior_shape_raises(self) -> None:
        beta_grid = make_beta_grid(n_points=9)
        with pytest.raises(ValueError):
            init_grid_mixture(
                batch_size=1, beta_grid=beta_grid, prior_pmf=torch.ones(5) / 5
            )

    def test_bad_switch_rate_raises(self) -> None:
        beta_grid = make_beta_grid(n_points=9, dtype=torch.float64)
        state = init_grid_mixture(batch_size=1, beta_grid=beta_grid)
        logits = torch.zeros((1, 4), dtype=torch.float64)
        tokens = torch.zeros((1,), dtype=torch.int64)
        with pytest.raises(ValueError):
            grid_mixture_update_batched(state, logits, tokens, switch_rate=1.0)
        with pytest.raises(ValueError):
            grid_mixture_update_batched(state, logits, tokens, switch_rate=-0.1)
