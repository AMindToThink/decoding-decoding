"""Tests for the streaming Bayesian β estimator.

Critical regression tests:

  * ``test_walkthrough_two_token_numbers`` — replicate the explicit numbers
    from bayesian-sampler/walkthrough.html (steps t=0,1,2 of the toy with
    V={A,B}, ℓ*=(1,-1), 3-point grid). The discrete-grid filter must
    reproduce β̂_t to 3 decimals at every step.

  * ``test_laplace_init_matches_prior_specs`` — η̂_0 = −σ_0²/2 and J_0 = 1/σ_0².

  * ``test_laplace_update_score_invariant_under_logit_shift`` — softmax is
    invariant under additive constants in logits; score and fisher must
    agree under that shift, since softmax-derived q is invariant.

  * ``test_laplace_update_recovers_true_beta_in_long_run`` — given a stream of
    logits ℓ_t = β_true · ℓ* (with x_t sampled from the same distribution), β̂
    converges to β_true under enough data. This is the matched-prior /
    Bayes-optimal toy.

  * ``test_batched_matches_per_element`` — batched update gives the same
    eta_hat / J as running per-element updates serially.
"""

from __future__ import annotations

import numpy as np
import torch

from decoding_decoding.counter_decode import (
    LaplaceState,
    discrete_grid_posterior_mean,
    discrete_grid_update,
    init_laplace,
    laplace_update,
    sample_under_beta,
)


# ---------------------------------------------------------------------------
# Discrete-grid filter — walkthrough regression
# ---------------------------------------------------------------------------


def test_walkthrough_two_token_numbers() -> None:
    """Reproduce walkthrough.html steps t=0,1,2 to 3 decimals.

    Setup: V = {A=0, B=1}, ℓ* = (1, -1), β grid = (0.5, 1.0, 2.0),
    prior = (0.4, 0.4, 0.2). Observation x_t = A at every step.

    Expected (from walkthrough):
      β̂_0 = E_{π_0}[β]                        = 1.000
      π_1 = (0.348, 0.419, 0.234)              ⇒ β̂_1 = 1.060
      π_2 = (0.298, 0.433, 0.269)              ⇒ β̂_2 = 1.120
      π_3 = (0.252, 0.442, 0.306)              ⇒ β̂_3 = 1.180
    """
    beta_grid = np.array([0.5, 1.0, 2.0])
    pi = np.array([0.4, 0.4, 0.2])
    ref = np.array([1.0, -1.0])
    x = 0  # token A

    assert abs(discrete_grid_posterior_mean(pi, beta_grid) - 1.000) < 1e-9

    # Walkthrough values are rounded to 3 decimals. Allow rounding slack.
    pi = discrete_grid_update(pi, beta_grid, ref, sampled_token_id=x)
    np.testing.assert_allclose(pi, np.array([0.348, 0.419, 0.234]), atol=1e-3)
    assert abs(discrete_grid_posterior_mean(pi, beta_grid) - 1.060) < 1e-3

    pi = discrete_grid_update(pi, beta_grid, ref, sampled_token_id=x)
    np.testing.assert_allclose(pi, np.array([0.298, 0.433, 0.269]), atol=1e-3)
    assert abs(discrete_grid_posterior_mean(pi, beta_grid) - 1.120) < 1e-3

    pi = discrete_grid_update(pi, beta_grid, ref, sampled_token_id=x)
    np.testing.assert_allclose(pi, np.array([0.252, 0.442, 0.306]), atol=1e-3)
    assert abs(discrete_grid_posterior_mean(pi, beta_grid) - 1.180) < 1e-3


def test_discrete_grid_unchanged_when_sampled_token_uniformly_likely() -> None:
    """If softmax(β·ℓ*)[x_t] is the same across all β on the grid, the
    posterior must equal the prior. Concretely: at ℓ* = (0, 0), every β
    gives a uniform 50/50, so observing any x doesn't update."""
    beta_grid = np.array([0.5, 1.0, 2.0])
    pi = np.array([0.4, 0.4, 0.2])
    ref = np.array([0.0, 0.0])
    pi_post = discrete_grid_update(pi, beta_grid, ref, sampled_token_id=0)
    np.testing.assert_allclose(pi_post, pi, atol=1e-12)


# ---------------------------------------------------------------------------
# Streaming Laplace filter — invariants
# ---------------------------------------------------------------------------


def test_laplace_init_matches_prior_specs() -> None:
    s = init_laplace(batch_size=4, sigma_0=0.5)
    expected_mu = -0.5 * 0.5 ** 2  # = -0.125
    expected_J = 1.0 / 0.5 ** 2    # = 4.0
    assert torch.allclose(s.eta_hat, torch.full((4,), expected_mu))
    assert torch.allclose(s.J, torch.full((4,), expected_J))
    # β̂_0 = exp(−σ_0²/2) ≈ 0.8825 with σ_0 = 0.5 — close to but below 1.
    assert torch.allclose(s.beta_hat(), torch.full((4,), float(np.exp(-0.125))))


def test_laplace_update_score_invariant_under_logit_shift() -> None:
    """Adding a constant c·1 to logits must not change score or fisher
    (softmax is shift-invariant). Quantitative: score and fisher are
    expectations under softmax, so the shift cancels exactly."""
    rng = np.random.default_rng(0)
    V = 50
    B = 7
    logits = torch.tensor(rng.normal(size=(B, V)), dtype=torch.float64)
    sampled = torch.tensor(rng.integers(0, V, size=B), dtype=torch.long)
    state = init_laplace(batch_size=B, sigma_0=0.5, dtype=torch.float64)

    s1, score1, fisher1 = laplace_update(state, logits, sampled)

    shift = torch.tensor(rng.normal(size=(B, 1)), dtype=torch.float64)
    s2, score2, fisher2 = laplace_update(state, logits + shift, sampled)

    torch.testing.assert_close(score1, score2, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(fisher1, fisher2, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(s1.eta_hat, s2.eta_hat, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(s1.J, s2.J, atol=1e-9, rtol=1e-9)


def test_laplace_score_sign_when_observation_is_max_logit() -> None:
    """If x_t is the argmax of ℓ_t, then ℓ_t[x_t] > E_q[ℓ_t] (Jensen on a
    spread-out softmax), so score > 0 and η̂ moves up — meaning the
    estimator infers β̂ has gotten larger (the model looks sharper)."""
    V = 32
    logits = torch.zeros(1, V)
    logits[0, 7] = 5.0  # spike, but spread among the rest.
    state = init_laplace(batch_size=1, sigma_0=0.5)
    new_state, score, fisher = laplace_update(
        state, logits, torch.tensor([7], dtype=torch.long)
    )
    assert score.item() > 0
    assert fisher.item() > 0
    assert new_state.eta_hat.item() > state.eta_hat.item()


def test_laplace_score_sign_when_observation_is_min_logit() -> None:
    """If x_t is a low-probability token under softmax(ℓ_t), score < 0,
    so η̂ moves down (β̂ shrinks: the model's prediction was less peaked
    than the data implies)."""
    V = 32
    logits = torch.zeros(1, V)
    logits[0, 7] = 5.0
    state = init_laplace(batch_size=1, sigma_0=0.5)
    new_state, score, _ = laplace_update(
        state, logits, torch.tensor([0], dtype=torch.long)  # not the argmax
    )
    assert score.item() < 0
    assert new_state.eta_hat.item() < state.eta_hat.item()


def test_batched_matches_per_element() -> None:
    """Running B trajectories in a batched call must match running each one
    serially in a batch-of-1 call. Key consistency check for vectorization."""
    rng = np.random.default_rng(1)
    V = 64
    B = 5
    state = init_laplace(batch_size=B, sigma_0=0.5, dtype=torch.float64)

    # Apply 4 sequential updates with random logits and sampled tokens.
    history_logits = []
    history_sampled = []
    for _ in range(4):
        logits = torch.tensor(rng.normal(size=(B, V)), dtype=torch.float64)
        sampled = torch.tensor(rng.integers(0, V, size=B), dtype=torch.long)
        history_logits.append(logits)
        history_sampled.append(sampled)
        state, _, _ = laplace_update(state, logits, sampled)

    # Now redo each batch element separately and compare.
    for b in range(B):
        s_b = init_laplace(batch_size=1, sigma_0=0.5, dtype=torch.float64)
        for logits, sampled in zip(history_logits, history_sampled):
            s_b, _, _ = laplace_update(s_b, logits[b:b+1], sampled[b:b+1])
        torch.testing.assert_close(state.eta_hat[b], s_b.eta_hat[0])
        torch.testing.assert_close(state.J[b], s_b.J[0])


def test_laplace_update_recovers_true_beta_in_long_run() -> None:
    """Set up a synthetic stream where ℓ_t = β_true · ℓ_ref each step and
    the sampled tokens are drawn from softmax(β_true·ℓ_ref) (matched-prior
    Bayes-optimality). β̂ should converge toward β_true.

    This is the toy that the walkthrough makes explicit — but here using
    the streaming Laplace filter rather than the discrete grid.
    """
    rng = torch.Generator().manual_seed(7)
    sigma_0 = 0.5
    beta_true = 1.6
    V = 20
    # Pick a generic reference logit shape.
    ell_ref = torch.linspace(2.0, -1.5, V, dtype=torch.float64)
    state = init_laplace(batch_size=1, sigma_0=sigma_0, dtype=torch.float64)

    # Sample from softmax(β_true · ℓ_ref) for each step's "observation".
    probs = torch.softmax(beta_true * ell_ref, dim=-1)
    n_steps = 4000
    samples = torch.multinomial(probs, num_samples=n_steps, replacement=True, generator=rng)

    # Apply updates — at each step the model "outputs" β̂·ℓ_ref logits.
    # Note: under matched-prior Bayes-optimality, the model's β̂ at each step
    # equals our β̂. We feed in β̂_t · ℓ_ref as the logits at step t. This is
    # exactly what the walkthrough does for the toy.
    eta_trace = []
    for t in range(n_steps):
        beta_hat_now = float(state.beta_hat().item())
        logits_t = (beta_hat_now * ell_ref).unsqueeze(0)  # (1, V)
        x = samples[t:t+1].to(torch.long)
        state, _, _ = laplace_update(state, logits_t, x)
        eta_trace.append(state.eta_hat.item())

    final_beta = float(state.beta_hat().item())
    # With 4000 obs we expect convergence to within ~5% of β_true.
    assert abs(np.log(final_beta) - np.log(beta_true)) < 0.05, (
        f"final β̂={final_beta:.3f} vs β_true={beta_true}; logs differ by "
        f"{np.log(final_beta) - np.log(beta_true):+.3f}"
    )
    # Posterior std should also have shrunk meaningfully from the prior.
    final_std = float(state.posterior_std_log_beta().item())
    assert final_std < 0.5 * sigma_0, f"posterior std {final_std} did not shrink enough"


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def test_sample_under_beta_zero_temperature_picks_argmax() -> None:
    """As β_dec → ∞, sampling concentrates on the argmax."""
    V = 16
    logits = torch.tensor([[0.1] * V])
    logits[0, 5] = 5.0
    beta = torch.tensor([100.0])
    g = torch.Generator().manual_seed(0)
    counts = np.zeros(V, dtype=np.int64)
    for _ in range(200):
        x = sample_under_beta(logits, beta, generator=g)
        counts[x.item()] += 1
    # All draws should be the argmax (5).
    assert counts[5] == 200, f"argmax sampling not concentrated: {counts}"


def test_sample_under_beta_uniform_at_zero() -> None:
    """β_dec = 0 collapses the distribution to uniform."""
    V = 8
    logits = torch.zeros(1, V)
    logits[0, 0] = 100.0  # huge logit but β=0 zeros it out
    beta = torch.tensor([0.0])
    g = torch.Generator().manual_seed(0)
    counts = np.zeros(V, dtype=np.int64)
    for _ in range(2000):
        x = sample_under_beta(logits, beta, generator=g)
        counts[x.item()] += 1
    # Each token should appear ~250 times under uniform; loose check.
    assert (counts > 100).all(), f"distribution not uniform under β=0: {counts}"
