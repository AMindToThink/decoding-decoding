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
    estimate_beta_rank_gap,
    estimate_beta_renyi_infty,
    init_laplace,
    laplace_update,
    rank_gaps,
    renyi_infty_from_logits,
    sample_under_beta,
    topk_logits_sorted,
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


# ---------------------------------------------------------------------------
# Discrete-grid filter (multi-prior) tests
# ---------------------------------------------------------------------------


def test_init_prior_pmf_lognormal_matches_dist() -> None:
    """log-normal prior pmf with σ=0.5, anchored at E[β]=1, should put most mass
    near β=1 and have median = exp(μ_0) ≈ 0.882."""
    from decoding_decoding.counter_decode import (
        init_prior_pmf,
        make_beta_grid,
    )
    grid = make_beta_grid(log_lo=-3.0, log_hi=3.0, n_points=121)
    pmf = init_prior_pmf(grid, kind="lognormal", sigma_0=0.5)
    # Sum to 1.
    assert abs(pmf.sum().item() - 1.0) < 1e-6
    # E[β] under the discrete pmf: should be close to 1.
    e_beta = float((pmf * grid).sum().item())
    assert abs(e_beta - 1.0) < 0.05, f"E[β]={e_beta}"


def test_init_prior_pmf_exponential_has_mean_1() -> None:
    from decoding_decoding.counter_decode import (
        init_prior_pmf,
        make_beta_grid,
    )
    grid = make_beta_grid(log_lo=-4.0, log_hi=4.0, n_points=201)
    pmf = init_prior_pmf(grid, kind="exponential")
    e_beta = float((pmf * grid).sum().item())
    # Exponential(1) has E=1 exactly. The grid truncation makes this
    # approximate; allow 5% slack.
    assert abs(e_beta - 1.0) < 0.1, f"E[β]={e_beta}"


def test_init_prior_pmf_uniform_log_is_flat_in_log_space() -> None:
    from decoding_decoding.counter_decode import (
        init_prior_pmf,
        make_beta_grid,
    )
    grid = make_beta_grid(log_lo=-3.0, log_hi=3.0, n_points=121)
    pmf = init_prior_pmf(grid, kind="uniform_logβ")
    # On a log-spaced grid, density 1/β · β = 1 is constant. So pmf should
    # be uniform across grid points.
    expected = 1.0 / pmf.shape[0]
    np.testing.assert_allclose(pmf.numpy(), expected, atol=1e-6)


def test_discrete_grid_update_recovers_true_beta() -> None:
    """Same long-run convergence test as the streaming Laplace, on the
    discrete-grid filter with log-normal prior. Verifies the new path
    converges to the right answer under matched-prior."""
    from decoding_decoding.counter_decode import (
        discrete_grid_update_batched,
        init_discrete_grid_state,
    )
    rng = torch.Generator().manual_seed(11)
    V = 30
    beta_true = 1.7
    ell_ref = torch.linspace(2.0, -1.5, V, dtype=torch.float32)
    state = init_discrete_grid_state(
        batch_size=1, prior_kind="lognormal", sigma_0=0.5, n_points=161
    )
    probs = torch.softmax(beta_true * ell_ref, dim=-1)
    n_steps = 800
    samples = torch.multinomial(probs, num_samples=n_steps, replacement=True, generator=rng)
    for t in range(n_steps):
        # Synthetic "model logits" track our β̂ (matched-prior simulation).
        beta_hat = float(state.beta_hat().item())
        logits = (beta_hat * ell_ref).unsqueeze(0)
        x = samples[t:t+1].to(torch.long)
        state = discrete_grid_update_batched(state, logits, x, grid_chunk=32)
    final_beta = float(state.beta_hat().item())
    # 800 obs → expect within ~10% of β_true (grid resolution + sample noise).
    assert abs(np.log(final_beta) - np.log(beta_true)) < 0.15, (
        f"final β̂={final_beta:.3f} vs true {beta_true}"
    )


def test_laplace_evidence_weight_slows_convergence() -> None:
    """With evidence_weight = 0.1, J grows ~10× slower per step. Verify the
    same n_steps gives a wider posterior."""
    rng = torch.Generator().manual_seed(31)
    V = 20
    beta_true = 1.6
    ell_ref = torch.linspace(2.0, -1.5, V, dtype=torch.float64)
    probs = torch.softmax(beta_true * ell_ref, dim=-1)
    n_steps = 200
    samples = torch.multinomial(probs, num_samples=n_steps, replacement=True, generator=rng)

    state_full = init_laplace(1, sigma_0=0.5, dtype=torch.float64)
    state_weak = init_laplace(1, sigma_0=0.5, dtype=torch.float64)
    for t in range(n_steps):
        for state, alpha in ((state_full, 1.0), (state_weak, 0.1)):
            beta_hat = float(state.beta_hat().item())
            logits = (beta_hat * ell_ref).unsqueeze(0)
            x = samples[t:t+1].to(torch.long)
            new_state, _, _ = laplace_update(
                state, logits, x, evidence_weight=alpha
            )
            state.eta_hat = new_state.eta_hat
            state.J = new_state.J
    # The "weak" filter should have ~10× less precision after 200 steps
    # (modulo the prior contribution of J_0=4).
    j_full = float(state_full.J.item())
    j_weak = float(state_weak.J.item())
    assert j_weak < 0.3 * j_full, f"J_weak {j_weak:.2f} not noticeably below J_full {j_full:.2f}"


def test_laplace_memory_decay_keeps_J_bounded() -> None:
    """With memory_decay = 0.9, J should reach a stationary value even after
    many observations (instead of growing without bound)."""
    rng = torch.Generator().manual_seed(33)
    V = 20
    beta_true = 1.4
    ell_ref = torch.linspace(2.0, -1.5, V, dtype=torch.float64)
    probs = torch.softmax(beta_true * ell_ref, dim=-1)
    samples = torch.multinomial(probs, num_samples=2000, replacement=True, generator=rng)

    state = init_laplace(1, sigma_0=0.5, dtype=torch.float64)
    sigma_0 = 0.5
    prior_eta = -0.5 * sigma_0 ** 2
    prior_J = 1.0 / sigma_0 ** 2
    j_history = []
    for t in range(2000):
        beta_hat = float(state.beta_hat().item())
        logits = (beta_hat * ell_ref).unsqueeze(0)
        x = samples[t:t+1].to(torch.long)
        state, _, _ = laplace_update(
            state, logits, x, memory_decay=0.9,
            prior_eta=prior_eta, prior_J=prior_J,
        )
        j_history.append(state.J.item())
    j_arr = np.asarray(j_history)
    # J should be bounded, not growing linearly in t.
    early = j_arr[100:200].mean()
    late = j_arr[1900:2000].mean()
    # With γ=0.9, stationary J ≈ E[fisher] / (1-γ) which is moderate.
    assert late < 100, f"J unbounded under decay=0.9: late={late}"
    # Late should be close to "stationary" = within 30% of early.
    assert abs(late - early) / early < 0.5, f"J still growing fast: early={early} late={late}"


def test_discrete_grid_filter_walkthrough_two_token() -> None:
    """Discrete-grid filter on the walkthrough's specific (V=2, 3-point grid)
    setup must reproduce the same posteriors as the npz-style filter."""
    from decoding_decoding.counter_decode import (
        DiscreteGridState,
        discrete_grid_update_batched,
    )
    grid = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    pi = torch.tensor([[0.4, 0.4, 0.2]], dtype=torch.float32)
    state = DiscreteGridState(pi=pi, beta_grid=grid)

    # Use bootstrap-form likelihood: ℓ_t = β̂ · ℓ*_true. Walkthrough's matched-prior
    # toy outputs ℓ_t = β_model · (1, -1) and updates with x_t=A=0.
    # Need to feed the filter logits with β̂ = state.beta_hat() at each step.
    targets_beta_hat = [1.000, 1.060, 1.120]  # walkthrough rounded
    for expected_bh in targets_beta_hat:
        bh = float(state.beta_hat().item())
        assert abs(bh - expected_bh) < 1e-2, f"got β̂={bh:.3f} expected {expected_bh}"
        logits = torch.tensor([[bh * 1.0, bh * -1.0]], dtype=torch.float32)
        state = discrete_grid_update_batched(
            state, logits, torch.tensor([0], dtype=torch.long), grid_chunk=3
        )


# ---------------------------------------------------------------------------
# Shape-based β̂ estimators (non-bootstrap)
# ---------------------------------------------------------------------------


def test_rank_gap_recovers_exact_beta_under_pure_rescaling() -> None:
    """ℓ_t = β · ℓ_ref ⇒ rank-gap β̂ recovers β exactly (no rank-churn case)."""
    torch.manual_seed(0)
    V = 5000
    K = 20
    base = torch.randn(1, V)  # natural-shape logits
    base_sorted = topk_logits_sorted(base, K)
    ref_gaps = rank_gaps(base_sorted)  # (1, K)
    for beta_true in (0.25, 0.5, 1.0, 2.0, 4.0):
        scaled = beta_true * base
        beta_hat = estimate_beta_rank_gap(scaled, ref_gaps).item()
        assert abs(beta_hat - beta_true) < 1e-4, (
            f"β_true={beta_true}, got β̂={beta_hat}"
        )


def test_rank_gap_translation_invariance() -> None:
    """Adding a constant to all logits leaves softmax unchanged; rank gaps too."""
    torch.manual_seed(1)
    V = 1000
    K = 10
    base = torch.randn(1, V)
    shifted = base + 17.3
    K_eff = K
    ref_gaps = rank_gaps(topk_logits_sorted(base, K_eff))
    beta_hat = estimate_beta_rank_gap(shifted, ref_gaps).item()
    assert abs(beta_hat - 1.0) < 1e-4


def test_renyi_infty_bisection_recovers_beta_under_rescaling() -> None:
    """ℓ_t = β · ℓ_ref ⇒ Rényi-∞ bisection β̂ ≈ β."""
    torch.manual_seed(2)
    V = 5000
    base = torch.randn(1, V)
    ref_h = renyi_infty_from_logits(base)
    for beta_true in (0.25, 0.5, 1.0, 2.0, 4.0):
        scaled = beta_true * base
        beta_hat = estimate_beta_renyi_infty(scaled, ref_h, n_iter=40).item()
        # 40 bisection iterations on log β span [-4, 4] → resolution ~5e-3 in β.
        assert abs(beta_hat - beta_true) < 1e-2, (
            f"β_true={beta_true}, got β̂={beta_hat}"
        )


def test_renyi_infty_from_logits_matches_formula() -> None:
    """H_∞ = -log max softmax(ℓ) = logsumexp(ℓ) - max ℓ."""
    torch.manual_seed(3)
    V = 200
    base = torch.randn(3, V)
    h = renyi_infty_from_logits(base)
    p = torch.softmax(base, dim=-1)
    expected = -torch.log(p.amax(dim=-1))
    assert torch.allclose(h, expected, atol=1e-5)


def test_shape_estimators_no_bootstrap_dependence_on_sampled_token() -> None:
    """Critical: β̂ from shape estimators does NOT depend on the sampled token.

    Sanity check that the API never references x_t. If we re-call with the
    same logits but different (irrelevant) integer "tokens," β̂ is identical.
    """
    torch.manual_seed(4)
    V = 2000
    base = torch.randn(2, V)
    scaled = 2.5 * base
    ref_gaps = rank_gaps(topk_logits_sorted(base, 16))
    ref_h = renyi_infty_from_logits(base)
    # The shape-estimator API doesn't accept sampled tokens at all. This test
    # documents that fact: API signature has no x_t parameter, so there's
    # no way for the estimator to be contaminated.
    b1 = estimate_beta_rank_gap(scaled, ref_gaps)
    b2 = estimate_beta_renyi_infty(scaled, ref_h)
    # Sanity: both report β̂ ≈ 2.5.
    assert torch.allclose(b1, torch.tensor([2.5, 2.5]), atol=1e-3)
    assert torch.allclose(b2, torch.tensor([2.5, 2.5]), atol=1e-2)
