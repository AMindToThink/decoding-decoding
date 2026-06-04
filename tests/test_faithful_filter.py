"""Unit tests for the β-corrected ``laplace_update_faithful``.

These pin three properties (TDD, CPU-only):

1. At η̂ = 0 (β̂ = 1) the faithful update is *identical* to the original
   ``laplace_update`` — re-linearizing at β=1 is the original.
2. Over a stream of tokens drawn from ``softmax(β_true · ℓ)`` the faithful
   filter *recovers* β_true, while the original ``laplace_update`` *lags*
   toward β=1 (the documented single-Newton-step bias).
3. The η-space chain-rule factors β̂ (score) and β̂² (Fisher) are present —
   the bootstrap "cancellation" of the original is, by design, broken here.
"""

from __future__ import annotations

import math

import torch

from decoding_decoding.counter_decode import (
    LaplaceState,
    laplace_update,
    laplace_update_faithful,
)


def test_faithful_equals_original_at_beta_one() -> None:
    """At η̂ = 0 (β̂ = 1) faithful ≡ original, exactly."""
    torch.manual_seed(0)
    B, V = 3, 40
    logits = torch.randn(B, V)
    x = torch.randint(0, V, (B,))
    state = LaplaceState(eta_hat=torch.zeros(B), J=torch.full((B,), 2.0))

    s_o, score_o, fisher_o = laplace_update(state, logits, x)
    s_f, score_f, fisher_f = laplace_update_faithful(state, logits, x)

    torch.testing.assert_close(s_f.eta_hat, s_o.eta_hat)
    torch.testing.assert_close(s_f.J, s_o.J)
    torch.testing.assert_close(score_f, score_o)
    torch.testing.assert_close(fisher_f, fisher_o)


def test_faithful_recovers_known_beta_while_original_is_biased() -> None:
    """Stream from softmax(β_true·ℓ): faithful → β_true; original is badly biased.

    On a stationary stream from a fixed logit vector the original
    ``laplace_update`` never re-linearizes, so its β=1 score keeps a nonzero
    mean every step and its η grows ~log(N) without bound (it does not converge
    to the MLE). The faithful filter re-linearizes, so its score → 0 at the true
    β and it converges. We assert the *consistent vs inconsistent* contrast,
    direction-agnostically.
    """
    torch.manual_seed(0)
    V = 60
    logits = torch.randn(1, V)                # fixed logit vector, well-conditioned
    beta_true = 2.0
    log_beta_true = math.log(beta_true)

    probs = torch.softmax(beta_true * logits, dim=-1)        # (1, V)
    N = 5000
    gen = torch.Generator().manual_seed(123)
    tokens = torch.multinomial(
        probs, num_samples=N, replacement=True, generator=gen
    ).squeeze(0)                                             # (N,)

    def run(update_fn) -> float:
        # Prior centered at β=1 (η=0) so the data drives the estimate.
        st = LaplaceState(eta_hat=torch.zeros(1), J=torch.full((1,), 1.0))
        for t in range(N):
            st, _, _ = update_fn(st, logits, tokens[t : t + 1])
        return float(st.eta_hat.item())

    eta_faithful = run(laplace_update_faithful)
    eta_original = run(laplace_update)

    # Faithful recovers the true log β.
    assert abs(eta_faithful - log_beta_true) < 0.1, (eta_faithful, log_beta_true)
    # Original is badly biased (does not converge to the MLE)...
    assert abs(eta_original - log_beta_true) > 1.0, eta_original
    # ...and the faithful estimate is far closer to the truth.
    assert abs(eta_faithful - log_beta_true) < 0.2 * abs(eta_original - log_beta_true)


def test_faithful_carries_chain_rule_factors() -> None:
    """At β̂ ≠ 1 the returned score/Fisher carry the β̂, β̂² factors."""
    torch.manual_seed(1)
    V = 30
    logits = torch.randn(1, V)
    x = torch.randint(0, V, (1,))
    eta = torch.tensor([0.5])                  # β̂ = exp(0.5) ≈ 1.6487 ≠ 1
    state = LaplaceState(eta_hat=eta.clone(), J=torch.tensor([2.0]))

    _, score, fisher = laplace_update_faithful(state, logits, x)

    beta = torch.exp(eta)
    q = torch.softmax(beta.unsqueeze(-1) * logits, dim=-1)
    E_q = (q * logits).sum(-1)
    Var_q = (q * (logits - E_q.unsqueeze(-1)) ** 2).sum(-1)
    ell_x = logits.gather(-1, x.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(score, beta * (ell_x - E_q))
    torch.testing.assert_close(fisher, beta * beta * Var_q)
    # The β factor is non-trivial: differs from the un-β'd (original-style) version.
    assert not torch.allclose(score, ell_x - E_q)
    assert not torch.allclose(fisher, Var_q)
