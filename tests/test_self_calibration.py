"""CPU unit tests for the self-calibration helpers (scripts/run_self_calibration.py).

Pins the pure, model-free pieces:
  - `per_token_nll` is exact and τ=1 reproduces the β=1 NLL bit-for-bit;
  - per-token τ vectors are applied position-wise;
  - the filter trajectory is strictly causal (β̂ used at position t depends only
    on x_{<t});
  - `pooled_mle_beta` recovers a known generating β.

The model/GPU pieces (generation, the full recovery run) are exercised by the
GPU smoke run, not here.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Load the script module by path (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "run_self_calibration",
    Path(__file__).resolve().parent.parent / "scripts" / "run_self_calibration.py",
)
sc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sc)  # type: ignore[union-attr]

from decoding_decoding.counter_decode import laplace_update, laplace_update_faithful


def test_per_token_nll_exact_at_beta_one() -> None:
    torch.manual_seed(0)
    B, U, V = 2, 7, 30
    logits = torch.randn(B, U, V)
    tokens = torch.randint(0, V, (B, U))
    got = sc.per_token_nll(logits_arr=logits, tokens=tokens, tau=1.0)
    logq = F.log_softmax(logits, dim=-1)
    want = (-logq.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)).double().numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_per_token_nll_per_token_tau() -> None:
    torch.manual_seed(1)
    B, U, V = 2, 5, 25
    logits = torch.randn(B, U, V)
    tokens = torch.randint(0, V, (B, U))
    tau = torch.rand(B, U) * 1.5 + 0.5            # (B,U) in [0.5, 2.0)
    got = sc.per_token_nll(logits_arr=logits, tokens=tokens, tau=tau)
    want = np.empty((B, U))
    for b in range(B):
        for u in range(U):
            lq = F.log_softmax(tau[b, u] * logits[b, u], dim=-1)
            want[b, u] = float(-lq[tokens[b, u]])
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_filter_trajectory_is_causal() -> None:
    """Changing token x_k must not change β̂_t for t ≤ k (β̂_t is recorded pre-update)."""
    torch.manual_seed(2)
    B, U, V = 3, 12, 20
    logits = torch.randn(B, U, V)
    tokens_a = torch.randint(0, V, (B, U))
    k = 5
    tokens_b = tokens_a.clone()
    tokens_b[:, k] = (tokens_a[:, k] + 1) % V      # perturb only position k

    for update_fn in (laplace_update, laplace_update_faithful):
        lb_a, _ = sc.run_filter_trajectory(
            logits_arr=logits, tokens=tokens_a, update_fn=update_fn,
            sigma_0=0.5, device="cpu",
        )
        lb_b, _ = sc.run_filter_trajectory(
            logits_arr=logits, tokens=tokens_b, update_fn=update_fn,
            sigma_0=0.5, device="cpu",
        )
        # positions 0..k use only x_{<=k-1}? recorded[:, t] is pre-update at t,
        # so it depends on x_{<t}; perturbing x_k affects recorded[:, t] for t > k only.
        np.testing.assert_allclose(lb_a[:, : k + 1], lb_b[:, : k + 1], atol=1e-6)
        assert not np.allclose(lb_a[:, k + 1 :], lb_b[:, k + 1 :])


def test_pooled_mle_recovers_known_beta() -> None:
    torch.manual_seed(3)
    V, N = 50, 6000
    ell = torch.randn(1, V)
    beta_true = 1.6
    probs = torch.softmax(beta_true * ell, dim=-1)
    gen = torch.Generator().manual_seed(7)
    toks = torch.multinomial(probs, num_samples=N, replacement=True, generator=gen).squeeze(0)
    logits_arr = ell.unsqueeze(1).expand(1, N, V).contiguous()     # (1, N, V) all = ell
    tokens = toks.unsqueeze(0)                                     # (1, N)
    beta_mle = sc.pooled_mle_beta(logits_arr=logits_arr, tokens=tokens)
    assert abs(math.log(beta_mle) - math.log(beta_true)) < 0.1, (beta_mle, beta_true)
