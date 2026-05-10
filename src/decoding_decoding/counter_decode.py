"""Streaming Bayesian inference of β (inverse-temperature) for counter-decoding.

Implements the per-step algorithm from bayesian-sampler/counter_decoding.html
(section 02). At each step, we observe the model's pre-decoding logits ℓ_t and
the sampled token x_t, and update a Laplace approximation to the posterior on
η = log β. The bootstrap reference is ℓ*_t = ℓ_t / β̂, which under matched-prior
Bayes-optimality recovers the natural reference ℓ*_true.

Two implementations live here:

  1. ``LaplaceState`` + ``laplace_update`` — the production streaming Laplace
     update over log β (used inside the generation loop, vectorized over a
     batch of trajectories on GPU).

  2. ``discrete_grid_update`` — a discrete-grid Bayesian filter over β (used
     to validate the algorithm against the walkthrough's explicit numbers,
     and to provide a non-Laplace cross-check during development).

Algorithm (Laplace, per step):
    β̂      = exp(η̂)
    q      = softmax(ℓ_t)              # = model's pre-decoding distribution P_φ
    E_q    = Σ_j q[j] · ℓ_t[j]
    Var_q  = Σ_j q[j] · (ℓ_t[j] − E_q)²
    score  = ℓ_t[x_t] − E_q              # dlog p(x|η) / dη at η̂
    fisher = Var_q                        # expected Fisher information
    J     ← J + fisher
    η̂    ← η̂ + score / J                # Newton step

Note that the bootstrap reference ℓ*_t = ℓ_t / β̂ never appears explicitly in
score or fisher — the β̂ factors cancel out. This is by design (see the
counter_decoding memo, "Two implementation notes").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Streaming Laplace filter (production)
# ---------------------------------------------------------------------------


@dataclass
class LaplaceState:
    """Posterior state on η = log β as a Gaussian (mode, precision)."""

    eta_hat: torch.Tensor  # (B,) running posterior mode of log β
    J: torch.Tensor        # (B,) running posterior precision (positive)

    def beta_hat(self) -> torch.Tensor:
        """Current point estimate β̂ = exp(η̂)."""
        return torch.exp(self.eta_hat)

    def posterior_std_log_beta(self) -> torch.Tensor:
        """Marginal std of log β under the Laplace approximation: 1/sqrt(J)."""
        return 1.0 / torch.sqrt(self.J)


def init_laplace(
    batch_size: int,
    *,
    sigma_0: float = 0.5,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> LaplaceState:
    """Initialize the Laplace filter.

    Prior: log β ~ N(μ_0, σ_0²) with μ_0 = −σ_0²/2 (so E[β] = 1 exactly).

    Mode at start: η̂_0 = μ_0 (so β̂_0 = exp(−σ_0²/2), close to 1 for small σ_0).
    Precision at start: J_0 = 1/σ_0².

    Args:
        batch_size: number of independent trajectories.
        sigma_0: prior std of log β.
        device, dtype: where to place the tensors.
    """
    if sigma_0 <= 0:
        raise ValueError(f"sigma_0 must be positive, got {sigma_0}")
    mu_0 = -0.5 * sigma_0 ** 2
    eta = torch.full((batch_size,), float(mu_0), device=device, dtype=dtype)
    J = torch.full((batch_size,), float(1.0 / sigma_0 ** 2), device=device, dtype=dtype)
    return LaplaceState(eta_hat=eta, J=J)


def laplace_update(
    state: LaplaceState,
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
) -> Tuple[LaplaceState, torch.Tensor, torch.Tensor]:
    """One streaming Laplace update from a single per-trajectory observation.

    Formula (per-batch element):

        q       = softmax(ℓ_t)                    # (V,) — model's P_φ
        E_q     = Σ_j q[j] · ℓ_t[j]               # scalar — expected logit
        Var_q   = Σ_j q[j] · (ℓ_t[j] − E_q)²       # scalar — logit variance
        score   = ℓ_t[x_t] − E_q
        fisher  = Var_q
        J_new   = J + fisher
        η̂_new   = η̂ + score / J_new

    Code correspondence (lines below):
        - ``log_q = log_softmax(logits)`` : computes log q (numerically stable).
        - ``E_q  = (q * logits).sum(-1)`` : Σ_j q[j] · ℓ_t[j].
        - ``Var_q = (q * (logits - E_q[..., None])**2).sum(-1)`` : Σ_j q[j] · (...)².
        - ``score = ell_x - E_q`` : ℓ_t[x_t] − E_q.
        - ``J_new = state.J + Var_q`` : Σ over t of fisher_t (online accumulation).
        - ``eta_new = state.eta_hat + score / J_new`` : Newton step.

    Args:
        state: prior ``LaplaceState`` (batch B).
        logits: ``(B, V)`` model's pre-decoding logits for this step.
        sampled_token_ids: ``(B,)`` int64 token id sampled at this step.

    Returns:
        (new_state, score, fisher) — new_state has shape-(B,) eta_hat and J;
        score and fisher are returned for diagnostics (shape (B,)).
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got shape {tuple(logits.shape)}")
    if sampled_token_ids.dim() != 1 or sampled_token_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"sampled_token_ids must be (B,) matching logits batch; "
            f"got {tuple(sampled_token_ids.shape)} vs B={logits.shape[0]}"
        )
    # Compute q = softmax(logits) in a numerically stable way.
    log_q = torch.log_softmax(logits, dim=-1)         # (B, V)
    q = torch.exp(log_q)                              # (B, V)

    # Aggregation #1: E_q[ℓ_t] = Σ_j q[j] · ℓ_t[j]   (over vocab axis V)
    E_q = (q * logits).sum(dim=-1)                    # (B,)

    # Aggregation #2: Var_q[ℓ_t] = Σ_j q[j] · (ℓ_t[j] − E_q)²
    centered = logits - E_q.unsqueeze(-1)             # (B, V)
    Var_q = (q * centered * centered).sum(dim=-1)     # (B,)

    # Score uses the sampled token's logit.
    ell_x = logits.gather(-1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)  # (B,)
    score = ell_x - E_q                               # (B,)

    # Streaming Newton step.
    fisher = Var_q                                    # (B,)
    J_new = state.J + fisher
    eta_new = state.eta_hat + score / J_new

    return LaplaceState(eta_hat=eta_new, J=J_new), score, fisher


# ---------------------------------------------------------------------------
# Discrete-grid Bayesian filter (cross-check)
# ---------------------------------------------------------------------------


def discrete_grid_update(
    prior_pi: np.ndarray,
    beta_grid: np.ndarray,
    ref_logits: np.ndarray,
    sampled_token_id: int,
) -> np.ndarray:
    """Exact Bayes update on a discrete grid over β.

    This matches the walkthrough.html companion exactly: at each step we
    multiply the prior by softmax(β · ℓ*_t)[x_t] for every β on the grid and
    renormalize. The walkthrough's β-grid is (0.5, 1.0, 2.0) with prior
    weights (0.4, 0.4, 0.2) and ℓ*_true = (1, -1); see test_counter_decode.

    Args:
        prior_pi: (G,) probabilities over the β grid (sum to 1).
        beta_grid: (G,) β values (matching prior_pi).
        ref_logits: (V,) the *reference* logits ℓ*_t for this step.
        sampled_token_id: int — observed token id at this step.

    Returns:
        posterior_pi: (G,) renormalized posterior over the grid.
    """
    if prior_pi.shape != beta_grid.shape or prior_pi.ndim != 1:
        raise ValueError(
            f"prior_pi and beta_grid must be 1D and same shape; "
            f"got {prior_pi.shape} vs {beta_grid.shape}"
        )
    if not (0 <= sampled_token_id < ref_logits.shape[0]):
        raise ValueError(
            f"sampled_token_id={sampled_token_id} out of range for ref_logits"
            f" of length {ref_logits.shape[0]}"
        )
    # log P(x_t | β) = β·ℓ[x_t] − logsumexp(β·ℓ)  for each β on the grid.
    scaled = beta_grid[:, None] * ref_logits[None, :]                   # (G, V)
    log_lik = scaled[:, sampled_token_id] - _logsumexp_axis(scaled, -1)  # (G,)
    # log unnorm posterior, then renormalize in log-space.
    log_unnorm = np.log(prior_pi + 1e-300) + log_lik
    log_unnorm -= log_unnorm.max()
    unnorm = np.exp(log_unnorm)
    return unnorm / unnorm.sum()


def _logsumexp_axis(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (np.log(np.exp(a - m).sum(axis=axis, keepdims=True)) + m).squeeze(axis)


def discrete_grid_posterior_mean(pi: np.ndarray, beta_grid: np.ndarray) -> float:
    """β̂ = Σ_g π[g] · β_grid[g]."""
    return float(np.sum(pi * beta_grid))


# ---------------------------------------------------------------------------
# Sampling under counter-decoded distribution
# ---------------------------------------------------------------------------


def sample_under_beta(
    logits: torch.Tensor,
    beta_dec: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample x_t ~ Categorical(softmax(β_dec · ℓ_t)) per batch element.

    Args:
        logits: (B, V).
        beta_dec: (B,) per-trajectory effective inverse-temperature multiplier.
        generator: optional torch.Generator for reproducibility.

    Returns:
        sampled_token_ids: (B,) int64.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    if beta_dec.shape != (logits.shape[0],):
        raise ValueError(
            f"beta_dec must be (B,)={logits.shape[0]}, got {tuple(beta_dec.shape)}"
        )
    scaled = logits * beta_dec.unsqueeze(-1)               # (B, V)
    probs = torch.softmax(scaled, dim=-1)                  # (B, V)
    # torch.multinomial respects the optional generator.
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return sampled.squeeze(-1).to(dtype=torch.long)
