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

import math
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
    *,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
    prior_eta: float = 0.0,
    prior_J: float = 0.0,
) -> Tuple[LaplaceState, torch.Tensor, torch.Tensor]:
    """One streaming Laplace update from a single per-trajectory observation.

    Formula (per-batch element):

        q       = softmax(ℓ_t)                    # (V,) — model's P_φ
        E_q     = Σ_j q[j] · ℓ_t[j]               # scalar — expected logit
        Var_q   = Σ_j q[j] · (ℓ_t[j] − E_q)²       # scalar — logit variance
        score   = ℓ_t[x_t] − E_q
        fisher  = Var_q
        # optional "fuzzy window" decay (default no-op):
        η̂      ← γ · η̂ + (1 − γ) · η_prior
        J       ← γ · J  + (1 − γ) · J_prior
        # Newton step with optional evidence weight α:
        J_new   = J + α · fisher
        η̂_new   = η̂ + α · score / J_new

    The (γ, α) hyperparameters are off (1.0, 1.0) by default, recovering the
    plain matched-prior Laplace from the counter_decoding memo. They can be
    used to give the filter a "fuzzy window" where past evidence is decayed
    and per-step evidence is downweighted, useful when matched-prior
    Bayes-optimality is suspect.

    Code correspondence:
        - ``log_q = log_softmax(logits)`` : computes log q.
        - ``E_q  = (q * logits).sum(-1)`` : Σ_j q[j] · ℓ_t[j].
        - ``Var_q = ...`` : Σ_j q[j] · (...)².
        - ``score = ell_x - E_q`` : ℓ_t[x_t] − E_q.
        - decay: ``eta = γ·eta + (1-γ)·eta_prior`` etc.
        - ``J_new = J + α·Var_q`` : per-step evidence accumulation.
        - ``eta_new = eta + α·score / J_new`` : Newton step.

    Args:
        state: prior ``LaplaceState`` (batch B).
        logits: ``(B, V)`` model's pre-decoding logits for this step.
        sampled_token_ids: ``(B,)`` int64 token id sampled at this step.
        evidence_weight: α ∈ (0, 1]. Per-step weight on (score, fisher).
        memory_decay: γ ∈ [0, 1]. Per-step decay of accumulated state toward prior.
        prior_eta: η₀ for the decay anchor.
        prior_J: J₀ for the decay anchor (precision).

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

    # Optional decay toward prior (γ=1 ⇒ no decay).
    if memory_decay < 1.0:
        eta_decayed = memory_decay * state.eta_hat + (1.0 - memory_decay) * prior_eta
        J_decayed = memory_decay * state.J + (1.0 - memory_decay) * prior_J
    else:
        eta_decayed = state.eta_hat
        J_decayed = state.J

    # Streaming Newton step (with optional per-step evidence weight α).
    fisher = Var_q                                    # (B,) — raw observation Fisher
    fisher_eff = evidence_weight * fisher
    score_eff = evidence_weight * score
    J_new = J_decayed + fisher_eff
    eta_new = eta_decayed + score_eff / J_new

    return LaplaceState(eta_hat=eta_new, J=J_new), score, fisher


def laplace_update_faithful(
    state: LaplaceState,
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    *,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
    prior_eta: float = 0.0,
    prior_J: float = 0.0,
) -> Tuple[LaplaceState, torch.Tensor, torch.Tensor]:
    """Streaming Laplace update that RE-LINEARIZES at the running estimate β̂.

    ``laplace_update`` evaluates the score and Fisher once at β=1 and never
    re-evaluates, so its β̂ is a single damped Newton step from the prior mode —
    accurate only near β=1 and biased away from 1 otherwise. This variant
    evaluates them at the current estimate β̂ = exp(η̂), making each token a
    proper Fisher-scoring (natural-gradient) step in η = log β. Over a stream it
    is a *consistent* estimator of β (converges to the MLE), where the original
    lags.

    For the 1-parameter exponential family ``P(x | η) = softmax(e^η · ℓ)[x]`` the
    EXACT score and expected Fisher in η are (note dβ/dη = β)::

        β̂      = exp(η̂)
        q_β     = softmax(β̂ · ℓ)                  # re-linearize at β̂, NOT at 1
        E_qβ    = Σ_j q_β[j] · ℓ_j
        Var_qβ  = Σ_j q_β[j] · (ℓ_j − E_qβ)²
        score_η  = β̂  · (ℓ_{x} − E_qβ)             # = d/dη log P(x|η)
        fisher_η = β̂² · Var_qβ                     # = E[−d²/dη² log P(x|η)]

    The β̂ / β̂² chain-rule factors are what make this consistent in η-space;
    ``laplace_update`` drops them (exact only at β̂=1). Because q_β depends on β̂,
    the bootstrap reference ℓ* = ℓ/β̂ no longer cancels out of score/fisher — the
    "cancellation" noted for ``laplace_update`` is *by design* broken here (that
    cancellation IS the β=1 linearization).

    Recursion is identical in structure to ``laplace_update``; the linearization
    point is the *decayed* η̂ (the belief we step from). One re-linearized Newton
    step per token.

    Args / Returns mirror ``laplace_update`` exactly. The returned ``(score,
    fisher)`` are the η-space quantities ``score_η, fisher_η`` (carrying β̂, β̂²).
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got shape {tuple(logits.shape)}")
    if sampled_token_ids.dim() != 1 or sampled_token_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"sampled_token_ids must be (B,) matching logits batch; "
            f"got {tuple(sampled_token_ids.shape)} vs B={logits.shape[0]}"
        )

    # Decay toward prior FIRST (γ=1 ⇒ no decay); we linearize at the decayed belief.
    if memory_decay < 1.0:
        eta_decayed = memory_decay * state.eta_hat + (1.0 - memory_decay) * prior_eta
        J_decayed = memory_decay * state.J + (1.0 - memory_decay) * prior_J
    else:
        eta_decayed = state.eta_hat
        J_decayed = state.J

    # Re-linearize the tilted distribution at the current estimate β̂ = exp(η̂).
    beta_hat = torch.exp(eta_decayed)                      # (B,)
    log_q = torch.log_softmax(beta_hat.unsqueeze(-1) * logits, dim=-1)  # (B, V)
    q = torch.exp(log_q)                                   # (B, V)

    E_q = (q * logits).sum(dim=-1)                         # (B,) — E_{q_β}[ℓ]
    centered = logits - E_q.unsqueeze(-1)                  # (B, V)
    Var_q = (q * centered * centered).sum(dim=-1)          # (B,) — Var_{q_β}[ℓ]

    ell_x = logits.gather(-1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)  # (B,)

    # η-space score and expected Fisher carry the chain-rule factors β̂, β̂².
    score = beta_hat * (ell_x - E_q)                       # (B,) — score_η
    fisher = beta_hat * beta_hat * Var_q                   # (B,) — fisher_η

    score_eff = evidence_weight * score
    fisher_eff = evidence_weight * fisher
    J_new = J_decayed + fisher_eff
    eta_new = eta_decayed + score_eff / J_new

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
# Vectorized discrete-grid filter (production, supports arbitrary priors)
# ---------------------------------------------------------------------------


@dataclass
class DiscreteGridState:
    """Posterior on β as a pmf over a (log-spaced) grid, batched over trajectories."""

    pi: torch.Tensor          # (B, G) — posterior probabilities, sum to 1 along G
    beta_grid: torch.Tensor   # (G,) — β values, e.g. log-spaced

    def beta_hat(self) -> torch.Tensor:
        """Posterior mean: (B,) = Σ_g π[g] · β_grid[g]."""
        return (self.pi * self.beta_grid.unsqueeze(0)).sum(dim=-1)

    def posterior_std_log_beta(self) -> torch.Tensor:
        """Posterior std of log β under the discrete grid."""
        log_beta = torch.log(self.beta_grid)                      # (G,)
        m = (self.pi * log_beta.unsqueeze(0)).sum(dim=-1)         # (B,)
        v = (self.pi * (log_beta.unsqueeze(0) - m.unsqueeze(-1)) ** 2).sum(dim=-1)
        return torch.sqrt(v.clamp_min(0.0))


def make_beta_grid(
    log_lo: float = -4.0,
    log_hi: float = 4.0,
    n_points: int = 161,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Log-spaced β grid in [exp(log_lo), exp(log_hi)] with `n_points` points."""
    if n_points < 3:
        raise ValueError(f"need ≥3 grid points, got {n_points}")
    log_grid = torch.linspace(log_lo, log_hi, n_points, device=device, dtype=dtype)
    return torch.exp(log_grid)


def init_prior_pmf(
    beta_grid: torch.Tensor,
    *,
    kind: str,
    sigma_0: float = 0.5,
    gamma_shape: float = 2.0,
    invgamma_shape: float = 3.0,
    halfcauchy_scale: float = 1.0,
) -> torch.Tensor:
    """Compute initial prior pmf over the β grid.

    Available kinds (scipy.stats based; library implementations preferred per
    repo CLAUDE.md):
      - "lognormal":      log β ~ N(μ_0, σ_0²) with μ_0 = -σ_0²/2 (E[β]=1).
      - "exponential":    β ~ Exp(rate=1)  (max-ent over β>0 with E[β]=1).
      - "gamma":          β ~ Gamma(shape, rate) with rate = shape (so E[β]=1).
      - "invgamma":       β ~ InverseGamma(α=invgamma_shape, scale=invgamma_shape-1)
                           (E[β]=1 for α>1; heavy right tail).
      - "halfcauchy_logβ": log β ~ HalfCauchy(scale)  centered at 0 (folded), so β≥1.
      - "cauchy_logβ":    log β ~ Cauchy(loc=0, scale=halfcauchy_scale) (heavy two-sided).
      - "uniform_logβ":   log β ~ Uniform(log_lo, log_hi) — flat in log space (improper at limit).

    Returns: (G,) pmf normalized to sum to 1, on the same device/dtype as beta_grid.
    """
    from scipy import stats as _scipy_stats

    grid_np = beta_grid.detach().cpu().numpy().astype(np.float64)
    log_grid_np = np.log(grid_np)

    if kind == "lognormal":
        # scipy lognorm: s=sigma, scale=exp(mu).
        mu = -0.5 * sigma_0 ** 2
        rv = _scipy_stats.lognorm(s=sigma_0, scale=np.exp(mu))
        density = rv.pdf(grid_np)
    elif kind == "exponential":
        rv = _scipy_stats.expon(scale=1.0)
        density = rv.pdf(grid_np)
    elif kind == "gamma":
        # E[β]=shape/rate=1 ⇒ rate=shape; scipy gamma uses scale=1/rate=1/shape.
        rv = _scipy_stats.gamma(a=gamma_shape, scale=1.0 / gamma_shape)
        density = rv.pdf(grid_np)
    elif kind == "invgamma":
        # scipy invgamma: a=α, scale=β; E[X]=β/(α-1).
        # Set α=invgamma_shape, β=α-1 to give E[X]=1.
        if invgamma_shape <= 1.0:
            raise ValueError("invgamma shape must be > 1 for finite mean")
        rv = _scipy_stats.invgamma(a=invgamma_shape, scale=invgamma_shape - 1.0)
        density = rv.pdf(grid_np)
    elif kind == "halfcauchy_logβ":
        # log β ~ HalfCauchy(scale) starts at 0; means β ≥ 1.
        # Density on log β ⇒ density on β = halfcauchy_pdf(log β) / β.
        rv = _scipy_stats.halfcauchy(scale=halfcauchy_scale)
        density = rv.pdf(np.maximum(log_grid_np, 0.0)) / np.maximum(grid_np, 1e-300)
        # Zero out grid points with log β < 0 (β < 1).
        density = np.where(log_grid_np >= 0.0, density, 0.0)
    elif kind == "cauchy_logβ":
        # log β ~ Cauchy(loc=0, scale=halfcauchy_scale). E[β] doesn't exist,
        # but median β = 1.
        rv = _scipy_stats.cauchy(loc=0.0, scale=halfcauchy_scale)
        density = rv.pdf(log_grid_np) / np.maximum(grid_np, 1e-300)
    elif kind == "uniform_logβ":
        # log β uniform on the grid range. Density on β = const / β.
        density = 1.0 / np.maximum(grid_np, 1e-300)
    else:
        raise ValueError(f"unknown prior kind {kind!r}")

    # Convert to discrete pmf by integrating the density over each grid cell;
    # use trapezoid rule on log β (equally spaced) → density(g) · Δlog β · grid(g)
    # because dβ = β·dlog β. With log-spaced grid, Δlog β is constant and cancels
    # in normalization, so we only need pmf_g ∝ pdf_β(β_g) · β_g.
    weight = density * grid_np
    weight = np.maximum(weight, 1e-300)
    pmf = weight / weight.sum()
    return torch.tensor(pmf, device=beta_grid.device, dtype=beta_grid.dtype)


def discrete_grid_update_batched(
    state: DiscreteGridState,
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    *,
    grid_chunk: int = 32,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
    prior_pmf: torch.Tensor | None = None,
) -> DiscreteGridState:
    """LEGACY bootstrap-grid filter — kept as a baseline, NOT the certificate.

    Carries **no per-sequence guarantee** (its experts are not fixed — see the
    NOTE below) and reports the posterior **mean** β̂ = Σ_g π[g]·β_g, i.e. the
    "posterior-mean trap" flagged in ``paper/grid_certificate_walkthrough.html``:
    the mean of a skewed posterior over a log-spaced grid is a biased point
    estimate, not the grid MLE. For the guarantee-bearing filter that scores
    fixed experts on raw logits and reports the grid MAP, use
    ``grid_mixture_update_batched`` (the certificate).

    Likelihood at grid point β_g for trajectory n (using bootstrap reference
    ℓ*_t = ℓ_t / β̂_t per-trajectory):

        log P(x_t | β_g, ℓ*_t)
          = β_g · ℓ*_t[x_t] − logsumexp_v(β_g · ℓ*_t[v])

    NOTE (guarantees): because ℓ*_t = ℓ_t / β̂_t depends on the *aggregate*
    posterior state, the per-grid-point "experts" here are not fixed forecasters,
    so the per-sequence regret guarantees of the Bayes-mixture / aggregating
    algorithm do NOT apply to this function. For the guarantee-bearing variant
    that scores experts on raw logits, see ``grid_mixture_update_batched``.

    Implementation: loop over `grid_chunk` β values at a time to keep peak
    memory at O(B · grid_chunk · V) instead of O(B · G · V).

    Args:
        state: prior DiscreteGridState (B, G) over β_grid.
        logits: (B, V) model's pre-decoding logits.
        sampled_token_ids: (B,) int64.
        grid_chunk: how many grid points to evaluate per inner pass.

    Returns:
        new DiscreteGridState with updated `pi`.
    """
    B, V = logits.shape
    G = state.beta_grid.shape[0]
    if state.pi.shape != (B, G):
        raise ValueError(
            f"state.pi shape {tuple(state.pi.shape)} does not match (B={B}, G={G})"
        )

    beta_hat = state.beta_hat()                   # (B,)
    ell_star = logits / beta_hat.unsqueeze(-1)    # (B, V)
    ell_star_x = ell_star.gather(-1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)  # (B,)

    log_lik = torch.empty((B, G), device=logits.device, dtype=torch.float32)
    beta_grid = state.beta_grid                   # (G,)
    for start in range(0, G, grid_chunk):
        stop = min(start + grid_chunk, G)
        bg = beta_grid[start:stop]                # (Gc,)
        # (B, Gc, V) — peak memory; chunked so this stays bounded.
        scaled = bg.view(1, -1, 1) * ell_star.unsqueeze(1)
        lse = torch.logsumexp(scaled.float(), dim=-1)            # (B, Gc)
        # β_g · ℓ*[x] − lse
        log_lik[:, start:stop] = bg.unsqueeze(0) * ell_star_x.unsqueeze(-1) - lse

    # Optional decay: mix toward prior pmf.
    if memory_decay < 1.0:
        if prior_pmf is None:
            raise ValueError("memory_decay < 1 requires prior_pmf to mix toward")
        pi_decayed = memory_decay * state.pi + (1.0 - memory_decay) * prior_pmf.unsqueeze(0)
    else:
        pi_decayed = state.pi

    # Bayes update in log space, with optional evidence weighting.
    log_post = torch.log(pi_decayed.clamp_min(1e-30)) + evidence_weight * log_lik
    log_post = log_post - log_post.max(dim=-1, keepdim=True).values
    post = torch.exp(log_post)
    post = post / post.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return DiscreteGridState(pi=post, beta_grid=state.beta_grid)


def init_discrete_grid_state(
    *,
    batch_size: int,
    prior_kind: str,
    sigma_0: float = 0.5,
    log_lo: float = -4.0,
    log_hi: float = 4.0,
    n_points: int = 161,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    **prior_kwargs,
) -> DiscreteGridState:
    """Build a DiscreteGridState ready for streaming updates."""
    beta_grid = make_beta_grid(
        log_lo=log_lo, log_hi=log_hi, n_points=n_points, device=device, dtype=dtype
    )
    prior_pmf = init_prior_pmf(beta_grid, kind=prior_kind, sigma_0=sigma_0, **prior_kwargs)
    pi = prior_pmf.unsqueeze(0).expand(batch_size, -1).contiguous()
    return DiscreteGridState(pi=pi, beta_grid=beta_grid)


# ---------------------------------------------------------------------------
# Guarantee-bearing grid mixture filter (prediction with expert advice)
# ---------------------------------------------------------------------------
#
# Each grid point β_g is a fixed "expert" forecasting P_g(x_t) = softmax(β_g·ℓ_t)[x_t]
# on the RAW logits (no bootstrap rescaling — that is the one difference from
# ``discrete_grid_update_batched``, and it is load-bearing: fixed experts are what
# make the regret guarantees below hold). The filter is the Bayes mixture over
# experts, which for log loss is exactly Vovk's Aggregating Algorithm at η=1
# (log loss is 1-mixable; Cesa-Bianchi & Lugosi, "Prediction, Learning, and
# Games", ch. 3 & 9). Guarantees, per-sequence (i.e. for ARBITRARY token
# streams, with no assumption that tokens were sampled from any P_β):
#
#   Forecasting:  Σ_t −ln(mixture prob of x_t) ≤ L_n(β_g) + ln(1/π₀[g])  ∀g,
#                 since mix_loss = −ln Σ_g π₀[g]·e^{−L_n[g]} (telescoping).
#                 Uniform prior ⇒ regret vs best grid expert ≤ ln G.
#   Estimation:   with uniform prior the grid MAP equals argmin_g L_n(β_g)
#                 (follow-the-leader) — the in-hindsight best-fit "effective"
#                 inverse temperature on the grid. L_n(β) is convex in β with
#                 L_n''(β) = Σ_t Var_{softmax(βℓ_t)}(ℓ_t), so the level set
#                 {β : L_n(β) ≤ min L_n + c} is an interval whose width is the
#                 honest, data-dependent uncertainty (wide exactly when peaked
#                 logits make β unidentifiable). ``grid_mixture_level_set``
#                 reads this interval off the stored loss profile.
#   Drift:        ``switch_rate`` α > 0 is Fixed-Share (Herbster & Warmuth 1998,
#                 share-to-initial-prior variant; cf. ``memory_decay`` in the
#                 bootstrap filter): exact Bayes under a switching prior, with
#                 per-sequence regret ≈ m·ln(Gn/m) vs the best m-segment β
#                 sequence. Tracking/forecasting guarantee only — the MAP/level-set
#                 estimation reading is for the static (α=0) case.
#
# These guarantees are relative to the in-hindsight best-fit β (the sequence
# MLE / KL projection), NOT the dial a counterparty actually set; under
# misspecification (top-k, blacklists, GIGO loops) the former is the natural
# definition of effective temperature. Point-estimate filters (the Laplace
# updates above ≈ online Newton step) provably cannot carry uniform per-sequence
# guarantees for losses of this type (Foster et al. 2018, "Logistic Regression:
# The Importance of Being Improper", arXiv:1803.09349) — the mixture is improper
# and sidesteps that obstruction.


@dataclass
class GridMixtureState:
    """State of the expert-advice grid mixture: loss profile + forecaster weights."""

    cum_loss: torch.Tensor   # (B, G) — L_n(β_g): cumulative log loss per expert
    log_w: torch.Tensor      # (B, G) — normalized log forecaster weights
    mix_loss: torch.Tensor   # (B,)   — cumulative log loss of the mixture forecast
    beta_grid: torch.Tensor  # (G,)   — β values (fixed experts)
    log_prior: torch.Tensor  # (G,)   — log π₀ over the grid

    def beta_map(self) -> torch.Tensor:
        """(B,) grid MAP = argmax_g (log π₀[g] − L_n[g]); uniform prior ⇒ argmin L_n."""
        idx = (self.log_prior.unsqueeze(0) - self.cum_loss).argmax(dim=-1)
        return self.beta_grid[idx]

    def realized_regret(self) -> torch.Tensor:
        """(B,) mixture loss minus the best expert's loss (≤ regret_bound())."""
        return self.mix_loss - self.cum_loss.min(dim=-1).values

    def regret_bound(self) -> torch.Tensor:
        """(B,) the per-sequence guarantee ln(1/π₀[best expert]); uniform ⇒ ln G."""
        best = self.cum_loss.argmin(dim=-1)
        return -self.log_prior[best]


def init_grid_mixture(
    *,
    batch_size: int,
    beta_grid: torch.Tensor,
    prior_pmf: torch.Tensor | None = None,
) -> GridMixtureState:
    """Initialize the grid mixture filter.

    Args:
        batch_size: number of independent trajectories B.
        beta_grid: (G,) β values, e.g. from ``make_beta_grid``.
        prior_pmf: (G,) prior over the grid (e.g. from ``init_prior_pmf``);
            default uniform. Non-uniform priors loosen the regret bound for
            expert g to ln(1/π₀[g]) and shift the MAP by the log-prior.
    """
    if beta_grid.dim() != 1:
        raise ValueError(f"beta_grid must be 1D, got shape {tuple(beta_grid.shape)}")
    G = beta_grid.shape[0]
    if prior_pmf is None:
        log_prior = torch.full(
            (G,), -math.log(G), device=beta_grid.device, dtype=beta_grid.dtype
        )
    else:
        if prior_pmf.shape != (G,):
            raise ValueError(
                f"prior_pmf shape {tuple(prior_pmf.shape)} does not match grid ({G},)"
            )
        if (prior_pmf <= 0).any():
            raise ValueError("prior_pmf must be strictly positive on the grid")
        prior_pmf = prior_pmf.to(device=beta_grid.device, dtype=beta_grid.dtype)
        log_prior = torch.log(prior_pmf / prior_pmf.sum())
    zeros = torch.zeros(
        (batch_size, G), device=beta_grid.device, dtype=beta_grid.dtype
    )
    return GridMixtureState(
        cum_loss=zeros,
        log_w=log_prior.unsqueeze(0).expand(batch_size, -1).contiguous(),
        mix_loss=torch.zeros(batch_size, device=beta_grid.device, dtype=beta_grid.dtype),
        beta_grid=beta_grid,
        log_prior=log_prior,
    )


def grid_mixture_update_batched(
    state: GridMixtureState,
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    *,
    grid_chunk: int = 32,
    switch_rate: float = 0.0,
) -> GridMixtureState:
    """One expert-advice step: score every expert on raw logits, Bayes-mix.

    Per step, for each grid point β_g:

        log P_g(x_t) = β_g · ℓ_t[x_t] − logsumexp_v(β_g · ℓ_t[v])
        mix step loss = −logsumexp_g(log w[g] + log P_g(x_t))
        w ← normalize(w · P_g(x_t));  then Fixed-Share if switch_rate α > 0:
        w ← (1 − α) · w + α · π₀

    Evidence weighting is deliberately NOT exposed: any weight ≠ 1 destroys the
    1-mixability of log loss and with it every guarantee documented above.

    Args:
        state: prior GridMixtureState (B, G).
        logits: (B, V) RAW pre-decoding logits (no bootstrap rescaling).
        sampled_token_ids: (B,) int64.
        grid_chunk: grid points per inner pass (memory O(B·grid_chunk·V)).
        switch_rate: Fixed-Share α ∈ [0, 1); 0 = pure Bayes (static guarantee).

    Returns:
        new GridMixtureState.
    """
    if not 0.0 <= switch_rate < 1.0:
        raise ValueError(f"switch_rate must be in [0, 1), got {switch_rate}")
    B, V = logits.shape
    G = state.beta_grid.shape[0]
    if state.cum_loss.shape != (B, G):
        raise ValueError(
            f"state.cum_loss shape {tuple(state.cum_loss.shape)} does not match (B={B}, G={G})"
        )
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    ell_x = logits.gather(-1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)  # (B,)
    beta_grid = state.beta_grid
    log_lik = torch.empty((B, G), device=logits.device, dtype=state.cum_loss.dtype)
    for start in range(0, G, grid_chunk):
        stop = min(start + grid_chunk, G)
        bg = beta_grid[start:stop]                              # (Gc,)
        scaled = bg.view(1, -1, 1) * logits.unsqueeze(1)        # (B, Gc, V)
        lse = torch.logsumexp(scaled.to(state.cum_loss.dtype), dim=-1)  # (B, Gc)
        log_lik[:, start:stop] = bg.unsqueeze(0) * ell_x.unsqueeze(-1) - lse

    step_mix_log_prob = torch.logsumexp(state.log_w + log_lik, dim=-1)  # (B,)

    log_w = state.log_w + log_lik
    log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
    if switch_rate > 0.0:
        # Fixed-Share toward the initial prior (Herbster & Warmuth 1998).
        log_w = torch.logaddexp(
            log_w + math.log(1.0 - switch_rate),
            state.log_prior.unsqueeze(0) + math.log(switch_rate),
        )

    return GridMixtureState(
        cum_loss=state.cum_loss - log_lik,
        log_w=log_w,
        mix_loss=state.mix_loss - step_mix_log_prob,
        beta_grid=state.beta_grid,
        log_prior=state.log_prior,
    )


def grid_mixture_level_set(
    state: GridMixtureState,
    c: torch.Tensor | float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence error bar: bounds in log β of {g : L_n[g] ≤ min L_n + c}.

    L_n(β) is convex in β, so the sub-level set is an interval; its width is a
    distribution-free, data-dependent uncertainty for the in-hindsight best-fit
    β. Default threshold c = ln(1/π₀[best expert]) (uniform prior: ln G), the
    same constant as the regret guarantee.

    "Distribution-free" in both senses: no assumption on the token process, and
    no log-normal prior / Gaussian-posterior approximation either — with a
    uniform prior this is a profile-likelihood interval around the grid MLE,
    and the prior enters only through the default threshold c.

    Returns:
        (lo, hi): each (B,), bounds in log β. Resolution-limited by the grid:
        the continuum MLE can sit up to one grid spacing outside.
    """
    if c is None:
        c_t = state.regret_bound()                              # (B,)
    else:
        c_t = torch.as_tensor(
            c, device=state.cum_loss.device, dtype=state.cum_loss.dtype
        ).expand(state.cum_loss.shape[0])
    min_loss = state.cum_loss.min(dim=-1, keepdim=True).values  # (B, 1)
    mask = state.cum_loss <= min_loss + c_t.unsqueeze(-1)       # (B, G)
    log_grid = torch.log(state.beta_grid).unsqueeze(0)          # (1, G)
    lo = torch.where(mask, log_grid, torch.inf).min(dim=-1).values
    hi = torch.where(mask, log_grid, -torch.inf).max(dim=-1).values
    return lo, hi


# ---------------------------------------------------------------------------
# Shape-based β̂ estimators (non-bootstrap)
# ---------------------------------------------------------------------------
#
# Both estimators read β_model directly from the shape of ℓ_t (the model's own
# prediction over the next-token distribution), not from the sampled token x_t.
# Because they don't condition on x_t, they break the bootstrap circularity
# inherent to streaming Bayesian filters: there is no fixed point at which
# β̂ → β_target.
#
# Convention: ℓ_t may be uncentered. Under a pure β-rescaling ℓ ↦ β·ℓ, both
# the rank-gap statistic Δ_k and the Rényi-∞ entropy H_∞ are equivariant in a
# specific way:
#   Δ_k(βℓ)        = β · Δ_k(ℓ)               (exact)
#   H_∞(softmax(βℓ)) is monotone in β       (monotonically increasing → log V
#                                              as β → ∞, → 0 as β → 0⁺).
#
# Reference convention: capture the reference at t=0 (first generated token,
# after consuming the prompt). At t=0 we *assume* β_model ≈ 1 because the
# model has not yet drifted under any committed continuation. Subsequent
# β̂_t is reported relative to that reference. This is a free choice of gauge
# (any anchor would do); the corrector only cares about *ratios*, so as long
# as the reference is consistent the cancellation of β_target / β̂_t works.


def topk_logits_sorted(logits: torch.Tensor, K: int) -> torch.Tensor:
    """Return the top-(K+1) logits sorted in descending order.

    Args:
        logits: (B, V).
        K: number of rank-gaps to support. Returns (B, K+1) so we can form K gaps.
    """
    if K < 1:
        raise ValueError(f"K must be ≥ 1, got {K}")
    top, _ = torch.topk(logits, k=K + 1, dim=-1, sorted=True)
    return top  # (B, K+1)


def rank_gaps(top_sorted_logits: torch.Tensor) -> torch.Tensor:
    """Δ_k = ℓ_(k) − ℓ_(k+1) for k = 1..K.

    Args:
        top_sorted_logits: (B, K+1), the output of `topk_logits_sorted`.
    Returns:
        gaps: (B, K), all non-negative since input is sorted descending.
    """
    return top_sorted_logits[..., :-1] - top_sorted_logits[..., 1:]


def estimate_beta_rank_gap(
    logits: torch.Tensor,
    reference_gaps: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rank-gap β̂ estimator.

    Formula (per trajectory):
        ℓ_(k)^(t) sorted descending; Δ_k^(t) = ℓ_(k)^(t) − ℓ_(k+1)^(t).
        β̂_t = median_{k=1..K} ( Δ_k^(t) / Δ_k^(ref) ).

    Under ℓ ↦ β · ℓ each Δ_k scales by exactly β (β-equivariant), so the
    ratio is an unbiased per-rank estimate of β. The median over k makes the
    statistic robust to changes in which tokens occupy each rank between
    reference and current step.

    Args:
        logits: (B, V) current step logits (uncentered is fine — gaps are
            translation-invariant).
        reference_gaps: (B, K) reference gaps captured at t=0 per trajectory.
        eps: tiny epsilon to avoid divide-by-zero on degenerate references.

    Returns:
        beta_hat: (B,) point estimate of β_model_t relative to the t=0 reference.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    if reference_gaps.dim() != 2:
        raise ValueError(
            f"reference_gaps must be (B, K), got {tuple(reference_gaps.shape)}"
        )
    B = logits.shape[0]
    if reference_gaps.shape[0] != B:
        raise ValueError(
            f"reference_gaps batch dim {reference_gaps.shape[0]} != logits batch dim {B}"
        )
    K = reference_gaps.shape[-1]
    top = topk_logits_sorted(logits, K)        # (B, K+1)
    gaps = rank_gaps(top)                       # (B, K)
    ratios = gaps / reference_gaps.clamp_min(eps)
    return ratios.median(dim=-1).values         # (B,)


def renyi_infty_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """H_∞(softmax(ℓ)) = log Σ exp ℓ − max ℓ = −log max softmax(ℓ).

    Args:
        logits: (B, V).
    Returns:
        h_inf: (B,) Rényi-∞ entropy in nats, ≥ 0.
    """
    return torch.logsumexp(logits, dim=-1) - logits.amax(dim=-1)


def estimate_beta_renyi_infty(
    logits: torch.Tensor,
    reference_h_inf: torch.Tensor,
    *,
    log_beta_lo: float = -4.0,
    log_beta_hi: float = 4.0,
    n_iter: int = 30,
) -> torch.Tensor:
    """Rényi-∞ β̂ estimator via bisection.

    Find β̂_t such that H_∞(softmax(ℓ_t / β̂_t)) = H_∞^(ref).

    Monotonicity argument: as c grows, ℓ/c flattens, softmax(ℓ/c) approaches
    uniform, H_∞ → log V. As c → 0⁺, softmax(ℓ/c) concentrates on argmax,
    H_∞ → 0. So H_∞(softmax(ℓ/c)) is strictly monotone-increasing in c.

    Interpretation: if H_∞(softmax(ℓ_t)) is SMALLER than reference (sharper
    than at t=0), then β̂ > 1 — the model has self-imprinted. Dividing ℓ_t by
    β̂ recovers the reference shape.

    Args:
        logits: (B, V).
        reference_h_inf: (B,) reference H_∞ captured at t=0.
        log_beta_lo, log_beta_hi: bisection bracket on log β̂.
        n_iter: bisection iterations (30 → ~1e-9 precision on log β̂).

    Returns:
        beta_hat: (B,) point estimate.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got {tuple(logits.shape)}")
    B = logits.shape[0]
    if reference_h_inf.shape != (B,):
        raise ValueError(
            f"reference_h_inf must be (B,)={B}, got {tuple(reference_h_inf.shape)}"
        )
    device = logits.device
    dtype = logits.dtype
    lo = torch.full((B,), float(log_beta_lo), device=device, dtype=dtype)
    hi = torch.full((B,), float(log_beta_hi), device=device, dtype=dtype)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        beta = torch.exp(mid)
        scaled = logits / beta.unsqueeze(-1)
        h = renyi_infty_from_logits(scaled)
        too_low = h < reference_h_inf          # need bigger β
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    return torch.exp(0.5 * (lo + hi))


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
