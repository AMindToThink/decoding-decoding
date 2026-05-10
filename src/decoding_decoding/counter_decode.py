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
    """Bayes update on the discrete grid for a batch of trajectories.

    Likelihood at grid point β_g for trajectory n (using bootstrap reference
    ℓ*_t = ℓ_t / β̂_t per-trajectory):

        log P(x_t | β_g, ℓ*_t)
          = β_g · ℓ*_t[x_t] − logsumexp_v(β_g · ℓ*_t[v])

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
