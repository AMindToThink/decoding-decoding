"""Counter-decoding generation harness.

Step-by-step batched decoding under a per-step temperature schedule, supporting
both the corrected arm (β_dec_t = β_target / β̂_t with β̂_t from the streaming
Laplace filter) and the uncorrected arm (β_dec_t = β_target fixed).

Why HuggingFace transformers and not vLLM:
  * The corrected sampling distribution at step t depends on per-trajectory
    Laplace state, which depends on logits and observations from earlier
    steps. We need full vocab logits per step (for E_q, Var_q in the Laplace
    update). Manual stepwise control is straightforward in transformers.
  * Per-trajectory full-vocab logit save at sparse positions is needed for
    the SVD analysis in step 1 of the plan.

Output layout (under `out_dir`):
    out_dir/
        config.json
        manifest.parquet                   # one row per trajectory
        traces/{trace_id}.npz              # per-trajectory data
        sparse_logits/{trace_id}.npz       # per-trajectory full-vocab logits at sparse t

Per-trajectory npz schema (traces/):
    top_token_ids          (L, top_n) int32   — top-N tokens of P_φ each step
    top_logprobs           (L, top_n) float32 — corresponding natural-log probs
    sampled_token_ids      (L,)       int64
    beta_hat_pre           (L,)       float32 — β̂ at the start of step t
    beta_dec               (L,)       float32 — β_dec used at step t
    J_pre                  (L,)       float32 — J at the start of step t
    score                  (L,)       float32 — Laplace score added at step t
    fisher                 (L,)       float32 — Laplace fisher (Var_q) at step t
    max_p_phi              (L,)       float32 — max prob under P_φ (full vocab)
    entropy_phi            (L,)       float32 — entropy(P_φ) in nats (full vocab)
    max_p_sample           (L,)       float32 — max prob under P_sample (full vocab)
    entropy_sample         (L,)       float32 — entropy(P_sample) in nats (full vocab)
    eta_hat_post_final     scalar     float32 — final η̂ AFTER the last update
    J_post_final           scalar     float32 — final J AFTER the last update

Per-trajectory sparse_logits npz (only for arm="uncorrected"):
    positions  (P,)    int32   — the step indices at which we saved
    logits     (P, V)  float16 — pre-decoding logits at those steps
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from decoding_decoding.counter_decode import (
    DiscreteGridState,
    GridMixtureState,
    LaplaceState,
    discrete_grid_update_batched,
    estimate_beta_rank_gap,
    estimate_beta_renyi_infty,
    grid_mixture_level_set,
    grid_mixture_update_batched,
    init_discrete_grid_state,
    init_grid_mixture,
    init_laplace,
    laplace_update,
    make_beta_grid,
    rank_gaps,
    renyi_infty_from_logits,
    sample_under_beta,
    topk_logits_sorted,
)


SHAPE_ESTIMATORS = ("rank_gap", "renyi_infty")
from decoding_decoding.data_layout import (
    MANIFEST_FILENAME,
    encode_params,
    save_trace,
    trace_relpath,
    upsert_manifest,
)
from decoding_decoding.prompts import PROMPTS


MODEL_NAME = "Qwen/Qwen2.5-3B"
TOP_LOGPROBS = 200
SEED_BASE = 42
COUNTER_SEED_OFFSET = 300_000
DEFAULT_SPARSE_POSITIONS: tuple[int, ...] = (8, 16, 32, 64, 128, 199)

# Certificate backend (prior_kind="grid_mixture") default grid — matches
# scripts/grid_certificate_experiments.py (GRID_LOG_LO/HI/POINTS) so the
# in-the-loop filter shares the certificate's grid semantics. Overridable per
# run via prior_kwargs {log_lo, log_hi, n_points}.
GRID_MIXTURE_LOG_LO, GRID_MIXTURE_LOG_HI, GRID_MIXTURE_POINTS = -3.0, 3.0, 121


# --------------------------------------------------------------------------
# Per-trajectory specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterDecodeSpec:
    prompt_id: int
    prompt: str
    beta_target: float
    corrected: bool
    sigma_0: float
    run_idx: int
    seed: int

    @property
    def trace_id(self) -> str:
        arm = "corr" if self.corrected else "unco"
        # Encode beta_target with two decimal digits but no decimal in the id.
        bt_str = f"{self.beta_target:.2f}".replace(".", "p")
        return f"counter_p{self.prompt_id:02d}_b{bt_str}_{arm}_r{self.run_idx:02d}"

    @property
    def condition_label(self) -> str:
        arm = "corr" if self.corrected else "unco"
        return f"btarget={self.beta_target:.2f}_arm={arm}"


def _build_specs(
    *,
    beta_target_values: Sequence[float],
    corrected_arms: Sequence[bool],
    n_runs_per_prompt: int,
    sigma_0: float,
    prompts: Sequence[str] = PROMPTS,
) -> list[CounterDecodeSpec]:
    specs: list[CounterDecodeSpec] = []
    for bt in beta_target_values:
        for corr in corrected_arms:
            for pid, prompt in enumerate(prompts):
                for r in range(n_runs_per_prompt):
                    seed = COUNTER_SEED_OFFSET + int(round(bt * 100)) * 100_000 + (
                        1 if corr else 0
                    ) * 10_000 + pid * 100 + r
                    specs.append(
                        CounterDecodeSpec(
                            prompt_id=pid,
                            prompt=prompt,
                            beta_target=float(bt),
                            corrected=bool(corr),
                            sigma_0=float(sigma_0),
                            run_idx=r,
                            seed=seed,
                        )
                    )
    return specs


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def _load_model_and_tokenizer(
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load Qwen2.5-3B with HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Left-pad so that logits[:, -1, :] is always the next-token distribution
    # for the real (non-padded) sequence end.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    return model, tok


# --------------------------------------------------------------------------
# Generation core
# --------------------------------------------------------------------------


def _full_vocab_summary_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (max_prob, entropy_nats) on the full vocabulary.

    logits: (B, V) — return shape (B,) for both stats.
    Entropy uses the natural log: H = −Σ_v p_v log p_v.
    """
    log_p = torch.log_softmax(logits.float(), dim=-1)
    p = torch.exp(log_p)
    max_prob = p.max(dim=-1).values
    entropy = -(p * log_p).sum(dim=-1)
    return max_prob, entropy


def _topn_extract(logits: torch.Tensor, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return top-N (token_ids, log_probs) per batch element.

    Uses log_softmax over the full vocab (so the saved logprobs are the
    true natural-log probabilities under P_φ, sortable across batches).
    """
    log_p = torch.log_softmax(logits.float(), dim=-1)
    top_lp, top_idx = torch.topk(log_p, k=n, dim=-1)
    return top_idx.cpu().numpy().astype(np.int32), top_lp.cpu().numpy().astype(np.float32)


@dataclass
class GenerationResult:
    """Per-trajectory data for a batch of `B` trajectories run in parallel.

    NOTE — diagnostic-slot semantics depend on ``prior_kind`` (recorded in each
    trace's params, so consumers can tell which applies):
      * Laplace / discrete-grid backends: ``J_pre`` is a precision (1/var on
        log β), ``score`` is a log-β-change score, ``fisher`` a precision change.
      * ``grid_mixture`` (certificate) backend: there is NO precision analog, so
        the slots are repurposed — ``J_pre`` (and ``J_post_final``) carry the
        level-set WIDTH in log β (uncertainty, not precision), ``score`` carries
        the realized regret (mixture loss − best-expert loss), and ``fisher``
        carries the regret bound ln(1/π₀[best]) (uniform prior ⇒ ln G). β̂
        (``beta_hat_pre``) is the grid MAP in every backend.
    """

    sampled_token_ids: np.ndarray   # (B, T) int64
    top_token_ids: np.ndarray       # (B, T, top_n) int32
    top_logprobs: np.ndarray        # (B, T, top_n) float32
    beta_hat_pre: np.ndarray        # (B, T) float32  — β̂ at start of step t
    beta_dec: np.ndarray            # (B, T) float32
    J_pre: np.ndarray               # (B, T) float32 — precision; grid_mixture: level-set width
    score: np.ndarray               # (B, T) float32 — grid_mixture: realized regret
    fisher: np.ndarray              # (B, T) float32 — grid_mixture: regret bound (ln G)
    max_p_phi: np.ndarray           # (B, T) float32
    entropy_phi: np.ndarray         # (B, T) float32
    max_p_sample: np.ndarray        # (B, T) float32
    entropy_sample: np.ndarray      # (B, T) float32
    eta_hat_post_final: np.ndarray  # (B,) float32
    J_post_final: np.ndarray        # (B,) float32
    decoded_text: list[str]         # length B
    # Optional: full-vocab logits at chosen positions (only for uncorrected arm).
    sparse_positions: np.ndarray | None  # (P,) int32
    sparse_logits: np.ndarray | None     # (B, P, V) float16


def _generate_batch(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    beta_target_per_traj: torch.Tensor,   # (B,)
    corrected_per_traj: torch.Tensor,     # (B,) bool
    sigma_0_per_traj: torch.Tensor,       # (B,)
    seeds_per_traj: Sequence[int],
    max_tokens: int,
    top_n: int,
    sparse_positions: tuple[int, ...] | None,
    save_sparse_for_mask: torch.Tensor,   # (B,) bool — True where we save sparse logits
    device: str,
    prior_kind: str = "lognormal_laplace",
    prior_kwargs: dict | None = None,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
    correction_exponent: float = 1.0,
    switch_rate: float = 0.0,
) -> GenerationResult:
    """Run `len(prompts)` trajectories in parallel through `max_tokens` steps.

    Each trajectory has its own β_target, arm, σ_0, and seed. The Laplace state
    is per-trajectory. Sampling RNG is per-trajectory via separate generators
    (one shared CUDA generator and we use random uint64 per step → not seedable
    per traj).

    NOTE on per-trajectory seeding: torch.multinomial does not accept a vector
    of generators. We therefore mix per-trajectory seeds into the logits via
    a torch generator seeded with hash(seed, step) and call multinomial on
    one trajectory at a time to keep deterministic per-traj reproducibility.
    For correctness we just need different draws — the per-traj seed mixing
    handles that.
    """
    B = len(prompts)
    assert beta_target_per_traj.shape == (B,)
    assert corrected_per_traj.shape == (B,)
    assert sigma_0_per_traj.shape == (B,)
    assert save_sparse_for_mask.shape == (B,)

    # Tokenize and left-pad.
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True)
    input_ids: torch.Tensor = enc["input_ids"].to(device)              # (B, P)
    attention_mask: torch.Tensor = enc["attention_mask"].to(device)    # (B, P)

    P = input_ids.shape[1]
    V = model.config.vocab_size

    # Initialize per-trajectory β estimator state. Four backends:
    #   - "lognormal_laplace": original streaming Laplace on log β (vectorized).
    #   - other priors via discrete grid: "lognormal", "exponential", "gamma",
    #     "invgamma", "halfcauchy_logβ", "cauchy_logβ", "uniform_logβ".
    #   - "grid_mixture": guarantee-bearing certificate filter (expert-advice
    #     grid mixture / Vovk AA). β̂ is the grid MAP; the only knob is
    #     switch_rate; evidence_weight/memory_decay are forbidden (they break
    #     1-mixability). See GridMixtureState in counter_decode.py.
    #   - shape estimators "rank_gap" / "renyi_infty": non-bootstrap, read β̂
    #     from ℓ_t shape relative to a t=0 reference. No filter state, just a
    #     stored reference per trajectory.
    # Within one batch we require all trajectories share the same prior_kind
    # (the design choice is per-condition, not per-trajectory).
    laplace_state: LaplaceState | None = None
    grid_state: DiscreteGridState | None = None
    mixture_state: GridMixtureState | None = None
    laplace_prior_eta: float = 0.0
    laplace_prior_J: float = 0.0
    grid_prior_pmf: torch.Tensor | None = None
    # Shape-estimator references (captured at t=0).
    rank_gap_K: int = 20
    rank_gap_reference: torch.Tensor | None = None     # (B, K)
    renyi_reference: torch.Tensor | None = None        # (B,)
    pkw = dict(prior_kwargs or {})
    if prior_kind in SHAPE_ESTIMATORS:
        # No streaming state; references will be captured at t=0 below.
        pass
    elif prior_kind == "lognormal_laplace":
        eta_hat = torch.empty(B, device=device, dtype=torch.float32)
        J = torch.empty(B, device=device, dtype=torch.float32)
        for b in range(B):
            s = init_laplace(1, sigma_0=float(sigma_0_per_traj[b].item()),
                             device=device, dtype=torch.float32)
            eta_hat[b] = s.eta_hat[0]
            J[b] = s.J[0]
        laplace_state = LaplaceState(eta_hat=eta_hat, J=J)
        # Anchor for memory_decay.
        sigma_0_val = float(sigma_0_per_traj[0].item())
        laplace_prior_eta = -0.5 * sigma_0_val ** 2
        laplace_prior_J = 1.0 / sigma_0_val ** 2
    elif prior_kind == "grid_mixture":
        # Certificate backend. NOT a copy of the discrete-grid branch: experts
        # are FIXED forecasters on RAW logits, β̂ is the grid MAP, and the prior
        # is uniform (clean ln G regret bound). evidence_weight and memory_decay
        # are forbidden — they destroy the 1-mixability the guarantees rest on
        # (grid_mixture_update_batched would raise on evidence_weight anyway).
        # The only knob is switch_rate (Fixed-Share tracking). sigma_0 is unused.
        if evidence_weight != 1.0:
            raise ValueError(
                "prior_kind='grid_mixture' forbids evidence_weight != 1 "
                "(it breaks the 1-mixability of log loss and voids the certificate)"
            )
        if memory_decay != 1.0:
            raise ValueError(
                "prior_kind='grid_mixture' forbids memory_decay != 1 "
                "(no analog; use switch_rate for tracking instead)"
            )
        gm_pkw = dict(pkw)
        beta_grid = make_beta_grid(
            log_lo=gm_pkw.pop("log_lo", GRID_MIXTURE_LOG_LO),
            log_hi=gm_pkw.pop("log_hi", GRID_MIXTURE_LOG_HI),
            n_points=gm_pkw.pop("n_points", GRID_MIXTURE_POINTS),
            device=device,
            dtype=torch.float32,
        )
        if gm_pkw:
            raise ValueError(
                f"prior_kind='grid_mixture' got unexpected prior_kwargs "
                f"{sorted(gm_pkw)}; only log_lo/log_hi/n_points are accepted "
                "(the certificate uses a uniform prior by construction)"
            )
        mixture_state = init_grid_mixture(batch_size=B, beta_grid=beta_grid)
    else:
        sigma_0_val = float(sigma_0_per_traj[0].item())
        if not (sigma_0_per_traj == sigma_0_per_traj[0]).all():
            raise ValueError(
                "discrete-grid filter currently shares prior across the batch; "
                "all sigma_0_per_traj must be identical"
            )
        grid_state = init_discrete_grid_state(
            batch_size=B,
            prior_kind=prior_kind,
            sigma_0=sigma_0_val,
            device=device,
            dtype=torch.float32,
            **pkw,
        )
        grid_prior_pmf = grid_state.pi[0].clone()  # (G,) — anchor for decay

    # Prefill
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    logits_t: torch.Tensor = out.logits[:, -1, :].float()  # (B, V)
    past = out.past_key_values

    # For shape estimators, compute the natural-temperature reference from
    # the LAST few real prompt positions. Each is at β_model = 1 (prompt is
    # given text, not model output), and the last-K window captures the
    # model's logit shape RIGHT BEFORE generation starts — which is the
    # state we actually want β̂ to be calibrated against. Averaging over
    # ALL prompt positions conflates with diffuse early-prompt positions
    # and yields a too-sharp reference; averaging over the last K=5 gives
    # noise reduction without that bias.
    REF_WINDOW = 5
    if prior_kind in SHAPE_ESTIMATORS:
        prompt_logits = out.logits.float()                              # (B, P, V)
        mask_f = attention_mask.to(prompt_logits.dtype)                  # (B, P)
        # Build a weight mask over the last REF_WINDOW REAL positions per row.
        # cum_from_end counts real positions counting back from the end.
        rev_cum = mask_f.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])     # (B, P)
        in_window = (rev_cum <= REF_WINDOW) & (mask_f > 0)               # (B, P)
        w = in_window.to(prompt_logits.dtype)                            # (B, P)
        if prior_kind == "rank_gap":
            top_p = topk_logits_sorted(
                prompt_logits.reshape(-1, V), rank_gap_K
            )                                                            # (B*P, K+1)
            gaps_p = rank_gaps(top_p).reshape(B, P, rank_gap_K)          # (B, P, K)
            ww = w.unsqueeze(-1)                                         # (B, P, 1)
            denom = ww.sum(dim=1).clamp_min(1.0)                         # (B, 1)
            rank_gap_reference = (gaps_p * ww).sum(dim=1) / denom        # (B, K)
        elif prior_kind == "renyi_infty":
            h_p = renyi_infty_from_logits(
                prompt_logits.reshape(-1, V)
            ).reshape(B, P)                                              # (B, P)
            denom = w.sum(dim=1).clamp_min(1.0)                          # (B,)
            renyi_reference = (h_p * w).sum(dim=1) / denom               # (B,)

    # Storage
    T = max_tokens
    sampled = torch.zeros(B, T, device=device, dtype=torch.long)
    top_ids = np.zeros((B, T, top_n), dtype=np.int32)
    top_lps = np.zeros((B, T, top_n), dtype=np.float32)
    beta_hat_pre_arr = np.zeros((B, T), dtype=np.float32)
    beta_dec_arr = np.zeros((B, T), dtype=np.float32)
    J_pre_arr = np.zeros((B, T), dtype=np.float32)
    score_arr = np.zeros((B, T), dtype=np.float32)
    fisher_arr = np.zeros((B, T), dtype=np.float32)
    max_p_phi_arr = np.zeros((B, T), dtype=np.float32)
    entropy_phi_arr = np.zeros((B, T), dtype=np.float32)
    max_p_sample_arr = np.zeros((B, T), dtype=np.float32)
    entropy_sample_arr = np.zeros((B, T), dtype=np.float32)

    sparse_buf: dict[int, torch.Tensor] = {}
    sp_positions: list[int] = []
    if sparse_positions is not None and save_sparse_for_mask.any().item():
        sp_positions = [p for p in sparse_positions if 0 <= p < T]

    # Per-trajectory generator for reproducible sampling.
    # CUDA's per-tensor-row sampling with shared generator is OK; for true
    # per-traj reproducibility we sample row-by-row with per-traj generators.
    # That's slow if B is large. Compromise: one generator, seeded by hash
    # of all seeds. Reproducibility-per-trajectory only when seed is unique.
    combined_seed = int(np.uint64(np.array(list(seeds_per_traj), dtype=np.uint64).sum()))
    gen = torch.Generator(device=device).manual_seed(combined_seed % (2 ** 31 - 1))

    for t in range(T):
        if laplace_state is not None:
            beta_hat_now = laplace_state.beta_hat()
            J_now = laplace_state.J
        elif grid_state is not None:
            beta_hat_now = grid_state.beta_hat()
            # No "J" in the discrete filter; record posterior precision on log β as 1/std².
            std_lb = grid_state.posterior_std_log_beta()
            J_now = 1.0 / std_lb.clamp_min(1e-3) ** 2
        elif mixture_state is not None:
            # Certificate backend: β̂ is the grid MAP. There is NO precision
            # analog J; the honest per-step uncertainty is the level-set WIDTH
            # in log β. We carry that width in the J_pre slot (see the
            # GenerationResult note — for grid_mixture, J_pre is a width, NOT a
            # precision). Read from the pre-update state.
            beta_hat_now = mixture_state.beta_map()
            lo_t, hi_t = grid_mixture_level_set(mixture_state)
            J_now = hi_t - lo_t                       # (B,) width in log β
        elif prior_kind == "rank_gap":
            assert rank_gap_reference is not None
            beta_hat_now = estimate_beta_rank_gap(logits_t, rank_gap_reference)
            J_now = torch.zeros(B, device=device, dtype=torch.float32)
        elif prior_kind == "renyi_infty":
            assert renyi_reference is not None
            beta_hat_now = estimate_beta_renyi_infty(logits_t, renyi_reference)
            J_now = torch.zeros(B, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"unknown prior_kind: {prior_kind}")

        # β_dec_t for corrected arm: β_target / β̂_t^p
        #   p = 1: full correction (cancel the imprint fully)
        #   p = 0: no correction (β_target alone — equivalent to the uncorrected arm)
        #   p ∈ (0, 1): partial correction — assume the LLM's belief is only
        #              partially up-to-date with what β̂_t says.
        # Uncorrected arm uses β_target unchanged.
        beta_hat_corrected = beta_hat_now ** correction_exponent
        beta_dec_t = torch.where(
            corrected_per_traj,
            beta_target_per_traj / beta_hat_corrected,
            beta_target_per_traj,
        )

        # Summary stats on P_φ (full vocab) at this step.
        max_p_phi_t, entropy_phi_t = _full_vocab_summary_stats(logits_t)

        # Stats on P_sample = softmax(β_dec · ℓ_t).
        sample_logits = logits_t * beta_dec_t.unsqueeze(-1)
        max_p_sample_t, entropy_sample_t = _full_vocab_summary_stats(sample_logits)

        # Top-N for diagnostics / archival.
        ti, tl = _topn_extract(logits_t, top_n)

        # Sample under β_dec.
        x_t = sample_under_beta(logits_t, beta_dec_t, generator=gen)  # (B,)

        # Streaming filter update (Laplace OR discrete-grid), with optional
        # evidence_weight (α) and memory_decay (γ). Shape estimators have no
        # state to update — β̂ is read off ℓ_t and a fixed reference.
        if laplace_state is not None:
            laplace_state, score_t, fisher_t = laplace_update(
                laplace_state,
                logits_t,
                x_t,
                evidence_weight=evidence_weight,
                memory_decay=memory_decay,
                prior_eta=laplace_prior_eta,
                prior_J=laplace_prior_J,
            )
        elif grid_state is not None:
            grid_state = discrete_grid_update_batched(
                grid_state,
                logits_t,
                x_t,
                evidence_weight=evidence_weight,
                memory_decay=memory_decay,
                prior_pmf=grid_prior_pmf,
            )
            # Diagnostic score / fisher are not strictly defined here. Use placeholders
            # that still convey something: log-difference in β̂ (score-like) and
            # change in posterior precision (fisher-like).
            new_bh = grid_state.beta_hat()
            score_t = torch.log(new_bh) - torch.log(beta_hat_now)
            new_J = 1.0 / grid_state.posterior_std_log_beta().clamp_min(1e-3) ** 2
            fisher_t = (new_J - J_now).clamp_min(0.0)
        elif mixture_state is not None:
            # Certificate update: score experts on RAW logits, Bayes-mix. The
            # only knob is switch_rate (Fixed-Share). The score/fisher diagnostic
            # slots are repurposed (see GenerationResult note):
            #   score  ← realized regret  = mixture loss − best-expert loss
            #   fisher ← regret bound     = ln(1/π₀[best]) (uniform ⇒ ln G)
            # so a consumer can verify realized ≤ bound straight from the trace.
            mixture_state = grid_mixture_update_batched(
                mixture_state, logits_t, x_t, switch_rate=switch_rate
            )
            score_t = mixture_state.realized_regret()
            fisher_t = mixture_state.regret_bound()
        else:
            # Shape estimators: no filter update. β̂_{t+1} will be computed from
            # the next step's ℓ_{t+1} and the same fixed reference.
            score_t = torch.zeros(B, device=device, dtype=torch.float32)
            fisher_t = torch.zeros(B, device=device, dtype=torch.float32)

        # Save per-step diagnostics.
        sampled[:, t] = x_t
        top_ids[:, t, :] = ti
        top_lps[:, t, :] = tl
        beta_hat_pre_arr[:, t] = beta_hat_now.detach().cpu().numpy()
        beta_dec_arr[:, t] = beta_dec_t.detach().cpu().numpy()
        J_pre_arr[:, t] = J_now.detach().cpu().numpy()
        score_arr[:, t] = score_t.detach().cpu().numpy()
        fisher_arr[:, t] = fisher_t.detach().cpu().numpy()
        max_p_phi_arr[:, t] = max_p_phi_t.detach().cpu().numpy()
        entropy_phi_arr[:, t] = entropy_phi_t.detach().cpu().numpy()
        max_p_sample_arr[:, t] = max_p_sample_t.detach().cpu().numpy()
        entropy_sample_arr[:, t] = entropy_sample_t.detach().cpu().numpy()

        if t in sp_positions:
            sparse_buf[t] = logits_t.detach().to(torch.float16).cpu()

        if t < T - 1:
            # Decode-step forward pass with KV cache.
            new_token = x_t.unsqueeze(-1)  # (B, 1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, device=device, dtype=attention_mask.dtype)],
                dim=-1,
            )
            with torch.no_grad():
                out = model(
                    input_ids=new_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            logits_t = out.logits[:, -1, :].float()
            past = out.past_key_values

    # Decoded text per trajectory.
    decoded_text: list[str] = []
    for b in range(B):
        ids = sampled[b].cpu().tolist()
        decoded_text.append(tokenizer.decode(ids, skip_special_tokens=False))

    # Stack sparse logits if any.
    sparse_pos_arr: np.ndarray | None = None
    sparse_logits_arr: np.ndarray | None = None
    if sp_positions and sparse_buf:
        sparse_pos_arr = np.array(sorted(sparse_buf.keys()), dtype=np.int32)
        # (B, P, V) float16 — only meaningful where save_sparse_for_mask is True;
        # we still store all rows for layout simplicity but the caller decides
        # which to write to disk.
        stacked = torch.stack([sparse_buf[int(p)] for p in sparse_pos_arr], dim=1)  # (B, P, V)
        sparse_logits_arr = stacked.numpy().astype(np.float16)

    if laplace_state is not None:
        eta_final = laplace_state.eta_hat.detach().cpu().numpy().astype(np.float32)
        J_final = laplace_state.J.detach().cpu().numpy().astype(np.float32)
    elif grid_state is not None:
        # For the discrete filter, η̂_final is log(posterior mean), and J_final
        # is the inverse posterior variance on log β.
        eta_final = torch.log(grid_state.beta_hat()).detach().cpu().numpy().astype(np.float32)
        J_final = (
            (1.0 / grid_state.posterior_std_log_beta().clamp_min(1e-3) ** 2)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    elif mixture_state is not None:
        # Certificate filter: η̂_final = log(grid MAP); J_final carries the final
        # level-set WIDTH in log β (uncertainty, not a precision — same override
        # as the per-step J_pre slot).
        eta_final = torch.log(mixture_state.beta_map()).detach().cpu().numpy().astype(np.float32)
        lo_f, hi_f = grid_mixture_level_set(mixture_state)
        J_final = (hi_f - lo_f).detach().cpu().numpy().astype(np.float32)
    else:
        # Shape estimator: no posterior. Take the last-step β̂ as the "final"
        # estimate, and zero out the precision (J undefined for shape readout).
        eta_final = np.log(beta_hat_pre_arr[:, -1]).astype(np.float32)
        J_final = np.zeros(B, dtype=np.float32)

    return GenerationResult(
        sampled_token_ids=sampled.cpu().numpy().astype(np.int64),
        top_token_ids=top_ids,
        top_logprobs=top_lps,
        beta_hat_pre=beta_hat_pre_arr,
        beta_dec=beta_dec_arr,
        J_pre=J_pre_arr,
        score=score_arr,
        fisher=fisher_arr,
        max_p_phi=max_p_phi_arr,
        entropy_phi=entropy_phi_arr,
        max_p_sample=max_p_sample_arr,
        entropy_sample=entropy_sample_arr,
        eta_hat_post_final=eta_final,
        J_post_final=J_final,
        decoded_text=decoded_text,
        sparse_positions=sparse_pos_arr,
        sparse_logits=sparse_logits_arr,
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _save_trace_extended(
    out_dir: Path,
    trace_id: str,
    result: GenerationResult,
    b_idx: int,
) -> None:
    """Write per-trajectory data to traces/{trace_id}.npz.

    We extend the basic two-array layout from data_layout.save_trace with
    additional per-step arrays. The base layout is preserved so existing
    consumers (analyze.py) can read top_token_ids and top_logprobs.
    """
    npz_path = Path(out_dir) / "traces" / f"{trace_id}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.parent / f"{npz_path.stem}.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            top_token_ids=result.top_token_ids[b_idx].astype(np.int32),
            top_logprobs=result.top_logprobs[b_idx].astype(np.float32),
            sampled_token_ids=result.sampled_token_ids[b_idx].astype(np.int64),
            beta_hat_pre=result.beta_hat_pre[b_idx].astype(np.float32),
            beta_dec=result.beta_dec[b_idx].astype(np.float32),
            J_pre=result.J_pre[b_idx].astype(np.float32),
            score=result.score[b_idx].astype(np.float32),
            fisher=result.fisher[b_idx].astype(np.float32),
            max_p_phi=result.max_p_phi[b_idx].astype(np.float32),
            entropy_phi=result.entropy_phi[b_idx].astype(np.float32),
            max_p_sample=result.max_p_sample[b_idx].astype(np.float32),
            entropy_sample=result.entropy_sample[b_idx].astype(np.float32),
            eta_hat_post_final=np.asarray(result.eta_hat_post_final[b_idx], dtype=np.float32),
            J_post_final=np.asarray(result.J_post_final[b_idx], dtype=np.float32),
        )
    tmp.replace(npz_path)


def _save_sparse_logits(
    out_dir: Path,
    trace_id: str,
    positions: np.ndarray,
    logits: np.ndarray,
) -> None:
    """Save full-vocab logits at sparse positions to sparse_logits/{trace_id}.npz."""
    p = Path(out_dir) / "sparse_logits" / f"{trace_id}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f"{p.stem}.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            positions=positions.astype(np.int32),
            logits=logits.astype(np.float16),
        )
    tmp.replace(p)


def _write_config(out_dir: Path, **kwargs) -> None:
    cfg = {"model": MODEL_NAME, **kwargs}
    (Path(out_dir) / "config.json").write_text(json.dumps(cfg, indent=2))


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------


def run_counter_decode_experiment(
    out_dir: Path,
    *,
    beta_target_values: Sequence[float] = (0.5, 0.7, 1.0, 1.3, 1.7, 2.0),
    corrected_arms: Sequence[bool] = (True, False),
    n_runs_per_prompt: int = 3,
    max_tokens: int = 200,
    sigma_0: float = 0.5,
    sparse_positions: tuple[int, ...] = DEFAULT_SPARSE_POSITIONS,
    top_n: int = TOP_LOGPROBS,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    prior_kind: str = "lognormal_laplace",
    prior_kwargs: dict | None = None,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
    correction_exponent: float = 1.0,
    switch_rate: float = 0.0,
) -> None:
    """Generate counter-decoding trajectories for the F0 sweep.

    For every (β_target, arm, prompt, run) trajectory:
      * runs `max_tokens` decoding steps
      * applies the streaming Laplace filter for diagnostics in both arms
      * applies β_dec_t = β_target/β̂_t (corrected) or β_target (uncorrected)
      * saves top-N logprobs + per-step diagnostics + sparse full-vocab logits

    All trajectories under one (β_target, arm) condition are batched together
    in a single decode loop to maximize GPU utilization. The sigma_0 hyperparam
    is shared across the whole run.

    Args:
        out_dir: target data directory.
        beta_target_values: sweep values for β_target.
        corrected_arms: which arms to run, e.g. (True, False).
        n_runs_per_prompt: trajectories per prompt per (β, arm) condition.
        max_tokens: number of tokens to generate per trajectory.
        sigma_0: prior std on log β.
        sparse_positions: step indices at which to save full-vocab logits
            (only applied to the uncorrected arm; needed for the SVD step).
        top_n: how many top logprobs to save per step.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CUDA device selection: respect CUDA_VISIBLE_DEVICES; default to GPU 0.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"[counter-decode] loading {MODEL_NAME} ...")
    model, tokenizer = _load_model_and_tokenizer(device=device, dtype=dtype)

    specs = _build_specs(
        beta_target_values=beta_target_values,
        corrected_arms=corrected_arms,
        n_runs_per_prompt=n_runs_per_prompt,
        sigma_0=sigma_0,
    )
    print(f"[counter-decode] {len(specs)} trajectories total")

    # Group specs by (β_target, arm) so we can batch them.
    from collections import defaultdict
    by_cond: dict[tuple[float, bool], list[CounterDecodeSpec]] = defaultdict(list)
    for s in specs:
        by_cond[(s.beta_target, s.corrected)].append(s)

    new_rows: list[dict] = []

    for (bt, corr), batch_specs in sorted(by_cond.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cond_t0 = time.time()
        B = len(batch_specs)
        prompts_b = [s.prompt for s in batch_specs]
        beta_target_per = torch.tensor([bt] * B, dtype=torch.float32, device=device)
        corrected_per = torch.tensor([corr] * B, dtype=torch.bool, device=device)
        sigma_0_per = torch.tensor([sigma_0] * B, dtype=torch.float32, device=device)
        seeds = [s.seed for s in batch_specs]

        # Save sparse logits only for uncorrected (step-1 SVD).
        save_sparse_mask = torch.tensor(
            [not corr] * B, dtype=torch.bool, device=device
        )

        print(
            f"[counter-decode] β_target={bt:.2f} arm={'corrected' if corr else 'uncorrected'} "
            f"| {B} trajectories × {max_tokens} tokens"
        )
        result = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts_b,
            beta_target_per_traj=beta_target_per,
            corrected_per_traj=corrected_per,
            sigma_0_per_traj=sigma_0_per,
            seeds_per_traj=seeds,
            max_tokens=max_tokens,
            top_n=top_n,
            sparse_positions=sparse_positions if not corr else None,
            save_sparse_for_mask=save_sparse_mask,
            device=device,
            prior_kind=prior_kind,
            prior_kwargs=prior_kwargs,
            evidence_weight=evidence_weight,
            memory_decay=memory_decay,
            correction_exponent=correction_exponent,
            switch_rate=switch_rate,
        )

        for b, spec in enumerate(batch_specs):
            _save_trace_extended(out_dir, spec.trace_id, result, b)
            if (
                not spec.corrected
                and result.sparse_positions is not None
                and result.sparse_logits is not None
            ):
                _save_sparse_logits(
                    out_dir,
                    spec.trace_id,
                    result.sparse_positions,
                    result.sparse_logits[b],
                )
            params = {
                "beta_target": float(spec.beta_target),
                "corrected": bool(spec.corrected),
                "sigma_0": float(spec.sigma_0),
                "prior_kind": str(prior_kind),
                "evidence_weight": float(evidence_weight),
                "memory_decay": float(memory_decay),
                "correction_exponent": float(correction_exponent),
                "switch_rate": float(switch_rate),
            }
            new_rows.append(
                {
                    "trace_id": spec.trace_id,
                    "family": "counter_decode",
                    "prompt_id": spec.prompt_id,
                    "prompt_text": spec.prompt,
                    "condition_label": spec.condition_label,
                    "params_json": encode_params(params),
                    "run_idx": spec.run_idx,
                    "seed": spec.seed,
                    "length": max_tokens,
                    "sampled_token_ids": list(map(int, result.sampled_token_ids[b].tolist())),
                    "decoded_text": result.decoded_text[b],
                    "trace_path": trace_relpath(spec.trace_id),
                }
            )
        elapsed = time.time() - cond_t0
        print(
            f"[counter-decode]   ↳ {elapsed:.1f}s  ({B * max_tokens / elapsed:.0f} tok/s effective)"
        )

    upsert_manifest(out_dir, new_rows)
    _write_config(
        out_dir,
        family="counter_decode",
        beta_target_values=list(beta_target_values),
        corrected_arms=list(corrected_arms),
        n_runs_per_prompt=int(n_runs_per_prompt),
        max_tokens=int(max_tokens),
        sigma_0=float(sigma_0),
        sparse_positions=list(sparse_positions),
        top_n=int(top_n),
        n_prompts=len(PROMPTS),
        prompts_sha256=hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
        prior_kind=str(prior_kind),
        prior_kwargs=prior_kwargs or {},
        evidence_weight=float(evidence_weight),
        memory_decay=float(memory_decay),
        correction_exponent=float(correction_exponent),
        switch_rate=float(switch_rate),
    )
    print(f"[counter-decode] wrote {len(new_rows)} trajectories to {out_dir}")
