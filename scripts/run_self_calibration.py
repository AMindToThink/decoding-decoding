"""Streaming temperature self-calibration: controlled recovery (Part 1).

This script implements the experiment in
`~/.claude/plans/it-is-time-to-polished-robin.md`. Part 1 (controlled recovery)
is the clean method demonstration:

  - Generate Qwen self-text at a KNOWN decoder inverse-temperature β = 1/T (seed
    with a short real prefix, then sample every token from softmax(β·ℓ_t)).
  - Run the streaming Laplace filter over the generated tokens — both the
    original (linearize-at-β=1) and the β-corrected `laplace_update_faithful`.
  - (a) Recovery: does β̂_t recover the imposed β = 1/T? The faithful filter is a
        consistent estimator; the original is biased (it never re-linearizes).
  - (b) Prediction: applying the causal β̂_t as a per-token temperature, does it
        lower held-out NLL vs the naive β=1 model and approach the known-optimal
        global temperature 1/T? (Exact per-token NLL — no interpolation.)

Causality: the β̂_t used to score token x_t is recorded BEFORE the filter
consumes x_t (it depends only on x_{<t}).

Parts 0 (calibration survey) and 2 (heterogeneity gate + sweep) are added in
later steps; this file is structured so its helpers are reused there.

Run (GPU0):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_self_calibration.py \
        --part recovery --temps 0.5 0.7 1.0 1.5 2.0 --n-seq 64

Outputs:
    data/self_calibration/recovery.npz                     (raw arrays)
    results/counter_decode/self_calibration/recovery.json  (auditable numbers)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize, stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = REPO_ROOT / "data" / "self_calibration"
RESULTS_OUT = REPO_ROOT / "results" / "counter_decode" / "self_calibration"


# ---------------------------------------------------------------------------
# Shared helpers (reused by survey/gate later)
# ---------------------------------------------------------------------------


def generate_at_beta(
    *,
    model,
    seed_ids: torch.Tensor,        # (B, seed_len) int64 on device
    gen_len: int,
    beta_dec: float,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively sample `gen_len` tokens from softmax(beta_dec · ℓ_t).

    Uses an incremental KV cache. Returns:
        logits_arr: (B, gen_len, V) fp16 — the per-position pre-decoding logits
                    ℓ_t that produced each generated token (kept on device).
        tokens:     (B, gen_len) int64 — the sampled tokens x_t ~ softmax(β·ℓ_t).

    The pair (logits_arr[:, t], tokens[:, t]) is exactly "token x_t was drawn
    from softmax(beta_dec · ℓ_t)", which is what the filter must recover.
    """
    from decoding_decoding.counter_decode import sample_under_beta

    B = seed_ids.shape[0]
    V = model.config.vocab_size
    logits_arr = torch.empty((B, gen_len, V), dtype=torch.float16, device=device)
    tokens = torch.empty((B, gen_len), dtype=torch.long, device=device)
    beta_vec = torch.full((B,), float(beta_dec), device=device, dtype=torch.float32)

    cur = seed_ids
    attn = torch.ones_like(seed_ids)
    past = None
    with torch.no_grad():
        for t in range(gen_len):
            out = model(
                input_ids=cur, attention_mask=attn, past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            logits_t = out.logits[:, -1, :]                      # (B, V) fp16
            logits_arr[:, t, :] = logits_t
            nxt = sample_under_beta(logits_t.float(), beta_vec, generator=generator)
            tokens[:, t] = nxt
            cur = nxt.unsqueeze(-1)                               # (B, 1)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], dim=1)
    return logits_arr, tokens


def run_filter_trajectory(
    *,
    logits_arr: torch.Tensor,      # (B, U, V) fp16 on device
    tokens: torch.Tensor,          # (B, U) int64 on device
    update_fn,                     # laplace_update or laplace_update_faithful
    sigma_0: float,
    device: str,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
) -> np.ndarray:
    """Stream `update_fn` over the sequence; return log β̂_t recorded BEFORE each
    update (causal: log_beta_pre[:, t] depends only on x_{<t}). Shape (B, U)."""
    from decoding_decoding.counter_decode import init_laplace

    B, U, _ = logits_arr.shape
    prior_eta = -0.5 * sigma_0 ** 2
    prior_J = 1.0 / sigma_0 ** 2
    state = init_laplace(B, sigma_0=sigma_0, device=device, dtype=torch.float32)
    log_beta_pre = np.zeros((B, U), dtype=np.float32)
    for t in range(U):
        log_beta_pre[:, t] = state.eta_hat.detach().cpu().numpy()   # pre-update (causal)
        state, _, _ = update_fn(
            state,
            logits_arr[:, t, :].float(),
            tokens[:, t],
            evidence_weight=evidence_weight,
            memory_decay=memory_decay,
            prior_eta=prior_eta,
            prior_J=prior_J,
        )
    final_eta = state.eta_hat.detach().cpu().numpy()                # (B,)
    return log_beta_pre, final_eta


def per_token_nll(
    *,
    logits_arr: torch.Tensor,      # (B, U, V) fp16 on device
    tokens: torch.Tensor,          # (B, U) int64 on device
    tau,                           # float scalar OR (B, U) tensor of per-token temps
    chunk: int = 4096,
) -> np.ndarray:
    """Exact per-token NLL = logsumexp(τ·ℓ) − τ·ℓ[x], chunked over (B·U) rows.

    No grid / interpolation — exact. Returns (B, U) float64 NLL in nats.
    """
    B, U, V = logits_arr.shape
    flat_logits = logits_arr.reshape(B * U, V)
    flat_tokens = tokens.reshape(B * U)
    if isinstance(tau, (int, float)):
        flat_tau = torch.full((B * U,), float(tau), device=logits_arr.device)
    else:
        flat_tau = tau.reshape(B * U).to(logits_arr.device)
    out = np.empty(B * U, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, B * U, chunk):
            lg = flat_logits[i : i + chunk].float()                 # (m, V)
            tk = flat_tokens[i : i + chunk]
            ta = flat_tau[i : i + chunk].unsqueeze(-1)              # (m, 1)
            logq = F.log_softmax(ta * lg, dim=-1)
            lp = logq.gather(-1, tk.unsqueeze(-1)).squeeze(-1)      # (m,)
            out[i : i + chunk] = (-lp).double().cpu().numpy()
    return out.reshape(B, U)


def pooled_mle_beta(
    *,
    logits_arr: torch.Tensor,
    tokens: torch.Tensor,
    bracket: tuple[float, float] = (0.05, 20.0),
) -> float:
    """β minimizing mean per-token NLL over ALL (sequence, position) tokens.

    This is the Guo-MLE global temperature. For text generated at β=1/T it must
    land near 1/T (the sanity gate)."""

    def mean_nll(beta: float) -> float:
        return float(per_token_nll(logits_arr=logits_arr, tokens=tokens, tau=float(beta)).mean())

    res = optimize.minimize_scalar(
        mean_nll, bounds=bracket, method="bounded", options={"xatol": 1e-4}
    )
    return float(res.x)


def load_seed_prompts(*, n_seq: int, seed_len: int, tokenizer, seed: int) -> torch.Tensor:
    """Real short prefixes (half WikiText, half WritingPrompts) → (n_seq, seed_len)."""
    from decoding_decoding.natural_text_filter import (
        _tokenize_to_length,
        load_wikitext_passages,
        load_writingprompts_passages,
    )

    n_wt = n_seq // 2
    n_wp = n_seq - n_wt
    wt = load_wikitext_passages(n=n_wt, min_tokens=seed_len, tokenizer=tokenizer, seed=seed)
    wp = load_writingprompts_passages(n=n_wp, min_tokens=seed_len, tokenizer=tokenizer, seed=seed)
    ids = _tokenize_to_length(wt + wp, tokenizer, length=seed_len)   # (n_seq, seed_len) CPU
    return ids


# ---------------------------------------------------------------------------
# Part 1 — controlled recovery
# ---------------------------------------------------------------------------


def run_recovery(args, model, tokenizer) -> dict:
    from decoding_decoding.counter_decode import laplace_update, laplace_update_faithful

    device = args.device
    gen_len = args.length - args.seed_len
    if gen_len <= args.warmup + 5:
        raise ValueError(f"gen_len={gen_len} too small for warmup={args.warmup}")

    seed_ids_cpu = load_seed_prompts(
        n_seq=args.n_seq, seed_len=args.seed_len, tokenizer=tokenizer, seed=args.seed
    )
    seed_ids = seed_ids_cpu.to(device)

    summary: dict = {
        "config": {
            "temps": list(args.temps),
            "n_seq": args.n_seq,
            "seed_len": args.seed_len,
            "length": args.length,
            "gen_len": gen_len,
            "warmup": args.warmup,
            "sigma_0": args.sigma0,
            "seed": args.seed,
        },
        "by_temp": {},
    }
    raw: dict = {}

    for T in args.temps:
        beta_true = 1.0 / T
        log_beta_true = math.log(beta_true)
        gen = torch.Generator(device="cpu").manual_seed(args.seed + int(round(T * 1000)))
        t0 = time.time()
        # Generation samples on-device; sample_under_beta uses a CPU generator via multinomial,
        # so move probs to CPU there is avoided — pass a device generator instead.
        dev_gen = torch.Generator(device=device).manual_seed(args.seed + int(round(T * 1000)))
        logits_arr, tokens = generate_at_beta(
            model=model, seed_ids=seed_ids, gen_len=gen_len,
            beta_dec=beta_true, device=device, generator=dev_gen,
        )

        # Filters (both); causal log β̂_t recorded pre-update.
        lb_faith, eta_faith = run_filter_trajectory(
            logits_arr=logits_arr, tokens=tokens, update_fn=laplace_update_faithful,
            sigma_0=args.sigma0, device=device,
        )
        lb_orig, eta_orig = run_filter_trajectory(
            logits_arr=logits_arr, tokens=tokens, update_fn=laplace_update,
            sigma_0=args.sigma0, device=device,
        )

        # Recovery: settled estimate = mean log β̂ over the LAST quarter of the
        # trajectory (past the convergence transient), per sequence. Also keep the
        # post-warmup median as a robust secondary.
        w = args.warmup
        late_start = int(0.75 * gen_len)
        settled_faith = lb_faith[:, late_start:].mean(axis=1)   # (B,)
        settled_orig = lb_orig[:, late_start:].mean(axis=1)
        med_faith = np.median(lb_faith[:, w:], axis=1)
        med_orig = np.median(lb_orig[:, w:], axis=1)

        # Prediction NLL (post-warmup positions): baselines + adaptive.
        tau_faith = torch.from_numpy(np.exp(lb_faith)).to(device)   # (B,U) causal temps
        tau_orig = torch.from_numpy(np.exp(lb_orig)).to(device)
        nll_b1 = per_token_nll(logits_arr=logits_arr, tokens=tokens, tau=1.0)[:, w:]
        nll_oracle = per_token_nll(logits_arr=logits_arr, tokens=tokens, tau=beta_true)[:, w:]
        nll_faith = per_token_nll(logits_arr=logits_arr, tokens=tokens, tau=tau_faith)[:, w:]
        nll_orig = per_token_nll(logits_arr=logits_arr, tokens=tokens, tau=tau_orig)[:, w:]

        # Per-sequence mean NLL (the resampling unit).
        seq_b1 = nll_b1.mean(axis=1)
        seq_oracle = nll_oracle.mean(axis=1)
        seq_faith = nll_faith.mean(axis=1)
        seq_orig = nll_orig.mean(axis=1)

        def paired_t(a, b):  # mean(a-b) and t-stat of (a-b) vs 0 over sequences
            d = a - b
            t = stats.ttest_rel(a, b)
            return float(d.mean()), float(t.statistic), float(t.pvalue)

        d_faith_b1 = paired_t(seq_faith, seq_b1)   # negative = faithful beats β=1
        d_orig_b1 = paired_t(seq_orig, seq_b1)
        d_faith_oracle = paired_t(seq_faith, seq_oracle)

        # Sanity: pooled MLE β must ≈ 1/T.
        mle_beta = pooled_mle_beta(logits_arr=logits_arr, tokens=tokens)

        # GIGO/coherence diagnostics.
        with torch.no_grad():
            gen_logq = F.log_softmax(beta_true * logits_arr[:, w:, :].float().reshape(-1, logits_arr.shape[-1]), dim=-1)
            gen_entropy = float((-(gen_logq.exp() * gen_logq).sum(-1)).mean())   # nats, mean H of generating dist
        toks_np = tokens.cpu().numpy()[:, w:]
        repeat_rate = float(np.mean(toks_np[:, 1:] == toks_np[:, :-1]))
        distinct_ratio = float(np.mean([len(np.unique(r)) / r.shape[0] for r in toks_np]))

        summary["by_temp"][f"{T}"] = {
            "T": T,
            "beta_true": beta_true,
            "log_beta_true": log_beta_true,
            "recovery": {
                # Settled estimate (last 25% of trajectory) — the headline.
                "faithful_settled_logbeta_mean": float(settled_faith.mean()),
                "faithful_settled_logbeta_se": float(settled_faith.std(ddof=1) / math.sqrt(len(settled_faith))),
                "faithful_settled_bias_vs_true": float(settled_faith.mean() - log_beta_true),
                "original_settled_logbeta_mean": float(settled_orig.mean()),
                "original_settled_bias_vs_true": float(settled_orig.mean() - log_beta_true),
                # Secondary: post-warmup median + final value.
                "faithful_median_logbeta_mean": float(med_faith.mean()),
                "original_median_logbeta_mean": float(med_orig.mean()),
                "final_eta_faithful_mean": float(eta_faith.mean()),
                "final_eta_original_mean": float(eta_orig.mean()),
            },
            "prediction_nll_nats": {
                "beta1_mean": float(seq_b1.mean()),
                "oracle_1overT_mean": float(seq_oracle.mean()),
                "adaptive_faithful_mean": float(seq_faith.mean()),
                "adaptive_original_mean": float(seq_orig.mean()),
                "faithful_minus_beta1": {"mean": d_faith_b1[0], "t": d_faith_b1[1], "p": d_faith_b1[2]},
                "original_minus_beta1": {"mean": d_orig_b1[0], "t": d_orig_b1[1], "p": d_orig_b1[2]},
                "faithful_minus_oracle": {"mean": d_faith_oracle[0], "t": d_faith_oracle[1], "p": d_faith_oracle[2]},
            },
            "sanity_pooled_mle_beta": mle_beta,
            "sanity_mle_vs_truth_logratio": float(math.log(mle_beta) - log_beta_true),
            "diagnostics": {
                "gen_dist_entropy_nats": gen_entropy,
                "repeat_rate": repeat_rate,
                "distinct_token_ratio": distinct_ratio,
            },
        }
        raw[f"lb_faith_T{T}"] = lb_faith
        raw[f"lb_orig_T{T}"] = lb_orig
        print(
            f"[recovery] T={T} beta_true={beta_true:.3f} (log={log_beta_true:+.3f}) | "
            f"faithful settled logβ̂={settled_faith.mean():+.3f} (bias {settled_faith.mean()-log_beta_true:+.3f}) | "
            f"original {settled_orig.mean():+.3f} (bias {settled_orig.mean()-log_beta_true:+.3f}) | "
            f"MLE β={mle_beta:.3f} | NLL β1={seq_b1.mean():.4f} faith={seq_faith.mean():.4f} "
            f"oracle={seq_oracle.mean():.4f} orig={seq_orig.mean():.4f} | rep={repeat_rate:.2f} | {time.time()-t0:.1f}s"
        )
        del logits_arr, tokens, tau_faith, tau_orig
        torch.cuda.empty_cache()

    return summary, raw


# ---------------------------------------------------------------------------
# Part 0 — calibration survey across text types
# ---------------------------------------------------------------------------

# Low-resource / constructed languages via wikimedia/wikipedia (config = dump.lang).
_WIKI_LANG = {"yoruba": "20231101.yo", "esperanto": "20231101.eo", "telugu": "20231101.te"}


def _stream_text_passages(
    *, hf_name, hf_config, field, n, min_tokens, tokenizer, char_per_tok=2, max_scan=400_000
) -> list[str]:
    """Stream `hf_name`, keep passages with >= min_tokens tokens until we have n.

    A char prefilter (len < min_tokens*char_per_tok) skips obvious stubs cheaply
    before tokenising. Fails LOUD if it cannot reach n (no silent skip)."""
    from datasets import load_dataset

    ds = load_dataset(hf_name, hf_config, streaming=True) if hf_config else load_dataset(hf_name, streaming=True)
    split = next(iter(ds.keys()))
    out: list[str] = []
    scanned = 0
    for ex in ds[split]:
        scanned += 1
        if scanned > max_scan:
            break
        t = ex.get(field)
        if not isinstance(t, str) or len(t) < min_tokens * char_per_tok:
            continue
        if len(tokenizer(t, add_special_tokens=False)["input_ids"]) >= min_tokens:
            out.append(t)
            if len(out) >= n:
                return out
    raise RuntimeError(
        f"{hf_name}[{hf_config}] field={field!r}: only {len(out)}/{n} passages "
        f">= {min_tokens} tokens after scanning {scanned} rows"
    )


def _recaman_sequence(n_terms: int) -> list[int]:
    """OEIS A005132 (Recamán): a(0)=0; a(n)=a(n-1)-n if positive & unseen, else a(n-1)+n.

    Deterministic but erratic (0,1,3,6,2,7,13,20,12,21,11,...) — a 'confidently wrong'
    test: a hidden rule the model can't infer locally, so it may be mis-temperatured."""
    seq, seen = [0], {0}
    for i in range(1, n_terms):
        prev = seq[-1]
        cand = prev - i
        nxt = cand if (cand > 0 and cand not in seen) else prev + i
        seq.append(nxt)
        seen.add(nxt)
    return seq


def load_survey_ids(*, key, n, length, tokenizer, seed) -> torch.Tensor:
    """Return (n, length) int64 token ids for survey dataset `key`."""
    from decoding_decoding.natural_text_filter import (
        _tokenize_to_length,
        load_wikitext_passages,
        load_writingprompts_passages,
    )

    if key == "recaman":
        # One long deterministic sequence, rendered as decimals, chunked into n×length.
        seq = _recaman_sequence(40_000)
        text = ", ".join(str(x) for x in seq)
        flat = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        need = n * length
        if flat.shape[0] < need:
            raise RuntimeError(f"recaman: {flat.shape[0]} tokens < needed {need}; raise n_terms")
        return flat[:need].reshape(n, length).contiguous()

    if key in ("wikitext", "shuffled_wikitext"):
        texts = load_wikitext_passages(n=n, min_tokens=length, tokenizer=tokenizer, seed=seed)
    elif key == "writingprompts":
        texts = load_writingprompts_passages(n=n, min_tokens=length, tokenizer=tokenizer, seed=seed)
    elif key in _WIKI_LANG:
        texts = _stream_text_passages(
            hf_name="wikimedia/wikipedia", hf_config=_WIKI_LANG[key], field="text",
            n=n, min_tokens=length, tokenizer=tokenizer,
        )
    elif key == "gutenberg":
        texts = _stream_text_passages(
            hf_name="sedthh/gutenberg_english", hf_config=None, field="TEXT",
            n=n, min_tokens=length, tokenizer=tokenizer,
        )
    elif key == "abc_music":
        texts = _stream_text_passages(
            hf_name="sander-wood/irishman", hf_config=None, field="abc notation",
            n=n, min_tokens=length, tokenizer=tokenizer, char_per_tok=1,
        )
    else:
        raise ValueError(f"unknown survey dataset key {key!r}")

    ids = _tokenize_to_length(texts, tokenizer, length=length)       # (n, length)
    if key == "shuffled_wikitext":
        # Permute tokens within each passage → destroys local structure (extreme
        # control: the model is maximally over-confident on broken n-grams).
        g = torch.Generator().manual_seed(seed + 777)
        for i in range(ids.shape[0]):
            ids[i] = ids[i][torch.randperm(ids.shape[1], generator=g)]
    return ids


def forward_window(*, model, ids, warmup, device, batch=16):
    """Forward `ids` (n, L); return (logits_win (n,U,V) fp16, targets (n,U)) for the
    filter window: logits at positions [warmup-1 : L-1] predict tokens [warmup : L]."""
    n, L = ids.shape
    U = L - warmup
    V = model.config.vocab_size
    logits_win = torch.empty((n, U, V), dtype=torch.float16, device=device)
    targets = torch.empty((n, U), dtype=torch.long, device=device)
    with torch.no_grad():
        for s in range(0, n, batch):
            b = ids[s : s + batch].to(device)
            out = model(input_ids=b, attention_mask=torch.ones_like(b), use_cache=False)
            logits_win[s : s + b.shape[0]] = out.logits[:, warmup - 1 : L - 1, :].to(torch.float16)
            targets[s : s + b.shape[0]] = b[:, warmup:L]
            del out
            torch.cuda.empty_cache()
    return logits_win, targets


def run_survey(args, model, tokenizer) -> dict:
    device = args.device
    L, w = args.length, args.warmup
    ntr, nte = args.n_train, args.n_test
    summary = {
        "config": {
            "datasets": list(args.datasets), "n_train": ntr, "n_test": nte,
            "length": L, "warmup": w, "seed": args.seed,
        },
        "by_dataset": {},
    }
    for key in args.datasets:
        t0 = time.time()
        ids = load_survey_ids(key=key, n=ntr + nte, length=L, tokenizer=tokenizer, seed=args.seed)
        logits_win, targets = forward_window(
            model=model, ids=ids, warmup=w, device=device, batch=args.batch
        )
        # Global Guo-MLE temperature fit on TRAIN; evaluate on TEST.
        beta_mle = pooled_mle_beta(logits_arr=logits_win[:ntr], tokens=targets[:ntr])
        nll_b1 = per_token_nll(logits_arr=logits_win[ntr:], tokens=targets[ntr:], tau=1.0)
        nll_mle = per_token_nll(logits_arr=logits_win[ntr:], tokens=targets[ntr:], tau=beta_mle)
        seq_b1, seq_mle = nll_b1.mean(axis=1), nll_mle.mean(axis=1)
        gain = float((seq_b1 - seq_mle).mean())            # +nats = temperature helps
        tt = stats.ttest_rel(seq_b1, seq_mle)
        # Mean entropy of the model's β=1 predictive distribution (difficulty proxy).
        with torch.no_grad():
            lw = logits_win[ntr:].float().reshape(-1, logits_win.shape[-1])
            ent = 0.0
            cn = 4096
            for i in range(0, lw.shape[0], cn):
                lq = F.log_softmax(lw[i : i + cn], dim=-1)
                ent += float((-(lq.exp() * lq).sum(-1)).sum())
            ent /= lw.shape[0]
        summary["by_dataset"][key] = {
            "n_train": ntr, "n_test": nte,
            "beta_mle": beta_mle,
            "optimal_T": 1.0 / beta_mle,
            "nll_beta1_mean": float(seq_b1.mean()),
            "nll_optT_mean": float(seq_mle.mean()),
            "global_T_gain_nats": gain,
            "gain_t": float(tt.statistic),
            "gain_p": float(tt.pvalue),
            "mean_beta1_entropy_nats": ent,
        }
        print(
            f"[survey] {key:18s} optT={1.0/beta_mle:5.2f} (β={beta_mle:.3f}) | "
            f"NLL β1={seq_b1.mean():7.3f} optT={seq_mle.mean():7.3f} | "
            f"gain={gain:+.4f} nats (t={tt.statistic:+.1f}, p={tt.pvalue:.1e}) | "
            f"H_β1={ent:.2f} | {time.time()-t0:.1f}s"
        )
        del logits_win, targets
        torch.cuda.empty_cache()

    # Rank by miscalibration (global-T gain).
    ranked = sorted(summary["by_dataset"].items(), key=lambda kv: -kv[1]["global_T_gain_nats"])
    summary["ranked_by_gain"] = [k for k, _ in ranked]
    return summary


# ---------------------------------------------------------------------------
# Part 2 — within-passage-CV heterogeneity gate (+ faithful adaptive)
# ---------------------------------------------------------------------------


def pertoken_oracle_nll(*, logits_arr, tokens, n_grid=41):
    """Min over a log-spaced τ-grid of per-token NLL (the absolute ceiling for ANY
    temperature scheme; in-sample per token). Returns (n, m) float64."""
    taus = np.exp(np.linspace(-1.5, 1.5, n_grid))
    best = None
    for tau in taus:
        nll = per_token_nll(logits_arr=logits_arr, tokens=tokens, tau=float(tau))
        best = nll if best is None else np.minimum(best, nll)
    return best


def run_gate(args, model, tokenizer) -> dict:
    """Within-passage cross-validated heterogeneity gate (the critic's leak-free design).

    Split each passage in half. Fit a per-passage temperature on the FIRST half, score
    the SECOND half — compare to a single global temperature (fit on the first halves,
    pooled). If per-passage ≈ global, there is no exploitable across-passage temperature
    heterogeneity. Also report the per-token oracle (ceiling) and the faithful filter's
    causal adaptive NLL on the second half. All gains are relative to the global T
    (positive = beats global)."""
    from decoding_decoding.counter_decode import laplace_update_faithful

    device, L, w = args.device, args.length, args.warmup
    summary = {"config": {"datasets": list(args.datasets), "n": args.n_gate,
                          "length": L, "warmup": w, "sigma_0": args.sigma0}, "by_dataset": {}}
    for key in args.datasets:
        t0 = time.time()
        ids = load_survey_ids(key=key, n=args.n_gate, length=L, tokenizer=tokenizer, seed=args.seed)
        Lw, Tw = forward_window(model=model, ids=ids, warmup=w, device=device, batch=args.batch)
        n, U = Tw.shape
        half = U // 2
        sec_l, sec_t = Lw[:, half:, :], Tw[:, half:]

        beta_global = pooled_mle_beta(logits_arr=Lw[:, :half, :], tokens=Tw[:, :half])
        nll_global = per_token_nll(logits_arr=sec_l, tokens=sec_t, tau=beta_global).mean(axis=1)

        nll_cv = np.empty(n)
        betas_pp = np.empty(n)
        for i in range(n):
            bi = pooled_mle_beta(logits_arr=Lw[i : i + 1, :half, :], tokens=Tw[i : i + 1, :half])
            betas_pp[i] = bi
            nll_cv[i] = per_token_nll(logits_arr=sec_l[i : i + 1], tokens=sec_t[i : i + 1], tau=bi).mean()

        nll_tok = pertoken_oracle_nll(logits_arr=sec_l, tokens=sec_t).mean(axis=1)

        lb_faith, _ = run_filter_trajectory(
            logits_arr=Lw, tokens=Tw, update_fn=laplace_update_faithful,
            sigma_0=args.sigma0, device=device,
        )
        tau_faith = torch.from_numpy(np.exp(lb_faith[:, half:])).to(device)
        nll_faith = per_token_nll(logits_arr=sec_l, tokens=sec_t, tau=tau_faith).mean(axis=1)

        def gain(better):  # mean(global - better) over passages; +mean = beats global
            tt = stats.ttest_rel(nll_global, better)
            return {"mean": float((nll_global - better).mean()), "t": float(tt.statistic), "p": float(tt.pvalue)}

        summary["by_dataset"][key] = {
            "n": n, "half_tokens": half, "beta_global": beta_global,
            "perpassage_beta_std": float(betas_pp.std()),
            "nll_global_mean": float(nll_global.mean()),
            "nll_perpassage_cv_mean": float(nll_cv.mean()),
            "nll_pertoken_oracle_mean": float(nll_tok.mean()),
            "nll_faithful_adaptive_mean": float(nll_faith.mean()),
            "perpassage_cv_gain_over_global": gain(nll_cv),
            "pertoken_oracle_gain_over_global": gain(nll_tok),
            "faithful_adaptive_gain_over_global": gain(nll_faith),
        }
        s = summary["by_dataset"][key]
        print(
            f"[gate] {key:18s} β_global={beta_global:.3f} β_pp std={betas_pp.std():.3f} | "
            f"gain vs global (nats): per-passage-CV={s['perpassage_cv_gain_over_global']['mean']:+.4f} "
            f"(t={s['perpassage_cv_gain_over_global']['t']:+.1f}) | "
            f"per-token-oracle={s['pertoken_oracle_gain_over_global']['mean']:+.4f} | "
            f"faithful-adaptive={s['faithful_adaptive_gain_over_global']['mean']:+.4f} "
            f"(t={s['faithful_adaptive_gain_over_global']['t']:+.1f}) | {time.time()-t0:.1f}s"
        )
        del Lw, Tw, tau_faith
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["recovery", "survey", "gate"], default="recovery")
    p.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.7, 1.0, 1.5, 2.0])
    p.add_argument("--n-seq", type=int, default=64)
    p.add_argument("--seed-len", type=int, default=24)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--sigma0", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=12345)
    # survey-only
    p.add_argument("--datasets", type=str, nargs="+", default=[
        "wikitext", "writingprompts", "gutenberg", "esperanto",
        "yoruba", "telugu", "abc_music", "recaman", "shuffled_wikitext",
    ])
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-test", type=int, default=80)
    p.add_argument("--n-gate", type=int, default=120)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--cuda-device", type=str, default="0")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_device)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    from decoding_decoding.natural_text_filter import load_model_and_tokenizer

    print(f"[self-cal] loading model on {args.device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) ...")
    model, tok = load_model_and_tokenizer(device=args.device)

    if args.part == "recovery":
        summary, raw = run_recovery(args, model, tok)
        (RESULTS_OUT / "recovery.json").write_text(json.dumps(summary, indent=2))
        np.savez(DATA_OUT / "recovery.npz", **raw, allow_pickle=True)
        print(f"[self-cal] wrote {RESULTS_OUT / 'recovery.json'}")
    elif args.part == "survey":
        summary = run_survey(args, model, tok)
        (RESULTS_OUT / "survey.json").write_text(json.dumps(summary, indent=2))
        print(f"[self-cal] wrote {RESULTS_OUT / 'survey.json'}")
        print(f"[self-cal] ranked by global-T gain: {summary['ranked_by_gain']}")
    elif args.part == "gate":
        summary = run_gate(args, model, tok)
        (RESULTS_OUT / "gate.json").write_text(json.dumps(summary, indent=2))
        print(f"[self-cal] wrote {RESULTS_OUT / 'gate.json'}")


if __name__ == "__main__":
    main()
