"""Exp A: per-corpus calibration baseline — streaming β̂ vs batch MLE, test PPL.

For each corpus (WikiText-103, WritingPrompts):
  1. Load 2N passages tokenized to exactly L tokens. Passage-level split:
     first N → calibration, last N → test.
  2. On the calibration half:
       β̂_streaming = final eta_hat of the existing Laplace filter (one-pass).
       β̂_MLE      = scipy.optimize.minimize_scalar over per-token NLL
                    = −(1/T) Σ_t log softmax(β · ℓ_t)[x_{t+1}].
  3. On the test half: per-passage total NLL at four β settings:
       T = 1 (no calibration)
       β̂_streaming (own corpus)
       β̂_MLE (own corpus)
       β̂_MLE (the *other* corpus) — does a globally fit β hurt the wrong corpus?
  4. Per-passage Δlog-PPL = mean per-token NLL(β) − mean per-token NLL(1).
     Paired t-test against zero on the test split.

Output goes to:
  data/calibration_baseline/results.npz        (raw per-passage NLLs etc.)
  results/counter_decode/calibration_baseline/summary.json   (auditable numbers)

Notes
-----
* Per the critic, the headline of Exp A is the test-PPL drop on WritingPrompts,
  not the β̂_streaming ≈ β̂_MLE consistency (the streaming Newton is itself a
  one-pass approximation to the batch MAP, so they should agree closely by
  construction). The consistency tile is still reported as a sanity check.
* WikiText prediction is "no detectable test-PPL change", framed as a null —
  see decoding-modeling.tex framing.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize, stats

from decoding_decoding.counter_decode import init_laplace, laplace_update
from decoding_decoding.natural_text_filter import (
    DEFAULT_SIGMA_0,
    DEFAULT_WARMUP,
    load_model_and_tokenizer,
    load_wikitext_passages,
    load_writingprompts_passages,
    _tokenize_to_length,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = REPO_ROOT / "data" / "calibration_baseline"
RESULTS_OUT = REPO_ROOT / "results" / "counter_decode" / "calibration_baseline"


@dataclass(frozen=True)
class CalibrationFit:
    """Calibration-half outputs: per-passage + pooled streaming β̂, MLE β̂."""

    corpus: str
    n_passages_calibration: int
    # Per-passage filter (state RESETS each passage); operational quantity.
    log_beta_streaming_median: float       # median final η̂ across passages
    log_beta_streaming_per_passage: np.ndarray   # (n_calibration,)
    # Pooled filter (state PERSISTS across all calibration tokens); apples-to-apples vs MLE.
    log_beta_streaming_pooled: float       # single-trajectory final η̂
    # Pooled MLE via scipy.
    log_beta_mle: float                    # single scalar over all calibration tokens
    nll_at_mle: float                      # per-token NLL at MLE β
    nll_at_one: float                      # per-token NLL at β = 1
    total_filter_steps: int                # n_passages * (length - warmup)


@dataclass(frozen=True)
class TestEval:
    """Test-half per-passage NLLs at each β candidate."""

    corpus: str
    n_passages_test: int
    beta_labels: list[str]                 # ordered list of β-setting names
    beta_values: list[float]               # ordered list of β values used
    per_passage_total_nll: np.ndarray      # (n_betas, n_test) total NLL per passage
    per_passage_n_tokens: np.ndarray       # (n_test,) num filter tokens per passage


# ---------------------------------------------------------------------------
# Calibration half: filter + MLE
# ---------------------------------------------------------------------------


def _run_calibration_half(
    *,
    corpus: str,
    texts: list[str],
    model,
    tokenizer,
    length: int,
    warmup: int,
    sigma_0: float,
    batch_size: int,
    device: str,
) -> CalibrationFit:
    """Run the streaming filter and accumulate (ℓ_t, x_{t+1}) for MLE fit.

    Memory: holds the full calibration logits on GPU as float16 to fit a
    Q-RTX-8000 (46 GB). For 64×236 positions × 151936 vocab × 2 bytes that's
    ≈ 4.6 GB.
    """
    n_pass = len(texts)
    U = length - warmup
    log_beta_streaming = np.zeros(n_pass, dtype=np.float64)

    # Pre-allocate logits + targets buffers on GPU.
    V = model.config.vocab_size
    logits_all = torch.empty(
        (n_pass, U, V), dtype=torch.float16, device=device
    )
    targets_all = torch.empty((n_pass, U), dtype=torch.long, device=device)

    for chunk_start in range(0, n_pass, batch_size):
        chunk_texts = texts[chunk_start:chunk_start + batch_size]
        ids = _tokenize_to_length(chunk_texts, tokenizer, length=length).to(device)
        B = ids.shape[0]
        attn = torch.ones_like(ids)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        all_logits = out.logits.float()       # (B, length, V)

        # Stream the Laplace filter.
        state = init_laplace(B, sigma_0=sigma_0, device=device, dtype=torch.float32)
        for k in range(U):
            seq_pos = warmup - 1 + k
            target_pos = warmup + k
            logits_t = all_logits[:, seq_pos, :]
            x_next = ids[:, target_pos]
            state, _, _ = laplace_update(state, logits_t, x_next)

            # Stash for MLE fit.
            logits_all[chunk_start:chunk_start + B, k, :] = logits_t.to(torch.float16)
            targets_all[chunk_start:chunk_start + B, k] = x_next

        log_beta_streaming[chunk_start:chunk_start + B] = (
            state.eta_hat.detach().cpu().numpy()
        )

        del all_logits, out
        torch.cuda.empty_cache()

    # MLE: minimize per-token NLL = -mean log softmax(β·ℓ)[x] over (n_pass*U) tokens.
    # Chunk along the token axis to keep peak GPU memory bounded (each fp32
    # logit chunk is chunk*V*4 bytes; with chunk=512 V=152K that's ~310 MB).
    flat_logits = logits_all.view(-1, V)                     # (n_pass*U, V), fp16
    flat_targets = targets_all.view(-1)                      # (n_pass*U,)

    def per_token_nll(beta: float) -> float:
        total_nll = 0.0
        n_total = 0
        chunk = 512
        with torch.no_grad():
            for i in range(0, flat_logits.shape[0], chunk):
                chunk_logits = flat_logits[i:i + chunk].float()
                chunk_targets = flat_targets[i:i + chunk]
                log_q = F.log_softmax(beta * chunk_logits, dim=-1)
                lp = log_q.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
                total_nll += float(-lp.sum().item())
                n_total += chunk_logits.shape[0]
        return total_nll / n_total

    # Per-token NLL is convex in log β. minimize_scalar with brent over a wide bracket.
    res = optimize.minimize_scalar(
        per_token_nll, bounds=(0.05, 10.0), method="bounded",
        options={"xatol": 1e-4},
    )
    beta_mle = float(res.x)
    nll_at_mle = float(res.fun)
    nll_at_one = per_token_nll(1.0)

    log_beta_streaming_median = float(np.median(log_beta_streaming))

    # ----- Pooled streaming filter: one trajectory across ALL calibration tokens.
    # State is initialized once and never reset. Per-token Newton step.
    pooled_state = init_laplace(1, sigma_0=sigma_0, device=device, dtype=torch.float32)
    pool_chunk = 256
    with torch.no_grad():
        for i in range(0, flat_logits.shape[0], pool_chunk):
            chunk_logits = flat_logits[i:i + pool_chunk].float()    # (chunk, V)
            chunk_targets = flat_targets[i:i + pool_chunk]
            for j in range(chunk_logits.shape[0]):
                pooled_state, _, _ = laplace_update(
                    pooled_state,
                    chunk_logits[j:j+1, :],
                    chunk_targets[j:j+1],
                )
    log_beta_streaming_pooled = float(pooled_state.eta_hat.detach().cpu().numpy()[0])

    # Free the giant GPU stash before test eval.
    del logits_all, targets_all, flat_logits, flat_targets
    torch.cuda.empty_cache()

    return CalibrationFit(
        corpus=corpus,
        n_passages_calibration=n_pass,
        log_beta_streaming_median=log_beta_streaming_median,
        log_beta_streaming_per_passage=log_beta_streaming,
        log_beta_streaming_pooled=log_beta_streaming_pooled,
        log_beta_mle=float(np.log(beta_mle)),
        nll_at_mle=nll_at_mle,
        nll_at_one=nll_at_one,
        total_filter_steps=n_pass * U,
    )


# ---------------------------------------------------------------------------
# Test half: per-passage NLL at each β
# ---------------------------------------------------------------------------


def _run_test_half(
    *,
    corpus: str,
    texts: list[str],
    model,
    tokenizer,
    length: int,
    warmup: int,
    beta_settings: list[tuple[str, float]],
    batch_size: int,
    device: str,
) -> TestEval:
    """Forward-pass each test passage and evaluate NLL at each β candidate.

    Streaming over batches; no giant stash.
    """
    n_pass = len(texts)
    U = length - warmup
    n_betas = len(beta_settings)

    per_passage_total_nll = np.zeros((n_betas, n_pass), dtype=np.float64)
    per_passage_n_tokens = np.full((n_pass,), U, dtype=np.int64)

    beta_tensors = torch.tensor(
        [b for _, b in beta_settings], device=device, dtype=torch.float32
    )    # (n_betas,)

    for chunk_start in range(0, n_pass, batch_size):
        chunk_texts = texts[chunk_start:chunk_start + batch_size]
        ids = _tokenize_to_length(chunk_texts, tokenizer, length=length).to(device)
        B = ids.shape[0]
        attn = torch.ones_like(ids)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        all_logits = out.logits.float()              # (B, length, V)
        # Slice to the filter window.
        logits_win = all_logits[:, warmup - 1:length - 1, :]   # (B, U, V)
        targets = ids[:, warmup:length]                        # (B, U)

        for i, (_label, beta_val) in enumerate(beta_settings):
            with torch.no_grad():
                scaled = beta_val * logits_win                  # (B, U, V)
                log_q = F.log_softmax(scaled, dim=-1)
                lp = log_q.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, U)
                # Total NLL per passage (sum over filter positions).
                per_passage_nll = (-lp).sum(dim=-1).detach().cpu().numpy()  # (B,)
                per_passage_total_nll[i, chunk_start:chunk_start + B] = per_passage_nll

        del all_logits, logits_win, targets, out
        torch.cuda.empty_cache()

    return TestEval(
        corpus=corpus,
        n_passages_test=n_pass,
        beta_labels=[label for label, _ in beta_settings],
        beta_values=[float(b) for _, b in beta_settings],
        per_passage_total_nll=per_passage_total_nll,
        per_passage_n_tokens=per_passage_n_tokens,
    )


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run_one_corpus(
    *,
    corpus: str,
    texts: list[str],
    n_calibration: int,
    n_test: int,
    model,
    tokenizer,
    length: int,
    warmup: int,
    sigma_0: float,
    batch_size: int,
    device: str,
) -> tuple[CalibrationFit, list[str], list[str]]:
    """Calibration-half run; returns fit + the two text splits.

    We do the test-half pass later (after both corpora's calibrations are
    done) so we can include cross-corpus β̂.
    """
    assert len(texts) >= n_calibration + n_test, (
        f"need ≥{n_calibration + n_test} passages; got {len(texts)}"
    )
    calibration_texts = texts[:n_calibration]
    test_texts = texts[n_calibration:n_calibration + n_test]

    t0 = time.time()
    fit = _run_calibration_half(
        corpus=corpus,
        texts=calibration_texts,
        model=model,
        tokenizer=tokenizer,
        length=length,
        warmup=warmup,
        sigma_0=sigma_0,
        batch_size=batch_size,
        device=device,
    )
    print(
        f"[calib:{corpus}] "
        f"streaming median log β̂ = {fit.log_beta_streaming_median:+.4f}  "
        f"streaming pooled log β̂ = {fit.log_beta_streaming_pooled:+.4f}  "
        f"MLE log β̂ = {fit.log_beta_mle:+.4f}  "
        f"(took {time.time() - t0:.1f}s)"
    )
    return fit, calibration_texts, test_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calibration", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA_0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[calib] loading model ...")
    model, tok = load_model_and_tokenizer(device=args.device)

    n_needed = args.n_calibration + args.n_test
    print(f"[calib] loading {n_needed} WikiText passages ...")
    wt_texts = load_wikitext_passages(
        n=n_needed, min_tokens=args.length, tokenizer=tok, seed=args.seed
    )
    print(f"[calib] loading {n_needed} WritingPrompts passages ...")
    wp_texts = load_writingprompts_passages(
        n=n_needed, min_tokens=args.length, tokenizer=tok, seed=args.seed
    )

    fit_wt, cal_wt, test_wt = run_one_corpus(
        corpus="wikitext", texts=wt_texts,
        n_calibration=args.n_calibration, n_test=args.n_test,
        model=model, tokenizer=tok,
        length=args.length, warmup=args.warmup, sigma_0=args.sigma0,
        batch_size=args.batch_size, device=args.device,
    )
    fit_wp, cal_wp, test_wp = run_one_corpus(
        corpus="writingprompts", texts=wp_texts,
        n_calibration=args.n_calibration, n_test=args.n_test,
        model=model, tokenizer=tok,
        length=args.length, warmup=args.warmup, sigma_0=args.sigma0,
        batch_size=args.batch_size, device=args.device,
    )

    # Beta settings shared across both corpora's test splits.
    beta_wt_stream_med = float(np.exp(fit_wt.log_beta_streaming_median))
    beta_wt_stream_pooled = float(np.exp(fit_wt.log_beta_streaming_pooled))
    beta_wt_mle = float(np.exp(fit_wt.log_beta_mle))
    beta_wp_stream_med = float(np.exp(fit_wp.log_beta_streaming_median))
    beta_wp_stream_pooled = float(np.exp(fit_wp.log_beta_streaming_pooled))
    beta_wp_mle = float(np.exp(fit_wp.log_beta_mle))

    print(
        f"[calib] β̂ table:\n"
        f"  WT  stream_med={beta_wt_stream_med:.4f}  stream_pooled={beta_wt_stream_pooled:.4f}  MLE={beta_wt_mle:.4f}\n"
        f"  WP  stream_med={beta_wp_stream_med:.4f}  stream_pooled={beta_wp_stream_pooled:.4f}  MLE={beta_wp_mle:.4f}"
    )

    print("[calib:test] WikiText test ...")
    t0 = time.time()
    test_eval_wt = _run_test_half(
        corpus="wikitext",
        texts=test_wt,
        model=model, tokenizer=tok,
        length=args.length, warmup=args.warmup,
        beta_settings=[
            ("T1", 1.0),
            ("WT_stream_med", beta_wt_stream_med),
            ("WT_stream_pooled", beta_wt_stream_pooled),
            ("WT_MLE", beta_wt_mle),
            ("WP_MLE", beta_wp_mle),
        ],
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("[calib:test] WritingPrompts test ...")
    t0 = time.time()
    test_eval_wp = _run_test_half(
        corpus="writingprompts",
        texts=test_wp,
        model=model, tokenizer=tok,
        length=args.length, warmup=args.warmup,
        beta_settings=[
            ("T1", 1.0),
            ("WP_stream_med", beta_wp_stream_med),
            ("WP_stream_pooled", beta_wp_stream_pooled),
            ("WP_MLE", beta_wp_mle),
            ("WT_MLE", beta_wt_mle),
        ],
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Save raw arrays ---
    np.savez(
        DATA_OUT / "results.npz",
        # WikiText
        wt_log_beta_streaming_per_passage=fit_wt.log_beta_streaming_per_passage,
        wt_log_beta_streaming_median=fit_wt.log_beta_streaming_median,
        wt_log_beta_streaming_pooled=fit_wt.log_beta_streaming_pooled,
        wt_log_beta_mle=fit_wt.log_beta_mle,
        wt_nll_at_mle=fit_wt.nll_at_mle,
        wt_nll_at_one_calibration=fit_wt.nll_at_one,
        wt_total_filter_steps=fit_wt.total_filter_steps,
        wt_test_beta_labels=np.array(test_eval_wt.beta_labels, dtype=object),
        wt_test_beta_values=np.array(test_eval_wt.beta_values, dtype=np.float64),
        wt_test_per_passage_total_nll=test_eval_wt.per_passage_total_nll,
        wt_test_per_passage_n_tokens=test_eval_wt.per_passage_n_tokens,
        # WritingPrompts
        wp_log_beta_streaming_per_passage=fit_wp.log_beta_streaming_per_passage,
        wp_log_beta_streaming_median=fit_wp.log_beta_streaming_median,
        wp_log_beta_streaming_pooled=fit_wp.log_beta_streaming_pooled,
        wp_log_beta_mle=fit_wp.log_beta_mle,
        wp_nll_at_mle=fit_wp.nll_at_mle,
        wp_nll_at_one_calibration=fit_wp.nll_at_one,
        wp_total_filter_steps=fit_wp.total_filter_steps,
        wp_test_beta_labels=np.array(test_eval_wp.beta_labels, dtype=object),
        wp_test_beta_values=np.array(test_eval_wp.beta_values, dtype=np.float64),
        wp_test_per_passage_total_nll=test_eval_wp.per_passage_total_nll,
        wp_test_per_passage_n_tokens=test_eval_wp.per_passage_n_tokens,
        # Meta
        length=args.length,
        warmup=args.warmup,
        sigma_0=args.sigma0,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        allow_pickle=True,
    )
    print(f"[calib] raw arrays → {DATA_OUT / 'results.npz'}")

    # --- Summarize ---
    summary = _summarize(fit_wt, fit_wp, test_eval_wt, test_eval_wp, args)
    (RESULTS_OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[calib] summary → {RESULTS_OUT / 'summary.json'}")


def _summarize(
    fit_wt: CalibrationFit,
    fit_wp: CalibrationFit,
    test_wt: TestEval,
    test_wp: TestEval,
    args,
) -> dict:
    out: dict = {
        "config": {
            "length": args.length,
            "warmup": args.warmup,
            "sigma_0": args.sigma0,
            "n_calibration": args.n_calibration,
            "n_test": args.n_test,
        },
        "calibration": {
            "wikitext": {
                "log_beta_streaming_median": fit_wt.log_beta_streaming_median,
                "log_beta_streaming_pooled": fit_wt.log_beta_streaming_pooled,
                "log_beta_mle": fit_wt.log_beta_mle,
                "nll_at_one_calibration": fit_wt.nll_at_one,
                "nll_at_mle_calibration": fit_wt.nll_at_mle,
                "consistency_gap_pooled_vs_mle": (
                    fit_wt.log_beta_streaming_pooled - fit_wt.log_beta_mle
                ),
                "aggregation_gap_median_vs_pooled": (
                    fit_wt.log_beta_streaming_median - fit_wt.log_beta_streaming_pooled
                ),
            },
            "writingprompts": {
                "log_beta_streaming_median": fit_wp.log_beta_streaming_median,
                "log_beta_streaming_pooled": fit_wp.log_beta_streaming_pooled,
                "log_beta_mle": fit_wp.log_beta_mle,
                "nll_at_one_calibration": fit_wp.nll_at_one,
                "nll_at_mle_calibration": fit_wp.nll_at_mle,
                "consistency_gap_pooled_vs_mle": (
                    fit_wp.log_beta_streaming_pooled - fit_wp.log_beta_mle
                ),
                "aggregation_gap_median_vs_pooled": (
                    fit_wp.log_beta_streaming_median - fit_wp.log_beta_streaming_pooled
                ),
            },
        },
        "test": {},
    }

    for name, ev in (("wikitext", test_wt), ("writingprompts", test_wp)):
        per_token_nll = ev.per_passage_total_nll / ev.per_passage_n_tokens[None, :]
        per_token_ppl = np.exp(per_token_nll)
        # Δlog-PPL per passage = log PPL(β) − log PPL(T=1) = per_token_nll(β) − per_token_nll(1)
        nll_t1 = per_token_nll[0]    # by construction first row is T=1
        deltas = per_token_nll - nll_t1[None, :]    # (n_betas, n_test)

        out["test"][name] = {
            "beta_labels": ev.beta_labels,
            "beta_values": ev.beta_values,
            "per_token_nll_mean": [float(v) for v in per_token_nll.mean(axis=1)],
            "per_token_ppl_mean": [float(v) for v in per_token_ppl.mean(axis=1)],
            "delta_log_ppl_mean": [float(v) for v in deltas.mean(axis=1)],
            "delta_log_ppl_se": [
                float(v) for v in (deltas.std(axis=1, ddof=1) / np.sqrt(deltas.shape[1]))
            ],
        }
        # Paired t-test of each β vs T=1.
        t_stats = []
        p_vals = []
        for i in range(deltas.shape[0]):
            if i == 0:
                t_stats.append(0.0)
                p_vals.append(1.0)
                continue
            t, p = stats.ttest_1samp(deltas[i], 0.0)
            t_stats.append(float(t))
            p_vals.append(float(p))
        out["test"][name]["paired_t_vs_T1"] = t_stats
        out["test"][name]["paired_p_vs_T1"] = p_vals

    return out


if __name__ == "__main__":
    main()
