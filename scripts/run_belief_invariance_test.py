"""Belief-invariance test: a cleaner primary metric for (γ, α, p) selection.

Per critic feedback on the closed-loop sweep:

  Fix β_target = 1. For each prompt:
    1. Self-generate K=80 tokens at β_prefix ∈ {0.5, 1, 2}. This builds a
       prefix with a known imprint.
    2. Append a natural-text continuation (from WritingPrompts / WikiText)
       of length N=120 tokens after the prefix.
    3. Run the streaming filter through (prefix + continuation), letting
       β̂_t accumulate as the filter sees the prefix and then the natural
       continuation.
    4. At each position in the continuation, compute the corrected
       sampler's distribution `softmax((β_target / β̂_t^p) · ℓ_t)`
       and evaluate its log-prob on the natural next-token.
    5. Mean negative-log-prob over the continuation = CE per condition.

  A well-calibrated (γ, α, p) makes CE INVARIANT to β_prefix: the corrector
  successfully undoes the prefix's imprint. The primary metric is the
  variance of CE across β_prefix; lower variance = better.

This is a direct invariance test on natural-text ground truth — much
cleaner than self-generation closed-loop.

Output:
  data/counter_decode_belief_invariance/results.npz
  results/counter_decode/belief_invariance/summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from decoding_decoding.counter_decode import (
    LaplaceState, init_laplace, laplace_update,
)
from decoding_decoding.natural_text_filter import (
    DEFAULT_SIGMA_0,
    load_model_and_tokenizer,
    load_writingprompts_passages,
    load_wikitext_passages,
)
from decoding_decoding.prompts import PROMPTS


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = REPO_ROOT / "data" / "counter_decode_belief_invariance"
RESULTS_OUT = REPO_ROOT / "results" / "counter_decode" / "belief_invariance"


@dataclass
class Condition:
    gamma: float
    alpha: float
    p: float
    beta_prefix: float
    sigma_0: float
    beta_target: float = 1.0


def _self_generate_prefix(
    *,
    prompt_ids: torch.Tensor,        # (1, P)
    beta_prefix: float,
    n_tokens: int,
    model,
    device: str,
    seed: int,
) -> torch.Tensor:
    """Self-generate n_tokens at β_prefix from prompt_ids, return the new tokens (1, n)."""
    ids = prompt_ids.clone()
    attn = torch.ones_like(ids)
    gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=True)
    logits_t = out.logits[:, -1, :].float()
    past = out.past_key_values

    sampled = []
    for _ in range(n_tokens):
        decoder_log_q = F.log_softmax(beta_prefix * logits_t, dim=-1)
        decoder_probs = decoder_log_q.exp()
        x_next = torch.multinomial(decoder_probs, num_samples=1, generator=gen).squeeze(-1)
        sampled.append(int(x_next.item()))
        attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=attn.dtype)], dim=-1)
        with torch.no_grad():
            out = model(
                input_ids=x_next.unsqueeze(-1),
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
        logits_t = out.logits[:, -1, :].float()
        past = out.past_key_values
    return torch.tensor([sampled], dtype=torch.long, device=device)


def _eval_corrected_sampler_on_continuation(
    *,
    prompt_ids: torch.Tensor,        # (1, P)
    prefix_ids: torch.Tensor,        # (1, K)
    continuation_ids: torch.Tensor,  # (1, N)
    sigma_0: float,
    gamma: float,
    alpha: float,
    p: float,
    beta_target: float,
    model,
    device: str,
) -> dict:
    """Run the streaming filter through (prompt + prefix + continuation), and
    at each continuation position, evaluate the corrected sampler's log-prob
    on the actual natural next-token.
    """
    full_ids = torch.cat([prompt_ids, prefix_ids, continuation_ids], dim=-1)  # (1, P+K+N)
    P = prompt_ids.shape[1]
    K = prefix_ids.shape[1]
    N = continuation_ids.shape[1]
    L = full_ids.shape[1]

    attn = torch.ones_like(full_ids)
    with torch.no_grad():
        out = model(input_ids=full_ids, attention_mask=attn, use_cache=False)
    logits = out.logits[0].float()    # (L, V)

    state = init_laplace(1, sigma_0=sigma_0, device=device, dtype=torch.float32)
    prior_eta = -0.5 * sigma_0 ** 2
    prior_J = 1.0 / sigma_0 ** 2

    # Run the filter through ALL positions starting at position 0 (after the prompt).
    # The filter sees logits at position i predicting token at position i+1.
    # Start with no filtering before position P-1 (so the filter is fresh on prefix start).
    filter_start = P - 1     # logits position; predicts token at P
    log_beta_traj = []
    neg_log_p_sampler = np.zeros(N, dtype=np.float64)   # NLL of natural tokens
    naturals_at_continuation = continuation_ids[0].detach().cpu().numpy()

    for i in range(filter_start, L - 1):
        ℓ = logits[i:i + 1, :]
        x_next = full_ids[0, i + 1:i + 2]

        # If we're inside the continuation, evaluate the corrected sampler on the
        # NATURAL next-token (which equals x_next here because we're using full_ids).
        # The corrected sampler uses β̂_t computed BEFORE seeing x_next.
        beta_hat = float(torch.exp(state.eta_hat).item())
        if i >= P + K - 1:
            cont_idx = i - (P + K - 1)
            if 0 <= cont_idx < N:
                if p == 0.0:
                    decoder_logits = beta_target * ℓ
                else:
                    decoder_logits = (beta_target / (beta_hat ** p)) * ℓ
                log_q = F.log_softmax(decoder_logits, dim=-1)
                lp = log_q.gather(-1, x_next.unsqueeze(-1)).squeeze(-1).item()
                neg_log_p_sampler[cont_idx] = -lp

        # Update the filter with the natural-text token.
        state, _, _ = laplace_update(
            state, ℓ, x_next,
            evidence_weight=alpha,
            memory_decay=gamma,
            prior_eta=prior_eta,
            prior_J=prior_J,
        )
        log_beta_traj.append(float(state.eta_hat.detach().cpu().item()))

    return {
        "neg_log_p_per_token": neg_log_p_sampler,        # (N,)
        "log_beta_traj": np.array(log_beta_traj, dtype=np.float32),
        "naturals_at_continuation": naturals_at_continuation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=24)
    parser.add_argument("--prefix-len", type=int, default=80)
    parser.add_argument("--cont-len", type=int, default=120)
    parser.add_argument("--prompt-tokens", type=int, default=20)
    parser.add_argument("--sigma0", type=float, default=0.2)
    parser.add_argument("--corpus", choices=["wikitext", "writingprompts"],
                        default="writingprompts")
    parser.add_argument("--betas-prefix", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--gammas", type=str, default="0.85,0.95,1.0")
    parser.add_argument("--alphas", type=str, default="0.5,1.0")
    parser.add_argument("--ps", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    betas_prefix = [float(x) for x in args.betas_prefix.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    ps = [float(x) for x in args.ps.split(",")]

    print(f"[inv] grid: γ={gammas} α={alphas} p={ps} β_prefix={betas_prefix}")
    print(f"[inv] {args.n_prompts} prompts × prefix_len {args.prefix_len} + cont_len {args.cont_len}")

    print("[inv] loading model + corpus ...")
    model, tok = load_model_and_tokenizer()

    # Load natural-text continuations: each passage gives prompt + ground-truth continuation.
    # We slice [prompt_tokens : prompt_tokens+prefix_len+cont_len], so min_tokens
    # must include all three plus a safety buffer.
    min_tokens = args.prompt_tokens + args.prefix_len + args.cont_len + 50
    if args.corpus == "writingprompts":
        passages = load_writingprompts_passages(
            n=args.n_prompts, min_tokens=min_tokens, tokenizer=tok, seed=args.seed
        )
    else:
        passages = load_wikitext_passages(
            n=args.n_prompts, min_tokens=min_tokens, tokenizer=tok, seed=args.seed
        )

    # Tokenize each passage and pull out prompt + continuation segments.
    prompt_token_lists: list[list[int]] = []
    continuation_token_lists: list[list[int]] = []
    for txt in passages:
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        prompt_token_lists.append(ids[:args.prompt_tokens])
        continuation_token_lists.append(
            ids[args.prompt_tokens + args.prefix_len:
                args.prompt_tokens + args.prefix_len + args.cont_len]
        )
    # The natural continuation should start at position prompt_tokens+prefix_len of the
    # source text, but the actual prefix used is SELF-GENERATED — they don't connect
    # textually. The continuation is still natural text, but disconnected from the prefix.

    # Self-generate prefixes once per (prompt, β_prefix); reuse across (γ,α,p).
    print("[inv] self-generating prefixes ...")
    prefixes: dict[float, list[torch.Tensor]] = {bp: [] for bp in betas_prefix}
    for bp in betas_prefix:
        t0 = time.time()
        for pi, ptl in enumerate(prompt_token_lists):
            prompt_ids = torch.tensor([ptl], dtype=torch.long, device="cuda")
            pref = _self_generate_prefix(
                prompt_ids=prompt_ids,
                beta_prefix=bp,
                n_tokens=args.prefix_len,
                model=model,
                device="cuda",
                seed=args.seed + pi + int(bp * 100),
            )
            prefixes[bp].append(pref)
        print(f"  β_prefix={bp}: prefixes generated in {time.time() - t0:.1f}s")

    # For each (γ, α, p), evaluate CE on continuations under each β_prefix.
    print("[inv] evaluating corrected sampler ...")
    records: list[dict] = []
    n_configs = len(gammas) * len(alphas) * len(ps)
    for ci, (gamma, alpha, p) in enumerate(product(gammas, alphas, ps)):
        t0 = time.time()
        ce_by_prefix: dict[float, list[float]] = {}
        for bp in betas_prefix:
            per_prompt_ce: list[float] = []
            for pi, ptl in enumerate(prompt_token_lists):
                prompt_ids = torch.tensor([ptl], dtype=torch.long, device="cuda")
                pref = prefixes[bp][pi]
                cont = torch.tensor(
                    [continuation_token_lists[pi]], dtype=torch.long, device="cuda"
                )
                if cont.shape[1] < args.cont_len:
                    raise RuntimeError(
                        f"prompt {pi}: continuation has {cont.shape[1]} tokens, need {args.cont_len}"
                    )
                res = _eval_corrected_sampler_on_continuation(
                    prompt_ids=prompt_ids,
                    prefix_ids=pref,
                    continuation_ids=cont,
                    sigma_0=args.sigma0,
                    gamma=gamma, alpha=alpha, p=p,
                    beta_target=1.0,
                    model=model,
                    device="cuda",
                )
                per_prompt_ce.append(float(np.mean(res["neg_log_p_per_token"])))
            ce_by_prefix[bp] = per_prompt_ce
        ce_arr = np.array([[ce_by_prefix[bp][pi] for bp in betas_prefix]
                           for pi in range(args.n_prompts)])  # (n_prompts, n_betas)
        # Variance of CE across β_prefix, per prompt; then mean across prompts.
        var_per_prompt = ce_arr.var(axis=1, ddof=1)
        mean_var = float(np.mean(var_per_prompt))
        mean_ce_per_prefix = ce_arr.mean(axis=0).tolist()
        records.append({
            "gamma": gamma, "alpha": alpha, "p": p,
            "mean_var_across_prefix": mean_var,
            "mean_ce_per_prefix": dict(zip([str(b) for b in betas_prefix], mean_ce_per_prefix)),
            "n_prompts": args.n_prompts,
        })
        elapsed = time.time() - t0
        print(
            f"  [{ci + 1}/{n_configs}] γ={gamma}, α={alpha}, p={p}: "
            f"var across β_prefix = {mean_var:.4f}, "
            f"mean CE = {dict(zip([str(b) for b in betas_prefix], [f'{c:.3f}' for c in mean_ce_per_prefix]))}, "
            f"({elapsed:.1f}s)"
        )

    summary = {
        "config": {
            "n_prompts": args.n_prompts,
            "prefix_len": args.prefix_len,
            "cont_len": args.cont_len,
            "prompt_tokens": args.prompt_tokens,
            "sigma_0": args.sigma0,
            "corpus": args.corpus,
            "betas_prefix": betas_prefix,
            "gammas": gammas,
            "alphas": alphas,
            "ps": ps,
        },
        "records": records,
        "top_configs": sorted(records, key=lambda r: r["mean_var_across_prefix"])[:10],
    }
    (RESULTS_OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[inv] summary → {RESULTS_OUT / 'summary.json'}")

    # Print top 10.
    print("\n=== Top 10 (γ, α, p) by CE-invariance-across-β_prefix ===")
    print(f"{'γ':>6} {'α':>6} {'p':>6} | {'var':>9} | per-β_prefix CE")
    for r in summary["top_configs"]:
        ce_str = "  ".join(f"{k}:{v:.3f}" for k, v in r["mean_ce_per_prefix"].items())
        print(f"{r['gamma']:>6.2f} {r['alpha']:>6.2f} {r['p']:>6.2f} | "
              f"{r['mean_var_across_prefix']:>9.4f} | {ce_str}")


if __name__ == "__main__":
    main()
