"""Exp C: does the model's distribution respond to its own decoder?

Generate the same N prompts under three β_dec settings:
  β_dec = 0.5  (T = 2.0, "flat decoder", high-entropy text expected)
  β_dec = 1.0  (T = 1.0, no temperature)
  β_dec = 2.0  (T = 0.5, "sharp decoder", low-entropy text expected)

At each step, record the model's pre-decoding entropy H(softmax(ℓ_t)).
Hypothesis: at late steps (after the model has conditioned on enough of its
own generation), entropy under β_dec=2 should be lower than under β_dec=0.5
— the model has formed a belief that it is operating under a sharp decoder
and sharpens its pre-decoding distribution to match.

This is a direct test of the project's "model belief about decoder" framing
(see decoding-modeling.tex), distinct from the calibration question of
Exp A.

Outputs:
  data/entropy_probe/results.npz
  results/counter_decode/entropy_probe/summary.json
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
from scipy import stats

from decoding_decoding.natural_text_filter import load_model_and_tokenizer
from decoding_decoding.prompts import PROMPTS


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = REPO_ROOT / "data" / "entropy_probe"
RESULTS_OUT = REPO_ROOT / "results" / "counter_decode" / "entropy_probe"


@dataclass(frozen=True)
class GenResult:
    """Per-step model entropy + sampled tokens under one β_dec setting."""

    beta_dec: float
    entropies: np.ndarray            # (n_prompts, n_steps) pre-decoding H(softmax(ℓ_t))
    sampled_token_ids: np.ndarray    # (n_prompts, n_steps)
    target_logprobs: np.ndarray      # (n_prompts, n_steps) log softmax(ℓ_t)[x_t] (β=1)


def _generate_under_beta(
    *,
    prompts: list[str],
    beta_dec: float,
    model,
    tokenizer,
    n_steps: int,
    warmup_tokens: int,
    device: str,
    seed: int,
) -> GenResult:
    """Generate ``n_steps`` tokens per prompt at fixed β_dec, recording entropy.

    Same algorithm as filter_pass_self_gen but with arbitrary β_dec on the
    sampler and no filter state. Pre-decoding logits ℓ_t are what we record
    entropy of (NOT softmax(β_dec·ℓ_t)).
    """
    # Tokenize prompts to a fixed prompt length (truncate or pad-right).
    enc = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    # Trim each to a minimum length so all prompts have ≥ warmup_tokens tokens.
    if any(len(t) < warmup_tokens for t in enc):
        raise ValueError(
            f"prompt has fewer than {warmup_tokens} tokens after tokenization"
        )
    enc = [t[:warmup_tokens] for t in enc]
    ids = torch.tensor(enc, dtype=torch.long, device=device)   # (B, warmup_tokens)
    B = ids.shape[0]
    attn = torch.ones_like(ids)

    gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=True)
    logits_t = out.logits[:, -1, :].float()
    past = out.past_key_values

    entropies = np.zeros((B, n_steps), dtype=np.float32)
    sampled = np.zeros((B, n_steps), dtype=np.int64)
    target_lp = np.zeros((B, n_steps), dtype=np.float32)

    for k in range(n_steps):
        log_q = F.log_softmax(logits_t, dim=-1)
        q = log_q.exp()
        H = -(q * log_q).sum(dim=-1)
        entropies[:, k] = H.detach().cpu().numpy()

        # Sample x_t ~ softmax(β_dec · ℓ_t). Decoder applied here, NOT to ℓ_t
        # used for entropy recording.
        decoder_logits = beta_dec * logits_t
        decoder_log_q = F.log_softmax(decoder_logits, dim=-1)
        decoder_probs = decoder_log_q.exp()
        x_next = torch.multinomial(decoder_probs, num_samples=1, generator=gen).squeeze(-1)
        sampled[:, k] = x_next.detach().cpu().numpy()
        target_lp[:, k] = (
            log_q.gather(-1, x_next.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
        )

        if k < n_steps - 1:
            attn = torch.cat(
                [attn, torch.ones(B, 1, device=device, dtype=attn.dtype)], dim=-1
            )
            with torch.no_grad():
                out = model(
                    input_ids=x_next.unsqueeze(-1),
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
            logits_t = out.logits[:, -1, :].float()
            past = out.past_key_values

    return GenResult(
        beta_dec=float(beta_dec),
        entropies=entropies,
        sampled_token_ids=sampled,
        target_logprobs=target_lp,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--warmup-tokens", type=int, default=12,
                        help="number of leading tokens taken from each prompt")
    parser.add_argument("--beta-dec", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--late-window-start", type=int, default=80,
                        help="aggregate entropy over steps in [late_window_start, n_steps)")
    args = parser.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[entropy-probe] loading model ...")
    model, tok = load_model_and_tokenizer(device=args.device)

    prompts = list(PROMPTS)
    print(f"[entropy-probe] N_prompts={len(prompts)}, n_steps={args.n_steps}, "
          f"warmup_tokens={args.warmup_tokens}, β_dec={args.beta_dec}")

    results: dict[float, GenResult] = {}
    for i, beta_dec in enumerate(args.beta_dec):
        t0 = time.time()
        print(f"[entropy-probe] generating under β_dec={beta_dec} ...")
        results[beta_dec] = _generate_under_beta(
            prompts=prompts,
            beta_dec=beta_dec,
            model=model,
            tokenizer=tok,
            n_steps=args.n_steps,
            warmup_tokens=args.warmup_tokens,
            device=args.device,
            seed=args.seed + i,
        )
        print(f"  done in {time.time() - t0:.1f}s")

    # Save raw arrays.
    save_payload = {
        "beta_dec_values": np.array(args.beta_dec, dtype=np.float32),
        "warmup_tokens": args.warmup_tokens,
        "n_steps": args.n_steps,
        "late_window_start": args.late_window_start,
    }
    for beta_dec in args.beta_dec:
        r = results[beta_dec]
        key = f"beta_{beta_dec:.2f}".replace(".", "_")
        save_payload[f"{key}_entropies"] = r.entropies
        save_payload[f"{key}_sampled_token_ids"] = r.sampled_token_ids
        save_payload[f"{key}_target_logprobs"] = r.target_logprobs
    np.savez_compressed(DATA_OUT / "results.npz", **save_payload, allow_pickle=True)
    print(f"[entropy-probe] arrays → {DATA_OUT / 'results.npz'}")

    # Summarize.
    summary: dict = {
        "config": {
            "n_prompts": len(prompts),
            "n_steps": args.n_steps,
            "warmup_tokens": args.warmup_tokens,
            "beta_dec_values": args.beta_dec,
            "late_window_start": args.late_window_start,
            "model": "Qwen/Qwen2.5-3B",
        },
        "by_beta_dec": {},
        "paired_diffs": {},
    }

    # Per-β late-window stats.
    for beta_dec in args.beta_dec:
        r = results[beta_dec]
        late = r.entropies[:, args.late_window_start:]   # (n_prompts, late_window)
        per_prompt_mean = late.mean(axis=1)               # (n_prompts,)
        summary["by_beta_dec"][f"{beta_dec}"] = {
            "late_entropy_mean_over_prompts": float(per_prompt_mean.mean()),
            "late_entropy_median_over_prompts": float(np.median(per_prompt_mean)),
            "late_entropy_se_over_prompts": float(per_prompt_mean.std(ddof=1)
                                                  / np.sqrt(per_prompt_mean.shape[0])),
        }

    # Paired difference: H(β_dec=low) - H(β_dec=high) per prompt.
    # Expectation (if model forms a belief about its decoder):
    #   H(β_dec=0.5) > H(β_dec=1.0) > H(β_dec=2.0)
    #   so (β=0.5 minus β=2.0) > 0 per prompt.
    if 0.5 in args.beta_dec and 2.0 in args.beta_dec:
        late_low = results[0.5].entropies[:, args.late_window_start:].mean(axis=1)
        late_high = results[2.0].entropies[:, args.late_window_start:].mean(axis=1)
        diff = late_low - late_high          # (n_prompts,)
        t, p = stats.ttest_1samp(diff, 0.0)
        summary["paired_diffs"]["H_betadec_0.5_minus_2.0"] = {
            "n_prompts": int(diff.shape[0]),
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "se": float(diff.std(ddof=1) / np.sqrt(diff.shape[0])),
            "t_stat": float(t),
            "p_value": float(p),
            "n_positive": int((diff > 0).sum()),
        }
    if 0.5 in args.beta_dec and 1.0 in args.beta_dec:
        late_low = results[0.5].entropies[:, args.late_window_start:].mean(axis=1)
        late_one = results[1.0].entropies[:, args.late_window_start:].mean(axis=1)
        diff = late_low - late_one
        t, p = stats.ttest_1samp(diff, 0.0)
        summary["paired_diffs"]["H_betadec_0.5_minus_1.0"] = {
            "n_prompts": int(diff.shape[0]),
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "se": float(diff.std(ddof=1) / np.sqrt(diff.shape[0])),
            "t_stat": float(t),
            "p_value": float(p),
            "n_positive": int((diff > 0).sum()),
        }
    if 1.0 in args.beta_dec and 2.0 in args.beta_dec:
        late_one = results[1.0].entropies[:, args.late_window_start:].mean(axis=1)
        late_high = results[2.0].entropies[:, args.late_window_start:].mean(axis=1)
        diff = late_one - late_high
        t, p = stats.ttest_1samp(diff, 0.0)
        summary["paired_diffs"]["H_betadec_1.0_minus_2.0"] = {
            "n_prompts": int(diff.shape[0]),
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "se": float(diff.std(ddof=1) / np.sqrt(diff.shape[0])),
            "t_stat": float(t),
            "p_value": float(p),
            "n_positive": int((diff > 0).sum()),
        }

    (RESULTS_OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[entropy-probe] summary → {RESULTS_OUT / 'summary.json'}")

    # Print headline result.
    print("\n=== Entropy probe summary ===")
    for b in args.beta_dec:
        s = summary["by_beta_dec"][f"{b}"]
        print(
            f"  β_dec={b}: late-window H mean over prompts = {s['late_entropy_mean_over_prompts']:.3f} ± {s['late_entropy_se_over_prompts']:.3f}"
        )
    if "H_betadec_0.5_minus_2.0" in summary["paired_diffs"]:
        s = summary["paired_diffs"]["H_betadec_0.5_minus_2.0"]
        print(
            f"  ΔH(β=0.5 − β=2.0): mean = {s['mean']:+.3f} (SE {s['se']:.3f}), "
            f"t = {s['t_stat']:+.2f}, p = {s['p_value']:.3g}, "
            f"n positive = {s['n_positive']}/{s['n_prompts']}"
        )


if __name__ == "__main__":
    main()
