"""Exp C control: does the model's entropy track the sampler β, or just content?

Pre-warm each prompt with 80 steps of β=0.5 sampling (builds a "garbage"
prefix per Exp C). Then SWITCH the sampler β at step 80 to one of
{0.5 [no switch], 1.0, 2.0} and continue for another 70 steps. Record
pre-decoding entropy throughout.

Hypothesis under the project framing ("model believes its decoder β"):
  After the switch, model entropy at steps 81+ should drop SHARPLY toward
  the post-switch β's late-window value (from Exp C: 2.17 nats at β=1.0,
  1.17 nats at β=2.0).
Hypothesis under the content/garbage-in-garbage-out account:
  Model entropy after the switch should stay elevated because the prefix
  is unparseable; the new tokens only gradually replace the bad context.

This experiment is decisive between the two — it varies the sampler β
while holding the early prefix fixed (modulo the post-switch sampler's
own evolution).
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

from decoding_decoding.natural_text_filter import load_model_and_tokenizer
from decoding_decoding.prompts import PROMPTS


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = REPO_ROOT / "data" / "entropy_probe_swap"
RESULTS_OUT = REPO_ROOT / "results" / "counter_decode" / "entropy_probe"


@dataclass(frozen=True)
class SwapRun:
    beta_pre: float
    beta_post: float
    switch_step: int
    entropies: np.ndarray            # (n_prompts, n_steps)
    sampled_token_ids: np.ndarray    # (n_prompts, n_steps)


def _generate_with_switch(
    *,
    prompts: list[str],
    beta_pre: float,
    beta_post: float,
    switch_step: int,
    n_steps: int,
    warmup_tokens: int,
    model, tokenizer,
    device: str,
    seed: int,
) -> SwapRun:
    enc = [tokenizer(p, add_special_tokens=False)["input_ids"][:warmup_tokens]
           for p in prompts]
    ids = torch.tensor(enc, dtype=torch.long, device=device)
    B = ids.shape[0]
    attn = torch.ones_like(ids)
    gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=True)
    logits_t = out.logits[:, -1, :].float()
    past = out.past_key_values

    entropies = np.zeros((B, n_steps), dtype=np.float32)
    sampled = np.zeros((B, n_steps), dtype=np.int64)

    for k in range(n_steps):
        log_q = F.log_softmax(logits_t, dim=-1)
        q = log_q.exp()
        H = -(q * log_q).sum(dim=-1)
        entropies[:, k] = H.detach().cpu().numpy()

        # Decoder β: β_pre for steps < switch_step, β_post afterward.
        beta_dec = beta_pre if k < switch_step else beta_post
        decoder_log_q = F.log_softmax(beta_dec * logits_t, dim=-1)
        decoder_probs = decoder_log_q.exp()
        x_next = torch.multinomial(
            decoder_probs, num_samples=1, generator=gen
        ).squeeze(-1)
        sampled[:, k] = x_next.detach().cpu().numpy()

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

    return SwapRun(
        beta_pre=float(beta_pre),
        beta_post=float(beta_post),
        switch_step=int(switch_step),
        entropies=entropies,
        sampled_token_ids=sampled,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--switch-step", type=int, default=80)
    parser.add_argument("--warmup-tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[swap] loading model ...")
    model, tok = load_model_and_tokenizer(device=args.device)

    prompts = list(PROMPTS)
    print(f"[swap] N_prompts={len(prompts)}  n_steps={args.n_steps}  "
          f"switch_step={args.switch_step}  warmup_tokens={args.warmup_tokens}")

    runs: dict[str, SwapRun] = {}
    # The four conditions: 0.5 prefix → {0.5, 1.0, 2.0} post-switch
    # plus a 2.0 prefix → 0.5 post-switch (mirror).
    conditions = [
        ("pre05_post05", 0.5, 0.5),    # no-switch baseline (= Exp C β=0.5)
        ("pre05_post10", 0.5, 1.0),    # switch to β=1
        ("pre05_post20", 0.5, 2.0),    # switch to β=2
        ("pre20_post05", 2.0, 0.5),    # mirror: sharp → flat
        ("pre20_post20", 2.0, 2.0),    # no-switch β=2 baseline
    ]
    for i, (label, beta_pre, beta_post) in enumerate(conditions):
        t0 = time.time()
        print(f"[swap] {label}: β_pre={beta_pre} → β_post={beta_post}")
        runs[label] = _generate_with_switch(
            prompts=prompts,
            beta_pre=beta_pre, beta_post=beta_post,
            switch_step=args.switch_step,
            n_steps=args.n_steps, warmup_tokens=args.warmup_tokens,
            model=model, tokenizer=tok,
            device=args.device, seed=args.seed + i,
        )
        print(f"  done in {time.time() - t0:.1f}s")

    # Save raw arrays.
    payload = {
        "switch_step": args.switch_step,
        "n_steps": args.n_steps,
        "warmup_tokens": args.warmup_tokens,
    }
    for label, r in runs.items():
        payload[f"{label}_entropies"] = r.entropies
        payload[f"{label}_sampled_token_ids"] = r.sampled_token_ids
        payload[f"{label}_beta_pre"] = r.beta_pre
        payload[f"{label}_beta_post"] = r.beta_post
    np.savez_compressed(DATA_OUT / "results.npz", **payload, allow_pickle=True)
    print(f"[swap] arrays → {DATA_OUT / 'results.npz'}")

    # Summarize: post-switch entropy trajectory per condition.
    summary: dict = {
        "config": {
            "n_prompts": len(prompts),
            "n_steps": args.n_steps,
            "switch_step": args.switch_step,
            "warmup_tokens": args.warmup_tokens,
        },
        "post_switch_trajectory": {},
        "pre_switch_means": {},
    }
    for label, r in runs.items():
        post = r.entropies[:, args.switch_step:]    # (n_prompts, n_post)
        pre = r.entropies[:, :args.switch_step]
        # Mean trajectory across prompts.
        med_traj = np.median(post, axis=0).tolist()
        summary["post_switch_trajectory"][label] = {
            "median_H_post_step_0_5_10_20_50": [
                float(med_traj[0]),
                float(med_traj[5]) if len(med_traj) > 5 else None,
                float(med_traj[10]) if len(med_traj) > 10 else None,
                float(med_traj[20]) if len(med_traj) > 20 else None,
                float(med_traj[50]) if len(med_traj) > 50 else None,
            ],
            "median_H_post_full_traj": med_traj,
            "median_H_post_final_window_mean": float(np.median(post[:, -30:])),
            "beta_pre": r.beta_pre,
            "beta_post": r.beta_post,
        }
        summary["pre_switch_means"][label] = {
            "median_H_pre": float(np.median(pre)),
        }

    (RESULTS_OUT / "swap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[swap] summary → {RESULTS_OUT / 'swap_summary.json'}")

    # Print headline result.
    print("\n=== Swap-prefix summary (median H at offset from switch) ===")
    print(f"{'condition':<22} {'pre-med':>8} | {'+0':>6} {'+5':>6} {'+10':>6} {'+20':>6} {'+50':>6} {'final30':>8}")
    for label, r in runs.items():
        post = r.entropies[:, args.switch_step:]
        pre_med = float(np.median(r.entropies[:, :args.switch_step]))
        med_traj = np.median(post, axis=0)
        finalwin = float(np.median(post[:, -30:]))
        print(
            f"{label:<22} {pre_med:>8.3f} | "
            f"{med_traj[0]:>6.3f} {med_traj[5]:>6.3f} {med_traj[10]:>6.3f} "
            f"{med_traj[20]:>6.3f} {med_traj[50]:>6.3f} {finalwin:>8.3f}"
        )


if __name__ == "__main__":
    main()
