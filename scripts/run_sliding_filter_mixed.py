"""Exp B: sliding β̂_t on alternating-genre corpus.

Reframed in light of Exp A (filter ≠ MLE): rather than asking whether
sliding β̂_t beats batch MLE at test PPL (it cannot, since the filter is
not a Guo-style calibrator at all), we ask whether the filter's sliding
β̂_t signal tracks genre transitions in real heterogeneous text.

Construction:
  Alternate FULL documents (per critic Exp A audit): wiki article (256
  tokens) → WritingPrompts story (256 tokens) → wiki → WP → ... × 4 pairs.
  Total 8 × 256 = 2048 tokens.

The filter is run with γ = memory_decay ∈ {0.90, 0.95, 0.98, 0.99}.
Within each genre block, exclude a 20-token boundary window on each side
when summarizing the filter signal — per critic's earlier flag that the
seam introduces artifact.

Outputs:
  data/sliding_filter_mixed/results.npz
  results/counter_decode/sliding_filter_mixed/summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from decoding_decoding.counter_decode import init_laplace, laplace_update
from decoding_decoding.natural_text_filter import (
    DEFAULT_SIGMA_0,
    load_model_and_tokenizer,
    load_wikitext_passages,
    load_writingprompts_passages,
    _tokenize_to_length,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = REPO_ROOT / "data" / "sliding_filter_mixed"
RESULTS_OUT = REPO_ROOT / "results" / "counter_decode" / "sliding_filter_mixed"


def _build_mixed_sequence(
    wiki: list[str], wp: list[str], chunk_tokens: int, n_pairs: int, tokenizer,
) -> tuple[torch.Tensor, list[tuple[int, str]]]:
    """Construct a single sequence of alternating (wiki, wp) chunks.

    Returns:
      ids: (1, n_pairs * 2 * chunk_tokens) int64
      boundaries: list of (start_index_in_sequence, genre) per chunk
    """
    chunks: list[list[int]] = []
    boundaries: list[tuple[int, str]] = []
    cursor = 0
    for p in range(n_pairs):
        wt_ids = tokenizer(wiki[p], add_special_tokens=False)["input_ids"][:chunk_tokens]
        wp_ids = tokenizer(wp[p], add_special_tokens=False)["input_ids"][:chunk_tokens]
        if len(wt_ids) < chunk_tokens or len(wp_ids) < chunk_tokens:
            raise ValueError(
                f"pair {p}: not enough tokens (wiki={len(wt_ids)}, wp={len(wp_ids)})"
            )
        boundaries.append((cursor, "wikitext"))
        cursor += chunk_tokens
        chunks.append(wt_ids)
        boundaries.append((cursor, "writingprompts"))
        cursor += chunk_tokens
        chunks.append(wp_ids)
    flat = [tok for ch in chunks for tok in ch]
    ids = torch.tensor([flat], dtype=torch.long)
    return ids, boundaries


def _run_sliding_filter(
    *,
    ids: torch.Tensor,           # (1, T)
    model, tokenizer,
    sigma_0: float,
    memory_decay: float,
    warmup: int,
    device: str,
) -> dict:
    """Run a single-sequence sliding β̂_t filter with the given γ.

    Returns per-step trajectories of log β̂, J, score, fisher, model entropy.
    """
    ids = ids.to(device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    all_logits = out.logits[0].float()    # (T, V)
    T = all_logits.shape[0]

    state = init_laplace(1, sigma_0=sigma_0, device=device, dtype=torch.float32)
    U = T - warmup
    log_beta_hat = np.zeros(U, dtype=np.float32)
    J_arr = np.zeros(U, dtype=np.float32)
    score_arr = np.zeros(U, dtype=np.float32)
    fisher_arr = np.zeros(U, dtype=np.float32)
    entropy_arr = np.zeros(U, dtype=np.float32)

    for k in range(U):
        seq_pos = warmup - 1 + k
        target_pos = warmup + k
        if target_pos >= T:
            break
        logits_t = all_logits[seq_pos:seq_pos + 1, :]
        x_next = ids[0, target_pos:target_pos + 1]

        log_beta_hat[k] = state.eta_hat.detach().cpu().numpy()[0]
        J_arr[k] = state.J.detach().cpu().numpy()[0]
        log_q = F.log_softmax(logits_t, dim=-1)
        entropy_arr[k] = float(-(log_q.exp() * log_q).sum(dim=-1).detach().cpu().numpy()[0])

        state, score, fisher = laplace_update(
            state, logits_t, x_next,
            memory_decay=memory_decay,
            prior_eta=-0.5 * sigma_0 ** 2,
            prior_J=1.0 / sigma_0 ** 2,
        )
        score_arr[k] = score.detach().cpu().numpy()[0]
        fisher_arr[k] = fisher.detach().cpu().numpy()[0]

    return {
        "log_beta_hat": log_beta_hat,
        "J": J_arr,
        "score": score_arr,
        "fisher": fisher_arr,
        "model_entropy": entropy_arr,
        "memory_decay": memory_decay,
        "warmup": warmup,
        "n_filter_steps": U,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=4,
                        help="number of (wiki, wp) chunk pairs in the mixed sequence")
    parser.add_argument("--chunk-tokens", type=int, default=256,
                        help="tokens per chunk")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA_0)
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.90, 0.95, 0.98, 0.99, 1.00])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[mix] loading model ...")
    model, tok = load_model_and_tokenizer(device=args.device)

    n_needed = args.n_pairs
    print(f"[mix] loading {n_needed} wiki + {n_needed} WP passages ...")
    wt = load_wikitext_passages(
        n=n_needed, min_tokens=args.chunk_tokens, tokenizer=tok, seed=args.seed
    )
    wp = load_writingprompts_passages(
        n=n_needed, min_tokens=args.chunk_tokens, tokenizer=tok, seed=args.seed
    )

    ids, boundaries = _build_mixed_sequence(
        wt, wp, chunk_tokens=args.chunk_tokens, n_pairs=args.n_pairs, tokenizer=tok,
    )
    print(f"[mix] built sequence of {ids.shape[1]} tokens, "
          f"{len(boundaries)} chunks alternating")

    runs: dict[float, dict] = {}
    for gamma in args.gammas:
        t0 = time.time()
        print(f"[mix] running filter at γ={gamma} ...")
        runs[gamma] = _run_sliding_filter(
            ids=ids,
            model=model, tokenizer=tok,
            sigma_0=args.sigma0,
            memory_decay=gamma,
            warmup=args.warmup,
            device=args.device,
        )
        print(f"  done in {time.time() - t0:.1f}s")

    # Save raw arrays.
    payload = {
        "chunk_tokens": args.chunk_tokens,
        "n_pairs": args.n_pairs,
        "warmup": args.warmup,
        "sigma_0": args.sigma0,
        "boundaries_start": np.array([b for b, _ in boundaries], dtype=np.int64),
        "boundaries_genre": np.array([g for _, g in boundaries], dtype=object),
        "ids": ids.cpu().numpy(),
    }
    for gamma, r in runs.items():
        key = f"gamma_{gamma:.2f}".replace(".", "_")
        for f in ["log_beta_hat", "J", "score", "fisher", "model_entropy"]:
            payload[f"{key}_{f}"] = r[f]
    np.savez_compressed(DATA_OUT / "results.npz", **payload, allow_pickle=True)
    print(f"[mix] arrays → {DATA_OUT / 'results.npz'}")

    # Summary: per-chunk mean β̂ and mean H, excluding 20-token boundary windows.
    summary: dict = {
        "config": {
            "n_pairs": args.n_pairs,
            "chunk_tokens": args.chunk_tokens,
            "warmup": args.warmup,
            "sigma_0": args.sigma0,
            "gammas": args.gammas,
            "boundary_exclusion": 20,
        },
        "per_gamma": {},
    }
    bexc = 20
    for gamma, r in runs.items():
        per_chunk_log_beta = []
        per_chunk_H = []
        per_chunk_genre = []
        for i, (start, genre) in enumerate(boundaries):
            # Filter-step indices for this chunk (after warmup):
            #   start - warmup    (clamp at 0) ... start + chunk_tokens - warmup
            chunk_start = max(start - args.warmup, 0)
            chunk_end = start + args.chunk_tokens - args.warmup
            chunk_end = min(chunk_end, r["log_beta_hat"].shape[0])
            cs_inner = chunk_start + bexc
            ce_inner = chunk_end - bexc
            if ce_inner <= cs_inner:
                continue
            per_chunk_log_beta.append(
                float(np.mean(r["log_beta_hat"][cs_inner:ce_inner]))
            )
            per_chunk_H.append(
                float(np.mean(r["model_entropy"][cs_inner:ce_inner]))
            )
            per_chunk_genre.append(genre)

        wt_lb = [v for v, g in zip(per_chunk_log_beta, per_chunk_genre) if g == "wikitext"]
        wp_lb = [v for v, g in zip(per_chunk_log_beta, per_chunk_genre) if g == "writingprompts"]
        wt_H = [v for v, g in zip(per_chunk_H, per_chunk_genre) if g == "wikitext"]
        wp_H = [v for v, g in zip(per_chunk_H, per_chunk_genre) if g == "writingprompts"]

        summary["per_gamma"][f"{gamma}"] = {
            "per_chunk_log_beta": per_chunk_log_beta,
            "per_chunk_H": per_chunk_H,
            "per_chunk_genre": per_chunk_genre,
            "wt_log_beta_mean": float(np.mean(wt_lb)) if wt_lb else None,
            "wp_log_beta_mean": float(np.mean(wp_lb)) if wp_lb else None,
            "wt_log_beta_minus_wp_log_beta": (
                (float(np.mean(wt_lb)) - float(np.mean(wp_lb)))
                if wt_lb and wp_lb else None
            ),
            "wt_H_mean": float(np.mean(wt_H)) if wt_H else None,
            "wp_H_mean": float(np.mean(wp_H)) if wp_H else None,
        }

    (RESULTS_OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[mix] summary → {RESULTS_OUT / 'summary.json'}")

    # Headline print.
    print("\n=== Sliding filter on alternating genres ===")
    print(f"{'γ':>6} | {'mean log β̂ WT':>14} {'mean log β̂ WP':>14} {'WT−WP':>8}  {'H WT':>6} {'H WP':>6}")
    for gamma in args.gammas:
        s = summary["per_gamma"][f"{gamma}"]
        wt_lb = s["wt_log_beta_mean"]
        wp_lb = s["wp_log_beta_mean"]
        diff = s["wt_log_beta_minus_wp_log_beta"]
        hwt = s["wt_H_mean"]
        hwp = s["wp_H_mean"]
        print(
            f"{gamma:>6.2f} | "
            f"{wt_lb:+14.4f} {wp_lb:+14.4f} {diff:+8.4f}  "
            f"{hwt:>6.3f} {hwp:>6.3f}"
        )


if __name__ == "__main__":
    main()
