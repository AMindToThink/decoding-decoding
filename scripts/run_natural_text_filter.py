"""Run the streaming Laplace β̂ filter on natural text + self-generation arms.

Per corpus (WikiText-103, WritingPrompts):
  * natural arm  — filter sees real next-tokens.
  * self_gen arm — model samples its own next-tokens at β_dec = 1.
The same prompt prefix is used for both arms within a corpus.

Saves to data/natural_text_filter/results.npz.

Usage:
    uv run python scripts/run_natural_text_filter.py [--n 64] [--length 256] [--warmup 20]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from decoding_decoding.natural_text_filter import (
    DEFAULT_LENGTH,
    DEFAULT_SIGMA_0,
    DEFAULT_WARMUP,
    filter_pass_natural,
    filter_pass_self_gen,
    load_model_and_tokenizer,
    load_wikitext_passages,
    load_writingprompts_passages,
    save_results,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "natural_text_filter"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64, help="passages per corpus")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA_0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "results.npz")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run-ntf] N={args.n} length={args.length} warmup={args.warmup} σ_0={args.sigma0}")
    print("[run-ntf] loading model ...")
    model, tok = load_model_and_tokenizer()

    print(f"[run-ntf] loading {args.n} WikiText-103 passages ...")
    wt = load_wikitext_passages(
        n=args.n, min_tokens=args.length, tokenizer=tok, seed=args.seed
    )

    print(f"[run-ntf] loading {args.n} WritingPrompts stories ...")
    wp = load_writingprompts_passages(
        n=args.n, min_tokens=args.length, tokenizer=tok, seed=args.seed
    )

    all_results = []

    t0 = time.time()
    print("[run-ntf] WikiText / natural ...")
    all_results += filter_pass_natural(
        wt,
        corpus="wikitext",
        model=model,
        tokenizer=tok,
        sigma_0=args.sigma0,
        warmup=args.warmup,
        length=args.length,
        batch_size=args.batch_size,
    )
    print(f"[run-ntf]   ↳ done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("[run-ntf] WritingPrompts / natural ...")
    all_results += filter_pass_natural(
        wp,
        corpus="writingprompts",
        model=model,
        tokenizer=tok,
        sigma_0=args.sigma0,
        warmup=args.warmup,
        length=args.length,
        batch_size=args.batch_size,
    )
    print(f"[run-ntf]   ↳ done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("[run-ntf] WikiText / self_gen ...")
    all_results += filter_pass_self_gen(
        wt,
        corpus="wikitext",
        model=model,
        tokenizer=tok,
        sigma_0=args.sigma0,
        warmup=args.warmup,
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"[run-ntf]   ↳ done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("[run-ntf] WritingPrompts / self_gen ...")
    all_results += filter_pass_self_gen(
        wp,
        corpus="writingprompts",
        model=model,
        tokenizer=tok,
        sigma_0=args.sigma0,
        warmup=args.warmup,
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )
    print(f"[run-ntf]   ↳ done in {time.time() - t0:.1f}s")

    print(f"[run-ntf] saving {len(all_results)} PassageResults to {args.out}")
    save_results(all_results, args.out)
    print("[run-ntf] done.")


if __name__ == "__main__":
    main()
