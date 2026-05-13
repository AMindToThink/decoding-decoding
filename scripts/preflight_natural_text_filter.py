"""Pre-flight check before the full natural-text filter run.

The critic flagged: if Var_q (per-step Fisher) is much smaller than the prior
precision J_0 = 1/σ_0² = 4, the filter is prior-dominated and the null check is
vacuous. Verify Var_q is at least order 1 by running on a single passage from
each corpus.

Also runs both arms (self_gen + natural) on the same 4 passages to make sure
the per-arm pipelines are wired correctly before committing the full sweep.

Usage:
    uv run python scripts/preflight_natural_text_filter.py
"""

from __future__ import annotations

import numpy as np

from decoding_decoding.natural_text_filter import (
    DEFAULT_SIGMA_0,
    filter_pass_natural,
    filter_pass_self_gen,
    load_model_and_tokenizer,
    load_wikitext_passages,
    load_writingprompts_passages,
)


def _describe(name: str, results):
    print(f"\n{name}")
    print("-" * len(name))
    for r in results:
        fisher_med = float(np.median(r.fisher))
        fisher_mean = float(np.mean(r.fisher))
        final_log_beta = float(r.log_beta_hat[-1])
        final_J = float(r.J[-1])
        total_fisher = float(np.sum(r.fisher))
        print(
            f"  passage {r.passage_id} [{r.corpus}/{r.arm}]  "
            f"median Var_q={fisher_med:.3f}  mean Var_q={fisher_mean:.3f}  "
            f"final log β̂={final_log_beta:+.3f}  final J={final_J:.1f}  "
            f"ΣFisher={total_fisher:.1f}"
        )
        prev = r.text_preview.replace("\n", " ")[:90]
        print(f"      preview: {prev!r}")


def main() -> None:
    print("[preflight] loading model ...")
    model, tok = load_model_and_tokenizer()

    print("[preflight] loading 2 passages from each corpus ...")
    wt = load_wikitext_passages(n=2, min_tokens=300, tokenizer=tok, seed=0)
    wp = load_writingprompts_passages(n=2, min_tokens=300, tokenizer=tok, seed=0)

    print("[preflight] running natural arm on WikiText (2 passages) ...")
    nat_wt = filter_pass_natural(wt, corpus="wikitext", model=model, tokenizer=tok)
    _describe("WikiText / natural", nat_wt)

    print("[preflight] running natural arm on WritingPrompts (2 passages) ...")
    nat_wp = filter_pass_natural(
        wp, corpus="writingprompts", model=model, tokenizer=tok
    )
    _describe("WritingPrompts / natural", nat_wp)

    print("[preflight] running self_gen arm on WikiText prompts (2 passages) ...")
    sg_wt = filter_pass_self_gen(wt, corpus="wikitext", model=model, tokenizer=tok)
    _describe("WikiText / self_gen", sg_wt)

    print("[preflight] running self_gen arm on WritingPrompts prompts (2 passages) ...")
    sg_wp = filter_pass_self_gen(
        wp, corpus="writingprompts", model=model, tokenizer=tok
    )
    _describe("WritingPrompts / self_gen", sg_wp)

    # Sanity verdict.
    print()
    sigma_0 = DEFAULT_SIGMA_0
    J0 = 1.0 / sigma_0 ** 2
    fishers = [
        float(np.median(r.fisher)) for r in (nat_wt + nat_wp + sg_wt + sg_wp)
    ]
    median_var_q = float(np.median(fishers))
    print(
        f"[preflight] σ_0={sigma_0} → J_0={J0:.2f}; median Var_q across passages={median_var_q:.3f}"
    )
    if median_var_q < 0.05:
        print(
            "[preflight] WARN: Var_q < 0.05 — prior may dominate even at the end. "
            "Filter test may be uninformative."
        )
    elif median_var_q < 0.5:
        print(
            "[preflight] CAUTION: Var_q < 0.5. Prior dominates for many steps. "
            "Asymptotic β̂ should still be informative if length is large enough."
        )
    else:
        print(
            "[preflight] OK: Var_q ≥ 0.5 → prior washes out within ~ J_0 / Var_q steps."
        )


if __name__ == "__main__":
    main()
