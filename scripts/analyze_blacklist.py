"""Analyze the blacklist experiment and write all outputs:

  results/tables/blacklist_summary.json
      Per-banned-rank effect sizes (early vs late mean p_banned, paired diff
      against baseline) with cluster-bootstrap CIs. Plus per-position curves
      smoothed for plotting.

  results/figures/fig_blacklist_pbanned.png
      p_banned(i) vs generation position, faceted by banned-rank, with the
      seed-naive baseline (k=inf, no ban) overlaid as a control.

  results/figures/fig_blacklist_dropoff.png
      Late-minus-early Δ p_banned for each rank, forest plot with CIs.

  results/figures/fig_blacklist_floor_diagnostic.png
      Per-position fraction of (prompt, run) where the banned token had fallen
      OUT of the saved top-200 — this is upward bias in our p_banned estimate
      (we substitute the floor probability), so this fraction is a sanity
      check for how much of the "drop" is censored.

  results/blacklist_summary.md
      Human-readable rendering, all numbers imported from the JSON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoding_decoding.blacklist_analyze import (
    baseline_p_for_token,
    load_blacklist_condition,
    position_mean_with_ci,
    smooth_along_position,
)


# ------------------------------ tiny helpers ------------------------------ #


def _bootstrap_scalar(arr: np.ndarray, n_resamples: int) -> tuple[float, float, float]:
    """Bootstrap a scalar; arr (n_prompts, n_runs)."""
    pt, lo, hi = position_mean_with_ci(arr[..., None], n_resamples=n_resamples)
    return float(pt[0]), float(lo[0]), float(hi[0])


def _paired(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Pair t, c on (n_prompts, n_runs); average c if its run dim differs."""
    if t.shape[1] == c.shape[1]:
        return t - c
    return t - c.mean(axis=1, keepdims=True)


# ------------------------------ main ------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--out-fig-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "figures",
    )
    parser.add_argument(
        "--out-tables-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "tables",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "blacklist_summary.md",
    )
    parser.add_argument(
        "--banned-tokens-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "tables"
        / "banned_tokens.json",
    )
    parser.add_argument("--early-start", type=int, default=25)
    parser.add_argument("--early-end", type=int, default=75)
    parser.add_argument("--late-start", type=int, default=500)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--smooth-window", type=int, default=20)
    args = parser.parse_args()
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)
    args.out_tables_dir.mkdir(parents=True, exist_ok=True)

    banned_meta = json.loads(args.banned_tokens_json.read_text())["banned_token_meta"]
    n_ranks = len(banned_meta)

    print(f"loading {n_ranks} blacklist conditions...")
    cond_data = []
    for rank in range(1, n_ranks + 1):
        cd = load_blacklist_condition(args.data_dir, rank)
        print(
            f"  rank {rank} (token id {cd.banned_token_id}): "
            f"p_banned shape {cd.p_banned.shape}; "
            f"in-top-200 fraction across all (p, r, i): {cd.in_topn.mean():.3f}"
        )
        cond_data.append(cd)

    # ---------- compute per-condition stats ----------
    per_rank: list[dict] = []
    per_rank_curves: list[dict] = []  # for plotting
    for cd, meta in zip(cond_data, banned_meta, strict=True):
        rank = cd.banned_rank
        token_id = cd.banned_token_id
        p_t = cd.p_banned   # (n_prompts, n_runs, L_t)
        # The baseline (top-k=∞) traces may have been extended past L_t (e.g.
        # to L=2048), so truncate to match for paired comparisons.
        p_c_full = baseline_p_for_token(args.data_dir, token_id)
        L_t = p_t.shape[-1]
        p_c = p_c_full[..., :L_t]

        # Position-mean curves with CIs.
        t_pt, t_lo, t_hi = position_mean_with_ci(p_t, n_resamples=2000)
        c_pt, c_lo, c_hi = position_mean_with_ci(p_c, n_resamples=2000)

        # Early/late effects.
        early_t = p_t[..., args.early_start:args.early_end].mean(axis=-1)
        late_t = p_t[..., args.late_start:].mean(axis=-1)
        early_c = p_c[..., args.early_start:args.early_end].mean(axis=-1)
        late_c = p_c[..., args.late_start:].mean(axis=-1)

        e_t = _bootstrap_scalar(early_t, args.n_resamples)
        l_t = _bootstrap_scalar(late_t, args.n_resamples)
        e_c = _bootstrap_scalar(early_c, args.n_resamples)
        l_c = _bootstrap_scalar(late_c, args.n_resamples)

        # Within-condition: late − early in treatment.
        d_within = _bootstrap_scalar(late_t - early_t, args.n_resamples)
        # Treatment − control late: paired by (prompt, run).
        d_late_paired = _bootstrap_scalar(_paired(late_t, late_c), args.n_resamples)
        # Treatment − control early: paired (sanity check; should be small).
        d_early_paired = _bootstrap_scalar(_paired(early_t, early_c), args.n_resamples)

        per_rank.append({
            "banned_rank": rank,
            "banned_token_id": token_id,
            "decoded": meta["decoded"],
            "baseline_frequency": meta["fraction"],
            "early_treatment": list(e_t),
            "late_treatment": list(l_t),
            "early_control": list(e_c),
            "late_control": list(l_c),
            "late_minus_early_within_treatment": list(d_within),
            "treatment_minus_control_at_late": list(d_late_paired),
            "treatment_minus_control_at_early": list(d_early_paired),
            "fraction_in_top200": float(cd.in_topn.mean()),
        })

        # Smoothed for plotting; also track in_topn fraction.
        t_curve = smooth_along_position(t_pt, args.smooth_window)
        c_curve = smooth_along_position(c_pt, args.smooth_window)
        in_top_curve = cd.in_topn.mean(axis=(0, 1))
        per_rank_curves.append({
            "banned_rank": rank,
            "decoded": meta["decoded"],
            "t_pt": t_pt, "t_lo": t_lo, "t_hi": t_hi,
            "c_pt": c_pt, "c_lo": c_lo, "c_hi": c_hi,
            "t_smooth": t_curve, "c_smooth": c_curve,
            "in_top200": in_top_curve,
        })

    summary = {
        "provenance": {
            "script": "scripts/analyze_blacklist.py",
            "argv": sys.argv,
            "generated_at": dt.datetime.now().isoformat(),
            "data_dir": str(args.data_dir),
            "L": int(cond_data[0].p_banned.shape[2]),
            "n_prompts": int(cond_data[0].p_banned.shape[0]),
            "n_runs_per_condition": int(cond_data[0].p_banned.shape[1]),
            "early_window": [args.early_start, args.early_end],
            "late_start": args.late_start,
            "n_resamples": args.n_resamples,
            "smooth_window": args.smooth_window,
        },
        "per_rank": per_rank,
    }

    out_json = args.out_tables_dir / "blacklist_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    # ---------- plots ----------
    fig_pbanned(per_rank_curves, args.out_fig_dir / "fig_blacklist_pbanned.png")
    fig_dropoff(per_rank, args.out_fig_dir / "fig_blacklist_dropoff.png")
    fig_floor(per_rank_curves, args.out_fig_dir / "fig_blacklist_floor_diagnostic.png")

    # ---------- markdown (read from JSON) ----------
    md = render_md(summary)
    args.out_md.write_text(md)
    print(f"wrote {args.out_md}")


def fig_pbanned(curves: list[dict], out_path: Path) -> None:
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, curves):
        L = c["t_pt"].shape[0]
        xs = np.arange(L)
        ax.plot(xs, c["t_smooth"], color="C0", label="banned (treatment)", lw=1.4)
        ax.fill_between(xs, c["t_lo"], c["t_hi"], color="C0", alpha=0.18)
        ax.plot(xs, c["c_smooth"], color="C1", label="baseline (top-k=∞)", lw=1.4)
        ax.fill_between(xs, c["c_lo"], c["c_hi"], color="C1", alpha=0.18)
        ax.set_yscale("log")
        ax.set_title(f"banned rank {c['banned_rank']}: {c['decoded']!r}")
        ax.set_xlabel("position")
        ax.set_ylabel("p_banned (log)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("p_banned vs. position, treatment vs. unrestricted-baseline control", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_dropoff(per_rank: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(per_rank) + 2.2))
    ys = np.arange(len(per_rank))[::-1]
    labels = []
    for y, r in zip(ys, per_rank):
        d_pt, d_lo, d_hi = r["late_minus_early_within_treatment"]
        ax.errorbar([d_pt], [y], xerr=[[d_pt - d_lo], [d_hi - d_pt]],
                    fmt="o", color="C0", capsize=4, lw=1.4)
        # Place the value label below the marker via pixel-offset annotation, so
        # it stays clear of the title on the topmost row regardless of axis range.
        ax.annotate(
            f"{d_pt:+.5f} [{d_lo:+.5f}, {d_hi:+.5f}]",
            xy=(d_pt, y),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )
        labels.append(
            f"rank {r['banned_rank']}: {r['decoded']!r} "
            f"(baseline freq {r['baseline_frequency']:.3f})"
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_ylim(ys.min() - 0.6, ys.max() + 0.4)
    ax.set_yticklabels(labels)
    ax.set_xlabel("late − early p_banned (within blacklist treatment)")
    ax.set_title("Δ p_banned: does the model learn its own ban?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_floor(curves: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for c in curves:
        ax.plot(np.arange(c["in_top200"].shape[0]), c["in_top200"],
                label=f"rank {c['banned_rank']}: {c['decoded']!r}", lw=1.3)
    ax.set_xlabel("position")
    ax.set_ylabel("fraction of (prompt, run) with banned token in saved top-200")
    ax.set_title("Diagnostic: how often the banned token is captured in top-200")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _fmt(triple: list[float]) -> str:
    pt, lo, hi = triple
    return f"{pt:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def render_md(summary: dict) -> str:
    p = summary["provenance"]
    lines = [
        f"<!-- Generated by: {p['script']} {' '.join(p['argv'][1:])} -->",
        f"<!-- Source: {p['data_dir']}/manifest.parquet (family=blacklist) -->",
        f"<!-- Generated at: {p['generated_at']} -->",
        "<!-- All numbers below are read from results/tables/blacklist_summary.json. -->",
        "",
        "# Blacklist self-prediction — does the LM catch its own ban?",
        "",
        f"L = {p['L']}, prompts = {p['n_prompts']}, runs per (prompt, banned-rank) = "
        f"{p['n_runs_per_condition']}. "
        f"early window = positions {p['early_window'][0]}–{p['early_window'][1]}, "
        f"late = positions ≥ {p['late_start']}.",
        "",
        "## Effect sizes (95% cluster-bootstrap CI)",
        "",
        "| banned rank | token | baseline freq | early p_banned | late p_banned "
        "| late − early (within treatment) | T − C late | T − C early (sanity) "
        "| in-top-200 fraction |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["per_rank"]:
        lines.append(
            f"| {r['banned_rank']} | `{r['decoded']}` | {r['baseline_frequency']:.4f} "
            f"| {_fmt(r['early_treatment'])} | {_fmt(r['late_treatment'])} "
            f"| {_fmt(r['late_minus_early_within_treatment'])} "
            f"| {_fmt(r['treatment_minus_control_at_late'])} "
            f"| {_fmt(r['treatment_minus_control_at_early'])} "
            f"| {r['fraction_in_top200']:.3f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "* **late − early (within treatment)**: most direct test of the hypothesis. "
        "Negative ⇒ the model assigns lower softmax prob to the banned token over time.",
        "* **T − C late**: paired contrast against the baseline (k=∞, no ban). Asks "
        "whether the late-position prob in the banned condition is below the "
        "baseline-condition prob *for the same prompts*. Negative ⇒ ban is detected.",
        "* **T − C early**: should be near zero — at early positions there's no ban "
        "history yet, so any difference is RNG/batch noise. Use as a sanity check.",
        "* **in-top-200 fraction**: when the banned token has fallen out of the "
        "saved top-200, we substitute the row's floor probability, which is an "
        "*upper bound*. A lower fraction here means the reported drop is, if "
        "anything, an underestimate.",
        "",
        "Figures: `results/figures/fig_blacklist_pbanned.png` (curves), "
        "`fig_blacklist_dropoff.png` (forest), "
        "`fig_blacklist_floor_diagnostic.png` (in-top-200 fraction over time).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
