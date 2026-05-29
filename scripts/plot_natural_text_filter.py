"""Plot results of the natural-text filter experiment.

Reads data/natural_text_filter/results.npz and produces:

  F_ntf_1_log_beta_vs_fisher.png : log β̂ vs ΣFisher per (corpus, arm). Headline.
  F_ntf_2_log_beta_vs_t.png      : log β̂ vs filter-step t (sanity / cache check).
  F_ntf_3_final_distribution.png : boxplot of final log β̂ per (corpus, arm).
  F_ntf_4_score_over_t.png       : per-step score mean + IQR per condition.
  F_ntf_5_entropy_over_t.png     : per-step model entropy per corpus.
  F_ntf_6_entropy_vs_finalbeta.png : scatter — final log β̂ vs mean H(ℓ).

Plus a JSON summary with per-condition asymptotic stats.

Usage:
    uv run python scripts/plot_natural_text_filter.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoding_decoding.natural_text_filter import (
    DEFAULT_SIGMA_0,
    load_results,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_NPZ = REPO_ROOT / "data" / "natural_text_filter" / "results.npz"
OUT_DIR = REPO_ROOT / "results" / "counter_decode" / "natural_text"

# Visual conventions.
COLORS = {
    ("wikitext", "self_gen"): "#1f5e8c",
    ("wikitext", "natural"): "#a8412a",
    ("writingprompts", "self_gen"): "#6b9bd2",
    ("writingprompts", "natural"): "#d97a4a",
}
LABELS = {
    ("wikitext", "self_gen"): "WikiText / self_gen (β_dec=1)",
    ("wikitext", "natural"): "WikiText / natural",
    ("writingprompts", "self_gen"): "WritingPrompts / self_gen (β_dec=1)",
    ("writingprompts", "natural"): "WritingPrompts / natural",
}
CONDITIONS = list(COLORS.keys())


def _by_condition(results) -> dict[tuple[str, str], list]:
    out: dict[tuple[str, str], list] = defaultdict(list)
    for r in results:
        out[(r.corpus, r.arm)].append(r)
    return out


def _stack(rs, attr: str) -> np.ndarray:
    """Stack one PassageResult attribute into (n_passages, U)."""
    return np.stack([getattr(r, attr) for r in rs], axis=0)


def _cumulative_fisher(rs) -> np.ndarray:
    """Σ_τ Var_q(τ) up to each step, shape (n_passages, U).

    Equivalent to (J_t - J_0) when memory_decay=1.0 and evidence_weight=1.0.
    Computed from the per-step ``fisher`` array, cumulative.
    """
    fisher = _stack(rs, "fisher")
    return np.cumsum(fisher, axis=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(RESULTS_NPZ)
    print(f"[plot-ntf] loaded {len(results)} PassageResults")

    by_cond = _by_condition(results)
    summary: dict = {"sigma_0": DEFAULT_SIGMA_0, "rows": []}

    # ------------------------------------------------------------------
    # Plot 1: log β̂ vs ΣFisher (headline)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    # For ΣFisher x-axis we need a common grid per condition. ΣFisher is
    # per-passage and monotone; absolute values differ across passages and
    # corpora. Plot each condition over ITS OWN cumfisher range so we don't
    # hide tail behavior of low-entropy corpora.
    for cond in CONDITIONS:
        rs = by_cond[cond]
        cf = _cumulative_fisher(rs)
        lb = _stack(rs, "log_beta_hat")
        cf_max_med = float(np.median(cf[:, -1]))
        grid = np.linspace(0.0, cf_max_med, 200)
        interp_rows = np.zeros((len(rs), len(grid)), dtype=np.float64)
        for i in range(len(rs)):
            interp_rows[i] = np.interp(grid, cf[i], lb[i])
        med = np.median(interp_rows, axis=0)
        q25 = np.quantile(interp_rows, 0.25, axis=0)
        q75 = np.quantile(interp_rows, 0.75, axis=0)
        ax.plot(grid, med, color=COLORS[cond], label=LABELS[cond], linewidth=1.6)
        ax.fill_between(grid, q25, q75, color=COLORS[cond], alpha=0.18)
    ax.axhline(0.0, color="#6b6453", linewidth=0.7, linestyle=":", label="prior mean log β̂ = 0")
    ax.set_xlabel(r"cumulative Fisher $\sum_\tau \mathrm{Var}_q(\tau) = J_t - J_0$ (effective sample size)")
    ax.set_ylabel(r"log $\hat\beta_t$ (median across passages, IQR band)")
    ax.set_title("F_ntf_1 — log β̂ trajectory vs cumulative Fisher")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ntf_1_log_beta_vs_fisher.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-ntf] wrote {OUT_DIR / 'F_ntf_1_log_beta_vs_fisher.png'}")

    # ------------------------------------------------------------------
    # Plot 2: log β̂ vs t (filter step)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for cond in CONDITIONS:
        rs = by_cond[cond]
        lb = _stack(rs, "log_beta_hat")
        t = np.arange(lb.shape[1])
        med = np.median(lb, axis=0)
        q25 = np.quantile(lb, 0.25, axis=0)
        q75 = np.quantile(lb, 0.75, axis=0)
        ax.plot(t, med, color=COLORS[cond], label=LABELS[cond], linewidth=1.4)
        ax.fill_between(t, q25, q75, color=COLORS[cond], alpha=0.18)
    ax.axhline(0.0, color="#6b6453", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"filter step $t$ (0 = first observation after warmup)")
    ax.set_ylabel(r"log $\hat\beta_t$ (median + IQR)")
    ax.set_title("F_ntf_2 — log β̂ trajectory vs filter step")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ntf_2_log_beta_vs_t.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-ntf] wrote {OUT_DIR / 'F_ntf_2_log_beta_vs_t.png'}")

    # ------------------------------------------------------------------
    # Plot 3: distribution of final log β̂
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    data = []
    labels = []
    colors = []
    for cond in CONDITIONS:
        rs = by_cond[cond]
        lb = _stack(rs, "log_beta_hat")
        data.append(lb[:, -1])
        labels.append(LABELS[cond])
        colors.append(COLORS[cond])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0.0, color="#6b6453", linewidth=0.7, linestyle=":")
    ax.set_ylabel(r"final log $\hat\beta_t$ (one point per passage)")
    ax.set_title("F_ntf_3 — distribution of asymptotic log β̂")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ntf_3_final_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-ntf] wrote {OUT_DIR / 'F_ntf_3_final_distribution.png'}")

    # ------------------------------------------------------------------
    # Plot 4: per-step score
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for cond in CONDITIONS:
        rs = by_cond[cond]
        sc = _stack(rs, "score")
        t = np.arange(sc.shape[1])
        med = np.median(sc, axis=0)
        q25 = np.quantile(sc, 0.25, axis=0)
        q75 = np.quantile(sc, 0.75, axis=0)
        ax.plot(t, med, color=COLORS[cond], label=LABELS[cond], linewidth=1.2)
        ax.fill_between(t, q25, q75, color=COLORS[cond], alpha=0.12)
    ax.axhline(0.0, color="#6b6453", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"filter step $t$")
    ax.set_ylabel(r"score = $\ell_t[x_t] - E_q[\ell_t]$ (median, IQR)")
    ax.set_title("F_ntf_4 — per-step score over t")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ntf_4_score_over_t.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-ntf] wrote {OUT_DIR / 'F_ntf_4_score_over_t.png'}")

    # ------------------------------------------------------------------
    # Plot 5: model entropy over t (corpus characterization)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for cond in CONDITIONS:
        rs = by_cond[cond]
        H = _stack(rs, "model_entropy")
        t = np.arange(H.shape[1])
        med = np.median(H, axis=0)
        q25 = np.quantile(H, 0.25, axis=0)
        q75 = np.quantile(H, 0.75, axis=0)
        ax.plot(t, med, color=COLORS[cond], label=LABELS[cond], linewidth=1.2)
        ax.fill_between(t, q25, q75, color=COLORS[cond], alpha=0.12)
    ax.set_xlabel(r"filter step $t$")
    ax.set_ylabel(r"$H(\mathrm{softmax}(\ell_t))$ in nats (median + IQR)")
    ax.set_title("F_ntf_5 — model entropy over t (corpus characterization)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ntf_5_entropy_over_t.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-ntf] wrote {OUT_DIR / 'F_ntf_5_entropy_over_t.png'}")

    # ------------------------------------------------------------------
    # Plot 6: final log β̂ vs mean model entropy (per passage)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    for cond in CONDITIONS:
        rs = by_cond[cond]
        H_mean = np.array([float(np.mean(r.model_entropy)) for r in rs])
        lb_final = np.array([float(r.log_beta_hat[-1]) for r in rs])
        ax.scatter(
            H_mean,
            lb_final,
            color=COLORS[cond],
            label=LABELS[cond],
            alpha=0.7,
            s=30,
            edgecolors="white",
            linewidth=0.5,
        )
    ax.axhline(0.0, color="#6b6453", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"mean model entropy per passage (nats)")
    ax.set_ylabel(r"final log $\hat\beta_t$")
    ax.set_title("F_ntf_6 — asymptotic β̂ vs corpus entropy under the model")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ntf_6_entropy_vs_finalbeta.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-ntf] wrote {OUT_DIR / 'F_ntf_6_entropy_vs_finalbeta.png'}")

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------
    for cond in CONDITIONS:
        rs = by_cond[cond]
        lb_final = np.array([float(r.log_beta_hat[-1]) for r in rs])
        H_mean = np.array([float(np.mean(r.model_entropy)) for r in rs])
        fisher_total = np.array([float(np.sum(r.fisher)) for r in rs])
        # SE of the median: asymptotic 1.2533·σ/√n.
        n = len(rs)
        std = float(np.std(lb_final, ddof=1))
        se_mean = std / np.sqrt(n)
        se_med = 1.2533 * se_mean
        # Trimmed median (drop 3 worst on each tail).
        sorted_lb = np.sort(lb_final)
        trim_med = float(np.median(sorted_lb[3:-3]))
        summary["rows"].append(
            {
                "corpus": cond[0],
                "arm": cond[1],
                "n_passages": n,
                "log_beta_final_median": float(np.median(lb_final)),
                "log_beta_final_iqr_low": float(np.quantile(lb_final, 0.25)),
                "log_beta_final_iqr_high": float(np.quantile(lb_final, 0.75)),
                "log_beta_final_mean": float(np.mean(lb_final)),
                "log_beta_final_std": std,
                "log_beta_final_se_mean": float(se_mean),
                "log_beta_final_se_median": float(se_med),
                "log_beta_final_trimmed_median": trim_med,
                "median_t_stat_vs_zero": float(np.median(lb_final) / se_med),
                "mean_model_entropy_median": float(np.median(H_mean)),
                "total_fisher_median": float(np.median(fisher_total)),
            }
        )

    # Paired natural − self_gen differences per corpus.
    paired: dict = {}
    for corpus in ("wikitext", "writingprompts"):
        nat = sorted(by_cond[(corpus, "natural")], key=lambda r: r.passage_id)
        sg = sorted(by_cond[(corpus, "self_gen")], key=lambda r: r.passage_id)
        diff = np.array([
            float(n.log_beta_hat[-1]) - float(s.log_beta_hat[-1])
            for n, s in zip(nat, sg)
        ])
        nn = len(diff)
        mean_diff = float(diff.mean())
        std_diff = float(diff.std(ddof=1))
        se_diff = std_diff / np.sqrt(nn)
        paired[corpus] = {
            "n_pairs": nn,
            "mean_diff": mean_diff,
            "median_diff": float(np.median(diff)),
            "std_diff": std_diff,
            "se_diff": se_diff,
            "t_stat": mean_diff / se_diff if se_diff > 0 else 0.0,
        }
    summary["paired_natural_minus_self_gen"] = paired
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[plot-ntf] wrote {OUT_DIR / 'summary.json'}")

    # Compact terminal report.
    print()
    print(
        f"{'corpus':>16} {'arm':>10} {'N':>4} {'log β̂_∞ med':>14} {'IQR':>14} "
        f"{'mean H':>9} {'med ΣF':>9}"
    )
    print("-" * 80)
    for row in summary["rows"]:
        iqr_str = f"[{row['log_beta_final_iqr_low']:+.3f}, {row['log_beta_final_iqr_high']:+.3f}]"
        print(
            f"{row['corpus']:>16} {row['arm']:>10} {row['n_passages']:>4} "
            f"{row['log_beta_final_median']:>+14.3f} {iqr_str:>14} "
            f"{row['mean_model_entropy_median']:>9.3f} {row['total_fisher_median']:>9.1f}"
        )


if __name__ == "__main__":
    main()
