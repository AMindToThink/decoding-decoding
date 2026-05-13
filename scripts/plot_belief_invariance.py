"""Plot the belief-invariance test results.

Figures:
  F_bi_1_var_heatmap.png   — CE variance across β_prefix in (γ, p) plane
                              (one panel per α). Lower = better.
  F_bi_2_ce_vs_p.png        — per-β_prefix CE vs p, panels by (γ, α).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "results" / "counter_decode" / "belief_invariance" / "summary.json"
OUT_DIR = REPO_ROOT / "results" / "counter_decode" / "belief_invariance"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run run_belief_invariance_test.py first")
    summary = json.loads(DATA.read_text())
    gammas = summary["config"]["gammas"]
    alphas = summary["config"]["alphas"]
    ps = summary["config"]["ps"]
    bps = summary["config"]["betas_prefix"]
    records = summary["records"]

    # F_bi_1: variance heatmap
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), squeeze=False)
    for ai, alpha in enumerate(alphas):
        Z = np.full((len(gammas), len(ps)), np.nan)
        for gi, g in enumerate(gammas):
            for pi, p in enumerate(ps):
                rec = next((r for r in records
                            if r["gamma"] == g and r["alpha"] == alpha and r["p"] == p), None)
                if rec is None:
                    continue
                Z[gi, pi] = rec["mean_var_across_prefix"]
        ax = axes[0, ai]
        im = ax.imshow(np.log10(Z + 1e-3), origin="lower", aspect="auto",
                       cmap="viridis_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels([f"{p}" for p in ps])
        ax.set_yticks(range(len(gammas)))
        ax.set_yticklabels([f"{g}" for g in gammas])
        ax.set_xlabel("correction exponent p")
        ax.set_ylabel("memory decay γ")
        ax.set_title(f"α = {alpha}")
        for gi in range(len(gammas)):
            for pi in range(len(ps)):
                v = Z[gi, pi]
                if not np.isnan(v):
                    ax.text(pi, gi, f"{v:.2f}", ha="center", va="center",
                            color="white" if np.log10(v + 1e-3) > 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, label="log10(var) of CE across β_prefix")
    fig.suptitle(
        "Belief-invariance test: per-prompt variance of CE on natural-text "
        "continuation across β_prefix ∈ {0.5, 1, 2}.\n"
        "Lower = better (corrector cancels prefix imprint). Cell label = raw "
        "variance in nats².",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_bi_1_var_heatmap.png", dpi=140)
    plt.close(fig)

    # F_bi_2: per-β_prefix CE vs p (panel per (γ, α)).
    fig, axes = plt.subplots(len(gammas), len(alphas),
                              figsize=(5 * len(alphas), 3.2 * len(gammas)),
                              squeeze=False, sharex=True)
    bp_colors = {0.5: "#aa3333", 1.0: "#666666", 2.0: "#117733"}
    for gi, g in enumerate(gammas):
        for ai, alpha in enumerate(alphas):
            ax = axes[gi, ai]
            for bp in bps:
                ys = []
                xs = []
                for p in ps:
                    rec = next((r for r in records
                                if r["gamma"] == g and r["alpha"] == alpha and r["p"] == p), None)
                    if rec is None:
                        continue
                    ce = rec["mean_ce_per_prefix"][str(bp)]
                    xs.append(p)
                    ys.append(ce)
                ax.plot(xs, ys, "o-",
                        color=bp_colors.get(bp, "#222"),
                        label=f"β_prefix={bp}", linewidth=1.6, markersize=6)
            ax.set_title(f"γ={g}, α={alpha}")
            ax.set_xlabel("correction exponent p")
            ax.set_ylabel("mean CE on natural-text continuation (nats)")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if gi == 0 and ai == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "CE on natural-text continuation by p, color = β_prefix.  Invariance "
        "across β_prefix at p=0 is preserved trivially (no correction);\n"
        "increasing p amplifies the divergence sharply for β_prefix=0.5 "
        "(prefix-induced filter bias overcorrects).",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_bi_2_ce_vs_p.png", dpi=140)
    plt.close(fig)

    print(f"[plot-bi] figures → {OUT_DIR}/F_bi_*.png")


if __name__ == "__main__":
    main()
