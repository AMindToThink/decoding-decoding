"""Plot Exp A figures from data/calibration_baseline/results.npz.

Produces three figures in results/counter_decode/calibration_baseline/:
  F_cal_1_beta_estimators.png  — per-corpus β̂ from each estimator
  F_cal_2_nll_curve.png        — NLL surface vs β with each estimator marked
  F_cal_3_test_ppl.png         — per-passage test PPL distribution per β setting

Usage: uv run python scripts/plot_calibration_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "calibration_baseline" / "results.npz"
OUT_DIR = REPO_ROOT / "results" / "counter_decode" / "calibration_baseline"


def _plot_beta_estimators(npz: np.lib.npyio.NpzFile) -> None:
    """Per-corpus β̂ values from each estimator."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = [
        ("WikiText", "stream median (per-passage)", float(npz["wt_log_beta_streaming_median"])),
        ("WikiText", "stream pooled", float(npz["wt_log_beta_streaming_pooled"])),
        ("WikiText", "Guo MLE",     float(npz["wt_log_beta_mle"])),
        ("WritingPrompts", "stream median (per-passage)", float(npz["wp_log_beta_streaming_median"])),
        ("WritingPrompts", "stream pooled", float(npz["wp_log_beta_streaming_pooled"])),
        ("WritingPrompts", "Guo MLE",     float(npz["wp_log_beta_mle"])),
    ]
    labels = [f"{c}\n{e}" for c, e, _ in rows]
    vals = [v for _, _, v in rows]
    colors = ["#4477aa", "#aa3333", "#117733"] * 2
    ax.barh(range(len(rows)), vals, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("log β̂   (log β̂ = 0 ⇒ β̂ = 1)")
    ax.set_title("Calibration estimators: streaming filter vs Guo batch MLE")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_cal_1_beta_estimators.png", dpi=140)
    plt.close(fig)


def _plot_nll_curve(npz: np.lib.npyio.NpzFile) -> None:
    """NLL surface vs log β on the calibration set, with each estimator marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, prefix, title in (
        (axes[0], "wt", "WikiText calibration set"),
        (axes[1], "wp", "WritingPrompts calibration set"),
    ):
        grid = npz[f"{prefix}_log_beta_grid"]
        curve = npz[f"{prefix}_nll_curve"]
        ax.plot(grid, curve, color="#222", linewidth=1.8, label="per-token NLL")
        # Mark each estimator's location.
        markers = [
            (npz[f"{prefix}_log_beta_streaming_median"], "#4477aa",
             "stream median"),
            (npz[f"{prefix}_log_beta_streaming_pooled"], "#aa3333",
             "stream pooled"),
            (npz[f"{prefix}_log_beta_mle"], "#117733", "Guo MLE"),
        ]
        for lb, color, label in markers:
            lb = float(lb)
            if grid.min() <= lb <= grid.max():
                yv = float(np.interp(lb, grid, curve))
                ax.axvline(lb, color=color, linewidth=1.0, linestyle="--", alpha=0.7)
                ax.plot(lb, yv, "o", color=color, label=f"{label}: log β̂={lb:+.3f}")
            else:
                ax.axvline(lb, color=color, linewidth=1.0, linestyle=":", alpha=0.5,
                           label=f"{label}: log β̂={lb:+.3f} (off-grid)")
        ax.axvline(0.0, color="grey", linewidth=0.6, linestyle="-")
        ax.set_xlabel("log β")
        ax.set_ylabel("per-token NLL (nats)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Where each estimator sits on the calibration NLL surface\n"
        "(the surface is nearly flat near β=1 — small gradient, but small curvature too, "
        "so one-step Newton overshoots)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_cal_2_nll_curve.png", dpi=140)
    plt.close(fig)


def _plot_test_ppl(npz: np.lib.npyio.NpzFile) -> None:
    """Per-passage test PPL at each β setting, boxplot per corpus."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, prefix, title in (
        (axes[0], "wt", "WikiText test PPL per passage"),
        (axes[1], "wp", "WritingPrompts test PPL per passage"),
    ):
        labels = list(npz[f"{prefix}_test_beta_labels"])
        total_nll = npz[f"{prefix}_test_per_passage_total_nll"]      # (n_betas, n_test)
        n_tokens = npz[f"{prefix}_test_per_passage_n_tokens"]
        per_token_nll = total_nll / n_tokens[None, :]
        per_token_ppl = np.exp(per_token_nll)
        positions = np.arange(len(labels))
        bp = ax.boxplot(
            [per_token_ppl[i] for i in range(len(labels))],
            positions=positions, showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], ["#cccccc", "#4477aa", "#aa3333", "#117733", "#ddcc77"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("per-token PPL")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", alpha=0.3)
    fig.suptitle(
        "Per-passage test PPL across β settings (log y-axis).  "
        "Pooled streaming β̂ is catastrophic; MLE β̂ ≈ no change",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_cal_3_test_ppl.png", dpi=140)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run run_calibration_baseline.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz = np.load(DATA, allow_pickle=True)
    _plot_beta_estimators(npz)
    _plot_nll_curve(npz)
    _plot_test_ppl(npz)
    print(f"[plot-cal] figures → {OUT_DIR}/F_cal_*.png")


if __name__ == "__main__":
    main()
