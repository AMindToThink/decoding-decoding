"""Plot Exp B figures from data/sliding_filter_mixed/results.npz.

Figures:
  F_mix_1_log_beta_trajectory.png  — log β̂_t vs t, with chunk boundaries marked
  F_mix_2_gamma_comparison.png     — per-chunk mean log β̂ vs chunk index, by γ
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "sliding_filter_mixed" / "results.npz"
OUT_DIR = REPO_ROOT / "results" / "counter_decode" / "sliding_filter_mixed"


GAMMAS = [0.90, 0.95, 0.98, 0.99, 1.00]


def _plot_trajectory(npz: np.lib.npyio.NpzFile) -> None:
    chunk_tokens = int(npz["chunk_tokens"])
    warmup = int(npz["warmup"])
    boundaries_start = list(npz["boundaries_start"])
    boundaries_genre = list(npz["boundaries_genre"])

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"0.90": "#aa3333", "0.95": "#bb6622", "0.98": "#117733",
              "0.99": "#226699", "1.00": "#666666"}
    for gamma in GAMMAS:
        key = f"gamma_{gamma:.2f}".replace(".", "_")
        traj = npz[f"{key}_log_beta_hat"]
        # x-axis: filter step → sequence position = warmup + k.
        x = np.arange(traj.shape[0]) + warmup
        c = colors.get(f"{gamma:.2f}", "#222")
        ax.plot(x, traj, color=c, linewidth=1.1, label=f"γ = {gamma}", alpha=0.85)

    # Shade chunks by genre.
    for i, (start, genre) in enumerate(zip(boundaries_start, boundaries_genre)):
        end = start + chunk_tokens
        color = "#ddeedd" if str(genre) == "wikitext" else "#fde9c8"
        ax.axvspan(start, end, color=color, alpha=0.45, zorder=0)
    # Vertical lines at every boundary.
    for start in boundaries_start[1:]:
        ax.axvline(start, color="grey", linewidth=0.6, linestyle="--")

    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("sequence position  t")
    ax.set_ylabel("log β̂_t")
    ax.set_title(
        "Sliding β̂_t over alternating wiki / WritingPrompts chunks  "
        "(green = wiki, orange = WP)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_mix_1_log_beta_trajectory.png", dpi=140)
    plt.close(fig)


def _plot_gamma_comparison(npz: np.lib.npyio.NpzFile) -> None:
    import json
    summary = json.loads(
        (OUT_DIR / "summary.json").read_text()
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for gamma in GAMMAS:
        s = summary["per_gamma"][f"{gamma}"]
        per_chunk = s["per_chunk_log_beta"]
        per_genre = s["per_chunk_genre"]
        positions = np.arange(len(per_chunk))
        # WT chunks circle, WP chunks square.
        for x, v, g in zip(positions, per_chunk, per_genre):
            marker = "o" if g == "wikitext" else "s"
            color = "#117733" if g == "wikitext" else "#aa6622"
            ax.scatter(x, v, marker=marker, color=color, s=40,
                       edgecolor="black", linewidth=0.5)
        ax.plot(positions, per_chunk, color={
            "0.90": "#aa3333", "0.95": "#bb6622", "0.98": "#117733",
            "0.99": "#226699", "1.00": "#666666"
        }[f"{gamma:.2f}"], linewidth=1.0, label=f"γ={gamma}", alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("chunk index in mixed sequence")
    ax.set_ylabel("mean log β̂  (within 20-token boundary-stripped middle)")
    ax.set_title("Per-chunk filter signal by γ.  Green ● = wiki, orange ■ = WP.")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_mix_2_gamma_comparison.png", dpi=140)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run run_sliding_filter_mixed.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz = np.load(DATA, allow_pickle=True)
    _plot_trajectory(npz)
    _plot_gamma_comparison(npz)
    print(f"[plot-mix] figures → {OUT_DIR}/F_mix_*.png")


if __name__ == "__main__":
    main()
