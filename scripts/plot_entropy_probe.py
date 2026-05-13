"""Plot Exp C figures from data/entropy_probe/results.npz.

Produces:
  F_ep_1_entropy_trajectory.png  — median H(t) per β_dec over generation steps
  F_ep_2_paired_late_entropy.png  — per-prompt paired late-window H
  F_ep_3_sample_generations.png   — first ~30 tokens of one sample per β_dec
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoding_decoding.natural_text_filter import load_model_and_tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "entropy_probe" / "results.npz"
OUT_DIR = REPO_ROOT / "results" / "counter_decode" / "entropy_probe"


# Natural-text H benchmarks (from results/counter_decode/natural_text/macros.tex).
NATURAL_TEXT_H = {
    "WikiText / self_gen (β=1)": 2.691,
    "WikiText / natural":         1.983,
    "WritingPrompts / self_gen (β=1)": 2.939,
    "WritingPrompts / natural":   2.847,
}


def _key(beta_dec: float) -> str:
    return f"beta_{beta_dec:.2f}".replace(".", "_")


def _plot_trajectory(npz: np.lib.npyio.NpzFile) -> None:
    beta_vals = list(npz["beta_dec_values"])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = {"0.5": "#aa3333", "1.0": "#666666", "2.0": "#117733"}
    for b in beta_vals:
        H = npz[f"{_key(b)}_entropies"]    # (n_prompts, n_steps)
        med = np.median(H, axis=0)
        lo, hi = np.percentile(H, [25, 75], axis=0)
        c = colors.get(f"{b}", "#222")
        ax.plot(med, color=c, linewidth=1.8, label=f"β_dec = {b}  (T = {1/b:.2f})")
        ax.fill_between(np.arange(med.shape[0]), lo, hi, color=c, alpha=0.18)
    # Reference lines for natural text entropies.
    for name, h in NATURAL_TEXT_H.items():
        ax.axhline(h, color="grey", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.text(med.shape[0] * 0.99, h + 0.05, name, color="grey",
                fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("generation step  k")
    ax.set_ylabel("pre-decoding model entropy  H(softmax(ℓ_t))  (nats)")
    ax.set_title("Model's pre-decoding entropy responds to its own decoder β")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ep_1_entropy_trajectory.png", dpi=140)
    plt.close(fig)


def _plot_paired_late(npz: np.lib.npyio.NpzFile) -> None:
    late_start = int(npz["late_window_start"])
    H_low = npz[f"{_key(0.5)}_entropies"][:, late_start:].mean(axis=1)
    H_one = npz[f"{_key(1.0)}_entropies"][:, late_start:].mean(axis=1)
    H_high = npz[f"{_key(2.0)}_entropies"][:, late_start:].mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(H_low))
    ax.scatter(positions, H_low, color="#aa3333", s=24, label="β_dec=0.5 (T=2.0)")
    ax.scatter(positions, H_one, color="#666666", s=24, label="β_dec=1.0 (T=1.0)")
    ax.scatter(positions, H_high, color="#117733", s=24, label="β_dec=2.0 (T=0.5)")
    # Connect each prompt with a thin gray line.
    for i in range(len(H_low)):
        ax.plot([i, i, i], [H_low[i], H_one[i], H_high[i]],
                color="grey", alpha=0.3, linewidth=0.7)
    ax.set_xlabel("prompt index")
    ax.set_ylabel(f"mean H over steps [{late_start}, end)  (nats)")
    ax.set_title("Per-prompt late-window entropy: every prompt shows the ordering")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ep_2_paired_late_entropy.png", dpi=140)
    plt.close(fig)


def _plot_sample_generations(npz: np.lib.npyio.NpzFile) -> None:
    """Decode and show first ~80 tokens of one sample per β_dec — for the
    garbage-in-garbage-out audit."""
    from decoding_decoding.prompts import PROMPTS

    model, tok = load_model_and_tokenizer()
    sample_prompt_idx = 0   # use the first prompt for visualization

    fig, axes = plt.subplots(3, 1, figsize=(11, 4.5))
    for ax, b, color in zip(
        axes,
        [0.5, 1.0, 2.0],
        ["#aa3333", "#666666", "#117733"],
    ):
        sampled = npz[f"{_key(b)}_sampled_token_ids"][sample_prompt_idx, :60]
        decoded = tok.decode(sampled.tolist(), skip_special_tokens=False)
        # Replace newlines + tabs so we can see them.
        decoded = decoded.replace("\n", "\\n ").replace("\t", "\\t ")
        ax.text(
            0.01, 0.5,
            f"β_dec = {b} (T = {1/b:.2f}):\n«{PROMPTS[sample_prompt_idx]}» " + decoded[:600],
            fontsize=8, family="monospace", va="center",
            wrap=True, color="black",
        )
        ax.set_facecolor((1.0, 1.0, 1.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color(color)
        ax.spines["left"].set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Sample generations (prompt 0, first 60 tokens) at each β_dec  —  "
        "audit for garbage-in-garbage-out",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ep_3_sample_generations.png", dpi=140)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run run_entropy_probe.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz = np.load(DATA, allow_pickle=True)
    _plot_trajectory(npz)
    _plot_paired_late(npz)
    _plot_sample_generations(npz)
    print(f"[plot-ep] figures → {OUT_DIR}/F_ep_*.png")


if __name__ == "__main__":
    main()
