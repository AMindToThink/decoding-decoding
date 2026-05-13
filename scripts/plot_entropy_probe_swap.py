"""Plot Exp C swap-prefix control from data/entropy_probe_swap/results.npz.

Two figures:
  F_ep_4_swap_trajectory.png  — median H(t) per switch condition, with switch step marked
  F_ep_5_swap_recovery.png    — median H in step-offsets-from-switch windows
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "entropy_probe_swap" / "results.npz"
OUT_DIR = REPO_ROOT / "results" / "counter_decode" / "entropy_probe"


CONDITIONS = [
    ("pre05_post05", 0.5, 0.5, "#cc6666", "no-switch β=0.5"),
    ("pre05_post10", 0.5, 1.0, "#cc9933", "0.5 → 1.0 (test: does β=1 recover model?)"),
    ("pre05_post20", 0.5, 2.0, "#117733", "0.5 → 2.0"),
    ("pre20_post05", 2.0, 0.5, "#3366aa", "2.0 → 0.5 (mirror)"),
    ("pre20_post20", 2.0, 2.0, "#666666", "no-switch β=2.0"),
]


def _plot_trajectory(npz: np.lib.npyio.NpzFile) -> None:
    switch_step = int(npz["switch_step"])
    n_steps = int(npz["n_steps"])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for label, _bp, _bs, color, desc in CONDITIONS:
        H = npz[f"{label}_entropies"]
        med = np.median(H, axis=0)
        lo, hi = np.percentile(H, [25, 75], axis=0)
        ax.plot(med, color=color, linewidth=1.7, label=desc)
        ax.fill_between(np.arange(n_steps), lo, hi, color=color, alpha=0.13)
    ax.axvline(switch_step, color="black", linewidth=1.2, linestyle="--",
               label=f"sampler switch at step {switch_step}")
    ax.set_xlabel("generation step  k")
    ax.set_ylabel("pre-decoding model entropy  H(softmax(ℓ_t))  (nats)")
    ax.set_title(
        "Swap-prefix control: does switching the sampler β mid-generation "
        "recover model coherence?"
    )
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ep_4_swap_trajectory.png", dpi=140)
    plt.close(fig)


def _plot_recovery(npz: np.lib.npyio.NpzFile) -> None:
    """Show median entropy at offsets +0, +5, +10, +20, +50 from the switch."""
    switch_step = int(npz["switch_step"])
    offsets = [0, 5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, _bp, _bs, color, desc in CONDITIONS:
        H = npz[f"{label}_entropies"]   # (n, n_steps)
        post = H[:, switch_step:]
        med_at = [float(np.median(post[:, o])) for o in offsets]
        ax.plot(offsets, med_at, "o-", color=color, linewidth=1.7, label=desc, markersize=6)
    ax.set_xlabel("step offset after sampler switch")
    ax.set_ylabel("median H across prompts (nats)")
    ax.set_title(
        "Post-switch entropy by offset.\n"
        "If the model 'believed its decoder' and updated quickly, β=0.5→1.0 should fall to ~2 nats. It does not."
    )
    ax.set_xticks(offsets)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F_ep_5_swap_recovery.png", dpi=140)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run run_entropy_probe_swap.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz = np.load(DATA, allow_pickle=True)
    _plot_trajectory(npz)
    _plot_recovery(npz)
    print(f"[plot-swap] figures → {OUT_DIR}/F_ep_4*.png and F_ep_5*.png")


if __name__ == "__main__":
    main()
