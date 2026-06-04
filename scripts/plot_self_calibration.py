"""Figures for the self-calibration headline (Claim 1, Part 1).

Reads results/counter_decode/self_calibration/{recovery.json, recovery.npz}
(written by run_self_calibration.py --part recovery) and emits:

  F_selfcal_recovery.png       — mean β̂_t trajectories, faithful vs original, with the
                                 known target log(1/T) dashed (recovery; paper/appendix).
  F_selfcal_recovery_talk.png  — single clean panel (filter β̂_t → log(1/T)); for the
                                 talk, which does not introduce the faithful/original split.
  F_selfcal_nll_gain.png       — per-T NLL improvement over β=1 (nats), adaptive β̂_t vs
                                 the oracle 1/T (the filter captures nearly all the gain).

Usage: uv run python scripts/plot_self_calibration.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = REPO_ROOT / "results" / "counter_decode" / "self_calibration"
NPZ = REPO_ROOT / "data" / "self_calibration" / "recovery.npz"


def main() -> None:
    summary = json.loads((BASE / "recovery.json").read_text())
    raw = np.load(NPZ, allow_pickle=True)
    by_T = summary["by_temp"]
    Ts = sorted((float(k) for k in by_T), key=float)
    warmup = summary["config"]["warmup"]
    cmap = plt.get_cmap("viridis")
    colors = {T: cmap(i / max(1, len(Ts) - 1)) for i, T in enumerate(Ts)}

    # --- Figure 1: recovery trajectories (faithful vs original) ---
    fig, (axf, axo) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for T in Ts:
        target = math.log(1.0 / T)
        lbf = raw[f"lb_faith_T{T}"].mean(axis=0)       # (gen_len,)
        lbo = raw[f"lb_orig_T{T}"].mean(axis=0)
        x = np.arange(lbf.shape[0])
        axf.plot(x, lbf, color=colors[T], label=f"T={T}")
        axf.axhline(target, color=colors[T], ls=":", lw=1, alpha=0.7)
        axo.plot(x, lbo, color=colors[T])
        axo.axhline(target, color=colors[T], ls=":", lw=1, alpha=0.7)
    for ax, ttl in ((axf, "faithful (β-corrected)"), (axo, "original (β=1 linearization)")):
        ax.axvline(warmup, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("generated position")
        ax.set_title(ttl)
        ax.grid(alpha=0.2)
    axf.set_ylabel(r"$\log\hat\beta_t$  (dotted = true $\log(1/T)$)")
    axf.legend(fontsize=8, ncol=2, loc="best")
    axo.set_ylim(axf.get_ylim()[0] - 1, axo.get_ylim()[1])  # show divergence
    fig.suptitle("Recovering a known decoder temperature: faithful converges to the truth, original does not")
    fig.tight_layout()
    out1 = BASE / "F_selfcal_recovery.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 1b (talk): single clean panel, no faithful/original framing ---
    # The talk never introduces the β=1-linearization bug, so it shows only the
    # working filter recovering the truth. Same data as the left panel above.
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for T in Ts:
        target = math.log(1.0 / T)
        lbf = raw[f"lb_faith_T{T}"].mean(axis=0)
        x = np.arange(lbf.shape[0])
        ax.plot(x, lbf, color=colors[T], lw=1.8, label=f"T={T}")
        ax.axhline(target, color=colors[T], ls=":", lw=1, alpha=0.7)
    ax.axvline(warmup, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("generated position")
    ax.set_ylabel(r"estimated $\log\hat\beta_t$   (dotted = true $\log(1/T)$)")
    ax.set_title("The filter reads the decoder temperature off the text")
    ax.legend(fontsize=9, ncol=2, loc="best", title="generated at")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out1b = BASE / "F_selfcal_recovery_talk.png"
    fig.savefig(out1b, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-T NLL gain over β=1 (adaptive β̂_t vs oracle) ---
    faith_gain = [by_T[f"{T}"]["prediction_nll_nats"]["beta1_mean"]
                  - by_T[f"{T}"]["prediction_nll_nats"]["adaptive_faithful_mean"] for T in Ts]
    oracle_gain = [by_T[f"{T}"]["prediction_nll_nats"]["beta1_mean"]
                   - by_T[f"{T}"]["prediction_nll_nats"]["oracle_1overT_mean"] for T in Ts]
    coherent = [by_T[f"{T}"]["diagnostics"]["gen_dist_entropy_nats"] < 3.0 for T in Ts]
    x = np.arange(len(Ts))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.2, oracle_gain, width=0.4, label="oracle (known $1/T$)", color="#bbbbbb")
    ax.bar(x + 0.2, faith_gain, width=0.4, label="adaptive $\\hat\\beta_t$ (filter)", color="#1f77b4")
    ax.axhline(0, color="k", lw=0.8)
    for i, (fg, coh) in enumerate(zip(faith_gain, coherent)):
        ax.annotate("coherent" if coh else "near-random", (i, max(fg, 0)),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7,
                    color="green" if coh else "darkred")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={T}" for T in Ts])
    ax.set_ylabel("NLL improvement over $\\beta{=}1$ (nats)")
    ax.set_title("Causal per-token temperature beats the default model, nearly matching the oracle")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out2 = BASE / "F_selfcal_nll_gain.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out1}")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
