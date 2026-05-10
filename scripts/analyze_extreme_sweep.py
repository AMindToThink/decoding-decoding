"""Analyze the extreme β_target sweep.

After the user's observation that sample entropy is roughly invariant under
correction (Marie Curie at β=2 reads dull), this analysis investigates:

  1. Does the corrected β̂ converge to β_target regardless of β_target,
     making β_dec → 1 (corrector neutralizes itself)?
  2. How far can β_target be pushed before BOTH arms degenerate?

The extreme sweep covers β_target ∈ {0.05, 0.125, 0.25, 0.5, 1.0, 2.0,
4.0, 8.0, 16.0}, generated half on GPU 0 and half on GPU 1 into separate
data dirs that we merge here.

Outputs:
    results/counter_decode/extreme_sweep/
        figures/F0_extreme.png            — concentration drift over t
        figures/fixed_point.png           — β̂_late and β_dec_late vs β_target
        figures/degeneration.png          — distinct-trigrams over t per condition
        tables/extreme_summary.json
        tables/sample_texts.json          — picked example outputs

Usage:
    uv run python scripts/analyze_extreme_sweep.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from decoding_decoding.counter_decode_analyze import (
    f0_concentration,
    load_counter_condition,
)
from decoding_decoding.data_layout import MANIFEST_FILENAME


REPO_ROOT = Path(__file__).resolve().parent.parent
EXTREME_DIRS = [
    REPO_ROOT / "data" / "counter_decode_extreme" / "gpu0",
    REPO_ROOT / "data" / "counter_decode_extreme" / "gpu1",
]
RESULTS_DIR = REPO_ROOT / "results" / "counter_decode" / "extreme_sweep"


def _merged_load(beta_target: float, corrected: bool):
    """Load condition data from whichever GPU's dir contains it."""
    for d in EXTREME_DIRS:
        manifest = d / MANIFEST_FILENAME
        if not manifest.exists():
            continue
        m = pl.read_parquet(manifest)
        # Match on condition_label.
        arm = "corr" if corrected else "unco"
        label = f"btarget={beta_target:.2f}_arm={arm}"
        if (m.filter(pl.col("condition_label") == label).height) > 0:
            return load_counter_condition(d, beta_target=beta_target, corrected=corrected)
    raise FileNotFoundError(
        f"no traces for β_target={beta_target} corrected={corrected} across "
        f"{[str(d) for d in EXTREME_DIRS]}"
    )


def _decoded_for(beta_target: float, corrected: bool, prompt_id: int = 0, run_idx: int = 0) -> tuple[str, str]:
    """Return (prompt, decoded_text) for one sample trajectory."""
    arm = "corr" if corrected else "unco"
    label = f"btarget={beta_target:.2f}_arm={arm}"
    for d in EXTREME_DIRS:
        manifest = d / MANIFEST_FILENAME
        if not manifest.exists():
            continue
        m = pl.read_parquet(manifest)
        sub = m.filter(
            (pl.col("condition_label") == label)
            & (pl.col("prompt_id") == prompt_id)
            & (pl.col("run_idx") == run_idx)
        )
        if sub.height > 0:
            row = sub.row(0, named=True)
            return row["prompt_text"], row["decoded_text"]
    return "", ""


def distinct_trigrams_over_window(
    sampled_token_ids: np.ndarray,
    *,
    window: int = 50,
) -> np.ndarray:
    """For each step t, distinct trigrams in tokens [t-window..t] / window.

    sampled_token_ids: (P, R, T)
    returns: (P, R, T) float32 — value at t < window-1 is the same as at window-1.
    """
    P, R, T = sampled_token_ids.shape
    out = np.zeros((P, R, T), dtype=np.float32)
    for p in range(P):
        for r in range(R):
            seq = sampled_token_ids[p, r]
            for t in range(T):
                start = max(0, t - window + 1)
                slice_ = seq[start: t + 1]
                if len(slice_) < 3:
                    out[p, r, t] = 1.0
                    continue
                trigrams = [tuple(slice_[i:i+3]) for i in range(len(slice_) - 2)]
                if not trigrams:
                    out[p, r, t] = 1.0
                else:
                    out[p, r, t] = len(set(trigrams)) / len(trigrams)
    return out


def main() -> None:
    fig_dir = RESULTS_DIR / "figures"
    table_dir = RESULTS_DIR / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    beta_targets = [0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    summary: dict = {"beta_targets": beta_targets, "rows": []}
    sample_texts: dict = {}

    # Per (β_target, arm), load and compute headline stats.
    rows = []
    for bt in beta_targets:
        for corr in (True, False):
            try:
                cond = _merged_load(bt, corr)
            except FileNotFoundError as e:
                print(f"[warn] {e}")
                continue
            # Late-trajectory means.
            T = cond.beta_hat_pre.shape[-1]
            late = slice(max(0, T - 50), T)
            row = {
                "beta_target": bt,
                "corrected": corr,
                "beta_hat_t0": float(cond.beta_hat_pre[..., 0].mean()),
                "beta_hat_late": float(cond.beta_hat_pre[..., late].mean()),
                "beta_dec_late": float(cond.beta_dec[..., late].mean()),
                "max_p_sample_t0": float(cond.max_p_sample[..., 0].mean()),
                "max_p_sample_late": float(cond.max_p_sample[..., late].mean()),
                "entropy_sample_t0": float(cond.entropy_sample[..., 0].mean()),
                "entropy_sample_late": float(cond.entropy_sample[..., late].mean()),
                "max_p_phi_late": float(cond.max_p_phi[..., late].mean()),
                "entropy_phi_late": float(cond.entropy_phi[..., late].mean()),
            }
            rows.append(row)

            # Save a sample text snippet (prompt 0, run 0).
            prompt, decoded = _decoded_for(bt, corr)
            arm = "corr" if corr else "unco"
            sample_texts[f"β={bt}_arm={arm}"] = {
                "prompt": prompt[:160],
                "decoded": decoded[:600],
            }

    summary["rows"] = rows

    # ---- Plot 1: fixed-point demonstration ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bts = sorted(set(r["beta_target"] for r in rows if r["corrected"]))
    bh_late_corr = [next(r["beta_hat_late"] for r in rows if r["corrected"] and r["beta_target"] == bt) for bt in bts]
    bd_late_corr = [next(r["beta_dec_late"] for r in rows if r["corrected"] and r["beta_target"] == bt) for bt in bts]
    bh_late_unco = [next(r["beta_hat_late"] for r in rows if not r["corrected"] and r["beta_target"] == bt) for bt in bts]

    log_bt = np.log(bts)
    axes[0].plot(log_bt, np.log(bh_late_corr), "o-", color="#2d5d4f", label=r"corrected $\hat\beta_{\rm late}$")
    axes[0].plot(log_bt, np.log(bh_late_unco), "s-", color="#a8412a", label=r"uncorrected $\hat\beta_{\rm late}$")
    axes[0].plot(log_bt, log_bt, ":", color="#6b6453", label=r"$\hat\beta = \beta_{\rm target}$")
    axes[0].set_xlabel(r"$\log \beta_{\rm target}$")
    axes[0].set_ylabel(r"$\log \hat\beta_{\rm late}$")
    axes[0].set_title("β̂ late-trajectory asymptote vs β_target")
    axes[0].grid(True, linewidth=0.3, alpha=0.4)
    axes[0].legend(fontsize=9)

    axes[1].plot(log_bt, np.log(bd_late_corr), "o-", color="#2d5d4f", label=r"corrected $\beta_{\rm dec, late}$")
    axes[1].axhline(0.0, color="#6b6453", linestyle=":", label=r"$\beta_{\rm dec}=1$ (corrector neutralized)")
    axes[1].set_xlabel(r"$\log \beta_{\rm target}$")
    axes[1].set_ylabel(r"$\log \beta_{\rm dec, late}$")
    axes[1].set_title("Effective decoder β late vs β_target (corrected arm)")
    axes[1].grid(True, linewidth=0.3, alpha=0.4)
    axes[1].legend(fontsize=9)
    fig.suptitle("Fixed-point demonstration: corrector self-neutralizes at large t", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fixed_point.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: F0-style for extreme range ----
    n_rows = len(beta_targets)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.5 * n_rows), sharex=True, squeeze=False)
    for i, bt in enumerate(beta_targets):
        try:
            corr = _merged_load(bt, True)
            unco = _merged_load(bt, False)
        except FileNotFoundError:
            continue
        f0_corr = f0_concentration(corr, n_resamples=300)
        f0_unco = f0_concentration(unco, n_resamples=300)
        T = f0_corr.max_p_sample.point.shape[0]
        ts = np.arange(T)
        for j, (key, label) in enumerate(
            zip(["max_p_sample", "entropy_sample"],
                [r"max-prob $P_{\rm sample}$", r"entropy $P_{\rm sample}$ (nats)"])
        ):
            ax = axes[i, j]
            for f0, color, name in (
                (f0_corr, "#2d5d4f", "corrected"),
                (f0_unco, "#a8412a", "uncorrected"),
            ):
                summ = getattr(f0, key)
                ax.plot(ts, summ.point, color=color, linewidth=1.3, label=name)
                ax.fill_between(ts, summ.ci_low, summ.ci_high, color=color, alpha=0.15)
            if j == 0:
                ax.set_ylabel(rf"$\beta_{{\rm target}}={bt}$", fontsize=10)
            if i == 0:
                ax.set_title(label, fontsize=10)
                ax.legend(fontsize=8, loc="best")
            if i == n_rows - 1:
                ax.set_xlabel(r"$t$")
            ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.suptitle("F0 extreme — concentration drift across 9 β_target values (corrected vs uncorrected)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "F0_extreme.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: distinct-trigram diversity over t (degeneration metric) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    cmap = plt.get_cmap("inferno")
    n = len(beta_targets)
    for i, bt in enumerate(beta_targets):
        try:
            corr = _merged_load(bt, True)
            unco = _merged_load(bt, False)
        except FileNotFoundError:
            continue
        d_corr = distinct_trigrams_over_window(corr.sampled_token_ids, window=40)
        d_unco = distinct_trigrams_over_window(unco.sampled_token_ids, window=40)
        c = cmap(0.15 + 0.65 * i / max(1, n - 1))
        ax.plot(d_corr.mean(axis=(0, 1)), color=c, linewidth=1.4, label=rf"corr β_t={bt}")
        ax.plot(d_unco.mean(axis=(0, 1)), color=c, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("distinct-trigram fraction (window=40)")
    ax.set_title("Degeneration metric: lower → more repetition. Solid=corrected, Dashed=uncorrected")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "degeneration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    (table_dir / "extreme_summary.json").write_text(json.dumps(summary, indent=2))
    (table_dir / "sample_texts.json").write_text(json.dumps(sample_texts, indent=2))

    print(f"[extreme] wrote {len(rows)} rows; figures and tables in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
