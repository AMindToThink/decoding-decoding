"""Analyze the (σ_0, γ) × β_target sweep for stable distinct entropy.

Reads:
    data/counter_decode_entropy_stability/s*_g*/...

Produces:
    results/counter_decode/entropy_stability/
        figures/entropy_curves.png       — entropy_sample(t) per (σ_0, γ),
                                           one panel per (σ_0, γ), curves per β_target
        figures/entropy_vs_btarget.png   — late entropy vs β_target, per (σ_0, γ)
        figures/stability_landscape.png  — for each (σ_0, γ), |entropy_slope| and
                                           range(entropy across β_target) — find sweet spot
        tables/entropy_stability_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from decoding_decoding.data_layout import MANIFEST_FILENAME


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_BASE = REPO_ROOT / "data" / "counter_decode_entropy_stability"
RESULTS_DIR = REPO_ROOT / "results" / "counter_decode" / "entropy_stability"

SIGMA_VALUES = (0.1, 0.15, 0.2, 0.3, 0.5)
GAMMA_VALUES = (1.0, 0.97, 0.9)
BETA_TARGETS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _cell_dir(sigma_0: float, gamma: float) -> Path:
    return DATA_BASE / f"s{sigma_0}_g{gamma}".replace(".", "p")


def _load_traces(out_dir: Path, beta_target: float) -> Optional[dict]:
    manifest_path = out_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    m = pl.read_parquet(manifest_path)
    label = f"btarget={beta_target:.2f}_arm=corr"
    sub = m.filter(pl.col("condition_label") == label).sort(["prompt_id", "run_idx"])
    if sub.height == 0:
        return None
    bh, bd, mps, ent_s = [], [], [], []
    for row in sub.iter_rows(named=True):
        npz = np.load(out_dir / row["trace_path"])
        bh.append(npz["beta_hat_pre"])
        bd.append(npz["beta_dec"])
        mps.append(npz["max_p_sample"])
        ent_s.append(npz["entropy_sample"])
    return {
        "beta_hat_pre": np.stack(bh),
        "beta_dec": np.stack(bd),
        "max_p_sample": np.stack(mps),
        "entropy_sample": np.stack(ent_s),
    }


def _entropy_slope(ent: np.ndarray) -> float:
    T = ent.shape[-1]
    half = T // 2
    tail = ent[..., half:]
    t_axis = np.arange(tail.shape[-1])
    flat = tail.reshape(-1, tail.shape[-1])
    slopes = []
    for row in flat:
        m, _ = np.polyfit(t_axis, row, 1)
        slopes.append(m)
    return float(np.mean(slopes))


def collect_summary() -> Dict:
    """For each (σ_0, γ), gather per-β_target late stats and full trajectories."""
    out = {"records": [], "per_t": {}}
    for sigma_0 in SIGMA_VALUES:
        for gamma in GAMMA_VALUES:
            cell = _cell_dir(sigma_0, gamma)
            for bt in BETA_TARGETS:
                tr = _load_traces(cell, bt)
                if tr is None:
                    continue
                T = tr["entropy_sample"].shape[-1]
                late = slice(T - 30, T)
                rec = {
                    "sigma_0": sigma_0,
                    "gamma": gamma,
                    "beta_target": bt,
                    "entropy_sample_late": float(np.mean(tr["entropy_sample"][..., late])),
                    "max_p_sample_late": float(np.mean(tr["max_p_sample"][..., late])),
                    "beta_hat_late": float(np.mean(tr["beta_hat_pre"][..., late])),
                    "beta_dec_late": float(np.mean(tr["beta_dec"][..., late])),
                    "entropy_slope_2nd_half": _entropy_slope(tr["entropy_sample"]),
                }
                out["records"].append(rec)
                out["per_t"][f"s={sigma_0}_g={gamma}_b={bt}"] = {
                    "entropy_sample": tr["entropy_sample"].mean(axis=0).tolist(),
                    "beta_hat": tr["beta_hat_pre"].mean(axis=0).tolist(),
                    "beta_dec": tr["beta_dec"].mean(axis=0).tolist(),
                    "max_p_sample": tr["max_p_sample"].mean(axis=0).tolist(),
                }
    return out


def plot_entropy_curves(summary: Dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    nS, nG = len(SIGMA_VALUES), len(GAMMA_VALUES)
    fig, axes = plt.subplots(nG, nS, figsize=(3.2 * nS, 2.5 * nG), sharex=True, sharey=True)
    cmap = plt.cm.coolwarm
    for j, sigma_0 in enumerate(SIGMA_VALUES):
        for i, gamma in enumerate(GAMMA_VALUES):
            ax = axes[i, j]
            for bt in BETA_TARGETS:
                key = f"s={sigma_0}_g={gamma}_b={bt}"
                d = summary["per_t"].get(key)
                if d is None:
                    continue
                T = len(d["entropy_sample"])
                t = np.arange(T)
                color = cmap(np.log(bt) / (2 * np.log(BETA_TARGETS[-1])) + 0.5)
                ax.plot(t, d["entropy_sample"], color=color, lw=1.0, label=f"β={bt}")
            ax.set_title(f"σ₀={sigma_0}, γ={gamma}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == nG - 1:
                ax.set_xlabel("token t")
            if j == 0:
                ax.set_ylabel("entropy(P_sample)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Sample entropy trajectories by (σ_0, γ); curves = β_target",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_vs_btarget(summary: Dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(GAMMA_VALUES), figsize=(5 * len(GAMMA_VALUES), 4), sharey=True)
    cmap = plt.cm.viridis
    for i, gamma in enumerate(GAMMA_VALUES):
        ax = axes[i]
        for j, sigma_0 in enumerate(SIGMA_VALUES):
            color = cmap(j / max(1, len(SIGMA_VALUES) - 1))
            xs, ys = [], []
            for bt in BETA_TARGETS:
                rec = next((r for r in summary["records"]
                            if r["sigma_0"] == sigma_0 and r["gamma"] == gamma and r["beta_target"] == bt), None)
                if rec is None:
                    continue
                xs.append(bt)
                ys.append(rec["entropy_sample_late"])
            ax.plot(xs, ys, "o-", color=color, label=f"σ₀={sigma_0}", lw=1.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta_{target}$ (log axis)")
        if i == 0:
            ax.set_ylabel("late entropy(P_sample)")
        ax.set_title(f"γ = {gamma}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Late sample entropy vs β_target — looking for monotone non-degenerate curves",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_vs_btarget.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_stability_landscape(summary: Dict, fig_dir: Path) -> None:
    """Per (σ_0, γ): mean |entropy_slope| (across β_target) and entropy range
    (max - min across β_target). Sweet spot = high range, low slope."""
    nS, nG = len(SIGMA_VALUES), len(GAMMA_VALUES)
    slope_grid = np.full((nG, nS), np.nan)
    range_grid = np.full((nG, nS), np.nan)
    nondeg_grid = np.full((nG, nS), np.nan)  # min over β_target of (1 - max_p_sample_late)
    for i, gamma in enumerate(GAMMA_VALUES):
        for j, sigma_0 in enumerate(SIGMA_VALUES):
            ents, slopes, maxp = [], [], []
            for bt in BETA_TARGETS:
                rec = next((r for r in summary["records"]
                            if r["sigma_0"] == sigma_0 and r["gamma"] == gamma and r["beta_target"] == bt), None)
                if rec is None:
                    continue
                ents.append(rec["entropy_sample_late"])
                slopes.append(abs(rec["entropy_slope_2nd_half"]))
                maxp.append(rec["max_p_sample_late"])
            if not ents:
                continue
            range_grid[i, j] = max(ents) - min(ents)
            slope_grid[i, j] = float(np.mean(slopes))
            nondeg_grid[i, j] = 1.0 - max(maxp)  # bigger = less degenerate

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, grid, title, cmap_name in zip(
        axes,
        [range_grid, slope_grid, nondeg_grid],
        ["entropy range across β_target (bigger = better)",
         "mean |entropy slope| (smaller = more stable)",
         "min (1 − max P_sample) (bigger = less degenerate)"],
        ["viridis", "magma_r", "viridis"],
    ):
        im = ax.imshow(grid, aspect="auto", cmap=cmap_name)
        ax.set_xticks(range(nS), [str(s) for s in SIGMA_VALUES])
        ax.set_yticks(range(nG), [str(g) for g in GAMMA_VALUES])
        ax.set_xlabel(r"$\sigma_0$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title, fontsize=10)
        for i in range(nG):
            for j in range(nS):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2g}", ha="center", va="center",
                            color="white" if cmap_name == "magma_r" else "black",
                            fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("Stability landscape: where is entropy distinct, stable, and non-degenerate?",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "stability_landscape.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_dir = RESULTS_DIR / "figures"
    table_dir = RESULTS_DIR / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summary()
    if not summary["records"]:
        raise SystemExit("No data — sweep not finished?")

    with (table_dir / "entropy_stability_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_entropy_curves(summary, fig_dir)
    plot_entropy_vs_btarget(summary, fig_dir)
    plot_stability_landscape(summary, fig_dir)

    # Print compact summary.
    print(f"{'σ_0':>5} {'γ':>5} {'β':>5} {'ent':>7} {'maxP':>6} "
          f"{'β̂':>6} {'βdec':>6} {'slope':>9}")
    for r in summary["records"]:
        print(f"{r['sigma_0']:>5.2f} {r['gamma']:>5.2f} {r['beta_target']:>5.2f} "
              f"{r['entropy_sample_late']:>7.3f} {r['max_p_sample_late']:>6.3f} "
              f"{r['beta_hat_late']:>6.3f} {r['beta_dec_late']:>6.3f} "
              f"{r['entropy_slope_2nd_half']:>9.5f}")


if __name__ == "__main__":
    main()
