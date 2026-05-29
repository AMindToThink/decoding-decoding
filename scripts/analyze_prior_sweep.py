"""Analyze the multi-prior and fuzzy-window sweeps.

Reads:
    data/counter_decode_priors/prior_*/...
    data/counter_decode_fuzzy/a*_g*/...

Produces:
    results/counter_decode/prior_sweep/
        figures/prior_beta_hat_traj.png    — β̂ trajectory per prior
        figures/prior_max_p_sample.png     — max P_sample late vs prior, per β_target
        figures/prior_pmf_visual.png       — visualization of each tested prior
        figures/fuzzy_window_traj.png      — β̂ trajectory per (α, γ) config
        tables/prior_summary.json
        tables/fuzzy_summary.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as sps

from decoding_decoding.data_layout import MANIFEST_FILENAME, decode_params


REPO_ROOT = Path(__file__).resolve().parent.parent
PRIOR_BASE = REPO_ROOT / "data" / "counter_decode_priors"
FUZZY_BASE = REPO_ROOT / "data" / "counter_decode_fuzzy"
RESULTS_DIR = REPO_ROOT / "results" / "counter_decode" / "prior_sweep"


PRIORS_TESTED = (
    "lognormal_laplace",
    "lognormal",
    "exponential",
    "gamma",
    "invgamma",
    "halfcauchy_logβ",
    "cauchy_logβ",
    "uniform_logβ",
)


def _condition_dir_for_prior(prior: str) -> Path:
    return PRIOR_BASE / f"prior_{prior.replace('_logβ', '_logbeta')}"


def _load_traces_for(out_dir: Path, beta_target: float, corrected: bool = True):
    """Load all per-step arrays for one (β_target, corrected) condition under
    a single experiment dir."""
    m = pl.read_parquet(out_dir / MANIFEST_FILENAME)
    arm = "corr" if corrected else "unco"
    label = f"btarget={beta_target:.2f}_arm={arm}"
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


def plot_priors_visual(fig_dir: Path) -> None:
    """Visualize the prior density on β for each tested prior (analytical)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    beta = np.exp(np.linspace(-4, 4, 1000))

    rows = []
    rows.append(("lognormal (σ=0.5, μ=-σ²/2)", sps.lognorm(s=0.5, scale=np.exp(-0.125)).pdf(beta)))
    rows.append(("exponential (rate=1)", sps.expon(scale=1.0).pdf(beta)))
    rows.append(("gamma(α=2, rate=2)", sps.gamma(a=2, scale=0.5).pdf(beta)))
    rows.append(("invgamma(α=3, scale=2)", sps.invgamma(a=3, scale=2).pdf(beta)))
    # halfcauchy on log β: density on β = halfcauchy(log β) / β, support log β ≥ 0.
    log_beta = np.log(beta)
    hc_density = np.where(log_beta >= 0, sps.halfcauchy(scale=1).pdf(np.maximum(log_beta, 0)) / np.maximum(beta, 1e-300), 0)
    rows.append(("halfcauchy(log β)", hc_density))
    cc_density = sps.cauchy(scale=1).pdf(log_beta) / np.maximum(beta, 1e-300)
    rows.append(("cauchy(log β)", cc_density))
    # uniform_logβ: density on β = const / β.
    rows.append(("uniform(log β) on [-4, 4]", np.where((log_beta >= -4) & (log_beta <= 4), 1 / np.maximum(beta, 1e-300), 0)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, dens in rows:
        # Left: density on β (linear axis).
        axes[0].plot(beta, dens, label=label, linewidth=1.4)
        # Right: density on log β = density on β · β (Jacobian).
        axes[1].plot(log_beta, dens * beta, label=label, linewidth=1.4)
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.05, 30)
    axes[0].set_ylim(0, 2.5)
    axes[0].set_xlabel(r"$\beta$ (log axis)")
    axes[0].set_ylabel(r"density on $\beta$")
    axes[0].grid(True, linewidth=0.3, alpha=0.4)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_xlim(-4, 4)
    axes[1].set_xlabel(r"$\log \beta$")
    axes[1].set_ylabel(r"density on $\log \beta$ (Jacobian-corrected)")
    axes[1].grid(True, linewidth=0.3, alpha=0.4)
    fig.suptitle("Tested priors on β (and their pushforwards onto log β)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "prior_pmf_visual.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_priors_results(fig_dir: Path, table_dir: Path) -> dict:
    """For each prior, show β̂ trajectory and late max-p_sample / entropy_sample."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"rows": []}
    beta_targets = [0.25, 1.0, 4.0]

    fig, axes = plt.subplots(len(beta_targets), 1, figsize=(10, 3 * len(beta_targets)), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, bt in enumerate(beta_targets):
        ax = axes[i]
        ax.axhline(np.log(bt), color="#6b6453", linestyle=":", linewidth=0.8,
                   label=rf"$\log \beta_{{\rm target}} = {np.log(bt):.2f}$")
        ax.set_title(rf"$\beta_{{\rm target}} = {bt}$")
        ax.set_ylabel(r"$\log \hat\beta_t$")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        for j, prior in enumerate(PRIORS_TESTED):
            d = _condition_dir_for_prior(prior)
            if not (d / MANIFEST_FILENAME).exists():
                continue
            data = _load_traces_for(d, bt, corrected=True)
            if data is None:
                continue
            log_bh = np.log(np.clip(data["beta_hat_pre"], 1e-9, None))  # (P*R, T)
            T = log_bh.shape[-1]
            ts = np.arange(T)
            med = np.median(log_bh, axis=0)
            q25 = np.quantile(log_bh, 0.25, axis=0)
            q75 = np.quantile(log_bh, 0.75, axis=0)
            color = cmap(j % 10)
            ax.plot(ts, med, color=color, linewidth=1.3, label=prior)
            ax.fill_between(ts, q25, q75, color=color, alpha=0.10)

            late_mps = data["max_p_sample"][..., -50:].mean()
            late_ent = data["entropy_sample"][..., -50:].mean()
            late_bh = data["beta_hat_pre"][..., -50:].mean()
            late_bd = data["beta_dec"][..., -50:].mean()
            summary["rows"].append({
                "prior": prior,
                "beta_target": bt,
                "beta_hat_late": float(late_bh),
                "beta_dec_late": float(late_bd),
                "max_p_sample_late": float(late_mps),
                "entropy_sample_late": float(late_ent),
            })
        if i == 0:
            ax.legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel(r"$t$")
    fig.tight_layout()
    fig.savefig(fig_dir / "prior_beta_hat_traj.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Bar plot of late max P_sample per prior, per β_target.
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    width = 0.10
    x = np.arange(len(beta_targets))
    for j, prior in enumerate(PRIORS_TESTED):
        rows = [r for r in summary["rows"] if r["prior"] == prior]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["beta_target"])
        if {r["beta_target"] for r in rows} != set(beta_targets):
            # Align — fill missing with NaN.
            row_by_bt = {r["beta_target"]: r for r in rows}
            rows = [row_by_bt[b] if b in row_by_bt else {"max_p_sample_late": np.nan} for b in beta_targets]
        vals = [r["max_p_sample_late"] for r in rows]
        ax.bar(x + j * width, vals, width=width, label=prior, color=cmap(j % 10))
    ax.set_xticks(x + width * (len(PRIORS_TESTED) - 1) / 2)
    ax.set_xticklabels([rf"β_t = {b}" for b in beta_targets])
    ax.set_ylabel("late max P_sample (mean over last 50 tokens)")
    ax.set_title("Effect of prior on late-trajectory sample concentration (corrected arm)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "prior_max_p_sample.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    (table_dir / "prior_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_fuzzy_window_results(fig_dir: Path, table_dir: Path) -> dict:
    """Compare β̂ trajectories under different (α, γ) settings."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"rows": []}
    beta_targets = [0.25, 4.0]

    cmap = plt.get_cmap("viridis")
    configs = sorted(p.name for p in FUZZY_BASE.glob("a*_g*"))
    if not configs:
        return summary

    fig, axes = plt.subplots(len(beta_targets), 1, figsize=(10, 3 * len(beta_targets)), sharex=True)
    for i, bt in enumerate(beta_targets):
        ax = axes[i]
        ax.axhline(np.log(bt), color="#6b6453", linestyle=":", linewidth=0.8)
        ax.set_title(rf"$\beta_{{\rm target}} = {bt}$")
        ax.set_ylabel(r"$\log \hat\beta_t$")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        for j, cfg in enumerate(configs):
            d = FUZZY_BASE / cfg
            data = _load_traces_for(d, bt, corrected=True)
            if data is None:
                continue
            log_bh = np.log(np.clip(data["beta_hat_pre"], 1e-9, None))
            T = log_bh.shape[-1]
            ts = np.arange(T)
            med = np.median(log_bh, axis=0)
            color = cmap(0.1 + 0.7 * j / max(1, len(configs) - 1))
            label = cfg.replace("a", "α=").replace("_g", ", γ=").replace("p", ".")
            ax.plot(ts, med, color=color, linewidth=1.4, label=label)

            late_mps = data["max_p_sample"][..., -50:].mean()
            late_bh = data["beta_hat_pre"][..., -50:].mean()
            late_bd = data["beta_dec"][..., -50:].mean()
            summary["rows"].append({
                "config": cfg,
                "beta_target": bt,
                "beta_hat_late": float(late_bh),
                "beta_dec_late": float(late_bd),
                "max_p_sample_late": float(late_mps),
            })
        if i == 0:
            ax.legend(fontsize=9, loc="best")
    axes[-1].set_xlabel(r"$t$")
    fig.suptitle("Fuzzy-window: $\\hat\\beta_t$ trajectories under various (α, γ) settings", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "fuzzy_window_traj.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    (table_dir / "fuzzy_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    fig_dir = RESULTS_DIR / "figures"
    table_dir = RESULTS_DIR / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    plot_priors_visual(fig_dir)
    print("[analyze] prior visuals done")
    if PRIOR_BASE.exists():
        plot_priors_results(fig_dir, table_dir)
        print("[analyze] prior sweep done")
    if FUZZY_BASE.exists():
        plot_fuzzy_window_results(fig_dir, table_dir)
        print("[analyze] fuzzy window sweep done")
    print(f"[analyze] outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
