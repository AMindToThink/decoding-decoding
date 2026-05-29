"""Analyze the σ_0 (tight prior) sweep.

Reads:
    data/counter_decode_tight_prior/sigma_*/...

Produces:
    results/counter_decode/tight_prior/
        figures/tight_prior_traj.png       — β̂_t, β_dec_t, entropy_sample_t,
                                             max_p_sample_t for each (σ_0, β_target)
        figures/tight_prior_late_stats.png — late stable values vs σ_0, per β_target
        tables/tight_prior_summary.json    — per-(σ_0, β_target) numerical stats
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from decoding_decoding.data_layout import MANIFEST_FILENAME


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_BASE = REPO_ROOT / "data" / "counter_decode_tight_prior"
RESULTS_DIR = REPO_ROOT / "results" / "counter_decode" / "tight_prior"

SIGMA_VALUES = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
BETA_TARGETS = (0.25, 1.0, 4.0)


def _sigma_dir(sigma_0: float) -> Path:
    return DATA_BASE / f"sigma_{sigma_0}".replace(".", "p")


def _load_traces(out_dir: Path, beta_target: float) -> Optional[dict]:
    manifest_path = out_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    m = pl.read_parquet(manifest_path)
    label = f"btarget={beta_target:.2f}_arm=corr"
    sub = m.filter(pl.col("condition_label") == label).sort(["prompt_id", "run_idx"])
    if sub.height == 0:
        return None
    bh, bd, mps, ent_s, ent_phi, mph = [], [], [], [], [], []
    for row in sub.iter_rows(named=True):
        npz = np.load(out_dir / row["trace_path"])
        bh.append(npz["beta_hat_pre"])
        bd.append(npz["beta_dec"])
        mps.append(npz["max_p_sample"])
        ent_s.append(npz["entropy_sample"])
        ent_phi.append(npz["entropy_phi"])
        mph.append(npz["max_p_phi"])
    return {
        "beta_hat_pre": np.stack(bh),
        "beta_dec": np.stack(bd),
        "max_p_sample": np.stack(mps),
        "entropy_sample": np.stack(ent_s),
        "entropy_phi": np.stack(ent_phi),
        "max_p_phi": np.stack(mph),
    }


def _entropy_slope(ent: np.ndarray) -> float:
    """Mean per-token slope of entropy over the second half of the trajectory.

    Aggregates across (n_prompts, n_runs); returns nats per token."""
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
    out = {"records": [], "per_t": {}}
    for sigma_0 in SIGMA_VALUES:
        sd = _sigma_dir(sigma_0)
        for bt in BETA_TARGETS:
            tr = _load_traces(sd, bt)
            if tr is None:
                continue
            T = tr["beta_hat_pre"].shape[-1]
            late = slice(T - 30, T)
            rec = {
                "sigma_0": sigma_0,
                "beta_target": bt,
                "beta_hat_late": float(np.mean(tr["beta_hat_pre"][..., late])),
                "beta_dec_late": float(np.mean(tr["beta_dec"][..., late])),
                "entropy_sample_late": float(np.mean(tr["entropy_sample"][..., late])),
                "entropy_phi_late": float(np.mean(tr["entropy_phi"][..., late])),
                "max_p_sample_late": float(np.mean(tr["max_p_sample"][..., late])),
                "max_p_phi_late": float(np.mean(tr["max_p_phi"][..., late])),
                "entropy_slope_2nd_half": _entropy_slope(tr["entropy_sample"]),
            }
            out["records"].append(rec)
            out["per_t"][f"sigma={sigma_0}_beta={bt}"] = {
                "beta_hat": tr["beta_hat_pre"].mean(axis=0).tolist(),
                "beta_dec": tr["beta_dec"].mean(axis=0).tolist(),
                "entropy_sample": tr["entropy_sample"].mean(axis=0).tolist(),
                "entropy_phi": tr["entropy_phi"].mean(axis=0).tolist(),
                "max_p_sample": tr["max_p_sample"].mean(axis=0).tolist(),
            }
    return out


def plot_trajectories(summary: Dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
    cmap = plt.cm.viridis
    sigma_norm = {s: cmap(i / max(1, len(SIGMA_VALUES) - 1)) for i, s in enumerate(SIGMA_VALUES)}

    for j, bt in enumerate(BETA_TARGETS):
        for sigma_0 in SIGMA_VALUES:
            key = f"sigma={sigma_0}_beta={bt}"
            d = summary["per_t"].get(key)
            if d is None:
                continue
            T = len(d["beta_hat"])
            t = np.arange(T)
            color = sigma_norm[sigma_0]
            axes[0, j].plot(t, np.log(np.array(d["beta_hat"])), color=color, lw=1.2,
                            label=f"σ₀={sigma_0}")
            axes[1, j].plot(t, np.log(np.array(d["beta_dec"])), color=color, lw=1.2)
            axes[2, j].plot(t, d["entropy_sample"], color=color, lw=1.2)
            axes[3, j].plot(t, d["max_p_sample"], color=color, lw=1.2)

        axes[0, j].axhline(np.log(bt), color="black", lw=0.6, ls="--",
                           label=f"log β_target={np.log(bt):.2f}")
        axes[0, j].axhline(0.0, color="grey", lw=0.5, ls=":")
        axes[1, j].axhline(np.log(bt), color="black", lw=0.6, ls="--")
        axes[1, j].axhline(0.0, color="grey", lw=0.5, ls=":")
        axes[0, j].set_title(f"β_target = {bt}")
        if j == 0:
            axes[0, 0].legend(fontsize=7, loc="best")

    axes[0, 0].set_ylabel(r"$\log \hat\beta_t$")
    axes[1, 0].set_ylabel(r"$\log \beta_{dec,t}$")
    axes[2, 0].set_ylabel("entropy(P_sample)")
    axes[3, 0].set_ylabel("max P_sample")
    for j in range(3):
        axes[3, j].set_xlabel("token index t")

    fig.suptitle("Tight-prior σ_0 sweep: corrected arm trajectories", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "tight_prior_traj.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_late_stats(summary: Dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    sigmas = np.array(SIGMA_VALUES)
    for bt in BETA_TARGETS:
        bh = []
        bd = []
        es = []
        mp = []
        for s in sigmas:
            r = next((r for r in summary["records"]
                     if r["sigma_0"] == s and r["beta_target"] == bt), None)
            if r is None:
                bh.append(np.nan); bd.append(np.nan); es.append(np.nan); mp.append(np.nan)
                continue
            bh.append(np.log(r["beta_hat_late"]))
            bd.append(np.log(r["beta_dec_late"]))
            es.append(r["entropy_sample_late"])
            mp.append(r["max_p_sample_late"])
        axes[0].plot(sigmas, bh, "o-", label=f"β_target={bt}")
        axes[1].plot(sigmas, bd, "o-", label=f"β_target={bt}")
        axes[2].plot(sigmas, es, "o-", label=f"β_target={bt}")
        axes[3].plot(sigmas, mp, "o-", label=f"β_target={bt}")
        axes[0].axhline(np.log(bt), color="grey", ls=":", lw=0.7)
        axes[1].axhline(np.log(bt), color="grey", ls=":", lw=0.7)

    for ax, lbl in zip(
        axes,
        [r"late $\log \hat\beta$", r"late $\log \beta_{dec}$", "late entropy(P_sample)", "late max P_sample"],
    ):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_0$")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Tight-prior σ_0 sweep: late-trajectory stats vs σ_0", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "tight_prior_late_stats.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_dir = RESULTS_DIR / "figures"
    table_dir = RESULTS_DIR / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summary()
    if not summary["records"]:
        raise SystemExit("No records loaded — did the sweep finish writing?")

    with (table_dir / "tight_prior_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_trajectories(summary, fig_dir)
    plot_late_stats(summary, fig_dir)

    # Print a tidy summary table.
    print(f"{'σ_0':>6} {'β_target':>8} {'β̂_late':>10} {'β_dec_late':>11} "
          f"{'ent_smpl':>9} {'maxP':>7} {'ent_slope':>10}")
    for r in summary["records"]:
        print(f"{r['sigma_0']:>6.3f} {r['beta_target']:>8.2f} "
              f"{r['beta_hat_late']:>10.3f} {r['beta_dec_late']:>11.3f} "
              f"{r['entropy_sample_late']:>9.3f} {r['max_p_sample_late']:>7.3f} "
              f"{r['entropy_slope_2nd_half']:>10.5f}")


if __name__ == "__main__":
    main()
