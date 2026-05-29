"""Analyze the shape-estimator sweep (rank-gap and Rényi-∞).

Reads:
    data/counter_decode_shape/est_rank_gap/...
    data/counter_decode_shape/est_renyi_infty/...

Compares each new estimator against the baseline streaming Laplace at the
sweet-spot σ_0 = 0.2 (from the prior_sweep findings) on a shared
β_target ∈ {0.25, 0.5, 1, 2, 4} sweep.

Produces:
    results/counter_decode/shape_estimators/
        figures/
            shape_traj.png            — β̂_t, β_dec_t, entropy_sample_t,
                                        max_p_sample_t for each (estimator, β_target)
            shape_vs_btarget.png      — late entropy vs β_target per estimator
            shape_vs_baseline.png     — shape estimators vs Laplace σ_0=0.2 baseline
        tables/
            shape_summary.json
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
SHAPE_BASE = REPO_ROOT / "data" / "counter_decode_shape"
ENTROPY_BASE = REPO_ROOT / "data" / "counter_decode_entropy_stability"
RESULTS_DIR = REPO_ROOT / "results" / "counter_decode" / "shape_estimators"

ESTIMATORS = ("rank_gap", "renyi_infty")
BETA_TARGETS = (0.25, 0.5, 1.0, 2.0, 4.0)
# Use σ_0=0.2, γ=1.0 Laplace as the best baseline (from entropy-stability sweep).
BASELINE_TAG = "s0p2_g1p0"


def _est_dir(est: str) -> Path:
    return SHAPE_BASE / f"est_{est}"


def _baseline_dir() -> Path:
    return ENTROPY_BASE / BASELINE_TAG


def _load_traces(out_dir: Path, beta_target: float) -> Optional[dict]:
    manifest = out_dir / MANIFEST_FILENAME
    if not manifest.exists():
        return None
    m = pl.read_parquet(manifest)
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
        "beta_hat": np.stack(bh),
        "beta_dec": np.stack(bd),
        "max_p_sample": np.stack(mps),
        "entropy_sample": np.stack(ent_s),
        "entropy_phi": np.stack(ent_phi),
        "max_p_phi": np.stack(mph),
    }


def _entropy_slope(ent: np.ndarray) -> float:
    T = ent.shape[-1]
    half = T // 2
    tail = ent[..., half:]
    t = np.arange(tail.shape[-1])
    flat = tail.reshape(-1, tail.shape[-1])
    slopes = [np.polyfit(t, row, 1)[0] for row in flat]
    return float(np.mean(slopes))


def collect_summary() -> Dict:
    rec = {"records": [], "per_t": {}}
    sources = [(est, _est_dir(est)) for est in ESTIMATORS]
    sources.append(("laplace_baseline", _baseline_dir()))
    for label, base in sources:
        for bt in BETA_TARGETS:
            tr = _load_traces(base, bt)
            if tr is None:
                continue
            T = tr["beta_hat"].shape[-1]
            late = slice(max(T - 30, 0), T)
            r = {
                "estimator": label,
                "beta_target": bt,
                "beta_hat_late": float(np.mean(tr["beta_hat"][..., late])),
                "beta_dec_late": float(np.mean(tr["beta_dec"][..., late])),
                "entropy_sample_late": float(np.mean(tr["entropy_sample"][..., late])),
                "entropy_phi_late": float(np.mean(tr["entropy_phi"][..., late])),
                "max_p_sample_late": float(np.mean(tr["max_p_sample"][..., late])),
                "max_p_phi_late": float(np.mean(tr["max_p_phi"][..., late])),
                "entropy_slope_2nd_half": _entropy_slope(tr["entropy_sample"]),
            }
            rec["records"].append(r)
            rec["per_t"][f"{label}|{bt}"] = {
                "beta_hat": tr["beta_hat"].mean(axis=0).tolist(),
                "beta_dec": tr["beta_dec"].mean(axis=0).tolist(),
                "entropy_sample": tr["entropy_sample"].mean(axis=0).tolist(),
                "entropy_phi": tr["entropy_phi"].mean(axis=0).tolist(),
                "max_p_sample": tr["max_p_sample"].mean(axis=0).tolist(),
            }
    return rec


def plot_trajectories(summary: Dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = ESTIMATORS + ("laplace_baseline",)
    cmap = plt.cm.coolwarm
    fig, axes = plt.subplots(4, len(labels), figsize=(4 * len(labels), 11), sharex=True)
    for j, est in enumerate(labels):
        for bt in BETA_TARGETS:
            d = summary["per_t"].get(f"{est}|{bt}")
            if d is None:
                continue
            T = len(d["beta_hat"])
            t = np.arange(T)
            color = cmap(np.log(bt) / (2 * np.log(BETA_TARGETS[-1])) + 0.5)
            axes[0, j].plot(t, np.log(np.array(d["beta_hat"])), color=color, lw=1.0,
                            label=f"β={bt}")
            axes[1, j].plot(t, np.log(np.array(d["beta_dec"])), color=color, lw=1.0)
            axes[2, j].plot(t, d["entropy_sample"], color=color, lw=1.0)
            axes[3, j].plot(t, d["max_p_sample"], color=color, lw=1.0)
        axes[0, j].set_title(f"estimator: {est}", fontsize=10)
        for ax in axes[:, j]:
            ax.grid(True, alpha=0.3)
        axes[0, j].axhline(0.0, color="grey", lw=0.5, ls=":")
    axes[0, 0].set_ylabel(r"$\log \hat\beta_t$")
    axes[1, 0].set_ylabel(r"$\log \beta_{dec,t}$")
    axes[2, 0].set_ylabel("entropy(P_sample)")
    axes[3, 0].set_ylabel("max P_sample")
    for j in range(len(labels)):
        axes[3, j].set_xlabel("token t")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Shape estimators vs Laplace baseline: per-step trajectories", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "shape_traj.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_late_vs_btarget(summary: Dict, fig_dir: Path) -> None:
    labels = ESTIMATORS + ("laplace_baseline",)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for est in labels:
        bts, bh, bd, ent, mp = [], [], [], [], []
        for r in summary["records"]:
            if r["estimator"] != est:
                continue
            bts.append(r["beta_target"])
            bh.append(np.log(r["beta_hat_late"]))
            bd.append(np.log(r["beta_dec_late"]))
            ent.append(r["entropy_sample_late"])
            mp.append(r["max_p_sample_late"])
        order = np.argsort(bts)
        bts_a = np.array(bts)[order]
        axes[0].plot(bts_a, np.array(bh)[order], "o-", label=est)
        axes[1].plot(bts_a, np.array(bd)[order], "o-", label=est)
        axes[2].plot(bts_a, np.array(ent)[order], "o-", label=est)
        axes[3].plot(bts_a, np.array(mp)[order], "o-", label=est)
    for ax, lbl in zip(
        axes,
        [r"late $\log\hat\beta$", r"late $\log\beta_{dec}$",
         "late entropy(P_sample)", "late max P_sample"],
    ):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta_{target}$")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Late-trajectory stats vs β_target", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "shape_vs_btarget.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_dir = RESULTS_DIR / "figures"
    table_dir = RESULTS_DIR / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summary()
    if not summary["records"]:
        raise SystemExit("No data yet — sweep not finished?")

    with (table_dir / "shape_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_trajectories(summary, fig_dir)
    plot_late_vs_btarget(summary, fig_dir)

    print(f"{'estimator':>20} {'β':>5} {'β̂':>7} {'β_dec':>8} {'ent':>7} "
          f"{'maxP':>6} {'slope':>9}")
    for r in summary["records"]:
        print(f"{r['estimator']:>20} {r['beta_target']:>5.2f} "
              f"{r['beta_hat_late']:>7.3f} {r['beta_dec_late']:>8.3f} "
              f"{r['entropy_sample_late']:>7.3f} {r['max_p_sample_late']:>6.3f} "
              f"{r['entropy_slope_2nd_half']:>9.5f}")


if __name__ == "__main__":
    main()
