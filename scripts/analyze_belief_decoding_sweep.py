"""Analyze the (γ, α, p) joint belief-decoding sweep.

Reads:    data/counter_decode_belief_sweep/g*_a*_p*/...
Produces: results/counter_decode/belief_sweep/
            summary.json
            F_bs_1_tracking.png
            F_bs_2_heatmaps_alpha{α}.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from decoding_decoding.data_layout import MANIFEST_FILENAME


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_BASE = REPO_ROOT / "data" / "counter_decode_belief_sweep"
RESULTS_DIR = REPO_ROOT / "results" / "counter_decode" / "belief_sweep"


def _tag(gamma: float, alpha: float, p: float) -> str:
    return f"g{gamma:.2f}_a{alpha:.2f}_p{p:.2f}".replace(".", "p")


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
        "beta_hat_pre": np.stack(bh),    # (n_prompts, T)
        "beta_dec": np.stack(bd),
        "max_p_sample": np.stack(mps),
        "entropy_sample": np.stack(ent_s),
    }


def _discover_grid() -> tuple[list[float], list[float], list[float], list[float]]:
    """Discover the (γ, α, p, β_target) values actually present on disk."""
    rx = re.compile(r"g(\d+p\d+)_a(\d+p\d+)_p(\d+p\d+)")
    gammas, alphas, ps = set(), set(), set()
    for d in DATA_BASE.glob("g*_a*_p*"):
        mat = rx.fullmatch(d.name)
        if not mat:
            continue
        g = float(mat.group(1).replace("p", "."))
        a = float(mat.group(2).replace("p", "."))
        p = float(mat.group(3).replace("p", "."))
        gammas.add(g); alphas.add(a); ps.add(p)
    # β_targets from one sample manifest.
    bts: set[float] = set()
    for d in DATA_BASE.glob("g*_a*_p*"):
        mp = d / MANIFEST_FILENAME
        if not mp.exists():
            continue
        m = pl.read_parquet(mp)
        for row in m.iter_rows(named=True):
            label = row["condition_label"]
            m2 = re.match(r"btarget=([\d.]+)_arm=corr", label)
            if m2:
                bts.add(float(m2.group(1)))
        break
    return sorted(gammas), sorted(alphas), sorted(ps), sorted(bts)


def collect_summary() -> dict:
    gammas, alphas, ps, bts = _discover_grid()
    out = {
        "grid": {"gammas": gammas, "alphas": alphas, "ps": ps, "beta_targets": bts},
        "records": [],
    }
    for g in gammas:
        for a in alphas:
            for p in ps:
                tag = _tag(g, a, p)
                for bt in bts:
                    tr = _load_traces(DATA_BASE / tag, bt)
                    if tr is None:
                        continue
                    T = tr["entropy_sample"].shape[-1]
                    late = slice(T - 30, T)
                    rec = {
                        "gamma": g, "alpha": a, "p": p, "beta_target": bt,
                        "beta_hat_late_mean": float(np.mean(tr["beta_hat_pre"][..., late])),
                        "beta_dec_late_mean": float(np.mean(tr["beta_dec"][..., late])),
                        "entropy_sample_late_mean": float(np.mean(tr["entropy_sample"][..., late])),
                        "max_p_sample_late_mean": float(np.mean(tr["max_p_sample"][..., late])),
                        "max_p_sample_late_p95": float(np.percentile(tr["max_p_sample"][..., late], 95)),
                        "entropy_sample_late_std": float(np.std(np.mean(tr["entropy_sample"][..., late], axis=-1))),
                    }
                    out["records"].append(rec)
    return out


def _condition_quality(records: list[dict], gamma: float, alpha: float, p: float, bts: list[float]) -> dict:
    """Define one scalar 'tracking quality' per (γ, α, p):

      - want entropy at β=2 (low temp) < entropy at β=1 < entropy at β=0.5
      - want max-P at β=2 not saturating (< 0.85)
      - want max-P at β=0.5 not saturating either
    """
    cond = {r["beta_target"]: r for r in records
            if r["gamma"] == gamma and r["alpha"] == alpha and r["p"] == p}
    if not all(bt in cond for bt in bts):
        return {"valid": False}
    H = [cond[bt]["entropy_sample_late_mean"] for bt in bts]
    M = [cond[bt]["max_p_sample_late_mean"] for bt in bts]
    monotone = all(H[i] >= H[i + 1] for i in range(len(H) - 1))   # decreasing in β_target
    H_spread = H[0] - H[-1]
    max_saturating = max(M)
    # Total "badness": penalize saturating tails + small entropy spread.
    badness = max(0.0, max_saturating - 0.7) + max(0.0, 1.0 - H_spread)
    return {
        "valid": True,
        "H_by_bt": dict(zip([str(b) for b in bts], H)),
        "M_by_bt": dict(zip([str(b) for b in bts], M)),
        "monotone": bool(monotone),
        "H_spread": H_spread,
        "max_saturating": max_saturating,
        "badness": badness,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = collect_summary()
    bts = summary["grid"]["beta_targets"]
    gammas = summary["grid"]["gammas"]
    alphas = summary["grid"]["alphas"]
    ps = summary["grid"]["ps"]

    # Per-config quality metric.
    qualities = []
    for g in gammas:
        for a in alphas:
            for p in ps:
                q = _condition_quality(summary["records"], g, a, p, bts)
                if q.get("valid"):
                    qualities.append({"gamma": g, "alpha": a, "p": p, **q})
    summary["qualities"] = qualities

    # Sort by badness (ascending) to identify best configs.
    qualities_sorted = sorted(qualities, key=lambda q: q["badness"])
    summary["top_configs"] = qualities_sorted[:5]

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[bs] summary → {RESULTS_DIR / 'summary.json'}")

    # ------- Figure 1: entropy & max-P vs β_target, color by p, panels by α, line by γ.
    fig, axes = plt.subplots(len(alphas), 2, figsize=(11, 3.4 * len(alphas)), squeeze=False)
    p_cmap = plt.cm.plasma
    p_colors = {p_: p_cmap(i / max(1, len(ps) - 1)) for i, p_ in enumerate(ps)}
    line_styles = {0.85: ":", 0.95: "--", 1.00: "-"}
    for ai, alpha in enumerate(alphas):
        ax_H = axes[ai, 0]
        ax_M = axes[ai, 1]
        for g in gammas:
            for p in ps:
                xs, H_ys, M_ys = [], [], []
                for bt in bts:
                    rec = next(
                        (r for r in summary["records"]
                         if r["gamma"] == g and r["alpha"] == alpha
                         and r["p"] == p and r["beta_target"] == bt),
                        None,
                    )
                    if rec is None:
                        continue
                    xs.append(bt)
                    H_ys.append(rec["entropy_sample_late_mean"])
                    M_ys.append(rec["max_p_sample_late_mean"])
                ls = line_styles.get(round(g, 2), "-")
                ax_H.plot(xs, H_ys, color=p_colors[p], linestyle=ls,
                          marker="o", markersize=4,
                          label=f"γ={g}, p={p}" if ai == 0 and alpha == alphas[0] else None,
                          alpha=0.9)
                ax_M.plot(xs, M_ys, color=p_colors[p], linestyle=ls,
                          marker="o", markersize=4, alpha=0.9)
        ax_H.set_title(f"α = {alpha}: late sample entropy")
        ax_H.set_xlabel("β_target")
        ax_H.set_ylabel("entropy_sample (nats)")
        ax_H.grid(alpha=0.3)
        ax_M.set_title(f"α = {alpha}: late max-P_sample")
        ax_M.set_xlabel("β_target")
        ax_M.set_ylabel("max P_sample")
        ax_M.grid(alpha=0.3)
        ax_M.axhline(0.85, color="grey", linewidth=0.6, linestyle=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(8, len(labels)), fontsize=7, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / "F_bs_1_tracking.png", dpi=140)
    plt.close(fig)

    # ------- Figure 2: heatmap of "badness" in (γ, p) plane, one panel per α.
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), squeeze=False)
    grid_badness = np.full((len(gammas), len(ps)), np.nan)
    vmin, vmax = 0.0, 1.0
    for ai, alpha in enumerate(alphas):
        for gi, g in enumerate(gammas):
            for pi, p in enumerate(ps):
                q = next(
                    (q for q in qualities
                     if q["gamma"] == g and q["alpha"] == alpha and q["p"] == p), None
                )
                if q is None:
                    continue
                grid_badness[gi, pi] = q["badness"]
        ax = axes[0, ai]
        im = ax.imshow(grid_badness, origin="lower", aspect="auto",
                       cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels([f"{p}" for p in ps])
        ax.set_yticks(range(len(gammas)))
        ax.set_yticklabels([f"{g}" for g in gammas])
        ax.set_xlabel("correction exponent p")
        ax.set_ylabel("memory decay γ")
        ax.set_title(f"α = {alpha}: closed-loop badness")
        # Annotate each cell with the value.
        for gi in range(len(gammas)):
            for pi in range(len(ps)):
                v = grid_badness[gi, pi]
                if not np.isnan(v):
                    ax.text(pi, gi, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, label="badness (lower = better)")
    fig.suptitle("Closed-loop badness across (γ, p); lower = corrected sampler tracks β_target",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "F_bs_2_heatmap.png", dpi=140)
    plt.close(fig)

    # Print top 5.
    print("\n=== Top 5 configs by closed-loop badness ===")
    print(f"{'γ':>6} {'α':>6} {'p':>6} | {'H@0.5':>7} {'H@1':>7} {'H@2':>7}   {'M@0.5':>7} {'M@1':>7} {'M@2':>7} | {'spread':>7} {'maxM':>7} | {'badness':>8}")
    for q in qualities_sorted[:10]:
        H_str = " ".join(f"{q['H_by_bt'][str(bt)]:>7.3f}" for bt in bts)
        M_str = " ".join(f"{q['M_by_bt'][str(bt)]:>7.3f}" for bt in bts)
        print(
            f"{q['gamma']:>6.2f} {q['alpha']:>6.2f} {q['p']:>6.2f} | "
            f"{H_str}   {M_str} | "
            f"{q['H_spread']:>7.3f} {q['max_saturating']:>7.3f} | "
            f"{q['badness']:>8.3f}"
        )


if __name__ == "__main__":
    main()
