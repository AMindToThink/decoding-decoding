"""Generate the F0/F3/F1/F2 figures for counter-decoding.

Reads from data/counter_decode/ (the layout written by run_counter_decode.py)
and writes figures + JSON tables to results/counter_decode/.

Usage:
    uv run python scripts/plot_counter_decode.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoding_decoding.counter_decode_analyze import (
    CounterConditionData,
    f0_concentration,
    f1_svd_over_positions,
    f1_svd_per_prompt_at_position,
    f2_calibration_at_position,
    f3_beta_eff,
    load_counter_condition,
    normalize_svd_scale,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "counter_decode"
RESULTS_DIR = REPO_ROOT / "results" / "counter_decode"


# -----------------------------------------------------------------------------
# F0 — concentration drift
# -----------------------------------------------------------------------------


def plot_f0(
    beta_targets: list[float],
    out_dir: Path,
    *,
    fig_dir: Path,
) -> dict:
    """Two-panel layout per β_target (max-prob and entropy of P_sample).

    Layout:
        rows: β_target (one per row)
        cols: P_sample max-prob | P_sample entropy | P_φ max-prob | P_φ entropy
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(beta_targets)
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(16, 3 * n_rows), squeeze=False, sharex=True
    )

    summary: dict = {"beta_targets": beta_targets, "rows": []}

    for i, bt in enumerate(beta_targets):
        cond_corr = load_counter_condition(out_dir, beta_target=bt, corrected=True)
        cond_unco = load_counter_condition(out_dir, beta_target=bt, corrected=False)
        f0_corr = f0_concentration(cond_corr)
        f0_unco = f0_concentration(cond_unco)

        T = f0_corr.max_p_sample.point.shape[0]
        ts = np.arange(T)

        for j, (key, label, ax) in enumerate(
            zip(
                ["max_p_sample", "entropy_sample", "max_p_phi", "entropy_phi"],
                [
                    r"max-prob of $P_{\rm sample}$",
                    r"entropy of $P_{\rm sample}$",
                    r"max-prob of $P_{\phi}$",
                    r"entropy of $P_{\phi}$",
                ],
                axes[i],
            )
        ):
            for f0, color, name in (
                (f0_corr, "#2d5d4f", "corrected"),
                (f0_unco, "#a8412a", "uncorrected"),
            ):
                summ = getattr(f0, key)
                ax.plot(ts, summ.point, label=name, color=color, linewidth=1.4)
                ax.fill_between(ts, summ.ci_low, summ.ci_high, color=color, alpha=0.15)
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(rf"$\beta_{{\rm target}}={bt}$", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel(r"$t$")
            ax.grid(True, linewidth=0.3, alpha=0.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=9, loc="best")

        # Persist last-100-step means for sanity inspection.
        last_slice = slice(T - min(50, T), T)
        summary["rows"].append(
            {
                "beta_target": float(bt),
                "max_p_sample_corr_late": float(
                    f0_corr.max_p_sample.point[last_slice].mean()
                ),
                "max_p_sample_unco_late": float(
                    f0_unco.max_p_sample.point[last_slice].mean()
                ),
                "entropy_sample_corr_late": float(
                    f0_corr.entropy_sample.point[last_slice].mean()
                ),
                "entropy_sample_unco_late": float(
                    f0_unco.entropy_sample.point[last_slice].mean()
                ),
                "max_p_phi_corr_late": float(
                    f0_corr.max_p_phi.point[last_slice].mean()
                ),
                "max_p_phi_unco_late": float(
                    f0_unco.max_p_phi.point[last_slice].mean()
                ),
                "entropy_phi_corr_late": float(
                    f0_corr.entropy_phi.point[last_slice].mean()
                ),
                "entropy_phi_unco_late": float(
                    f0_unco.entropy_phi.point[last_slice].mean()
                ),
            }
        )
    fig.suptitle(
        "F0 — concentration drift, corrected vs uncorrected", fontsize=12, y=1.0
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "F0_concentration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return summary


# -----------------------------------------------------------------------------
# F3 — closed-loop stability
# -----------------------------------------------------------------------------


def plot_f3(
    beta_targets: list[float],
    out_dir: Path,
    *,
    fig_dir: Path,
) -> dict:
    """Panels per β_target showing log β_eff (median + IQR) for both arms,
    plus a cross-panel summary of E(t) on a log-y scale."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = len(beta_targets)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.5), sharex=True)

    summary: dict = {"beta_targets": beta_targets, "rows": []}

    for i, bt in enumerate(beta_targets):
        cond_corr = load_counter_condition(out_dir, beta_target=bt, corrected=True)
        cond_unco = load_counter_condition(out_dir, beta_target=bt, corrected=False)
        f3_corr = f3_beta_eff(cond_corr)
        f3_unco = f3_beta_eff(cond_unco)
        T = f3_corr.median_log_beta_eff.shape[0]
        ts = np.arange(T)

        # Top row: log β_eff trajectory (median + IQR).
        ax = axes[0, i]
        ax.axhline(np.log(bt), color="#6b6453", linewidth=0.8, linestyle=":")
        for f3, color, name in (
            (f3_corr, "#2d5d4f", "corrected"),
            (f3_unco, "#a8412a", "uncorrected"),
        ):
            ax.plot(ts, f3.median_log_beta_eff, color=color, label=name, linewidth=1.4)
            half_iqr = 0.5 * f3.iqr_log_beta_eff
            ax.fill_between(
                ts,
                f3.median_log_beta_eff - half_iqr,
                f3.median_log_beta_eff + half_iqr,
                color=color,
                alpha=0.15,
            )
        ax.set_title(rf"$\beta_{{\rm target}}={bt}$", fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        if i == 0:
            ax.set_ylabel(r"$\log \beta_{\rm eff}$ (median ± IQR)", fontsize=10)
            ax.legend(fontsize=9, loc="best")

        # Bottom row: E(t) on log-y.
        ax = axes[1, i]
        for f3, color, name in (
            (f3_corr, "#2d5d4f", "corrected"),
            (f3_unco, "#a8412a", "uncorrected"),
        ):
            ax.plot(ts, f3.median_abs_log_err + 1e-12, color=color, label=name, linewidth=1.4)
        ax.set_yscale("log")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        if i == 0:
            ax.set_ylabel(r"$E(t) = $ median $|\log\beta_{\rm eff} - \log\beta_{\rm target}|$", fontsize=9)
        ax.set_xlabel(r"$t$")

        summary["rows"].append(
            {
                "beta_target": float(bt),
                "median_abs_log_err_corr_at_T": float(f3_corr.median_abs_log_err[-1]),
                "median_abs_log_err_unco_at_T": float(f3_unco.median_abs_log_err[-1]),
                "iqr_log_beta_eff_corr_at_T": float(f3_corr.iqr_log_beta_eff[-1]),
                "iqr_log_beta_eff_unco_at_T": float(f3_unco.iqr_log_beta_eff[-1]),
            }
        )

    fig.suptitle("F3 — closed-loop stability", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "F3_closed_loop.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_f1_2(
    beta_targets: list[float],
    out_dir: Path,
    *,
    fig_dir: Path,
) -> dict:
    """F1.2 — empirical β̂_t drift under uncorrected decoding for each β_dec.

    One curve per β_dec. Y axis: log β̂_pre (the streaming Laplace estimate).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    cmap = plt.get_cmap("inferno")
    n = len(beta_targets)
    summary: dict = {"beta_targets": beta_targets, "rows": []}
    for idx, bt in enumerate(beta_targets):
        cond = load_counter_condition(out_dir, beta_target=bt, corrected=False)
        bh = cond.beta_hat_pre  # (P, R, T)
        flat = np.log(bh).reshape(-1, bh.shape[-1])
        median_log = np.median(flat, axis=0)
        q25 = np.quantile(flat, 0.25, axis=0)
        q75 = np.quantile(flat, 0.75, axis=0)
        c = cmap(0.15 + 0.65 * idx / max(1, n - 1))
        ts = np.arange(median_log.shape[0])
        ax.plot(ts, median_log, color=c, label=rf"$\beta_{{\rm dec}}={bt}$", linewidth=1.4)
        ax.fill_between(ts, q25, q75, color=c, alpha=0.12)
        summary["rows"].append(
            {
                "beta_dec": float(bt),
                "median_log_beta_hat_at_T": float(median_log[-1]),
                "iqr_log_beta_hat_at_T": float(q75[-1] - q25[-1]),
            }
        )
    ax.axhline(0.0, color="#6b6453", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\log\hat\beta_t$ (streaming Laplace)")
    ax.set_title("F1.2 — empirical drift of streaming β̂ under uncorrected decoding")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.2_beta_hat_drift.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return summary


# -----------------------------------------------------------------------------
# Step 1 (SVD) and Step 2 (calibration) — full-vocab logits
# -----------------------------------------------------------------------------


def plot_f1_and_f2_svd(
    beta_targets: list[float],
    out_dir: Path,
    *,
    fig_dir: Path,
) -> dict:
    """F1.1 — R(t) over positions, F1.2-style β̄(k) vs β_k, F2.1 — Δ(t,k),
    and F2.2 — scatter of streaming β̂ vs SVD β̄ (scale-anchored at t=0)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        svds_raw = f1_svd_over_positions(out_dir, beta_dec_values=beta_targets)
    except FileNotFoundError as e:
        print(f"[warn] sparse_logits not found, skipping SVD plots: {e}")
        return {"rows": []}
    if not svds_raw:
        return {"rows": []}
    # Normalize β̄ scale by anchoring at the earliest available position
    # (t=0 if present; otherwise the smallest position we have).
    anchor_pos = min(s.position for s in svds_raw)
    svds = normalize_svd_scale(svds_raw, anchor_at_position=anchor_pos)

    summary: dict = {"rows": [], "scale_anchor": int(anchor_pos)}

    # F1.1: R(t) vs t.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    positions = [s.position for s in svds]
    R_vals = [s.R for s in svds]
    ax.plot(positions, R_vals, "o-", color="#1f5e8c")
    ax.set_xlabel(r"$t$ (sparse position)")
    ax.set_ylabel(r"$R(t) = \sum_{i\geq 2}\sigma_i^2 / \sum_i \sigma_i^2$")
    ax.set_title("F1.1 — off-axis residual fraction (multiplicativity test)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.1_R_of_t.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Empirical β̄(k, t) recovered from SVD over time, one curve per k.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    cmap = plt.get_cmap("inferno")
    K = len(beta_targets)
    bk_arr = np.asarray(beta_targets, dtype=np.float64)
    log_bar_table = np.zeros((len(svds), K), dtype=np.float64)
    for ti, s in enumerate(svds):
        log_bar_table[ti, :] = np.log(np.maximum(s.beta_recovered, 1e-9))
    for ki, bk in enumerate(beta_targets):
        c = cmap(0.15 + 0.65 * ki / max(1, K - 1))
        ax.plot(
            positions,
            log_bar_table[:, ki],
            "o-",
            color=c,
            label=rf"$\beta_{{\rm dec}}={bk}$",
        )
        ax.axhline(np.log(bk), color=c, linestyle=":", linewidth=0.6)
    ax.set_xlabel(r"$t$ (sparse position)")
    ax.set_ylabel(r"$\log \bar\beta_t(k)$  (scale anchor at $t=$" + f"{anchor_pos})")
    ax.set_title("F1.2-svd — empirical SVD-recovered β̄(k, t) over time")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.2-svd_beta_bar.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # F2.1 — Δ_t(k) = log β̂ - log β̄ over t, one curve per β_k.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    delta_table = np.zeros((len(svds), K), dtype=np.float64)
    beta_hat_table = np.zeros((len(svds), K), dtype=np.float64)
    for ti, s in enumerate(svds):
        cal = f2_calibration_at_position(
            out_dir,
            position=s.position,
            beta_dec_values=beta_targets,
            svd=s,
        )
        beta_hat_table[ti, :] = cal.beta_hat_streaming
        delta_table[ti, :] = cal.log_gap
    for ki, bk in enumerate(beta_targets):
        c = cmap(0.15 + 0.65 * ki / max(1, K - 1))
        ax.plot(positions, delta_table[:, ki], "o-", color=c, label=rf"$\beta_{{\rm dec}}={bk}$")
    ax.axhline(0.0, color="#6b6453", linestyle=":", linewidth=0.6)
    ax.set_xlabel(r"$t$ (sparse position)")
    ax.set_ylabel(r"$\Delta_t(k) = \log\hat\beta - \log\bar\beta$")
    ax.set_title("F2.1 — streaming-vs-SVD calibration gap (anchored at t=0)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "F2.1_calibration_gap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # F2.2: scatter log β̂ vs log β̄_norm.
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    log_hat = np.log(np.maximum(beta_hat_table, 1e-9)).reshape(-1)
    log_bar = log_bar_table.reshape(-1)
    ax.scatter(log_bar, log_hat, alpha=0.6, color="#a8412a")
    lo = float(min(log_hat.min(), log_bar.min()) - 0.2)
    hi = float(max(log_hat.max(), log_bar.max()) + 0.2)
    ax.plot([lo, hi], [lo, hi], color="#6b6453", linewidth=0.8, linestyle=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\log \bar\beta_t(k)$  (SVD, anchored)")
    ax.set_ylabel(r"$\log \hat\beta_t(k)$  (streaming Laplace)")
    ax.set_title("F2.2 — calibration scatter (identity = matched-prior)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "F2.2_calibration_scatter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    for ti, s in enumerate(svds):
        summary["rows"].append(
            {
                "position": int(s.position),
                "R": float(s.R),
                "singular_values": s.singular_values.tolist(),
                "beta_recovered_norm": s.beta_recovered.tolist(),
                "beta_hat_streaming": beta_hat_table[ti, :].tolist(),
                "log_gap": delta_table[ti, :].tolist(),
            }
        )
    return summary


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--beta-targets",
        type=str,
        default="0.5,0.7,1.0,1.3,1.7,2.0",
    )
    args = parser.parse_args()

    beta_targets = [float(x) for x in args.beta_targets.split(",")]
    fig_dir = args.results_dir / "figures"
    table_dir = args.results_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("[plot] F0 ...")
    f0_summary = plot_f0(beta_targets, args.data_dir, fig_dir=fig_dir)
    (table_dir / "F0_summary.json").write_text(json.dumps(f0_summary, indent=2))

    print("[plot] F3 ...")
    f3_summary = plot_f3(beta_targets, args.data_dir, fig_dir=fig_dir)
    (table_dir / "F3_summary.json").write_text(json.dumps(f3_summary, indent=2))

    print("[plot] F1.2 ...")
    f12_summary = plot_f1_2(beta_targets, args.data_dir, fig_dir=fig_dir)
    (table_dir / "F1.2_summary.json").write_text(json.dumps(f12_summary, indent=2))

    print("[plot] F1.1 + F2.2 ...")
    svd_summary = plot_f1_and_f2_svd(beta_targets, args.data_dir, fig_dir=fig_dir)
    (table_dir / "F1_F2_svd_summary.json").write_text(json.dumps(svd_summary, indent=2))

    print("[plot] F1.4 (per-prompt SVD) ...")
    pp_summary = plot_per_prompt_svd(beta_targets, args.data_dir, fig_dir=fig_dir)
    (table_dir / "F1.4_per_prompt_summary.json").write_text(json.dumps(pp_summary, indent=2))

    print(f"[plot] all figures and tables in {args.results_dir}")


def plot_per_prompt_svd(
    beta_targets: list[float],
    out_dir: Path,
    *,
    fig_dir: Path,
) -> dict:
    """F1.4 — per-prompt SVD comparison.

    For each sparse position, run the SVD per-prompt (each prompt's runs are
    averaged separately, so the principal-direction recovery uses only
    one prompt's ℓ*_t at a time). Plots:
        - F1.4_R_per_prompt: median + IQR of R^(p)(t) across prompts vs t.
          Pop-up: cross-prompt-aggregated SVD's R(t) on the same axes
          (the "noisy upper bound" the plan describes).
        - F1.4_beta_per_prompt: median + IQR of recovered log β̄^(p)(k) across
          prompts at the largest sparse t, vs the streaming β̂.

    These give a tighter test than the global SVD plots.
    """
    import warnings
    fig_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = Path(out_dir) / "sparse_logits"
    any_npz = next(sparse_dir.glob("*.npz"), None)
    if any_npz is None:
        return {"rows": []}
    positions = [int(p) for p in np.load(any_npz)["positions"].tolist()]

    summary: dict = {"positions": positions, "rows": []}

    R_med = []
    R_q25 = []
    R_q75 = []
    log_beta_med = []
    log_beta_q25 = []
    log_beta_q75 = []
    for pos in positions:
        try:
            res = f1_svd_per_prompt_at_position(
                out_dir, position=pos, beta_dec_values=beta_targets
            )
        except ValueError as e:
            warnings.warn(str(e))
            continue
        R_med.append(res["R_median"])
        R_q25.append(res["R_iqr"][0])
        R_q75.append(res["R_iqr"][1])
        log_beta_med.append(res["log_beta_median_across_prompts"])
        log_beta_q25.append(res["log_beta_iqr_across_prompts"][0])
        log_beta_q75.append(res["log_beta_iqr_across_prompts"][1])
        summary["rows"].append(
            {
                "position": int(pos),
                "n_prompts": int(res["n_prompts"]),
                "R_median": float(res["R_median"]),
                "R_iqr_low": float(res["R_iqr"][0]),
                "R_iqr_high": float(res["R_iqr"][1]),
                "log_beta_median": list(map(float, res["log_beta_median_across_prompts"].tolist())),
            }
        )

    R_med = np.asarray(R_med)
    R_q25 = np.asarray(R_q25)
    R_q75 = np.asarray(R_q75)
    log_beta_med = np.stack(log_beta_med, axis=0)  # (P, K)
    log_beta_q25 = np.stack(log_beta_q25, axis=0)  # (P, K)
    log_beta_q75 = np.stack(log_beta_q75, axis=0)  # (P, K)

    # F1.4 R(t): per-prompt vs aggregated.
    svds_global = f1_svd_over_positions(out_dir, beta_dec_values=beta_targets)
    R_global = np.asarray([s.R for s in svds_global])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(positions, R_med, "o-", color="#1f5e8c", label="median per-prompt $R^{(p)}(t)$")
    ax.fill_between(positions, R_q25, R_q75, color="#1f5e8c", alpha=0.2,
                    label="25/75% across prompts")
    ax.plot(positions, R_global, "s--", color="#a8412a",
            label="global aggregated $R(t)$ (noisy upper bound)")
    ax.set_xlabel(r"$t$ (sparse position)")
    ax.set_ylabel(r"off-axis residual fraction $R(t)$")
    ax.set_title("F1.4 — per-prompt vs global multiplicativity test")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.4_R_per_prompt.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # F1.4 β̄^(p)(k) at the latest position vs decoder values.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    last_idx = -1
    bk_arr = np.asarray(beta_targets, dtype=np.float64)
    ax.plot(np.log(bk_arr), log_beta_med[last_idx], "o-", color="#1f5e8c",
            label=rf"per-prompt median $\log\bar\beta^{{(p)}}(k)$ at $t={positions[last_idx]}$")
    ax.fill_between(np.log(bk_arr), log_beta_q25[last_idx], log_beta_q75[last_idx],
                    color="#1f5e8c", alpha=0.2, label="IQR across prompts")
    lo = float(min(np.log(bk_arr).min(), log_beta_q25[last_idx].min()) - 0.2)
    hi = float(max(np.log(bk_arr).max(), log_beta_q75[last_idx].max()) + 0.2)
    ax.plot([lo, hi], [lo, hi], color="#6b6453", linewidth=0.8, linestyle=":",
            label="identity $\\bar\\beta = \\beta_{\\rm dec}$")
    ax.set_xlabel(r"$\log \beta_{\rm dec}$")
    ax.set_ylabel(r"$\log \bar\beta^{(p)}(k)$ (per-prompt SVD)")
    ax.set_title(f"F1.4-bar — recovered β̄ vs decoder β at t={positions[last_idx]}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.4_beta_per_prompt.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return summary


if __name__ == "__main__":
    main()
