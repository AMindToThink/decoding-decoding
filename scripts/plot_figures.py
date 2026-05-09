"""Produce the 5 plan figures from saved data/.

Usage:
    uv run python scripts/plot_figures.py [--data-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoding_decoding.analyze import (
    CONDITIONS,
    bootstrap_position_mean,
    cliff_log_per_step,
    cumulative_topk_mass_per_step,
    load_condition,
    smooth_rolling_mean,
)


# --------------------------------- helpers --------------------------------- #


def _cliff_arr_per_run(top_logprobs: np.ndarray, r: int) -> np.ndarray:
    """(n_prompts, n_runs, L, top_n) -> (n_prompts, n_runs, L) cliff at rank r."""
    return cliff_log_per_step(top_logprobs, r)


def _broadcast_match_runs(
    src: np.ndarray, target_runs: int
) -> np.ndarray:
    """Tile a (n_prompts, 1, L) metric to (n_prompts, target_runs, L) for paired
    contrasts against a multi-run condition. Used for k=1 (one run/prompt) when
    we want to pair against k=inf's 8 runs.
    """
    if src.shape[1] == target_runs:
        return src
    if src.shape[1] != 1:
        raise ValueError(
            f"cannot broadcast {src.shape[1]} runs to {target_runs}; only 1 -> N supported"
        )
    return np.broadcast_to(src, (src.shape[0], target_runs, src.shape[2])).copy()


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    return smooth_rolling_mean(arr.reshape(1, -1), window)[0]


# --------------------------------- figures --------------------------------- #


def fig1_main_effect(data: dict, out_path: Path) -> None:
    """Cliff delta_K vs position, faceted by condition K. Three treatment
    panels (K=1, 3, 5), each overlaying the seed-matched control (top-k=inf)
    at the same rank. 95% cluster-bootstrap CI bands."""
    treatment_ks = ("1", "3", "5")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, k in zip(axes, treatment_ks):
        rank = int(k)
        for label, condition_k, color in [
            (f"top-k = {k}", k, "C0"),
            ("top-k = ∞", "inf", "C1"),
        ]:
            cd = data[condition_k]
            metric = _cliff_arr_per_run(cd.top_logprobs, rank)
            mean, lo, hi = bootstrap_position_mean(metric, n_resamples=2000)
            L = mean.shape[0]
            xs = np.arange(L)
            ax.plot(xs, _smooth(mean, 20), color=color, label=label, lw=1.4)
            ax.fill_between(xs, _smooth(lo, 20), _smooth(hi, 20), color=color, alpha=0.18)
        ax.set_title(f"δ_{rank} = log p_{rank} − log p_{rank + 1}")
        ax.set_xlabel("generation position")
        ax.set_ylabel("cliff (nats)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Fig 1. Cliff at rank K vs. position, treatment vs. control", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig2_cliff_heatmap(data: dict, out_path: Path) -> None:
    """δ_r over (rank r, position i) as a heatmap, one panel per condition.
    A bright horizontal stripe at r = K is the H3 visual signature."""
    ranks = np.arange(1, 51)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, k in zip(axes, CONDITIONS):
        cd = data[k]
        # cliff at each rank: shape (len(ranks), n_prompts, n_runs, L) → mean to (len(ranks), L)
        per_rank_means = np.stack(
            [_cliff_arr_per_run(cd.top_logprobs, r).mean(axis=(0, 1)) for r in ranks],
            axis=0,
        )
        im = ax.imshow(
            per_rank_means,
            aspect="auto",
            origin="lower",
            extent=(0, per_rank_means.shape[1], ranks[0] - 0.5, ranks[-1] + 0.5),
            cmap="viridis",
        )
        if k != "inf":
            ax.axhline(int(k), color="red", lw=0.8, ls="--", alpha=0.7)
        ax.set_title(f"top-k = {k}")
        ax.set_xlabel("position")
        if ax is axes[0]:
            ax.set_ylabel("evaluation rank r")
        fig.colorbar(im, ax=ax, fraction=0.045, label="δ_r (nats)" if ax is axes[-1] else "")
    fig.suptitle("Fig 2. Cliff δ_r at every rank, faceted by sampling condition", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig3_concentration(data: dict, out_path: Path) -> None:
    """Cumulative top-K mass vs position, one line per condition. CI bands."""
    show_ks = [1, 3, 5, 10, 50]
    fig, axes = plt.subplots(1, len(show_ks), figsize=(4 * len(show_ks), 4), sharey=False)
    for ax, k_eval in zip(axes, show_ks):
        for color, condition_k in zip(["C0", "C2", "C3", "C1"], CONDITIONS):
            cd = data[condition_k]
            metric = cumulative_topk_mass_per_step(cd.top_logprobs, k_eval)
            mean, lo, hi = bootstrap_position_mean(metric, n_resamples=2000)
            xs = np.arange(mean.shape[0])
            ax.plot(xs, _smooth(mean, 20), color=color, label=f"top-k = {condition_k}", lw=1.3)
            ax.fill_between(xs, _smooth(lo, 20), _smooth(hi, 20), color=color, alpha=0.15)
        ax.set_title(f"S_{k_eval} (cumulative top-{k_eval} mass)")
        ax.set_xlabel("generation position")
        ax.set_ylabel("probability mass")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.01)
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("Fig 3. Concentration of probability on the top-K vs. position", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig4_specificity_diagonal(data: dict, out_path: Path, late_start: int = 500) -> None:
    """For each condition, mean δ_r at late positions (i >= late_start), as a
    function of evaluation rank r. Hypothesis: condition K's curve peaks at r = K
    relative to the control curve."""
    ranks = np.arange(1, 21)

    fig, (ax_raw, ax_diff) = plt.subplots(1, 2, figsize=(12, 4.2))

    # Helper: for a condition, return (means, lows, highs) over ranks.
    def per_rank_stats(condition_k: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cd = data[condition_k]
        means, lows, highs = [], [], []
        for r in ranks:
            metric = _cliff_arr_per_run(cd.top_logprobs, r)[..., late_start:]
            late_mean_per_run = metric.mean(axis=-1, keepdims=True)  # (n_prompts, n_runs, 1)
            point, lo, hi = bootstrap_position_mean(late_mean_per_run, n_resamples=2000)
            means.append(point[0])
            lows.append(lo[0])
            highs.append(hi[0])
        return np.asarray(means), np.asarray(lows), np.asarray(highs)

    inf_means, inf_lows, inf_highs = per_rank_stats("inf")

    for color, condition_k in zip(["C0", "C2", "C3"], ("1", "3", "5")):
        m, lo, hi = per_rank_stats(condition_k)
        ax_raw.plot(ranks, m, color=color, label=f"top-k = {condition_k}", lw=1.4)
        ax_raw.fill_between(ranks, lo, hi, color=color, alpha=0.18)

        # Difference vs control. Compute as paired bootstrap on per-(prompt, run) late mean.
        ax_diff.plot(
            ranks,
            m - inf_means,
            color=color,
            label=f"top-k = {condition_k} minus top-k = ∞",
            lw=1.4,
        )

    ax_raw.plot(ranks, inf_means, color="C1", label="top-k = ∞ (control)", lw=1.4)
    ax_raw.fill_between(ranks, inf_lows, inf_highs, color="C1", alpha=0.18)
    ax_raw.set_xlabel("evaluation rank r")
    ax_raw.set_ylabel(f"mean δ_r over positions ≥ {late_start}")
    ax_raw.set_title("Raw cliff curves")
    ax_raw.legend(loc="best", fontsize=8)
    ax_raw.grid(alpha=0.3)

    ax_diff.axhline(0, color="black", lw=0.8)
    ax_diff.set_xlabel("evaluation rank r")
    ax_diff.set_ylabel("treatment − control (nats)")
    ax_diff.set_title("Specificity: bump at r = K?")
    ax_diff.legend(loc="best", fontsize=8)
    ax_diff.grid(alpha=0.3)

    fig.suptitle("Fig 4. Specificity diagonal — does the cliff peak at r = K?", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig6_rank_probability(data: dict, out_path: Path, late_start: int = 500) -> None:
    """For each condition, plot mean p_r at late positions as a function of rank r.

    Two panels:
        Left  — linear y, ranks 1..20: shows where the cliff sits.
        Right — log y, ranks 1..200: shows the full tail shape.

    CI bands via cluster bootstrap by prompt over per-(prompt, run) late means.
    """
    n_top_n = data["inf"].top_logprobs.shape[-1]
    ranks_left = np.arange(1, 21)
    ranks_right = np.arange(1, n_top_n + 1)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.4))
    color_map = {"1": "C0", "3": "C2", "5": "C3", "inf": "C1"}

    for condition_k in CONDITIONS:
        cd = data[condition_k]
        # Probabilities at every rank, late-position-mean per (prompt, run).
        # Shape: (n_prompts, n_runs, top_n)
        late_probs = np.exp(cd.top_logprobs[..., late_start:, :]).mean(axis=-2)

        # Bootstrap-mean per rank: feed (n_prompts, n_runs, n_ranks) into
        # bootstrap_position_mean which treats the last axis as positions.
        point, lo, hi = bootstrap_position_mean(late_probs, n_resamples=2000)
        color = color_map[condition_k]

        ax_lin.plot(
            ranks_left,
            point[: len(ranks_left)],
            color=color,
            label=f"top-k = {condition_k}",
            lw=1.5,
        )
        ax_lin.fill_between(
            ranks_left,
            lo[: len(ranks_left)],
            hi[: len(ranks_left)],
            color=color,
            alpha=0.2,
        )

        # Log-y panel uses every rank we saved.
        ax_log.plot(ranks_right, point, color=color, label=f"top-k = {condition_k}", lw=1.4)
        ax_log.fill_between(ranks_right, lo, hi, color=color, alpha=0.18)

        # Vertical guide lines at each treatment K.
        if condition_k != "inf":
            ax_lin.axvline(int(condition_k), color=color, ls=":", alpha=0.5, lw=1.0)
            ax_log.axvline(int(condition_k), color=color, ls=":", alpha=0.5, lw=1.0)

    ax_lin.set_xlabel("rank r")
    ax_lin.set_ylabel(f"E[p_r] over positions ≥ {late_start}")
    ax_lin.set_title("Linear y, ranks 1..20")
    ax_lin.legend(loc="best", fontsize=9)
    ax_lin.grid(alpha=0.3)
    ax_lin.set_xticks([1, 3, 5, 10, 15, 20])

    ax_log.set_xlabel("rank r (log)")
    ax_log.set_ylabel("E[p_r] (log)")
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_title(f"Log–log, full top-{n_top_n}")
    ax_log.legend(loc="best", fontsize=9)
    ax_log.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Fig 6. Probability assigned to rank r at late positions, by condition",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig5_forest(data: dict, out_path: Path, late_start: int = 500) -> None:
    """Forest plot of (treatment − control) cliff at r = K, averaged over late positions."""
    treatment_ks = ("1", "3", "5")

    inf_data = data["inf"]

    rows: list[tuple[str, float, float, float]] = []
    for k in treatment_ks:
        cd = data[k]
        rank = int(k)

        # Treatment metric averaged over late positions, shape (n_prompts, n_runs)
        t_metric = _cliff_arr_per_run(cd.top_logprobs, rank)[..., late_start:].mean(axis=-1)
        # Control metric averaged over late positions, shape (n_prompts, n_runs_inf)
        c_metric = _cliff_arr_per_run(inf_data.top_logprobs, rank)[..., late_start:].mean(axis=-1)
        n_runs_t = t_metric.shape[1]
        n_runs_c = c_metric.shape[1]
        # Pair seed-matched runs: in our seed scheme, run r of each condition uses
        # the same seed. For k=1 (n_runs=1) we broadcast against the first c-run.
        if n_runs_t == n_runs_c:
            paired_diff = t_metric - c_metric  # (n_prompts, n_runs)
        elif n_runs_t == 1:
            # Pair k=1 deterministic run against c_metric averaged over its runs.
            paired_diff = t_metric - c_metric.mean(axis=1, keepdims=True)
        else:
            raise ValueError(f"unexpected runs_t={n_runs_t} runs_c={n_runs_c}")

        # Bootstrap over prompts. The (n_prompts, n_runs) -> (n_prompts, n_runs, 1) trick.
        point, lo, hi = bootstrap_position_mean(
            paired_diff[..., None], n_resamples=10_000
        )
        rows.append((k, float(point[0]), float(lo[0]), float(hi[0])))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ys = np.arange(len(rows))[::-1]
    for y, (k, p, lo, hi) in zip(ys, rows):
        ax.errorbar(
            [p], [y],
            xerr=[[p - lo], [hi - p]],
            fmt="o", color="C0", capsize=4, lw=1.5,
        )
        ax.text(p, y + 0.18, f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]", ha="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"top-k = {r[0]}, rank = {r[0]}" for r in rows])
    ax.set_xlabel("(treatment − control) cliff δ_K at late positions (nats)")
    ax.set_title(f"Fig 5. Effect-size forest — late positions ≥ {late_start}, 95% CI")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------- main ---------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "figures",
    )
    parser.add_argument(
        "--late-start",
        type=int,
        default=500,
        help="positions >= this are treated as 'late' for late-position-mean figures",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="optional truncation of the position axis (e.g. to compare at fixed L)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading saved data per condition…")
    data: dict = {}
    for k in CONDITIONS:
        cd = load_condition(args.data_dir, k, max_length=args.max_length)
        print(f"  condition {k}: top_logprobs.shape = {cd.top_logprobs.shape}")
        data[k] = cd

    print("Plotting figures…")
    fig1_main_effect(data, args.out_dir / "fig1_main_effect.png")
    fig2_cliff_heatmap(data, args.out_dir / "fig2_cliff_heatmap.png")
    fig3_concentration(data, args.out_dir / "fig3_concentration.png")
    fig4_specificity_diagonal(
        data, args.out_dir / "fig4_specificity.png", late_start=args.late_start
    )
    fig5_forest(data, args.out_dir / "fig5_forest.png", late_start=args.late_start)
    fig6_rank_probability(
        data, args.out_dir / "fig6_rank_probability.png", late_start=args.late_start
    )
    print(f"All figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
