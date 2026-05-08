"""Write a one-page markdown summary of the headline effect sizes and verdicts.

Reads:    data/run_metadata.parquet, data/logprobs/*.npz
Writes:   results/summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from decoding_decoding.analyze import (
    bootstrap_position_mean,
    cliff_log_per_step,
    load_condition,
)


def _verdict(point: float, lo: float, hi: float) -> str:
    if lo > 0:
        return "supports H2 (treatment > control)"
    if hi < 0:
        return "REJECTS H2 (treatment < control)"
    return "null at this L (CI crosses zero)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "summary.md",
    )
    parser.add_argument("--late-start", type=int, default=500)
    args = parser.parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {k: load_condition(args.data_dir, k) for k in ("1", "3", "5", "inf")}
    L = data["inf"].top_logprobs.shape[2]
    n_prompts = data["inf"].top_logprobs.shape[0]
    n_runs_inf = data["inf"].top_logprobs.shape[1]

    lines: list[str] = []
    lines.append("# Top-K self-prediction — headline results\n")
    lines.append(
        f"Length L = {L}, prompts = {n_prompts}, runs per (prompt, condition) = "
        f"{{1: 1, 3: 8, 5: 8, inf: {n_runs_inf}}}, model = Qwen/Qwen2.5-3B (base, fp16).\n"
    )

    # Per-condition cliff at rank K, late-position mean.
    lines.append("## H2 — treatment vs. control cliff at r = K, positions ≥ "
                 f"{args.late_start} (95% cluster-bootstrap CI)\n")
    lines.append("| condition | δ_K (treatment) | δ_K (control top-k=∞) | difference (T − C) | verdict |")
    lines.append("|---|---|---|---|---|")

    inf_data = data["inf"]
    for k in ("1", "3", "5"):
        rank = int(k)
        cd = data[k]

        t_metric = cliff_log_per_step(cd.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)
        c_metric = cliff_log_per_step(inf_data.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)
        # Bootstrap each separately, plus the difference.
        t_pt, t_lo, t_hi = bootstrap_position_mean(t_metric[..., None], n_resamples=10_000)
        c_pt, c_lo, c_hi = bootstrap_position_mean(c_metric[..., None], n_resamples=10_000)

        # Paired difference: align run dims (k=1 has runs=1; broadcast against c_metric mean).
        if t_metric.shape[1] == c_metric.shape[1]:
            paired = t_metric - c_metric
        elif t_metric.shape[1] == 1:
            paired = t_metric - c_metric.mean(axis=1, keepdims=True)
        else:
            raise ValueError(f"unexpected runs shapes {t_metric.shape} vs {c_metric.shape}")
        d_pt, d_lo, d_hi = bootstrap_position_mean(paired[..., None], n_resamples=10_000)

        lines.append(
            f"| top-k = {k} | {float(t_pt[0]):+.4f} [{float(t_lo[0]):+.4f}, {float(t_hi[0]):+.4f}] "
            f"| {float(c_pt[0]):+.4f} [{float(c_lo[0]):+.4f}, {float(c_hi[0]):+.4f}] "
            f"| {float(d_pt[0]):+.4f} [{float(d_lo[0]):+.4f}, {float(d_hi[0]):+.4f}] "
            f"| {_verdict(float(d_pt[0]), float(d_lo[0]), float(d_hi[0]))} |"
        )

    lines.append("")
    lines.append("## H1 — concentration over time")
    lines.append("Cliff δ_K averaged early (positions 25–75) vs. late (positions ≥ 500), within each condition.\n")
    lines.append("| condition | δ_K early | δ_K late | late − early |")
    lines.append("|---|---|---|---|")

    for k in ("1", "3", "5", "inf"):
        rank = int(k) if k != "inf" else 5  # for inf condition, evaluate at rank 5
        cd = data[k]
        cliffs = cliff_log_per_step(cd.top_logprobs, rank)
        early = cliffs[..., 25:75].mean(axis=-1)
        late = cliffs[..., args.late_start:].mean(axis=-1)
        e_pt, e_lo, e_hi = bootstrap_position_mean(early[..., None], n_resamples=10_000)
        l_pt, l_lo, l_hi = bootstrap_position_mean(late[..., None], n_resamples=10_000)
        diff_pt, diff_lo, diff_hi = bootstrap_position_mean(
            (late - early)[..., None], n_resamples=10_000
        )
        lines.append(
            f"| top-k = {k} (rank {rank}) "
            f"| {float(e_pt[0]):+.4f} [{float(e_lo[0]):+.4f}, {float(e_hi[0]):+.4f}] "
            f"| {float(l_pt[0]):+.4f} [{float(l_lo[0]):+.4f}, {float(l_hi[0]):+.4f}] "
            f"| {float(diff_pt[0]):+.4f} [{float(diff_lo[0]):+.4f}, {float(diff_hi[0]):+.4f}] |"
        )

    # H3 — specificity: at late positions, is (treatment − control) δ_r largest exactly at r = K?
    lines.append("")
    lines.append("## H3 — specificity (does the cliff peak at r = K?)\n")
    lines.append(
        "For each treatment condition K, compare the (treatment − control) cliff at "
        f"the matched rank r=K against ranks r=K−1 and r=K+1, all averaged over positions ≥ {args.late_start}.\n"
    )
    lines.append("| condition | Δδ_{K-1} | Δδ_{K} | Δδ_{K+1} | peaks at r=K? |")
    lines.append("|---|---|---|---|---|")

    def _paired_diff_at_rank(treatment_k: str, rank: int) -> tuple[float, float, float]:
        cd_t = data[treatment_k]
        cd_c = inf_data
        t = cliff_log_per_step(cd_t.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)
        c = cliff_log_per_step(cd_c.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)
        if t.shape[1] == c.shape[1]:
            paired = t - c
        elif t.shape[1] == 1:
            paired = t - c.mean(axis=1, keepdims=True)
        else:
            raise ValueError(f"unexpected runs shapes {t.shape} vs {c.shape}")
        pt, lo, hi = bootstrap_position_mean(paired[..., None], n_resamples=10_000)
        return float(pt[0]), float(lo[0]), float(hi[0])

    for k in ("3", "5"):
        K = int(k)
        d_minus = _paired_diff_at_rank(k, K - 1)
        d_at = _paired_diff_at_rank(k, K)
        d_plus = _paired_diff_at_rank(k, K + 1)
        peaks = d_at[0] > d_minus[0] and d_at[0] > d_plus[0]
        lines.append(
            f"| top-k = {k} "
            f"| {d_minus[0]:+.4f} [{d_minus[1]:+.4f}, {d_minus[2]:+.4f}] "
            f"| {d_at[0]:+.4f} [{d_at[1]:+.4f}, {d_at[2]:+.4f}] "
            f"| {d_plus[0]:+.4f} [{d_plus[1]:+.4f}, {d_plus[2]:+.4f}] "
            f"| {'yes' if peaks else 'NO'} |"
        )
    # k=1 has no left-neighbor (rank 0 is undefined), so skip the symmetric test for it.
    lines.append(
        "\nNote: top-k=1 omitted because rank 0 is undefined; the H3 contrast needs both neighbors."
    )

    lines.append("")
    lines.append("## Caveats")
    lines.append(
        "- **k=1 effect is partly tautological.** Greedy decoding yields deterministic, "
        "highly predictable text; large δ_1 may reflect the model's general predictability "
        "on its own greedy continuations rather than self-recognition of the sampling regime. "
        "The k=3 and k=5 contrasts are the cleaner tests."
    )
    lines.append(
        "- **fp16 noise floor on Turing.** vLLM Triton kernels in fp16 give ~1e-2 nats "
        "drift on individual logprobs across batch positions; cliffs (consecutive differences) "
        "absorb the additive log-Z drift. The reported CIs reflect cluster-bootstrap variance "
        "across prompts, which is well above this noise floor."
    )
    lines.append("")
    lines.append("Figures: see `results/figures/`.")

    args.out_path.write_text("\n".join(lines))
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
