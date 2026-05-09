"""Compute and persist headline numbers for the topk experiment.

Two outputs are produced:

  results/tables/topk_summary.json
      Machine-readable, full-precision source of truth.
      Schema:
        {
          "provenance": { "script": ..., "data_dir": ..., "args": {...},
                          "n_prompts": ..., "L": ..., "late_start": ... },
          "h2": [{ "condition_k": "1", "rank": 1, "treatment": [pt, lo, hi],
                   "control":   [pt, lo, hi], "diff": [pt, lo, hi],
                   "verdict":   "supports H2 (treatment > control)" }, ...],
          "h1": [{ "condition_k": "...", "rank": ..., "early": [...],
                   "late": [...], "late_minus_early": [...] }, ...],
          "h3": [{ "treatment_k": "3",
                   "Δδ_K_minus_1": [...], "Δδ_K": [...], "Δδ_K_plus_1": [...],
                   "peaks_at_K": false }, ...]
        }

  results/summary.md
      Human-readable markdown rendered from the JSON. Numbers are read from
      the JSON and formatted; nothing is computed here. Provenance comment
      at the top.

Reads:
    data/manifest.parquet, data/traces/*.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from decoding_decoding.analyze import (
    bootstrap_position_mean,
    cliff_log_per_step,
    load_condition,
)


# ============================================================ stats helpers


def _verdict_h2(lo: float, hi: float) -> str:
    if lo > 0:
        return "supports H2 (treatment > control)"
    if hi < 0:
        return "REJECTS H2 (treatment < control)"
    return "null at this L (CI crosses zero)"


def _bootstrap_scalar(arr_per_pr: np.ndarray, n_resamples: int = 10_000) -> tuple[float, float, float]:
    """arr_per_pr shape (n_prompts, n_runs); bootstrap a scalar mean."""
    pt, lo, hi = bootstrap_position_mean(arr_per_pr[..., None], n_resamples=n_resamples)
    return float(pt[0]), float(lo[0]), float(hi[0])


def _paired_diff(t_pr: np.ndarray, c_pr: np.ndarray) -> np.ndarray:
    """Pair t_pr (n_prompts, n_runs_t) and c_pr (n_prompts, n_runs_c).
    If runs match, element-wise. If t has 1 run, average c across runs."""
    if t_pr.shape[1] == c_pr.shape[1]:
        return t_pr - c_pr
    if t_pr.shape[1] == 1:
        return t_pr - c_pr.mean(axis=1, keepdims=True)
    raise ValueError(f"unexpected runs shapes {t_pr.shape} vs {c_pr.shape}")


# ============================================================ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "summary.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "tables" / "topk_summary.json",
    )
    parser.add_argument("--late-start", type=int, default=500)
    parser.add_argument("--early-start", type=int, default=25)
    parser.add_argument("--early-end", type=int, default=75)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    data = {
        k: load_condition(args.data_dir, k, max_length=args.max_length)
        for k in ("1", "3", "5", "inf")
    }
    L = data["inf"].top_logprobs.shape[2]
    n_prompts = data["inf"].top_logprobs.shape[0]
    n_runs_inf = data["inf"].top_logprobs.shape[1]
    inf_data = data["inf"]

    # ---- H2 — treatment vs control cliff at r=K, late positions
    h2_rows: list[dict] = []
    for k in ("1", "3", "5"):
        rank = int(k)
        cd = data[k]
        t_pr = cliff_log_per_step(cd.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)
        c_pr = cliff_log_per_step(inf_data.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)

        t = _bootstrap_scalar(t_pr, n_resamples=args.n_resamples)
        c = _bootstrap_scalar(c_pr, n_resamples=args.n_resamples)
        d = _bootstrap_scalar(_paired_diff(t_pr, c_pr), n_resamples=args.n_resamples)

        h2_rows.append({
            "condition_k": k,
            "rank": rank,
            "treatment": list(t),
            "control": list(c),
            "diff": list(d),
            "verdict": _verdict_h2(d[1], d[2]),
        })

    # ---- H1 — concentration over time (early vs late within condition)
    h1_rows: list[dict] = []
    for k in ("1", "3", "5", "inf"):
        rank = int(k) if k != "inf" else 5
        cd = data[k]
        cliffs = cliff_log_per_step(cd.top_logprobs, rank)
        early = cliffs[..., args.early_start:args.early_end].mean(axis=-1)
        late = cliffs[..., args.late_start:].mean(axis=-1)
        e = _bootstrap_scalar(early, n_resamples=args.n_resamples)
        l = _bootstrap_scalar(late, n_resamples=args.n_resamples)
        d = _bootstrap_scalar(late - early, n_resamples=args.n_resamples)
        h1_rows.append({
            "condition_k": k,
            "rank": rank,
            "early": list(e),
            "late": list(l),
            "late_minus_early": list(d),
        })

    # ---- H3 — specificity: does (treatment − control) cliff peak at r=K?
    def paired_diff_at_rank(treatment_k: str, rank: int) -> tuple[float, float, float]:
        cd_t = data[treatment_k]
        t_pr = cliff_log_per_step(cd_t.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)
        c_pr = cliff_log_per_step(inf_data.top_logprobs, rank)[..., args.late_start:].mean(axis=-1)
        return _bootstrap_scalar(_paired_diff(t_pr, c_pr), n_resamples=args.n_resamples)

    h3_rows: list[dict] = []
    for k in ("3", "5"):
        K = int(k)
        d_minus = paired_diff_at_rank(k, K - 1)
        d_at = paired_diff_at_rank(k, K)
        d_plus = paired_diff_at_rank(k, K + 1)
        peaks = bool(d_at[0] > d_minus[0] and d_at[0] > d_plus[0])
        h3_rows.append({
            "treatment_k": k,
            "K": K,
            "delta_minus": list(d_minus),
            "delta_at": list(d_at),
            "delta_plus": list(d_plus),
            "peaks_at_K": peaks,
        })

    summary = {
        "provenance": {
            "script": "scripts/write_summary.py",
            "argv": sys.argv,
            "generated_at": dt.datetime.now().isoformat(),
            "data_dir": str(args.data_dir),
            "n_prompts": n_prompts,
            "L": L,
            "n_runs_per_condition": {"1": 1, "3": 8, "5": 8, "inf": int(n_runs_inf)},
            "late_start": args.late_start,
            "early_window": [args.early_start, args.early_end],
            "n_resamples": args.n_resamples,
            "model": "Qwen/Qwen2.5-3B (base, fp16)",
        },
        "h2": h2_rows,
        "h1": h1_rows,
        "h3": h3_rows,
    }

    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")

    # ---- Render markdown FROM the JSON.
    md = render_markdown(summary)
    args.out_md.write_text(md)
    print(f"wrote {args.out_md}")


def _fmt(triple: list[float]) -> str:
    pt, lo, hi = triple
    return f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def render_markdown(summary: dict) -> str:
    p = summary["provenance"]
    lines: list[str] = []
    lines.append(
        f"<!-- Generated by: {p['script']} {' '.join(p['argv'][1:])} -->"
    )
    lines.append(f"<!-- Source data: {p['data_dir']}/manifest.parquet + traces/ -->")
    lines.append(f"<!-- Generated at: {p['generated_at']} -->")
    lines.append(f"<!-- All numbers below are read from results/tables/topk_summary.json. -->\n")

    lines.append("# Top-K self-prediction — headline results\n")
    lines.append(
        f"Length L = {p['L']}, prompts = {p['n_prompts']}, runs per (prompt, condition) = "
        f"{p['n_runs_per_condition']}, model = {p['model']}.\n"
    )

    # H2
    lines.append(f"## H2 — treatment vs. control cliff at r = K, positions ≥ {p['late_start']} (95% cluster-bootstrap CI)\n")
    lines.append("| condition | δ_K (treatment) | δ_K (control top-k=∞) | difference (T − C) | verdict |")
    lines.append("|---|---|---|---|---|")
    for row in summary["h2"]:
        lines.append(
            f"| top-k = {row['condition_k']} | {_fmt(row['treatment'])} | {_fmt(row['control'])} "
            f"| {_fmt(row['diff'])} | {row['verdict']} |"
        )

    # H1
    lines.append("")
    lines.append("## H1 — concentration over time\n")
    lines.append(
        f"Cliff δ_K averaged early (positions {p['early_window'][0]}–{p['early_window'][1]}) "
        f"vs. late (positions ≥ {p['late_start']}), within each condition.\n"
    )
    lines.append("| condition | δ_K early | δ_K late | late − early |")
    lines.append("|---|---|---|---|")
    for row in summary["h1"]:
        lines.append(
            f"| top-k = {row['condition_k']} (rank {row['rank']}) | {_fmt(row['early'])} "
            f"| {_fmt(row['late'])} | {_fmt(row['late_minus_early'])} |"
        )

    # H3
    lines.append("")
    lines.append("## H3 — specificity (does the cliff peak at r = K?)\n")
    lines.append(
        f"For each treatment condition K, compare the (treatment − control) cliff at "
        f"the matched rank r=K against ranks r=K−1 and r=K+1, all averaged over "
        f"positions ≥ {p['late_start']}.\n"
    )
    lines.append("| condition | Δδ_{K-1} | Δδ_{K} | Δδ_{K+1} | peaks at r=K? |")
    lines.append("|---|---|---|---|---|")
    for row in summary["h3"]:
        peak = "yes" if row["peaks_at_K"] else "NO"
        lines.append(
            f"| top-k = {row['treatment_k']} | {_fmt(row['delta_minus'])} | "
            f"{_fmt(row['delta_at'])} | {_fmt(row['delta_plus'])} | {peak} |"
        )
    lines.append(
        "\nNote: top-k=1 omitted because rank 0 is undefined; the H3 contrast "
        "needs both neighbors."
    )

    # Caveats
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
    return "\n".join(lines)


if __name__ == "__main__":
    main()
