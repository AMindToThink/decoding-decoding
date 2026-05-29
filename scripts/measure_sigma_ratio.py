"""Measure σ_ε / σ_* from existing N=3 sparse-logits data.

The per-prompt SVD noise floor is

    R_noise ≈ (K-1) · r² / N / (‖β‖² + K · r² / N)

where r = σ_ε / σ_*:
  * σ_ε is the per-entry trajectory-to-trajectory standard deviation of the
    centered logits at fixed (prompt, β_dec, t) — i.e., across the N runs.
  * σ_* is the per-entry spread of the natural-direction logits across vocab
    (the "signal" magnitude — RMS of centered logits within one ℓ_t vector).
  * K = number of decoder values in the sweep, ‖β‖² = Σ_k β_k².

If r ≈ 1, my original R_aggregation_caveat.tex floor estimate (~1.6% at N=30)
holds. If r ≫ 1 (the critic's worry), the floor scales as r² and the planned
N=30 rerun gives no improvement over N=3.

Reads from data/counter_decode/sparse_logits/, writes a summary table and a
PNG heatmap to results/counter_decode/sigma_ratio/.

Usage:
    uv run python scripts/measure_sigma_ratio.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from decoding_decoding.data_layout import MANIFEST_FILENAME, decode_params


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "counter_decode"
OUT_DIR = REPO_ROOT / "results" / "counter_decode" / "sigma_ratio"


def _gather_uncorrected_traces(manifest: pl.DataFrame) -> dict[tuple[int, float, int], str]:
    """Returns {(prompt_id, beta_dec, run_idx): trace_id} for the uncorrected arm only."""
    out: dict[tuple[int, float, int], str] = {}
    for row in manifest.filter(pl.col("family") == "counter_decode").iter_rows(named=True):
        params = decode_params(row["params_json"])
        if bool(params["corrected"]):
            continue
        key = (int(row["prompt_id"]), float(params["beta_target"]), int(row["run_idx"]))
        out[key] = row["trace_id"]
    return out


def _load_sparse_block(
    data_dir: Path, trace_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (positions, logits) for one trace's sparse_logits file."""
    npz = np.load(data_dir / "sparse_logits" / f"{trace_id}.npz")
    return np.asarray(npz["positions"], dtype=np.int32), np.asarray(
        npz["logits"], dtype=np.float32
    )


def _compute_per_condition(
    runs: np.ndarray,
) -> tuple[float, float]:
    """For one (prompt, β_dec, t) cell, compute (σ_ε, σ_*).

    Args:
        runs: (N, V) float32 — per-run logit vectors at one position.

    Returns:
        sigma_eps: RMS over vocab entries of (per-entry std across N runs).
        sigma_star: std over vocab entries of the across-runs averaged centered
            logit (i.e., spread of the recovered natural direction).
    """
    # Per-vector centering: softmax-invariant.
    centered = runs - runs.mean(axis=-1, keepdims=True)
    # σ_ε per entry: std across N runs.
    std_per_entry = centered.std(axis=0, ddof=1)  # (V,)
    sigma_eps = float(np.sqrt(np.mean(std_per_entry ** 2)))
    # σ_*: spread (std over vocab) of the across-runs averaged centered logits.
    avg = centered.mean(axis=0)  # (V,)
    sigma_star = float(avg.std(ddof=0))
    return sigma_eps, sigma_star


def _noise_floor(r: float, *, K: int, beta_norm_sq: float, N: int) -> float:
    """Predicted per-prompt SVD noise floor for the given r = σ_ε/σ_*."""
    num = (K - 1) * r * r / N
    den = beta_norm_sq + K * r * r / N
    return num / den


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[sigma-ratio] reading manifest from {DATA_DIR}")
    manifest = pl.read_parquet(DATA_DIR / MANIFEST_FILENAME)
    by_key = _gather_uncorrected_traces(manifest)

    # Index by (prompt, β_dec) → list of (positions, (P,V) logits) sorted by run_idx.
    by_pk: dict[tuple[int, float], list[tuple[int, np.ndarray]]] = defaultdict(list)
    positions_canonical: np.ndarray | None = None
    for (pid, bk, ridx), tid in by_key.items():
        pos, lps = _load_sparse_block(DATA_DIR, tid)
        if positions_canonical is None:
            positions_canonical = pos
        elif not np.array_equal(positions_canonical, pos):
            raise ValueError(f"position mismatch in {tid}: {pos} vs {positions_canonical}")
        by_pk[(pid, bk)].append((ridx, lps))

    assert positions_canonical is not None
    positions = positions_canonical.tolist()

    beta_values = sorted({bk for _, bk in by_pk.keys()})
    prompt_ids = sorted({pid for pid, _ in by_pk.keys()})

    K = len(beta_values)
    beta_norm_sq = float(sum(bv * bv for bv in beta_values))
    print(
        f"[sigma-ratio] K={K} β_values={beta_values} ‖β‖²={beta_norm_sq:.2f} "
        f"prompts={len(prompt_ids)} positions={positions}"
    )

    # Compute σ_ε, σ_*, r per (prompt, β_dec, t).
    # Shape: per_prompt[pid, bi, ti] → (sigma_eps, sigma_star)
    n_prompts = len(prompt_ids)
    n_betas = len(beta_values)
    n_pos = len(positions)
    sigma_eps_arr = np.zeros((n_prompts, n_betas, n_pos), dtype=np.float64)
    sigma_star_arr = np.zeros((n_prompts, n_betas, n_pos), dtype=np.float64)

    for pi, pid in enumerate(prompt_ids):
        for bi, bk in enumerate(beta_values):
            runs_for_pk = sorted(by_pk[(pid, bk)], key=lambda x: x[0])
            stack = np.stack([lps for _, lps in runs_for_pk], axis=0)  # (N, P_pos, V)
            for ti in range(n_pos):
                eps, star = _compute_per_condition(stack[:, ti, :])
                sigma_eps_arr[pi, bi, ti] = eps
                sigma_star_arr[pi, bi, ti] = star

    # Per (β_dec, t) median across prompts.
    ratio_arr = sigma_eps_arr / np.maximum(sigma_star_arr, 1e-9)
    ratio_med = np.median(ratio_arr, axis=0)  # (n_betas, n_pos)
    ratio_q25 = np.quantile(ratio_arr, 0.25, axis=0)
    ratio_q75 = np.quantile(ratio_arr, 0.75, axis=0)
    eps_med = np.median(sigma_eps_arr, axis=0)
    star_med = np.median(sigma_star_arr, axis=0)

    # Summary rows for a table.
    rows: list[dict] = []
    for bi, bk in enumerate(beta_values):
        for ti, t in enumerate(positions):
            r = float(ratio_med[bi, ti])
            row = {
                "beta_dec": float(bk),
                "t": int(t),
                "sigma_eps_median": float(eps_med[bi, ti]),
                "sigma_star_median": float(star_med[bi, ti]),
                "ratio_r_median": r,
                "ratio_r_q25": float(ratio_q25[bi, ti]),
                "ratio_r_q75": float(ratio_q75[bi, ti]),
                "R_noise_floor_N3": _noise_floor(r, K=K, beta_norm_sq=beta_norm_sq, N=3),
                "R_noise_floor_N10": _noise_floor(r, K=K, beta_norm_sq=beta_norm_sq, N=10),
                "R_noise_floor_N30": _noise_floor(r, K=K, beta_norm_sq=beta_norm_sq, N=30),
                "R_noise_floor_N100": _noise_floor(r, K=K, beta_norm_sq=beta_norm_sq, N=100),
            }
            rows.append(row)

    # Write JSON summary.
    summary = {
        "K": K,
        "beta_values": beta_values,
        "beta_norm_sq": beta_norm_sq,
        "n_prompts": n_prompts,
        "positions": positions,
        "rows": rows,
    }
    (OUT_DIR / "sigma_ratio.json").write_text(json.dumps(summary, indent=2))
    print(f"[sigma-ratio] wrote {OUT_DIR / 'sigma_ratio.json'}")

    # Print a compact table.
    print()
    print(
        f"{'β_dec':>6} {'t':>5} {'σ_ε':>8} {'σ_*':>8} {'r':>6} "
        f"{'floor@N=3':>11} {'@N=10':>9} {'@N=30':>9} {'@N=100':>9}"
    )
    print("-" * 80)
    for row in rows:
        print(
            f"{row['beta_dec']:>6.2f} {row['t']:>5d} "
            f"{row['sigma_eps_median']:>8.3f} {row['sigma_star_median']:>8.3f} "
            f"{row['ratio_r_median']:>6.2f} "
            f"{row['R_noise_floor_N3']:>11.3f} {row['R_noise_floor_N10']:>9.3f} "
            f"{row['R_noise_floor_N30']:>9.3f} {row['R_noise_floor_N100']:>9.3f}"
        )

    # Heatmap of r over (β_dec, t).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    im = ax.imshow(
        ratio_med, aspect="auto", origin="lower", cmap="magma",
        extent=[-0.5, n_pos - 0.5, -0.5, n_betas - 0.5],
    )
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([str(t) for t in positions])
    ax.set_yticks(range(n_betas))
    ax.set_yticklabels([f"{b:.2f}" for b in beta_values])
    ax.set_xlabel("t (sparse position)")
    ax.set_ylabel(r"$\beta_{\rm dec}$")
    ax.set_title(r"Median $r = \sigma_\varepsilon / \sigma_*$ across prompts")
    fig.colorbar(im, ax=ax, label="r")

    ax = axes[1]
    # Predicted N=30 floor as a heatmap.
    floor_30 = np.zeros_like(ratio_med)
    for bi in range(n_betas):
        for ti in range(n_pos):
            floor_30[bi, ti] = _noise_floor(
                float(ratio_med[bi, ti]), K=K, beta_norm_sq=beta_norm_sq, N=30
            )
    im = ax.imshow(
        floor_30, aspect="auto", origin="lower", cmap="viridis",
        extent=[-0.5, n_pos - 0.5, -0.5, n_betas - 0.5],
    )
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([str(t) for t in positions])
    ax.set_yticks(range(n_betas))
    ax.set_yticklabels([f"{b:.2f}" for b in beta_values])
    ax.set_xlabel("t (sparse position)")
    ax.set_ylabel(r"$\beta_{\rm dec}$")
    ax.set_title(r"Predicted per-prompt $R$ noise floor at $N=30$")
    fig.colorbar(im, ax=ax, label="R floor")

    fig.suptitle(
        "σ_ε / σ_* and predicted N=30 noise floor (existing N=3 sparse logits)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sigma_ratio_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[sigma-ratio] wrote {OUT_DIR / 'sigma_ratio_heatmap.png'}")

    # Also a small line plot: r(t) for each β_dec.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    cmap = plt.get_cmap("inferno")
    for bi, bk in enumerate(beta_values):
        c = cmap(0.15 + 0.65 * bi / max(1, n_betas - 1))
        ax.plot(positions, ratio_med[bi, :], "o-", color=c, label=rf"$\beta_{{\rm dec}}={bk}$")
        ax.fill_between(
            positions, ratio_q25[bi, :], ratio_q75[bi, :], color=c, alpha=0.12
        )
    ax.set_xlabel(r"$t$ (sparse position)")
    ax.set_ylabel(r"$r = \sigma_\varepsilon / \sigma_*$ (median across prompts)")
    ax.set_title(r"Trajectory-to-trajectory noise ratio (IQR across prompts)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ratio_over_t.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[sigma-ratio] wrote {OUT_DIR / 'ratio_over_t.png'}")


if __name__ == "__main__":
    main()
