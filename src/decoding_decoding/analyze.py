"""Load saved generations and compute per-condition, per-position metrics + CIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from decoding_decoding.bootstrap import cluster_bootstrap_per_position


CONDITIONS: tuple[str, ...] = ("1", "3", "5", "inf")


@dataclass(frozen=True)
class ConditionData:
    """Stacked saved data for a single condition."""

    condition_k: str
    top_token_ids: np.ndarray  # (n_prompts, n_runs, L, top_n)  int32
    top_logprobs: np.ndarray   # (n_prompts, n_runs, L, top_n)  fp32

    @property
    def shape(self) -> tuple[int, ...]:
        return self.top_logprobs.shape


def load_condition(out_dir: Path, condition_k: str) -> ConditionData:
    """Stack all saved (prompt, run) npz files for one condition."""
    out_dir = Path(out_dir)
    meta = pl.read_parquet(out_dir / "run_metadata.parquet")
    sub = (
        meta.filter(pl.col("condition_k") == condition_k)
        .sort(["prompt_id", "run_idx"])
    )
    grouped = sub.group_by("prompt_id", maintain_order=True).agg(pl.col("logprobs_npz"))
    files_per_prompt = grouped["logprobs_npz"].to_list()
    n_prompts = len(files_per_prompt)
    runs_counts = {len(fs) for fs in files_per_prompt}
    if len(runs_counts) != 1:
        raise ValueError(
            f"condition {condition_k}: ragged runs per prompt ({runs_counts})"
        )
    n_runs = runs_counts.pop()

    lp_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    for prompt_files in files_per_prompt:
        per_run_lp: list[np.ndarray] = []
        per_run_id: list[np.ndarray] = []
        for npz_name in prompt_files:
            data = np.load(out_dir / "logprobs" / npz_name)
            per_run_lp.append(np.asarray(data["top_logprobs"]))
            per_run_id.append(np.asarray(data["top_token_ids"]))
        lp_blocks.append(np.stack(per_run_lp, axis=0))
        id_blocks.append(np.stack(per_run_id, axis=0))
    return ConditionData(
        condition_k=condition_k,
        top_token_ids=np.stack(id_blocks, axis=0),
        top_logprobs=np.stack(lp_blocks, axis=0),
    )


def cliff_log_per_step(top_logprobs: np.ndarray, r: int) -> np.ndarray:
    """Vectorized cliff at rank r over (..., L, top_n) → (..., L). r is 1-indexed."""
    if r < 1 or top_logprobs.shape[-1] < r + 1:
        raise ValueError(f"insufficient ranks for cliff at r={r}")
    return top_logprobs[..., r - 1] - top_logprobs[..., r]


def cumulative_topk_mass_per_step(top_logprobs: np.ndarray, k: int) -> np.ndarray:
    """Vectorized S_k = sum_{j<k} p_j over (..., L, top_n) → (..., L)."""
    if k < 1 or top_logprobs.shape[-1] < k:
        raise ValueError(f"insufficient ranks for top-k mass at k={k}")
    return np.exp(top_logprobs[..., :k]).sum(axis=-1)


def entropy_topn_per_step(top_logprobs: np.ndarray) -> np.ndarray:
    """Per-step entropy over the top-N saved entries (lower bound on true entropy)."""
    p = np.exp(top_logprobs)
    return -(p * top_logprobs).sum(axis=-1)


def bootstrap_position_mean(
    metric: np.ndarray,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrap cluster_bootstrap_per_position. Input shape (n_prompts, n_runs, L)."""
    return cluster_bootstrap_per_position(
        metric, n_resamples=n_resamples, rng_seed=rng_seed
    )


def smooth_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean along the last axis. Edge handling: same-array mode."""
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(arr, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    for i in range(flat.shape[0]):
        out_flat = np.convolve(flat[i], kernel, mode="same")
        out.reshape(-1, arr.shape[-1])[i] = out_flat
    return out
