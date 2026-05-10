"""Load saved generations and compute per-condition, per-position metrics + CIs.

Reads the unified layout described in `data_layout.py`:
    data/manifest.parquet           one row per trace
    data/traces/{trace_id}.npz      per-trace top_token_ids + top_logprobs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import uniform_filter1d

from decoding_decoding.bootstrap import cluster_bootstrap_per_position
from decoding_decoding.data_layout import MANIFEST_FILENAME, load_trace


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


def _stack_traces(
    out_dir: Path,
    sub: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a manifest subset filtered to one condition, stack per-trace npz
    arrays into (n_prompts, n_runs, L, top_n) blocks.

    All traces in `sub` must share the same `length` (truncation to the min
    is the caller's responsibility if not).
    """
    grouped = sub.group_by("prompt_id", maintain_order=True).agg(
        [pl.col("trace_id"), pl.col("run_idx")]
    )
    n_prompts = len(grouped)
    runs_counts = {len(rs) for rs in grouped["trace_id"].to_list()}
    if len(runs_counts) != 1:
        raise ValueError(f"ragged runs per prompt: {runs_counts}")

    lp_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    for trace_ids, run_idxs in zip(
        grouped["trace_id"].to_list(),
        grouped["run_idx"].to_list(),
        strict=True,
    ):
        # Order by run_idx to keep the run-axis stable across conditions.
        order = np.argsort(np.asarray(run_idxs))
        ordered_ids = [trace_ids[i] for i in order]
        per_run_lp: list[np.ndarray] = []
        per_run_id: list[np.ndarray] = []
        for tid in ordered_ids:
            ids, lps = load_trace(out_dir, tid)
            per_run_id.append(ids)
            per_run_lp.append(lps)
        id_blocks.append(np.stack(per_run_id, axis=0))
        lp_blocks.append(np.stack(per_run_lp, axis=0))
    _ = n_prompts  # documented for the reader
    return np.stack(id_blocks, axis=0), np.stack(lp_blocks, axis=0)


def load_condition(
    out_dir: Path,
    condition_k: str,
    *,
    family: str = "topk",
    max_length: int | None = None,
) -> ConditionData:
    """Stack all saved traces for one top-k condition.

    `max_length` truncates the position axis (useful when comparing the same
    saved data at L=1024 and L=2048 reproducibly).
    """
    out_dir = Path(out_dir)
    meta = pl.read_parquet(out_dir / MANIFEST_FILENAME)
    label = f"k={condition_k}"
    sub = (
        meta.filter((pl.col("family") == family) & (pl.col("condition_label") == label))
        .sort(["prompt_id", "run_idx"])
    )
    if len(sub) == 0:
        raise ValueError(
            f"no traces matching family={family} condition_label={label}"
        )
    lengths = set(sub["length"].to_list())
    if len(lengths) != 1:
        # Truncate to the min if asked to do so via max_length; otherwise fail.
        L_min = min(lengths)
        if max_length is None:
            raise ValueError(
                f"condition {condition_k}: traces have different lengths {lengths}; "
                f"pass max_length to truncate"
            )
        L_use = min(L_min, max_length)
    else:
        L_use = lengths.pop() if max_length is None else min(lengths.pop(), max_length)

    ids, lps = _stack_traces(out_dir, sub)
    return ConditionData(
        condition_k=condition_k,
        top_token_ids=ids[..., :L_use, :],
        top_logprobs=lps[..., :L_use, :],
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
    """Centered rolling mean along the last axis. Boundary mode 'nearest'
    extends the edge value, so a constant input stays constant end-to-end
    (avoids the zero-padding dilution that np.convolve(mode='same') causes)."""
    if window <= 1:
        return arr
    return uniform_filter1d(arr.astype(np.float64), size=window, axis=-1, mode="nearest")
