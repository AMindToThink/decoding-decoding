"""Statistical analysis for the blacklist experiment.

Hypothesis (H_blacklist):
  When sampling at top-k=∞, T=1 with token b's logit forced to -∞, the
  model's *unmodified* softmax probability on b — that is, the probability
  the model would have assigned absent the bias — should DECREASE over the
  course of the generation. The intuition: the absence of b in the past
  context is itself a statistical fingerprint, and the model may learn to
  expect that the next token also won't be b.

Stronger hypothesis: tokens that were more frequent in the unrestricted
baseline should show a faster / larger drop, because the absence of a
high-frequency token is more informative.

Per-step measurement:
  At each generation step i in a blacklisted-rank-r trace, record p_b(i)
  = softmax probability the model assigns to the banned token. (We read
  this off `top_logprobs` if b is in the saved top-200; otherwise we use
  the smallest top-200 prob as an upper bound, which strongly suppressed
  cases will hit.)

Estimators:
  * mean_p_b(i) over (prompt, run) per position; cluster bootstrap by prompt.
  * effect: late-mean − early-mean per condition (paired by (prompt, run)).
  * cross-rank monotonicity: ordering of effects across banned ranks.

Outputs (machine-readable, importable):
  results/tables/blacklist_summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from decoding_decoding.bootstrap import cluster_bootstrap_per_position
from decoding_decoding.data_layout import MANIFEST_FILENAME, decode_params, load_trace


@dataclass(frozen=True)
class BlacklistConditionData:
    banned_rank: int
    banned_token_id: int
    # (n_prompts, n_runs, L) — model's softmax prob on the banned token.
    p_banned: np.ndarray
    # (n_prompts, n_runs, L) — boolean: was the banned token in top-200 at this step?
    in_topn: np.ndarray
    # The lower bound (= min of top-N probs) used when the banned token was OUT of top-N.
    floor_p: np.ndarray   # (n_prompts, n_runs, L)


def load_blacklist_condition(
    data_dir: Path, banned_rank: int
) -> BlacklistConditionData:
    """Stack per-trace data for a single banned-rank condition.

    Returns an object whose `p_banned[p, r, i]` is the softmax probability
    on the banned token at step i (with `in_topn[p, r, i]` indicating whether
    the value was read directly from the saved top-N or was clipped to the
    floor probability).
    """
    data_dir = Path(data_dir)
    manifest = pl.read_parquet(data_dir / MANIFEST_FILENAME)
    sub = manifest.filter(
        (pl.col("family") == "blacklist")
        & (pl.col("condition_label") == f"blacklist_rank{banned_rank}")
    ).sort(["prompt_id", "run_idx"])
    if len(sub) == 0:
        raise ValueError(f"no traces matching blacklist_rank{banned_rank}")
    grouped = sub.group_by("prompt_id", maintain_order=True).agg(
        [pl.col("trace_id"), pl.col("run_idx"), pl.col("params_json")]
    )
    runs_counts = {len(r) for r in grouped["trace_id"].to_list()}
    if len(runs_counts) != 1:
        raise ValueError(f"ragged runs per prompt: {runs_counts}")

    p_blocks: list[np.ndarray] = []
    intop_blocks: list[np.ndarray] = []
    floor_blocks: list[np.ndarray] = []

    banned_token_id: int | None = None
    for tids, ridxs, pjsons in zip(
        grouped["trace_id"].to_list(),
        grouped["run_idx"].to_list(),
        grouped["params_json"].to_list(),
        strict=True,
    ):
        order = np.argsort(np.asarray(ridxs))
        ordered = [(tids[i], pjsons[i]) for i in order]
        prompt_p: list[np.ndarray] = []
        prompt_intop: list[np.ndarray] = []
        prompt_floor: list[np.ndarray] = []
        for tid, pjson in ordered:
            params = decode_params(pjson)
            tok = int(params["banned_token_id"])
            if banned_token_id is None:
                banned_token_id = tok
            elif banned_token_id != tok:
                raise ValueError(
                    f"trace {tid}: banned_token_id {tok} != expected {banned_token_id}"
                )
            ids, lps = load_trace(data_dir, tid)
            # Find banned token's column at each step (or -1).
            mask = ids == tok                    # (L, top_n)
            in_topn = mask.any(axis=1)           # (L,)
            # Index of the column where banned token sits (0 where missing).
            col = mask.argmax(axis=1)
            row = np.arange(ids.shape[0])
            banned_lp = lps[row, col]
            # Where missing, use min-of-row (the floor), as an upper bound.
            floor = lps[:, -1]
            banned_lp = np.where(in_topn, banned_lp, floor)
            prompt_p.append(np.exp(banned_lp))
            prompt_intop.append(in_topn)
            prompt_floor.append(np.exp(floor))
        p_blocks.append(np.stack(prompt_p, axis=0))
        intop_blocks.append(np.stack(prompt_intop, axis=0))
        floor_blocks.append(np.stack(prompt_floor, axis=0))

    return BlacklistConditionData(
        banned_rank=banned_rank,
        banned_token_id=int(banned_token_id) if banned_token_id is not None else -1,
        p_banned=np.stack(p_blocks, axis=0),
        in_topn=np.stack(intop_blocks, axis=0),
        floor_p=np.stack(floor_blocks, axis=0),
    )


def baseline_p_for_token(
    data_dir: Path, token_id: int
) -> np.ndarray:
    """Per-position mean p(token_id) under the unrestricted top-k=∞ control.

    Returns shape (n_prompts, n_runs, L). For positions where the token wasn't
    in the saved top-200, we substitute the row's floor probability (a known
    upper bound on missing tokens).
    """
    data_dir = Path(data_dir)
    manifest = pl.read_parquet(data_dir / MANIFEST_FILENAME)
    sub = manifest.filter(
        (pl.col("family") == "topk") & (pl.col("condition_label") == "k=inf")
    ).sort(["prompt_id", "run_idx"])
    grouped = sub.group_by("prompt_id", maintain_order=True).agg(
        [pl.col("trace_id"), pl.col("run_idx")]
    )

    blocks: list[np.ndarray] = []
    for tids, ridxs in zip(
        grouped["trace_id"].to_list(),
        grouped["run_idx"].to_list(),
        strict=True,
    ):
        order = np.argsort(np.asarray(ridxs))
        per_run: list[np.ndarray] = []
        for tid in [tids[i] for i in order]:
            ids, lps = load_trace(data_dir, tid)
            mask = ids == token_id
            in_topn = mask.any(axis=1)
            col = mask.argmax(axis=1)
            row = np.arange(ids.shape[0])
            tok_lp = lps[row, col]
            floor = lps[:, -1]
            tok_lp = np.where(in_topn, tok_lp, floor)
            per_run.append(np.exp(tok_lp))
        blocks.append(np.stack(per_run, axis=0))
    return np.stack(blocks, axis=0)


def position_mean_with_ci(
    metric: np.ndarray, n_resamples: int = 10_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """metric: (n_prompts, n_runs, L) -> (point, lo, hi) each (L,)."""
    return cluster_bootstrap_per_position(metric, n_resamples=n_resamples)


def smooth_along_position(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean along the last axis."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.empty_like(flat, dtype=np.float64)
    for i in range(flat.shape[0]):
        out[i] = np.convolve(flat[i], kernel, mode="same")
    return out.reshape(arr.shape).astype(arr.dtype)
