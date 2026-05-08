"""Per-step metrics computed from saved top-N logprobs.

Input convention (matches what we save to .npz from the vLLM harness):
    top_logprobs: np.ndarray of shape (L, N), dtype float32.
        top_logprobs[i, j] = log p(token at rank j+1 | prefix at step i),
        sorted descending in j (so j=0 is the most probable token).
        Values are NATURAL log-probabilities from the model's *unmodified*
        softmax (vLLM `logprobs_mode="raw_logprobs"`), not from any post-
        truncation distribution.

All public functions are pure and return arrays of shape (L,).
"""

from __future__ import annotations

import numpy as np


def cliff_log(top_logprobs: np.ndarray, r: int) -> np.ndarray:
    """Per-step log-ratio cliff at rank r:  delta_r[i] = log p_r - log p_{r+1}.

    Sorted-descending input guarantees delta_r >= 0.
    r is 1-indexed (r=1 is the gap between rank 1 and rank 2).
    """
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    if top_logprobs.ndim != 2:
        raise ValueError(f"top_logprobs must be 2D (L, N), got shape {top_logprobs.shape}")
    if top_logprobs.shape[1] < r + 1:
        raise ValueError(
            f"need at least r+1={r + 1} logprob columns to compute cliff at rank {r}, "
            f"got {top_logprobs.shape[1]}"
        )
    return top_logprobs[:, r - 1] - top_logprobs[:, r]


def cliff_raw(top_logprobs: np.ndarray, r: int) -> np.ndarray:
    """Per-step raw-probability cliff at rank r:  Delta_r[i] = p_r - p_{r+1}."""
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    if top_logprobs.shape[1] < r + 1:
        raise ValueError(
            f"need at least r+1={r + 1} logprob columns, got {top_logprobs.shape[1]}"
        )
    return np.exp(top_logprobs[:, r - 1]) - np.exp(top_logprobs[:, r])


def cumulative_topk_mass(top_logprobs: np.ndarray, k: int) -> np.ndarray:
    """Per-step sum of probabilities over the top-k entries:  S_k[i] = sum_{j<k} p_j."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if top_logprobs.shape[1] < k:
        raise ValueError(f"need at least k={k} logprob columns, got {top_logprobs.shape[1]}")
    return np.exp(top_logprobs[:, :k]).sum(axis=1)


def entropy_topn(top_logprobs: np.ndarray) -> np.ndarray:
    """Per-step entropy over the top-N entries: H[i] = -sum_j p_j log p_j.

    Note: this is a lower bound on the true entropy of the full vocabulary,
    since we only have access to the top-N probabilities. Use as a relative
    measure across conditions, not as an absolute entropy estimate.
    """
    p = np.exp(top_logprobs)
    # element-wise p * logp; logp is already given.
    return -(p * top_logprobs).sum(axis=1)
