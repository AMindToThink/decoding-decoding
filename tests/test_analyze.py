"""Unit tests for analyze.py — vectorized metrics consistent with metrics.py."""

from __future__ import annotations

import numpy as np

from decoding_decoding.analyze import (
    cliff_log_per_step,
    cumulative_topk_mass_per_step,
    entropy_topn_per_step,
    smooth_rolling_mean,
)
from decoding_decoding.metrics import (
    cliff_log,
    cumulative_topk_mass,
    entropy_topn,
)


def _fixture_logprobs(n_prompts: int, n_runs: int, L: int, top_n: int) -> np.ndarray:
    """Random sorted-descending log-probabilities, shape (n_prompts, n_runs, L, top_n)."""
    rng = np.random.default_rng(0)
    raw = rng.random((n_prompts, n_runs, L, top_n)) + 1e-3
    raw = np.sort(raw, axis=-1)[..., ::-1]
    raw = raw / raw.sum(axis=-1, keepdims=True)
    return np.log(raw).astype(np.float32)


def test_vectorized_cliff_matches_metrics_module() -> None:
    arr = _fixture_logprobs(2, 3, 7, 30)
    for r in [1, 5, 20]:
        vec = cliff_log_per_step(arr, r)  # (2, 3, 7)
        for p in range(2):
            for run in range(3):
                expect = cliff_log(arr[p, run], r)
                np.testing.assert_allclose(vec[p, run], expect, rtol=1e-6)


def test_vectorized_cumulative_mass_matches_metrics_module() -> None:
    arr = _fixture_logprobs(2, 3, 5, 25)
    for k in [1, 5, 25]:
        vec = cumulative_topk_mass_per_step(arr, k)
        for p in range(2):
            for run in range(3):
                expect = cumulative_topk_mass(arr[p, run], k)
                np.testing.assert_allclose(vec[p, run], expect, rtol=1e-6)


def test_vectorized_entropy_matches_metrics_module() -> None:
    arr = _fixture_logprobs(2, 3, 5, 30)
    vec = entropy_topn_per_step(arr)
    for p in range(2):
        for run in range(3):
            expect = entropy_topn(arr[p, run])
            np.testing.assert_allclose(vec[p, run], expect, rtol=1e-6)


def test_smooth_rolling_mean_window_one_is_identity() -> None:
    arr = np.arange(10.0).reshape(1, -1)
    np.testing.assert_array_equal(smooth_rolling_mean(arr, 1), arr)


def test_smooth_rolling_mean_constant_is_constant() -> None:
    arr = np.full((2, 16), 3.5)
    np.testing.assert_allclose(smooth_rolling_mean(arr, 5)[:, 5:-5], 3.5)
