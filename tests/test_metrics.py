"""Unit tests for metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from decoding_decoding.metrics import (
    cliff_log,
    cliff_raw,
    cumulative_topk_mass,
    entropy_topn,
)


def _logprobs(rows: list[list[float]]) -> np.ndarray:
    """Helper: build a (L, N) log-probability array from a list of probability rows.

    Each row's probabilities are sorted descending and (loosely) treated as the
    top-N entries. Values are converted to natural log.
    """
    arr = np.asarray(rows, dtype=np.float64)
    assert (arr > 0).all(), "all probabilities must be > 0"
    # Verify descending sort within each row (input convention).
    assert (np.diff(arr, axis=1) <= 0).all(), "probabilities must be sorted descending"
    return np.log(arr).astype(np.float32)


def test_cliff_log_non_negative_on_sorted_input() -> None:
    rng = np.random.default_rng(0)
    raw = rng.random((50, 30)) + 1e-3
    raw = np.sort(raw, axis=1)[:, ::-1]
    raw = raw / raw.sum(axis=1, keepdims=True)
    logp = np.log(raw).astype(np.float32)
    for r in [1, 2, 5, 10, 20]:
        delta = cliff_log(logp, r)
        assert (delta >= -1e-7).all(), f"cliff_log negative at r={r}"


def test_cliff_log_fixture() -> None:
    logp = _logprobs([[0.5, 0.3, 0.1]])
    # delta_1 = log 0.5 - log 0.3 = log(5/3)
    expected_d1 = math.log(0.5) - math.log(0.3)
    expected_d2 = math.log(0.3) - math.log(0.1)
    np.testing.assert_allclose(cliff_log(logp, 1)[0], expected_d1, rtol=1e-6)
    np.testing.assert_allclose(cliff_log(logp, 2)[0], expected_d2, rtol=1e-6)


def test_cliff_raw_fixture() -> None:
    logp = _logprobs([[0.5, 0.3, 0.1]])
    np.testing.assert_allclose(cliff_raw(logp, 1)[0], 0.2, atol=1e-6)
    np.testing.assert_allclose(cliff_raw(logp, 2)[0], 0.2, atol=1e-6)


def test_cumulative_topk_mass_fixture() -> None:
    logp = _logprobs([[0.5, 0.3, 0.1, 0.05]])
    np.testing.assert_allclose(cumulative_topk_mass(logp, 1)[0], 0.5, atol=1e-6)
    np.testing.assert_allclose(cumulative_topk_mass(logp, 2)[0], 0.8, atol=1e-6)
    np.testing.assert_allclose(cumulative_topk_mass(logp, 3)[0], 0.9, atol=1e-6)


def test_entropy_topn_fixture() -> None:
    logp = _logprobs([[0.5, 0.5]])  # bernoulli
    expected = -2 * 0.5 * math.log(0.5)  # = log 2
    np.testing.assert_allclose(entropy_topn(logp)[0], expected, atol=1e-6)


def test_cliff_log_bounds_check() -> None:
    logp = _logprobs([[0.7, 0.3]])  # only 2 ranks
    with pytest.raises(ValueError):
        cliff_log(logp, 2)  # would need rank 3
    with pytest.raises(ValueError):
        cliff_log(logp, 0)


def test_mean_of_log_ratios_differs_from_log_of_mean_ratios() -> None:
    """Regression guard.

    The plan specifies aggregating cliff_log values across runs as a
    *mean of log-ratios*, NOT a log of the mean ratio. On any non-degenerate
    fixture these two quantities differ (Jensen's inequality), and we want the
    test suite to fail loudly if a future refactor accidentally swaps them.
    """
    run_a = _logprobs([[0.5, 0.1]])
    run_b = _logprobs([[0.9, 0.05]])
    cliffs = np.stack([cliff_log(run_a, 1), cliff_log(run_b, 1)])  # (2 runs, 1 step)

    mean_of_log_ratios = float(cliffs.mean(axis=0)[0])

    # The wrong aggregation: take ratios in linear space, average, then log.
    ratios = np.stack(
        [np.exp(cliff_log(run_a, 1)), np.exp(cliff_log(run_b, 1))]
    )
    log_of_mean_ratios = float(np.log(ratios.mean(axis=0))[0])

    assert not math.isclose(mean_of_log_ratios, log_of_mean_ratios, rel_tol=1e-3), (
        "mean-of-log-ratios coincides with log-of-mean-ratios on this fixture, "
        "which means we'd lose the discriminative test"
    )
    # The mean-of-log-ratios should be (log 5 + log 18) / 2.
    expected_mean_of_logs = (math.log(5.0) + math.log(0.9 / 0.05)) / 2
    np.testing.assert_allclose(mean_of_log_ratios, expected_mean_of_logs, rtol=1e-6)
