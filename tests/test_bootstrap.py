"""Unit tests for bootstrap.py."""

from __future__ import annotations

import numpy as np

from decoding_decoding.bootstrap import (
    cluster_bootstrap_per_position,
    paired_cluster_bootstrap_per_position,
    position_mean,
)


def test_position_mean_matches_numpy_mean() -> None:
    rng = np.random.default_rng(0)
    metric = rng.normal(size=(7, 3, 50)).astype(np.float64)
    np.testing.assert_allclose(position_mean(metric), metric.mean(axis=(0, 1)))


def test_bootstrap_point_estimate_is_unweighted_population_mean() -> None:
    """The reported point estimate must equal the original population mean,
    not a bootstrap-resampled mean — the bootstrap is only for the CI."""
    rng = np.random.default_rng(1)
    metric = rng.normal(size=(8, 4, 32)).astype(np.float64)
    point, _, _ = cluster_bootstrap_per_position(metric, n_resamples=200)
    np.testing.assert_allclose(point, metric.mean(axis=(0, 1)), rtol=1e-12)


def test_bootstrap_ci_is_degenerate_for_constant_metric() -> None:
    metric = np.full((8, 4, 16), 3.5, dtype=np.float64)
    point, lo, hi = cluster_bootstrap_per_position(metric, n_resamples=200)
    np.testing.assert_allclose(point, 3.5)
    np.testing.assert_allclose(lo, 3.5)
    np.testing.assert_allclose(hi, 3.5)


def test_paired_bootstrap_ci_is_zero_for_identical_inputs() -> None:
    rng = np.random.default_rng(2)
    metric = rng.normal(size=(6, 3, 12)).astype(np.float64)
    point, lo, hi = paired_cluster_bootstrap_per_position(
        metric, metric, n_resamples=200
    )
    np.testing.assert_allclose(point, 0.0, atol=1e-12)
    np.testing.assert_allclose(lo, 0.0, atol=1e-12)
    np.testing.assert_allclose(hi, 0.0, atol=1e-12)


def test_bootstrap_ci_widens_with_smaller_n_prompts() -> None:
    """Sanity check: fewer prompts → wider CI (under same per-position variance)."""
    rng = np.random.default_rng(3)
    big = rng.normal(size=(40, 4, 20)).astype(np.float64)
    small = big[:8]  # take 8 prompts out of 40
    _, lo_big, hi_big = cluster_bootstrap_per_position(big, n_resamples=500, rng_seed=0)
    _, lo_sm, hi_sm = cluster_bootstrap_per_position(small, n_resamples=500, rng_seed=0)
    width_big = (hi_big - lo_big).mean()
    width_sm = (hi_sm - lo_sm).mean()
    assert width_sm > width_big, (
        f"expected wider CI with fewer prompts; got width small={width_sm:.4f} "
        f"vs big={width_big:.4f}"
    )
