"""Cluster bootstrap by prompt.

We have repeated measurements per prompt (multiple runs sharing the same
prompt context). Resampling at the (prompt, run) level would underestimate
variance because runs within a prompt are correlated. Cluster bootstrap
resamples whole prompts (with all of their runs preserved together) and
recomputes the position-wise mean each time.

The point estimate is the unweighted mean over the original (prompt, run)
pairs at each position — bootstrap is used only to derive the CI.
"""

from __future__ import annotations

import numpy as np


def _validate_3d(metric: np.ndarray, name: str) -> None:
    if metric.ndim != 3:
        raise ValueError(
            f"{name} must be 3D (n_prompts, n_runs, n_positions), got shape {metric.shape}"
        )


def position_mean(metric: np.ndarray) -> np.ndarray:
    """Unweighted mean across (prompt, run) per position. Shape (n_positions,)."""
    _validate_3d(metric, "metric")
    return metric.mean(axis=(0, 1))


def cluster_bootstrap_per_position(
    metric: np.ndarray,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster bootstrap CI for a single-condition metric.

    Args:
        metric: shape (n_prompts, n_runs, n_positions). NaNs are not handled.
        n_resamples: number of bootstrap resamples (B).
        rng_seed: seed for the resampling RNG.
        confidence: e.g. 0.95 for a 95% percentile CI.

    Returns:
        (point_estimate, ci_low, ci_high), each of shape (n_positions,).
        point_estimate is the unweighted mean over (prompt, run); the CI is
        derived from the bootstrap distribution of the same statistic under
        prompt-level resampling.
    """
    _validate_3d(metric, "metric")
    n_prompts = metric.shape[0]
    rng = np.random.default_rng(rng_seed)

    point = position_mean(metric)

    # Bootstrap loop: resample prompt indices with replacement.
    boot_means = np.empty((n_resamples, metric.shape[2]), dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        boot_means[b] = metric[idx].mean(axis=(0, 1))

    alpha = (1.0 - confidence) / 2.0
    ci_low = np.quantile(boot_means, alpha, axis=0)
    ci_high = np.quantile(boot_means, 1.0 - alpha, axis=0)
    return point, ci_low, ci_high


def paired_cluster_bootstrap_per_position(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster bootstrap CI for the difference (a - b), paired by (prompt, run).

    Both arrays must share the same (n_prompts, n_runs, n_positions) shape;
    pairing is by index so the caller is responsible for matching seeds.
    """
    _validate_3d(metric_a, "metric_a")
    _validate_3d(metric_b, "metric_b")
    if metric_a.shape != metric_b.shape:
        raise ValueError(
            f"shape mismatch: metric_a {metric_a.shape} vs metric_b {metric_b.shape}"
        )
    diff = metric_a - metric_b
    return cluster_bootstrap_per_position(
        diff,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
        confidence=confidence,
    )
