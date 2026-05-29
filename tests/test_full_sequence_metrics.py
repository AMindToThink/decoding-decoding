"""Unit tests for the go/no-go metrics.

These pin the decision-relevant statistics: KL between candidate distributions,
the fraction of the total correction recovered by an estimator, the affine
(temperature-collinearity) decomposition, and bootstrap CIs.

The temperature-collinearity guard is the load-bearing one: the scalar-linear
Laplace twist is provably equivalent to a position-dependent temperature change,
so a candidate-correction that is *affine in the logit* is recoverable by a mere
global-temperature retune and must NOT be credited to the structured twist. The
affine_residual_fraction isolates the non-temperature part of the correction.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from decoding_decoding.full_sequence.metrics import (
    affine_residual_fraction,
    bootstrap_ci,
    kl_divergence,
    softmax,
    target_from_correction,
)


def test_softmax_normalizes_and_is_shift_invariant():
    logw = np.array([1.0, 2.0, 3.0])
    p = softmax(logw)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    p_shift = softmax(logw + 100.0)
    np.testing.assert_allclose(p, p_shift, atol=1e-12)


def test_kl_zero_for_identical():
    p = softmax(np.array([0.2, 1.1, -0.3]))
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_matches_manual():
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    # Tolerance loosened past kl_divergence's internal eps=1e-12 guard, which
    # perturbs the value by ~3.5e-12; the math is exact otherwise.
    assert kl_divergence(p, q) == pytest.approx(expected, abs=1e-9)


def test_kl_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = softmax(rng.standard_normal(6))
        q = softmax(rng.standard_normal(6))
        assert kl_divergence(p, q) >= -1e-12


def test_target_from_correction_recovers_qg():
    # q_g(v) ∝ exp(g_pt_v + c_v). Building it from g_pt and c must match a direct
    # softmax of the sum.
    g_pt = np.array([0.1, -0.5, 2.0, 0.3])
    c = np.array([0.0, 1.0, -1.0, 0.5])
    got = target_from_correction(g_pt, c)
    expected = softmax(g_pt + c)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_affine_residual_fraction_zero_when_correction_is_affine_in_logit():
    # If c_v = a + b*ell_v exactly, the correction is pure temperature change;
    # the non-affine residual fraction must be ~0.
    rng = np.random.default_rng(1)
    ell = rng.standard_normal(50)
    c = 0.7 + 1.3 * ell  # exactly affine
    frac = affine_residual_fraction(c, ell)
    assert frac == pytest.approx(0.0, abs=1e-12)


def test_affine_residual_fraction_one_when_orthogonal_to_logit():
    # A correction with zero linear relationship to the logit (and de-meaned)
    # has residual fraction ~1: none of it is temperature-collinear.
    n = 200
    ell = np.linspace(-2, 2, n)
    rng = np.random.default_rng(2)
    # Construct c orthogonal to ell: a quadratic component, then remove the
    # linear projection so OLS slope is ~0.
    raw = ell ** 2
    # Remove linear fit on ell so only the non-affine part remains.
    A = np.vstack([np.ones(n), ell]).T
    coef, *_ = np.linalg.lstsq(A, raw, rcond=None)
    c = raw - A @ coef + raw.mean()  # residual-only (plus a constant, harmless)
    frac = affine_residual_fraction(c, ell)
    assert frac == pytest.approx(1.0, abs=1e-6)


def test_affine_residual_fraction_constant_correction_is_zero_variance():
    # A constant c carries no correction at all (cancels in renormalization):
    # Var(c)=0; defined to return 0 (nothing to explain, nothing residual).
    ell = np.array([0.1, 0.5, -0.3, 2.0])
    c = np.full(4, 1.234)
    assert affine_residual_fraction(c, ell) == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_ci_brackets_mean_and_is_deterministic_with_seed():
    rng = np.random.default_rng(3)
    data = rng.normal(loc=5.0, scale=1.0, size=500)
    lo, mid, hi = bootstrap_ci(data, statistic=np.mean, n_boot=1000, seed=42)
    assert lo < mid < hi
    assert mid == pytest.approx(float(np.mean(data)), abs=1e-12)
    assert lo < 5.0 < hi
    # Determinism: same seed -> identical CI.
    lo2, mid2, hi2 = bootstrap_ci(data, statistic=np.mean, n_boot=1000, seed=42)
    assert (lo, mid, hi) == (lo2, mid2, hi2)


def test_bootstrap_ci_lower_bound_positive_for_clearly_positive_data():
    data = np.full(200, 0.5) + np.random.default_rng(4).normal(0, 0.01, 200)
    lo, _mid, _hi = bootstrap_ci(data, statistic=np.mean, n_boot=500, seed=1)
    assert lo > 0.0
