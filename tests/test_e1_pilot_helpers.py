"""Unit tests for the pure helpers in scripts/e1_synthetic_pilot.py.

The pilot only calibrates a pre-registered threshold, but its misfit measure must
be correct or the threshold is garbage. These tests pin the three numerical
helpers; the end-to-end self-validation (control misfit ≈ 0, greedy β̂ → grid
edge, ban-argmax β̂ < 0) lives in the pilot's own output.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "e1_synthetic_pilot", _REPO / "scripts" / "e1_synthetic_pilot.py"
)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    p = pilot._softmax(logits)
    assert np.allclose(p.sum(axis=-1), 1.0)
    # Monotone: larger logit -> larger prob.
    assert p[0, 2] > p[0, 1] > p[0, 0]


def test_entropy_matches_uniform_and_degenerate() -> None:
    V = 8
    uniform = np.full((1, V), 1.0 / V)
    assert pilot._entropy(uniform)[0] == pytest.approx(math.log(V))
    degenerate = np.zeros((1, V))
    degenerate[0, 0] = 1.0
    assert pilot._entropy(degenerate)[0] == pytest.approx(0.0)


def test_topk_sampling_distribution_keeps_exactly_k() -> None:
    logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])  # (1, 6), already sorted
    q = pilot._sampling_distribution(logits, "topk3")
    assert (q[0] > 0).sum() == 3
    # The three surviving tokens are the top-3 logits, renormalized.
    assert np.allclose(q[0, 3:], 0.0)
    assert q[0].sum() == pytest.approx(1.0)
    # Probabilities of survivors are the renormalized softmax of the top-3.
    top3 = pilot._softmax(logits[:, :3])[0]
    assert np.allclose(q[0, :3], top3)


def test_blacklist_argmax_zeros_the_top_logit() -> None:
    logits = np.array([[1.0, 9.0, 2.0, 3.0]])  # argmax at index 1
    q = pilot._sampling_distribution(logits, "blacklist_argmax")
    assert q[0, 1] == 0.0
    assert q[0].sum() == pytest.approx(1.0)
    # Mass redistributes over the remaining tokens (full softmax on them).
    rest = pilot._softmax(logits[:, [0, 2, 3]])[0]
    assert np.allclose(q[0, [0, 2, 3]], rest)


def test_control_distribution_is_plain_softmax() -> None:
    logits = np.array([[1.0, 2.0, 3.0]])
    q = pilot._sampling_distribution(logits, "control")
    assert np.allclose(q, pilot._softmax(logits))


def test_blacklist_fixed_zeros_the_popular_token() -> None:
    logits = np.array([[5.0, 1.0, 2.0, 3.0]])  # POPULAR_TOKEN is index 0
    q = pilot._sampling_distribution(logits, "blacklist_fixed")
    assert q[0, pilot.POPULAR_TOKEN] == 0.0
    assert q[0].sum() == pytest.approx(1.0)


def test_arm_logits_biases_only_blacklist_fixed() -> None:
    logits = np.zeros((3, 5))
    # Non-blacklist_fixed arms see the unbiased logits (identity).
    for arm in ("control", "topk5", "blacklist_argmax"):
        assert np.array_equal(pilot._arm_logits(logits, arm), logits)
    # blacklist_fixed adds POPULAR_BIAS to POPULAR_TOKEN, leaves the rest.
    biased = pilot._arm_logits(logits, "blacklist_fixed")
    assert biased[0, pilot.POPULAR_TOKEN] == pytest.approx(pilot.POPULAR_BIAS)
    others = [v for v in range(5) if v != pilot.POPULAR_TOKEN]
    assert np.array_equal(biased[:, others], logits[:, others])


def test_grid_misfit_control_near_zero_small_stream() -> None:
    """A well-specified short stream has misfit ≈ 0 (family contains the truth)."""
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 1.5, size=(400, 25))
    q = pilot._sampling_distribution(logits, "control")
    tokens = pilot._sample_tokens(q, np.random.default_rng(1))
    out = pilot._grid_misfit(logits, tokens, q)
    # β̂ near 1 (log β̂ near 0) and misfit small for a well-specified stream.
    assert abs(out["log_beta_map"]) < 0.3
    assert abs(out["misfit"]) < 0.1
