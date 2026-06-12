"""Tests for the pure helpers in scripts/grid_certificate_experiments.py.

The heavy filter math is already covered by tests/test_grid_mixture.py; here we
test the experiment script's stream generator, continuum-MLE helper, and the
Jeffreys coverage posterior, so the pre-registered analyses rest on verified
parts.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location(
    "grid_certificate_experiments",
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "grid_certificate_experiments.py",
)
assert _SPEC is not None and _SPEC.loader is not None
gce = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gce)


class TestSyntheticStream:
    def test_deterministic_per_seed_and_shapes(self) -> None:
        a_logits, a_tokens = gce.make_synthetic_stream(
            7, n_steps=50, vocab=25, beta_true=1.0, logit_scale=1.5
        )
        b_logits, b_tokens = gce.make_synthetic_stream(
            7, n_steps=50, vocab=25, beta_true=1.0, logit_scale=1.5
        )
        c_logits, _ = gce.make_synthetic_stream(
            8, n_steps=50, vocab=25, beta_true=1.0, logit_scale=1.5
        )
        assert a_logits.shape == (50, 25) and a_tokens.shape == (50,)
        assert a_logits.dtype == np.float64 and a_tokens.dtype == np.int64
        np.testing.assert_array_equal(a_logits, b_logits)
        np.testing.assert_array_equal(a_tokens, b_tokens)
        assert not np.array_equal(a_logits, c_logits)
        assert a_tokens.min() >= 0 and a_tokens.max() < 25

    def test_tokens_follow_target_temperature(self) -> None:
        """At large β the sampled token should usually be the argmax logit."""
        logits, tokens = gce.make_synthetic_stream(
            0, n_steps=400, vocab=25, beta_true=8.0, logit_scale=1.5
        )
        frac_argmax = float((tokens == logits.argmax(axis=-1)).mean())
        assert frac_argmax > 0.9


class TestContinuumMle:
    def test_recovers_beta_on_well_specified_stream(self) -> None:
        logits, tokens = gce.make_synthetic_stream(
            1, n_steps=4000, vocab=25, beta_true=1.5, logit_scale=1.5
        )
        log_mle = gce.continuum_log_mle(logits, tokens)
        assert abs(log_mle - math.log(1.5)) < 0.1

    def test_is_a_local_minimum(self) -> None:
        logits, tokens = gce.make_synthetic_stream(
            2, n_steps=500, vocab=25, beta_true=1.0, logit_scale=1.5
        )
        log_mle = gce.continuum_log_mle(logits, tokens)
        f = lambda lb: gce.cumulative_loss_at(math.exp(lb), logits, tokens)
        assert f(log_mle) <= f(log_mle - 0.1) + 1e-9
        assert f(log_mle) <= f(log_mle + 0.1) + 1e-9


class TestGridRun:
    def test_map_near_truth_well_specified(self) -> None:
        logits, tokens = gce.make_synthetic_stream(
            3, n_steps=1000, vocab=25, beta_true=1.0, logit_scale=1.5
        )
        state = gce.run_stream_through_grid(
            torch.tensor(logits), torch.tensor(tokens)
        )
        assert abs(math.log(state.beta_map()[0].item())) < 0.3


class TestJeffreys:
    def test_posterior_probability_monotone_in_successes(self) -> None:
        p_low = gce.jeffreys_prob_at_least(180, 200, 0.95)
        p_high = gce.jeffreys_prob_at_least(199, 200, 0.95)
        assert 0.0 <= p_low < p_high <= 1.0
