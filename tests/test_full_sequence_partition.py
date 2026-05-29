"""Unit tests for the future-partition-function (F) computation.

These test the *correctness-critical* core of the full-sequence go/no-go test:
the depth-h lookahead estimate of the future partition function

    F(x_{1:t}) = sum_{x_{t+1:N}} prod_{s>t} P(x_s | x_{<s})^{1/T}

against an independent brute-force enumeration oracle. If this math is wrong,
every downstream go/no-go number is wrong, so it is pinned with exact-value tests
on tiny synthetic models where the full sum is enumerable.

Convention (matches full-sequence/files/01_decoding_target.md, verified in
full-sequence/review/fs_review_math.md):
  - log F is in nats.
  - depth-h lookahead truncates the continuation at h steps, with leaf value
    log F = 0 (i.e. F = 1) at the truncation horizon.
  - depth h = N - len(prefix) reproduces the exact fixed-N F.
  - depth-1 lookahead equals the tempered one-step log-partition
    log sum_v P(v | prefix)^{1/T}.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from decoding_decoding.full_sequence.partition import (
    brute_force_log_F,
    recursive_log_F,
    tempered_log_partition,
)


def _make_synthetic_model(vocab: int, seed: int = 0):
    """A deterministic, prefix-dependent toy LM over `vocab` tokens.

    Returns next_log_probs_fn(prefix: tuple[int,...]) -> np.ndarray of shape
    (vocab,) holding normalized log-probabilities. The distribution genuinely
    depends on the whole prefix (via a hash), so F is not trivially constant.
    """
    rng_master = np.random.default_rng(seed)
    base = rng_master.standard_normal(vocab)

    def next_log_probs_fn(prefix: tuple[int, ...]) -> np.ndarray:
        # Mix a per-prefix deterministic perturbation into the base logits.
        h = (hash((seed, prefix)) % 100_000) / 100_000.0
        local_rng = np.random.default_rng(int(h * 1_000_000) + 1)
        logits = base + 0.9 * local_rng.standard_normal(vocab)
        logits = logits - logits.max()
        logZ = math.log(np.exp(logits).sum())
        return logits - logZ  # normalized log-probs

    return next_log_probs_fn


def test_tempered_log_partition_matches_manual():
    # log sum_v exp(log_p_v / T)
    log_probs = np.log(np.array([0.5, 0.3, 0.2]))
    T = 0.7
    expected = math.log(np.sum(np.array([0.5, 0.3, 0.2]) ** (1.0 / T)))
    got = tempered_log_partition(log_probs, T)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_tempered_log_partition_T1_is_log_sum_probs_is_zero():
    # At T=1, sum_v P(v) = 1 (locally normalized) -> log partition = 0.
    log_probs = np.log(np.array([0.4, 0.35, 0.25]))
    got = tempered_log_partition(log_probs, 1.0)
    assert got == pytest.approx(0.0, abs=1e-12)


def test_depth1_recursive_equals_tempered_one_step_partition():
    model = _make_synthetic_model(vocab=4, seed=1)
    T = 0.6
    prefix = (2, 0)
    expected = tempered_log_partition(model(prefix), T)
    got = recursive_log_F(model, prefix, depth=1, T=T, vocab=4)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_recursive_full_depth_equals_brute_force_small():
    vocab, N = 3, 4
    model = _make_synthetic_model(vocab=vocab, seed=7)
    T = 0.8
    prefix = (1,)
    depth = N - len(prefix)
    brute = brute_force_log_F(model, prefix, N=N, T=T, vocab=vocab)
    rec = recursive_log_F(model, prefix, depth=depth, T=T, vocab=vocab)
    assert rec == pytest.approx(brute, rel=1e-10, abs=1e-10)


def test_recursive_full_depth_equals_brute_force_T1():
    # At T=1 the fixed-N F is exactly 1 (the future conditionals sum to 1 at
    # every step), so log F = 0 for any prefix.
    vocab, N = 3, 4
    model = _make_synthetic_model(vocab=vocab, seed=3)
    prefix = (0, 2)
    brute = brute_force_log_F(model, prefix, N=N, T=1.0, vocab=vocab)
    assert brute == pytest.approx(0.0, abs=1e-9)
    rec = recursive_log_F(model, prefix, depth=N - len(prefix), T=1.0, vocab=vocab)
    assert rec == pytest.approx(0.0, abs=1e-9)


def test_brute_force_matches_explicit_enumeration():
    # Cross-check brute_force_log_F against a hand-rolled enumeration to ensure
    # the oracle itself is correct (not just self-consistent with the recursion).
    vocab, N = 2, 3
    model = _make_synthetic_model(vocab=vocab, seed=11)
    T = 0.5
    prefix = (1,)
    # Enumerate continuations of length N - len(prefix) = 2 explicitly.
    terms = []
    for cont in itertools.product(range(vocab), repeat=N - len(prefix)):
        logw = 0.0
        cur = prefix
        for tok in cont:
            logw += model(cur)[tok] / T
            cur = cur + (tok,)
        terms.append(logw)
    expected = float(np.logaddexp.reduce(terms))
    got = brute_force_log_F(model, prefix, N=N, T=T, vocab=vocab)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_candidate_conditioning_F_varies_with_candidate():
    # The whole method rests on F(prefix, v) varying across candidates v.
    # On a prefix-dependent model it must not be constant.
    vocab, N = 3, 4
    model = _make_synthetic_model(vocab=vocab, seed=5)
    T = 0.7
    prefix = (2,)
    vals = [
        recursive_log_F(model, prefix + (v,), depth=N - len(prefix) - 1, T=T, vocab=vocab)
        for v in range(vocab)
    ]
    assert np.std(vals) > 1e-6
