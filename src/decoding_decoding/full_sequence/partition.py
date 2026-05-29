"""Future partition function F via depth-h lookahead.

Implements the sequence-level-temperature future partition function from
``full-sequence/files/01_decoding_target.md`` (math verified in
``full-sequence/review/fs_review_math.md``):

    F(x_{1:t}) = sum_{x_{t+1:N}} prod_{s>t} P(x_s | x_{<s})^{1/T},   F(x_{1:N}) = 1,

with backward recursion

    F(x_{1:t}) = sum_{x_{t+1}} P(x_{t+1} | x_{<=t})^{1/T} F(x_{1:t+1}).

In the log domain this is the soft (log-sum-exp) Bellman backup

    log F(x_{1:t}) = logsumexp_v [ (1/T) log P(v | x_{1:t}) + log F(x_{1:t}, v) ].

These reference implementations are model-agnostic: they take a callable
``next_log_probs_fn(prefix) -> np.ndarray`` returning normalized log-probs over
the vocabulary for the given prefix. They are deliberately simple (per-prefix
calls, no batching) so they can serve as the test oracle; the runner uses a
batched leaf computation that must agree with ``tempered_log_partition`` /
``recursive_log_F`` on tiny models.

NOTE (repo rule): this is NOT multi-pass scoring. Each call to
``next_log_probs_fn`` is one forward pass over a distinct prefix; the recursion
explores the *continuation tree*, it does not re-score the same tokens with
growing prefixes for the same position.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np

NextLogProbsFn = Callable[[tuple[int, ...]], np.ndarray]


def tempered_log_partition(log_probs: np.ndarray, T: float) -> float:
    """log sum_v P(v)^{1/T} = logsumexp_v[ (1/T) log P(v) ].

    This is the depth-1 lookahead value: the exact one-step tempered partition.

    Args:
        log_probs: (V,) normalized log-probabilities.
        T: temperature (> 0).

    Returns:
        Scalar log-partition in nats.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    a = np.asarray(log_probs, dtype=np.float64) / T
    m = float(a.max())
    return m + float(np.log(np.exp(a - m).sum()))


def recursive_log_F(
    next_log_probs_fn: NextLogProbsFn,
    prefix: tuple[int, ...],
    *,
    depth: int,
    T: float,
    vocab: int,
) -> float:
    """Depth-``depth`` lookahead estimate of log F(prefix).

    Truncates the continuation tree at ``depth`` steps with leaf value
    log F = 0 (F = 1). ``depth = N - len(prefix)`` reproduces the exact fixed-N
    log F; ``depth = 1`` reproduces ``tempered_log_partition``.

    Args:
        next_log_probs_fn: prefix -> (vocab,) normalized log-probs.
        prefix: committed token ids.
        depth: lookahead horizon in tokens (>= 0).
        T: temperature (> 0).
        vocab: vocabulary size.

    Returns:
        log F estimate in nats.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if depth == 0:
        return 0.0
    log_probs = np.asarray(next_log_probs_fn(prefix), dtype=np.float64)
    if log_probs.shape != (vocab,):
        raise ValueError(
            f"next_log_probs_fn returned shape {log_probs.shape}, expected ({vocab},)"
        )
    if depth == 1:
        return tempered_log_partition(log_probs, T)
    terms = np.empty(vocab, dtype=np.float64)
    for v in range(vocab):
        child = recursive_log_F(
            next_log_probs_fn, prefix + (v,), depth=depth - 1, T=T, vocab=vocab
        )
        terms[v] = log_probs[v] / T + child
    m = float(terms.max())
    return m + float(np.log(np.exp(terms - m).sum()))


def brute_force_log_F(
    next_log_probs_fn: NextLogProbsFn,
    prefix: tuple[int, ...],
    *,
    N: int,
    T: float,
    vocab: int,
) -> float:
    """Exact fixed-N log F(prefix) by explicit enumeration of all continuations.

    Independent oracle for the recursion: enumerates V^{N-len(prefix)} full
    continuations and log-sum-exps their tempered log-weights. Only tractable for
    tiny (vocab, N); used in tests.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    horizon = N - len(prefix)
    if horizon < 0:
        raise ValueError(f"prefix longer ({len(prefix)}) than N ({N})")
    if horizon == 0:
        return 0.0
    terms: list[float] = []
    for cont in itertools.product(range(vocab), repeat=horizon):
        logw = 0.0
        cur = prefix
        for tok in cont:
            logw += float(next_log_probs_fn(cur)[tok]) / T
            cur = cur + (tok,)
        terms.append(logw)
    arr = np.asarray(terms, dtype=np.float64)
    m = float(arr.max())
    return m + float(np.log(np.exp(arr - m).sum()))
