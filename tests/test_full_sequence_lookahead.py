"""Tests for the batched lookahead glue used by the GPU runner.

Two load-bearing pieces, both tested model-free via synthetic callables/tensors:

1. ``candidate_leaf_log_F`` — depth-1 exact log F(prefix, v) for each candidate v
   via a single batched forward over [prefix+v for v]. Must equal the per-prefix
   reference ``tempered_log_partition`` of the model's next-token logits.

2. ``candidate_filter_latent`` — the vectorized candidate-conditioned filter
   latent theta(v). This is the efficiency claim (one length-K tensor op, no
   python loop over the vocab). It MUST equal ``laplace_update`` applied
   per-candidate, otherwise the twist is evaluated at the wrong latent.
"""
from __future__ import annotations

import numpy as np
import torch

from decoding_decoding.counter_decode import init_laplace, laplace_update
from decoding_decoding.full_sequence.lookahead import (
    candidate_filter_latent,
    candidate_leaf_log_F,
)
from decoding_decoding.full_sequence.partition import tempered_log_partition


def test_candidate_leaf_log_F_matches_reference():
    vocab = 12
    rng = np.random.default_rng(0)
    # Synthetic "model": last-position logits depend on the last token id only,
    # so we can predict the leaf distribution for prefix+v deterministically.
    table = rng.standard_normal((vocab, vocab)).astype(np.float32)

    def forward_last_logits(batch_ids: torch.Tensor) -> torch.Tensor:
        # batch_ids: (B, L). Return logits at last position: (B, vocab).
        last_tok = batch_ids[:, -1]
        return torch.from_numpy(table[last_tok.numpy()])

    prefix = torch.tensor([3, 7, 1], dtype=torch.long)
    candidates = torch.tensor([0, 5, 9, 2], dtype=torch.long)
    T = 0.7

    got = candidate_leaf_log_F(forward_last_logits, prefix, candidates, T=T)

    # Reference: for each candidate v, leaf logits are table[v] (last token = v),
    # depth-1 log F = tempered_log_partition(log_softmax(table[v]), T).
    expected = np.empty(len(candidates), dtype=np.float64)
    for i, v in enumerate(candidates.tolist()):
        logits_v = torch.from_numpy(table[v])
        log_p = torch.log_softmax(logits_v, dim=-1).numpy()
        expected[i] = tempered_log_partition(log_p, T)

    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_candidate_filter_latent_matches_laplace_update_per_candidate():
    vocab = 30
    rng = np.random.default_rng(1)
    logits = torch.from_numpy(rng.standard_normal(vocab).astype(np.float32))
    candidates = torch.tensor([0, 4, 11, 29, 7], dtype=torch.long)

    state = init_laplace(1, sigma_0=0.5, dtype=torch.float32)

    # Vectorized: theta(v) for all candidates at once.
    got = candidate_filter_latent(state, logits, candidates)

    # Reference: run the real laplace_update once per candidate, read eta_hat.
    expected = np.empty(len(candidates), dtype=np.float64)
    for i, v in enumerate(candidates.tolist()):
        new_state, _score, _fisher = laplace_update(
            state, logits.unsqueeze(0), torch.tensor([v], dtype=torch.long)
        )
        expected[i] = float(new_state.eta_hat.item())

    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_candidate_filter_latent_respects_evidence_and_decay_knobs():
    vocab = 20
    rng = np.random.default_rng(2)
    logits = torch.from_numpy(rng.standard_normal(vocab).astype(np.float32))
    candidates = torch.tensor([1, 2, 3], dtype=torch.long)
    state = init_laplace(1, sigma_0=0.3, dtype=torch.float32)

    got = candidate_filter_latent(
        state, logits, candidates, evidence_weight=0.5, memory_decay=0.95
    )
    expected = np.empty(len(candidates), dtype=np.float64)
    for i, v in enumerate(candidates.tolist()):
        new_state, _s, _f = laplace_update(
            state,
            logits.unsqueeze(0),
            torch.tensor([v], dtype=torch.long),
            evidence_weight=0.5,
            memory_decay=0.95,
        )
        expected[i] = float(new_state.eta_hat.item())
    np.testing.assert_allclose(got, expected, atol=1e-5)
