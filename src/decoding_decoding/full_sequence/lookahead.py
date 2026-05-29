"""Batched lookahead glue for the go/no-go runner.

Two helpers that bridge the pure-math core (``partition.py``) and the streaming
Laplace filter (``counter_decode.py``) to a real model's forward passes.

``candidate_leaf_log_F`` computes the **exact** depth-1 future partition function
log F(prefix, v) for each candidate v with a single batched forward over the K
sequences ``[prefix + v]``. depth-1 is exact for the one-step term and dominates
F at mild temperature (the tempered tail decays); the runner can extend to
depth-2 by recursion if a horizon check is wanted.

``candidate_filter_latent`` computes the candidate-conditioned filter latent
theta(v) = eta after hypothetically emitting v, for ALL candidates in one
length-K tensor op. This is exact (it reproduces ``laplace_update`` per
candidate) and cheap: the Fisher/precision update is candidate-independent, and
the only candidate-dependent term is the centred logit ell_v - E_q. This is the
"no python loop over the vocab" efficiency claim from the review.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from decoding_decoding.counter_decode import LaplaceState
from decoding_decoding.full_sequence.partition import tempered_log_partition

ForwardLastLogitsFn = Callable[[torch.Tensor], torch.Tensor]


def candidate_leaf_log_F(
    forward_last_logits: ForwardLastLogitsFn,
    prefix_ids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    T: float,
    batch_size: int = 64,
) -> np.ndarray:
    """Exact depth-1 log F(prefix, v) for each candidate v.

    Args:
        forward_last_logits: maps (B, L) token ids -> (B, V) logits at the LAST
            position. (One model forward pass, no KV-cache surgery required.)
        prefix_ids: (L,) committed prefix token ids.
        candidates: (K,) candidate next-token ids.
        T: temperature (> 0).
        batch_size: max sequences per forward call.

    Returns:
        (K,) array of log F(prefix, v) in nats.
    """
    if prefix_ids.dim() != 1:
        raise ValueError(f"prefix_ids must be 1-D, got {tuple(prefix_ids.shape)}")
    if candidates.dim() != 1:
        raise ValueError(f"candidates must be 1-D, got {tuple(candidates.shape)}")
    K = candidates.shape[0]
    L = prefix_ids.shape[0]
    device = prefix_ids.device
    out = np.empty(K, dtype=np.float64)

    for start in range(0, K, batch_size):
        chunk = candidates[start:start + batch_size]
        b = chunk.shape[0]
        # Build [prefix + v] for each candidate v in the chunk: (b, L+1).
        seqs = torch.empty((b, L + 1), dtype=torch.long, device=device)
        seqs[:, :L] = prefix_ids.unsqueeze(0).expand(b, L)
        seqs[:, L] = chunk
        last_logits = forward_last_logits(seqs)  # (b, V)
        log_p = torch.log_softmax(last_logits.float(), dim=-1)  # (b, V)
        log_p_np = log_p.detach().cpu().numpy()
        for j in range(b):
            out[start + j] = tempered_log_partition(log_p_np[j], T)
    return out


def candidate_filter_latent(
    state: LaplaceState,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    *,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
    prior_eta: float = 0.0,
    prior_J: float = 0.0,
) -> np.ndarray:
    """theta(v) = filter latent (eta = log beta) after hypothetically emitting v.

    Vectorized over candidates. Reproduces ``laplace_update`` exactly: the
    precision update J_new = decay(J) + alpha * Var_q and E_q, Var_q are
    candidate-independent; the only candidate-dependent term is the score
    ell_v - E_q. Single-trajectory state (batch dim 1) is assumed (the runner
    advances one prefix at a time).

    Args:
        state: current LaplaceState (batch size 1).
        logits: (V,) model logits at the current position.
        candidates: (K,) candidate next-token ids.
        evidence_weight, memory_decay, prior_eta, prior_J: filter knobs,
            matching ``laplace_update`` semantics.

    Returns:
        (K,) array of theta(v) = eta_hat after emitting candidate v.
    """
    if logits.dim() != 1:
        raise ValueError(f"logits must be 1-D (V,), got {tuple(logits.shape)}")
    ell = logits.float()
    log_q = torch.log_softmax(ell, dim=-1)
    q = torch.exp(log_q)
    E_q = torch.sum(q * ell)
    centered = ell - E_q
    Var_q = torch.sum(q * centered * centered)

    eta = float(state.eta_hat.reshape(-1)[0].item())
    J = float(state.J.reshape(-1)[0].item())
    eta_decayed = memory_decay * eta + (1.0 - memory_decay) * prior_eta
    J_decayed = memory_decay * J + (1.0 - memory_decay) * prior_J
    J_new = J_decayed + evidence_weight * float(Var_q.item())

    score_cand = centered[candidates].detach().cpu().numpy().astype(np.float64)
    theta = eta_decayed + evidence_weight * score_cand / J_new
    return theta
