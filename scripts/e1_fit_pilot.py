"""Synthetic pilot for E1 fit-metric design (NOT the real experiment).

Question: under top-k / blacklist truncation, is the certificate's best-expert
per-token loss L_n(beta_hat)/n HIGHER or LOWER than a control (untruncated)
stream? And what does the KL-projection residual min_beta KL(P_decode || P_beta)
look like? This decides which fit metric (and which hypothesis sign) goes in the
pre-registration.

Synthetic Gaussian logits, matching the H1 generator (vocab 25, scale 1.5).
"""

from __future__ import annotations

import numpy as np
import torch

from decoding_decoding.counter_decode import (
    grid_mixture_level_set,
    grid_mixture_update_batched,
    init_grid_mixture,
    make_beta_grid,
)

VOCAB = 25
SCALE = 1.5
N = 1000
GRID_LO, GRID_HI, GRID_POINTS = -3.0, 3.0, 121


def softmax_np(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_logits(rng, n, vocab, scale):
    return rng.normal(0.0, scale, size=(n, vocab))


def decode_dist(logits, *, kind, k=None):
    """Return the per-step decode distribution (n, V) for a decoding rule."""
    z = logits.copy()
    if kind == "control":
        pass
    elif kind == "topk":
        # keep top-k logits, set rest to -inf
        idx = np.argsort(-z, axis=-1)[:, k:]
        for t in range(z.shape[0]):
            z[t, idx[t]] = -np.inf
    elif kind == "blacklist":
        # ban the top-1 token (rank-0) per step
        top1 = np.argmax(logits, axis=-1)
        z[np.arange(z.shape[0]), top1] = -np.inf
    else:
        raise ValueError(kind)
    return softmax_np(z, axis=-1)


def sample_tokens(rng, dist):
    n, V = dist.shape
    out = np.empty(n, dtype=np.int64)
    for t in range(n):
        out[t] = rng.choice(V, p=dist[t])
    return out


def grid_loss_profile(logits, tokens):
    """Run the certificate filter; return (beta_map, width, best_expert_mean_loss)."""
    beta_grid = make_beta_grid(log_lo=GRID_LO, log_hi=GRID_HI, n_points=GRID_POINTS, dtype=torch.float64)
    state = init_grid_mixture(batch_size=1, beta_grid=beta_grid)
    lt = torch.tensor(logits, dtype=torch.float64)
    tk = torch.tensor(tokens, dtype=torch.long)
    for t in range(logits.shape[0]):
        state = grid_mixture_update_batched(state, lt[t:t+1], tk[t:t+1])
    beta_map = float(state.beta_map()[0])
    lo, hi = grid_mixture_level_set(state)
    width = float((hi - lo)[0])
    best_mean_loss = float(state.cum_loss.min(dim=-1).values[0]) / logits.shape[0]
    return beta_map, width, best_mean_loss


def kl_projection_residual(logits, decode_d):
    """min_beta mean_t KL(decode_d[t] || softmax(beta*logits[t])) over the grid."""
    beta_grid = np.exp(np.linspace(GRID_LO, GRID_HI, GRID_POINTS))
    best = np.inf
    P = decode_d  # (n, V)
    logP = np.where(P > 0, np.log(P), 0.0)
    entropy_term = (P * logP).sum(axis=-1)  # (n,)  = -H(P) ... actually sum P logP = -H
    for b in beta_grid:
        logQ = b * logits - np.log(np.exp(b * logits - (b*logits).max(-1, keepdims=True)).sum(-1, keepdims=True)) - (b*logits).max(-1, keepdims=True)
        # KL(P||Q) = sum P (logP - logQ)
        kl = (P * (logP - logQ)).sum(axis=-1)  # (n,)
        m = kl.mean()
        if m < best:
            best = m
    return best


def main():
    rows = []
    for kind, k in [("control", None), ("topk", 5), ("topk", 3), ("blacklist", None)]:
        bmaps, widths, losses, kls = [], [], [], []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            logits = make_logits(rng, N, VOCAB, SCALE)
            d = decode_dist(logits, kind=kind, k=k)
            # sample tokens with an independent stream so seeds are reproducible
            srng = np.random.default_rng(10_000 + seed)
            tokens = sample_tokens(srng, d)
            bmap, width, loss = grid_loss_profile(logits, tokens)
            kl = kl_projection_residual(logits, d)
            bmaps.append(bmap); widths.append(width); losses.append(loss); kls.append(kl)
        label = f"{kind}" + (f"_k{k}" if k else "")
        rows.append((label, np.mean(bmaps), np.mean(widths), np.mean(losses), np.mean(kls)))
        print(f"{label:14s}  beta_map={np.mean(bmaps):6.3f}  log_bmap={np.log(np.mean(bmaps)):+6.3f}  "
              f"width={np.mean(widths):.3f}  bestloss/n={np.mean(losses):.4f}  KLresid={np.mean(kls):.4f}")

    print("\n--- deltas vs control (best-expert loss/n and KL residual) ---")
    ctrl = rows[0]
    for label, bmap, width, loss, kl in rows[1:]:
        print(f"{label:14s}  dloss={loss-ctrl[3]:+.4f}  dKLresid={kl-ctrl[4]:+.4f}  d_log_bmap={np.log(bmap)-np.log(ctrl[1]):+.3f}")


if __name__ == "__main__":
    main()
