"""Go/no-go decision metrics.

All distributions are over the per-position candidate set (top-K tokens), with
the covered tempered mass reported separately by the runner. See
``full-sequence/GO_NO_GO_DESIGN.md`` for the pre-registered decision rule.

Key guard: ``affine_residual_fraction`` isolates the part of the candidate
correction ``c_v = log F(x_<t, v)`` that is NOT affine in the logit ``ell_v``.
Because the scalar-linear Laplace twist is provably a position-dependent
temperature change (the candidate latent is affine in ell_v), only the
non-affine residual is a correction that a global-temperature retune cannot
already capture.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def softmax(log_weights: np.ndarray) -> np.ndarray:
    """Numerically stable softmax of unnormalized log-weights -> prob vector."""
    a = np.asarray(log_weights, dtype=np.float64)
    m = float(a.max())
    e = np.exp(a - m)
    return e / e.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """KL(p || q) in nats over a shared finite support.

    Both inputs are probability vectors over the same candidate set. A small eps
    guards against log(0); inputs are assumed already normalized.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"p, q shape mismatch: {p.shape} vs {q.shape}")
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask] + eps) - np.log(q[mask] + eps))))


def target_from_correction(g_pt: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Sequence-level target over candidates: q_g(v) ∝ exp(g_pt_v + c_v).

    Args:
        g_pt: (K,) per-token tempered log-weights (1/T) log P(v | x_<t).
        c: (K,) candidate correction log F(x_<t, v).

    Returns:
        (K,) normalized probability vector.
    """
    g_pt = np.asarray(g_pt, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if g_pt.shape != c.shape:
        raise ValueError(f"g_pt, c shape mismatch: {g_pt.shape} vs {c.shape}")
    return softmax(g_pt + c)


def affine_residual_fraction(c: np.ndarray, ell: np.ndarray) -> float:
    """Fraction of Var_v(c) NOT explained by an affine fit c ≈ a + b·ell.

    OLS-fit ``c_v = a + b·ell_v + r_v`` over the candidate set; return
    ``Var(r) / Var(c)``. 0 => correction is pure temperature change
    (fully recoverable by retuning T); 1 => correction is orthogonal to the
    logit (genuinely non-temperature structure). If Var(c)=0 (constant
    correction, cancels in renormalization), returns 0.
    """
    c = np.asarray(c, dtype=np.float64)
    ell = np.asarray(ell, dtype=np.float64)
    if c.shape != ell.shape:
        raise ValueError(f"c, ell shape mismatch: {c.shape} vs {ell.shape}")
    var_c = float(np.var(c))
    if var_c <= 1e-18:
        return 0.0
    A = np.vstack([np.ones_like(ell), ell]).T  # (K, 2)
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    resid = c - A @ coef
    return float(np.var(resid) / var_c)


def cv_binned_predict(
    thetas: list[np.ndarray],
    cs: list[np.ndarray],
    *,
    n_bins: int = 8,
    n_folds: int = 5,
    seed: int = 0,
) -> list[np.ndarray]:
    """Cross-validated nonparametric 1-D map latent -> per-position de-meaned c.

    Fits a GLOBAL quantile-binned-mean map ``theta -> (c - mean_position(c))`` on
    training-fold positions and predicts held-out positions, so a position's own
    candidates never inform its own prediction (no in-sample leakage). Returns,
    per position, the out-of-fold predicted correction, re-de-meaned per position
    (the softmax level gauge cancels).

    This is the honest ceiling for "best smooth function of the 1-D latent": if it
    cannot beat a global temperature retune, no log-linear/scalar twist over this
    latent can either. ``thetas[i]`` and ``cs[i]`` are the (K_i,) candidate latent
    and correction arrays for evaluation position i.
    """
    P = len(thetas)
    if P != len(cs):
        raise ValueError(f"thetas/cs length mismatch: {P} vs {len(cs)}")
    ths = [np.asarray(t, dtype=np.float64) for t in thetas]
    ys = [np.asarray(c, dtype=np.float64) - float(np.mean(c)) for c in cs]
    preds: list[np.ndarray] = [np.zeros_like(t) for t in ths]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(P)
    folds = np.array_split(perm, min(n_folds, P)) if P > 0 else []

    for fold in folds:
        test_idx = set(int(i) for i in fold.tolist())
        train_idx = [i for i in range(P) if i not in test_idx]
        if not train_idx:
            continue
        train_th = np.concatenate([ths[i] for i in train_idx])
        train_y = np.concatenate([ys[i] for i in train_idx])
        edges = np.quantile(train_th, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if edges.shape[0] < 2:
            # degenerate (all theta equal): predict the (zero) mean.
            for i in test_idx:
                preds[i] = np.zeros_like(ths[i])
            continue
        inner = edges[1:-1]
        nb = edges.shape[0] - 1
        tb = np.clip(np.digitize(train_th, inner), 0, nb - 1)
        bin_means = np.zeros(nb, dtype=np.float64)
        for b in range(nb):
            m = tb == b
            bin_means[b] = float(train_y[m].mean()) if m.any() else 0.0
        for i in test_idx:
            bi = np.clip(np.digitize(ths[i], inner), 0, nb - 1)
            p = bin_means[bi]
            preds[i] = p - float(p.mean())
    return preds


def bootstrap_ci(
    data: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``statistic`` over rows of ``data``.

    Returns (lower, point, upper) where point = statistic(data) on the full
    sample and lower/upper are the alpha/2 and 1-alpha/2 percentiles of the
    bootstrap distribution. Deterministic given ``seed``.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    point = float(statistic(data))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = statistic(data[idx])
    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return lo, point, hi
