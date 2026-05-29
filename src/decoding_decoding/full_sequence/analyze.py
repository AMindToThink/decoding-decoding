"""Analysis + go/no-go decision logic for the full-sequence existence test.

Kept separate from the GPU runner (``scripts/run_full_sequence_go_no_go.py``) so
the decision-critical statistics are importable and unit-testable without a model.

Estimators compared per temperature (all vs the per-token baseline ``q_pt``):
  - ``q_bestT``     : best single global temperature (1 param). The "boring" ceiling.
  - ``q_filt_lin``  : scalar-LINEAR filter twist ``w*theta``. Provably a temperature
                      change (theta is affine in the logit) -> sanity check ~ bestT.
  - ``q_filt_cv``   : cross-validated nonparametric map of the FILTER latent theta.
                      The method's actual ceiling.
  - ``q_logit_cv``  : cross-validated nonparametric map of the bare centered logit
                      ``s = ell - E_q``. Tests whether the filter's prefix-state / J
                      normalization adds anything beyond the candidate logit alone.

The scrutiny-proof quantities are ``frac_X - frac_bestT``: fraction of the
sequence-level correction recovered BEYOND any global temperature retune.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decoding_decoding.full_sequence.metrics import (
    affine_residual_fraction,
    bootstrap_ci,
    cv_binned_predict,
    kl_divergence,
    softmax,
)


@dataclass
class PositionRecord:
    """Per-evaluation-position raw arrays needed for all downstream metrics."""

    corpus: str
    passage_id: int
    position: int
    candidates: list[int]
    ell: list[float]          # base logits at the candidate tokens
    log_p: list[float]        # base log-probs at the candidate tokens
    E_q: float                # full-vocab mean logit E_q[ell] (filter centering)
    covered_mass: dict        # T(str) -> covered tempered mass over the candidate set
    c_by_T: dict              # T(str) -> [log F(prefix, v)] over candidates (depth-h)
    theta: list[float]        # candidate-conditioned filter latent per candidate
    # Prefix state BEFORE candidate-conditioning (for the prefix-adaptive-temperature
    # steelman). Defaulted so older callers/fixtures stay valid.
    eta_prefix: float = 0.0   # filter latent eta_hat = log beta after the prefix
    J_prefix: float = 0.0     # filter precision J after the prefix
    recent_entropy: float = 0.0  # mean model entropy (nats) over the last few prefix tokens
    position_frac: float = 0.0   # t / max_prefix_len (position annealing predictor)


def fit_best_T(g_pt: np.ndarray, ell: np.ndarray, c: np.ndarray) -> np.ndarray:
    """q_bestT ∝ exp(beta_eff * ell): scalar beta_eff minimizing KL(q_g || q_bestT).

    q_g ∝ exp(g_pt + c). Grid search over a wide beta range; the global-temperature
    family is 1-parameter so a fine grid is exact enough for a ceiling.
    """
    q_g = softmax(g_pt + c)
    best, best_kl = None, np.inf
    for beta in np.linspace(0.05, 8.0, 160):
        q = softmax(beta * ell)
        kl = kl_divergence(q_g, q)
        if kl < best_kl:
            best_kl, best = kl, q
    return best


def aggregate(records: list[PositionRecord], T_list: list[float], seed: int) -> dict:
    """Compute the pre-registered metrics for each temperature, with bootstrap CIs."""
    results: dict = {}
    for T in T_list:
        Tk = str(T)
        rows = [r for r in records if Tk in r.c_by_T]
        if not rows:
            continue

        g_pts, ells, cs, thetas, ss = [], [], [], [], []
        covered, struct_frac = [], []
        for r in rows:
            ell = np.array(r.ell)
            c = np.array(r.c_by_T[Tk])
            g_pts.append(np.array(r.log_p) / T)
            ells.append(ell)
            cs.append(c)
            thetas.append(np.array(r.theta))
            ss.append(ell - r.E_q)  # centered logit = filter observation
            covered.append(r.covered_mass[Tk])
            struct_frac.append(affine_residual_fraction(c, ell))

        n = len(rows)

        # Cross-validated nonparametric ceilings (out-of-fold; no in-sample leak).
        c_pred_theta = cv_binned_predict(thetas, cs, n_bins=10, n_folds=5, seed=seed)
        c_pred_s = cv_binned_predict(ss, cs, n_bins=10, n_folds=5, seed=seed)

        # Scalar-linear filter twist coefficient (pooled de-meaned slope of c on theta).
        xs = np.concatenate([t - t.mean() for t in thetas])
        ys = np.concatenate([c - c.mean() for c in cs])
        w_hat = float(np.sum(xs * ys) / np.sum(xs * xs)) if np.var(xs) > 1e-18 else 0.0

        kl_g_pt = np.empty(n)
        kl_g_bestT = np.empty(n)
        kl_g_lin = np.empty(n)
        kl_g_filt = np.empty(n)
        kl_g_logit = np.empty(n)
        for i in range(n):
            g_pt, ell, c = g_pts[i], ells[i], cs[i]
            q_g = softmax(g_pt + c)
            kl_g_pt[i] = kl_divergence(q_g, softmax(g_pt))
            kl_g_bestT[i] = kl_divergence(q_g, fit_best_T(g_pt, ell, c))
            kl_g_lin[i] = kl_divergence(q_g, softmax(g_pt + w_hat * thetas[i]))
            kl_g_filt[i] = kl_divergence(q_g, softmax(g_pt + c_pred_theta[i]))
            kl_g_logit[i] = kl_divergence(q_g, softmax(g_pt + c_pred_s[i]))

        D_total = float(np.mean(kl_g_pt))

        def frac_rom(kl_other: np.ndarray) -> float:
            return 1.0 - float(np.mean(kl_other)) / D_total if D_total > 0 else 0.0

        def boot_frac_increment(kl_other: np.ndarray) -> list:
            """Bootstrap CI of (frac_other - frac_bestT) = mean(kl_bestT-kl_other)/D_total."""
            rng = np.random.default_rng(seed)
            pts = np.empty(2000)
            for b in range(2000):
                bi = rng.integers(0, n, size=n)
                den = float(np.mean(kl_g_pt[bi]))
                num = float(np.mean(kl_g_bestT[bi] - kl_other[bi]))
                pts[b] = num / den if den > 0 else 0.0
            point = (float(np.mean(kl_g_bestT - kl_other)) / D_total) if D_total > 0 else 0.0
            return [float(np.percentile(pts, 2.5)), point, float(np.percentile(pts, 97.5))]

        d_lo, d_pt, d_hi = bootstrap_ci(kl_g_pt, statistic=np.mean, n_boot=2000, seed=seed)

        results[Tk] = {
            "n_positions": int(n),
            "D_total_nats": D_total,
            "D_total_ci": [d_lo, d_pt, d_hi],
            "frac_bestT": frac_rom(kl_g_bestT),
            "frac_filter_linear": frac_rom(kl_g_lin),
            "frac_filter_cv": frac_rom(kl_g_filt),
            "frac_logit_cv": frac_rom(kl_g_logit),
            "frac_filter_cv_minus_bestT": boot_frac_increment(kl_g_filt),
            "frac_logit_cv_minus_bestT": boot_frac_increment(kl_g_logit),
            "frac_filter_linear_minus_bestT": boot_frac_increment(kl_g_lin),
            "affine_residual_fraction_mean": float(np.mean(struct_frac)),
            "w_hat": w_hat,
            "covered_mass_mean": float(np.mean(covered)),
            "kl_g_pt_mean": float(np.mean(kl_g_pt)),
            "kl_g_bestT_mean": float(np.mean(kl_g_bestT)),
            "kl_g_filter_cv_mean": float(np.mean(kl_g_filt)),
            "kl_g_logit_cv_mean": float(np.mean(kl_g_logit)),
        }
    return results


def go_no_go_decision(agg: dict) -> dict:
    """Apply the pre-registered GO/NO-GO rule (full-sequence/GO_NO_GO_DESIGN.md).

    GO at a temperature iff:
      1. the correction is real        : D_total CI lower bound > 1e-3 nats, AND
      2. beyond-temperature structure  : the FILTER-latent CV ceiling recovers
         >= 5% of the correction BEYOND best-T with bootstrap CI lower bound > 0.
    Also records whether the bare-logit CV ceiling clears the same bar: if the
    logit ceiling clears but the filter ceiling does not, the GO is conditional on
    a better latent coordinate (reported, never silently assumed).
    """
    decisions: dict = {}
    any_go = False
    for Tk, m in agg.items():
        d_ci_lo = m["D_total_ci"][0]
        filt_lo, filt_pt, _ = m["frac_filter_cv_minus_bestT"]
        logit_lo, logit_pt, _ = m["frac_logit_cv_minus_bestT"]
        correction_real = d_ci_lo > 1e-3
        filter_go = (filt_lo > 0.0) and (filt_pt >= 0.05)
        logit_go = (logit_lo > 0.0) and (logit_pt >= 0.05)
        go = bool(correction_real and filter_go)
        any_go = any_go or go
        decisions[Tk] = {
            "correction_real": bool(correction_real),
            "filter_go": bool(filter_go),
            "logit_go": bool(logit_go),
            "GO": go,
            "go_conditional_on_better_latent": bool(
                correction_real and logit_go and not filter_go
            ),
        }
    decisions["ANY_GO"] = any_go
    return decisions
