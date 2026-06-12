"""E1 threshold-calibration pilot (synthetic, CPU).

Purpose: pick the pre-registered misfit threshold δ for E1 from SYNTHETIC
truncated streams, never from the real model data (per REFRAME_PLAN.md §3 E1
requirement 3 and the prereg discipline). This script produces no paper number
on its own; its only output is the δ written by hand into
``paper/certificate_misspec_prereg.md`` BEFORE any real E1 run.

Goodness-of-fit quantity (the E1 "fit, not just width" measure):

    misfit = L_n(β̂)/n  −  (1/n) Σ_t H(q_t)

where
    L_n(β̂)/n = min_g cum_loss[g] / n   — best grid expert's per-token log loss
                                          (the grid MLE / KL projection of the
                                          observed stream onto the temperature
                                          family, scored on RAW full-vocab logits)
    H(q_t)    = entropy of the realized next-token distribution actually sampled
                from (full softmax for control; truncated-renormalized for top-k /
                blacklist). Computed from the SAME raw logits + the known
                truncation rule — so it is equally computable on real streams.

Because L_n(β̂)/n ≈ (1/n) Σ_t [H(q_t) + KL(q_t ‖ p_{β̂})] in expectation, the
misfit estimates the mean KL projection residual: ≈ 0 when the temperature family
contains the truth (control), strictly > 0 under truncation (no β reproduces a
truncated categorical with the full-vocab softmax family). This subtracts the
stream's own entropy, so it is NOT confounded by the truncated stream being
sharper (lower-entropy) than control — the failure mode of a raw "loss vs
control" comparison.

Arms: control (full softmax), top-k (k=5, k=1=greedy), blacklist-argmax (ban the
per-step most-probable token — the synthetic analog of banning a high-frequency
token). Sharpness sweep σ ∈ {1.5, 3.0, 5.0}: probes the concern that for PEAKY
model logits top-k removes only ~zero-mass tail tokens, shrinking the misfit.

Run: ``uv run scripts/e1_synthetic_pilot.py``  (writes JSON to data/grid_certificate/e1_pilot.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoding_decoding.counter_decode import (  # noqa: E402
    init_grid_mixture,
    grid_mixture_level_set,
    grid_mixture_update_batched,
    make_beta_grid,
)

# Match the existing grid-certificate prereg grid exactly.
GRID_LOG_LO, GRID_LOG_HI, GRID_POINTS = -3.0, 3.0, 121
VOCAB = 25
N_STEPS = 1000
N_SEEDS = 50
SIGMAS = (1.5, 3.0, 5.0)
TOPK_VALUES = (5, 1)
# blacklist_fixed: a systematically-frequent token (logit +POPULAR_BIAS every step,
# making it the modal choice a sizeable fraction of the time) that is then banned —
# the real-blacklist mechanism (ban a high-frequency token like " the"), and the
# representative arm for calibrating δ. ban-argmax is an UPPER bound on disruption;
# this is the realistic case the GPU experiment's blacklist arm will actually face.
POPULAR_TOKEN = 0
POPULAR_BIAS = 2.0
DATA_DIR = REPO_ROOT / "data" / "grid_certificate"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Row-wise entropy in nats; 0·log0 := 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    return terms.sum(axis=-1)


def _sampling_distribution(logits: np.ndarray, arm: str) -> np.ndarray:
    """Realized next-token distribution q_t per step for the given arm.

    arm ∈ {"control", "topk{K}", "blacklist_argmax", "blacklist_fixed"}. For
    "blacklist_fixed" the caller must pass logits ALREADY biased on POPULAR_TOKEN.
    Returns (T, V); each row sums to 1.
    """
    p = _softmax(logits)  # (T, V)
    if arm == "control":
        return p
    if arm.startswith("topk"):
        k = int(arm[4:])
        # Keep the top-k logits per row, zero the rest, renormalize.
        T, V = logits.shape
        kth = np.partition(logits, V - k, axis=-1)[:, V - k]  # kth-largest per row
        keep = logits >= kth[:, None]
        q = np.where(keep, p, 0.0)
        return q / q.sum(axis=-1, keepdims=True)
    if arm == "blacklist_argmax":
        T, V = logits.shape
        q = p.copy()
        q[np.arange(T), logits.argmax(axis=-1)] = 0.0
        return q / q.sum(axis=-1, keepdims=True)
    if arm == "blacklist_fixed":
        # Ban the fixed, systematically-frequent token (POPULAR_TOKEN).
        q = p.copy()
        q[:, POPULAR_TOKEN] = 0.0
        return q / q.sum(axis=-1, keepdims=True)
    raise ValueError(f"unknown arm {arm!r}")


def _arm_logits(logits: np.ndarray, arm: str) -> np.ndarray:
    """Raw logits the certificate scores on. blacklist_fixed biases POPULAR_TOKEN
    high (so banning it is disruptive); all other arms see the unbiased logits."""
    if arm == "blacklist_fixed":
        biased = logits.copy()
        biased[:, POPULAR_TOKEN] += POPULAR_BIAS
        return biased
    return logits


def _sample_tokens(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling, one draw per row of q (T, V) -> (T,)."""
    u = rng.random(q.shape[0])
    return (np.cumsum(q, axis=-1) < u[:, None]).sum(axis=-1).clip(0, q.shape[1] - 1)


def _grid_misfit(logits: np.ndarray, tokens: np.ndarray, q: np.ndarray) -> dict:
    """Run the grid mixture; return best-expert per-token loss, mean entropy, misfit, β̂, width."""
    lg = torch.tensor(logits, dtype=torch.float64).unsqueeze(1)   # (T, 1, V)
    tk = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)       # (T, 1)
    beta_grid = make_beta_grid(
        log_lo=GRID_LOG_LO, log_hi=GRID_LOG_HI, n_points=GRID_POINTS, dtype=torch.float64
    )
    state = init_grid_mixture(batch_size=1, beta_grid=beta_grid)
    for t in range(lg.shape[0]):
        state = grid_mixture_update_batched(state, lg[t], tk[t])
    n = lg.shape[0]
    best_loss_per_tok = float(state.cum_loss.min().item()) / n
    mean_entropy = float(_entropy(q).mean())
    lo, hi = grid_mixture_level_set(state)
    return {
        "best_loss_per_tok": best_loss_per_tok,
        "mean_entropy": mean_entropy,
        "misfit": best_loss_per_tok - mean_entropy,
        "log_beta_map": float(torch.log(state.beta_map()[0]).item()),
        "width": float((hi - lo)[0].item()),
        "realized_regret": float(state.realized_regret()[0].item()),
    }


def run_pilot() -> dict:
    arms = (
        ["control"]
        + [f"topk{k}" for k in TOPK_VALUES]
        + ["blacklist_argmax", "blacklist_fixed"]
    )
    results: dict = {"config": {
        "grid": [GRID_LOG_LO, GRID_LOG_HI, GRID_POINTS], "vocab": VOCAB,
        "n_steps": N_STEPS, "n_seeds": N_SEEDS, "sigmas": list(SIGMAS),
        "arms": arms, "beta_true": 1.0,
    }, "by_sigma": {}}

    for sigma in SIGMAS:
        per_arm = {a: [] for a in arms}
        for seed in range(N_SEEDS):
            # Same raw logits across arms (matched); independent token draws per arm.
            rng = np.random.default_rng(seed)
            logits = rng.normal(0.0, sigma, size=(N_STEPS, VOCAB))
            for arm_idx, arm in enumerate(arms):
                # Deterministic per-(seed, arm) token-sampling seed (no Python hash()).
                arm_rng = np.random.default_rng((seed + 1) * 1000 + arm_idx)
                arm_logits = _arm_logits(logits, arm)  # biased only for blacklist_fixed
                q = _sampling_distribution(arm_logits, arm)
                tokens = _sample_tokens(q, arm_rng)
                per_arm[arm].append(_grid_misfit(arm_logits, tokens, q))
        summary = {}
        for arm, recs in per_arm.items():
            misfits = np.array([r["misfit"] for r in recs])
            widths = np.array([r["width"] for r in recs])
            lbmaps = np.array([r["log_beta_map"] for r in recs])
            regrets = np.array([r["realized_regret"] for r in recs])
            summary[arm] = {
                "misfit_mean": float(misfits.mean()),
                "misfit_std": float(misfits.std(ddof=1)),
                "misfit_min": float(misfits.min()),
                "misfit_p05": float(np.percentile(misfits, 5)),
                "misfit_median": float(np.median(misfits)),
                "width_median": float(np.median(widths)),
                "log_beta_map_median": float(np.median(lbmaps)),
                "realized_regret_max": float(regrets.max()),
            }
        results["by_sigma"][f"{sigma:.1f}"] = summary

    # δ recommendation keyed off blacklist_fixed (the REPRESENTATIVE arm: ban a
    # systematically-frequent token, matching the real GPU blacklist). ban-argmax
    # over-estimates disruption; top-k under-estimates on peaky logits. δ should sit
    # comfortably BELOW the blacklist_fixed misfit floor (so a real effect of that
    # size clears it with power to spare) and well ABOVE control noise. The prereg
    # author picks and locks the final δ; this only reports the bracketing numbers.
    blf_mins = [results["by_sigma"][f"{s:.1f}"]["blacklist_fixed"]["misfit_min"] for s in SIGMAS]
    blf_medians = [results["by_sigma"][f"{s:.1f}"]["blacklist_fixed"]["misfit_median"] for s in SIGMAS]
    control_abs_max = max(
        abs(results["by_sigma"][f"{s:.1f}"]["control"]["misfit_mean"]) for s in SIGMAS
    )
    control_std_max = max(
        results["by_sigma"][f"{s:.1f}"]["control"]["misfit_std"] for s in SIGMAS
    )
    worst_blf_min = float(min(blf_mins))
    results["delta_recommendation"] = {
        "blacklist_fixed_misfit_min_across_sigma": blf_mins,
        "blacklist_fixed_misfit_median_across_sigma": blf_medians,
        "worst_case_blacklist_fixed_misfit_min": worst_blf_min,
        "control_misfit_abs_max": float(control_abs_max),
        "control_misfit_std_max": float(control_std_max),
        "suggested_delta_60pct_of_floor": float(round(0.6 * worst_blf_min, 3)),
        "note": "δ ≈ 0.6 × worst-case blacklist_fixed misfit floor; must exceed control |misfit| by ≫ its std",
    }
    return results


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = run_pilot()
    out = DATA_DIR / "e1_pilot.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    # Human-readable summary.
    for sigma in SIGMAS:
        s = results["by_sigma"][f"{sigma:.1f}"]
        print(f"\nσ={sigma}:")
        for arm, st in s.items():
            print(
                f"  {arm:18s} misfit mean={st['misfit_mean']:+.4f} "
                f"min={st['misfit_min']:+.4f} median={st['misfit_median']:+.4f} | "
                f"width med={st['width_median']:.3f} logβ̂ med={st['log_beta_map_median']:+.3f}"
            )
    dr = results["delta_recommendation"]
    print(
        f"\nδ recommendation (keyed off blacklist_fixed, the realistic arm):\n"
        f"  blacklist_fixed misfit floor (min over seeds) per σ = {dr['blacklist_fixed_misfit_min_across_sigma']}\n"
        f"  worst-case floor = {dr['worst_case_blacklist_fixed_misfit_min']:.4f}\n"
        f"  control |misfit| max = {dr['control_misfit_abs_max']:.4f} (std max {dr['control_misfit_std_max']:.4f})\n"
        f"  suggested δ ≈ 0.6×floor = {dr['suggested_delta_60pct_of_floor']:.3f}"
    )


if __name__ == "__main__":
    main()
