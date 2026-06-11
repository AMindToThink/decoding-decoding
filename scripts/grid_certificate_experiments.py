"""Pre-registered grid-certificate experiments H1-H4.

Protocol: paper/grid_certificate_prereg.md (committed BEFORE this script ran).
Every hypothesis, sample size, and threshold below is pinned there; do not
change them in response to data. Anything labeled "exploratory" carries no
threshold.

Usage:
    uv run scripts/grid_certificate_experiments.py synthetic   # H1 + H3 data
    uv run scripts/grid_certificate_experiments.py model       # H2 + H4 data (GPU)
    uv run scripts/grid_certificate_experiments.py analyze     # tests + results.json
    uv run scripts/grid_certificate_experiments.py all
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Tuple

import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoding_decoding.counter_decode import (  # noqa: E402
    GridMixtureState,
    LaplaceState,
    grid_mixture_level_set,
    grid_mixture_update_batched,
    init_grid_mixture,
    init_laplace,
    laplace_update,
    laplace_update_faithful,
    make_beta_grid,
    sample_under_beta,
)

# --- Pinned by the pre-registration (do not edit after data exist) -----------
GRID_LOG_LO, GRID_LOG_HI, GRID_POINTS = -3.0, 3.0, 121
GRID_SPACING = (GRID_LOG_HI - GRID_LOG_LO) / (GRID_POINTS - 1)  # 0.05 in log β
H1_SEEDS, H1_STEPS, H1_VOCAB, H1_SCALE, H1_BETA = 200, 1000, 25, 1.5, 1.0
H1_NOMINAL_COVERAGE = 0.95
H2_WIDTH_THRESHOLD = 0.2
H3_SEEDS, H3_STEPS, H3_SCALE_NATURAL, H3_SCALE_PEAKED = 200, 300, 1.5, 8.0
H4_BETA_TARGETS = (0.5, 1.0, 2.0)
H4_NOMINAL_AGREEMENT = 0.90
MODEL_TOKENS = 1000
MODEL_SIGMA0 = 0.2
GENERATION_SEED = 0
SNAPSHOT_STEPS = (200, 500, 1000)

DATA_DIR = REPO_ROOT / "data" / "grid_certificate"


# --- Pure helpers (unit-tested in tests/test_grid_certificate_experiments.py) -


def make_synthetic_stream(
    seed: int,
    *,
    n_steps: int,
    vocab: int,
    beta_true: float,
    logit_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Well-specified stream: logits ~ N(0, scale²) i.i.d., x_t ~ softmax(β·ℓ_t).

    Returns (logits (T, V) float64, tokens (T,) int64), deterministic per seed.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, logit_scale, size=(n_steps, vocab))
    scaled = beta_true * logits
    p = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    u = rng.random(n_steps)
    tokens = (np.cumsum(p, axis=-1) < u[:, None]).sum(axis=-1)
    return logits, np.minimum(tokens, vocab - 1).astype(np.int64)


def cumulative_loss_at(beta: float, logits: np.ndarray, tokens: np.ndarray) -> float:
    """L_n(β) = Σ_t [logsumexp(β·ℓ_t) − β·ℓ_t[x_t]]."""
    scaled = beta * logits
    lse = logsumexp(scaled, axis=-1)
    return float((lse - scaled[np.arange(len(tokens)), tokens]).sum())


def continuum_log_mle(
    logits: np.ndarray,
    tokens: np.ndarray,
    *,
    log_lo: float = GRID_LOG_LO,
    log_hi: float = GRID_LOG_HI,
) -> float:
    """log of argmin_β L_n(β) over [e^log_lo, e^log_hi] (bounded Brent)."""
    res = minimize_scalar(
        lambda b: cumulative_loss_at(b, logits, tokens),
        bounds=(math.exp(log_lo), math.exp(log_hi)),
        method="bounded",
    )
    if not res.success:
        raise RuntimeError(f"continuum MLE optimization failed: {res.message}")
    return math.log(res.x)


def run_stream_through_grid(
    logits: torch.Tensor, tokens: torch.Tensor
) -> GridMixtureState:
    """Stream one (T, V) / (T,) trajectory through the grid mixture (float64)."""
    return run_streams_through_grid(logits.unsqueeze(1), tokens.unsqueeze(1))


def run_streams_through_grid(
    logits: torch.Tensor, tokens: torch.Tensor
) -> GridMixtureState:
    """Stream (T, B, V) logits / (T, B) tokens through the grid mixture."""
    dtype = logits.dtype
    beta_grid = make_beta_grid(
        log_lo=GRID_LOG_LO, log_hi=GRID_LOG_HI, n_points=GRID_POINTS,
        device=logits.device, dtype=dtype,
    )
    state = init_grid_mixture(batch_size=logits.shape[1], beta_grid=beta_grid)
    for t in range(logits.shape[0]):
        state = grid_mixture_update_batched(state, logits[t], tokens[t])
    return state


def jeffreys_prob_at_least(successes: int, n: int, p0: float) -> float:
    """P(rate ≥ p0 | data) under the Jeffreys Beta(s+½, f+½) posterior."""
    return float(1.0 - stats.beta.cdf(p0, successes + 0.5, n - successes + 0.5))


# --- Synthetic experiments (H1, H3) ------------------------------------------


def run_synthetic() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # H1: coverage of the continuum MLE.
    streams = [
        make_synthetic_stream(
            s, n_steps=H1_STEPS, vocab=H1_VOCAB, beta_true=H1_BETA, logit_scale=H1_SCALE
        )
        for s in range(H1_SEEDS)
    ]
    logits = torch.tensor(
        np.stack([sl for sl, _ in streams], axis=1), dtype=torch.float64
    )  # (T, B, V)
    tokens = torch.tensor(np.stack([tk for _, tk in streams], axis=1))  # (T, B)
    state = run_streams_through_grid(logits, tokens)
    lo, hi = grid_mixture_level_set(state)
    log_mles = np.array(
        [continuum_log_mle(sl, tk) for sl, tk in streams], dtype=np.float64
    )
    covered = (
        (lo.numpy() - GRID_SPACING <= log_mles) & (log_mles <= hi.numpy() + GRID_SPACING)
    )

    # H3: bar width, natural vs peaked logits.
    widths: dict[str, np.ndarray] = {}
    for label, scale in (("natural", H3_SCALE_NATURAL), ("peaked", H3_SCALE_PEAKED)):
        arm = [
            make_synthetic_stream(
                s, n_steps=H3_STEPS, vocab=H1_VOCAB, beta_true=H1_BETA, logit_scale=scale
            )
            for s in range(H3_SEEDS)
        ]
        arm_logits = torch.tensor(
            np.stack([sl for sl, _ in arm], axis=1), dtype=torch.float64
        )
        arm_tokens = torch.tensor(np.stack([tk for _, tk in arm], axis=1))
        arm_state = run_streams_through_grid(arm_logits, arm_tokens)
        a_lo, a_hi = grid_mixture_level_set(arm_state)
        widths[label] = (a_hi - a_lo).numpy()

    np.savez(
        DATA_DIR / "synthetic.npz",
        h1_lo=lo.numpy(),
        h1_hi=hi.numpy(),
        h1_log_mles=log_mles,
        h1_covered=covered,
        h3_width_natural=widths["natural"],
        h3_width_peaked=widths["peaked"],
    )
    print(
        f"synthetic done: H1 covered {int(covered.sum())}/{H1_SEEDS}, "
        f"H3 median width natural={np.median(widths['natural']):.3f} "
        f"peaked={np.median(widths['peaked']):.3f}"
    )


# --- Model experiments (H2, H4) -----------------------------------------------


def run_model(device: str = "cuda") -> None:
    """Generate fresh Qwen2.5-3B streams; run grid + both Laplace filters online."""
    from decoding_decoding.counter_decode_generate import _load_model_and_tokenizer
    from decoding_decoding.prompts import PROMPTS

    if not torch.cuda.is_available():
        raise RuntimeError("model experiments need a GPU (prereg: Qwen2.5-3B)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model, tok = _load_model_and_tokenizer(device=device)

    out: dict[str, np.ndarray] = {}
    for bt in H4_BETA_TARGETS:
        B = len(PROMPTS)
        enc = tok(list(PROMPTS), return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        beta_vec = torch.full((B,), float(bt), device=device, dtype=torch.float32)
        gen = torch.Generator(device=device).manual_seed(GENERATION_SEED)

        beta_grid = make_beta_grid(
            log_lo=GRID_LOG_LO, log_hi=GRID_LOG_HI, n_points=GRID_POINTS,
            device=device, dtype=torch.float32,
        )
        grid_state = init_grid_mixture(batch_size=B, beta_grid=beta_grid)
        faithful = init_laplace(B, sigma_0=MODEL_SIGMA0, device=device)
        original = init_laplace(B, sigma_0=MODEL_SIGMA0, device=device)

        with torch.no_grad():
            fwd = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True
            )
        logits_t = fwd.logits[:, -1, :].float()
        past = fwd.past_key_values

        snapshots: dict[int, dict[str, np.ndarray]] = {}
        for t in range(MODEL_TOKENS):
            x_t = sample_under_beta(logits_t, beta_vec, generator=gen)
            grid_state = grid_mixture_update_batched(
                grid_state, logits_t, x_t, grid_chunk=16
            )
            faithful, _, _ = laplace_update_faithful(faithful, logits_t, x_t)
            original, _, _ = laplace_update(original, logits_t, x_t)
            if t + 1 in SNAPSHOT_STEPS:
                snapshots[t + 1] = {
                    "cum_loss": grid_state.cum_loss.cpu().numpy(),
                    "mix_loss": grid_state.mix_loss.cpu().numpy(),
                    "eta_faithful": faithful.eta_hat.cpu().numpy(),
                    "eta_original": original.eta_hat.cpu().numpy(),
                }
            if t < MODEL_TOKENS - 1:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(B, 1, device=device, dtype=attention_mask.dtype),
                    ],
                    dim=-1,
                )
                with torch.no_grad():
                    fwd = model(
                        input_ids=x_t.unsqueeze(-1),
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                logits_t = fwd.logits[:, -1, :].float()
                past = fwd.past_key_values

        key = f"bt{bt:.2f}".replace(".", "p")
        for n_tok, snap in snapshots.items():
            for field, arr in snap.items():
                out[f"{key}_n{n_tok}_{field}"] = arr
        print(f"model streams done for beta_target={bt}")

    out["beta_grid"] = beta_grid.cpu().numpy()
    np.savez(DATA_DIR / "model_streams.npz", **out)


# --- Analysis (the pre-registered tests) ---------------------------------------


def _level_set_from_cum_loss(
    cum_loss: np.ndarray, log_grid: np.ndarray, c: float
) -> Tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of {g : L[g] ≤ min L + c} per row, in log β."""
    mask = cum_loss <= cum_loss.min(axis=-1, keepdims=True) + c
    lo = np.where(mask, log_grid[None, :], np.inf).min(axis=-1)
    hi = np.where(mask, log_grid[None, :], -np.inf).max(axis=-1)
    return lo, hi


def analyze() -> None:
    synth = np.load(DATA_DIR / "synthetic.npz")
    streams = np.load(DATA_DIR / "model_streams.npz")
    log_grid = np.log(streams["beta_grid"])
    c = math.log(GRID_POINTS)
    results: dict = {"threshold_c": c}

    # H1 — coverage (exact binomial, one-sided: fail if significantly < 0.95).
    covered = synth["h1_covered"]
    s, n = int(covered.sum()), int(covered.size)
    h1_test = stats.binomtest(s, n, H1_NOMINAL_COVERAGE, alternative="less")
    results["H1"] = {
        "covered": s,
        "n": n,
        "p_value": h1_test.pvalue,
        "fails": h1_test.pvalue < 0.05,
        "jeffreys_P_coverage_ge_0.95": jeffreys_prob_at_least(s, n, H1_NOMINAL_COVERAGE),
    }

    # H2 — non-vacuousness on real β_target=1 streams at n=1000.
    cum = streams["bt1p00_n1000_cum_loss"]
    lo, hi = _level_set_from_cum_loss(cum, log_grid, c)
    widths = hi - lo
    k = int((widths < H2_WIDTH_THRESHOLD).sum())
    h2_test = stats.binomtest(k, widths.size, 0.5, alternative="greater")
    boot = stats.bootstrap(
        (widths,), np.median, n_resamples=10_000, method="percentile",
        random_state=np.random.default_rng(0),
    )
    results["H2"] = {
        "n_streams": int(widths.size),
        "n_below_threshold": k,
        "median_width": float(np.median(widths)),
        "median_ci95": [
            float(boot.confidence_interval.low),
            float(boot.confidence_interval.high),
        ],
        "p_value": h2_test.pvalue,
        "holds": h2_test.pvalue < 0.05,
    }

    # H3 — peaked wider than natural (Mann–Whitney U, one-sided).
    w_nat, w_peak = synth["h3_width_natural"], synth["h3_width_peaked"]
    u_stat, h3_p = stats.mannwhitneyu(w_peak, w_nat, alternative="greater")
    results["H3"] = {
        "median_natural": float(np.median(w_nat)),
        "median_peaked": float(np.median(w_peak)),
        "p_value": float(h3_p),
        "holds": h3_p < 0.05,
        "rank_biserial": float(2.0 * u_stat / (w_nat.size * w_peak.size) - 1.0),
    }

    # H4 — faithful Laplace inside the certified bar (pooled over β_targets).
    agree, agree_orig, per_bt_widths, per_bt_regret = [], [], {}, {}
    for bt in H4_BETA_TARGETS:
        key = f"bt{bt:.2f}".replace(".", "p")
        cum = streams[f"{key}_n1000_cum_loss"]
        lo, hi = _level_set_from_cum_loss(cum, log_grid, c)
        eta_f = streams[f"{key}_n1000_eta_faithful"]
        eta_o = streams[f"{key}_n1000_eta_original"]
        agree.extend(((lo <= eta_f) & (eta_f <= hi)).tolist())
        agree_orig.extend(((lo <= eta_o) & (eta_o <= hi)).tolist())
        per_bt_widths[str(bt)] = float(np.median(hi - lo))
        mix = streams[f"{key}_n1000_mix_loss"]
        per_bt_regret[str(bt)] = float(np.median(mix - cum.min(axis=-1)))
    s4, n4 = int(np.sum(agree)), len(agree)
    h4_test = stats.binomtest(s4, n4, H4_NOMINAL_AGREEMENT, alternative="less")
    results["H4"] = {
        "agree": s4,
        "n": n4,
        "p_value": h4_test.pvalue,
        "fails": h4_test.pvalue < 0.05,
    }

    # Exploratory (no thresholds; labeled).
    expl_width_by_n = {}
    for n_tok in SNAPSHOT_STEPS:
        cum = streams[f"bt1p00_n{n_tok}_cum_loss"]
        lo, hi = _level_set_from_cum_loss(cum, log_grid, c)
        expl_width_by_n[str(n_tok)] = float(np.median(hi - lo))
    results["exploratory"] = {
        "original_laplace_agreement": f"{int(np.sum(agree_orig))}/{len(agree_orig)}",
        "median_width_by_beta_target_n1000": per_bt_widths,
        "median_realized_regret_by_beta_target (bound ln G = %.3f)" % c: per_bt_regret,
        "median_width_by_n_bt1": expl_width_by_n,
    }

    out_path = DATA_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwritten to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["synthetic", "model", "analyze", "all"])
    args = parser.parse_args()
    if args.stage in ("synthetic", "all"):
        run_synthetic()
    if args.stage in ("model", "all"):
        run_model()
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
