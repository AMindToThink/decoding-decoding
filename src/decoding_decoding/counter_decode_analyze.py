"""Analysis for the counter-decoding F0/F1/F2/F3 plots.

Loads the per-trajectory npz files written by counter_decode_generate.py and
computes condition-level statistics with cluster-bootstrap CIs by prompt.

Public entry points:

  * ``load_counter_condition`` — stack per-trajectory data for one
    (β_target, arm) condition into (n_prompts, n_runs, T) arrays for each
    per-step diagnostic.

  * ``f0_concentration`` — produce the F0 headline data: corrected vs
    uncorrected concentration on P_sample and P_φ over t.

  * ``f3_beta_eff`` — produce F3 closed-loop stability data: β_eff = β_dec
    × β̂_pre per step, and the absolute log-error against β_target.

  * ``f1_svd_recovery`` — produce step-1 SVD data: at each saved sparse
    position, decompose the centered trajectory-averaged decoder-logit
    matrix M_t = [ℓ̄_t(k)]_k and return (R(t), σ_1, recovered β̄_t(k), ...).

  * ``f2_calibration_gap`` — produce step-2 calibration data: log β̂ - log β̄.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from decoding_decoding.bootstrap import cluster_bootstrap_per_position
from decoding_decoding.data_layout import MANIFEST_FILENAME, decode_params


PER_STEP_KEYS: tuple[str, ...] = (
    "sampled_token_ids",
    "beta_hat_pre",
    "beta_dec",
    "J_pre",
    "score",
    "fisher",
    "max_p_phi",
    "entropy_phi",
    "max_p_sample",
    "entropy_sample",
)


@dataclass(frozen=True)
class CounterConditionData:
    """Per-(β_target, arm) stacked trajectory data."""

    beta_target: float
    corrected: bool
    sigma_0: float
    # Each is (n_prompts, n_runs, T).
    sampled_token_ids: np.ndarray
    beta_hat_pre: np.ndarray
    beta_dec: np.ndarray
    J_pre: np.ndarray
    score: np.ndarray
    fisher: np.ndarray
    max_p_phi: np.ndarray
    entropy_phi: np.ndarray
    max_p_sample: np.ndarray
    entropy_sample: np.ndarray
    # Trace ids in (prompt, run) order (n_prompts × n_runs flat list).
    trace_ids: list[str]


def _stack_per_step(
    out_dir: Path,
    sub: pl.DataFrame,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Group sub by prompt_id, stack each per-step array into (P, R, T)."""
    grouped = sub.group_by("prompt_id", maintain_order=True).agg(
        [pl.col("trace_id"), pl.col("run_idx")]
    )
    runs_counts = {len(rs) for rs in grouped["trace_id"].to_list()}
    if len(runs_counts) != 1:
        raise ValueError(f"ragged runs per prompt: {runs_counts}")

    per_key_blocks: dict[str, list[np.ndarray]] = {k: [] for k in PER_STEP_KEYS}
    flat_trace_ids: list[str] = []

    for trace_ids, run_idxs in zip(
        grouped["trace_id"].to_list(),
        grouped["run_idx"].to_list(),
        strict=True,
    ):
        order = np.argsort(np.asarray(run_idxs))
        ordered_ids = [trace_ids[i] for i in order]
        per_key_run: dict[str, list[np.ndarray]] = {k: [] for k in PER_STEP_KEYS}
        for tid in ordered_ids:
            npz = np.load(Path(out_dir) / "traces" / f"{tid}.npz")
            for k in PER_STEP_KEYS:
                per_key_run[k].append(np.asarray(npz[k]))
            flat_trace_ids.append(tid)
        for k in PER_STEP_KEYS:
            per_key_blocks[k].append(np.stack(per_key_run[k], axis=0))  # (R, T)

    stacked = {k: np.stack(per_key_blocks[k], axis=0) for k in PER_STEP_KEYS}  # (P, R, T)
    return stacked, flat_trace_ids


def load_counter_condition(
    out_dir: Path,
    *,
    beta_target: float,
    corrected: bool,
) -> CounterConditionData:
    """Stack all trajectories for one (β_target, arm) condition.

    Filters the manifest by family=='counter_decode' and the requested
    (β_target, corrected) combination, sorts by (prompt_id, run_idx), and
    returns per-step arrays of shape (n_prompts, n_runs, T) for each of
    PER_STEP_KEYS.
    """
    out_dir = Path(out_dir)
    manifest = pl.read_parquet(out_dir / MANIFEST_FILENAME)
    sub = manifest.filter(pl.col("family") == "counter_decode").sort(
        ["prompt_id", "run_idx"]
    )
    # Filter by params.
    keep_idx = []
    sigma_0_seen = None
    for i, row in enumerate(sub.iter_rows(named=True)):
        params = decode_params(row["params_json"])
        if (
            abs(params["beta_target"] - beta_target) < 1e-6
            and bool(params["corrected"]) == bool(corrected)
        ):
            keep_idx.append(i)
            if sigma_0_seen is None:
                sigma_0_seen = float(params["sigma_0"])
    if not keep_idx:
        raise ValueError(
            f"no traces matching β_target={beta_target} corrected={corrected}"
        )
    sub2 = sub[keep_idx]
    stacked, trace_ids = _stack_per_step(out_dir, sub2)
    return CounterConditionData(
        beta_target=float(beta_target),
        corrected=bool(corrected),
        sigma_0=float(sigma_0_seen) if sigma_0_seen is not None else float("nan"),
        sampled_token_ids=stacked["sampled_token_ids"],
        beta_hat_pre=stacked["beta_hat_pre"],
        beta_dec=stacked["beta_dec"],
        J_pre=stacked["J_pre"],
        score=stacked["score"],
        fisher=stacked["fisher"],
        max_p_phi=stacked["max_p_phi"],
        entropy_phi=stacked["entropy_phi"],
        max_p_sample=stacked["max_p_sample"],
        entropy_sample=stacked["entropy_sample"],
        trace_ids=trace_ids,
    )


# ---------------------------------------------------------------------------
# F0 — concentration drift on P_sample and P_φ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectorySummary:
    """Mean ± CI of one (n_prompts, n_runs, T) metric across trajectories."""

    point: np.ndarray  # (T,)
    ci_low: np.ndarray  # (T,)
    ci_high: np.ndarray  # (T,)


@dataclass(frozen=True)
class F0Curves:
    """All four F0 curves for one (β_target, arm) condition."""

    beta_target: float
    corrected: bool
    max_p_phi: TrajectorySummary
    entropy_phi: TrajectorySummary
    max_p_sample: TrajectorySummary
    entropy_sample: TrajectorySummary


def _summarize(metric: np.ndarray, *, n_resamples: int, seed: int) -> TrajectorySummary:
    point, lo, hi = cluster_bootstrap_per_position(
        metric, n_resamples=n_resamples, rng_seed=seed
    )
    return TrajectorySummary(point=point, ci_low=lo, ci_high=hi)


def f0_concentration(
    cond: CounterConditionData,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> F0Curves:
    """Compute the four F0 curves with cluster-bootstrap CIs by prompt."""
    return F0Curves(
        beta_target=cond.beta_target,
        corrected=cond.corrected,
        max_p_phi=_summarize(cond.max_p_phi, n_resamples=n_resamples, seed=seed),
        entropy_phi=_summarize(cond.entropy_phi, n_resamples=n_resamples, seed=seed + 1),
        max_p_sample=_summarize(cond.max_p_sample, n_resamples=n_resamples, seed=seed + 2),
        entropy_sample=_summarize(cond.entropy_sample, n_resamples=n_resamples, seed=seed + 3),
    )


# ---------------------------------------------------------------------------
# F3 — closed-loop stability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class F3Curves:
    """β_eff trajectories and absolute log-error for one (β_target, arm)."""

    beta_target: float
    corrected: bool
    # (n_prompts, n_runs, T) per-trajectory β_eff = β_dec · β̂_pre
    beta_eff: np.ndarray
    # (T,) median |log β_eff - log β_target| across all trajectories
    median_abs_log_err: np.ndarray
    # (T,) IQR of log β_eff across trajectories (q75 - q25)
    iqr_log_beta_eff: np.ndarray
    # (T,) median log β_eff
    median_log_beta_eff: np.ndarray


def f3_beta_eff(cond: CounterConditionData) -> F3Curves:
    """β_eff = β_dec · β̂_pre.

    Under matched-prior Bayes-optimality, β̂ = β_model, so this is exactly
    the effective inverse-temperature that the natural reference logits ℓ*
    experience after counter-decoding. Corrected arm: β_eff = β_target by
    construction. Uncorrected arm: β_eff = β_target · β̂_pre, which drifts.

    Args:
        cond: stacked condition data.

    Returns:
        F3Curves with per-trajectory and aggregated (median, IQR) measures.
    """
    bt = cond.beta_target
    beta_eff = cond.beta_dec * cond.beta_hat_pre   # (P, R, T)
    log_eff = np.log(beta_eff)
    # Flatten across (prompt, run) to compute medians per t.
    flat = log_eff.reshape(-1, log_eff.shape[-1])  # (P*R, T)
    median_log = np.median(flat, axis=0)
    q25 = np.quantile(flat, 0.25, axis=0)
    q75 = np.quantile(flat, 0.75, axis=0)
    abs_err = np.abs(flat - np.log(bt))
    median_abs_err = np.median(abs_err, axis=0)
    return F3Curves(
        beta_target=cond.beta_target,
        corrected=cond.corrected,
        beta_eff=beta_eff,
        median_abs_log_err=median_abs_err,
        iqr_log_beta_eff=q75 - q25,
        median_log_beta_eff=median_log,
    )


# ---------------------------------------------------------------------------
# Step 1 — SVD recovery (full-vocab sparse logits, uncorrected only)
# ---------------------------------------------------------------------------


def _load_sparse_logits_for_condition(
    out_dir: Path, trace_ids: Iterable[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Load (positions, logits) for a list of trace ids.

    All traces in `trace_ids` must share the same `positions` array (which
    they do by construction since it's a single config-level setting).

    Returns:
        positions: (P,) int32
        logits: (N, P, V) float32 — stacked across trajectories in input order.
    """
    out_dir = Path(out_dir)
    positions: np.ndarray | None = None
    arrays: list[np.ndarray] = []
    for tid in trace_ids:
        npz = np.load(out_dir / "sparse_logits" / f"{tid}.npz")
        pos = np.asarray(npz["positions"])
        if positions is None:
            positions = pos.astype(np.int32)
        elif not np.array_equal(positions, pos):
            raise ValueError(
                f"sparse position mismatch: {pos.tolist()} vs {positions.tolist()}"
            )
        arrays.append(np.asarray(npz["logits"]).astype(np.float32))
    return positions if positions is not None else np.empty(0, dtype=np.int32), np.stack(arrays, axis=0)


def _center_logits(logits: np.ndarray) -> np.ndarray:
    """Subtract per-row mean: ℓ̃ = ℓ − (1/V)·Σ_j ℓ_j · 1.

    Softmax is invariant to this additive shift, so removing the per-row mean
    isolates the shape direction. Operates along the last axis.
    """
    return logits - logits.mean(axis=-1, keepdims=True)


@dataclass(frozen=True)
class SvdAtPosition:
    """Step-1 SVD diagnostics at one sparse position t."""

    position: int
    # Off-axis residual fraction R(t) = Σ_{i≥2} σ_i² / Σ_i σ_i².
    R: float
    # Singular values σ_1,...,σ_K (length K = number of decoder values).
    singular_values: np.ndarray
    # Recovered β̄_t(k), one per decoder value, sign-aligned with the
    # principal left singular vector.
    beta_recovered: np.ndarray  # (K,)
    # The decoder values (β_k) themselves, in the same order as columns.
    beta_dec_values: np.ndarray  # (K,)
    # The principal left singular vector (a unit-norm direction in R^V),
    # interpreted as the recovered ℓ̂*_t up to sign and scale.
    u1: np.ndarray  # (V,)


def f1_svd_recovery_at_position(
    out_dir: Path,
    *,
    position: int,
    beta_dec_values: list[float],
    n_runs_per_prompt: int | None = None,
) -> SvdAtPosition:
    """At one sparse position, build the K-column matrix M_t and decompose.

    Steps:
      1. For each decoder value β_k in `beta_dec_values`, find the uncorrected
         traces under that decoder, load their sparse logits at `position`,
         and average across all (prompt, run) pairs to get ℓ̄_t(k) ∈ R^V.
      2. Subtract the per-row mean to remove the constant degree of freedom.
      3. Stack into M_t ∈ R^{V × K} and compute economy SVD.
      4. R(t) = Σ_{i≥2} σ_i² / Σ_i σ_i².
      5. Sign-align u_1 with ℓ̄_t(0) and recover β̄_t(k) = σ_1 · v_{1k}
         (with sign possibly flipped to keep β̄'s positive).

    NOTE: this uses the *trajectory-averaged* logits (averaged over prompt and
    run) to test the multiplicativity of the imprint. This matches step 1 of
    the plan ("At each large-enough $t$, average across trajectories").
    """
    out_dir = Path(out_dir)
    manifest = pl.read_parquet(out_dir / MANIFEST_FILENAME)
    sub_unco = manifest.filter(
        (pl.col("family") == "counter_decode")
        & (pl.col("condition_label").str.contains("arm=unco"))
    ).sort(["prompt_id", "run_idx"])

    avg_logits_by_k: list[np.ndarray] = []
    for bk in beta_dec_values:
        rows = []
        for r in sub_unco.iter_rows(named=True):
            params = decode_params(r["params_json"])
            if abs(params["beta_target"] - bk) < 1e-6 and not params["corrected"]:
                rows.append(r["trace_id"])
        if not rows:
            raise ValueError(
                f"no uncorrected traces for β_dec={bk} (need them for SVD step 1)"
            )
        if n_runs_per_prompt is not None:
            # Sanity: total trace count divisible by runs per prompt
            if len(rows) % n_runs_per_prompt:
                raise ValueError(
                    f"β_dec={bk}: {len(rows)} traces not divisible by "
                    f"{n_runs_per_prompt} runs/prompt"
                )
        positions, logits_n_p_V = _load_sparse_logits_for_condition(out_dir, rows)
        # Find this position's index.
        idx = int(np.where(positions == position)[0][0])
        avg = logits_n_p_V[:, idx, :].mean(axis=0)  # (V,) average across all traj
        avg_logits_by_k.append(avg)

    # M_t ∈ R^{V × K}, centered.
    M = np.stack([_center_logits(a) for a in avg_logits_by_k], axis=1).astype(np.float64)
    # Economy SVD on the (V, K) matrix.
    U, S, Vt = np.linalg.svd(M, full_matrices=False)  # U: (V,K) S: (K,) Vt: (K,K)
    R = float((S[1:] ** 2).sum() / max(1e-30, (S ** 2).sum()))

    u1 = U[:, 0]
    s1 = float(S[0])
    v1 = Vt[0]  # (K,)
    # Sign convention: pick u1 to align with the column of M with the largest
    # β_dec value (so β̄ comes out positive there). Equivalently flip if v1
    # for the largest β has the wrong sign.
    bk_arr = np.asarray(beta_dec_values, dtype=np.float64)
    largest_k = int(np.argmax(bk_arr))
    if v1[largest_k] < 0:
        u1 = -u1
        v1 = -v1
    beta_recovered = s1 * v1

    return SvdAtPosition(
        position=int(position),
        R=R,
        singular_values=S.astype(np.float64),
        beta_recovered=beta_recovered,
        beta_dec_values=bk_arr,
        u1=u1,
    )


def f1_svd_over_positions(
    out_dir: Path,
    *,
    beta_dec_values: list[float],
    positions: list[int] | None = None,
) -> list[SvdAtPosition]:
    """Run f1_svd_recovery_at_position for every sparse position present."""
    if positions is None:
        # Auto-detect from a single sparse_logits file.
        sparse_dir = Path(out_dir) / "sparse_logits"
        any_npz = next(sparse_dir.glob("*.npz"), None)
        if any_npz is None:
            raise FileNotFoundError(f"no sparse_logits/*.npz in {sparse_dir}")
        positions = [int(p) for p in np.load(any_npz)["positions"].tolist()]
    return [
        f1_svd_recovery_at_position(
            out_dir, position=p, beta_dec_values=beta_dec_values
        )
        for p in positions
    ]


def f1_svd_per_prompt_at_position(
    out_dir: Path,
    *,
    position: int,
    beta_dec_values: list[float],
) -> dict:
    """Per-prompt SVD: separates "true imprint off-axis" from prompt mixing.

    For each prompt p:
      1. For each β_k, average the runs at this prompt (no cross-prompt mixing).
      2. Build M_t^(p) ∈ R^{V × K}, center each row, SVD.
      3. Record R^(p)(t), σ_1^(p), and σ_1^(p) · v_{1,k}^(p) (one β̄ per k per p).

    Returns a dict with arrays of shape (n_prompts,) for R, and (n_prompts, K)
    for beta_recovered (per-prompt). Per-prompt scale normalization is applied
    so each prompt's β̄(k) has geometric mean 1 across k.
    """
    out_dir = Path(out_dir)
    manifest = pl.read_parquet(out_dir / MANIFEST_FILENAME)
    sub_unco = manifest.filter(
        (pl.col("family") == "counter_decode")
        & (pl.col("condition_label").str.contains("arm=unco"))
    ).sort(["prompt_id", "run_idx"])
    K = len(beta_dec_values)

    # Group: prompt_id → list of (β_target, trace_id) entries.
    by_prompt: dict[int, dict[float, list[str]]] = {}
    for row in sub_unco.iter_rows(named=True):
        params = decode_params(row["params_json"])
        if params["corrected"]:
            continue
        bt = float(params["beta_target"])
        if not any(abs(bt - bk) < 1e-6 for bk in beta_dec_values):
            continue
        by_prompt.setdefault(int(row["prompt_id"]), {}).setdefault(bt, []).append(
            row["trace_id"]
        )

    R_per_prompt: list[float] = []
    beta_recovered_per_prompt: list[np.ndarray] = []  # each (K,)
    sigma1_per_prompt: list[float] = []

    for pid in sorted(by_prompt.keys()):
        avg_logits_by_k: list[np.ndarray] = []
        for bk in beta_dec_values:
            tids = by_prompt[pid].get(bk, [])
            if not tids:
                raise ValueError(
                    f"prompt {pid} has no traces for β_dec={bk}"
                )
            positions, lps_n_p_V = _load_sparse_logits_for_condition(out_dir, tids)
            idx = int(np.where(positions == position)[0][0])
            avg = lps_n_p_V[:, idx, :].mean(axis=0)  # (V,)
            avg_logits_by_k.append(avg)
        M = np.stack([_center_logits(a) for a in avg_logits_by_k], axis=1).astype(
            np.float64
        )
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R_p = float((S[1:] ** 2).sum() / max(1e-30, (S ** 2).sum()))
        s1 = float(S[0])
        v1 = Vt[0]
        # Sign convention: align with the largest β_k.
        bk_arr = np.asarray(beta_dec_values, dtype=np.float64)
        if v1[int(np.argmax(bk_arr))] < 0:
            v1 = -v1
        beta_rec = s1 * v1  # (K,)
        # Per-prompt scale normalization: geometric mean across k → 1.
        gm = float(np.exp(np.mean(np.log(np.maximum(np.abs(beta_rec), 1e-9)))))
        beta_rec_norm = beta_rec / gm

        R_per_prompt.append(R_p)
        sigma1_per_prompt.append(s1)
        beta_recovered_per_prompt.append(beta_rec_norm)

    R_arr = np.asarray(R_per_prompt, dtype=np.float64)
    beta_arr = np.stack(beta_recovered_per_prompt, axis=0)  # (n_prompts, K)
    s1_arr = np.asarray(sigma1_per_prompt, dtype=np.float64)
    return {
        "position": int(position),
        "n_prompts": int(beta_arr.shape[0]),
        "R_per_prompt": R_arr,
        "R_median": float(np.median(R_arr)),
        "R_iqr": (float(np.quantile(R_arr, 0.25)), float(np.quantile(R_arr, 0.75))),
        "sigma1_per_prompt": s1_arr,
        # Per-prompt β̄_norm; geometric mean across prompts gives the
        # "consensus" direction.
        "beta_recovered_per_prompt": beta_arr,
        "beta_dec_values": np.asarray(beta_dec_values, dtype=np.float64),
        "log_beta_median_across_prompts": np.median(np.log(np.maximum(np.abs(beta_arr), 1e-9)), axis=0),
        "log_beta_iqr_across_prompts": (
            np.quantile(np.log(np.maximum(np.abs(beta_arr), 1e-9)), 0.25, axis=0),
            np.quantile(np.log(np.maximum(np.abs(beta_arr), 1e-9)), 0.75, axis=0),
        ),
    }


def normalize_svd_scale(
    svds: list[SvdAtPosition],
    *,
    anchor_at_position: int = 0,
) -> list[SvdAtPosition]:
    """SVD recovers β̄ only up to scale. Anchor by setting β̄ at t=0 to 1.

    At t=0 the model has no observations, so β_model^(0) ≈ 1 (under the
    matched-prior assumption with E_π_0[β] = 1). The SVD's β̄(k) at t=0
    is therefore the unknown scale factor c. We rescale β̄(k, t) ←
    β̄(k, t) / c for all t so that the result is on the same scale as β.

    Returns a new list of SvdAtPosition with `beta_recovered` rescaled.
    The principal vector ``u1`` is left untouched (the rescaling is into v
    and absorbed into σ).
    """
    anchor = next((s for s in svds if s.position == anchor_at_position), None)
    if anchor is None:
        raise ValueError(
            f"anchor_at_position={anchor_at_position} not found among SVD positions; "
            f"have {[s.position for s in svds]}"
        )
    # At t=0 with all columns equal, β̄(k=*) is approximately constant. Use
    # the geometric mean as the scale.
    c = float(np.exp(np.mean(np.log(np.maximum(anchor.beta_recovered, 1e-9)))))
    out: list[SvdAtPosition] = []
    for s in svds:
        out.append(
            SvdAtPosition(
                position=s.position,
                R=s.R,
                singular_values=s.singular_values,
                beta_recovered=s.beta_recovered / c,
                beta_dec_values=s.beta_dec_values,
                u1=s.u1,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Step 2 — calibration gap log β̂ - log β̄
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationGap:
    """At one sparse position, comparison of streaming β̂ vs SVD-recovered β̄."""

    position: int
    beta_dec_values: np.ndarray  # (K,)
    beta_hat_streaming: np.ndarray  # (K,) — mean β̂_pre across prompts/runs at this t
    beta_recovered_svd: np.ndarray  # (K,) — from f1_svd_recovery_at_position
    log_gap: np.ndarray  # log β̂ - log β̄ per decoder value


def f2_calibration_at_position(
    out_dir: Path,
    *,
    position: int,
    beta_dec_values: list[float],
    svd: SvdAtPosition,
) -> CalibrationGap:
    """Compute the calibration gap Δ(t, k) = log β̂_streaming − log β̄_svd.

    β̂_streaming is the streaming Laplace estimator's β̂_pre at step t,
    averaged across (prompt, run) trajectories at decoder β_k. β̄_svd is
    σ_1·v_{1,k} from the SVD decomposition above.
    """
    if abs(svd.position - position) > 0:
        raise ValueError(f"svd at position {svd.position} != requested {position}")
    beta_hats: list[float] = []
    for bk in beta_dec_values:
        cond = load_counter_condition(out_dir, beta_target=bk, corrected=False)
        bh = cond.beta_hat_pre[..., position]  # (P, R)
        beta_hats.append(float(bh.mean()))
    bh_arr = np.asarray(beta_hats, dtype=np.float64)
    return CalibrationGap(
        position=int(position),
        beta_dec_values=np.asarray(beta_dec_values, dtype=np.float64),
        beta_hat_streaming=bh_arr,
        beta_recovered_svd=svd.beta_recovered.astype(np.float64),
        log_gap=np.log(bh_arr) - np.log(np.maximum(svd.beta_recovered, 1e-9)),
    )
