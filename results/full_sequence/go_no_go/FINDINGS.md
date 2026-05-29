# Full-sequence structured twist — Go/No-Go existence test: findings

**Decision: NO-GO for the proposed structured twist.** The sequence-level
temperature correction `F` is real, but it is **token-identity-shaped, not
low-dimensional in the filter's latent**, so a log-linear (or any) twist over the
existing 1-D Laplace latent recovers ≈0 of it beyond a temperature retune.
Confirmed on **gpt2-large** (800 positions) and **Qwen2.5-3B** (600 positions),
both WikiText + WritingPrompts, T ∈ {0.3, 0.5, 0.7, 0.9}.

All numbers below come from `summary.json` in the per-model subdirs. The results
tables are emitted verbatim by `scripts/build_full_sequence_findings_tables.py`,
and per-scalar macros by `scripts/build_full_sequence_macros.py`
(numbers-from-scripts; none hand-typed). Regenerate the tables with
`uv run python scripts/build_full_sequence_findings_tables.py`. Pre-registered
design: `full-sequence/GO_NO_GO_DESIGN.md`.

## Question

Before building the streaming-twist sampler + fits, decide cheaply whether `F`
has enough structure on real text — **beyond what a temperature retune already
captures** — for a low-dimensional twist to recover a statistically significant,
scrutiny-proof fraction of it.

## Method (one paragraph)

Per evaluation position (committed prefix `x_<t`), form the top-K candidate set and
compute the **exact depth-1 future partition**
`c_v = log F(x_<t, v) = logsumexp_w[(1/T) log P(w | x_<t, v)]` with one batched
forward over `[x_<t · v]` (full-vocab leaf sum — no biased subsampling). The
sequence-level target over candidates is `q_g(v) ∝ exp((1/T) log P(v|x_<t) + c_v)`;
the per-token baseline is `q_pt(v) ∝ exp((1/T) log P(v|x_<t))`. We measure how much
of `D_total = E[KL(q_g ‖ q_pt)]` each estimator recovers: an optimal temperature
`q_bestT`; the scalar-linear filter twist `w·θ`; a **cross-validated nonparametric
map of the filter latent θ** (`frac_filter_CV`); and a CV map of the bare centered
logit `s = ℓ − E_q` (`frac_logit_CV`). Decision metric: `frac_X − frac_bestT`
(fraction of `F` recovered **beyond** temperature tuning), bootstrap CIs over
positions.

## The load-bearing subtlety

The Laplace candidate latent `θ(v) = η̂ + α(ℓ_v − E_q)/J` is **affine in the
candidate logit ℓ_v**, so the spec's scalar-*linear* twist
`P_v^{1/T}·exp(w·θ(v)) ∝ exp(β_eff·ℓ_v)` is *exactly* a temperature change and can
never beat `q_bestT`. We therefore credit only the increment **beyond** temperature
and reach beyond it only via a **nonlinear** map of the latent. Confirmed in data:
`frac_filter_linear ≈ 0` (rounds to the bestT line) at every T.

**Label note:** the implemented `q_bestT` fits the optimal temperature **per
position** (an oracle retuning T separately at each position) — a *stronger* baseline
than the single global temperature named in the design doc. So `filter_CV − bestT < 0`
means the latent twist is below even a per-position oracle temperature. The filter
twist also fails the weaker bar: its absolute `frac_filter_CV ≈ 0` is below what a
single global temperature recovers (steelman below). Either way: NO-GO.

## Results

Tables below are pasted verbatim from
`scripts/build_full_sequence_findings_tables.py` (overall block, both corpora
pooled). Negative signs are ASCII `-`.

### gpt2-large (n = 800 positions/T)

| T | D_total (nats) [95% CI] | frac_bestT (per-pos oracle) | frac_filter_CV | frac_logit_CV | (filter_CV − bestT) [95% CI] | affine_resid_frac | GO |
|---|--------------------------|------------------------------|----------------|---------------|------------------------------|-------------------|-----|
| 0.3 | 0.5068 [0.4488, 0.5716] | 0.487 | 0.000 | 0.013 | -0.487 [-0.513, -0.459] | 0.963 | No |
| 0.5 | 0.2219 [0.2002, 0.2447] | 0.335 | 0.000 | 0.013 | -0.335 [-0.362, -0.308] | 0.962 | No |
| 0.7 | 0.0757 [0.0694, 0.0820] | 0.255 | -0.000 | 0.009 | -0.255 [-0.279, -0.232] | 0.961 | No |
| 0.9 | 0.0077 [0.0072, 0.0082] | 0.192 | -0.000 | 0.004 | -0.192 [-0.213, -0.173] | 0.960 | No |

### Qwen2.5-3B (n = 600 positions/T)

| T | D_total (nats) [95% CI] | frac_bestT (per-pos oracle) | frac_filter_CV | frac_logit_CV | (filter_CV − bestT) [95% CI] | affine_resid_frac | GO |
|---|--------------------------|------------------------------|----------------|---------------|------------------------------|-------------------|-----|
| 0.3 | 0.3266 [0.2795, 0.3784] | 0.446 | -0.000 | 0.007 | -0.446 [-0.482, -0.407] | 0.963 | No |
| 0.5 | 0.1420 [0.1243, 0.1606] | 0.294 | -0.000 | 0.008 | -0.294 [-0.329, -0.260] | 0.962 | No |
| 0.7 | 0.0484 [0.0433, 0.0539] | 0.224 | -0.000 | 0.006 | -0.224 [-0.252, -0.197] | 0.960 | No |
| 0.9 | 0.0051 [0.0046, 0.0056] | 0.161 | 0.000 | 0.003 | -0.161 [-0.184, -0.139] | 0.957 | No |

Both models give the **same shape verdict**: `affine_resid_frac ≈ 0.96`,
`frac_filter_CV ≈ 0`, and every `filter_CV − bestT` CI well below 0. Qwen's
absolute `D_total` is modestly smaller than gpt2-large's (e.g. 0.33 vs 0.51 nats at
T=0.3) — a slightly smaller sequence-level correction — but the conclusion is
identical. (`frac_logit_CV` tracks `frac_filter_CV` to within ~0.01 → the filter's
prefix-state / J normalization adds essentially nothing over the bare candidate
logit at depth 1.)

## Why: F is real but not latent-shaped (four independent diagnostics)

1. **D_total CIs exclude 0** at every T, both models → the sequence-level correction
   is real and, at T=0.3–0.5, sizeable (0.14–0.51 nats).
2. **affine_residual_fraction ≈ 0.96** (both models) → ~96% of the within-position
   variance of `c_v` is NOT an affine function of the candidate logit.
3. **Spearman(c_v, logit_v) ≈ 0** and **per-position quadratic-in-logit R² ≈ 0.06**
   → no function of the candidate logit (hence no function of the affine latent θ)
   can recover `F`. Exact per-(model, T) values in `perposition_scrutiny.txt`
   (from `scripts/full_sequence_scrutiny_perposition.py`).
4. **Cross-validated 1-D map recovers `frac_filter_CV ≈ 0.00`** → confirmed
   out-of-sample, not a fitting artifact.

Mechanistically: `F(x_<t, v)` is the tempered future mass after committing token
`v`, which depends on **what token v is** (its continuations), not on v's current
probability. Two candidates with equal probability can have very different `F`. A
scalar latent that is a function of the logit is structurally blind to this — exactly
the expressiveness failure flagged in `full-sequence/review/fs_review_method.md` §3.

## Steelman: prefix-adaptive temperature (also NO-GO for this filter)

A different method the spec hints at is a **prefix-adaptive temperature** β\*_t
(not candidate-conditioning). Table pasted verbatim from
`scripts/build_full_sequence_findings_tables.py`; per-T detail in
`prefix_adaptive_temperature_steelman.txt`
(`scripts/full_sequence_steelman_adaptive_temp.py`). Fractions are of `D_total`:

| Model | T | global-T recovers | per-position-oracle-T recovers | adaptive-T headroom | Spearman(β*_t, filter prefix latent proxy) |
|-------|---|-------------------|-------------------------------|---------------------|---------------------------------------------|
| gpt2-large | 0.5 | +0.089 | +0.335 | +0.246 | -0.007 |
| gpt2-large | 0.7 | +0.028 | +0.256 | +0.228 | -0.007 |
| Qwen2.5-3B | 0.5 | +0.075 | +0.294 | +0.220 | -0.003 |
| Qwen2.5-3B | 0.7 | +0.025 | +0.226 | +0.200 | -0.017 |

So a **per-position oracle temperature has real headroom** (+0.20–0.25 of D beyond a
single global temperature) → "adaptive temperature" is a genuine phenomenon on both
models. **But the filter's prefix latent does not predict β\*_t** (Spearman between
−0.017 and −0.003 — indistinguishable from zero). The headroom exists but is
**not reachable from the existing filter's state.** (Proxy: candidate-averaged θ; a
dedicated `eta_prefix`/`J_prefix` readout would sharpen this but is unlikely to flip a
near-zero correlation.)

## Caveats / scope (honest bounds)

- **Depth-1 only.** `c_v` is the exact one-step tempered partition. A longer horizon
  can only *add* structure to `F`; it cannot make `F` more logit-shaped, so depth-1
  does not bias the NO-GO. (Depth-2 beam is supported by the runner but not needed to
  overturn a structural-shape conclusion.)
- **Ceiling, not realized estimator.** `frac_filter_CV` is the best cross-validated
  smooth map of the latent — an upper bound. The NO-GO is therefore conservative: the
  best *possible* 1-D-latent twist already recovers ≈0.
- **Top-K candidate set** (covered tempered mass ≥ 0.90 at T≥0.5, in summary); leaf
  partition is full-vocab.
- **KL-to-Q_g of the realized autoregressive sampler is NOT measured here** — that and
  any compounding over steps (review §2) would be Phase 4, which the NO-GO obviates.

## Decision and recommendation

**NO-GO** for the spec's structured twist as designed (a log-linear / scalar-latent
map of the surprise/logit latent). Two honest paths forward, neither matching the
current spec:

1. **If pursuing sequence-level tempering at all:** the twist must condition on
   **token identity / continuation structure** — a learned neural twist (Twisted SMC)
   or short-horizon exact lookahead — not a 1-D latent. Heavier than the repo's
   "wrap the existing filter" framing.
2. **If keeping the cheap-filter framing:** pivot to **prefix-adaptive temperature**,
   but first find a prefix feature that actually predicts β\*_t (the current Laplace
   latent does not). The oracle headroom (~+0.20–0.25 of D) bounds the best case.

The cheap go/no-go did its job: it ruled out a large build (sampler + two fits + SMC
eval) that the data show would have recovered ≈0 beyond a temperature change.
