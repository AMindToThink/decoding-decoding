# Pre-registration: grid-mixture certificate experiments (H1–H4)

Committed BEFORE any experiment below is run or analyzed. No hypothesis, test,
sample size, or threshold may change after seeing data; anything not listed
here is exploratory and must be labeled as such in the writeup.

**Estimator under test:** `grid_mixture_update_batched` /
`grid_mixture_level_set` (`src/decoding_decoding/counter_decode.py`,
commit d361e2a): Bayes mixture over fixed temperature experts on raw logits,
uniform prior, `switch_rate=0`. Grid for ALL experiments:
`make_beta_grid(log_lo=-3.0, log_hi=3.0, n_points=121)` (log-β spacing 0.05).
Error bar = level set at the default threshold c = ln G = ln 121 ≈ 4.796.
Filter math in float64 for synthetic streams, float32 for model streams.

**Deviation from the approved plan, declared up front:** the plan said "rerun
the existing sweep logs through the new filter." The saved sweeps
(`data/counter_decode_belief_sweep/`) store only top-200 logprobs, not
full-vocab logits; evaluating softmax(β·ℓ) on a truncated vocabulary biases
the partition function in a β-dependent way, so reusing them would silently
change the estimator. H2/H4 therefore use freshly generated streams: same
model (Qwen/Qwen2.5-3B, float16 weights), same prompt set
(`decoding_decoding.prompts.PROMPTS`, 32 prompts), open-loop (uncorrected)
temperature sampling, generation seed 0.

## H1 — Coverage (synthetic, well-specified)

The level-set bar contains the continuum MLE on well-specified streams.

- **Data:** 200 streams, seeds 0–199. Per stream: n = 1000 steps, vocab 25,
  logits i.i.d. N(0, 1.5²), tokens sampled from softmax(1.0 · ℓ_t) (β* = 1).
- **Statistic:** continuum MLE β̂ via `scipy.optimize.minimize_scalar`
  (bounded, [e⁻³, e³]) on L_n(β); success = log β̂ ∈ [lo − 0.05, hi + 0.05]
  (one grid spacing of slack per side).
- **Test:** exact binomial (`scipy.stats.binomtest`), one-sided,
  H0: coverage ≥ 0.95 vs H1: < 0.95. **Fail H1 if p < 0.05.**
- **Also report (descriptive):** Jeffreys Beta(s+½, f+½) posterior
  P(coverage ≥ 0.95).

## H2 — Non-vacuousness (real streams, β_target = 1)

The bar is tight enough to be interesting on natural model text.

- **Data:** 32 streams (one per prompt), Qwen2.5-3B, β_target = 1.0
  (plain sampling), n = 1000 generated tokens per stream, generation seed 0.
- **Statistic:** per-stream bar width w = hi − lo in log β at the final token.
- **Test:** one-sided sign test (exact binomial on #\{w < 0.2\} vs 0.5),
  H1: median width < 0.2. **H2 holds if p < 0.05.** Prediction from the
  curvature ballpark (Λ_n ≈ 3·n): width ≈ 2·√(2c/Λ_n) ≈ 0.11.
- **Also report (descriptive):** median width + 95% bootstrap CI
  (10,000 resamples, `scipy.stats.bootstrap`, percentile method).

## H3 — Regime degradation (synthetic)

The bar is wider when logits are peaked (temperature unidentifiable regime) —
the expected, honest direction.

- **Data:** 200 + 200 streams (seeds 0–199 per arm), n = 300, vocab 25,
  β* = 1; logit scale 1.5 ("natural") vs 8.0 ("peaked").
- **Test:** Mann–Whitney U (`scipy.stats.mannwhitneyu`), one-sided,
  H1: peaked widths stochastically larger. **H3 holds if p < 0.05.**
  Report rank-biserial effect size.

## H4 — Grid–Laplace agreement (real streams)

The faithful Laplace point estimate lies inside the grid's certified bar —
the condition for informally transferring the certificate to Laplace outputs.

- **Data:** 96 streams = 32 prompts × β_target ∈ {0.5, 1.0, 2.0},
  Qwen2.5-3B, n = 1000 tokens, generation seed 0 (the β_target = 1 arm is the
  same data as H2).
- **Statistic:** `laplace_update_faithful` run online on the same raw logits
  (defaults: σ₀ = 0.2 to match sweep convention; evidence_weight = 1,
  memory_decay = 1); success = final η̂ ∈ [lo, hi].
- **Test:** exact binomial, one-sided, H0: agreement ≥ 0.90 vs H1: < 0.90.
  **Fail H4 if p < 0.05.** Failure consequence (pre-committed): the paper
  scopes the certificate to grid outputs only.
- **Also report (exploratory, labeled):** same agreement rate for the
  production `laplace_update` (the original biased variant) — context for the
  paper's existing headline numbers, no threshold attached.

## Exploratory (no thresholds, must be labeled exploratory)

- Real-stream bar widths by β_target (0.5 / 1.0 / 2.0) — descriptive analogue
  of H3.
- Realized regret vs the ln G bound on real streams.
- Bar width as a function of n (200 / 500 / 1000 token prefixes).

## Analysis lock

Analysis script: `scripts/grid_certificate_experiments.py` (to be written
after this document is committed; the script must implement exactly the tests
above). Any rerun uses the same seeds. If a stream fails to generate (e.g.
EOS collapse before 1000 tokens), it is excluded and the exclusion count
reported; thresholds do not change.
