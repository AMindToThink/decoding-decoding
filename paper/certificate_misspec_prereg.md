# Pre-registration: certificate under misspecification + grid-resolution floor (E1, E4a)

Committed BEFORE E1 and E4a are run or analyzed. No confirmatory hypothesis,
test, sample size, or threshold below may change after seeing the real data;
anything not pinned here is exploratory/descriptive and is labeled so in the
writeup. Companion to `paper/grid_certificate_prereg.md` (H1–H4, already run);
absorbs the overclaim fixes in `paper/claim_audit.md` (esp. Claim 1).

**This document was revised once after an adversarial design critique** (fresh
agent, before any real data): it caught (a) E4a's original "peaked logits"
premise was backwards — peaked logits give *low* curvature and *wider* bars, so
the grid floor lives at high n, not peaked logits; (b) the width-ratio statistic
was undefined at the floor; (c) a strictly content-matched E1 control exists
(the rule's projection residual on a shared logit set). All three are fixed
below; the fixes were made before generating E1/E4a data.

**Estimator under test:** `grid_mixture_update_batched` / `grid_mixture_level_set`
(`src/decoding_decoding/counter_decode.py`): Bayes mixture over fixed temperature
experts on RAW logits, uniform prior, `switch_rate=0`. β̂ = grid MAP. Default
grid `make_beta_grid(log_lo=-3, log_hi=3, n_points=121)` (log-β step 0.05); E4a
also runs a 241-point grid (step 0.025). Error bar = level set at threshold
c = ln G. Filter math float64 (synthetic), float32 (model streams).

**Honesty labels (do not relabel):** E4a is **confirmatory** (synthetic data not
yet generated). E1 is **confirmatory on the fit hypotheses** (the shared-logit
projection-residual test) and **descriptive on the effective-β readout**. E2 is
**descriptive** (its data — `data/grid_certificate/model_streams.npz` snapshots —
is already committed and was analyzed for H4). E3 is **optional/stretch**.

---

## Pilot that set E1's threshold (synthetic, design-stage, NOT the experiment)

`scripts/e1_fit_pilot.py` (committed), Gaussian logits matching H1 (vocab 25,
scale 1.5, n=1000, 8 seeds), found:

| arm | log β̂ | bar width | best-expert loss/n | projection residual ρ_proj |
|---|---|---|---|---|
| control      | −0.01 | 0.11 | 2.36 | ~0.000 |
| top-k (k=5)  | +0.53 | 0.13 | 1.61 | 0.213 |
| top-k (k=3)  | +0.78 | 0.12 | 1.24 | 0.276 |
| blacklist (ban top-1) | −0.45 | 0.18 | 2.82 | 0.273 |

Two design facts this fixed:
- Best-expert *token loss* is **lower** under top-k (a truncated stream is more
  predictable), so raw token loss is confounded by stream entropy and is NOT the
  fit metric. The unconfounded quantity is the **temperature-family projection
  residual** ρ_proj = min_β mean_t KL(P_rule(·|ℓ_t) ‖ softmax(β·ℓ_t)): identically
  0 for the untruncated rule (β = 1 reproduces it) and clearly positive for every
  truncation/ban rule (no β reproduces a truncated categorical). The plan's draft
  hypothesis ("top-k loss exceeds control") was mis-signed; this replaces it.
- The headline tension is real: top-k gives a **narrow bar (0.12) AND poor fit
  (ρ_proj = 0.28)** — identifiability ≠ goodness of fit; and the blacklist rule
  registers β < 1 (banning the mode flattens the realized stream).

These set the metric and δ = 0.05 nats/token below; they are NOT the reported
result. ρ_proj is computed on real streams exactly, since E1 captures full-vocab
logits and the decode rules are known.

---

## E4a — Grid-resolution floor (synthetic, CPU, confirmatory)

H2 reported a median real-stream bar width of one coarse grid step (0.05) at
n = 1000 — possibly a grid-quantization floor rather than a measured data width.
E4a settles this by driving the *true* (continuum) bar width below the grid step
with **more data** (curvature Λ_n ∝ n, continuum half-width ∝ 1/√n), at fixed
natural logit scale, and asking whether the grid level set can still resolve it.

(Why n, not peaked logits: the level-set width is set by curvature
L''(β) = Σ_t Var_{softmax(βℓ_t)}(ℓ_t) — `counter_decode.py:564`. Peaked logits
concentrate the softmax, *lower* the variance, and *widen* the bar — consistent
with H3. A narrow bar / grid floor therefore comes from high curvature = large n,
not from peaking.)

- **Data:** H1-style synthetic streams (`make_synthetic_stream`), scale 1.5,
  β* = 1, float64, n ∈ {1000, 4000, 16000, 64000}. Seeds 0–199 for
  n ≤ 16000; seeds 0–99 for n = 64000 (a declared compute bound, not a threshold).
- **Resolutions:** coarse = 121 points (step 0.05); fine = 241 points (step
  0.025); range [−3, 3], uniform prior, `switch_rate = 0`.
- **Per stream, record:**
  1. grid level-set width w_grid = hi − lo (log β) at each resolution (an *inner*
     approximation: the bar collapses to 0 when the true interval sits inside one
     grid cell);
  2. **continuum** profile-likelihood interval width w_cont: root-find
     L_n(β) = min_β L_n + c on each side of the continuum MLE with
     `scipy.optimize.brentq` on [e⁻³, e³] (well-defined and never collapses to 0).
- **Statistics per n:** median w_cont, median w_grid (coarse, fine), and the
  **bar-collapse fraction** = fraction of streams with w_grid = 0, per resolution.
- **Confirmatory hypotheses (committed now):**
  - **H-E4a-1 (width is data-limited, shrinks with n).** Prediction: median w_cont
    is monotone decreasing in n and median w_cont(64000) / median w_cont(1000)
    < 0.35 (1/√64 ≈ 0.125 ideal; 0.35 is a robust ≥ ~3× shrink). **Confirm if
    monotone AND ratio < 0.35.** ⇒ the bar width is genuinely data-driven, not
    pinned at a constant.
  - **H-E4a-2 (the coarse grid floors where the continuum still resolves).**
    Pre-committed test point n = 16000 (curvature predicts w_cont ≈ 0.043,
    i.e. < one coarse step). **Confirm a coarse floor if, at n = 16000,** the
    coarse-grid bar-collapse fraction > 0.5 **AND** the fine-grid collapse
    fraction is strictly lower than the coarse-grid collapse fraction. ⇒ the
    coarse grid cannot resolve sub-step widths the continuum (and a finer grid)
    can.
  - If H-E4a-1 fails (width does not shrink), or H-E4a-2 fails (coarse grid does
    not floor at n = 16000, or the fine grid floors equally), the corresponding
    claim is reported as not confirmed; thresholds do not move.
- **Consequence for H2 (pre-committed):** if H-E4a-2 confirms, the paper reports
  H2's 0.05 median as at/near the coarse grid-resolution floor — a lower bound on
  resolvable width, not a measured data width — and points to the continuum
  interval (or a finer grid) as the honest width when curvature is high.

---

## E1 — Certificate under misspecification (model streams, GPU)

The walkthrough claims the certificate is most valuable under top-k / blacklist;
nothing measures it. E1 measures (i, **confirmatory**) that the temperature
family genuinely cannot fit a truncated/banned stream — content-matched, with the
control fit residual identically 0 — and (ii, **descriptive**) the effective β
each decoding rule registers in actual deployment.

### Generation (fresh streams, full-vocab raw logits)

- **Harness:** the HF-transformers path (extend
  `scripts/grid_certificate_experiments.py::run_model`, which already captures
  full logits and calls `sample_under_beta`). The legacy vLLM banning pipeline
  (`generate.py`) saves only top-200 logprobs, so it is reused for its
  *token-banning logic only*, not its storage.
- **Model / prompts / length:** Qwen/Qwen2.5-3B float16, the 32 `PROMPTS`,
  n = 1000 generated tokens, generation seed 0 (H2/H4 convention).
- **Arms (raw full-vocab logits ℓ_t captured every step in all arms):**
  1. **control** — plain temperature-1 sampling on the full vocab.
  2. **top-k** — k ∈ {3, 5} (the legacy `generate.py` CONDITIONS); logits outside
     the per-step top-k set to −∞ before sampling.
  3. **blacklist** — a fixed committed banned-token list; logit → −∞ before
     sampling. **Provenance lock:** the list is regenerated by
     `scripts/analyze_token_frequencies.py` (the legacy source) with its default
     config, written to `results/tables/banned_tokens.json`, and that file's
     sha256 is recorded in this prereg **before** the blacklist run. (The file is
     not in the repo today; committing it with its hash is part of locking E1.)

### (i) Confirmatory fit test — content-matched, on a shared logit set

The raw logits are identical whether or not a rule bans/truncates at the sampling
step, so the rule's *fittability* is a property of the rule applied to a fixed
logit set — no stream divergence, perfect content-matching (absorbs claim-audit
Claim 1). On the **control arm's** captured logits {ℓ_t} (32 streams × 1000 steps)
compute, for each rule r ∈ {topk3, topk5, blacklist}:

  ρ_proj^r = min_{β ∈ grid} mean_t KL( P_r(·|ℓ_t) ‖ softmax(β·ℓ_t) )   [nats/token],

where P_r is rule r applied to ℓ_t. ρ_proj^control ≡ 0 by construction (β = 1).
Per-stream (per-prompt) statistic; the three rules are evaluated on the *same*
logits, so the comparison is exactly paired.

- **Confirmatory hypotheses (δ = 0.05 nats/token, set from the pilot):**
  ρ_proj^topk3 > δ, ρ_proj^topk5 > δ, ρ_proj^blacklist > δ.
- **Test:** one-sample one-sided Wilcoxon signed-rank (`scipy.stats.wilcoxon`,
  alternative="greater") on (ρ_proj^r − δ) over 32 prompts; **Holm–Bonferroni**
  across the three rules; **confirm a rule if its adjusted p < 0.05.** Report
  median ρ_proj^r + 95% bootstrap CI. δ = 0.05 is conservative: the pilot put
  control ρ_proj ≈ 0 (grid-quantization floor verified < 0.001 nats) and every
  rule ρ_proj ≥ 0.21, so δ sits ~120× above control noise and well below effect.
- **Failure consequence (pre-committed):** a rule that fails is reported as fit
  by the temperature family within δ (no detectable misspecification residual);
  no fit-gap claim is made for it.

### (ii) Descriptive effective-β readout — realistic deployment (labeled)

On each rule's **own** generated stream (its realized, divergent trajectory) run
the online certificate and report: grid MAP β̂, level-set bar (lo, hi, width),
realized regret, and the realized fit residual at β̂. Pilot-based expectations
(descriptive, not tests): top-k → log β̂ > 0 (sharper); blacklist → log β̂ < 0
(banning the mode flattens the realized stream). Report (bar width, fit residual)
jointly to show top-k yields a **narrow bar with positive misfit** — the
"narrow bar ≠ good fit" point. Trajectories diverge across arms; this is the
deployment readout, not the controlled comparison (that is part (i)).

- **Rail policy (pre-committed):** if β̂ hits a grid rail (log β̂ ∈ {−3, +3}) on
  > 20% of an arm's streams, the arm is reported as unidentified-at-the-rail and
  the grid is widened for it; ρ and the bar are still reported but flagged. (A
  synthetic check found k ∈ {3, 5} land at finite interior β̂; k = 1 rails — k = 1
  is not run.)

### Verification row (NOT a hypothesis)

Realized regret ≤ ln G holds by the mixture Lemma for any token stream; reported
as machinery verification, never as a falsifiable finding.

### Honest caveats (prereg + paper)

The certificate certifies how well the *observed stream* is fit by the
temperature family — never the counterparty's dial. Top-k/blacklist have no dial;
the effective β is a projection, not a recovered setting. A narrow bar means L_n
is sharply curved (identifiable), not that any β fits well (ρ_proj > 0 quantifies
the misfit). The blacklist β < 1 reading is "the realized stream looks flatter,"
not "the decoder lowered temperature."

---

## E2 — Online-approximation lag (descriptive, CPU, zero new runs)

**Descriptive, not confirmatory** — the data (`model_streams.npz` snapshots) is
committed and was already analyzed for H4; not dressed as pre-registered. From
the stored snapshots: grid MAP (argmin of `cum_loss`) vs `eta_faithful` vs
`eta_original`, error in log β to the known dial β_target ∈ {0.5, 1.0, 2.0} at
n ∈ {200, 500, 1000}. Framed as *"how much does the online Laplace approximation
lag the batch-equivalent grid MAP"*, not "which estimator is better" (the grid
MAP is the grid-quantized batch MLE, so it winning is largely by construction).
Report the ½-grid-step quantization floor (0.025 in log β) when reading errors at
β_target = 1. The dial is a fair target only on these well-specified streams.

## E3 — Closed-loop (GIGO) arm with the certified filter (optional/stretch, GPU)

Minimal two-arm rerun (`run_counter_decode_experiment` with
`prior_kind="grid_mixture"` as the β̂ backend), β_target ∈ {0.5, 2.0}, few seeds.
**Confound stated in prereg and paper:** at β_target = 0.5 the stream degrades to
token salad (claim-audit Claims 3/5); the certificate then certifies the
*salad's* effective temperature — a narrow bar is NOT dial recovery. The only
claim is "the corrector consumes a certified estimate of the observed stream." If
that sentence cannot carry a section, **cut E3 rather than inflate it.**

---

## Analysis lock

E4a + E1 analysis extends `scripts/grid_certificate_experiments.py` (or a sibling
with the same subcommand / results.json / deterministic-seed structure); build
script mirrors `scripts/build_grid_certificate_table.py` (emit macros + table).
Fail fast on bad preconditions — no silent `continue`. Any rerun uses the same
seeds. Streams that fail to generate (e.g. EOS collapse before n) are excluded
and the exclusion count reported; thresholds do not change. `scripts/e1_fit_pilot.py`
is committed; δ = 0.05 was fixed before any real E1 stream existed. The blacklist
`banned_tokens.json` + sha256 is committed before the blacklist run.
