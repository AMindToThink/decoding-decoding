# Claim audit — `decoding_paper.tex`

Adversarial review of every claim in the report, done before finalizing. Method: 8 fresh-context
critique subagents (one per claim-cluster) source-traced each cited number to its JSON/script and
attacked whether the claim *follows* and whether the experiment *distinguishes* what it asserts.
Every load-bearing objection below was then re-verified by reading the source directly (file:line
given). **No GPU re-runs** — this audits reasoning + traceability, not reproduction.

How to read a row: **Claim** → what the prose asserts. **Trace** → do the numbers resolve to a
script? **Surviving objection** → the strongest attack left after discarding ones I could refute.
**Verdict** → `KEEP` / `TIGHTEN` / `CAVEAT` / `SPLIT` / `DOWNGRADE`. Verdicts that change a claim
are flagged **[ESCALATE]** — they are the author's call, not mine.

---

## TL;DR

**Good news — integrity is clean.** All 43 data-value macros used in prose are script-generated
and defined; no cross-section inconsistencies; the one hand-typed result number is `15\%` at
`decoding_paper.tex:264` (should reference the existing `\fzeroRfinal=0.15`). The
numbers-from-scripts discipline held.

**The problem is inferential overclaim, from four recurring root causes:**

1. **GIGO (garbage-in/garbage-out).** β<1 self-generation becomes token-salad within ~20 steps
   (the paper concedes this at `:410`). That off-distribution text contaminates: §4's "overshoot"
   mechanism, §4's SVD residual, and is the *uncontrolled confound* that guts §6's "decisive" framing.
2. **Pooled-vs-median estimator ambiguity.** The filter has two aggregations under one name.
   §5 uses the benign *median* for "hurts as calibrator" and the catastrophic *pooled* one-step
   estimator for "different fixed points" — inflating the latter claim.
3. **Invariance tested only at β_target=1** (the trivially-calibrated case), where the right
   action is "don't correct." That can't support §7's "γ<1 is load-bearing."
4. **§3's positive evidence is softer than "infers its decoder."** The control is not
   content-matched, and the top-k cliff does not peak at K.

**What still stands:** the *conceptual* thesis (an LM responds to decoder-shaped context) keeps
support — but its cleanest evidence may be the WritingPrompts natural-text offset (§5, paired
t=−3.80, p<10⁻³), not the §3 experiments the paper leans on. Honest posture: the belief-imprint
hypothesis is **plausible and partially supported, with the strongest individual claims needing
to be softened** — consistent with the paper's own "refined, not refuted" stance, but more so.

---

## Claim 7 (integrity audit) — **CLEAN**
- **Trace:** all prose macros defined & sourced; build script reads the 11 JSONs + 2 ingested
  macro files it claims. No orphans, no undefined macros.
- **Action:** replace hand-typed `15\%` (`decoding_paper.tex:264`) with `\fzeroRfinal`-derived
  macro (it restates the SVD residual already cited correctly one clause earlier at `:263`).

## Claim 1 — Positive evidence: blacklist & top-k self-prediction (§3) — **CAVEAT [ESCALATE]**
*This is the paper's stated "load-bearing positive evidence" (`:199-201`), so softening it matters most.*
- **Trace ✓:** `\blRankOneDeltaP`=−0.019, `\blRankOneTvc`=−0.016 CI[−0.019,−0.013], `\tkDeltaKthree`=+0.475,
  `\tkDeltaKfive`=+0.170 all resolve and match `blacklist_summary.json`, `topk_summary_L2048.json`.
- **Surviving objection (verified):**
  - **Control is not content-matched.** Treatment seeds `BLACKLIST_SEED_OFFSET+…` (`generate.py:422`)
    vs control seeds `SEED_BASE+…` (`generate.py:96`) — different realized token streams, sharing
    only the prompt. So "holding content fixed" (`:152`) is inaccurate, and treatment−control
    cannot separate "infers a blacklist" from "conditions on an altered context where the banned
    token is simply absent." Effect fades exactly with token frequency (rank-4 ` of`≈−0.004, rank-5
    ` to`≈0, CI crosses zero) — the signature a frequency/context-recession story predicts.
  - **Top-k cliff does not peak at K.** `peaks_at_K: false` for k=3 and k=5; the extra gap is
    *larger* at K−1 (0.864) than at K (0.475) than at K+1 (0.240) (`topk_summary_L2048.json:175-210`).
    So "not generic drift but a response to the truncation" (`:188-189`) overstates: the data are
    consistent with broad truncation-induced sharpening, not a K-specific signal.
- **Suggested wording:** the supported claim is *"the model conditions its predictions on its
  realized context, which the decoder shapes"* — weaker than *"infers its decoder."* Either
  (a) re-run the blacklist control as the **same** banned text with the ban lifted (content-matched
  counterfactual), or (b) drop the H3 specificity framing, report `peaks_at_K=false`, and downgrade
  "not generic drift." Consider promoting the §5 WritingPrompts offset as the cleaner positive signal.

## Claim 2 — Streaming Laplace filter / method (§2) — **REFRAME [ESCALATE]**
- **Trace ✓:** code matches the score/Fisher/Newton equations exactly (`counter_decode.py:153-180`);
  prior init `μ₀=−σ₀²/2`, `J₀=1/σ₀²` (`:85-87`). Tests prove shift-invariance, batched=serial, and
  recovery *under matched-prior synthetic data* — none test the real-LM imprint assumption.
- **Surviving objection (verified):** "a per-token Bayesian update, **not** an iterated Newton
  solver" (`:127-129`) is word-play. `score = ℓ_t[x_t]−E_q` is d/dβ log softmax(βℓ)[x] at β=1 and
  Var_q is the Fisher, so `η̂ = μ₀ + Σscore/(1/σ₀²+ΣVar)` is **algebraically one damped
  Newton/Fisher-scoring step from the prior** — which *is* a Laplace posterior update. The
  β̂-independence is a **definitional** modeling choice, and §5 itself (`:313-315`) admits it is the
  *source* of the bias. §2 sells it as a principled feature; §5 reveals it as the flaw.
- **Suggested wording:** call it a *prior-regularized one-step online MLE / Fisher-scoring update on
  log β*; cut the "Bayesian-not-Newton" sentence; and **define the two aggregations (median vs
  pooled) in §2**, since the paper later uses both under "the filter."

## Claim 3 — Corrector & concentration dynamics (§4) — **CAVEAT [ESCALATE]**
- **Trace ✓:** `\fzeroMaxPcorrHalf`=0.579, `\fzeroLogbhatHalf`=−4.74, `\fzeroEuncoHalf`=4.74,
  `\fzeroRfinal`=0.15 resolve and match F0/F1.2/F3/SVD JSONs.
- **Surviving objection (verified):**
  - **Overshoot is GIGO.** log β̂→−4.74 at β_target=0.5 is the filter ingesting token-salad logits
    (β=0.5 decoheres by ~step 20, per §6 `:410,529`). §4's mechanistic "overshoot of a belief"
    story contradicts §6's own concession.
  - **Asymmetry is partly metric geometry.** Corrected max-P sits ~0.52–0.60 across *all* targets
    (roughly symmetric); the apparent asymmetry is the *uncorrected* arm hitting the max-P=1 ceiling
    for β>1 vs spreading over the ~151k-vocab simplex for β<1. Bounded-above / unbounded-below.
  - **SVD residual.** `_center_logits` removes the *per-decoder* vocab mean (`counter_decode_analyze.py:301-307`),
    so the prose "centered **across-decoder** logit matrix" (`:262`) misdescribes it. K=6 columns,
    two GIGO-contaminated; R climbs monotonically with t. "15% not explained by scalar β" is partly
    the off-distribution low-β columns, not a stable structural residual.
  - **Double-count:** `\fzeroEuncoHalf`=4.74 and `|\fzeroLogbhatHalf|`=4.74 are the *same* number
    (E_unco = |log β̂ − log β_target|), presented at `:244` and `:260` as if two corroborating facts.
- **Suggested wording:** keep "corrected arm stays bounded / uncorrected runs away / E(T)≈0 is
  tautological" (all sound). Add a GIGO caveat to the confusion-regime β̂; soften "asymmetry" to note
  the metric geometry; fix "across-decoder"; present the overshoot and E_unco as one measurement.

## Claim 4 — Filter ≠ Guo MLE + calibration + Exp B (§5) — **SPLIT [ESCALATE]**
- **Trace ✓:** all cal* macros resolve. `\calTestPPLWPWPstreammed`=24.989 (median) vs
  `…streampooled`=143.398 (pooled) confirmed in `calibration_baseline/macros.tex:44-49`.
- **What KEEPS:** "the filter (median β̂) hurts as a calibrator" — PPL 19.55→24.99, t=+26.46,
  p=2×10⁻³⁴ — real, large, significant. And "calibration and imprint-detection are distinct
  *targets*" as a conceptual point.
- **Surviving objection (verified) — what does NOT keep:**
  - **"Different fixed points / asymptotic-equivalence does not survive"** rests on the *pooled*
    estimator, a single damped Newton step that **by construction cannot sit at the MLE** on a flat
    NLL surface. The experiment never tests the *large-data limit* the proposal actually invoked;
    `run_calibration_baseline.py:25-27` even says the two "should agree closely by construction."
  - **"Qwen already calibrated"** rests on MLE log β≈+0.0004 with a 3×10⁻⁷-nat gain on WikiText —
    machine noise; Qwen near-certainly trained on Wikipedia, so this is in-distribution-trivial.
    Hedge as corpus-specific.
  - **Exp B genre gap +0.054 is not significant** (n=4/genre, p≈0.07) and dwarfed 25:1 by entropy.
    The hedge "plausibly content-driven" concedes it shows nothing about the decoder.
- **Suggested wording:** keep "filter hurts / distinct targets"; reframe "different fixed points"
  as "a one-step estimator is not the global MLE" (don't claim the asymptotic limit was tested);
  hedge "Qwen calibrated" as WikiText-specific; **drop Exp B or state it as a null**.

## Claim 5 — Belief form: not a fast entropy read-out (§6) — **DOWNGRADE TO NULL [ESCALATE]**
- **Trace ✓:** `\expcLateHalf`=9.216, `\expcLateTwo`=1.172, `\expcSpread`=8.04 (t=+69.7),
  `\expcSwapPre`=8.56, `\expcSwapPostFifty`=8.91 all match `entropy_probe/summary.json`, `swap_summary.json`.
- **Surviving objection (verified):** the swap control **cannot distinguish** "belief is slow" from
  "β=0.5 content is irreversibly broken" — both predict the flat `pre05_post10`. The paper concedes
  β=0.5 = salad by ~step 20 (`:410`), which *is* the confound; the swap script's own comment states
  the GIGO prediction (`run_entropy_probe_swap.py:11-14`). The `pre05_post20` recovery is attributed
  to content re-anchoring (β=2 forces top-1) — conceding content *alone* moves entropy 9.5→0.33, so
  content (not belief) is the live variable across the whole experiment. And "slower, more
  structural — rank statistics, support shape, blacklist signatures" (`:447-449`) is **not measured
  in §6** — imported from §3.
- **Matches prior:** consistent with the project memory "Exp C only rules out a fast-entropy form."
- **Suggested wording:** drop "decisive / proves it / falsifies." State: entropy tracks realized
  content; β=0.5 damage is irreversible in-window; this experiment **neither supports nor refutes** a
  fast belief read-out — the positive case rests on §3/§5. (The swap design is the right instinct;
  it just lacks an arm with coherent post-switch content.)

## Claim 6 — Belief dynamics: closed-loop vs invariance + γ<1 (§7) — **OVERCLAIMS [ESCALATE]**
- **Trace ✓:** `\bsweepBestBadness`=0.011, `\binvCeFloor`=3.60, `\binvCeExplode`=19.44,
  `\binvVarCap`=0.07, `\binvVarFloor`=0.017 all resolve.
- **Surviving objection (verified):**
  - **Invariance runs only β_target=1**, where the correct action is p=0 (no correction). All p=0
    rows are byte-identical regardless of γ/α. So the criterion rewards *not correcting* — it cannot
    distinguish "γ<1 models the belief well" from "γ<1 throttles an action that shouldn't be taken."
  - **The CE explosion is driven by p, not γ** (3.60→19.44 as p:0→1 at γ=1). Calling γ "the
    load-bearing parameter" overstates a knob that only matters because it limits p.
  - **The recommended default (γ=0.95, α=1, p=0.5) is in no top row of either table** (invariance
    rank ~14th, var=0.0337; closed-loop top-5 all γ=1) — an eyeballed compromise with no row behind it.
  - **"Biases cancel inside the loop"** (`:500-502`) is a just-so story; the data show a
    self-consistent fixed point, not equal-and-opposite cancellation.
- **Suggested wording:** the §7 data support exactly one claim — *don't correct at β_target=1,
  because Qwen is calibrated and correction can only hurt.* The closed-loop "win" for full
  correction is a tautology of a self-generated metric. Recommending a specific (γ,α,p) for
  off-temperature use is not supported by a β_target=1-only test.

## Appendix candidate — Full-sequence NO-GO (`results/full_sequence/`) — **SOUND & TRACEABLE**
- All 4 headline numbers trace (D_total 0.33–0.51 nats CIs>0; affine_residual≈0.96; frac_filter_CV≈0;
  Spearman≈0), on gpt2-large + Qwen. Decision rule applied as pre-registered (and conservatively —
  the implemented oracle bestT is a *harder* bar than the registered global-T). K=48 candidates, so
  not a power artifact. No single-pass/multi-pass confusion (the K leaf forwards are legitimate
  depth-1 lookahead over distinct continuations). **Safe to state in one appendix paragraph as-is.**

---

## Bottom line for the author

Nothing here is fraud or a math error — the pipeline is honest and the numbers are real. The fix is
**wording and scope**: tighten §3 (control + peaks_at_K), reframe §2/§5 around "one-step estimator ≠
MLE," add GIGO caveats to §4, **downgrade §6 to a null**, and pull §7 back to "don't correct at
β_target=1." Doing so makes the paper *more* defensible, not less, and fits its own "refined, not
refuted" thesis. The cleanest surviving positive evidence is the WritingPrompts natural-text offset;
consider leading the positive case with it.
