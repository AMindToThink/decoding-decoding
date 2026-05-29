# Full-Sequence Structured Twist — Review & Implementation Plan

**Status:** the `full-sequence/` folder is a *design/method spec*, not a finished experiment.
It contains four markdown docs + two HTML memos. There is **no runner, no results JSON, no
findings doc** yet. This document (a) consolidates a five-agent adversarial review of the spec
(full reports under `full-sequence/review/`), and (b) proposes an implementation plan. **Nothing
in the spec or code has been edited.**

Reviews (saved, web-cross-referenced):
- `review/fs_review_math.md` — equation-by-equation audit
- `review/fs_review_method.md` — ML/RL soundness
- `review/fs_review_cites.md` — citation verification against the real papers
- `review/fs_review_concept.md` — conceptual / framing
- `review/fs_review_repo.md` — repo ground-truth inventory

---

## 1. One-paragraph verdict

The **core math is correct** — the Q_pt/Q_g ratio, the future-partition-function decomposition
`Q_g(x_t|x_<t) ∝ P(x_t|x_<t)^{1/T}·F(x_{1:t})`, the backward recursion, and the
candidate-conditioning argument all check out (`review/fs_review_math.md` §1–4, 9). The **citations
are accurate** (one attribution slip, below). The **framing has fixable defects** (one real
self-contradiction; novelty oversold). The **real risk is empirical, not mathematical**: the method
bets that a *log-linear twist over a 1-D streaming latent* recovers enough of the exact
sequence-tempering correction to beat per-token tempering by a measurable margin — and there are
strong a-priori reasons (LHTS's own null on open-ended GPT-2 MAUVE; a scalar latent provably cannot
represent F) to think that margin may be ~0 at the mild temperatures where the smooth estimator is
best-behaved. **The single most important change to the plan: run a cheap go/no-go existence test
*before* building any of the streaming machinery.**

---

## 2. Defects to rectify in the spec (the "mistakes")

Ordered by severity. Citations are `file:line` into `full-sequence/files/`.

### A. Real defects (must fix before/while implementing)

| # | Where | Defect | Fix |
|---|-------|--------|-----|
| A1 | `01:49-50` def. vs `03:100` (`if x_t==EOS: break`) and `03:150,166` (per-example anchor) | **Fixed-N vs EOS inconsistency.** F and Q_g are defined only for **fixed N**, but the sampling loop terminates early on EOS and fit B anchors at each prefix's own variable end. Sampling/fitting then target a *different* distribution than the fixed-N F corrects. | **Pick one regime.** Recommended: **fixed-N** — sample/score length-N windows; anchor only true length-N states. (Or: define an EOS-terminated F with terminal probabilities — strictly more work.) |
| A2 | `02:69`, `03:166` terminal anchor; see `review/fs_review_math.md` §5d, §7 | **Terminal boundary under-constrained.** A single soft anchor `log F̃(θ_N)=0` fixes only the one global additive gauge; it does **not** enforce `F=1` at the off-path length-N latents the `t=N` bootstrap target references. With `log F̃=w·φ(θ)` linear, the horizon residual is under-determined. | Anchor on **many** sampled length-N latents (or design `φ` so `φ(θ_N)=0` by construction). Document the soft-anchor weight as a real hyperparameter. |
| A3 | `03:171` ("subsample v … keep the chosen token") | **Biased estimator (latent correctness bug).** Subsampling inside a `logsumexp` is **not** unbiased (Jensen: `E[log Ŝ] ≤ log E[Ŝ]`); "keep the chosen token" makes the inclusion prob token-dependent, biasing `w` downward. Confirmed by the sampled-softmax-bias literature *and* the project's own MEMORY note. | Use the **full-vocab** logsumexp (affordable — it's a constant factor over the forward pass you already pay). If subsampling is ever forced, use an importance-corrected **sum-inside-log** and unit-test against exact. |
| A4 | `02:71`, `01:56` | **Citation attribution slip.** "the same squared-residual objective Twisted SMC *uses*" is wrong: Twisted SMC's own objective is **contrastive (CTL)**; squared-error / soft-Q-learning is a *prior-work baseline* (Mudgal 2023 / Han 2024) it unifies and beats. The math analogy is fine; the attribution is not. | Reword to "…the squared-residual / soft-Q-learning twist objective that Twisted SMC includes as a baseline (its own proposed objective is contrastive)." |
| A5 | `01:68`, `00:39`, `sequence_temperature.html:146` | **Internal self-contradiction.** "the imprint is the reason naive per-token sampling misses Q_g" contradicts the (correct) earlier claim that the mismatch "exists even for a model with zero in-context drift … a property of the operator" (`01:65-66`). | Reword: the **operator** mismatch (tempering does not commute with conditioning) is *why* per-token misses Q_g; the imprint only *modulates how large* the miss is. |

### B. Imprecise / oversold (fix in prose; do not block the method)

| # | Where | Issue | Fix |
|---|-------|-------|-----|
| B1 | `01:39` | "soft-Bellman backup" label is pinned to the **linear** (Feynman–Kac) recursion. The soft/log-sum-exp structure appears only after taking logs. | Move the label to the log-domain line: `log F = logsumexp_v[β log P + log F']`; call F the desirability function, `log F` the soft value. (Standard: Todorov 2009 / Kappen path-integral; Levine 2018.) |
| B2 | `00:README`, `02` framing | **Novelty oversold** outside the (honest) lit section. The target (LHTS), the proposal×twist decomposition and the squared-Bellman-residual fit (Twisted SMC) are all prior art. | Add one sentence up front: *"This is a special case of the Twisted SMC twist, restricting it to a log-linear map over a low-D streaming latent; the question is how much of the correction survives that restriction."* |
| B3 | `02:89-92`, `00:51` | **"Residual floor = imprint dimensionality" is not licensed**, and the "measurable margin" bar has **no pre-registered number**. A high floor confounds genuine dimensionality with a bad `φ` (nonlinearity), wrong latent coordinate, and off-distribution calibration. | Pre-register a margin (e.g. *X* nats KL-to-Q_g reduction, or *Y*% PPL/MAUVE at stated T). Before attributing a floor to "dimensionality," control `φ`'s function class (rich `φ` on scalar θ) separately from latent dimension. |
| B4 | `03:54-57` latent = surprise | The spec's *placeholder* latent is decayed mean **surprise**. The repo's **actual** filter tracks **log β** (inferred inverse-temperature) from the centered logit `ℓ_t[x_t]−E_q` — and the spec says "defer to the implementation." log β is arguably a *better* coordinate for a temperature twist, but the choice should be argued, not inherited. | Use the implemented filter's log-β latent; justify it on its own terms (natural conjugate observation for a tempering correction), or sweep coordinates. |

### C. Points in the method's favor the spec under-sells (keep, state explicitly)

- The inner `Σ_v` is computed **exactly** from full-vocab logits, so the classic
  residual-gradient **double-sampling bias does not arise** (`review/fs_review_math.md` §6). Because
  it's exact, the **full residual gradient is unbiased and available here** — the spec defaults to
  semi-gradient (stop-grad) without noting it could use the unbiased full gradient.
- The candidate-conditioned twist is **cheap and fully vectorizable** for the implemented filter
  (see §3, Phase 1): no Python loop over the 150k vocab is needed.

> **Multi-pass check (repo rule):** the spec does **not** contain growing-prefix multi-pass scoring;
> the sampling loop and fits use one forward pass per step. No multi-pass red flag here. The relevant
> adjacent point (full-vocab sum is exact, subsampling is not) is captured in A3.

---

## 3. Implementation strategy — **go/no-go first**

The reviewers converge on one strategic point: the streaming filter + log-linear map + semi-gradient
TD + chicken-and-egg refit is a large apparatus built to *approximate* F, and we do not yet know that
F's correction is even non-negligible on real text at usable T. Build the cheap test first.

### Phase 0 — GO/NO-GO existence test (do this before anything else)

**Goal:** decide whether the project is alive, with ~hours of compute and **no streaming sampler**.

- On a few hundred prefixes, compute a **depth-h exact lookahead** estimate of `log F(x_{<t},v)` for
  each candidate `v` (top-k candidates, h=1,2,3 — `k^h` forward passes per step), at `T ∈ {0.7, 0.9}`
  (and a small-T point, e.g. 0.3–0.5, where the gap is largest).
- **Measure two things:** (a) the cross-candidate **spread of `log F`** in nats vs. the spread already
  induced by `(1/T) log p_v`; (b) the **R²** of the streaming latent (log β) against the exact `log F`
  target.
- **Decision rule (pre-register):** if the `log F` spread ≪ tempered-logit spread at mild T → the
  correction is below noise; **stop, or pivot to small T**. If spread is non-negligible *and* R² is
  high → proceed to Phase 1. (This same depth-h lookahead doubles as the **distillation teacher** in
  Phase 3 and the **exact reference** in Phase 4.)

### Phase 1 — Infrastructure (small; mostly reuse)

- **`peek`/`commit` on the filter — nearly free.** `laplace_update` (`src/decoding_decoding/counter_decode.py:91-182`)
  is **already pure** (it returns a fresh `LaplaceState`, no mutation). `commit = laplace_update`;
  `peek` = the same call discarding the side-channel. The scout's "must add peek/commit" overstates the
  work.
- **Candidate-conditioning is vectorizable.** At a step, `J_new = J + α·Var_q` is **identical for every
  candidate** (Fisher doesn't depend on which token you hypothesize), and `score(v) = ℓ_t[v] − E_q` is
  just the centered-logit vector. So `η̂(v) = η̂ + α·(ℓ_t[v] − E_q)/J_new` is **one length-V tensor op** —
  no per-token loop over 150k tokens.
- Reuse: full-vocab `log_softmax`, `torch.logsumexp`, model+tokenizer loader, WikiText/WritingPrompts
  corpus loaders, the `(γ, α)` knobs already in `laplace_update`.

### Phase 2 — The sampler (`Q_g` sampler with candidate-conditioned twist)

- Implement `03_pseudocode.md` §3 against the **real** filter interface, with the fixes from §2:
  fixed-N horizon (A1), full-vocab reweight, max-subtracted twist for stability.
- The latent coordinate is **log β** (B4), `φ(θ)=θ` to start (scalar), `log F̃(θ)=w·θ`.

### Phase 3 — Fits

- **Prefer distillation over bootstrapping.** Regress `log F̃(θ)` on the **depth-h lookahead target**
  from Phase 0 (a fixed, correct-as-far-as-it-goes target). This sidesteps the semi-gradient / deadly-
  triad instability (`review/fs_review_method.md` §4) entirely. The spec lists distillation as option 3
  but defaults to the bootstrap; **the ranking should be reversed** given the soundness risks.
- If the bootstrap (fit B) is used anyway: **full-vocab** logsumexp (A3), document the semi-gradient
  fixed point ≠ argmin E[r²], anchor on many terminal latents (A2).
- **Fit A (filter knobs) is optional.** Per the concept review, the method is **imprint-agnostic** — F
  is defined and fit from base conditionals alone. Fold knobs into the joint fit, or warm-start from the
  existing `(γ, α, σ_0)` and skip Fit A in the first pass.

### Phase 4 — Evaluation (the part the spec is weakest on)

- **Primary metric: KL(realized sampler ‖ Q_g)**, estimated via SNIS weights
  `w = Q_g_unnorm / sampler_prob` on rollouts — *not* the Bellman residual. Low residual ⇏ low KL
  (`review/fs_review_method.md` §2): error accumulates over steps (no discounting) and compounds under
  autoregressive drift. Demote `E[r²]` to a diagnostic.
- **Honest baselines:** per-token tempering (`F≡1`); depth-h lookahead (upper reference); optionally
  add SMC resampling to the twist (turns it into honest Twisted-SMC with a cheap structured twist — a
  defensible fallback contribution if the standalone sampler underperforms).
- **Pre-register the win margin** (B3) and the **task** — pick a likelihood/constrained metric (XSUM
  perplexity-style, where LHTS *did* win) rather than open-ended MAUVE (where LHTS's full fine-tune got
  0.00). Report bootstrap CIs over prompts.

### Phase 5 — Toy validation (optional; uses ground-truth F)

- Mixture-of-HMMs (Xie-style) with exact F by DP. **Role is narrow:** a one-time *existence/falsification*
  test of the log-linear-in-latent functional form — it can only **falsify** (form can't fit even when
  the latent is sufficient by construction), never **validate** for the LLM (`review/fs_review_method.md`
  §3). Also the cleanest place to demonstrate the objective-goal gap (correlate `E[r²]` vs exact KL).
- **Absent in repo** — build from scratch only if Phases 0/4 justify it.

### Cross-cutting (repo conventions)

- **Numbers-from-scripts:** every inline number in any findings doc must come from a generated
  `macros.tex` (mirror `scripts/build_calibration_baseline_macros.py`). No hand-typed numbers.
- **Fail loud:** validate preconditions up front; no `try/except: continue` swallowing failed prefixes.
- **Tests:** unit-test the twist reweight against a brute-force fixed-N `F` on a tiny vocab; unit-test
  the logsumexp estimator vs exact; type hints throughout; `uv run pytest`.

---

## 4. Decisions you own (flagged, not assumed)

1. **Base model — Qwen2.5-3B vs GPT-2.** *Correction: the spec does NOT specify a model.* Every GPT-2
   mention in the spec (markdown + HTML) is inside the **LHTS citation** (describing that paper's
   experiments), not a recommendation for this project. The spec instead says "defer to the existing
   implementation … the repo already contains a working streaming filter" — and that filter runs on
   **Qwen2.5-3B**. So the spec points *toward* **Qwen** (maximal reuse of the filter + all prior
   counter-decode experiments; downside ~150k vocab → ~3× per-step twist cost). The only reasons to
   consider **GPT-2** are external, not from the spec: apples-to-apples comparison with LHTS/Twisted-SMC,
   a cheaper ~50k vocab, and a more tractable toy-DP (Phase 5). This is a genuine choice, but the default
   the spec implies is Qwen, not GPT-2.
2. **Scope — go/no-go first vs commit to the full pipeline.** Recommended: Phase 0 first, decide from its
   result. The alternative (build Phases 1–4 regardless) risks a large build whose headline comes out ~0.
3. **Target regime — fixed-N (recommended) vs EOS-terminated** (A1).
4. **Fit — distillation-from-lookahead (recommended) vs bootstrap** (Phase 3).

---

## 5. Reuse-vs-build

**Reuse (production-ready):** Laplace filter core + init (already pure → trivial peek/commit); full-vocab
`log_softmax` / `logsumexp`; Qwen loader; WikiText/WritingPrompts loaders; `(γ, α, σ_0)` knobs; the
`build_*_macros.py` numbers-from-scripts pattern; the `(γ,α,p)` sweep harness.

**Build:** Phase 0 depth-h lookahead (also teacher + reference); twist map `w·φ(θ)`; candidate-conditioned
fixed-N sampler; the fit (distillation first); SNIS KL-to-Q_g evaluator; optional toy-HMM DP.

**Note:** `double_sharpening.html` referenced in the spec = `in-context-learning-of-decoding.html` in the
repo root (filename is a spec placeholder). Experiments A and C exist (`calibration_baseline/`,
`entropy_probe/`); B is implicit; E is absent.
