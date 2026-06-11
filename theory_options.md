# Theory options for the counter-decoding project

Notes from the May 13 conversation about which theory targets are actually feasible
for this project. Working doc, not a deliverable — uses β freely.

Background: the streaming Laplace filter was *designer math* (concrete object: posterior
on log β; concrete operation: Newton step on log-likelihood). The math followed once the
design was committed. The Plan items in the older proposal draft are *discoverer math* —
proofs about properties of an algorithm under misspecification — and discoverer math
doesn't reliably appear on demand. These four options are ranked roughly by how close
they are to the designer-math mode that already worked once.

---

## 1. Closed-loop fixed-point analysis of the corrector

**What it would deliver.** Write down the coupled dynamics of `(β_model,t, β̂_t)` under the
working assumption that the model is doing in-context Bayes on its own inverse
temperature. Classify fixed points and their stability as a function of β_target.

**Why it's the natural next step.** The headline empirical finding ("corrector pins in
the boredom regime, overshoots in the confusion regime") is begging to be a stability
calculation. Three candidate fixed points are already visible:

- (a) β̂ → β_target (the framework's matched-prior prediction)
- (b) β̂ → 1 (model gives up imprinting; natural-reference logits)
- (c) β̂ → 0 (positive-feedback collapse through the bootstrap reference)

Each can be checked: substitute, compute the Jacobian, look at eigenvalues. The
empirical β̂ ≈ 0.009 at target 0.5 suggests (c) is the attractor in the confusion
regime — but this is a guess until the calculation is done.

**Catch.** The model's in-context Bayes is itself a black box. You'd have to either (i)
assume the simplest form (matched-prior log-normal updating, same as the filter), or (ii)
empirically fit a one-step `β_model,t+1 = f(β_model,t, x_t)` map and study *that*. Option
(i) is cleaner; option (ii) is more honest about misspecification.

**Same key as the filter derivation.** A designed dynamical system whose properties you
work out by direct calculation in a specific setting. Most likely to yield a
clean writable result.

## 2. 2-token toy model, closed form

**What it would deliver.** With `V = 2` and a fixed natural reference `ℓ*_true = (1, -1)`,
compute β̂_t analytically (or in closed-form recursion). Predict the steady state as a
function of β_target and verify against simulation. Prove statements like "for
β_target < 1, the trajectory β̂_t monotonically does X" — where X is whatever the
calculation actually says.

**Why feasible.** Your `bayesian-sampler/walkthrough.html` already works through one
trajectory at V=2 with explicit numbers at t = 0, 1, 2. Extending that to a recursion is
mechanical algebra. The closed-form makes the qualitative behavior unambiguous.

**Catch.** V=2 is a long way from V=50k. Things that hold at V=2 (e.g., specific
sigmoid identities) may not generalize. Treat as an existence proof for the
qualitative phenomenon, not a quantitative bound.

**Pairs naturally with #1.** Use the toy model as a sanity check on the fixed-point
analysis: the same fixed points should appear in the toy model's dynamics.

## 3. Identifiability / KL-projection lemma

**What it would deliver.** A statement of the form: when the off-axis residual fraction
`R` is nonzero, the population MLE on β has irreducible bias of magnitude `g(R)`, where
`g` is some explicit function from a 1-dimensional convex-analysis calculation.

**Why feasible.** This is a finite-d (1d!) projection problem. No asymptotics, no online
machinery. You set up the KL functional `D(P_true || P_β)` where `P_β =
softmax(β · ℓ*)`, take the argmin in β, and compare to whatever "true" β you've decided
to compare against. The off-axis residual `R` is the thing that creates the gap.

**Catch.** You have to *define* what the comparison "true β" is — there's no canonical
answer once R > 0. Candidates: leading singular vector of the centered logit matrix,
log-prob-weighted projection, etc. Different definitions give different bounds. This is
not a deal-breaker but is something to nail down before writing.

**Connects to #1 cleanly.** The KL projection IS what the asymptotic MLE converges to
under standard misspecification theory. So this lemma would compute the *target* that the
filter's β̂ would converge to in the well-specified-with-misspecification sense — and any
gap between β̂_∞ and the KL projection is then attributable to the streaming/closed-loop
dynamics, not to misspecification per se.

## 4. Filter-as-natural-gradient lemma

**What it would deliver.** Show that the Laplace update for `log β` is exactly the
online natural-gradient step on the cross-entropy loss `−log softmax(β · ℓ*_t)[x_t]`,
with step size `1/J_t` (the accumulated Fisher information). This is a one-paragraph
calculation; the score and Fisher are already the right objects.

**Why feasible.** It's a calculation, not a discovery. You write down the natural-
gradient update (Riemannian gradient under the Fisher metric), specialize to log β,
compare to the Laplace Newton step, see they're identical.

**What it buys you.** Once you've matched the filter to a known online-learning
algorithm, you can cite the natural-gradient consistency results from the
literature for the well-specified case (β̂_t → β_true at the parametric rate). It
*does not* address misspecification — but it justifies the design decision and lets
you point at theory rather than re-deriving it.

**Catch.** Small claim. Doesn't say anything about the project's headline puzzle
(confusion regime). Use as supporting infrastructure, not as the centerpiece.

---

## 5. Expert-advice certificate (distribution-free; IMPLEMENTED)

**Status.** Unlike anchors 1–4 this one is implemented (`grid_mixture_update_batched`,
`grid_mixture_level_set` in `src/decoding_decoding/counter_decode.py`) with the
experiments pre-registered in `paper/grid_certificate_prereg.md`. It is the
finite-sample, per-sequence twin of anchor #3 (whose KL projection is its population
limit) and it subsumes anchor #4 (the natural-gradient lemma describes the *proper*
point-estimate filter; this anchor explains why the *improper* mixture is the one that
carries guarantees).

**What it delivers.** Treat each grid value β_g as a fixed expert forecasting
`P_g(x_t) = softmax(β_g · ℓ_t)[x_t]` on the raw logits. The Bayes mixture over experts
is Vovk's Aggregating Algorithm for log loss (1-mixable), and the following hold for
EVERY token sequence — no assumption that tokens were sampled from any `P_β`:

1. *(Regret / mixability)* The mixture's cumulative log loss satisfies
   `mix_loss_n = −ln Σ_g π₀[g] e^{−L_n[g]} ≤ L_n[g] + ln(1/π₀[g])` for every g,
   where `L_n[g] = Σ_t [lse(β_g ℓ_t) − β_g ℓ_t[x_t]]`. Proof: the mixture's per-step
   log marginal telescopes into the log of the prior-weighted likelihood; drop all
   but term g. Uniform prior ⇒ regret vs the best grid expert ≤ ln G. No boundedness
   assumption, no vocabulary dependence.
2. *(MAP = follow-the-leader)* With uniform prior, the grid posterior mode is exactly
   `argmin_g L_n[g]` — the in-hindsight best-fit (effective) inverse temperature on
   the grid, with zero slack. (Do NOT use the posterior mean here: the regret bound
   controls the mixture's loss, and Jensen runs the wrong way — a bimodal posterior
   over {0.1, 10} forecasts fine while the mean β ≈ 5 is terrible.)
3. *(Level-set certificate)* `L_n(β)` is convex with `L_n''(β) = Σ_t Var_{q_{β,t}}(ℓ_t)`,
   so `{β : L_n(β) ≤ min L_n + c}` is an interval; its width is a data-dependent,
   distribution-free error bar for the effective temperature, computable directly
   from the stored loss profile. It widens exactly when peaked logits make β
   unidentifiable (the honest regime statement); note regret and estimation precision
   degrade in *opposite* directions in the realized Fisher mass `Λ_n`.
4. *(Drift, forecasting only)* `switch_rate` α > 0 is Fixed-Share (Herbster & Warmuth
   1998) = exact Bayes under a switching prior: regret ≈ m·ln(Gn/m) vs the best
   m-segment β sequence. The estimation reading (2)–(3) does NOT survive segmentation;
   claim tracking only.

**Why the mixture and not the Laplace filter.** The per-token loss in β has
exp-concavity constant ~`e^{−β·(logit gap)}`, so proper point-estimate updates (the
Laplace/ONS-style filter) cannot carry uniform per-sequence guarantees even in 1-d —
this is the Foster et al. 2018 "Importance of Being Improper" obstruction
(arXiv:1803.09349); the mixture is improper and sidesteps it. The Laplace filter stays
as the fast approximation; H4 of the prereg checks empirically whether its estimates
fall inside the certified bar.

**Catch (target honesty).** The guarantee is relative to the in-hindsight best-fit β,
NOT the dial the counterparty set. Under misspecification (top-k, blacklist, GIGO
closed loop) the former is the natural definition of effective temperature — and in the
confusion regime it certifies that β̂ ≈ tiny is a property of the emitted sequence, not
filter failure. The paper must say plainly: we do not certify recovery of the dial
setting. Also: the regret bounds need fixed experts — they do NOT apply to
`discrete_grid_update_batched`, whose bootstrap rescaling `ℓ* = ℓ/β̂` makes the experts
state-dependent.

**Citations.** Cesa-Bianchi & Lugosi, *Prediction, Learning, and Games* ch. 3 & 9
(mixability; continuous-prior mixture bound `≈ ½ ln Λ_n + const` vs the best β in a
bounded interval — note Shtarkov/NML/Xie–Barron do not apply directly because of the
per-round contexts ℓ_t); Herbster & Warmuth 1998; Foster, Kale, Luo, Mohri, Sridharan
2018.

---

## Suggested priority

If aiming for *one* theory anchor: do **#1** (closed-loop fixed-point) backed by **#2**
(toy model as sanity check). #1 directly addresses the project's headline empirical
finding. #2 makes #1's claims concrete and falsifiable in a fully tractable setting.

#3 is a clean independent lemma you can write up as a one-page side result if #1 turns
out to be intractable on the full system — it gives you *something* to point at that's
mathematically rigorous, even if it's not the whole story.

#4 is a half-page infrastructure note. Worth writing for completeness if any of the
others reference it.

#5 is implemented and pre-registered; it slots in as the rigorous "what the filter
provably does" infrastructure lemma (per-sequence certificate), paired with #3's
"what it converges to" (population target). It does not address #1's headline puzzle
and should not displace it.

## What was dropped

The adaptive-conformal / online-pinball framing from the earlier Plan is set aside as a
theorem target — coverage guarantees in this setting would require an
exchangeability-like property that isn't obvious in the LLM-sampling setup. Keep as
a one-sentence interpretive remark if it helps situate the work, but don't promise a
theorem.
