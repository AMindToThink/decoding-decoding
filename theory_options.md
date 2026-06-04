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

## Suggested priority

If aiming for *one* theory anchor: do **#1** (closed-loop fixed-point) backed by **#2**
(toy model as sanity check). #1 directly addresses the project's headline empirical
finding. #2 makes #1's claims concrete and falsifiable in a fully tractable setting.

#3 is a clean independent lemma you can write up as a one-page side result if #1 turns
out to be intractable on the full system — it gives you *something* to point at that's
mathematically rigorous, even if it's not the whole story.

#4 is a half-page infrastructure note. Worth writing for completeness if any of the
others reference it.

## What was dropped

The adaptive-conformal / online-pinball framing from the earlier Plan is set aside as a
theorem target — coverage guarantees in this setting would require an
exchangeability-like property that isn't obvious in the LLM-sampling setup. Keep as
a one-sentence interpretive remark if it helps situate the work, but don't promise a
theorem.
