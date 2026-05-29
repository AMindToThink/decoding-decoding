# Math audit — full-sequence structured-twist spec

Files audited (all in `/home/cs29824/matthew/decoding-decoding/full-sequence/files/`):
`00_README.md`, `01_decoding_target.md`, `02_structured_twist_method.md`, `03_pseudocode.md`.

Convention used throughout this audit: `x_<t = x_{1:t-1}` (tokens strictly before position t),
`x_{1:t}` includes `x_t`, `x_{≤t} = x_{1:t}`. `β := 1/T`.

---

## 1. Ratio claim  Q_pt(x)/Q_g(x) = Z / Π_t Z_t(x_<t)   — 01:25

Derive. With β = 1/T:

- Q_pt(x) = Π_t P(x_t|x_<t)^β / Z_t(x_<t),  Z_t(x_<t) = Σ_v P(v|x_<t)^β.
- Q_g(x)  = P(x_{1:N})^β / Z = (Π_t P(x_t|x_<t))^β / Z = Π_t P(x_t|x_<t)^β / Z.

Numerators are literally the same product Π_t P(x_t|x_<t)^β (01:23 is correct: (Π)^β = Π(·^β)).
Ratio:

  Q_pt/Q_g = [Π_t P^β / Π_t Z_t] / [Π_t P^β / Z] = Z / Π_t Z_t(x_<t).

**CORRECT, exactly.** Both Q_pt and Q_g are normalized distributions over V^N, so the ratio is a
genuine likelihood ratio (it integrates the right way: E_{Q_g}[Q_pt/Q_g] = 1, i.e.
E_{Q_g}[Z/Π_t Z_t] = 1 ⇒ Z = E_{Q_g}[Π_t Z_t]·... — consistent). The T=1 collapse claim
(01:27, "every Z_t = 1 and Z = 1") is **CORRECT** since P is locally normalized: at β=1,
Z_t = Σ_v P(v|x_<t) = 1 and Z = Σ_y P(y) = 1.

One nit (IMPRECISE, not wrong): 01:27 says strings through "high-Z_t prefixes are systematically
reweighted." Direction: Q_pt/Q_g = Z/Π Z_t, so large Π Z_t ⇒ Q_pt UNDER-weights that string
relative to Q_g. Fine, but the doc never states the sign; a reader could get it backwards.

---

## 2. Future partition function  Q_g(x_t|x_<t) ∝ P(x_t|x_<t)^β · F(x_{1:t})   — 01:33

Re-derive by marginalizing Q_g. The conditional is

  Q_g(x_t | x_<t) = Q_g(x_{1:t}) / Q_g(x_<t)            (def. of conditional from the joint marginals)

where the marginals of Q_g over a prefix are

  Q_g(x_{1:t}) = Σ_{x_{t+1:N}} Q_g(x_{1:N})
             = (1/Z) Σ_{x_{t+1:N}} Π_{s=1}^N P(x_s|x_<s)^β
             = (1/Z) [Π_{s=1}^t P(x_s|x_<s)^β] · [Σ_{x_{t+1:N}} Π_{s=t+1}^N P(x_s|x_<s)^β].

Define F(x_{1:t}) := Σ_{x_{t+1:N}} Π_{s>t} P(x_s|x_<s)^β. Then

  Q_g(x_{1:t}) = (1/Z) · P(x_<t)^β · P(x_t|x_<t)^β · F(x_{1:t}),
  Q_g(x_<t)    = (1/Z) · P(x_<t)^β · F(x_{1:t-1}),

where P(x_<t)^β := Π_{s=1}^{t-1} P(x_s|x_<s)^β. Dividing:

  Q_g(x_t|x_<t) = P(x_t|x_<t)^β · F(x_{1:t}) / F(x_{1:t-1}).

Since F(x_{1:t-1}) does not depend on x_t, it is a constant of the per-step normalization, so

  Q_g(x_t|x_<t) ∝ P(x_t|x_<t)^β · F(x_{1:t}).   ✓

**CORRECT.** All four checks the task asked for pass:
- The prefix factor P(x_<t)^β **cancels** exactly between numerator and denominator. ✓
- Z **cancels** too (it appears in both Q_g(x_{1:t}) and Q_g(x_<t)). ✓ The doc writes "∝"
  so the unmentioned 1/Z and 1/F(x_{1:t-1}) constants are correctly swept into proportionality.
  The exact per-step normalizer is Σ_{x_t} P(x_t|x_<t)^β F(x_{1:t}) = F(x_{1:t-1}) (this is the
  backward recursion, item 3), confirming consistency. ✓
- F(x_{1:t}) is a function of x_{1:t} **INCLUDING x_t**: the summand Π_{s>t} P(x_s|x_<s)^β has
  s>t terms whose conditioning sets x_<s all include x_t (e.g. P(x_{t+1}|x_{≤t}) depends on x_t). ✓
  This is exactly why the twist must be candidate-conditioned (item 4).
- F definition matches 01:33 verbatim. ✓

Sanity at t=N: empty sum convention gives F(x_{1:N}) = (empty product) = 1, matching 01:37. ✓
Sanity vs Z: F(x_{1:0}) = F(∅) = Σ_{x_{1:N}} Π P^β = Z. So Z = F(∅); the global normalizer is
the value-to-go of the empty prefix. The doc never states this but it is the clean closure and
is consistent.

Definitional note (correct, and matches the literature): "future partition function" F sums over
**futures only** (Π_{s>t}). It excludes the current tempered conditional P(x_t|x_<t)^β, which sits
outside F in the conditional. This is EXACTLY the convention of Twisted-SMC's optimal twist
ψ*_t(s_{1:t}) ∝ Σ_{s_{t+1:T}} p0(s_{t+1:T}|s_{1:t}) φ(s_{1:T}) (Zhao et al. 2024, Prop 3.2 / Eq. 12),
which is future-only and excludes the step-t factor — VERIFIED against the paper. With φ≡1 and base
measure P^β, F = ψ*_t. No convention mismatch. ✓

---

## 3. Backward recursion + "soft-Bellman" framing   — 01:37, 01:39

Recursion: F(x_{1:t}) = Σ_{x_{t+1}} P(x_{t+1}|x_{≤t})^β · F(x_{1:t+1}),  F(x_{1:N})=1.

Derive from the definition:

  F(x_{1:t}) = Σ_{x_{t+1:N}} Π_{s>t} P(x_s|x_<s)^β
            = Σ_{x_{t+1}} P(x_{t+1}|x_{≤t})^β  Σ_{x_{t+2:N}} Π_{s>t+1} P(x_s|x_<s)^β
            = Σ_{x_{t+1}} P(x_{t+1}|x_{≤t})^β · F(x_{1:t+1}).   ✓

**CORRECT.** Index on the conditioning set is right: the first emitted future token is x_{t+1},
conditioned on x_{≤t} = x_{1:t}, and F is then evaluated at the extended prefix x_{1:t+1}.
Base case F(x_{1:N})=1 correct (item 2). Pulling out only the first factor is valid because the
remaining product Π_{s>t+1} does not reference x_{t+1}'s *value* beyond it being in the conditioning
prefix — which it is, correctly carried into F(x_{1:t+1}). ✓

**"Soft-Bellman backup with reward β·logP and value logF" — IMPRECISE / loosely true, needs care.**

Take logs of the recursion:

  log F(x_{1:t}) = log Σ_{x_{t+1}} exp[ β log P(x_{t+1}|x_{≤t}) + log F(x_{1:t+1}) ].

This is EXACTLY a log-sum-exp (soft-max) backup: with "reward" r = β log P(x_{t+1}|x_{≤t}) and
"value-to-go" V = log F(x_{1:t+1}), we get V(s) = logΣ_a exp[r(s,a) + V(s')]. This is the
**soft / entropy-regularized Bellman optimality operator at inverse-temperature 1** (the
log-sum-exp is the standard "soft-max" / "mellowmax-at-β=1" backup of MaxEnt RL; Haarnoja et al.
2017 "Reinforcement Learning with Deep Energy-Based Policies", Ziebart 2010). So the *log-domain*
statement is precise: log F is a soft value-to-go and the operator is the soft-Bellman operator.

Where the doc is loose:
(a) **01:39 attaches the soft-Bellman label to the LINEAR recursion** ("F = Σ P^β F ... which is a
    soft-Bellman backup"). The linear recursion is an ORDINARY (un-discounted, transition-weighted)
    linear Bellman/Feynman–Kac sum, not a soft one. The "soft" (log-sum-exp) structure only appears
    after taking logs. The doc does say "value log F" in the same breath, so the intent is right, but
    as written it labels the wrong line. Tighten: "the recursion is linear in F (a Feynman–Kac /
    Chapman–Kolmogorov backward sum); equivalently, in the log domain it is the soft-Bellman backup
    log F = logsumexp_a[β log P + log F']." This is the standard path-integral / linearly-solvable
    MDP (Todorov 2009 / Kappen path-integral control) correspondence: the partition/desirability
    function is linear-recursive, its log is the soft value. Calling F itself the "desirability
    function z" and log F the soft value would be the most precise framing.
(b) The "reward" has NO action-independent state reward and NO entropy bonus term written out — but
    that's fine: in MaxEnt RL the entropy bonus is exactly what the logsumexp encodes. The
    discount γ is implicitly 1 and horizon is finite (N). Fine for fixed N.
(c) There is no policy being optimized here — F is a fixed backward sum (a *prediction* /
    Feynman–Kac object, log-partition of a fixed twisted measure), not the value of an optimized
    policy. "Soft value-to-go" is apt; "backup" is apt; just don't imply an argmax/control problem.

Net: the soft-Bellman correspondence is REAL and standard, but 01:39 pins the label on the linear
line. Mark **IMPRECISE (fix the line it's attached to / state log-domain explicitly).**

---

## 4. Candidate-conditioning argument   — 02:29-33

Claim A (prefix-only factor cancels): if twist = g(x_<t) depending only on committed prefix, then
Q_g(x_t|x_<t) ∝ P(x_t|x_<t)^β g(x_<t) ∝ P(x_t|x_<t)^β (since g(x_<t) is constant in x_t and divides
out of the step-t normalizer Σ_v P(v|x_<t)^β g(x_<t) = g(x_<t) Σ_v P(v|x_<t)^β).

**CORRECT.** g(x_<t) factors out of both numerator and the sum, cancels, leaving per-token tempering.
02:31 is exactly right and is the key implementation warning.

Claim B (converse — must evaluate twist at θ_t(v), latent AFTER hypothetically emitting candidate v):
The exact conditional is ∝ P(x_t=v|x_<t)^β F(x_{1:t-1}, v). F(x_{1:t-1}, v) depends on v (item 2),
i.e. it is the value-to-go of the prefix that ALREADY INCLUDES v. So the correct surrogate is
F̃ evaluated at the latent that reflects a prefix including v, i.e. θ after observing v. The doc's
θ_t(v) = u(θ_{t-1}, o_t(v)) (02:33) with observation o_t(v) = −log P(v|x_<t) is the right structure:
feed candidate v's surprise into the filter, evaluate F̃ at the resulting latent.

**CORRECT in structure.** The cross-candidate spread of F̃(θ after v) is exactly what reproduces
the v-dependence of F(x_{1:t-1}, v); a v-independent twist cannot, hence reduces to per-token. ✓

Caveat (unstated assumption, flag): this requires the latent θ to be a sufficient statistic such
that F(x_{1:t}) ≈ F̃(θ_t). The whole correction's fidelity rests on (i) θ capturing the
v-dependence and (ii) the observation o_t(v) = −logP(v|x_<t) being the right (sufficient) summary
of how v changes the future value. If two candidates with equal surprise lead to very different
future masses, a surprise-only scalar filter cannot separate them — this is the "imprint may be
>1-dimensional" failure mode the doc itself flags (02:86, residual floor). Math is fine; sufficiency
is an assumption, correctly acknowledged elsewhere.

---

## 5. CRITICAL — off-by-one / index consistency between 02 and 03

Definitions in play:
- 02:69 residual:  r_t = log F̃(θ_t) − log Σ_v P(v|x_{1:t})^β F̃(θ_{t+1}(v)),  anchor log F̃(θ_N)=0.
- 01:37 true recursion: F(x_{1:t}) = Σ_{x_{t+1}} P(x_{t+1}|x_{≤t})^β F(x_{1:t+1}).
- 02:67 text says it substitutes the ansatz into "F(x_{1:t}) = Σ_v P(v|x_{1:t})^β F(x_{1:t},v)".

### Finding 5a — conditioning-index typo in the conditional probability (02:67, 02:69). MINOR but real.

In 02:67 and 02:69 the summand probability is written **P(v | x_{1:t})^β**. Compare the true
recursion 01:37: the next token x_{t+1}=v is conditioned on x_{≤t} = x_{1:t}. So P(v|x_{1:t}) IS
correct for the token being summed (v = x_{t+1} conditioned on the length-t prefix). ✓ Good —
02:67/69 conditioning index is actually right. NO error here. (I checked because "P(v|x_{1:t})"
looks suspicious next to "F̃(θ_t)" on the LHS, but the v in the sum is the (t+1)-th token, so
conditioning on x_{1:t} is correct.)

### Finding 5b — what does "θ_t" mean? (the real ambiguity). NOTATION DRIFT, resolvable, must be stated.

Across the two docs θ_t is used with TWO meanings, and they are reconciled only if you read 03's
filter carefully:

- In 02:33, θ_t(v) = u(θ_{t-1}, o_t(v)) = "latent AFTER hypothetically emitting candidate v at step t."
  ⇒ here θ_t = latent reflecting a prefix of length t (INCLUDING x_t). Subscript t = "after x_t."
- In 02:69, r_t = log F̃(θ_t) − log Σ_v P(v|x_{1:t})^β F̃(θ_{t+1}(v)). For this to be the image of
  the true recursion F(x_{1:t}) = Σ_v P(v|x_{≤t})^β F(x_{1:t},v), we need:
    F̃(θ_t)        ↔ F(x_{1:t})        ⇒ θ_t reflects prefix length t (INCLUDING x_t),
    F̃(θ_{t+1}(v)) ↔ F(x_{1:t}, v=x_{t+1}) ⇒ θ_{t+1}(v) reflects prefix length t+1 (INCLUDING v).
  Self-consistent with the 02:33 meaning ("subscript = after that token"). ✓

So **within 02, θ_t = latent after committing x_t.** Consistent.

### Finding 5c — does 03 fit B (03:145-167) enforce the SAME recursion? OFF-BY-ONE IN THE LOOP. Real, load-bearing for correctness of the fit.

Walk 03 fit B literally. At top of the `for t in 1..len(x)` loop body (03:151-163):
- `state` entering iteration t is the latent committed through x_{<t} (it was last advanced at the
  END of iteration t−1 by `commit(state, −log p[x_t])` on line 163, using the previous iteration's
  p and x_t). So at line 153 `state` reflects prefix x_{<t} = length t−1. Call it θ_{t-1} in the
  02 convention (latent after x_{t-1}).
- `pred = log_twist(state)`  ⇒ pred = log F̃(θ_{t-1}).   **(reflects x_<t, i.e. BEFORE x_t)**
- target = logsumexp_v [ β log p[v] + stopgrad log F̃(peek(state, −log p[v])) ]
        = log Σ_v P(v|x_<t)^β F̃(θ after appending v to x_<t).
  Here p = base_conditional(x_<t) (line 151), so P(v|x_<t), and peek appends v ⇒ latent reflects
  x_<t · v = a length-t prefix whose last token is v. In 02 convention that is θ_t(v).

So 03's per-step equation is:

  log F̃(θ_{t-1}) = log Σ_v P(v|x_<t)^β F̃(θ_t(v)).            (03 fit B, as written)

Map to the true recursion. The true recursion at prefix length t−1 is:

  F(x_{1:t-1}) = Σ_v P(v|x_{1:t-1})^β F(x_{1:t-1}, v),   v = x_t.

Image under the ansatz (θ_{t-1} ↔ x_{1:t-1}, θ_t(v) ↔ x_{1:t-1}·v):

  log F̃(θ_{t-1}) = log Σ_v P(v|x_<t)^β F̃(θ_t(v)).            (correct target recursion)

**These MATCH.** 03 fit B as written IS the correct recursion — but indexed one step earlier than
02:69. Precisely:

- 02:69 writes the residual as r_t with LHS F̃(θ_t) and the sum producing θ_{t+1}(v).
- 03 fit B's iteration t writes the residual with LHS F̃(θ_{t-1}) and the sum producing θ_t(v).

They are the **same family of equations shifted by one in the loop counter** (02's "t" = 03's "t−1").
This is **notation drift, NOT a substantive off-by-one** in the fit's recursion: the consistency
condition being enforced (F̃(prefix) = logsumexp of P^β F̃(prefix·v) over v) is identical. ✓
Both are also consistent with the SAMPLING loop (03 §3): sampling evaluates F̃ at `peek(state, o_v)`
= latent after appending v to the current committed prefix x_<t (03:89) — exactly θ_t(v) in 02 /
the same object summed in fit B. So **sampling and fit B agree on which latent the twist is read at
(post-v).** ✓

### Finding 5d — the GENUINE off-by-one: the terminal anchor is applied to the WRONG state in 03 fit B. LOAD-BEARING.

02 anchor: log F̃(θ_N) = 0, i.e. the latent reflecting the FULL length-N prefix x_{1:N} (after x_N).
That matches F(x_{1:N}) = 1 (item 7).

03 fit B anchor (03:166): `loss += anchor_weight * (log_twist(state))**2` applied to `state` AFTER
the loop. But the loop `for t in 1..len(x)` updates `state` via `commit` on line 163 INSIDE each
iteration, including the last one. Trace the final iteration t = L (L = len(x)):
- line 153 pred uses state = θ_{L-1};
- line 163 commits x_L ⇒ state becomes θ_L (reflects full prefix x_{1:L}).
After the loop, `state` = θ_L = θ_N (if L = N). So the anchor `log_twist(state)` = log F̃(θ_N).
**This is actually correct** — the post-loop state is θ_N. ✓ Good.

BUT two real problems remain:

(1) **The final-step residual is never anchored into a valid recursion / there is a missing-or-extra
    term at the boundary.** In iteration t=L, the target is
    logsumexp_v[β log p[v] + log F̃(θ_L(v))] where p = P(·|x_{<L}) — this bootstraps F̃(θ_{L-1})
    off F̃(θ_L(v)). But F̃(θ_L(v)) for the *hypothetical* v is a length-L latent, and the only
    length-L latent that is anchored to 0 is θ_N along the ACTUAL path (v = x_N). The hypothetical
    off-path length-N latents θ_N(v≠x_N) are NOT anchored and are free parameters bootstrapped
    against nothing terminal. For fixed N this is the same situation as any finite-horizon TD with a
    terminal value: you need V(terminal) = 0 for ALL terminal states, not just the one on the
    sampled path. With log F̃(θ) = w·φ(θ) a single global anchor at θ_N (one point) does NOT force
    F̃(·)=1 (log=0) at every length-N latent. ⇒ The scale/level is pinned only at one latent value;
    elsewhere F̃ at the horizon is whatever the linear map says. See item 7 — this is the real
    content of the "only relative values matter" caveat and it interacts badly with a SINGLE anchor.

(2) **The loop runs `for t in 1..len(x)` and forms a bootstrap target at the LAST position t=L
    that needs F̃ at length-L (=N) latents, then ALSO anchors length-N. For a fixed-N target the
    last residual should be the one whose target is the anchor F(x_{1:N})=1, i.e. the recursion
    F(x_{1:N-1}) = Σ_v P(v|x_{<N})^β F(x_{1:N-1},v) with F̃(θ_N(v)) forced to 0.** As written the
    code does compute that residual (iteration t=N−1 in 03's counting bootstraps off θ_{N-1}? — no:
    iteration t produces target over θ_t(v); the iteration whose target lands on length-N latents is
    t=N). The anchor and the t=N bootstrap target then BOTH constrain length-N latents, which is
    redundant for the on-path point and silent off-path. This is messy but not strictly wrong; it is
    an **unstated modeling choice** (soft anchor via penalty weight vs hard terminal condition).

**Verdict item 5:**
- 5a: no error (conditioning indices in 02 are correct).
- 5b/5c: **notation drift only** — "θ_t" means "after x_t" in 02; 03's loop counter is shifted by
  one but enforces the identical recursion; sampling and fit B agree on reading the twist post-v.
  No substantive off-by-one in the recursion itself. State the convention explicitly to remove
  the trap.
- 5d: **genuine boundary defect** — a single-point terminal anchor does not enforce F̃=1 at all
  length-N latents that the t=N bootstrap target references; the horizon residual is
  under-constrained. This is load-bearing for whether the fit is well-posed (see item 7).

Corrected, consistent indexing (recommend adopting in BOTH docs):
  Let θ_k denote the latent after committing x_k (θ_0 = prior, reflects empty/prompt).
  Residual at step k (for k = 0 .. N−1):
    r_k = log F̃(θ_k) − logsumexp_v [ β log P(v|x_{1:k}) + sg log F̃(θ_{k+1}(v)) ],
    θ_{k+1}(v) = u(θ_k, −log P(v|x_{1:k})).
  Terminal: enforce log F̃(θ) = 0 for every latent reachable at length N (in practice: anchor on
  many sampled length-N latents, or fix the readout so φ(θ_N)=0 by construction), not just one.

---

## 6. Soft-Bellman residual fit: exact inner Σ_v ⇒ no double-sampling bias   — 02:71, 03:155-160

Claim: the inner Σ_v is computed EXACTLY from full-vocab logits (an exact expectation), not sampled,
so the classic residual-gradient double-sampling bias does NOT arise.

Background (Baird 1995 residual gradients; Sutton & Barto §11.5): the double-sampling problem occurs
when you minimize the squared Bellman residual E[(V(s) − E_{s'}[r+γV(s')])²] and the inner
expectation over the next state s' is INSIDE the square. The gradient of (E_{s'}[·])² needs the
product of TWO independent samples of s' to be unbiased; with one trajectory sample you get
E[(sample)²] ≠ (E[sample])², biasing the gradient. It only bites when the transition s→s' is
stochastic AND you estimate the inner expectation by sampling.

Here the "next state" is θ_{t+1}(v) and the "expectation" is the explicit, deterministic sum
Σ_v P(v|x_<t)^β F̃(θ_{t+1}(v)) over the FULL vocabulary, evaluated from one forward pass of logits
(03:158-160 loops all v; 02:71 "inner sum uses the model's logits from one forward pass"). Because
the inner sum is computed in closed form (not Monte-Carlo over v), there is **no second sampling
distribution to be biased by**: the target is a deterministic function of (state, logits), so
(target)² and its gradient are exact. **CONFIRMED CORRECT — double-sampling bias does not arise.**
This is a genuine advantage and the doc is right to claim it (implicitly). Worth stating explicitly
in the doc; it currently only says "exact expectation" indirectly.

Caveat the doc DOES note and is right about: **semi-gradient / stop-gradient bias remains.** 02:71
and 03:159 use `stop_grad` on the bootstrap target ("semi-gradient ... exactly as in soft
Q-learning"). Semi-gradient TD does NOT follow the gradient of any fixed objective (Sutton & Barto
§9.3, §11; the "deadly triad" / it is not a true gradient method) — the stop-gradient drops the
∂target/∂w term. Consequences correctly implied: convergence is to a fixed point of the projected
soft-Bellman operator, not a minimizer of E[r²]; with a linear-in-φ value class and on-policy-ish
data it typically converges, but off-distribution prefixes (the doc's own caveat, 02:61) can
destabilize it. So: **double-sampling bias: absent (correct claim).** **Semi-gradient
non-gradient bias: present (doc acknowledges via "semi-gradient," could be more explicit that the
fixed point ≠ argmin E[r²]).** If one instead used the FULL residual gradient (no stop-grad), then
because the inner sum is exact there would be NO double-sampling penalty for doing so — i.e. here
the residual-gradient method is actually unbiased and available, a point in the method's favor the
doc under-sells. Mark **CORRECT, with an upgrade opportunity (full residual gradient is unbiased
here; could be mentioned).**

Subsampling note (03:171 "subsample v for large |V|"): subsampling the inner sum REINTRODUCES the
double-sampling problem (inner expectation now Monte-Carlo). "Keeping the chosen token" does not fix
the bias; an importance-weighted / unbiased-estimator-inside-log caveat is missing. log(Σ over a
subsample) is also a biased estimator of log(Σ over V) (Jensen: E[log Σ̂] ≤ log E[Σ̂]). **Flag:
03:171 subsampling silently breaks the "exact expectation" property that item 6 relies on.**

---

## 7. Terminal anchor log F̃(θ_N)=0 vs F(x_{1:N})=1   — 02:69, 03:166, vs 01:37

F(x_{1:N}) = 1 (item 2/3) ⇒ log F(x_{1:N}) = 0 ⇒ anchoring log F̃(θ_N) = 0 is the right boundary
VALUE. **Consistent in value.** ✓

Does one anchor fix the scale ambiguity? The doc's premise (02:27, "only relative values matter,"
F̃ identified up to a per-step multiplicative constant) is itself worth checking:

- In SAMPLING (03 §3), at each step the twist enters ∝ P^β F̃(θ_t(v)) and is renormalized over v.
  A multiplicative constant c_t common to all v at step t DOES cancel. So for sampling, only the
  cross-candidate SHAPE matters; an arbitrary per-step level is irrelevant. ✓ (02:27 correct.)
- But the FIT (item 5/6) couples levels ACROSS steps: log F̃(θ_t) is regressed onto
  logsumexp_v[β logP + log F̃(θ_{t+1}(v))]. A per-step-arbitrary additive constant in log F̃ is
  NOT free here — the recursion ties level at t to level at t+1 (an additive constant a at every
  state would give r = a − logsumexp[...+a] = a − (a + logsumexp[...]) = −logsumexp[...]+0... wait:
  if log F̃ → log F̃ + a uniformly, LHS gains a, RHS gains a (logsumexp of (x+a) = logsumexp(x)+a),
  residual unchanged ⇒ a GLOBAL additive constant is a true gauge freedom of the residual). So the
  residual objective alone has a one-parameter gauge (overall additive constant on log F̃), and the
  **single anchor log F̃(θ_N)=0 is exactly the one constraint needed to fix that one global
  gauge.** In that sense, **anchoring at one terminal state DOES suffice to fix the scale
  ambiguity** the residual leaves — but ONLY the global additive one.

HOWEVER (the catch, ties to 5d): the recursion also references F̃ at length-N latents θ_N(v) for
hypothetical off-path v (the t=N−1 bootstrap target). The recursion forces those to satisfy
F̃(θ_{N-1}) = logsumexp_v[β logP + log F̃(θ_N(v))], but nothing forces each individual
log F̃(θ_N(v)) = 0. With log F̃ = w·φ(θ) linear, fixing log F̃ = 0 at the single point θ_N (on-path)
does NOT make log F̃ = 0 at other length-N latents unless φ is constant there. So the boundary
condition F(terminal)=1 (which holds for EVERY terminal in the exact problem) is only imposed at one
point. **The exact problem has |V|^{...} terminal conditions all equal to 0; the fit imposes one.**
For a sufficiently expressive φ this is a real approximation gap at the horizon; for a scalar φ(θ)=θ
it means the model can't even represent "0 at all terminals" unless w=0 there. **Mark: anchor VALUE
correct and the single global-gauge fix is correct; but "suffices to fix scale" is true only for the
global additive gauge — it does NOT enforce the full terminal boundary condition, which is a real
(acknowledged-adjacent but not stated) approximation error.** IMPRECISE/under-specified.

Minor: 03:166 implements the anchor as a SOFT penalty (anchor_weight · (log F̃(θ_N))²), so even the
global gauge is only softly fixed; the relative weight vs the per-step residuals is an unstated
hyperparameter that trades off boundary fidelity against interior consistency.

---

## 8. EOS / variable-length   — 01:49-50, 03:100

01:49-50: fixed-N keeps Q_g a clean distribution over V^N; EOS is "subtler." Correct and honest.

Is fixed-N F well-defined? Yes: F(x_{1:t}) = Σ_{x_{t+1:N}} Π_{s>t} P(x_s|x_<s)^β is a finite sum over
V^{N-t}, well-defined for every t ≤ N. ✓

**GENUINE INCONSISTENCY (the task flagged it; it is real). 03:100 `if x_t == EOS: break`.**
The sampling loop terminates early on EOS, but F was DEFINED for fixed length N (01:33, sum to N;
anchor F(x_{1:N})=1). Three concrete problems:

1. **Target mismatch.** If the sampler can stop at length L < N, the realized distribution is over
   variable-length strings, but F (and hence the twist) was constructed for the fixed-N target Q_g
   over V^N. The twist values F̃(θ_t) being applied were derived/fit under "exactly N more-or-fewer
   tokens follow with NO early stop." Sampling with EOS-break therefore samples a DIFFERENT target
   than the one the twist corrects for. The twist is, at best, the fixed-N twist applied to a
   variable-length process — a model mismatch, not just an approximation.

2. **What is EOS's tempered mass?** Under fixed-N, EOS is just another token in V and P(EOS|·)^β
   contributes to Z_t and to F like any token, and continuations after EOS are still summed to N. If
   instead EOS truly terminates, the correct object is a tempered distribution over a TREE with
   terminal probabilities (01:50 says exactly this), whose partition function is
   F_eos(x_{1:t}) = P(EOS|x_{≤t})^β · 1 + Σ_{v≠EOS} P(v|x_{≤t})^β F_eos(x_{1:t},v) with a different
   boundary (terminate-on-EOS) — NOT the fixed-N F. The doc never defines this object, yet the
   pseudocode samples under it. **Flag as an inconsistency: 03 samples a variable-length process
   while 01/02 only define/justify the fixed-N twist.**

3. **Fit B horizon.** 03 fit B loops `for t in 1..len(x)` over prefixes of NATURAL/model text whose
   lengths len(x) vary per example, and anchors at the per-example final state. So fit B implicitly
   treats EACH prefix's own end as "terminal" (anchor at len(x)), i.e. it fits a VARIABLE-horizon /
   EOS-style boundary — which is INCONSISTENT with 01/02's single fixed N and with the fixed-N F it
   claims to approximate. Either (a) the target is fixed-N and fit B must use length-N windows and
   only anchor true length-N states, or (b) the target is EOS-terminated and 01/02's F must be
   redefined with terminal probabilities. The docs mix the two. **This is the same root issue as
   5d and is the most consequential cross-doc inconsistency after the anchor problem.**

Verdict item 8: fixed-N F is well-defined (✓), but **03's EOS-break (03:100) and per-example anchor
in fit B (03:166) silently assume a variable-length/EOS-terminated target that 01/02 explicitly say
is "subtler" and do NOT define. Real inconsistency; must pick one regime.**

---

## 9. T → 0 limits   — 01:51, 02:85

β = 1/T → ∞. Claims:
- "log F becomes max-dominated (logΣexp collapses toward a max)": log F(x_{1:t}) =
  logΣ_{x_{t+1}} exp[β log P(x_{t+1}|x_{≤t}) + log F(x_{1:t+1})]. As β→∞ the β log P term dominates
  and logsumexp → max over the next token of [β logP + logF'] → β·(value of the most-probable
  continuation). So yes, F → (mass of the single max-prob continuation)^β; log F → β · max-path
  log-prob-to-go. **CORRECT** — softmax→max as inverse-temperature→∞ is standard.
- "value-to-go grows spiky / smooth approximation degrades": as β→∞, F as a function of the prefix
  becomes dominated by whether the prefix lies on a high-probability path; tiny changes in prefix
  (one token) can switch which continuation is the max ⇒ F̃ becomes a near-discontinuous function of
  θ, which a smooth scalar map cannot track. **CORRECT, well-motivated.**
- "the same regime where the local/global gap matters most": at T→0 both Q_pt and Q_g concentrate.
  Q_pt does greedy-per-step (argmax each conditional); Q_g does the global argmax path
  (max-probability sequence, i.e. exact MAP / Viterbi). These differ exactly by the label-bias /
  myopic-vs-global gap, which is maximal at T→0 (greedy ≠ MAP in general). **CORRECT.** Note Q_pt at
  T→0 → per-step greedy; Q_g at T→0 → MAP sequence; the gap is the classic greedy-vs-beam/Viterbi
  gap. The doc could state this concretely; as written it is correct but qualitative.

One sanity caveat (unstated, minor): the T→0 statements are about the IDEALIZED F; the method's
weakness there (02:85 "graceful degradation") is asserted, not derived. Fine for a spec.

Verdict item 9: **CORRECT.** All three T→0 claims hold under standard softmax→max asymptotics.

---

## Literature cross-reference (as requested)

- **Soft Bellman / MaxEnt RL, "reward β·logP, value logF":** The log-sum-exp backup
  V(s) = logΣ_a exp(r + V(s')) is the soft-Bellman (entropy-regularized) optimality operator at
  unit temperature (Ziebart 2010 MaxEnt IRL; Haarnoja et al. 2017, Soft Q-Learning; Levine 2018
  "RL and Control as Probabilistic Inference," eq. for the soft value/backward message). The
  doc's correspondence is STANDARD and CORRECT in the log domain. The exact-marginal F is the
  "backward message" / desirability function of linearly-solvable MDPs and path-integral control
  (Todorov 2009; Kappen 2005), where F is linear-recursive and log F is the soft value — confirming
  item 3's precise framing and the doc's looseness in labeling the LINEAR line "soft-Bellman."
- **Double sampling / residual gradient (Baird 1995; Sutton & Barto 2018 §11.5):** double-sampling
  bias requires a SAMPLED inner expectation over stochastic transitions. With an EXACT inner sum
  (full vocab), it does not arise — confirms item 6. Semi-gradient TD is not a true gradient method
  and converges to a projected-Bellman fixed point, not argmin E[r²] (Sutton & Barto §9.3-§9.4,
  §11) — confirms the residual semi-gradient caveat.
- **Twisted SMC (Zhao et al. 2024, arXiv 2404.17546) — VERIFIED against the paper (ar5iv):**
  Proposition 3.2 / Eq. (12) defines the optimal twist
    ψ*_t(s_{1:t}) ∝ Σ_{s_{t+1:T}} p0(s_{t+1:T} | s_{1:t}) φ(s_{1:T}),
  i.e. the expected future potential conditioned on the prefix s_{1:t}, a FUTURE-ONLY conditional
  partition function that **excludes the current step-t factor and marginalizes only over t+1:T.**
  This is STRUCTURALLY IDENTICAL to this doc's F(x_{1:t}) = Σ_{x_{t+1:N}} Π_{s>t} P(x_s|x_<s)^β
  (item 2): with the terminal potential set to φ ≡ 1 and the base model replaced by the
  tempered/unnormalized measure P^β, ψ*_t IS F. The future-only / excludes-step-t convention
  MATCHES this doc — so the "convention difference" I worried about does NOT arise; the doc's F and
  TSMC's ψ* use the same indexing. ✓
  Section 3.4 / Eqs. (19)-(20): with exponential terminal potential φ = exp(β r), the paper itself
  states twists correspond to Q-values and the twist recursion is a "soft Bellman recursion," with
  twists as value/critic approximations. This is the paper's OWN framing, and it confirms BOTH
  (a) the doc's soft-Bellman / soft-value-to-go correspondence (item 3) is standard and correct, and
  (b) the doc's claim (02:71) that this is "the same squared-residual objective Twisted SMC uses for
  its twist, specialized to a log-linear-in-latent value class" is ACCURATE. The squared-error twist
  loss being "analogous to soft Q-learning" is the paper's explicit statement (also crediting Lioutas
  et al. 2022).
  Net: the literature framing in 01/02 is corroborated almost verbatim by the cited paper. The one
  real refinement (item 3) stands: the SOFT-Bellman structure is in the LOG domain; F itself / ψ*
  itself obeys a LINEAR (Feynman–Kac) recursion, and the paper's "soft Bellman" label likewise
  applies to the log/exponential-potential form.

---

## Summary table

| # | Item | Verdict | Fix |
|---|------|---------|-----|
| 1 | Q_pt/Q_g = Z/ΠZ_t | CORRECT | (state reweighting sign) |
| 2 | Q_g(x_t|x_<t) ∝ P^β F, F def | CORRECT | note F=desirability, Z=F(∅) |
| 3 | backward recursion + soft-Bellman label | CORRECT (recursion); IMPRECISE (label on linear line) | move "soft-Bellman" to the log-domain line; call F desirability / log F soft value |
| 4 | candidate-conditioning + converse | CORRECT | state sufficiency assumption on o(v) |
| 5 | 02↔03 index consistency | NOTATION DRIFT (5b/5c, not a real off-by-one) + GENUINE boundary defect (5d) | adopt single θ_k convention; fix terminal anchor (5d/7/8) |
| 6 | exact inner Σ_v ⇒ no double-sampling | CORRECT | mention full-residual-gradient is unbiased here; flag 03:171 subsampling breaks it |
| 7 | anchor logF̃(θ_N)=0 vs F=1 | CORRECT value + global-gauge fix; IMPRECISE on "suffices" | one anchor fixes only the global additive gauge, not the full terminal BC |
| 8 | EOS / fixed-N | fixed-N F well-defined (CORRECT); 03 EOS-break + per-example anchor INCONSISTENT with 01/02 | pick fixed-N (length-N windows, anchor only true N) or define EOS-terminated F with terminal prob |
| 9 | T→0 limits | CORRECT | could state greedy(Q_pt)→ vs MAP(Q_g) gap concretely |
