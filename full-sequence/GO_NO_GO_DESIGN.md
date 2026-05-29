# Go/No-Go existence test — pre-registered design

**Purpose.** Decide, cheaply and before building the streaming sampler/fits, whether the
sequence-level temperature correction `F` has enough structure on real text — and whether the
proposed estimator can recover a statistically-significant fraction of it — to justify the full
pipeline. Recommended by the method-soundness review (`review/fs_review_method.md` §1, §7).

**Authoritative math** (from `files/01_decoding_target.md`, all verified correct in `review/fs_review_math.md`):
`Q_g(x_t|x_<t) ∝ P(x_t|x_<t)^{1/T} · F(x_{1:t})`, with
`F(x_{1:t}) = Σ_{x_{t+1:N}} Π_{s>t} P(x_s|x_<s)^{1/T}` and backward recursion
`F(x_{1:t}) = Σ_{x_{t+1}} P(x_{t+1}|x_{≤t})^{1/T} F(x_{1:t+1})`, `F(x_{1:N})=1`.

The per-step correction is the **cross-candidate** variation of `log F(x_{<t}, v)` over candidates `v`
(a v-independent factor cancels in renormalization — `review/fs_review_math.md` §4).

---

## The load-bearing subtlety: scalar-linear twist ≡ temperature change

The repo's Laplace filter (`src/decoding_decoding/counter_decode.py:91`) tracks `η̂ = log β`. Its
candidate-conditioned latent is
```
θ(v) = η̂_{t-1} + α·(ℓ_v − E_q)/J_t ,   J_t = J_{t-1} + α·Var_q   (J_t is v-independent)
```
which is **affine in the candidate logit `ℓ_v`** with a v-independent intercept. With the spec's
log-linear map `log F̃(θ)=w·θ`, the sampling weight is
```
p_v^{1/T} · exp(w·θ(v)) ∝ exp( (1/T + wα/J_t)·ℓ_v ),
```
i.e. **softmax at inverse-temperature `β_eff(t) = 1/T + wα/J_t`**. So the scalar-linear structured
twist is *exactly* a per-position temperature whose offset decays like `1/J_t`. It cannot represent
any part of `F` that is non-affine in `ℓ_v`.

**Consequence for the test:** raw "fraction of `F` explained" is inflated by temperature-collinearity.
A scrutiny-proof GO requires recovering a fraction of `F` **beyond the best global temperature retune**.

---

## What we compute (depth-h lookahead; exact at h=1)

For each evaluation position with committed prefix `x_<t`:
1. Base forward pass → logits `ℓ`, `log p = log_softmax(ℓ)`.
2. Candidate set = top-`K` tokens by tempered mass `p_v^{1/T}` (`K` s.t. covered mass ≥ 0.99, cap 64);
   report covered mass.
3. For each candidate `v`: ONE forward pass on `x_<t · v` → `log p(·|x_<t,v)` →
   **exact** `log F_1(x_<t,v) = logsumexp_w[ (1/T)·log p(w|x_<t,v) ]` (full-vocab leaf sum, no
   subsampling — avoids the biased-logsumexp bug `review/fs_review_math.md` §6 / `files/03:171`).
   Optionally h=2 by expanding each `v` over its own top-`K₂` (beam) to check horizon sensitivity.
   `log F_h(x_<t,v) = logsumexp_w[ (1/T)log p(w|x_<t,v) + log F_{h-1}(x_<t,v,w) ]`.
4. Candidate-conditioned filter latent `θ(v)` via a pure `peek` (no mutation needed — the existing
   `laplace_update` already returns a fresh state).

Let `c_v := log F_h(x_<t, v)` (the correction target) and `g^{pt}_v := (1/T)·log p_v` (per-token
tempered log-weight, i.e. `F≡1`). Define per-position distributions over the `K` candidates
(renormalized over the set; covered mass reported):
```
q_pt(v)    ∝ exp( g^{pt}_v )                         # F ≡ 1  (per-token tempering)
q_g(v)     ∝ exp( g^{pt}_v + c_v )                   # depth-h sequence-level target
q_bestT(v) ∝ exp( β_eff · ℓ_v )                      # best single global temperature (1 param, fit to q_g)
q_filt(v)  ∝ exp( g^{pt}_v + ŵ·θ(v) )                # scalar-linear Laplace twist (ŵ distilled from c_v)
q_rich(v)  ∝ exp( g^{pt}_v + ĉ_v )                   # ceiling: ĉ_v = flexible fit of c_v on θ (and on richer φ)
```

---

## Metrics (bootstrap CIs over prefixes)

- **D_total = E_t[ KL(q_g ‖ q_pt) ]** (nats): the *size* of the sequence-level correction. If ≈0,
  the whole direction is dead for this model/T (NO-GO regardless of estimator).
- **Affine decomposition** of `c_v`: per-position OLS `c_v = a + b·ℓ_v + r_v`;
  report `structured_frac = E_t[Var_v(r_v)] / E_t[Var_v(c_v)]` (non-temperature share of `F`).
- **Recovered fractions** of `D_total` (1 − E_t[KL(q_g‖·)]/D_total):
  - `frac_bestT` — global temperature retune (the boring ceiling).
  - `frac_filter` — the proposed scalar-linear estimator.
  - `frac_rich` — flexible 1-D-latent map (estimator-class ceiling).
- **Scrutiny-proof increment**: `frac_filter − frac_bestT` and `frac_rich − frac_bestT`
  (= fraction of `F` recovered **beyond** any global temperature). This is the number that must be
  significantly > 0.

## Pre-registered GO / NO-GO rule

**GO** iff **both**:
1. `D_total` bootstrap 95% CI lower bound > **1e-3 nats** (correction is real, not numerical dust), **and**
2. at least one of `{frac_filter − frac_bestT, frac_rich − frac_bestT}` has bootstrap 95% CI lower
   bound > **0** AND point estimate ≥ **0.05** (the estimator class recovers ≥5% of `F` *beyond*
   temperature retuning — a deliberately small but scrutiny-proof bar; "small fraction is fine" per
   user, but it must clear temperature-collinearity and be significant).

Rationale for the 5% / significance bar: the user is happy with a small fraction but it must (a) be
statistically distinguishable from zero (bootstrap CI) and (b) not be the trivial temperature-collinear
part (hence "beyond best global T"). If `frac_rich − frac_bestT` is significant but `frac_filter` is
not, the GO is conditional on **richer φ / multi-dim latent** (report that as the required design change,
do not silently proceed with the scalar-linear form).

**NO-GO** → stop and report; the honest finding ("sequence-level tempering ≈ per-token tempering at a
shifted temperature for this model/regime") is itself a clean result consistent with LHTS's null on
open-ended GPT-2 (`review/fs_review_cites.md` §1).

## Settings
- Models: `gpt2-large` first (matches LHTS/Twisted-SMC literature, ~50k vocab → cheap full-vocab
  leaf sum), then `Qwen/Qwen2.5-3B` (repo continuity). [User: both, GPT-2 first.]
- `T ∈ {0.5, 0.7, 0.9}` (0.9/0.7 = estimator-friendly mild regime; 0.5 = larger gap).
- Prefixes: WikiText-103 + WritingPrompts (reuse loaders) — natural text; plus model-generated
  per-token-sampled prefixes (the distribution the filter was calibrated on). ~300+ positions/condition.
- `h ∈ {1 (exact), 2 (beam)}`; report horizon sensitivity.
- Determinism: fixed seed; `model.eval()`; `torch.no_grad()`; `CUDA_VISIBLE_DEVICES=0`.
- Numbers-from-scripts: emit `summary.json` → `macros.tex`; no hand-typed numbers.

## Confounds explicitly guarded
- Temperature-collinearity (the subtlety above) → `frac − frac_bestT`.
- Biased subsampled logsumexp → full-vocab leaf sum only.
- Top-K truncation of the *candidate set* → report covered tempered mass; leaf sum is full-vocab.
- Off-distribution (filter calibrated on Q_pt) → evaluate on BOTH natural and model-sampled prefixes.
- Horizon truncation (h finite) → report h=1 vs h=2 agreement; note h=1 is exact for the one-step term.
