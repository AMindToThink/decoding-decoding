# Structured streaming twist: method

This method approximates the future partition function $F$ (see `01_decoding_target.md`) so that autoregressive sampling tracks the sequence-level tempered target $Q_{\mathrm g} \propto P^{1/T}/Z$ instead of the per-token approximation $Q_{\mathrm{pt}}$.

## Where the approximation lives

All approaches approximate the same exact conditional,

$$Q_{\mathrm{g}}(x_t \mid x_{<t}) \;\propto\; P(x_t \mid x_{<t})^{1/T}\, F(x_{1:t}),$$

and differ only in where the approximation is placed:

- **LHTS** — in model weights (fine-tune so the model's own conditionals approximate $Q_{\mathrm g}$).
- **Twisted SMC** — in a learned neural twist $\psi \approx F$, used as an SMC proposal.
- **Structured streaming twist (this method)** — in a streaming estimate of a low-dimensional latent $\theta_t$ governed by a few knobs (prior, evidence weight $\eta$, decay $\lambda$), plus a small log-linear map from the latent to the twist.

The ansatz is

$$F(x_{1:t}) \approx \tilde F(\theta_t),$$

which asserts the value-to-go factors through a low-dimensional sufficient statistic of the prefix. This is the imprint-is-low-dimensional hypothesis restated; its validity is empirical and is itself measured by the method (see "Residual floor" below).

## Defining the map $\tilde F$

Three facts make $\tilde F$ a small, fittable object rather than an arbitrary function.

**1. Only relative values matter.** $F(x_{1:t})$ enters $Q_{\mathrm g}$ inside a ratio — the step-$t$ normalizer is $\sum_v P(v\mid x_{<t})^{1/T} F(x_{1:t-1},v)$ — so an overall scale on $F$ at a step cancels. $\tilde F$ is identified only up to a per-step multiplicative constant; the object being fit is the *shape* of value-versus-latent, not its level.

**2. It must be candidate-conditioned.** A factor depending on the committed prefix $x_{<t}$ alone is constant across candidate next-tokens and cancels in the same renormalization:

$$Q_{\mathrm{g}}(x_t \mid x_{<t}) \;\propto\; P(x_t \mid x_{<t})^{1/T}\, g(x_{<t}) \;\propto\; P(x_t \mid x_{<t})^{1/T},$$

which is just per-token tempering. So the estimate must be evaluated at the latent one *would* hold after each candidate, $\theta_t(v) = u(\theta_{t-1}, o_t(v))$, with a candidate-dependent observation such as that candidate's surprise $o_t(v) = -\log P(v\mid x_{<t})$. The cross-candidate spread of $\tilde F$ is the entire correction. This is the single most important implementation point: a prefix-only twist does nothing.

**3. Start log-linear.** Parameterize

$$\log \tilde F(\theta) = w^{\top}\phi(\theta),$$

with features $\phi$ of the latent (for a scalar latent, $\phi(\theta)=\theta$ and $w$ is one coefficient). The per-step sampling rule is then

$$Q_{\mathrm{g}}(x_t\mid x_{<t}) \;\approx\; \frac{P(x_t\mid x_{<t})^{1/T}\,\exp\!\big(w^{\top}\phi(\theta_t(x_t))\big)}{\sum_v P(v\mid x_{<t})^{1/T}\,\exp\!\big(w^{\top}\phi(\theta_t(v))\big)}.$$

This is a value-function approximation whose features are the tracked latent.

## Two objects, two fits

The method has two parts with *different* supervision signals, both living on the target LLM and on text:

- **The filter** $u(\cdot)$ and its knobs (prior $\theta_0$, prior strength, evidence weight $\eta$, decay $\lambda$) — the latent dynamics. Supervised by the imprint trajectories. No $F$ involved. **The repo already has a working streaming filter; it is authoritative.** These docs do not design the filter — the knob names above are generic and may not match its actual parameterization. Where this description and the implemented filter differ, defer to the implementation; the contribution here is the twist map and the fitting procedures that wrap whatever filter exists.
- **The map** $\tilde F$ (coefficients $w$) — turning the latent into the per-candidate reweighting. Supervised by soft-Bellman self-consistency. No ground-truth $F$ involved either.

The knob values are properties of how the model's inferred latent drifts *on text* (drift rate, per-token informativeness, forgetting timescale). They cannot be transferred from a toy generative process; they must be fit on the LLM. The knobs are, in effect, a parameterization of the in-context inference, so fitting them on text and characterizing the imprint are the same measurement.

### Fitting the filter (no $F$)

The filter must reproduce the model's actual inferred-latent dynamics:

- **Experiment A** (shape features vs. position under each $\theta$) gives drift trajectories. Pushing the latent through the filter and reading out a predicted sharpness should match these curves; this pins evidence weight $\eta$ against prior strength.
- **Experiment C** (recovery length $M^\ast$ after switching to default sampling) is close to a direct readout of the forgetting knob $\lambda$, though it may entangle with $\eta$ and want joint fitting.

Caveat: A and C measure the imprint on *per-token*-sampled prefixes. If $Q_{\mathrm g}$ visits different prefix statistics, the filter is calibrated off-distribution; re-fit on $Q_{\mathrm g}$-sampled prefixes and iterate.

Fitting the filter requires a readout from the latent to the measured shape feature. This is trivial if the latent is *defined* as the tracked quantity (e.g. a decayed mean surprise) and the measured feature is the same quantity; otherwise a small readout map is fit alongside.

### Fitting the map (no ground-truth $F$)

The soft-Bellman recursion lets $\tilde F$ bootstrap. Substituting the ansatz into both sides of $F(x_{1:t}) = \sum_v P(v\mid x_{1:t})^{1/T} F(x_{1:t},v)$ and taking logs gives a residual

$$r_t \;=\; \log\tilde F(\theta_t) \;-\; \log\sum_v P(v\mid x_{1:t})^{1/T}\,\tilde F\big(\theta_{t+1}(v)\big), \qquad \log\tilde F(\theta_N)=0,$$

and $\{w, \text{knobs}\}$ are fit by minimizing $\mathbb{E}[r_t^2]$ over text prefixes, semi-gradient (stop-gradient on the bootstrap target), exactly as in soft Q-learning. The inner sum uses the model's logits from one forward pass; the terminal anchor $\log\tilde F(\theta_N)=0$ fixes the scale. **No ground-truth $F$ value is ever observed.** This is the same squared-residual objective Twisted SMC uses for its twist, specialized to a log-linear-in-latent value class.

### Supervision options, ranked by cost

1. **Bootstrapped soft-Bellman residual** — cheap, $F$-free, the default.
2. **Monte-Carlo rollouts** — roll out continuations, form importance-weighted estimates of $F$, regress $\log\tilde F$ onto them; direct but high-variance (the variance Twisted SMC's resampling exists to fight). Use as a spot-check on (1).
3. **Distillation** — run an expensive-but-correct sampler (Twisted SMC, LHTS, or short-horizon exact DP) as a teacher on text prefixes and regress the student (knobs $+\,w$) onto the teacher's relative twist $\log\frac{Q_{\mathrm g}(\cdot\mid x_{<t})}{Q_{\mathrm{pt}}(\cdot\mid x_{<t})}$. Targets the cross-candidate shape directly.

## Role of the toy model (experiment E)

Not calibration. Its only legitimate role is a one-time **method unit-test**: in a setting with clean ground-truth $F$ (exact by DP) and a latent that is sufficient by construction (Xie-style mixture-of-HMMs), can this estimator class fit the map at all? If a log-linear-in-latent twist cannot recover $F$ even there, the functional form is wrong, and this is discovered without debugging blind on the LLM (where $F$ is invisible). This is an existence proof. It does not transfer knob values: fitting the toy's latent dynamics to the LLM's measured imprint would require first measuring the LLM, at which point the knobs can be fit on the LLM directly.

## Scope and regime

- **Mild tempering is the friendly regime.** As $T\to 0$, $F$ becomes max-dominated, the value-to-go gets spiky, and a smooth latent estimator is weakest — the same regime where the gap matters most, and where the Bellman residual is hardest to drive down. Expect graceful degradation.
- **Scalar latent is defensible for temperature specifically.** Sharpness is roughly one-dimensional. Blacklist or frequency-penalty decoders require token-identity information a scalar cannot carry, so this method is temperature-shaped by design. Match estimator dimensionality to the decoder's latent dimensionality; that mapping is itself a statement of which decoders the method can touch.
- **Cheap by construction.** If the per-token observation has an exponential-family likelihood for the latent, the streaming update is conjugate and closed-form, which is what makes the per-candidate evaluation ($O(|V|)$ latent updates per step) cheap rather than prohibitive.

## Residual floor as a measurement

The irreducible Bellman-residual floor — how far below zero $\mathbb{E}[r_t^2]$ cannot be pushed with a low-dimensional latent class — directly measures how non-low-dimensional the imprint is. The quantity that limits the method also quantifies the phenomenon (this is experiment B answered from the other side). An irreducible residual is the *expected* case, not a bug to chase to zero; a stop criterion should be set in advance.

## The honest bar

The target is not "match $F$." It is "beat $F \equiv 1$ (per-token tempering) by a measurable margin with a few interpretable knobs and one coefficient vector $w$." Low floor for a first result; high ceiling if the latent is genuinely sufficient — and the residual floor reports which case obtains.
