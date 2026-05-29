# Structured streaming twist: reference pseudocode

Companion to `02_structured_twist_method.md`. Notation follows that doc:
the base model gives conditionals $P(\cdot \mid x_{<t})$; the latent state is
$\theta_t$; the twist is $\log\tilde F(\theta) = w^\top \phi(\theta)$; the target
temperature is $T$.

This is language-agnostic pseudocode. The filter is shown with one concrete
instantiation (exponential-forgetting Bayesian mean), but the interface
(`init`, `peek`, `commit`) is what matters; any conjugate streaming estimator
of a low-dimensional latent can be dropped in.

---

## 1. Configuration

```
T                      # target temperature
phi(theta) -> vector   # latent features; scalar latent: phi(theta) = [theta]
w                      # twist coefficients (fit offline, section 4)

# filter knobs (fit offline, section 3)
theta0                 # prior latent mean
kappa0                 # prior strength (pseudo-count)
eta                    # evidence weight per observation
lam                    # decay / forgetting factor in (0, 1]
```

## 2. The streaming filter

> **Defer to the existing implementation.** The repo already contains a working
> streaming filter, designed with experience this pseudocode does not have. The
> update shown below is an *illustrative placeholder* chosen only to expose the
> `init` / `peek` / `commit` interface and show where the filter plugs into the
> sampling loop and the fits. It is not a recommendation about the latent's
> parameterization, what it observes, or how it updates. Wherever this placeholder
> and the existing filter disagree, the existing filter is authoritative; treat
> the rest of this document (sampling loop, candidate-conditioning, both offline
> fits) as wrapping *that* filter through the same interface. The one externally
> imposed requirement is the candidate-conditioning contract in section 3 — the
> filter must support a side-effect-free `peek` so the twist can be evaluated per
> candidate token.

State is the posterior over the scalar latent, summarized as a mean `m` and an
effective pseudo-count `c`. `peek` returns the state one *would* hold after a
hypothetical observation without mutating anything; `commit` applies it for real.
The observation fed to the filter is a token's surprise under the base model.

```
function filter_init():
    return State(m = theta0, c = kappa0)

# exponential-forgetting conjugate-mean update; pure, returns new state
function peek(state, o):                  # o is an observation (a surprise value)
    c_new = lam * state.c + eta
    m_new = (lam * state.c * state.m + eta * o) / c_new
    return State(m = m_new, c = c_new)

function commit(state, o):
    return peek(state, o)                 # same update, used to advance the chain

function log_twist(state):
    return dot(w, phi(state.m))           # log F-tilde(theta)
```

## 3. Sampling loop (candidate-conditioned twist)

The correction is candidate-conditioned: for each candidate next-token `v`, the
twist is evaluated at the latent that committing `v` would produce. A
prefix-only twist would cancel in renormalization and reduce to per-token
tempering (see method doc, "must be candidate-conditioned").

```
function sample_sequence(prompt, N):
    x     = prompt
    state = filter_init()
    # advance the filter over the prompt so the latent reflects it
    for token in prompt_tokens_after_first:
        p_prev = base_conditional(x_up_to(token))     # P(. | prefix before token)
        state  = commit(state, surprise(p_prev, token))

    for t in 1..N:
        p = base_conditional(x)                       # vector P(. | x_<t) over vocab V

        tempered = p ** (1 / T)                        # unnormalized, length |V|
        logtw    = zeros(|V|)
        for v in V:
            o_v       = -log(p[v])                     # candidate surprise
            state_v   = peek(state, o_v)               # latent AFTER hypothetically emitting v
            logtw[v]  = log_twist(state_v)

        # candidate-conditioned reweighting, then renormalize
        weights = tempered * exp(logtw - max(logtw))   # subtract max for stability
        q       = weights / sum(weights)

        x_t = sample_categorical(q)
        x   = append(x, x_t)
        state = commit(state, -log(p[x_t]))            # advance filter with chosen token

        if x_t == EOS: break
    return x
```

Complexity per step: one base forward pass plus `O(|V|)` constant-time latent
peeks. With a closed-form (conjugate) `peek`, the overhead over plain per-token
sampling is a small constant factor.

## 4. Offline fit A — filter knobs (no ground-truth F)

Fit `(theta0, kappa0, eta, lam)` so the filter reproduces the measured imprint
dynamics. `readout(m)` maps the latent to the shape feature that was measured
(identity if the latent is defined as that feature, e.g. decayed mean surprise).

```
# Inputs:
#   exp_A: for each swept decoder setting, the averaged measured shape-feature
#          trajectory  f_target[t]  over positions t  (drift curves)
#   exp_C: measured recovery length M_star after switching to default sampling
#          (primarily informs lam)

function filter_fit(exp_A, exp_C):
    minimize over (theta0, kappa0, eta, lam):
        loss = 0
        for trajectory in exp_A:                       # one per swept setting
            state = filter_init()
            for t, observed_surprise in trajectory:
                state = commit(state, observed_surprise)
                loss += (readout(state.m) - f_target[t]) ** 2
        loss += recovery_penalty(lam, exp_C.M_star)    # match decay to recovery length
    return (theta0, kappa0, eta, lam)
```

Caveat (method doc): `exp_A`/`exp_C` are measured on per-token-sampled prefixes.
If `Q_g` visits different prefix statistics, re-collect trajectories under the
fitted sampler and refit; iterate.

## 5. Offline fit B — twist coefficients via soft-Bellman residual (no ground-truth F)

Bootstrapped fit of `w` (optionally jointly refining the knobs). The target uses
a stop-gradient, as in soft Q-learning. Anchored by `log F-tilde(theta_N) = 0`.

```
# Inputs: a corpus of text prefixes (from the model or from natural text)

function map_fit(prefixes):
    minimize over w (and optionally knobs):
        loss = 0
        for x in prefixes:
            state = filter_init()
            for t in 1..len(x):
                p = base_conditional(x_<t)                 # P(. | x_<t)

                pred = log_twist(state)                    # w . phi(theta_t)

                # bootstrap target: log sum_v p[v]^(1/T) * F-tilde(theta after v)
                acc = []
                for v in V:
                    state_v = peek(state, -log(p[v]))
                    acc.append( (1/T) * log(p[v]) + stop_grad(log_twist(state_v)) )
                target = logsumexp(acc)

                loss += (pred - target) ** 2
                state = commit(state, -log(p[x_t]))        # advance along the actual prefix

            # terminal anchor
            loss += anchor_weight * (log_twist(state) ** 2)
    return w
```

Notes:
- The inner `logsumexp` over `v` is full-vocab; subsample `v` for large `|V|`
  if needed, keeping the chosen token in the sample.
- Fits A and B can be run in sequence (knobs first from imprint data, then `w`
  by bootstrapping) or jointly. Neither requires observing a ground-truth `F`.
- Monte-Carlo rollout targets and distillation from an expensive correct sampler
  (method doc, section "Supervision options") are drop-in replacements for the
  bootstrap target if a check on the bootstrapped fit is wanted.

## 6. Validation (optional, uses ground-truth F)

On the toy model organism (experiment E), `F` is computable exactly by dynamic
programming over the known generative process. Use it only to confirm the
log-linear-in-latent class can fit `F` when the latent is sufficient by
construction — an existence test of the functional form, not a source of knob
values for the LLM.

```
function toy_existence_test():
    F_exact = dp_partition_function(toy_model, T)          # exact, by DP
    fit (w, knobs) on toy rollouts via map_fit / filter_fit
    report ||log F-tilde(theta_t) - log F_exact(x_1:t)||   # if large here, form is wrong
```
