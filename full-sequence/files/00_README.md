# Project context: temperature-correct decoding under in-context inference

## Goal

A base (non-instruction-tuned) language model adjusts its next-token distribution in response to the statistical signature that its own decoder leaves on the generated prefix. The research memo (`double_sharpening.html`) calls this the **imprint** and frames it as in-context Bayesian inference over decoder-induced latents. This sub-project asks a narrower, concrete question that falls out of that framing:

> What does it mean to "decode at temperature $T$," and how do we actually sample from that target given that the model conditions on its own prefix?

## The reframe

"Apply temperature $T$" has two inequivalent meanings for an autoregressive model:

- **Per-token (myopic) tempering** — temper each conditional and renormalize at every step. This is what vLLM and every standard inference stack do.
- **Sequence-level (global) tempering** — temper the joint over whole strings and renormalize once: $Q_{\mathrm g}(x_{1:N}) \propto P(x_{1:N})^{1/T}$.

These are different distributions for $T \neq 1$. The sequence-level object is the principled definition of "decode at temperature $T$"; per-token tempering is a biased approximation to it. The bias is exactly a missing **future partition function** $F$ (a soft value-to-go / twist). See `01_decoding_target.md`.

The proposed correction is a **structured streaming twist**: approximate $F$ with a low-dimensional streaming filter plus a small log-linear map, rather than fine-tuning the model (LHTS) or learning a neural twist (Twisted SMC). See `02_structured_twist_method.md` and `03_pseudocode.md`.

> **Scope note.** The repo already contains a working streaming filter, built with experience these docs do not have. The contribution described here is the **twist map** ($\tilde F$) and the **fitting procedures**, which wrap the existing filter. Any filter internals in these docs (update rules, knob names, observation choice) are illustrative placeholders; where they disagree with the implemented filter, the implementation is authoritative.

## Document map

| File | Contents |
|---|---|
| `00_README.md` | This orientation. |
| `01_decoding_target.md` | Per-token vs. sequence-level temperature; the future partition function $F$; relevant literature; the imprint-vs-normalization distinction. |
| `02_structured_twist_method.md` | The structured streaming twist: defining the map $\tilde F$, the two-part fit, where the supervision comes from, scope and failure modes. |
| `03_pseudocode.md` | Reference pseudocode for sampling and for both offline fits. |
| `double_sharpening.html` | The original research memo: the imprint phenomenon and experiments A–E. (Pre-existing.) |

## One distinction to keep straight

The **imprint** and the **local-vs-global normalization mismatch** are related but distinct, and conflating them causes errors:

- The normalization mismatch ($\prod_t Z_t$ vs. a single $Z$) exists even for a model with zero in-context drift. It is a property of the *operator* (tempering a locally normalized model).
- The imprint is a property of the *model*: $P_\phi(\cdot \mid x_{<t})$ actually responding to prefix statistics.

Consequently, **sampling from $Q_{\mathrm g}$ does not "remove" the imprint.** $Q_{\mathrm g}$ is defined entirely from the base $P$, and its own per-step conditional still has the model conditioning on the prefix (through both $P^{1/T}$ and $F$). A sequence model conditioning on its prefix is correct behavior, not a bug. The honest framing: $Q_{\mathrm g}$ is the target; the imprint is the model property that makes naive per-token sampling miss that target; the method's job is to sample $Q_{\mathrm g}$ correctly, which reduces to approximating $F$.

## How this connects to the memo's experiments

- **A** (shape-feature drift vs. position under a $\theta$ sweep) and **C** (imprint recovery length $M^\ast$ after switching to default sampling) supply the supervision for the streaming filter's knobs. No ground-truth $F$ needed.
- **B** (whether the imprint is multi-dimensional or scalar-summarizable) is answered from the other side by the method's irreducible Bellman-residual floor: how far the residual cannot be pushed down with a low-dim latent measures the imprint's dimensionality.
- **E** (toy transformer on a mixture-of-HMMs generative process, Xie et al. style) is where ground-truth $F$ is computable by dynamic programming. Its role here is a one-time existence test of the method's functional form, **not** calibration of knob values (those are LLM/text properties).

## Open questions specific to this sub-project

- **Latent dimensionality.** Is a scalar sharpness latent the right starting ansatz, or should the latent separate, e.g., an inferred-temperature dimension from an inferred-text-type dimension? The method's dimensionality should match the decoder's latent dimensionality; temperature is the scalar-friendly case, vocabulary-bias decoders are not.
- **Off-distribution filter calibration.** Experiments A and C measure the imprint on per-token-sampled prefixes. If $Q_{\mathrm g}$ visits different prefix statistics, the filter is calibrated slightly off-distribution and must be re-fit on $Q_{\mathrm g}$-sampled prefixes (a short chicken-and-egg iteration).
- **Stop criterion.** Because an irreducible residual is expected (the latent is not exactly sufficient), decide in advance what residual level counts as "good enough to beat $F \equiv 1$ by a useful margin."
- **Sign of net effect.** $F$ does mode-seeking lookahead at $T<1$; whether sequence-level tempering is net sharper or flatter than per-token in realized-entropy terms is model-dependent and should be measured, not assumed.
