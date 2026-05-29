# The decoding target: per-token vs. sequence-level temperature

## Setup

An autoregressive model is locally normalized. Write the length-$N$ joint as

$$P(x_{1:N}) = \prod_{t=1}^{N} P(x_t \mid x_{<t}).$$

"Temperature $T$" can mean two inequivalent operations on top of this.

## Two distributions

**Per-token (myopic) tempering** — what vLLM and standard inference stacks do. Temper each conditional and renormalize at every step:

$$Q_{\mathrm{pt}}(x_{1:N}) = \prod_{t=1}^{N} \frac{P(x_t \mid x_{<t})^{1/T}}{Z_t(x_{<t})}, \qquad Z_t(x_{<t}) = \sum_{v \in V} P(v \mid x_{<t})^{1/T}.$$

**Sequence-level (global) tempering** — the principled target. Temper the joint and renormalize once:

$$Q_{\mathrm{g}}(x_{1:N}) = \frac{P(x_{1:N})^{1/T}}{Z}, \qquad Z = \sum_{y_{1:N} \in V^{N}} P(y_{1:N})^{1/T}.$$

## Why they differ

Because $\left(\prod_t P(x_t\mid x_{<t})\right)^{1/T} = \prod_t P(x_t\mid x_{<t})^{1/T}$, the **numerators are identical**. The entire difference is in the denominator:

$$\frac{Q_{\mathrm{pt}}(x_{1:N})}{Q_{\mathrm{g}}(x_{1:N})} = \frac{Z}{\prod_{t=1}^{N} Z_t(x_{<t})}.$$

$Z$ is a single constant; $\prod_t Z_t(x_{<t})$ is **path-dependent**, varying with the specific string through every prefix it threads. Strings passing through high-$Z_t$ prefixes are systematically reweighted relative to sequence-level tempering. The two distributions coincide only at $T = 1$, where every $Z_t = 1$ and $Z = 1$. The discrepancy is exact, not an artifact of finite sampling.

## The future partition function

Marginalizing $Q_{\mathrm g}$ recovers its true per-step conditional:

$$Q_{\mathrm{g}}(x_t \mid x_{<t}) \;\propto\; \underbrace{P(x_t \mid x_{<t})^{1/T}}_{\text{tempered conditional}} \cdot \underbrace{F(x_{1:t})}_{\text{future partition function}}, \qquad F(x_{1:t}) = \sum_{x_{t+1:N}} \prod_{s>t} P(x_s \mid x_{<s})^{1/T}.$$

$F$ obeys a backward recursion

$$F(x_{1:t}) = \sum_{x_{t+1}} P(x_{t+1}\mid x_{\le t})^{1/T}\, F(x_{1:t+1}), \qquad F(x_{1:N}) = 1,$$

which is a soft-Bellman backup with per-step "reward" $\tfrac{1}{T}\log P$ and value $\log F$. So $\log F$ is a **soft value-to-go**, and computing it exactly is intractable (a sum over $|V|^{N-t}$ continuations).

> **The bug, stated cleanly:** per-token tempering is exactly the approximation $F(x_{1:t}) \equiv 1$. It ignores how the current token reshapes the tempered mass of all possible futures.

### Intuition for $F$

At each step $F$ upweights candidate next-tokens that lead to high tempered future mass. For $T<1$ this is mode-seeking lookahead (a softened beam search): tokens leading into regions where the model is confident accrue large $F$. Whether sequence-level tempering is *net* sharper or flatter than per-token in realized-entropy terms is model-dependent and should be measured rather than assumed. (Note the empirical mixed results for sequence-level tempering on open-ended generation; see LHTS below.)

## Boundary cases

- **Fixing $N$** keeps $Q_{\mathrm g}$ a clean distribution over $V^N$.
- **Variable length with EOS** is subtler: tempering occurs over a tree with terminal probabilities, and one must decide how short strings are reweighted against long ones. This requires an explicit modeling choice and is not handled by the fixed-$N$ formulation.
- As $T \to 0$, $\log F$ becomes max-dominated (the $\log\!\sum\!\exp$ collapses toward a max), the value-to-go grows spiky, and any smooth approximation of $F$ degrades — the same regime where the local/global gap matters most.

## Relevant literature

- **Long Horizon Temperature Scaling (LHTS).** Shih, Sadigh, Ermon, ICML 2023 (arXiv 2302.03686). Names per-token tempering "myopic" and targets the temperature-scaled joint instead, by fine-tuning a model to approximate $P^{1/T}/Z$ across a range of $T$. Reports mixed empirical results: did not improve MAUVE on open-ended GPT-2 generation, and both tempering forms tended to hurt it. Amortizes the correction into model weights.
- **Probabilistic Inference in Language Models via Twisted SMC.** Zhao et al., ICML 2024 (arXiv 2404.17546). Learns twist functions that estimate the expected future value of a sequence-level potential at each step, used as SMC proposals; twist learning via a squared-error objective analogous to soft Q-learning. The general inference-time route to sampling sequence-level targets. Amortizes the correction into a learned neural twist.
- **Local and Global Decoding in Text Generation.** Gareev et al. (arXiv 2410.10810). The same local-vs-global split for *truncation* rather than temperature: local renormalization at each step distorts the joint relative to globally normalized truncation. About equilibrium distortion, not prefix-imprint dynamics over position. (Cited in the research memo's reference list.)
- **Label bias / globally vs. locally normalized sequence models.** E.g. Andor et al. 2016 on globally normalized transition-based networks. The structural ancestor: locally normalized models cannot represent arbitrary global distributions, and tempering locally reintroduces a label-bias-flavored pathology.
- **In-context learning as implicit Bayesian inference.** Xie et al., ICLR 2022. Proves ICL emerges as Bayesian inference over a latent concept under a mixture-of-HMMs pretraining distribution. The theoretical basis for the imprint framing and the natural choice for the toy model organism (experiment E), where ground-truth $F$ is computable by dynamic programming.

## The imprint vs. the normalization mismatch

These are distinct and should not be conflated:

- The **local-vs-global normalization mismatch** ($\prod_t Z_t$ vs. $Z$) exists even for a model with zero in-context drift. It is a property of the operator.
- The **imprint** is a property of the model: $P_\phi(\cdot\mid x_{<t})$ responding to prefix statistics.

Sampling from $Q_{\mathrm g}$ does not remove the imprint. $Q_{\mathrm g}$ is defined from the base $P$, and its conditional $Q_{\mathrm g}(x_t\mid x_{<t}) \propto P(x_t\mid x_{<t})^{1/T} F(x_{1:t})$ still has $P$ conditioning on the prefix. The imprint is the reason naive per-token sampling misses $Q_{\mathrm g}$; the correction is to sample $Q_{\mathrm g}$ correctly, i.e. to approximate $F$.
