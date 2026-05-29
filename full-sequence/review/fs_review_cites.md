# Full-sequence spec: citation verification

Reviewer task: verify each CITED CLAIM in `full-sequence/files/` against the real paper.
Verdict scale: SUPPORTED / PARTIALLY SUPPORTED / UNSUPPORTED / WRONG-METADATA.

Sources fetched:
- https://arxiv.org/abs/2302.03686 , https://arxiv.org/html/2302.03686v2 , https://proceedings.mlr.press/v202/shih23a.html
- https://arxiv.org/abs/2404.17546 , https://arxiv.org/pdf/2404.17546
- https://arxiv.org/abs/2410.10810 , https://aclanthology.org/2024.findings-emnlp.854/
- https://arxiv.org/abs/1603.06042 , https://aclanthology.org/P16-1231/
- https://arxiv.org/abs/2111.02080

Where the spec is located:
- 01_decoding_target.md "Relevant literature" (lines 53-59)
- 00_README.md (lines 18, 45), 02_structured_twist_method.md (lines 13-15, 71, 77)
- sequence_temperature.html ("You're right, and you've rediscovered LHTS")

---

## 1. LHTS — Shih, Sadigh, Ermon, ICML 2023, arXiv 2302.03686

**Verdict: SUPPORTED (metadata correct; all sub-claims confirmed).**

Metadata: Title "Long Horizon Temperature Scaling"; authors Andy Shih, Dorsa Sadigh, Stefano Ermon; arXiv 2302.03686; published ICML 2023 (PMLR v202, shih23a). ALL CORRECT.

Sub-claims (spec: "Names per-token tempering 'myopic' and targets the temperature-scaled joint instead, by fine-tuning a model to approximate P^{1/T}/Z across a range of T. Reports mixed empirical results: did not improve MAUVE on open-ended GPT-2 generation, and both tempering forms tended to hurt it."):

- "myopic": SUPPORTED. Abstract verbatim: "autoregressive models rely on myopic temperature scaling that greedily optimizes the next token."
- targets temperature-scaled JOINT: SUPPORTED. Abstract: "a novel approach for sampling from temperature-scaled joint distributions ... optimizes for the long horizon likelihood of samples."
- fine-tuning across a range of T: SUPPORTED. Abstract: "finetuning a model on a range of temperatures produces a single model capable of generation with a controllable long horizon temperature parameter."
- MAUVE: SUPPORTED. Verbatim from the LHTS paper (HTML/PDF, OpenWebText, GPT-2, MAUVE-paper setup, 1000 generations, prompt length 30): "We find that LHTS does not improve MAUVE score, and that both forms of temperature scaling (myopic and LHTS) in general decrease MAUVE score." Table 3 values: No Scaling 0.76; Myopic-only (T=0.8) 0.57; LHTS (T=0.9) 0.41; both = 0.00 at T=0.0. This directly confirms the spec's "did not improve MAUVE on open-ended GPT-2 generation, and both tempering forms tended to hurt it." Overall the paper still reports "advantages over myopic temperature scaling in likelihood and sample quality" on OTHER tasks/domains, so the spec's "mixed empirical results" framing is fair — LHTS wins on some metrics, loses on open-ended-MAUVE.

"sequence-level tempering = what LHTS targets": ACCURATE. LHTS's target is literally P^{1/T} normalized over whole sequences (the joint), i.e. the spec's Q_g.

Headline framing "you've rediscovered LHTS": FAIR. LHTS names "myopic," writes the sequence-level target, and attacks the same missing-future-partition-function gap. The spec is careful to distinguish amortization route (LHTS = fine-tune into weights vs. the spec's streaming twist), which is correct.

---

## 2. Twisted SMC — Zhao et al., ICML 2024, arXiv 2404.17546

**Verdict: PARTIALLY SUPPORTED — metadata: minor title shortening; one nuance on the objective.**

Metadata: Real title is "Probabilistic Inference in Language Models via Twisted **Sequential Monte Carlo**." The spec writes "...via Twisted SMC" (00_README, 01_decoding_target line 56). "SMC" = "Sequential Monte Carlo," so this is an acceptable abbreviation, not a wrong title — but for a bibliography the full title should be spelled out. Authors Stephen Zhao, Rob Brekelmans, Alireza Makhzani, Roger Grosse; arXiv 2404.17546; ICML 2024. CORRECT.

Sub-claims (spec: "Learns twist functions that estimate the expected future value of a sequence-level potential at each step, used as SMC proposals; twist learning via a squared-error objective analogous to soft Q-learning."):

- twist functions estimate expected future value of a sequence-level potential at each step: SUPPORTED. Abstract verbatim: "we use learned twist functions to estimate the expected future value of the potential at each timestep ... [potential] over the full sequence."
- used as SMC proposals: SUPPORTED (the whole method is twisted SMC; twists define the intermediate targets/proposals).
- "twist learning via a squared-error objective analogous to soft Q-learning": PARTIALLY SUPPORTED — the spec describes a PRIOR-WORK BASELINE objective, not the paper's signature method. Verbatim from the PDF (extracted /tmp/twisted_smc.pdf):
  - "We instead propose a novel contrastive twist learning (CTL) method, and show experimentally that CTL outperforms previous methods including those based on a squared error or `soft Q-learning' (Mudgal et al., 2023; Han et al., 2024) and the path consistency objective of Nachum et al. (2017)."
  - "For learning the twists, we propose a novel contrastive method (CTL), and also evaluate twisted SMC using existing twist-learning losses based on ... approximate dynamic programming, or `soft Q-learning' style methods."
  - Framing: "we provide a unified framework which recovers a number of recent works as special cases" + "connecting our twist-learning framework with the rich literature of soft reinforcement learning."
  So: the squared-error / soft-Q-learning twist loss is REAL and IS discussed/evaluated in the paper — but it is an EXISTING method (attributed to Mudgal et al. 2023 / Han et al. 2024), which CTL is shown to BEAT. The paper's own contribution is CTL (contrastive), NOT squared error. The spec's clause "twist learning via a squared-error objective analogous to soft Q-learning" therefore attributes a baseline to the paper as if it were the paper's method.

CORRECTED STATEMENT (01_decoding_target line 56): "Learns twist functions estimating the expected future value of a sequence-level potential, used in SMC; their proposed objective is *contrastive* (CTL), and they unify it with prior squared-error / soft-Q-learning and path-consistency twist losses under a soft-RL framework." For 02_structured_twist_method.md line 71 ("This is the same squared-residual objective Twisted SMC uses for its twist"): this is WRONG as written — squared error is not the objective Twisted SMC "uses" (it uses CTL). Fix to: "This is the squared-residual / soft-Q-learning twist objective that Twisted SMC includes as a baseline within its unified soft-RL framework (its own proposed objective is contrastive)." The spec's METHOD is genuinely a squared-Bellman-residual objective, so the analogy to that baseline is apt — only the attribution ("the objective Twisted SMC uses") is off.

---

## 3. Gareev et al., "Local and Global Decoding in Text Generation," arXiv 2410.10810

**Verdict: SUPPORTED — but venue is more specific than spec gives (no venue stated; it's EMNLP 2024 Findings).**

Metadata: Title exact. Authors Daniel Gareev, Thomas Hofmann, Ezhilmathi Krishnasamy, Tiago Pimentel. arXiv 2410.10810 (14 Oct 2024). Published Findings of EMNLP 2024 (2024.findings-emnlp.854, pp. 14577–14597). The spec gives only the arXiv id and no venue/authors inline — acceptable for a memo, but a formal bibliography entry should add the four authors and the Findings-of-EMNLP-2024 venue (not main-conference EMNLP). Not WRONG, just incomplete.

Sub-claims (spec: "The same local-vs-global split for *truncation* rather than temperature ... local renormalization at each step distorts the joint relative to globally normalized truncation. About equilibrium distortion, not prefix-imprint dynamics over position."):

- truncation (top-k / top-p), not temperature: SUPPORTED. Abstract: "Traditional methods, such as top-$k$ and top-$\pi$, apply local normalisation to the model's output distribution, which can distort it." (top-$\pi$ is their notation for nucleus/top-p.)
- local renormalization distorts the joint vs. globally normalized version: SUPPORTED. They "introduce globally-normalised versions of these decoding methods" and study "the effect of this distortion."
- about equilibrium/distribution distortion, not prefix dynamics over position: SUPPORTED. The framing is distribution-level ("distortion is an important feature of local decoding algorithms"); they use independent Metropolis–Hastings to sample the global distribution. Nothing about position-indexed prefix dynamics. The spec's contrast ("not prefix-imprint dynamics") is fair.

The characterization is accurate. (Aside, not a spec error: the paper's empirical punchline is that GLOBAL decoding usually performs WORSE than local for top-k/top-p — i.e. distortion helps. The spec doesn't claim otherwise, but anyone leaning on this cite to argue "global is better" would be misreading the paper.)

---

## 4. Andor et al. 2016, Globally Normalized Transition-Based Neural Networks (label bias)

**Verdict: SUPPORTED.**

Metadata: Title "Globally Normalized Transition-Based Neural Networks"; authors Daniel Andor, Chris Alberti, David Weiss, Aliaksei Severyn, Alessandro Presta, Kuzman Ganchev, Slav Petrov, Michael Collins; ACL 2016 (P16-1231, pp. 2442–2452); arXiv 1603.06042. CORRECT (spec only says "Andor et al. 2016," which is fine).

Sub-claim (spec: "locally normalized models cannot represent arbitrary global distributions, and tempering locally reintroduces a label-bias-flavored pathology"; framed as "the structural ancestor"):

- label-bias / expressiveness: SUPPORTED. Abstract verbatim: "a key insight is that the label bias problem implies that globally normalized models can be strictly more expressive than locally normalized models." This directly supports "locally normalized models cannot represent arbitrary global distributions."
- "structural ancestor" framing: REASONABLE. Andor is the canonical modern statement that local normalization is strictly less expressive than global normalization (the label-bias argument), which is exactly the structural fact the spec's per-token-vs-sequence-level tempering gap instantiates. The "tempering locally reintroduces a label-bias-flavored pathology" is the spec's own analogy/interpretation (not a claim attributed to Andor), and it is a fair conceptual link, not a miscitation.

---

## 5. Xie et al., ICLR 2022, "An Explanation of In-Context Learning as Implicit Bayesian Inference," arXiv 2111.02080

**Verdict: SUPPORTED (with a DP plausibility caveat that is the spec's own extension, not a miscitation).**

Metadata: Title "An Explanation of In-context Learning as Implicit Bayesian Inference"; authors Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma; arXiv 2111.02080 (3 Nov 2021); ICLR 2022. CORRECT.

Sub-claims (spec: "Proves ICL emerges as Bayesian inference over a latent concept under a mixture-of-HMMs pretraining distribution"; used as toy model where "ground-truth F is computable by dynamic programming"):

- ICL as Bayesian inference over a latent concept: SUPPORTED. Abstract: "the LM must infer a latent document-level concept ... in-context learning occurs when the LM also infers a shared latent concept between examples in a prompt."
- mixture-of-HMMs pretraining distribution: SUPPORTED. Abstract: "a setting where the pretraining distribution is a mixture of HMMs." (Synthetic dataset = GINC.)
- "ground-truth F computable by DP on a mixture-of-HMMs": PLAUSIBLE / SOUND, and clearly the spec's OWN construction, not attributed to Xie. In a known finite mixture of HMMs the marginal sequence probability P(x_{1:N}) and tempered future partition functions are exactly computable by forward/backward DP over the (finite) hidden-state x latent-concept space, so F(x_{1:t}) = sum over continuations of prod P^{1/T} is DP-computable for fixed N. The spec uses Xie only as the toy generative process (experiment E), which is legitimate. Caveat for the authors: F as defined (tempered, P^{1/T}) is NOT a vanilla HMM forward-backward — the per-step factor is P(x_t|x_{<t})^{1/T}, where P(x_t|x_{<t}) is the HMM's predictive marginal (which already integrates the hidden state). So the DP must be over the joint (hidden-state distribution given prefix) AND track the tempered token factors; it is still polynomial-time but is a custom recursion, not off-the-shelf HMM DP. This is an implementation note, not a citation defect.

---

## Bottom line

No WRONG-METADATA. No substantively misstated paper FINDING. One real wording fix + metadata completeness:
1. (Real fix, 02_structured_twist_method.md line 71) "the same squared-residual objective Twisted SMC uses for its twist" is INACCURATE: Twisted SMC's proposed objective is **contrastive (CTL)**; squared-error / soft-Q-learning is a PRIOR-WORK baseline (Mudgal 2023, Han 2024) that the paper unifies and BEATS. Re-attribute: it's a baseline in their framework, not the objective they use. (The spec's own method really is a squared-Bellman-residual loss, so the math analogy is fine; only the attribution is wrong.) 01_decoding_target line 56's "twist learning via a squared-error objective analogous to soft Q-learning" has the same attribution slip.
2. (Metadata completeness) Spell out Twisted SMC's full title ("...via Twisted Sequential Monte Carlo") and add Gareev et al.'s authors + "Findings of EMNLP 2024" venue if these graduate into a formal bibliography. Repo rule: fetch metadata, don't hand-type. All five papers' authors/ids/years as given are CORRECT.
