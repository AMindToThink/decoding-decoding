# decoding-decoding

An autoregressive language model is trained to *predict* text, so when it
*generates* text it has an incentive to model its own decoder — the
temperature, top-*k* cutoff, or banned-token list that turns its logits into
tokens. This repo investigates whether a model (Qwen2.5-3B, with gpt2-large
as a secondary check) can be shown to track that decoder from its own
generated context, builds a streaming Bayesian estimator that reads an
effective inverse-temperature off a token stream online, and frames that
estimator as a grid of fixed-temperature experts equivalent to Vovk's
Aggregating Algorithm for log loss — which yields two per-sequence,
distribution-free guarantees (a forecast-regret bound and a level-set error
bar around the grid MLE). See `paper/writeup.tex` ("Knowing One's Self is
Wisdom: An Analysis of an LLM's Knowledge of Its Own Decoder") for the full
writeup, or `paper/decoding_paper.tex` for an earlier draft.

For exact steps to set up the environment, obtain the data, and re-run the
experiments and paper build, see **[REPRODUCE.md](REPRODUCE.md)**.
