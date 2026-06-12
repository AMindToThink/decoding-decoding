# Reframe plan: make the grid-mixture certificate the centerpiece

**Status:** approved direction (Matthew, 2026-06-12). Written for an implementing agent. A fresh-context critique pass has been folded in; the per-experiment honesty labels (confirmatory vs descriptive vs verification) are deliberate — do not relabel them.
**Read first:** `paper/grid_certificate_walkthrough.html` (narrative source), `paper/grid_certificate_prereg.md` (H1–H4, all pass), `paper/claim_audit.md` (overclaim fixes this reframe must absorb), `data/grid_certificate/results.json`.

## 0. Process guardrails (non-negotiable)

- Work in a git worktree; verify `git config user.name && git config user.email` = `AMindToThink` / Matthew's email before any commit. Commit per phase; **never push without asking**.
- `uv run` for everything; no pip/conda.
- **Pre-registration honesty.** Confirmatory hypotheses (thresholds committed before data exists) are only possible for experiments whose data has not been generated or seen. Anything computed from already-committed artifacts is **descriptive** and must be labeled so in the prereg doc and the paper. Never move thresholds after seeing data. Existing H1–H4 results are final — do not rerun unless a code change touches `grid_mixture_update_batched` itself (then rerun and report both).
- **No hand-typed numbers in prose.** Every scalar in the .tex resolves through `results/tables/paper_macros.tex` or `data/grid_certificate/macros.tex` (`\newcommand` from build scripts). Extend `tests/test_paper_macros.py` guards for every new macro.
- Use cheap subagents: Sonnet for exploration/bulk mechanical edits/test writing, Opus for paper prose. Keep main-loop usage minimal.
- **Out of scope (do not get distracted):** multi-model replication (GPT-2 etc.), nucleus/top-p arms, Fixed-Share drift experiments on real streams, rewriting the walkthrough HTML, refactoring legacy Laplace experiment scripts, touching `full_sequence/` work.

## 1. The reframe in one paragraph

The paper currently leads with the streaming Laplace filter and a corrector, then partially retracts (§5 "filter ≠ Guo MLE"; claim audit: 5/7 sections overclaim). The reframe inverts this: the **contribution is the certificate** — the grid Bayes mixture over inverse temperatures is Vovk's Aggregating Algorithm for log loss, giving two *per-sequence, distribution-free* guarantees on arbitrary token streams: (i) forecast regret ≤ ln G vs the best grid temperature in hindsight, and (ii) a level-set error bar around the grid MLE ("everything outside the bar fits the observed stream worse by ≥ c nats"). Under misspecification (top-k, blacklists, GIGO loops) the *effective* temperature is the only well-defined target, and the certificate is the tool that survives there — **with the honest caveat that it certifies fit to the observed stream, never the counterparty's dial, and a narrow bar means identifiability, not good fit** (E1 measures fit separately). The Laplace filter becomes the fast approximation, validated against the certificate (H4: faithful 88/96 in-bar; original 10/96). The legacy experiments become applications and stress tests, with the claim-audit fixes folded in.

## 2. Code changes (small, do these first)

Files: `src/decoding_decoding/counter_decode.py`, `src/decoding_decoding/counter_decode_generate.py`.

1. **Do NOT delete `discrete_grid_update_batched`.** It is live: the generation loop dispatches to it for all discrete-grid `prior_kind`s (`counter_decode_generate.py:313-328, 460-467`); `run_prior_sweep.py` and `run_fuzzy_window_sweep.py` depend on it. Mark it legacy/baseline in the docstring with a prominent pointer: "Legacy bootstrap variant; carries no per-sequence guarantee (experts not fixed); reports the posterior **mean** β̂ — the 'posterior-mean trap' flagged in the walkthrough. For the guarantee-bearing filter see `grid_mixture_update_batched`."
2. **Wire `grid_mixture_update_batched` into the generation loop** as a new `prior_kind="grid_mixture"` backend. **This is not a copy of the discrete-grid branch — a naive port crashes.** Specifics:
   - β̂ comes from `GridMixtureState.beta_map()` (the MAP). `GridMixtureState` has **no** `beta_hat()` or `posterior_std_log_beta()`; the loop calls those at `counter_decode_generate.py:405,407,471,473` — the new branch needs its own readout path. For the uncertainty/diagnostic slots, record the level-set width from `grid_mixture_level_set()` (and optionally `realized_regret()`); there is no precision analog `J`.
   - The discrete branch passes `evidence_weight`, `memory_decay`, `prior_pmf` to its update. `grid_mixture_update_batched` **raises on `evidence_weight≠1` and has no `memory_decay`** (deliberately — they destroy 1-mixability). The new branch must bypass that plumbing entirely; the only knob is `switch_rate` (default 0).
   - Unit-test the new branch (`tests/test_counter_decode.py` style): batched=serial, β̂ sane on synthetic streams, incompatible kwargs rejected.
3. No other refactors.

## 3. Experiments

Create `paper/certificate_misspec_prereg.md` covering E1 and E4a **before running them**; E2 and E3 are labeled below. Model experiments use Qwen2.5-3B float16 on GPU (same harness as `scripts/grid_certificate_experiments.py`). **Cache warning:** `data/grid_certificate/model_streams.npz` stores only loss/η *snapshots on the fixed 121-point grid* at n∈{200,500,1000} — it contains no logits or tokens, so nothing in it can be re-gridded or re-filtered; reuse is limited to reading those snapshots.

**Priority order: E4a and E2 are cheap and required. E1 is the headline new experiment (GPU). E3 and E4b are optional/stretch — skip them if time or GPU is short; the paper lands without them.**

**E4a — grid-resolution floor, synthetic (CPU, cheap, confirmatory).**
Median H2 width at n=1000 equals one grid step (0.05) — possibly instrument-limited. Regenerate H1-style synthetic streams (deterministic seeds) and run a 241-point grid (step 0.025), same range. Pre-register: does median width halve (data-limited below 0.05) or stick (genuine floor)? Also pre-register width thresholds for any *new* synthetic arms at β* ∈ {0.5, 2.0}. (The model-stream analog is E4b, optional, GPU — fresh generation required.)

**E2 — online-approximation lag (descriptive, CPU, zero new runs).**
From the existing `model_streams.npz` snapshots: grid MAP (argmin of stored `cum_loss`) vs `eta_faithful` vs `eta_original`, error in log β to the known dial β_target ∈ {0.5, 1.0, 2.0} at n ∈ {200, 500, 1000}. **This is descriptive, not confirmatory** — the data is committed and was already analyzed for H4; do not dress it as pre-registered. Frame it as *"how much does the online Laplace approximation lag the batch-equivalent grid MAP"*, not "which estimator is better": the grid MAP is the grid-quantized batch MLE, so it winning is largely by construction. Note the quantization floor (½ grid step = 0.025 in log β) when interpreting errors at β_target=1. The dial is a fair target only on these well-specified streams; the paper still says the guarantee is about the in-hindsight best fit.

**E1 — certificate under misspecification (headline, GPU, confirmatory where possible).**
The walkthrough *claims* the certificate is most valuable under top-k / blacklist; nothing measures it. Design requirements (all four are mandatory — each fixes a known flaw):
   1. **Generate fresh streams with full-vocab raw logits captured.** The legacy blacklist pipeline (`run_blacklist_experiment.py` → `generate.py`) saves only top-200 logprobs — reuse its token-banning logic, not its storage. Arms: (a) top-k (k matching the legacy top-k experiment), (b) token blacklist, (c) control.
   2. **Content-matched control** (per claim-audit Claim 1): same prompts, same sampling seeds, with the ban/truncation lifted — not a "matched β_target" arm (top-k has no dial to match; finding its effective β *is* the experiment).
   3. **Measure fit, not just width.** A narrow bar means L_n is sharply curved (identifiable), **not** that any temperature fits the truncated stream well — under top-k/blacklist no β reproduces the modified categorical, so the grid MLE is a projection. Report a goodness-of-fit quantity alongside the bar: best-expert mean loss L_n(β̂)/n vs the control arm's, and the realized-regret slack. Pre-registerable hypotheses live here (e.g. "best-expert per-token loss under top-k exceeds control by ≥ δ" — pick δ from a pilot on *synthetic* truncated streams, not from the real data).
   4. **Verification row, not hypothesis:** realized regret ≤ ln G holds by Lemma (i) for the mixture loss; report it as machinery verification, never as a falsifiable finding.
   Descriptive output: the effective β (grid MAP + bar) that top-k/blacklist register vs control.

**E3 — closed-loop (GIGO) arm with the certified filter (optional/stretch, GPU).**
Minimal F0-style two-arm rerun (`run_counter_decode.py` shape) with `prior_kind="grid_mixture"` as β̂ backend, β_target ∈ {0.5, 2.0}, few seeds. **Confound to state in prereg and paper:** at β_target=0.5 the stream degrades to token salad (audit Claims 3/5); the certificate then certifies the *salad's* effective temperature — a narrow bar is NOT dial recovery. The claim is strictly "the corrector consumes a certified estimate of the observed stream"; if that sentence can't carry a section, cut E3 rather than inflate it.

Each experiment extends `scripts/grid_certificate_experiments.py` or adds a sibling script with the same structure (subcommands, results.json, deterministic seeds). Build scripts emit macros/tables (`scripts/build_grid_certificate_table.py` pattern). **Fail fast**: validate preconditions, no silent `continue`.

## 4. Paper rewrite (`paper/decoding_paper.tex`)

New outline (certificate promoted from subsection §2.1.1 to the spine):

1. **Intro** — counter-decoding problem; why distribution-free matters (realistic counterparties are misspecified: top-k, blacklist, closed-loop); contributions: (a) certificate lemma + two guarantees, (b) pre-registered validation incl. misspecification, (c) Laplace as validated fast approximation, (d) what the certified estimator reveals about Qwen2.5-3B.
2. **The certificate** — grid mixture = Vovk AA; Lemma 1 (already in tex, lines 283-307); MAP = grid MLE identity; level-set bar; the caveats up front: in-hindsight target (never the dial), widening on peaked logits, guarantee attaches to the mixture, **narrow bar = identifiability ≠ goodness of fit**. Mine `grid_certificate_walkthrough.html` for prose — it is the best existing exposition (experts framing, telescoping proof, posterior-mean trap).
3. **Pre-registered validation** — H1–H4 (existing `tab_grid_certificate.tex` + gridCert* macros) + E4a; E1 results with its confirmatory/descriptive split made explicit.
4. **The fast approximation** — Laplace filters (current §2.1 content, *reframed per claim audit*: "prior-regularized one-step online MLE"; cut the "Bayesian-not-Newton" sentence); E2 lag analysis (labeled descriptive); H4 transfer.
5. **Applications: what the certified estimator reveals** — reframed legacy results: model-conditions-on-decoder-shaped-context evidence (§3, softened per audit Claim 1: not "infers its decoder"; report peaks-at-K−1 honestly), corrector dynamics (§4 with GIGO caveat per Claim 3; merge the double-counted 4.74), calibration survey (§7 — survives well, supports "don't correct at β_target=1"), natural-text offset (§5's surviving signal, paired t=−3.80).
6. **What the original framing got wrong** — short, honest section absorbing audit Claims 4–6: one-step ≠ global MLE (not "different fixed points"), Exp B reported as null, Exp C downgraded to "neither supports nor refutes", §9 reduced to "don't correct at β_target=1 because Qwen is calibrated". A feature of the reframe, not an appendix apology.
7. **Limitations** — single model/tokenizer; grid resolution floor; certificate certifies effective β of the observed stream, never the dial; closed-loop guarantee does not transfer.

Mechanics:
- Fix the one hand-typed number (`15\%` at decoding_paper.tex:264 → macro).
- New title/abstract centered on certification. Pick one and flag it for Matthew's sign-off in the final report (suggestion: "Certified Counter-Decoding: Distribution-Free Online Estimates of an LLM's Effective Decoding Temperature"); do not bikeshed.
- Keep semantic linebreaks (one sentence per line) — `latex-semantic-linebreaks` convention.
- All new numbers via macros; extend `tests/test_paper_macros.py` (defined-macro check + forbidden-literal regression check for every replaced number).
- Citations: new references via `scripts/build_bib.py` / `refs_ids.toml` (`bibliography-from-ids`), then a citation-claims verification pass.

## 5. Order of work & verification

1. Phase A (code): §2 items + unit tests → `uv run pytest` green → commit.
2. Phase B (prereg): `paper/certificate_misspec_prereg.md` (E1 + E4a confirmatory parts; E2/E3 sections labeled descriptive/optional) → commit *before any E1/E4a run*.
3. Phase C (experiments): E4a (CPU) → E2 (CPU, from snapshots) → E1 (GPU) → [optional E3, E4b] → analyze → build macros/tables → commit per experiment with results.json.
4. Phase D (paper): restructure per §4, one section per commit; `uv run pytest tests/test_paper_macros.py` after each; full build (latexmk) at the end; final claim-audit-style self-check that no softened claim regressed.
5. Final: full `uv run pytest`, paper builds clean, then stop and report (including the title proposal) — Matthew reviews before any push.
