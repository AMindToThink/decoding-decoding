# Slideshow plan — 5-minute talk

**Talk:** "Modeling and Calibrating: LLMs Figure out their Decoding Process"
(Reliable ML with Unreliable Black Boxes). ~5 min ⇒ **6 slides, ~45–50s each.**
Headline = **Claim 1 (controlled self-calibration)**.

All numbers below are script-generated (`results/counter_decode/self_calibration/recovery.json`
→ `macros.tex`; cited in the report as `\sc…` macros and Table~\ref{tab:recovery}).
Do **not** hand-type them onto slides — pull from the table/macros so they stay in sync.

---

## Slide 1 — The question (hook)
- An autoregressive LM is trained to *predict* text, so when it *generates*, it has an incentive
  to model its own decoder (temperature, top-k, blacklist).
- **Two questions:** (a) does it? (b) can we use that to make it predict better?
- One line: *"Can a model read the temperature of the text in front of it, and recalibrate itself?"*
- Visual: the feedback-loop cartoon (decoder → text → model conditions on its own text).

## Slide 2 — The tool: a streaming Bayesian filter on log β
- Per token, a Laplace filter updates a posterior on `η = log β` (inverse-temperature) from the
  prefix — online, causal, training-free. (Background: score / Fisher / Newton, 1 line each.)
- Knobs: evidence weight α, memory decay γ. Output: `β̂_t` from `x_{<t}`.
- Visual: one equation (the Newton update) + "β̂_t depends only on the past."

## Slide 3 — The honest negatives (fast; sets up Claim 1)
- Two things we checked and *rejected*, briefly:
  - **Closed-loop correction self-neutralizes** (β_dec→1; confirmed across 7 exact-grid priors —
    not an approximation artifact).
  - **Real creative writing is already ~temperature-calibrated** (best single T gives a
    non-significant +0.0036 nats; p=0.35). Nothing to fix there.
- Takeaway: *"To show self-calibration works, test it where the temperature is known and there's
  something to recover."* → motivates Slide 4.

## Slide 4 — **CLAIM 1 (headline): recovering a known temperature**
- **Setup:** generate Qwen self-text at a *known* β = 1/T (T ∈ {0.5,…,2.0}); run the filter over it.
  Sanity: pooled MLE recovers 1/T at every T.
- **Two results, one slide:**
  1. **Recovery** — the (β-corrected) *faithful* filter recovers the imposed log β from the prefix
     alone, **bias ≤ 0.10** across the range (e.g. −0.03 at T=0.7). The *original* filter is badly
     biased — overshoots for β>1, **diverges** for β<1 (bias −4.15 at T=2.0). → the β=1 fix matters.
  2. **Prediction** — applying β̂_t as a causal per-token temperature **beats the default β=1 model**:
     **−0.11 nats at T=0.7** (t=−24), up to −0.72 at T=2.0; within **0.006–0.025 nats of the oracle**
     (known-optimal 1/T); costs only ~0.006 nats at T=1 (nothing to fix).
- **Honesty line (say it out loud):** the biggest gains (T≥1.5) are on near-random text
  (gen-entropy 10–11 nats); the *coherent* sharper-than-default regimes (T=0.5/0.7) still win cleanly.
- Visual: **Fig A** — β̂_t trajectory → log(1/T), faithful vs original (recovery).
  **Fig B** — NLL bars: β=1 vs faithful vs oracle, per T. (To generate: task #10 plots.)

## Slide 5 — Does natural text present a temperature to recover? (the survey)
- Claim 1 is a *capability* demo on known-T text. Does *natural* text carry a recoverable
  temperature? Fit the best global T per text type; measure held-out NLL gain over β=1.
  **9 types**, in-distribution → far-OOD (low-resource langs, non-Latin script, ABC music,
  **Recamán integer sequence**, shuffled-text control).
- **Result: Qwen is temperature-robust everywhere.** Optimal T ∈ **[0.91, 1.08]**; best global
  temperature buys **≤0.003 nats on any real type** (0.028 on shuffled gibberish). Significant
  (on the atypical types) but *practically negligible*.
- **Directions make sense:** mildly *over*confident on OOD text (T>1); uniquely *under*confident on
  **Recamán (T=0.91)** — it's right so often it should be even sharper; the erratic jumps are rare.
  "Harder ≠ mis-temperatured": Yoruba is hardest yet T≈1.00.
- **Takeaway:** no meaningful miscalibration to exploit → the premise of an online calibrator on
  natural text doesn't hold. A clean negative; modern LLMs are temperature-robust far OOD.
- Visual: the survey table (`Table~\ref{tab:survey}`), ranked by gain.

## Slide 6 — Takeaways
- The streaming filter, **corrected**, recovers a decoder temperature from context and predicts
  that text materially better than the default model — using only the model's own logits.
- Correct Bayesian bookkeeping (the β/β² chain-rule terms) is what makes it work; the original
  filter is inconsistent.
- Honest scope: the survey shows even far-OOD text is temperature-robust (gains ≤0.003 nats), so
  the capability has little to bite on in the wild. An honest positive (Claim 1) **and** an honest
  negative (survey) — closed-loop correction was already a dead end.

---

## Build notes (for whoever makes the deck)
- Source of truth for every number: `results/counter_decode/self_calibration/recovery.json` and the
  paper's Table~\ref{tab:recovery}. Regenerate macros with
  `uv run python scripts/build_self_calibration_macros.py && uv run python scripts/build_paper_macros.py`.
- Figures A/B generated by `scripts/plot_self_calibration.py`:
  - Fig A = `results/counter_decode/self_calibration/F_selfcal_recovery.png` (β̂_t → log(1/T), faithful vs original)
  - Fig B = `results/counter_decode/self_calibration/F_selfcal_nll_gain.png` (per-T NLL gain vs β=1, faithful vs oracle)
- Keep it to 6 slides. Slide 4 is the one that must land; rehearse the honesty line so the big
  T≥1.5 numbers aren't oversold.
