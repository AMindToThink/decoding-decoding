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

## Slide 5 — Where it could matter for real (what's next)
- Claim 1 is a *capability* demo on known-T text — it does **not** claim natural text carries a
  recoverable temperature. The open question: **is there real text the model is mis-temperatured on?**
- **Calibration survey (running):** measure the optimal-T NLL gain across text types, especially
  ones rare/absent in training (low-resource languages, symbolic sequences: protein/chess/ABC,
  archaic English, shuffled control). At least one is likely genuinely miscalibrated for Qwen.
- If a type shows per-text headroom a global temperature misses → the causal filter could earn a
  *real* calibration win there (vs a no-filter entropy baseline). Honest about nulls.

## Slide 6 — Takeaways
- The streaming filter, **corrected**, recovers a decoder temperature from context and predicts
  that text materially better than the default model — using only the model's own logits.
- Correct Bayesian bookkeeping (the β/β² chain-rule terms) is what makes it work; the original
  filter is inconsistent.
- Honest scope: closed-loop correction is a dead end; real well-trained text is already calibrated;
  the live question is OOD/under-trained text — survey in progress.

---

## Build notes (for whoever makes the deck)
- Source of truth for every number: `results/counter_decode/self_calibration/recovery.json` and the
  paper's Table~\ref{tab:recovery}. Regenerate macros with
  `uv run python scripts/build_self_calibration_macros.py && uv run python scripts/build_paper_macros.py`.
- Figures A/B not yet generated — see task #10 (`scripts/plot_self_calibration.py`, to add).
- Keep it to 6 slides. Slide 4 is the one that must land; rehearse the honesty line so the big
  T≥1.5 numbers aren't oversold.
