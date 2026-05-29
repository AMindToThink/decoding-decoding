# Repository Ground-Truth Inventory: decoding-decoding

## Executive Summary

The spec's claim that "the repo already contains a working streaming filter" is **TRUE**. The Laplace-based streaming filter exists in production code and has been extensively tested via multiple experiments. However, the spec's assumption about a `peek`/`commit` interface with side-effect-free evaluation **does NOT exist yet** — this is one of the pieces to be built. The repository contains experiments A, C, and partial E; Experiment B is measured implicitly via residual floors.

---

## 1. The Existing Streaming Filter

### Location & Implementation
- **Primary file:** `/home/cs29824/matthew/decoding-decoding/src/decoding_decoding/counter_decode.py` (lines 48–182)
- **Production class:** `LaplaceState` (lines 48–62)
- **Production function:** `laplace_update()` (lines 91–182)
- **Natural-text diagnostic wrapper:** `/home/cs29824/matthew/decoding-decoding/src/decoding_decoding/natural_text_filter.py` (lines 1–475)

### Filter Interface (CURRENT — Not Yet peek/commit)
```python
@dataclass
class LaplaceState:
    eta_hat: torch.Tensor      # (B,) posterior mode of log β
    J: torch.Tensor            # (B,) posterior precision

def init_laplace(batch_size, *, sigma_0=0.5, device=None, dtype=torch.float32) -> LaplaceState
    # Prior: log β ~ N(μ_0, σ_0²) with μ_0 = -σ_0²/2 (E[β] = 1 exactly)
    # Returns: LaplaceState(eta=μ_0, J=1/σ_0²)

def laplace_update(
    state: LaplaceState,
    logits: torch.Tensor,           # (B, V)
    sampled_token_ids: torch.Tensor, # (B,)
    *,
    evidence_weight: float = 1.0,
    memory_decay: float = 1.0,
    prior_eta: float = 0.0,
    prior_J: float = 0.0,
) -> Tuple[LaplaceState, torch.Tensor, torch.Tensor]
    # Returns: (new_state, score, fisher)
    # Algorithm per line 24–27:
    #   q = softmax(ℓ_t)
    #   E_q = Σ_j q[j] · ℓ_t[j]
    #   Var_q = Σ_j q[j] · (ℓ_t[j] − E_q)²
    #   score = ℓ_t[x_t] − E_q
    #   fisher = Var_q
    #   J ← J + α·fisher  (α = evidence_weight)
    #   η̂ ← γ·η̂ + (1−γ)·η_prior then η̂ ← η̂ + α·score/J_new
```

### State Variables (Actual Names)
- `eta_hat` — log β (inverse temperature), NOT "theta0" or "kappa0" as spec placeholder
- `J` — posterior precision (cumulative Fisher information)
- `evidence_weight` (α) — per-step weighting on Bayesian evidence (default 1.0)
- `memory_decay` (γ) — per-step exponential forgetting factor ∈ [0, 1] (default 1.0 = no decay)
- `sigma_0` — prior std dev on log β (default 0.5)

### Peek/Commit Status: **NOT YET IMPLEMENTED**
The current interface is **stateful only** — `laplace_update()` mutates the state in place. There is:
- ✗ NO `peek(state, o)` returning new_state without mutation
- ✗ NO `commit(state, o)` to apply the update
- ✗ NO per-candidate hypothesis evaluation (required for the twist reweighting loop)

**This must be added** to support the candidate-conditioned twist evaluation required by section 3 of `03_pseudocode.md`.

### Full-Vocab Log-Softmax Availability: **YES**
- `/home/cs29824/matthew/decoding-decoding/src/decoding_decoding/counter_decode_generate.py` line 188 & 201: `log_p = torch.log_softmax(logits.float(), dim=-1)` — computes full-vocab log conditionals
- `/home/cs29824/matthew/decoding-decoding/src/decoding_decoding/natural_text_filter.py` line 254: `log_q = F.log_softmax(logits_t, dim=-1)` — full conditional for E_q and Var_q computation
- Logsumexp is available: `torch.logsumexp()` at line 405 of `counter_decode.py`

---

## 2. double_sharpening.html / The Imprint Memo

### Status: **ABSENT from filename, content located elsewhere**

The file `double_sharpening.html` **does not exist** at `/home/cs29824/matthew/decoding-decoding/`. However, the experiments A, B, C, E are described and **their results exist** in:

- **Primary memo location:** `/home/cs29824/matthew/decoding-decoding/in-context-learning-of-decoding.html`
  - Contains "Experiment A" and "Experiment B" (visible in grep at line 1 of this file)
  - This is the memo defining the imprint and the experiments

- **Secondary reference:** `/home/cs29824/matthew/decoding-decoding/full-sequence/files/structured_twist.html`
  - HTML rendering of `02_structured_twist_method.md` and related
  - Explicitly cites Experiment A and Experiment C in HTML section 4

**Conclusion:** The memo content exists; the filename `double_sharpening.html` is a spec placeholder. The authoritative source is `in-context-learning-of-decoding.html`.

---

## 3. Experiments A / B / C / E: Presence Map

### Experiment A: "Shape features vs. θ drift trajectory under decoder sweep"
**Status:** EXISTS — FULL RESULTS

- **Script:** `/home/cs29824/matthew/decoding-decoding/scripts/run_calibration_baseline.py` (lines 1–31)
  - Computes β̂_streaming (Laplace filter final state) vs. β̂_MLE on calibration set
  - Tests on held-out natural-text passages
  - Output: per-corpus β estimates, NLL curves, test PPL
- **Results dir:** `/home/cs29824/matthew/decoding-decoding/results/counter_decode/calibration_baseline/`
  - `summary.json` — auditable numbers (corpus-level pooled β_streaming, β_MLE, test PPL means)
  - `F_cal_*.png` — figures (beta estimators, NLL curve, test PPL)
  - `findings.pdf` — full writeup
  - **Corpora:** WikiText-103, WritingPrompts
  - **Headline:** WritingPrompts shows test-PPL drop; WikiText is null (see findings.pdf line 1)
- **Macro generation:** `scripts/build_calibration_baseline_macros.py` generates `macros.tex` for inline numbers

### Experiment B: "Imprint dimensionality — scalar vs. multi-dimensional"
**Status:** PARTIAL — Measured implicitly via residual floors

- **Direct measurement:** NOT run as standalone
- **Method:** Bellman-residual floor (the irreducible MSE of bootstrapped soft-Bellman recursion) measures how non-low-dimensional the imprint is
  - See `02_structured_twist_method.md` section "Residual floor as a measurement" (lines 89–92)
  - Spec says Experiment B is "answered from the other side" by fitting (section 41–44 of `00_README.md`)
- **Status:** Identified as a measurement framework, not yet executed

### Experiment C: "Recovery length M* after switching to default sampling"
**Status:** EXISTS — FULL RESULTS with closed-loop variants

- **Primary script:** `/home/cs29824/matthew/decoding-decoding/scripts/run_entropy_probe.py` (lines 1–50)
  - Generates at three β_dec settings (0.5, 1.0, 2.0), records pre-decoding entropy H(softmax(ℓ_t))
  - Hypothesis: entropy under sharp decoder should lower after the model conditions on its own generation
  - Tests model's belief formation about decoder
- **Results:** `/home/cs29824/matthew/decoding-decoding/results/counter_decode/entropy_probe/`
  - `summary.json` — per-setting entropy statistics
  - Figures in subdirectory

- **Belief-invariance variant:** `/home/cs29824/matthew/decoding-decoding/scripts/run_belief_invariance_test.py` (lines 1–50)
  - Cleaner metric: fix β_target=1, vary β_prefix ∈ {0.5, 1, 2} in warmup, measure CE invariance on natural continuation
  - Tests calibration of (γ, α, p) parameters via variance of cross-entropy across β_prefix
  - Results: `/home/cs29824/matthew/decoding-decoding/results/counter_decode/belief_invariance/`

- **Belief-decoding sweep:** `/home/cs29824/matthew/decoding-decoding/scripts/run_belief_decoding_sweep.py` (lines 1–107)
  - Sweeps (γ, α, p) grid: γ ∈ {0.85, 0.95, 1.00}, α ∈ {0.5, 1.0}, p ∈ {0, 0.25, 0.5, 0.75, 1.0}
  - β_target ∈ {0.5, 1.0, 2.0}, 32 prompts, 200 tokens/run
  - Success criterion: closed-loop stability (smooth monotonic entropy/max-P with β_target, no saturation)
  - Results: `/home/cs29824/matthew/decoding-decoding/data/counter_decode_belief_sweep/{g*_a*_p*}/` subdirs

### Experiment E: "Toy mixture-of-HMMs with exact DP"
**Status:** ABSENT — NOT YET IMPLEMENTED

- **Spec role** (per `02_structured_twist_method.md` lines 79–81, `00_README.md` lines 45):
  - One-time unit test: ground-truth F is computable by DP
  - Confirms log-linear-in-latent functional form is not fundamentally flawed
  - Does NOT calibrate LLM knob values (those must be fit on text)
- **Current status:** No toy HMM generator, no DP partition function, no ground-truth F data
- **Implications:** If implementing Fit B (twist map via soft-Bellman), Experiment E provides an existence proof but is deferred; method can be validated on LLM text alone

---

## 4. Model + Inference Stack

### Base Model
- **Model name:** `Qwen/Qwen2.5-3B` (hardcoded in multiple files)
- **Sources:**
  - `/home/cs29824/matthew/decoding-decoding/src/decoding_decoding/counter_decode_generate.py` line 85
  - `/home/cs29824/matthew/decoding-decoding/src/decoding_decoding/natural_text_filter.py` line 42

### Per-Token Log-Probabilities & Conditionals
- **Full-vocab logit access:** YES, via standard HF transformers `.logits` output (e.g., line 188 of `counter_decode_generate.py`)
- **Log-softmax:** `torch.log_softmax(logits, dim=-1)` computes full vocabulary conditionals
- **Logsumexp:** `torch.logsumexp()` available for stable reduction (line 405 of `counter_decode.py`)
- **Per-step stats from logits:**
  - E[ℓ] = Σ_j q[j] · ℓ[j] (line 157 of counter_decode.py)
  - Var[ℓ] = Σ_j q[j] · (ℓ[j] − E[ℓ])² (line 161)
  - Entropy H(q) computed inline where needed (natural_text_filter.py line 255)

### Sampling Under β
- **Function:** `sample_under_beta(logits, beta, ...)` in `counter_decode.py` (line 606+)
- **Supports:** full-vocab sampling with temperature scaling β · ℓ
- **Used in:** counter_decode_generate.py for the main generation loop

---

## 5. Project Tooling & Dependencies

### Build System
- **Package manager:** `uv` (Hatch backend)
- **Python:** >=3.11
- **pyproject.toml:** `/home/cs29824/matthew/decoding-decoding/pyproject.toml`

### Core Dependencies (Exact Versions)
```
torch==2.9.1
transformers (implicit, from HF model loading)
datasets>=4.8.5
numpy<2.3
scipy>=1.17.1
polars>=1.40.1
pyarrow>=24.0.0
matplotlib>=3.10.9
tqdm>=4.67.3
vllm==0.15.1 (installed but NOT used in counter_decode; see comment in counter_decode_generate.py)
```

### Testing & Build Macros
- **Test framework:** pytest (dev dependency)
- **Test dir:** Not visible in initial listing, but project structure supports pytest
- **Build-macros pattern:** YES, already in use
  - `/home/cs29824/matthew/decoding-decoding/scripts/build_calibration_baseline_macros.py` — generates `macros.tex` from `summary.json`
  - Reads auditable JSON (results), emits `\newcommand{}{}` TeX macros for paper inline numbers
  - This is the pattern to replicate for all results

### GPU/Device Setup
- **CUDA:** vLLM pinned to cu128 wheel (torch cu128 index) due to driver age (575.51.03 / CUDA 12.9)
- **Inference:** SDPA attention for efficiency (`attn_implementation="sdpa"` in model loading)

---

## 6. Reuse vs. Build: Implementation Checklist

### REUSE (Exists, Integration Needed)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Laplace filter core | `counter_decode.py:91–182` | Prod-ready | `laplace_update(state, logits, tokens, evidence_weight, memory_decay)` |
| Filter initialization | `counter_decode.py:64–88` | Prod-ready | `init_laplace(batch_size, sigma_0)` |
| Full-vocab log-softmax | `counter_decode_generate.py:188, 201` | Prod-ready | Used for E_q, Var_q, and twist reweighting |
| Logsumexp stable reduction | `counter_decode.py:405` | Prod-ready | `torch.logsumexp()` for log-domain arithmetic |
| Model + tokenizer loading | `natural_text_filter.py:461–474` | Prod-ready | Qwen 3B, float16 / SDPA, `model.eval()` |
| Corpus loaders (WikiText, WritingPrompts) | `natural_text_filter.py:80–165` | Prod-ready | Handles long-article filtering, tokenization |
| Experiment A results + macros | `scripts/build_calibration_baseline_macros.py` | Prod-ready | Per-corpus β̂_streaming, β̂_MLE, test PPL |
| Experiment C results + metric | `scripts/run_entropy_probe.py` + `run_belief_invariance_test.py` | Prod-ready | Entropy + cross-entropy invariance under decoder sweep |
| Parameter sweep framework | `scripts/run_belief_decoding_sweep.py` | Prod-ready | Grid over (γ, α, p), batched generation, per-config output dir |

### BUILD (Stub/Missing, Must Implement)

| Component | Spec Section | Where It's Needed | Implementation Notes |
|-----------|--------------|-------------------|----------------------|
| **Peek/Commit interface** | `03_pseudocode.md`:2, lines 53–64 | `filter.peek(state, observation) → new_state` (immutable) | Wrap Laplace update to return without mutation; enables O(V) candidate evaluation per step |
| **Twist map log_twist(θ)** | `02_structured_twist_method.md`:23–42 | Log-linear: `w^T φ(θ)` where φ is latent features | Start scalar latent: φ(θ) = [θ], w ∈ ℝ; expand if needed |
| **Sampling loop with candidate reweighting** | `03_pseudocode.md`:3, lines 73–102 | Main sampling: for each token v, peek state if v committed, evaluate twist, reweight | Tempered × exp(twist) / sum over vocab |
| **Fit A: Filter knob optimization** | `02_structured_twist_method.md`:54–62, `03_pseudocode.md`:4, lines 121–130 | Minimize MSE between filter readout(θ_t) and measured imprint drift (Experiment A data) | Input: Experiment A trajectories; output: (theta0, kappa0, eta, lam) |
| **Fit B: Soft-Bellman twist fit** | `02_structured_twist_method.md`:65–77, `03_pseudocode.md`:5, lines 137–167 | Minimize E[r_t²] where r_t = log F̃(θ_t) − logsumexp_v[...] | Iterate over text prefixes; stop-gradient on bootstrap target; no ground-truth F needed |
| **Experiment E: Toy HMM + DP** | `02_structured_twist_method.md`:79–81, `03_pseudocode.md`:6, lines 188–192 | Generate exact F by DP; validate functional form | Xie-style mixture-of-HMMs; implement DP partition function; confirm log-linear fit works |
| **Readout map (if needed)** | `02_structured_twist_method.md`:63 | If latent ≠ measured shape feature, small NN from θ to feature | Simple case: latent is decayed mean surprise, so readout is identity |

### Summary of Immediate Tasks

**To implement the method (3 + 2 fits):**

1. ✓ Laplace filter core exists
2. **MUST ADD:** Peek/commit wrapper for stateless evaluation
3. **MUST ADD:** Twist map class and sampling loop
4. **MUST ADD:** Fit A (filter knob optimization on Exp A data)
5. **MUST ADD:** Fit B (soft-Bellman for twist weights)
6. OPTIONAL: Fit on Exp E toy (validation only, deferred)

**Dependencies already satisfied:**
- Full-vocab log-softmax for reweighting
- Stable logsumexp
- Experiment A/C data and framework
- Parameter sweep infrastructure (Exp C)
- Build-macros pattern for paper

---

## 7. Model Internals & Imprint Indicators

### Imprint Mechanism (Established via Exps A & C)
- The Laplace filter's η̂ (log β estimate) **rises** when the model generates high-entropy text (wide distribution)
- The filter's η̂ **falls** when the model generates low-entropy text (sharp distribution)
- This is captured per-trajectory and per-corpus (Experiment A: Writingprompts shows effect; Wikitext is null)
- Recovery length M* (Experiment C) is how many tokens until η̂ returns to prior after a sharp-decoder prefix

### Latent Observability
- Observation fed to filter: **token surprise** o_t(v) = −log P(v | x_<t)
- Evidence weight α controls how much each surprise moves the posterior
- Memory decay γ controls exponential forgetting (0 = full reset, 1 = no decay)
- **Current best σ_0=0.2** per line 55 of `run_belief_decoding_sweep.py`

---

## Appendix: Directory Structure

```
/home/cs29824/matthew/decoding-decoding/
├── src/decoding_decoding/
│   ├── counter_decode.py              [Laplace filter + grid filter]
│   ├── natural_text_filter.py         [Filter diagnostics on WikiText/WP]
│   ├── counter_decode_generate.py     [Main sampling loop for Exps A,C]
│   ├── counter_decode_analyze.py      [Result analysis]
│   ├── generate.py, bootstrap.py      [Utilities]
├── scripts/
│   ├── run_calibration_baseline.py    [Experiment A: β̂_streaming vs β̂_MLE, test PPL]
│   ├── run_entropy_probe.py           [Experiment C: entropy under decoder sweep]
│   ├── run_belief_invariance_test.py  [Belief calibration variant]
│   ├── run_belief_decoding_sweep.py   [(γ,α,p) grid sweep, closed-loop]
│   ├── build_calibration_baseline_macros.py [A→TeX macros]
│   ├── plot_*.py, analyze_*.py        [Visualization + result assembly]
├── full-sequence/files/
│   ├── 00_README.md                   [This spec: overview & document map]
│   ├── 02_structured_twist_method.md  [Method design]
│   ├── 03_pseudocode.md               [Reference pseudocode]
│   ├── structured_twist.html          [HTML rendering of above]
├── in-context-learning-of-decoding.html [Original memo: Exps A,B content]
├── results/counter_decode/
│   ├── calibration_baseline/          [Experiment A results + summary.json, macros.tex]
│   ├── entropy_probe/                 [Experiment C: entropy trajectories]
│   ├── belief_invariance/             [Belief-invariance test results]
│   ├── belief_sweep/                  [Per-(γ,α,p) closed-loop runs]
│   └── ...
└── pyproject.toml                     [Dependencies, uv config, CUDA 12.8]
```

---

End of Inventory.
