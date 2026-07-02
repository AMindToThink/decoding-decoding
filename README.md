# decoding-decoding

Research code + LaTeX write-up on modeling how LLMs decode / in-context learning of
decoding. The paper is `decoding-modeling.tex` / `decoding-modeling.pdf` (and `paper/`).

## Environment
- Python ≥ 3.11 (pinned via `.python-version` = `3.11`), managed with
  [uv](https://docs.astral.sh/uv/).
- Exact dependency versions are locked in `uv.lock`.
- Recreate and run:
  ```bash
  uv sync
  uv run scripts/analyze_belief_decoding_sweep.py   # example
  ```

## Data
- Primary dataset: **`euclaise/writingprompts`** from the HuggingFace Hub, pulled
  automatically on first use via `datasets.load_dataset("euclaise/writingprompts")`
  (cached under `~/.cache/huggingface`). No manual download required.
- The local `data/` directory holds generated intermediates — regenerable by the
  scripts, not needed to start.

## Layout / entry points
- `src/decoding_decoding/` — the package.
- `scripts/` — parameter sweeps and analysis (`analyze_*.py`).
- Planning / theory notes: `REFRAME_PLAN.md`, `theory_options.md`,
  `notes-from-prof.md`, `long_context_capacity.md`.

---
> Reproducibility note: this README was added 2026-07-02 during a machine migration to
> capture setup and data provenance. Verify `uv sync` on a fresh checkout.
