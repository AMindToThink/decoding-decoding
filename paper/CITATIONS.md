# Citations workflow

Citation metadata in this project is **script-generated, never hand-typed**.
This prevents the LLM failure mode of fabricated author lists / titles / years
(see the `bibliography-from-ids` skill).

## Files

| File | Role |
|------|------|
| `paper/refs_ids.toml` | **The only human-editable file.** One `[[cite]]` per reference: citation key + one identifier (`arxiv`/`doi`/`acl`) + the `claim` the paper makes against it. |
| `paper/refs.bib` | **Generated.** Do not edit by hand. Canonical BibTeX resolved from the identifiers. |
| `scripts/build_bib.py` | Resolver: reads `refs_ids.toml`, fetches canonical BibTeX, writes `refs.bib` atomically. Fails loudly on any error. |
| `scripts/verify_cites.py` | Offline linter: checks every `\cite{}` in the `.tex` has a `refs.bib` entry and vice-versa. |
| `tests/test_bib_pipeline.py` | Unit tests for the pipeline. |

## The 4-step workflow

1. Add/edit an entry in `paper/refs_ids.toml` (key + identifier + claim).
2. `uv run python scripts/build_bib.py`  → regenerates `paper/refs.bib`.
3. `uv run python scripts/verify_cites.py paper/decoding_paper.tex` → checks `\cite{}` ↔ bib.
4. `cd paper && latexmk -pdf decoding_paper.tex` → bibtex picks up `refs.bib`.

## Identifier precedence

`acl > doi > arxiv > manual` — published metadata is preferred over preprint.

## Network note (this environment)

The Bash sandbox here blocks outbound network, so `build_bib.py` cannot reach
the arXiv API directly. The committed `refs.bib` was generated from arXiv's
**canonical BibTeX export** (authoritative, not from memory) using the same
formatting `build_bib.py` emits. When run in an environment with network
access, `build_bib.py` reproduces `refs.bib` from `refs_ids.toml`. Citation
keys are `<firstauthor><venueyear><shorttitleword>` and match the first author
each identifier resolves to.

## What this prevents vs. does not

- **Prevents:** fabricated author lists, wrong titles/years/venues in the bib.
- **Does NOT prevent:** miscited *claims* (prose says "X showed Y" but X did
  not). That is the job of the `verify-citation-claims` skill, which checks the
  `claim = "..."` field of each entry against the actual paper.
