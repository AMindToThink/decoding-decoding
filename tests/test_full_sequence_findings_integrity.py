"""Regression guard: every number in FINDINGS.md traces to a script.

Standing project discipline (CLAUDE.md "numbers-from-scripts"): no number in the
write-up may be hand-typed. The results/steelman tables are emitted verbatim by
``scripts/build_full_sequence_findings_tables.py``; the prose cites rounded ranges
that the same script computes from committed artifacts. This test asserts:

  1. the gpt2-large / Qwen results tables appear verbatim in FINDINGS.md,
  2. the steelman table appears verbatim (skipped if the gitignored records.jsonl
     is absent — a fresh checkout),
  3. every prose scalar (D_total range, affine_resid_frac, R^2_quad, Spearman)
     equals the script-computed value rounded as the prose rounds it, and
  4. previously hand-typed wrong values do not reappear.

The generator is loaded by file path (it lives in ``scripts/``, not the installed
package) and its ``ROOT`` is pinned to an absolute path so the test is independent
of pytest's working directory.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_PATH = REPO_ROOT / "scripts" / "build_full_sequence_findings_tables.py"
GO_NO_GO = REPO_ROOT / "results" / "full_sequence" / "go_no_go"
FINDINGS = GO_NO_GO / "FINDINGS.md"


def _load_generator():
    spec = importlib.util.spec_from_file_location("fs_findings_gen", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = GO_NO_GO  # pin to absolute path; module default is CWD-relative
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


@pytest.fixture(scope="module")
def findings_text():
    return FINDINGS.read_text()


def test_findings_exists(findings_text):
    assert findings_text.strip(), "FINDINGS.md is empty"


def test_gpt2_table_verbatim(gen, findings_text):
    table = gen.results_table(GO_NO_GO / "gpt2-large/summary.json")
    assert table in findings_text, (
        "gpt2-large results table in FINDINGS.md does not match the generator. "
        "Regenerate: uv run python scripts/build_full_sequence_findings_tables.py"
    )


def test_qwen_table_verbatim(gen, findings_text):
    table = gen.results_table(GO_NO_GO / "qwen2.5-3b/summary.json")
    assert table in findings_text, (
        "Qwen2.5-3B results table in FINDINGS.md does not match the generator. "
        "Regenerate: uv run python scripts/build_full_sequence_findings_tables.py"
    )


def test_steelman_table_verbatim(gen, findings_text):
    have_records = (
        (GO_NO_GO / "gpt2-large/records.jsonl").exists()
        and (GO_NO_GO / "qwen2.5-3b/records.jsonl").exists()
    )
    if not have_records:
        pytest.skip("records.jsonl (gitignored) absent — steelman table not checkable")
    table = gen.steelman_table()
    assert table in findings_text, (
        "Steelman table in FINDINGS.md does not match the generator. "
        "Regenerate: uv run python scripts/build_full_sequence_findings_tables.py"
    )


def test_prose_numbers_trace_to_script(gen, findings_text):
    """Each prose scalar equals the script-computed value, rounded as prose rounds."""
    p = gen.prose_stats()
    lo_d, hi_d = p["d_sizeable"]
    lo_a, hi_a = p["affine"]
    lo_r, hi_r = p["r2_quad"]

    # Diagnostic #1: "sizeable (0.14-0.51 nats)" — en-dash separator.
    assert f"{lo_d:.2f}–{hi_d:.2f} nats" in findings_text, p

    # Diagnostic #2: both models round to the same affine value -> "approx 0.96".
    assert lo_a == hi_a, f"affine range not single-valued at 2dp: {p['affine']}"
    assert f"affine_residual_fraction ≈ {hi_a:.2f}" in findings_text, p

    # Diagnostic #3: R^2_quad both round to the same value -> "approx 0.06".
    assert lo_r == hi_r, f"R2_quad range not single-valued at 2dp: {p['r2_quad']}"
    assert f"R² ≈ {hi_r:.2f}" in findings_text, p

    # Diagnostic #3: Spearman(c, logit) is near zero, justifying the prose "approx 0".
    assert p["spearman_abs_max"] < 0.1, p
    assert "Spearman(c_v, logit_v) ≈ 0" in findings_text

    # Discussion: "0.33 vs 0.51 nats" (Qwen vs gpt2-large at T=0.3); prose may
    # wrap the line before "T=0.3", so match only the load-bearing comparison.
    q03, g03 = p["d_T03_qwen_vs_gpt2"]
    assert f"{q03:.2f} vs {g03:.2f} nats" in findings_text, p

    # Abstract / table headers: n positions per model.
    assert f"{p['n_records']['gpt2-large']} positions" in findings_text, p
    assert f"{p['n_records']['qwen2.5-3b']} positions" in findings_text, p


def test_steelman_prose_numbers_trace_to_script(gen, findings_text):
    """Headroom and Spearman ranges in the steelman prose match the generator."""
    have_records = (
        (GO_NO_GO / "gpt2-large/records.jsonl").exists()
        and (GO_NO_GO / "qwen2.5-3b/records.jsonl").exists()
    )
    if not have_records:
        pytest.skip("records.jsonl (gitignored) absent — steelman prose not checkable")
    p = gen.steelman_prose_stats()
    lo_h, hi_h = p["headroom"]
    lo_s, hi_s = p["spearman"]
    # "(+0.20-0.25 of D ...)" — en-dash range; appears twice (diagnostic + recommendation).
    assert f"+{lo_h:.2f}–{hi_h:.2f} of D" in findings_text, p
    # "Spearman between -0.017 and -0.003" — most-negative .. least-negative.
    assert f"{lo_s:.3f} and {hi_s:.3f}".replace("-", "−") in findings_text, p


def test_no_stale_handtyped_values(findings_text):
    """Values that were once hand-typed wrong must not reappear."""
    # R^2_quad rounds to 0.06 (artifact 0.058-0.063); the old hand-typed "0.05" was wrong.
    assert "R² ≈ 0.05" not in findings_text
    # No leftover fabrication / drift markers from the build process.
    for marker in ("DRIFT", "FABRICAT", "TODO", "PLACEHOLDER"):
        assert marker not in findings_text, f"stale marker {marker!r} in FINDINGS.md"
