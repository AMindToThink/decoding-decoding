r"""Regression guards for the compiled tech report's imported numbers.

Enforces the "inline numbers come from scripts, never from memory" discipline:

  1. Every data-macro the paper references is defined in the generated
     paper_macros.tex.
  2. No script-generated result value appears hand-typed in the paper prose
     (it must be cited via its macro; the literal lives only in the generated
     macro/table files, never in decoding_paper.tex).
  3. Every \input'd generated artifact (macros + table bodies) exists.

Run:  uv run pytest tests/test_paper_macros.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "decoding_paper.tex"
MACROS = ROOT / "results" / "tables" / "paper_macros.tex"
TABLES_DIR = ROOT / "results" / "tables"

# Prefixes of macros that carry experimental data (vs. ordinary LaTeX/package
# control sequences). A used control sequence with one of these prefixes MUST
# resolve to a definition in paper_macros.tex.
DATA_PREFIXES = (
    "binv", "bl", "bsweep", "expb", "expc", "fzero", "tk",
    "cal", "logBetaFinal", "paired", "sigmaZero", "priorMode",
    "meanEntropy", "totalFisher", "nPassages",
)

NEWCMD_RE = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
USE_RE = re.compile(r"\\([A-Za-z]+)")


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        # cut at first unescaped %
        m = re.search(r"(?<!\\)%", line)
        out.append(line if m is None else line[: m.start()])
    return "\n".join(out)


def defined_macros() -> dict[str, str]:
    d: dict[str, str] = {}
    for line in MACROS.read_text().splitlines():
        m = NEWCMD_RE.match(line)
        if m:
            d[m.group(1)] = m.group(2)
    return d


def paper_text() -> str:
    return strip_comments(PAPER.read_text())


def test_macros_file_nonempty():
    d = defined_macros()
    assert len(d) > 50, f"expected many generated macros, got {len(d)}"


def test_referenced_data_macros_are_defined():
    """Every \\fzero*/\\cal*/... used in the paper is defined."""
    defined = set(defined_macros())
    used = set(USE_RE.findall(paper_text()))
    referenced_data = {
        name for name in used if name.startswith(DATA_PREFIXES)
    }
    missing = sorted(referenced_data - defined)
    assert not missing, f"data-macros used but not defined: {missing}"


SELFCAL_RE = re.compile(r"^sc[A-Z]")


def test_referenced_selfcal_macros_are_defined():
    """Every self-calibration macro (\\sc<Upper>...) used in the paper is defined."""
    defined = set(defined_macros())
    used = set(USE_RE.findall(paper_text()))
    referenced = {n for n in used if SELFCAL_RE.match(n)}
    assert referenced, "expected the paper to reference self-calibration (sc*) macros"
    missing = sorted(referenced - defined)
    assert not missing, f"self-calibration macros used but not defined: {missing}"


def test_all_macro_names_are_letter_only():
    """TeX control-sequence names must be letters only (no digits)."""
    bad = [n for n in defined_macros() if not n.isalpha()]
    assert not bad, f"macro names with non-letters break TeX parsing: {bad}"


def test_no_handtyped_result_numbers_in_prose():
    """A generated result value must not appear literally in the paper .tex.

    Forbidden = macro values that are 'distinctive' numbers (contain a decimal
    point and are at least 4 chars after stripping sign). This excludes config
    integers (200, 32) and the short beta/temperature literals (0.5, 1.0, 2.0)
    that legitimately appear as experimental settings.
    """
    # Decimals that are legitimate experimental *settings* written in prose
    # (the memory-decay grid), which happen to collide with some result value's
    # numeric core (e.g. the t-stat -0.95 vs. the config gamma=0.95).
    config_allow = {"0.90", "0.95", "0.98", "0.99"}
    text = paper_text()
    forbidden: dict[str, str] = {}  # numeric core -> macro name
    for name, val in defined_macros().items():
        core = val.lstrip("+-")
        if "." in core and len(core) >= 4 and core not in config_allow:
            forbidden[core] = name
    offenders = []
    for core, name in forbidden.items():
        # match the number as a standalone token (not part of a longer number)
        if re.search(r"(?<![\d.])" + re.escape(core) + r"(?![\d])", text):
            offenders.append(f"{core} (should be \\{name})")
    assert not offenders, (
        "hand-typed result numbers found in paper prose; cite the macro "
        f"instead: {sorted(offenders)}"
    )


def test_inputs_exist():
    """Every \\input{...} target referenced by the paper exists on disk."""
    text = paper_text()
    paper_dir = PAPER.parent
    missing = []
    for rel in re.findall(r"\\input\{([^}]+)\}", text):
        # resolve relative to the paper directory; \tabdir expands to a path
        rel = rel.replace("\\tabdir", "../results/tables")
        if not rel.endswith(".tex"):
            rel += ".tex"
        if not (paper_dir / rel).resolve().is_file():
            missing.append(rel)
    assert not missing, f"\\input targets missing: {missing}"


def test_table_bodies_present():
    expected = [
        "tab_f0_concentration", "tab_f12_drift", "tab_f3_closedloop",
        "tab_blacklist", "tab_topk", "tab_expb_genre", "tab_expc_late",
        "tab_expc_swap", "tab_bsweep_closed", "tab_binv_invariance",
    ]
    missing = [t for t in expected if not (TABLES_DIR / f"{t}.tex").is_file()]
    assert not missing, f"generated table bodies missing: {missing}"
