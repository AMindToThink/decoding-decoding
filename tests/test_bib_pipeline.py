"""Unit tests for the citation pipeline (build_bib + verify_cites).

Does not hit the network. build_bib's network resolvers are exercised via
a real run in CI or by hand; here we test the pure-logic pieces
(TOML parsing, manual-entry rendering, verify_cites regex) so regressions
in those layers fail fast without a live API.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
BUILD = SCRIPTS / "build_bib.py"
VERIFY = SCRIPTS / "verify_cites.py"


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


# ----------------------------------------------------------------------------
# verify_cites
# ----------------------------------------------------------------------------


def _write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def test_verify_happy_path(tmp_path: Path) -> None:
    tex = _write(tmp_path, "p.tex", r"""\cite{foo} and \citep{bar,baz}""")
    bib = _write(
        tmp_path,
        "refs.bib",
        """@misc{foo, title={A}}\n@article{bar, title={B}}\n@misc{baz, title={C}}\n""",
    )
    r = run(["uv", "run", "python", str(VERIFY), "--tex", str(tex), "--bib", str(bib)])
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_verify_missing_key_fails(tmp_path: Path) -> None:
    tex = _write(tmp_path, "p.tex", r"""\cite{known} and \cite{unknown}""")
    bib = _write(tmp_path, "refs.bib", """@misc{known, title={K}}\n""")
    r = run(["uv", "run", "python", str(VERIFY), "--tex", str(tex), "--bib", str(bib)])
    assert r.returncode == 2
    assert "unknown" in r.stderr
    # the resolved key should not be in the FAIL list
    assert "- known\n" not in r.stderr
    assert "1 \\cite key" in r.stderr


def test_verify_unused_key_fails(tmp_path: Path) -> None:
    tex = _write(tmp_path, "p.tex", r"""\cite{used}""")
    bib = _write(tmp_path, "refs.bib", """@misc{used}\n@misc{dead}\n""")
    r = run(["uv", "run", "python", str(VERIFY), "--tex", str(tex), "--bib", str(bib)])
    assert r.returncode == 3
    assert "dead" in r.stderr


def test_verify_unused_key_waivable(tmp_path: Path) -> None:
    tex = _write(tmp_path, "p.tex", r"""\cite{used}""")
    bib = _write(tmp_path, "refs.bib", """@misc{used}\n@misc{dead}\n""")
    r = run(
        [
            "uv", "run", "python", str(VERIFY),
            "--tex", str(tex),
            "--bib", str(bib),
            "--ignore-unused",
        ]
    )
    assert r.returncode == 0
    assert "dead" in r.stderr  # warning still printed
    assert "OK" in r.stdout


def test_verify_handles_natbib_variants(tmp_path: Path) -> None:
    tex = _write(
        tmp_path,
        "p.tex",
        r"""\citep{a}, \citet{b}, \citep[p.~5]{c}, \cite{d,e}, \citealp*{f}""",
    )
    bib = _write(
        tmp_path,
        "refs.bib",
        "\n".join(f"@misc{{{k}}}" for k in "abcdef") + "\n",
    )
    r = run(["uv", "run", "python", str(VERIFY), "--tex", str(tex), "--bib", str(bib)])
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_verify_ignores_commented_cites(tmp_path: Path) -> None:
    # Commented-out \cite lines should not be counted.
    tex = _write(tmp_path, "p.tex", "% \\cite{dead}\n\\cite{live}\n")
    bib = _write(tmp_path, "refs.bib", "@misc{live}\n")
    r = run(["uv", "run", "python", str(VERIFY), "--tex", str(tex), "--bib", str(bib)])
    assert r.returncode == 0, (r.stdout, r.stderr)


# ----------------------------------------------------------------------------
# build_bib
# ----------------------------------------------------------------------------


def test_build_offline_manual_only(tmp_path: Path) -> None:
    """--offline-manual-only should succeed on a TOML of only manual entries."""
    toml_path = tmp_path / "paper" / "refs_ids.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        """[[cite]]
key = "sample"
manual = true
entry = \"\"\"
@misc{sample, title={Sample}, author={Test Author}, year={2025}}
\"\"\"
claim = "A test claim"
""",
        encoding="utf-8",
    )
    # Monkey-copy build_bib to use this tmp project root.
    # Instead of monkeypatching, invoke the script with env overrides.
    env_root = tmp_path
    # The script resolves paths from its own location; to run against the
    # tmp TOML, we stage a symlink/copy of the script inside tmp.
    tmp_scripts = env_root / "scripts"
    tmp_scripts.mkdir(exist_ok=True)
    (tmp_scripts / "build_bib.py").write_text(BUILD.read_text(encoding="utf-8"), encoding="utf-8")
    r = run(["uv", "run", "python", str(tmp_scripts / "build_bib.py"), "--offline-manual-only"], cwd=env_root)
    assert r.returncode == 0, (r.stdout, r.stderr)
    bib = (env_root / "paper" / "refs.bib").read_text(encoding="utf-8")
    assert "@misc{sample" in bib
    assert "Test Author" in bib
    assert "% source: manual" in bib
    assert "% claim : A test claim" in bib


def test_build_rejects_duplicate_keys(tmp_path: Path) -> None:
    toml_path = tmp_path / "paper" / "refs_ids.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        """[[cite]]
key = "dup"
manual = true
entry = "@misc{dup}"

[[cite]]
key = "dup"
manual = true
entry = "@misc{dup}"
""",
        encoding="utf-8",
    )
    tmp_scripts = tmp_path / "scripts"
    tmp_scripts.mkdir(exist_ok=True)
    (tmp_scripts / "build_bib.py").write_text(BUILD.read_text(encoding="utf-8"), encoding="utf-8")
    r = run(["uv", "run", "python", str(tmp_scripts / "build_bib.py"), "--offline-manual-only"], cwd=tmp_path)
    assert r.returncode != 0
    assert "duplicate" in (r.stdout + r.stderr).lower()


def test_build_rejects_no_identifier(tmp_path: Path) -> None:
    toml_path = tmp_path / "paper" / "refs_ids.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        """[[cite]]
key = "orphan"
claim = "missing identifier"
""",
        encoding="utf-8",
    )
    tmp_scripts = tmp_path / "scripts"
    tmp_scripts.mkdir(exist_ok=True)
    (tmp_scripts / "build_bib.py").write_text(BUILD.read_text(encoding="utf-8"), encoding="utf-8")
    r = run(["uv", "run", "python", str(tmp_scripts / "build_bib.py"), "--offline-manual-only"], cwd=tmp_path)
    assert r.returncode != 0
    assert "no identifier" in (r.stdout + r.stderr).lower()


def test_build_manual_rekey(tmp_path: Path) -> None:
    """If a manual entry's inner @misc key differs from `key`, rewrite it."""
    toml_path = tmp_path / "paper" / "refs_ids.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        """[[cite]]
key = "newkey"
manual = true
entry = \"\"\"
@misc{oldkey, title={T}, author={A}, year={2025}}
\"\"\"
""",
        encoding="utf-8",
    )
    tmp_scripts = tmp_path / "scripts"
    tmp_scripts.mkdir(exist_ok=True)
    (tmp_scripts / "build_bib.py").write_text(BUILD.read_text(encoding="utf-8"), encoding="utf-8")
    r = run(["uv", "run", "python", str(tmp_scripts / "build_bib.py"), "--offline-manual-only"], cwd=tmp_path)
    assert r.returncode == 0, (r.stdout, r.stderr)
    bib = (tmp_path / "paper" / "refs.bib").read_text(encoding="utf-8")
    assert "@misc{newkey" in bib
    assert "oldkey" not in bib


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
