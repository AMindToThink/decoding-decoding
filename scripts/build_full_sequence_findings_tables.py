"""Emit the FINDINGS.md results tables directly from summary.json + records.

Numbers-from-scripts: the markdown tables in FINDINGS.md are produced here, never
hand-typed. Prints four blocks delimited by sentinels so they can be pasted
verbatim:

    <<<GPT2_TABLE>>> ... <<<END>>>
    <<<QWEN_TABLE>>> ... <<<END>>>
    <<<STEELMAN_TABLE>>> ... <<<END>>>
    <<<PROSE_STATS>>> ... <<<END>>>   (rounded ranges the prose cites)

The PROSE_STATS block is computed from committed artifacts only (summary.json +
perposition_scrutiny.txt), so the prose-number regression test does not depend on
the gitignored records.jsonl. The steelman table does need records.jsonl.

Usage: uv run python scripts/build_full_sequence_findings_tables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from decoding_decoding.full_sequence.metrics import kl_divergence, softmax

ROOT = Path("results/full_sequence/go_no_go")


def fmt_ci3(ci):
    return f"[{ci[0]:.4f}, {ci[2]:.4f}]"


def results_table(summary_path: Path) -> str:
    s = json.load(open(summary_path))["overall"]
    rows = [
        "| T | D_total (nats) [95% CI] | frac_bestT (per-pos oracle) | frac_filter_CV | frac_logit_CV | (filter_CV − bestT) [95% CI] | affine_resid_frac | GO |",
        "|---|--------------------------|------------------------------|----------------|---------------|------------------------------|-------------------|-----|",
    ]
    for T in sorted(s, key=float):
        m = s[T]
        fmb = m["frac_filter_cv_minus_bestT"]
        rows.append(
            f"| {T} | {m['D_total_nats']:.4f} {fmt_ci3(m['D_total_ci'])} "
            f"| {m['frac_bestT']:.3f} | {m['frac_filter_cv']:.3f} | {m['frac_logit_cv']:.3f} "
            f"| {fmb[1]:.3f} [{fmb[0]:.3f}, {fmb[2]:.3f}] "
            f"| {m['affine_residual_fraction_mean']:.3f} | No |"
        )
    return "\n".join(rows)


def steelman_row(records_path: Path, T: float) -> dict:
    recs = [json.loads(l) for l in open(records_path) if l.strip()]
    Tk = str(T)
    recs = [r for r in recs if Tk in r["c_by_T"]]
    betas = np.linspace(0.05, 8.0, 300)

    def kl_at(beta):
        tot = 0.0
        for r in recs:
            ell = np.array(r["ell"]); c = np.array(r["c_by_T"][Tk]); g = np.array(r["log_p"]) / T
            tot += kl_divergence(softmax(g + c), softmax(beta * ell))
        return tot / len(recs)

    gbest = betas[int(np.argmin([kl_at(b) for b in betas]))]
    bstar, proxy, klpt, klg, klpp = [], [], [], [], []
    for r in recs:
        ell = np.array(r["ell"]); c = np.array(r["c_by_T"][Tk]); g = np.array(r["log_p"]) / T
        q_g = softmax(g + c); klpt.append(kl_divergence(q_g, softmax(g)))
        kls = [kl_divergence(q_g, softmax(b * ell)) for b in betas]
        j = int(np.argmin(kls)); bstar.append(betas[j]); klpp.append(kls[j])
        klg.append(kl_divergence(q_g, softmax(gbest * ell))); proxy.append(float(np.mean(r["theta"])))
    D = float(np.mean(klpt)); bs = np.array(bstar)
    return {
        "frac_global": 1 - float(np.mean(klg)) / D,
        "frac_perpos": 1 - float(np.mean(klpp)) / D,
        "headroom": (float(np.mean(klg)) - float(np.mean(klpp))) / D,
        "spearman": float(stats.spearmanr(bs, proxy).correlation),
    }


def steelman_table() -> str:
    rows = [
        "| Model | T | global-T recovers | per-position-oracle-T recovers | adaptive-T headroom | Spearman(β*_t, filter prefix latent proxy) |",
        "|-------|---|-------------------|-------------------------------|---------------------|---------------------------------------------|",
    ]
    for label, recs in [
        ("gpt2-large", ROOT / "gpt2-large/records.jsonl"),
        ("Qwen2.5-3B", ROOT / "qwen2.5-3b/records.jsonl"),
    ]:
        for T in (0.5, 0.7):
            r = steelman_row(recs, T)
            rows.append(
                f"| {label} | {T} | {r['frac_global']:+.3f} | {r['frac_perpos']:+.3f} "
                f"| {r['headroom']:+.3f} | {r['spearman']:+.3f} |"
            )
    return "\n".join(rows)


def prose_stats() -> dict:
    """Rounded ranges the FINDINGS prose cites, from committed artifacts only.

    No records.jsonl dependency, so the prose regression test is robust to a
    fresh checkout. Returns the (lo, hi) ranges rounded exactly as the prose
    rounds them so a test can compare prose substrings to these.
    """
    summaries = [
        json.load(open(ROOT / "gpt2-large/summary.json"))["overall"],
        json.load(open(ROOT / "qwen2.5-3b/summary.json"))["overall"],
    ]
    # Diagnostic #1: D_total over the "sizeable" T in {0.3, 0.5}, both models.
    d_sizeable = [s[T]["D_total_nats"] for s in summaries for T in ("0.3", "0.5")]
    # Diagnostic #2: affine_residual_fraction over all T, both models.
    affine = [s[T]["affine_residual_fraction_mean"] for s in summaries for T in s]

    # Diagnostic #3: R2_quad + Spearman(c, logit) parsed from the scrutiny artifact.
    scrut = (ROOT / "perposition_scrutiny.txt").read_text()
    r2 = [float(m) for m in re.findall(r"R2_quad mean=([0-9.]+)", scrut)]
    rho = [
        float(m)
        for m in re.findall(r"per_pos_spearman\(c, logit\): mean=([+-][0-9.]+)", scrut)
    ]
    # Discussion: D_total at T=0.3, Qwen vs gpt2 ("0.33 vs 0.51 nats").
    d_qwen_03 = summaries[1]["0.3"]["D_total_nats"]
    d_gpt2_03 = summaries[0]["0.3"]["D_total_nats"]
    return {
        "d_sizeable": (round(min(d_sizeable), 2), round(max(d_sizeable), 2)),
        "affine": (round(min(affine), 2), round(max(affine), 2)),
        "r2_quad": (round(min(r2), 2), round(max(r2), 2)),
        "spearman_abs_max": round(max(abs(x) for x in rho), 3),
        "d_T03_qwen_vs_gpt2": (round(d_qwen_03, 2), round(d_gpt2_03, 2)),
        "n_records": {
            "gpt2-large": json.load(open(ROOT / "gpt2-large/summary.json"))["n_records_total"],
            "qwen2.5-3b": json.load(open(ROOT / "qwen2.5-3b/summary.json"))["n_records_total"],
        },
    }


def steelman_prose_stats() -> dict:
    """Rounded ranges the steelman prose cites. Needs records.jsonl (gitignored)."""
    rows = [
        steelman_row(ROOT / "gpt2-large/records.jsonl", T) for T in (0.5, 0.7)
    ] + [steelman_row(ROOT / "qwen2.5-3b/records.jsonl", T) for T in (0.5, 0.7)]
    head = [r["headroom"] for r in rows]
    rho = [r["spearman"] for r in rows]
    return {
        "headroom": (round(min(head), 2), round(max(head), 2)),
        # Spearman range as the prose writes it: most-negative .. least-negative, 3dp.
        "spearman": (round(min(rho), 3), round(max(rho), 3)),
    }


def prose_stats_block() -> str:
    p = prose_stats()
    return "\n".join(
        [
            f"D_total sizeable range (T in 0.3,0.5): {p['d_sizeable'][0]:.2f}-{p['d_sizeable'][1]:.2f} nats",
            f"affine_residual_fraction range: {p['affine'][0]:.2f}-{p['affine'][1]:.2f}",
            f"per-position quadratic-in-logit R2 range: {p['r2_quad'][0]:.2f}-{p['r2_quad'][1]:.2f}",
            f"max |Spearman(c, logit)|: {p['spearman_abs_max']:.3f}",
            f"D_total@T=0.3 Qwen vs gpt2: {p['d_T03_qwen_vs_gpt2'][0]:.2f} vs {p['d_T03_qwen_vs_gpt2'][1]:.2f} nats",
            f"n_records: gpt2-large={p['n_records']['gpt2-large']} qwen2.5-3b={p['n_records']['qwen2.5-3b']}",
        ]
    )


def steelman_prose_block() -> str:
    p = steelman_prose_stats()
    return "\n".join(
        [
            f"adaptive-T headroom range: +{p['headroom'][0]:.2f} to +{p['headroom'][1]:.2f} of D",
            f"steelman Spearman range: {p['spearman'][0]:+.3f} to {p['spearman'][1]:+.3f}",
        ]
    )


def main() -> None:
    print("<<<GPT2_TABLE>>>")
    print(results_table(ROOT / "gpt2-large/summary.json"))
    print("<<<END>>>")
    print("<<<QWEN_TABLE>>>")
    print(results_table(ROOT / "qwen2.5-3b/summary.json"))
    print("<<<END>>>")
    print("<<<STEELMAN_TABLE>>>")
    print(steelman_table())
    print("<<<END>>>")
    print("<<<PROSE_STATS>>>")
    print(prose_stats_block())
    print("<<<END>>>")
    print("<<<STEELMAN_PROSE>>>")
    print(steelman_prose_block())
    print("<<<END>>>")


if __name__ == "__main__":
    main()
