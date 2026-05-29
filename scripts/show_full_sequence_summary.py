"""Print a go/no-go summary.json as compact, greppable one-line-per-metric output.

Each line is prefixed 'ROW|' so it survives noisy stdout relays via grep.
Usage: uv run python scripts/show_full_sequence_summary.py <summary.json>
"""
from __future__ import annotations

import json
import sys


def fmt_ci(ci):
    return f"[{ci[0]:+.4f},{ci[1]:+.4f},{ci[2]:+.4f}]"


def show(path: str) -> None:
    s = json.load(open(path))
    print(f"ROW|model={s['model']} n_records={s['n_records_total']} temps={s['temperatures']}")
    dec = s.get("decision_overall", {})
    print(f"ROW|ANY_GO={dec.get('ANY_GO')}")
    for scope in ["overall"] + [f"by_corpus.{c}" for c in s.get("by_corpus", {})]:
        if scope == "overall":
            block = s["overall"]
        else:
            block = s["by_corpus"][scope.split(".", 1)[1]]
        for T, m in block.items():
            print(
                f"ROW|{scope}|T={T}|npos={m['n_positions']}"
                f"|Dtot={m['D_total_nats']:.5f}|Dci={fmt_ci(m['D_total_ci'])}"
                f"|fBestT={m['frac_bestT']:+.4f}|fFiltLin={m['frac_filter_linear']:+.4f}"
                f"|fFiltCV={m['frac_filter_cv']:+.4f}|fLogitCV={m['frac_logit_cv']:+.4f}"
                f"|filtCV_minus_bestT={fmt_ci(m['frac_filter_cv_minus_bestT'])}"
                f"|logitCV_minus_bestT={fmt_ci(m['frac_logit_cv_minus_bestT'])}"
                f"|affRes={m['affine_residual_fraction_mean']:.4f}"
                f"|w_hat={m['w_hat']:+.4f}|cov={m['covered_mass_mean']:.4f}"
            )
        if scope == "overall":
            for T, d in dec.items():
                if T == "ANY_GO":
                    continue
                print(f"ROW|decision|T={T}|correction_real={d['correction_real']}"
                      f"|filter_go={d['filter_go']}|logit_go={d['logit_go']}|GO={d['GO']}"
                      f"|cond_better_latent={d['go_conditional_on_better_latent']}")


if __name__ == "__main__":
    show(sys.argv[1])
