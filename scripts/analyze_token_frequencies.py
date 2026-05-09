"""Compute the global frequency rank of every sampled token in the
top-k=∞ baseline, and emit:
    results/tables/token_frequencies.json   — full ranking (top 50)
                                               + provenance + tokenizer info.
    results/tables/banned_tokens.json       — top-N banned-token specs ready
                                               for the blacklist experiment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

import polars as pl

from decoding_decoding.data_layout import MANIFEST_FILENAME
from decoding_decoding.generate import MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "tables",
    )
    parser.add_argument("--n-banned", type=int, default=5,
                        help="how many top-frequent tokens to include in banned_tokens.json")
    parser.add_argument("--top-show", type=int, default=50,
                        help="how many ranks to record in the full token_frequencies.json")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pl.read_parquet(args.data_dir / MANIFEST_FILENAME)
    inf_rows = manifest.filter(
        (pl.col("family") == "topk") & (pl.col("condition_label") == "k=inf")
    )
    if len(inf_rows) == 0:
        raise RuntimeError("no top-k=inf traces found in manifest; can't rank tokens")

    counter: Counter[int] = Counter()
    n_tokens = 0
    for sampled in inf_rows["sampled_token_ids"].to_list():
        counter.update(int(t) for t in sampled)
        n_tokens += len(sampled)
    print(f"tallied {n_tokens:,} sampled tokens across {len(inf_rows)} k=inf traces")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ranked = counter.most_common(args.top_show)
    rows: list[dict] = []
    for rank, (tid, count) in enumerate(ranked, start=1):
        decoded = tokenizer.decode([tid], skip_special_tokens=False)
        rows.append({
            "rank": rank,
            "token_id": int(tid),
            "count": int(count),
            "fraction": float(count) / n_tokens,
            "decoded": decoded,
        })

    full_payload = {
        "provenance": {
            "script": "scripts/analyze_token_frequencies.py",
            "argv": sys.argv,
            "generated_at": dt.datetime.now().isoformat(),
            "data_dir": str(args.data_dir),
            "n_traces": int(len(inf_rows)),
            "n_tokens_total": int(n_tokens),
            "model": MODEL_NAME,
        },
        "ranking": rows,
    }
    full_path = args.out_dir / "token_frequencies.json"
    full_path.write_text(json.dumps(full_payload, indent=2))
    print(f"wrote {full_path}")

    banned_payload = {
        "provenance": full_payload["provenance"]
        | {"script_purpose": "Top-N most-frequent token IDs to blacklist"},
        "banned_token_ids": [int(r["token_id"]) for r in rows[: args.n_banned]],
        "banned_token_meta": rows[: args.n_banned],
    }
    banned_path = args.out_dir / "banned_tokens.json"
    banned_path.write_text(json.dumps(banned_payload, indent=2))
    print(f"wrote {banned_path}")
    print("Top banned tokens:")
    for r in rows[: args.n_banned]:
        print(f"  rank {r['rank']:>2}: id={r['token_id']:>6}, count={r['count']:>6}, "
              f"frac={r['fraction']:.4f}, decoded={r['decoded']!r}")


if __name__ == "__main__":
    main()
