"""Run the blacklist experiment.

Reads results/tables/banned_tokens.json (produced by
scripts/analyze_token_frequencies.py) for the list of banned token IDs.

Usage:
    uv run python scripts/run_blacklist_experiment.py [--max-tokens 1024]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--banned-tokens-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "tables"
        / "banned_tokens.json",
    )
    args = parser.parse_args()

    if not args.banned_tokens_json.exists():
        raise FileNotFoundError(
            f"missing {args.banned_tokens_json}; run analyze_token_frequencies.py first"
        )
    payload = json.loads(args.banned_tokens_json.read_text())
    banned_ids: list[int] = payload["banned_token_ids"]
    print(f"banning {len(banned_ids)} token IDs from {args.banned_tokens_json}:")
    for r in payload["banned_token_meta"]:
        print(f"  rank {r['rank']}: id={r['token_id']}, decoded={r['decoded']!r}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from decoding_decoding.generate import run_blacklist_experiment
    from decoding_decoding.logging_utils import start_logging

    start_logging(args.data_dir / "logs", f"blacklist_L{args.max_tokens}")

    t0 = time.time()
    run_blacklist_experiment(
        args.data_dir, max_tokens=args.max_tokens, banned_token_ids=banned_ids
    )
    elapsed = time.time() - t0
    print(f"Total wall clock: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
