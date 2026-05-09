"""One-shot migration from the legacy data layout to the new unified layout.

LEGACY:
    data/run_metadata.parquet                 (cols: prompt_id, prompt_text,
                                               condition_k, run_idx, seed,
                                               token_ids, decoded_text, logprobs_npz)
    data/logprobs/{pid:02d}_k{K}_r{run:02d}.npz

NEW:
    data/manifest.parquet                     (data_layout.MANIFEST_SCHEMA)
    data/traces/{trace_id}.npz

We rewrite the manifest and rename each .npz into the new traces/ directory.
The npz internal contents are unchanged, so this is a metadata migration.

Idempotent: if data/manifest.parquet already exists with the new schema, exits
with no-op.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl

from decoding_decoding.data_layout import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    TRACES_SUBDIR,
    encode_params,
    topk_params,
    topk_trace_id,
    trace_relpath,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    legacy_meta_path = data_dir / "run_metadata.parquet"
    legacy_logprobs_dir = data_dir / "logprobs"
    new_traces_dir = data_dir / TRACES_SUBDIR
    new_manifest_path = data_dir / MANIFEST_FILENAME

    if new_manifest_path.exists():
        existing = pl.read_parquet(new_manifest_path)
        # Quick schema sanity: must contain the required keys.
        required = set(MANIFEST_SCHEMA.keys())
        if required.issubset(set(existing.columns)):
            print(f"manifest already in new schema ({len(existing)} rows). no-op.")
            return
        else:
            raise RuntimeError(
                f"{new_manifest_path} exists but missing keys: "
                f"{required - set(existing.columns)}"
            )

    if not legacy_meta_path.exists():
        raise FileNotFoundError(f"no legacy {legacy_meta_path}; nothing to migrate")
    if not legacy_logprobs_dir.exists():
        raise FileNotFoundError(f"no legacy {legacy_logprobs_dir}; nothing to migrate")

    print(f"reading legacy {legacy_meta_path}")
    legacy = pl.read_parquet(legacy_meta_path)
    print(f"  {len(legacy)} legacy rows; cols: {legacy.columns}")

    new_traces_dir.mkdir(parents=True, exist_ok=True)

    new_rows: list[dict] = []
    for row in legacy.iter_rows(named=True):
        prompt_id = int(row["prompt_id"])
        condition_k = str(row["condition_k"])
        run_idx = int(row["run_idx"])

        trace_id = topk_trace_id(prompt_id, condition_k, run_idx)
        new_rel = trace_relpath(trace_id)
        new_path = data_dir / new_rel
        old_path = legacy_logprobs_dir / row["logprobs_npz"]

        if not old_path.exists():
            raise FileNotFoundError(f"missing legacy file: {old_path}")

        if not args.dry_run and not new_path.exists():
            shutil.copy2(old_path, new_path)

        token_ids = list(row["token_ids"])
        new_rows.append(
            {
                "trace_id": trace_id,
                "family": "topk",
                "prompt_id": prompt_id,
                "prompt_text": row["prompt_text"],
                "condition_label": f"k={condition_k}",
                "params_json": encode_params(topk_params(condition_k)),
                "run_idx": run_idx,
                "seed": int(row["seed"]),
                "length": len(token_ids),
                "sampled_token_ids": token_ids,
                "decoded_text": row["decoded_text"],
                "trace_path": new_rel,
            }
        )

    new_df = pl.DataFrame(new_rows)
    print(f"  built manifest: {len(new_df)} rows")

    if args.dry_run:
        print(new_df.head(3))
        print("dry-run; not writing manifest or removing legacy files.")
        return

    write_manifest(data_dir, new_df)
    print(f"wrote {new_manifest_path}")

    # Verify counts match before removing legacy.
    n_npz_new = sum(1 for _ in new_traces_dir.glob("topk_*.npz"))
    if n_npz_new != len(new_df):
        raise RuntimeError(f"copied {n_npz_new} traces but manifest has {len(new_df)}")

    legacy_meta_path.rename(legacy_meta_path.with_suffix(".legacy.parquet"))
    print(f"renamed legacy metadata to {legacy_meta_path.with_suffix('.legacy.parquet')}")
    print(
        f"NOTE: legacy {legacy_logprobs_dir} kept as a backup; remove it manually "
        f"once you've confirmed the new layout works end-to-end."
    )


if __name__ == "__main__":
    main()
