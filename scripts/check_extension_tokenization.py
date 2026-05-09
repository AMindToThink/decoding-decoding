"""Sanity-check: confirm that tokenizer.encode(prompt_text, add_special_tokens=False)
matches the prompt tokens vLLM uses internally for the original generation,
i.e. that the extension's prefix == prompt + sampled has the right length and
no surprise BOS injection.

Loads a sample trace, prints prompt token count, sampled token count, and
the first/last few tokens of the (prompt + sampled) sequence.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from decoding_decoding.data_layout import MANIFEST_FILENAME
from decoding_decoding.generate import MODEL_NAME


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    data_dir = Path("data")
    manifest = pl.read_parquet(data_dir / MANIFEST_FILENAME)
    row = manifest.filter(pl.col("trace_id") == "topk_p00_kinf_r00").row(0, named=True)
    prompt_text = row["prompt_text"]
    sampled = list(row["sampled_token_ids"])

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"prompt text repr: {prompt_text[:60]!r}")
    print(f"prompt tokens (no special): {len(prompt_ids)}")
    print(f"sampled tokens: {len(sampled)}")
    print(f"prompt token ids[:10]: {prompt_ids[:10]}")
    print(f"sampled[0:5]: {sampled[:5]}")
    print(f"decoded(prompt_ids + sampled[:5]): "
          f"{tokenizer.decode(prompt_ids + sampled[:5], skip_special_tokens=False)!r}")

    # Also: the prompt's text decoded back from prompt_ids must match:
    re_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    if re_decoded != prompt_text:
        print(f"WARNING: round-trip mismatch")
        print(f"  original:   {prompt_text!r}")
        print(f"  re-decoded: {re_decoded!r}")
    else:
        print("OK: prompt round-trip is exact.")


if __name__ == "__main__":
    main()
