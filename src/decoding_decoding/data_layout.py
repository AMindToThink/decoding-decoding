"""Unified on-disk layout for all decoding-decoding experiments.

Goals:
  * Single source of truth for what runs exist (one parquet file).
  * Each trace has a single npz file holding ALL of its logprobs across the
    full current length, so extension is "load → append → save" with no chunk
    juggling.
  * Multiple experiment families (topk, blacklist, ...) coexist; the family
    name plus a JSON params blob describes the sampling condition.
  * Trace IDs are deterministic from the experimental coordinates, so the
    same (family, prompt_id, params, run_idx) always points at the same file.

Layout:
    data/
        config.json                # global run config (model, dtype, vllm version)
        manifest.parquet           # one row per trace
        traces/
            {trace_id}.npz         # top_token_ids (L, N) int32
                                   # top_logprobs  (L, N) float32

Manifest schema (one row per trace):
    trace_id: str (unique key)
    family: str ("topk" | "blacklist")
    prompt_id: int
    prompt_text: str
    condition_label: str           # human-readable, e.g. "k=3" or "blacklist_rank1"
    params_json: str               # JSON of the SamplingParams-relevant params
    run_idx: int
    seed: int
    length: int                    # current number of generated tokens
    sampled_token_ids: list[int]   # length == length, the tokens actually sampled
    decoded_text: str              # detokenized current full output
    trace_path: str                # relative to the data dir, e.g. "traces/topk_p00_k3_r00.npz"

Per-trace npz schema:
    top_token_ids: (length, top_n) int32
    top_logprobs:  (length, top_n) float32
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


MANIFEST_FILENAME = "manifest.parquet"
TRACES_SUBDIR = "traces"
CONFIG_FILENAME = "config.json"


def topk_trace_id(prompt_id: int, condition_k: str, run_idx: int) -> str:
    """Deterministic trace_id for the top-k family."""
    return f"topk_p{prompt_id:02d}_k{condition_k}_r{run_idx:02d}"


def blacklist_trace_id(prompt_id: int, banned_rank: int, run_idx: int) -> str:
    """Deterministic trace_id for the blacklist family.

    `banned_rank` is the global frequency rank (1..5) of the banned token in
    the unrestricted-baseline data — the *condition* of the experiment, not
    the token id, so multiple datasets can interpret it consistently.
    """
    return f"blacklist_p{prompt_id:02d}_b{banned_rank:01d}_r{run_idx:02d}"


def trace_relpath(trace_id: str) -> str:
    return f"{TRACES_SUBDIR}/{trace_id}.npz"


# ---- Manifest schema (used to coerce when writing) ---------------------- #

MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "trace_id": pl.Utf8,
    "family": pl.Utf8,
    "prompt_id": pl.Int32,
    "prompt_text": pl.Utf8,
    "condition_label": pl.Utf8,
    "params_json": pl.Utf8,
    "run_idx": pl.Int32,
    "seed": pl.Int64,
    "length": pl.Int32,
    "sampled_token_ids": pl.List(pl.Int64),
    "decoded_text": pl.Utf8,
    "trace_path": pl.Utf8,
}


def load_manifest(data_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(Path(data_dir) / MANIFEST_FILENAME)


def write_manifest(data_dir: Path, df: pl.DataFrame) -> None:
    """Write manifest atomically (write to .tmp, then rename)."""
    data_dir = Path(data_dir)
    df = df.cast(MANIFEST_SCHEMA, strict=True).sort("trace_id")
    out = data_dir / MANIFEST_FILENAME
    tmp = out.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(out)


def upsert_manifest(data_dir: Path, new_rows: list[dict[str, Any]]) -> None:
    """Insert or update rows in the manifest, keyed on trace_id.

    For an existing trace_id, the new row replaces the old (length / decoded
    text typically grew). For a new trace_id, it's appended.
    """
    data_dir = Path(data_dir)
    new_df = pl.DataFrame(new_rows).cast(MANIFEST_SCHEMA, strict=True)
    path = data_dir / MANIFEST_FILENAME
    if path.exists():
        existing = pl.read_parquet(path)
        new_ids = set(new_df["trace_id"].to_list())
        keep = existing.filter(~pl.col("trace_id").is_in(list(new_ids)))
        merged = pl.concat([keep, new_df], how="diagonal_relaxed").cast(
            MANIFEST_SCHEMA, strict=True
        )
    else:
        merged = new_df
    write_manifest(data_dir, merged)


# ---- Trace I/O ---------------------------------------------------------- #


def trace_path_for(data_dir: Path, trace_id: str) -> Path:
    return Path(data_dir) / trace_relpath(trace_id)


def load_trace(data_dir: Path, trace_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (top_token_ids, top_logprobs), both shape (length, top_n)."""
    npz = np.load(trace_path_for(data_dir, trace_id))
    return np.asarray(npz["top_token_ids"]), np.asarray(npz["top_logprobs"])


def save_trace(
    data_dir: Path,
    trace_id: str,
    top_token_ids: np.ndarray,
    top_logprobs: np.ndarray,
) -> None:
    """Atomically write a trace npz.

    Writes to {trace_id}.tmp.npz then renames over {trace_id}.npz so the file
    is either the old version or the new version, never partial.
    """
    p = trace_path_for(data_dir, trace_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f"{p.stem}.tmp.npz"  # keep .npz extension so np.savez doesn't append another
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            top_token_ids=top_token_ids.astype(np.int32),
            top_logprobs=top_logprobs.astype(np.float32),
        )
    tmp.replace(p)


def append_to_trace(
    data_dir: Path,
    trace_id: str,
    new_token_ids: np.ndarray,
    new_logprobs: np.ndarray,
) -> int:
    """Concatenate new step rows onto an existing trace. Returns new length."""
    old_ids, old_lps = load_trace(data_dir, trace_id)
    cat_ids = np.concatenate([old_ids, new_token_ids.astype(np.int32)], axis=0)
    cat_lps = np.concatenate([old_lps, new_logprobs.astype(np.float32)], axis=0)
    save_trace(data_dir, trace_id, cat_ids, cat_lps)
    return cat_ids.shape[0]


# ---- Params helpers ---------------------------------------------------- #


def topk_params(condition_k: str) -> dict[str, Any]:
    """SamplingParams-relevant params for the top-k family."""
    top_k = -1 if condition_k == "inf" else int(condition_k)
    return {"top_k": top_k}


def blacklist_params(banned_rank: int, banned_token_id: int) -> dict[str, Any]:
    """SamplingParams-relevant params for the blacklist family."""
    return {
        "top_k": -1,
        "banned_rank": int(banned_rank),
        "banned_token_id": int(banned_token_id),
    }


def encode_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def decode_params(params_json: str) -> dict[str, Any]:
    return json.loads(params_json)
