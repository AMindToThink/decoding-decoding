"""vLLM generation harness for the decoding-decoding experiments.

Two entry points:

    run_topk_experiment(out_dir, max_tokens)
        Fresh-start top-k experiment over (prompts × {1,3,5,inf} × runs).

    extend_topk_experiment(out_dir, target_length)
        Extend every existing topk trace (current length L0) to `target_length`
        by generating (target_length - L0) more tokens from the saved prefix.

    run_blacklist_experiment(out_dir, max_tokens, banned_token_ids)
        Generate (prompts × banned-token × runs) at top-k=inf, T=1, with the
        banned token's logit driven to a large negative value via logit_bias.
        Saves under the same data/manifest.parquet + data/traces/{trace_id}.npz
        layout as the topk family, with family="blacklist".

All families write into the same unified layout described in `data_layout.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from decoding_decoding.data_layout import (
    MANIFEST_FILENAME,
    append_to_trace,
    blacklist_params,
    blacklist_trace_id,
    encode_params,
    save_trace,
    topk_params,
    topk_trace_id,
    trace_relpath,
    upsert_manifest,
)
from decoding_decoding.prompts import PROMPTS


MODEL_NAME = "Qwen/Qwen2.5-3B"
TOP_LOGPROBS = 200
TEMPERATURE = 1.0
SEED_BASE = 42

# Conditions and their replication counts for the topk family.
# K=1 at T=1 is deterministic, so a single run per prompt suffices.
CONDITIONS: tuple[str, ...] = ("1", "3", "5", "inf")
RUNS_PER_CONDITION: dict[str, int] = {"1": 1, "3": 8, "5": 8, "inf": 8}

# Blacklist family.
BLACKLIST_RUNS: int = 8
# Logit bias used to "blacklist" a token. vLLM's logit_bias is added to the
# logit before softmax; -1e9 effectively zeros the resulting probability.
BLACKLIST_LOGIT_BIAS: float = -1e9
# Seed offset so the blacklist family doesn't collide with the topk family.
BLACKLIST_SEED_OFFSET: int = 100_000
# Seed offset for the L0→L1 extension of the topk family.
TOPK_EXTEND_SEED_OFFSET: int = 200_000


# ============================================================ topk family ===


@dataclass(frozen=True)
class TopkSpec:
    prompt_id: int
    prompt: str
    condition_k: str
    run_idx: int
    seed: int

    @property
    def trace_id(self) -> str:
        return topk_trace_id(self.prompt_id, self.condition_k, self.run_idx)


def _build_topk_specs(prompts: Sequence[str] = PROMPTS) -> list[TopkSpec]:
    specs: list[TopkSpec] = []
    for pid, prompt in enumerate(prompts):
        for k in CONDITIONS:
            for r in range(RUNS_PER_CONDITION[k]):
                specs.append(
                    TopkSpec(
                        prompt_id=pid,
                        prompt=prompt,
                        condition_k=k,
                        run_idx=r,
                        seed=SEED_BASE + pid * 1000 + r,
                    )
                )
    return specs


def _topk_int(condition_k: str) -> int:
    """vLLM convention: top_k=-1 disables top-k filtering."""
    return -1 if condition_k == "inf" else int(condition_k)


def _build_topk_sampling_params(spec: TopkSpec, max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=_topk_int(spec.condition_k),
        seed=spec.seed,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        logprobs=TOP_LOGPROBS,
        ignore_eos=True,
        detokenize=True,
    )


# ============================================================ logprob extraction


def extract_top_logprobs_array(
    step_logprobs: list,
    top_n: int = TOP_LOGPROBS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert vLLM's per-step logprob dicts to (L, top_n) arrays.

    Each element of `step_logprobs` is a dict {token_id: Logprob} (or None for
    steps that didn't request logprobs). We sort by `Logprob.logprob` descending
    and keep the top `top_n` entries — this guards against vLLM including an
    "off-list" sampled token that landed outside the top-N.
    """
    L = len(step_logprobs)
    token_ids = np.full((L, top_n), -1, dtype=np.int32)
    logprobs = np.full((L, top_n), -np.inf, dtype=np.float32)

    for i, step_dict in enumerate(step_logprobs):
        if step_dict is None:
            raise ValueError(f"step {i}: logprobs is None — was logprobs= requested?")
        items = sorted(step_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)
        items = items[:top_n]
        if len(items) < top_n:
            raise ValueError(
                f"step {i}: vLLM returned only {len(items)} logprobs, expected {top_n}"
            )
        for j, (tok_id, lp_obj) in enumerate(items):
            token_ids[i, j] = int(tok_id)
            logprobs[i, j] = float(lp_obj.logprob)
    return token_ids, logprobs


# ============================================================ vLLM lifecycle


def _make_llm(max_model_len: int):
    """Standard vLLM constructor used by all entry points in this module."""
    from vllm import LLM

    return LLM(
        model=MODEL_NAME,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        logprobs_mode="raw_logprobs",
        enforce_eager=True,
        max_model_len=max_model_len,
        seed=SEED_BASE,
        attention_config={"backend": "TRITON_ATTN"},
        max_logprobs=TOP_LOGPROBS,
    )


def _write_config(out_dir: Path, max_tokens: int, **extra) -> None:
    import vllm

    config = {
        "model": MODEL_NAME,
        "vllm_version": vllm.__version__,
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "max_tokens": max_tokens,
        "top_logprobs": TOP_LOGPROBS,
        "temperature": TEMPERATURE,
        "seed_base": SEED_BASE,
        "n_prompts": len(PROMPTS),
        "prompts_sha256": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
        **extra,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


# ============================================================ entry: fresh topk run


def run_topk_experiment(out_dir: Path, max_tokens: int) -> None:
    """End-to-end fresh top-k experiment. Writes manifest + traces."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    llm = _make_llm(max_model_len=max_tokens + 256)

    specs = _build_topk_specs()
    prompts = [s.prompt for s in specs]
    sampling_params = [_build_topk_sampling_params(s, max_tokens) for s in specs]

    print(f"Generating {len(specs)} sequences × {max_tokens} tokens...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    if len(outputs) != len(specs):
        raise RuntimeError(f"got {len(outputs)} outputs for {len(specs)} specs")

    new_rows: list[dict] = []
    for spec, output in zip(specs, outputs, strict=True):
        out0 = output.outputs[0]
        if len(out0.token_ids) != max_tokens:
            raise RuntimeError(
                f"spec {spec}: expected {max_tokens} generated tokens, got {len(out0.token_ids)}"
            )
        top_ids, top_lps = extract_top_logprobs_array(out0.logprobs, top_n=TOP_LOGPROBS)
        save_trace(out_dir, spec.trace_id, top_ids, top_lps)

        new_rows.append(
            {
                "trace_id": spec.trace_id,
                "family": "topk",
                "prompt_id": spec.prompt_id,
                "prompt_text": spec.prompt,
                "condition_label": f"k={spec.condition_k}",
                "params_json": encode_params(topk_params(spec.condition_k)),
                "run_idx": spec.run_idx,
                "seed": spec.seed,
                "length": max_tokens,
                "sampled_token_ids": list(out0.token_ids),
                "decoded_text": out0.text,
                "trace_path": trace_relpath(spec.trace_id),
            }
        )

    upsert_manifest(out_dir, new_rows)
    _write_config(
        out_dir,
        max_tokens=max_tokens,
        family="topk",
        conditions=list(CONDITIONS),
        runs_per_condition=dict(RUNS_PER_CONDITION),
    )
    print(f"Wrote {len(new_rows)} traces to {out_dir}")


# ============================================================ entry: extend topk


def _ensure_tokenizer():
    """Load the tokenizer separately (faster than spinning up vLLM just for this)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


def extend_topk_experiment(out_dir: Path, target_length: int) -> None:
    """Extend every topk trace from current `length` to `target_length`.

    For each trace whose current length L0 < target_length, we feed the model
    the original prompt + the L0 already-sampled tokens as a new prefix, then
    generate `target_length - L0` more tokens with the same top_k condition
    and a fresh-but-deterministic seed. Logprobs and token ids for the new
    positions are appended onto the existing npz; the manifest length is
    bumped.

    NOTE: this is *not* a continuation of the same RNG stream. vLLM resets
    the RNG per request, so positions ≥ L0 use a new seed offset
    (`TOPK_EXTEND_SEED_OFFSET`). For our experiment that doesn't matter —
    we just need a continuation under the same top-k condition from the
    prefix that was already generated under that condition.
    """
    out_dir = Path(out_dir)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    manifest = pl.read_parquet(out_dir / MANIFEST_FILENAME)
    topk = manifest.filter(pl.col("family") == "topk").sort(["prompt_id", "run_idx"])
    todo = topk.filter(pl.col("length") < target_length)
    if len(todo) == 0:
        print(f"all topk traces already at length >= {target_length}; nothing to do.")
        return

    # All extensions in this run share the same delta; if existing lengths are
    # ragged, fail loudly (the design assumes uniform L0 per family).
    starting_lengths = set(todo["length"].to_list())
    if len(starting_lengths) != 1:
        raise ValueError(
            f"extension expects all topk traces at one starting length, "
            f"got {starting_lengths}"
        )
    L0 = starting_lengths.pop()
    delta = target_length - L0
    print(
        f"extending {len(todo)} topk traces: {L0} -> {target_length} ({delta} new tokens each)"
    )

    tokenizer = _ensure_tokenizer()

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    # Build fresh-prefix prompts (as TokensPrompt to bypass re-tokenization at
    # the prompt/sampled boundary) and per-trace sampling params.
    prompt_inputs: list = []
    sampling_params: list = []
    trace_ids: list[str] = []
    seeds_new: list[int] = []
    max_prefix_len = 0

    for row in todo.iter_rows(named=True):
        trace_id: str = row["trace_id"]
        prompt_text: str = row["prompt_text"]
        sampled = list(row["sampled_token_ids"])
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        new_prefix_ids = prompt_ids + sampled
        max_prefix_len = max(max_prefix_len, len(new_prefix_ids))

        params = json.loads(row["params_json"])
        top_k = int(params["top_k"])
        seed_new = int(row["seed"]) + TOPK_EXTEND_SEED_OFFSET
        sp = SamplingParams(
            temperature=TEMPERATURE,
            top_p=1.0,
            top_k=top_k,
            seed=seed_new,
            max_tokens=delta,
            min_tokens=delta,
            logprobs=TOP_LOGPROBS,
            ignore_eos=True,
            detokenize=True,
        )

        prompt_inputs.append(TokensPrompt(prompt_token_ids=new_prefix_ids))
        sampling_params.append(sp)
        trace_ids.append(trace_id)
        seeds_new.append(seed_new)

    llm = _make_llm(max_model_len=max_prefix_len + delta + 64)
    print(f"  vLLM loaded; max_model_len = {max_prefix_len + delta + 64}")
    print(f"Generating extensions: {len(prompt_inputs)} sequences × {delta} tokens...")
    outputs = llm.generate(prompt_inputs, sampling_params, use_tqdm=True)
    if len(outputs) != len(prompt_inputs):
        raise RuntimeError(f"got {len(outputs)} outputs for {len(prompt_inputs)} requests")

    # Need original manifest rows keyed by trace_id for the upsert.
    manifest_by_id: dict[str, dict] = {
        r["trace_id"]: r for r in manifest.iter_rows(named=True)
    }

    updated_rows: list[dict] = []
    for trace_id, output, seed_new in zip(trace_ids, outputs, seeds_new, strict=True):
        out0 = output.outputs[0]
        if len(out0.token_ids) != delta:
            raise RuntimeError(
                f"trace {trace_id}: expected {delta} new tokens, got {len(out0.token_ids)}"
            )
        new_ids, new_lps = extract_top_logprobs_array(out0.logprobs, top_n=TOP_LOGPROBS)
        new_length = append_to_trace(out_dir, trace_id, new_ids, new_lps)
        prev = manifest_by_id[trace_id]
        merged_decoded = prev["decoded_text"] + out0.text
        merged_sampled = list(prev["sampled_token_ids"]) + list(out0.token_ids)
        updated_rows.append(
            {
                "trace_id": trace_id,
                "family": prev["family"],
                "prompt_id": prev["prompt_id"],
                "prompt_text": prev["prompt_text"],
                "condition_label": prev["condition_label"],
                "params_json": prev["params_json"],
                "run_idx": prev["run_idx"],
                "seed": prev["seed"],  # keep the original seed; the extension seed lives in config
                "length": new_length,
                "sampled_token_ids": merged_sampled,
                "decoded_text": merged_decoded,
                "trace_path": prev["trace_path"],
            }
        )

    upsert_manifest(out_dir, updated_rows)
    print(f"Updated {len(updated_rows)} traces; new length = {target_length}")


# ============================================================ entry: blacklist


@dataclass(frozen=True)
class BlacklistSpec:
    prompt_id: int
    prompt: str
    banned_rank: int   # 1..N, the global frequency rank
    banned_token_id: int
    run_idx: int
    seed: int

    @property
    def trace_id(self) -> str:
        return blacklist_trace_id(self.prompt_id, self.banned_rank, self.run_idx)


def _build_blacklist_specs(
    banned_token_ids: list[int],
    prompts: Sequence[str] = PROMPTS,
) -> list[BlacklistSpec]:
    specs: list[BlacklistSpec] = []
    for pid, prompt in enumerate(prompts):
        for rank, tid in enumerate(banned_token_ids, start=1):
            for r in range(BLACKLIST_RUNS):
                specs.append(
                    BlacklistSpec(
                        prompt_id=pid,
                        prompt=prompt,
                        banned_rank=rank,
                        banned_token_id=int(tid),
                        run_idx=r,
                        seed=BLACKLIST_SEED_OFFSET + pid * 1000 + rank * 100 + r,
                    )
                )
    return specs


def _build_blacklist_sampling_params(spec: BlacklistSpec, max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        seed=spec.seed,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        logprobs=TOP_LOGPROBS,
        ignore_eos=True,
        detokenize=True,
        logit_bias={spec.banned_token_id: BLACKLIST_LOGIT_BIAS},
    )


def run_blacklist_experiment(
    out_dir: Path,
    max_tokens: int,
    banned_token_ids: list[int],
) -> None:
    """Generate the blacklist family. `banned_token_ids[i]` is the rank-(i+1) banned token."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if not banned_token_ids:
        raise ValueError("banned_token_ids must be non-empty")

    llm = _make_llm(max_model_len=max_tokens + 256)

    specs = _build_blacklist_specs(banned_token_ids)
    prompts = [s.prompt for s in specs]
    sampling_params = [_build_blacklist_sampling_params(s, max_tokens) for s in specs]

    print(
        f"Blacklist: generating {len(specs)} sequences × {max_tokens} tokens "
        f"(banned ranks 1..{len(banned_token_ids)})"
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    if len(outputs) != len(specs):
        raise RuntimeError(f"got {len(outputs)} outputs for {len(specs)} specs")

    new_rows: list[dict] = []
    for spec, output in zip(specs, outputs, strict=True):
        out0 = output.outputs[0]
        if len(out0.token_ids) != max_tokens:
            raise RuntimeError(
                f"spec {spec}: expected {max_tokens} tokens, got {len(out0.token_ids)}"
            )
        # Sanity: the banned token must never appear in sampled output.
        if spec.banned_token_id in out0.token_ids:
            raise RuntimeError(
                f"spec {spec}: banned token id {spec.banned_token_id} appeared in sampled output"
            )
        top_ids, top_lps = extract_top_logprobs_array(out0.logprobs, top_n=TOP_LOGPROBS)
        save_trace(out_dir, spec.trace_id, top_ids, top_lps)
        new_rows.append(
            {
                "trace_id": spec.trace_id,
                "family": "blacklist",
                "prompt_id": spec.prompt_id,
                "prompt_text": spec.prompt,
                "condition_label": f"blacklist_rank{spec.banned_rank}",
                "params_json": encode_params(
                    blacklist_params(spec.banned_rank, spec.banned_token_id)
                ),
                "run_idx": spec.run_idx,
                "seed": spec.seed,
                "length": max_tokens,
                "sampled_token_ids": list(out0.token_ids),
                "decoded_text": out0.text,
                "trace_path": trace_relpath(spec.trace_id),
            }
        )

    upsert_manifest(out_dir, new_rows)
    print(f"Wrote {len(new_rows)} blacklist traces to {out_dir}")


# ---------- back-compat alias ---------- #
# The earlier code called the top-k entry point `run_experiment`. Keep that
# name working so external scripts referring to it don't break.
run_experiment = run_topk_experiment
