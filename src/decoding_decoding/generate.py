"""vLLM generation harness for the top-k self-prediction experiment.

Loads Qwen/Qwen2.5-3B (base) in fp16, builds a batch of (prompt × condition × run)
SamplingParams, runs one `LLM.generate` call, and writes:
    {out_dir}/run_metadata.parquet           — one row per generated sequence
    {out_dir}/logprobs/{pid}_k{K}_r{run}.npz — top-N logprobs per step
    {out_dir}/config.json                    — run configuration
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

from decoding_decoding.prompts import PROMPTS


MODEL_NAME = "Qwen/Qwen2.5-3B"
TOP_LOGPROBS = 200
TEMPERATURE = 1.0
SEED_BASE = 42

# Conditions and their replication counts.
# K=1 at T=1 is deterministic (greedy), so a single run per prompt suffices.
CONDITIONS: tuple[str, ...] = ("1", "3", "5", "inf")
RUNS_PER_CONDITION: dict[str, int] = {"1": 1, "3": 8, "5": 8, "inf": 8}


@dataclass(frozen=True)
class GenSpec:
    prompt_id: int
    prompt: str
    condition_k: str  # "1", "3", "5", or "inf"
    run_idx: int
    seed: int


def build_specs(prompts: Sequence[str] = PROMPTS) -> list[GenSpec]:
    """Build the full (prompt × condition × run) plan.

    Seeds are deterministic and *matched* across conditions: at fixed
    (prompt_id, run_idx) the seed is identical for K=3, K=5, K=inf, so the
    three trajectories diverge only after step 1.
    """
    specs: list[GenSpec] = []
    for pid, prompt in enumerate(prompts):
        for k in CONDITIONS:
            for r in range(RUNS_PER_CONDITION[k]):
                specs.append(
                    GenSpec(
                        prompt_id=pid,
                        prompt=prompt,
                        condition_k=k,
                        run_idx=r,
                        seed=SEED_BASE + pid * 1000 + r,
                    )
                )
    return specs


def _top_k_int(condition_k: str) -> int:
    """vLLM convention: top_k=-1 disables top-k filtering."""
    return -1 if condition_k == "inf" else int(condition_k)


def build_sampling_params(spec: GenSpec, max_tokens: int):
    """Construct vLLM SamplingParams for a single GenSpec."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=_top_k_int(spec.condition_k),
        seed=spec.seed,
        max_tokens=max_tokens,
        min_tokens=max_tokens,  # force exactly max_tokens (no early EOS)
        logprobs=TOP_LOGPROBS,
        ignore_eos=True,
        detokenize=True,
    )


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


def run_experiment(out_dir: Path, max_tokens: int) -> None:
    """End-to-end: load model, generate, save metadata and per-sequence npz."""
    import vllm
    from vllm import LLM

    out_dir = Path(out_dir)
    logprobs_dir = out_dir / "logprobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logprobs_dir.mkdir(exist_ok=True)

    # Pin to GPU 0. The user's machine has 2× RTX 8000; we want only GPU 0.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",  # Turing has no bf16 tensor-core path.
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        logprobs_mode="raw_logprobs",
        enforce_eager=True,  # avoid CUDA graph / torch.compile issues on SM 7.5.
        max_model_len=max_tokens + 256,  # prompt is short; +256 is generous.
        seed=SEED_BASE,
        # Turing (SM 7.5): FA2 and FlashInfer require SM 8.0+. Use Triton kernels.
        attention_config={"backend": "TRITON_ATTN"},
        # Default cap is 20; we save the top-200 logprobs per step.
        max_logprobs=TOP_LOGPROBS,
    )

    specs = build_specs()
    prompts = [s.prompt for s in specs]
    sampling_params = [build_sampling_params(s, max_tokens) for s in specs]

    print(f"Generating {len(specs)} sequences × {max_tokens} tokens...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    if len(outputs) != len(specs):
        raise RuntimeError(f"got {len(outputs)} outputs for {len(specs)} specs")

    metadata_rows: list[dict] = []
    for spec, output in zip(specs, outputs, strict=True):
        out0 = output.outputs[0]
        if len(out0.token_ids) != max_tokens:
            raise RuntimeError(
                f"spec {spec}: expected {max_tokens} generated tokens, got {len(out0.token_ids)}"
            )
        top_ids, top_lps = extract_top_logprobs_array(out0.logprobs, top_n=TOP_LOGPROBS)
        npz_name = (
            f"{spec.prompt_id:02d}_k{spec.condition_k}_r{spec.run_idx:02d}.npz"
        )
        np.savez_compressed(
            logprobs_dir / npz_name,
            top_token_ids=top_ids,
            top_logprobs=top_lps,
        )
        metadata_rows.append(
            {
                "prompt_id": spec.prompt_id,
                "prompt_text": spec.prompt,
                "condition_k": spec.condition_k,
                "run_idx": spec.run_idx,
                "seed": spec.seed,
                "token_ids": list(out0.token_ids),
                "decoded_text": out0.text,
                "logprobs_npz": npz_name,
            }
        )

    df = pl.DataFrame(metadata_rows)
    df.write_parquet(out_dir / "run_metadata.parquet")

    config = {
        "model": MODEL_NAME,
        "vllm_version": vllm.__version__,
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "max_tokens": max_tokens,
        "top_logprobs": TOP_LOGPROBS,
        "conditions": list(CONDITIONS),
        "runs_per_condition": dict(RUNS_PER_CONDITION),
        "temperature": TEMPERATURE,
        "seed_base": SEED_BASE,
        "n_prompts": len(PROMPTS),
        "prompts_sha256": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Wrote {len(specs)} sequences to {out_dir}")
