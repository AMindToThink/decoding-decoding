"""Unit tests for blacklist_analyze using synthetic on-disk data.

We construct a tiny manifest + traces directory inside tmp_path and verify:
  * load_blacklist_condition correctly extracts p_banned from a saved trace,
    using the banned_token_id stored in params_json.
  * When the banned token is OUT of the saved top-N, p_banned is filled in
    with the floor probability (last-rank prob) and in_topn is False.
  * baseline_p_for_token reads the same token's prob out of a topk=inf trace.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from decoding_decoding.blacklist_analyze import (
    baseline_p_for_token,
    load_blacklist_condition,
)
from decoding_decoding.data_layout import (
    blacklist_params,
    blacklist_trace_id,
    encode_params,
    save_trace,
    topk_params,
    topk_trace_id,
    trace_relpath,
    upsert_manifest,
)


def _make_logprobs(top_ids: list[list[int]]) -> np.ndarray:
    """Each row's logprobs are [-1.0, -2.0, -3.0, ...] descending."""
    L = len(top_ids)
    N = len(top_ids[0])
    lps = np.empty((L, N), dtype=np.float32)
    for i in range(L):
        lps[i] = -np.arange(1, N + 1, dtype=np.float32)
    return lps


def test_load_blacklist_condition_picks_correct_token(tmp_path: Path) -> None:
    banned_token = 999
    L, N = 4, 5
    # banned token sits at rank 2 (column 1) at every step.
    top_ids = np.tile(np.array([[100, banned_token, 200, 300, 400]]), (L, 1))
    top_lps = _make_logprobs(top_ids.tolist())
    trace_id = blacklist_trace_id(0, banned_rank=1, run_idx=0)
    save_trace(tmp_path, trace_id, top_ids, top_lps)
    upsert_manifest(tmp_path, [{
        "trace_id": trace_id,
        "family": "blacklist",
        "prompt_id": 0,
        "prompt_text": "x",
        "condition_label": "blacklist_rank1",
        "params_json": encode_params(blacklist_params(banned_rank=1, banned_token_id=banned_token)),
        "run_idx": 0,
        "seed": 1,
        "length": L,
        "sampled_token_ids": [1] * L,
        "decoded_text": "",
        "trace_path": trace_relpath(trace_id),
    }])

    cd = load_blacklist_condition(tmp_path, banned_rank=1)
    assert cd.banned_token_id == banned_token
    assert cd.p_banned.shape == (1, 1, L)
    # rank-2 logprob is -2.0, so p = exp(-2.0)
    np.testing.assert_allclose(cd.p_banned[0, 0], np.full(L, np.exp(-2.0)), atol=1e-6)
    assert cd.in_topn.all()


def test_load_blacklist_falls_back_to_floor_when_token_absent(tmp_path: Path) -> None:
    banned_token = 999
    L, N = 4, 5
    # The banned token is NOT in the top-N at any step.
    top_ids = np.tile(np.array([[100, 200, 300, 400, 500]]), (L, 1))
    top_lps = _make_logprobs(top_ids.tolist())
    trace_id = blacklist_trace_id(0, banned_rank=1, run_idx=0)
    save_trace(tmp_path, trace_id, top_ids, top_lps)
    upsert_manifest(tmp_path, [{
        "trace_id": trace_id,
        "family": "blacklist",
        "prompt_id": 0,
        "prompt_text": "x",
        "condition_label": "blacklist_rank1",
        "params_json": encode_params(blacklist_params(banned_rank=1, banned_token_id=banned_token)),
        "run_idx": 0,
        "seed": 1,
        "length": L,
        "sampled_token_ids": [1] * L,
        "decoded_text": "",
        "trace_path": trace_relpath(trace_id),
    }])

    cd = load_blacklist_condition(tmp_path, banned_rank=1)
    assert cd.banned_token_id == banned_token
    # floor is rank-N logprob = -5.0 → p = exp(-5)
    np.testing.assert_allclose(cd.p_banned[0, 0], np.full(L, np.exp(-5.0)), atol=1e-6)
    assert (~cd.in_topn).all()
    np.testing.assert_allclose(cd.floor_p[0, 0], np.full(L, np.exp(-5.0)), atol=1e-6)


def test_baseline_p_for_token_reads_topk_inf_traces(tmp_path: Path) -> None:
    target_token = 7
    L, N = 3, 5
    # target sits at rank 3 (column 2) at every step.
    top_ids = np.tile(np.array([[1, 2, target_token, 8, 9]]), (L, 1))
    top_lps = _make_logprobs(top_ids.tolist())
    trace_id = topk_trace_id(0, "inf", 0)
    save_trace(tmp_path, trace_id, top_ids, top_lps)
    upsert_manifest(tmp_path, [{
        "trace_id": trace_id,
        "family": "topk",
        "prompt_id": 0,
        "prompt_text": "x",
        "condition_label": "k=inf",
        "params_json": encode_params(topk_params("inf")),
        "run_idx": 0,
        "seed": 0,
        "length": L,
        "sampled_token_ids": [1] * L,
        "decoded_text": "",
        "trace_path": trace_relpath(trace_id),
    }])

    p = baseline_p_for_token(tmp_path, target_token)
    assert p.shape == (1, 1, L)
    # rank-3 → -3.0 → exp(-3.0)
    np.testing.assert_allclose(p[0, 0], np.full(L, np.exp(-3.0)), atol=1e-6)
