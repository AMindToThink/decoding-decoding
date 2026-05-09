"""Unit tests for data_layout — trace_id determinism and round-trip on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from decoding_decoding.data_layout import (
    MANIFEST_FILENAME,
    append_to_trace,
    blacklist_params,
    blacklist_trace_id,
    decode_params,
    encode_params,
    load_manifest,
    load_trace,
    save_trace,
    topk_params,
    topk_trace_id,
    trace_path_for,
    trace_relpath,
    upsert_manifest,
    write_manifest,
)


def test_topk_trace_id_is_deterministic() -> None:
    assert topk_trace_id(7, "3", 4) == "topk_p07_k3_r04"
    assert topk_trace_id(7, "inf", 0) == "topk_p07_kinf_r00"


def test_blacklist_trace_id_is_deterministic() -> None:
    assert blacklist_trace_id(0, 1, 0) == "blacklist_p00_b1_r00"
    assert blacklist_trace_id(31, 5, 7) == "blacklist_p31_b5_r07"


def test_trace_relpath_is_under_traces_subdir() -> None:
    rel = trace_relpath("topk_p00_k3_r00")
    assert rel == "traces/topk_p00_k3_r00.npz"


def test_topk_params_inf_maps_to_neg_one() -> None:
    assert topk_params("inf") == {"top_k": -1}
    assert topk_params("3") == {"top_k": 3}


def test_blacklist_params_includes_token_id() -> None:
    p = blacklist_params(banned_rank=2, banned_token_id=42)
    assert p == {"top_k": -1, "banned_rank": 2, "banned_token_id": 42}


def test_encode_decode_params_roundtrip() -> None:
    p = blacklist_params(3, 99)
    s = encode_params(p)
    assert isinstance(s, str)
    assert decode_params(s) == p


def test_save_load_trace_roundtrip(tmp_path: Path) -> None:
    ids = np.arange(40, dtype=np.int32).reshape(5, 8)
    lps = np.linspace(-3.0, 0.0, 40, dtype=np.float32).reshape(5, 8)
    save_trace(tmp_path, "topk_p00_k3_r00", ids, lps)
    np.testing.assert_array_equal(np.asarray(load_trace(tmp_path, "topk_p00_k3_r00")[0]), ids)
    np.testing.assert_allclose(load_trace(tmp_path, "topk_p00_k3_r00")[1], lps, rtol=0, atol=1e-6)
    assert (tmp_path / trace_relpath("topk_p00_k3_r00")).exists()


def test_append_to_trace_extends_length(tmp_path: Path) -> None:
    ids0 = np.arange(40, dtype=np.int32).reshape(5, 8)
    lps0 = np.zeros((5, 8), dtype=np.float32)
    save_trace(tmp_path, "topk_p00_k3_r00", ids0, lps0)
    ids1 = np.arange(40, 64, dtype=np.int32).reshape(3, 8)
    lps1 = np.ones((3, 8), dtype=np.float32)
    new_len = append_to_trace(tmp_path, "topk_p00_k3_r00", ids1, lps1)
    assert new_len == 8
    cat_ids, cat_lps = load_trace(tmp_path, "topk_p00_k3_r00")
    np.testing.assert_array_equal(cat_ids, np.concatenate([ids0, ids1], axis=0))
    np.testing.assert_array_equal(cat_lps, np.concatenate([lps0, lps1], axis=0))


def _row_for(trace_id: str, length: int) -> dict:
    return {
        "trace_id": trace_id,
        "family": "topk",
        "prompt_id": 0,
        "prompt_text": "hi",
        "condition_label": "k=3",
        "params_json": encode_params(topk_params("3")),
        "run_idx": 0,
        "seed": 42,
        "length": length,
        "sampled_token_ids": [1, 2, 3],
        "decoded_text": "x",
        "trace_path": trace_relpath(trace_id),
    }


def test_upsert_manifest_replaces_existing_row(tmp_path: Path) -> None:
    upsert_manifest(tmp_path, [_row_for("topk_p00_k3_r00", 1024)])
    df = load_manifest(tmp_path)
    assert df["length"].to_list() == [1024]

    upsert_manifest(tmp_path, [_row_for("topk_p00_k3_r00", 2048)])
    df = load_manifest(tmp_path)
    assert df["length"].to_list() == [2048]
    assert len(df) == 1  # not duplicated


def test_upsert_manifest_appends_new_rows(tmp_path: Path) -> None:
    upsert_manifest(tmp_path, [_row_for("topk_p00_k3_r00", 1024)])
    upsert_manifest(tmp_path, [_row_for("topk_p00_k3_r01", 1024)])
    df = load_manifest(tmp_path)
    assert sorted(df["trace_id"].to_list()) == ["topk_p00_k3_r00", "topk_p00_k3_r01"]
