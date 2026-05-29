"""End-to-end smoke tests for the counter-decoding generation harness.

These actually load a (small) HF model and run a few steps. They are
GPU-required and gated by env var DD_GPU_TESTS=1 to avoid blocking CI.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

if os.environ.get("DD_GPU_TESTS") != "1":
    pytest.skip("set DD_GPU_TESTS=1 to run GPU-required smoke tests", allow_module_level=True)


from decoding_decoding.counter_decode import init_laplace, laplace_update
from decoding_decoding.counter_decode_generate import (  # noqa: E402
    _generate_batch,
    _load_model_and_tokenizer,
)


@pytest.fixture(scope="module")
def small_model():
    """Load a tiny model for the smoke test. We override MODEL_NAME at the
    module attribute level so we don't pull down 3B weights for a smoke check.
    """
    import decoding_decoding.counter_decode_generate as cdg

    orig = cdg.MODEL_NAME
    cdg.MODEL_NAME = "sshleifer/tiny-gpt2"
    try:
        model, tok = _load_model_and_tokenizer(device="cuda", dtype=torch.float32)
        yield model, tok
    finally:
        cdg.MODEL_NAME = orig


def test_generate_batch_runs_and_shapes(small_model) -> None:
    model, tok = small_model
    prompts = ["hello world", "the quick brown fox"]
    B = len(prompts)
    T = 8

    res = _generate_batch(
        model=model,
        tokenizer=tok,
        prompts=prompts,
        beta_target_per_traj=torch.tensor([1.5, 1.5], dtype=torch.float32, device="cuda"),
        corrected_per_traj=torch.tensor([True, False], dtype=torch.bool, device="cuda"),
        sigma_0_per_traj=torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda"),
        seeds_per_traj=[1, 2],
        max_tokens=T,
        top_n=20,
        sparse_positions=(0, 4),
        save_sparse_for_mask=torch.tensor([False, True], dtype=torch.bool, device="cuda"),
        device="cuda",
    )

    # Shape checks.
    assert res.sampled_token_ids.shape == (B, T)
    assert res.top_token_ids.shape == (B, T, 20)
    assert res.top_logprobs.shape == (B, T, 20)
    assert res.beta_hat_pre.shape == (B, T)
    assert res.beta_dec.shape == (B, T)
    assert res.J_pre.shape == (B, T)
    assert res.score.shape == (B, T)
    assert res.fisher.shape == (B, T)
    assert res.max_p_phi.shape == (B, T)
    assert res.entropy_phi.shape == (B, T)
    assert res.max_p_sample.shape == (B, T)
    assert res.entropy_sample.shape == (B, T)
    assert res.eta_hat_post_final.shape == (B,)
    assert res.J_post_final.shape == (B,)
    assert len(res.decoded_text) == B

    # Sparse logits exist (for both rows in this case since we requested
    # them via sparse_positions; the decision to write to disk is downstream
    # of save_sparse_for_mask).
    assert res.sparse_positions is not None
    assert res.sparse_logits is not None
    assert res.sparse_positions.tolist() == [0, 4]
    assert res.sparse_logits.shape[1] == 2  # P=2
    assert res.sparse_logits.shape[0] == B
    assert res.sparse_logits.shape[2] == model.config.vocab_size

    # Sanity: sampled tokens are in vocab.
    V = model.config.vocab_size
    assert (res.sampled_token_ids >= 0).all()
    assert (res.sampled_token_ids < V).all()


def test_uncorrected_arm_has_constant_beta_dec(small_model) -> None:
    model, tok = small_model
    res = _generate_batch(
        model=model,
        tokenizer=tok,
        prompts=["hello"],
        beta_target_per_traj=torch.tensor([1.7], dtype=torch.float32, device="cuda"),
        corrected_per_traj=torch.tensor([False], dtype=torch.bool, device="cuda"),
        sigma_0_per_traj=torch.tensor([0.5], dtype=torch.float32, device="cuda"),
        seeds_per_traj=[42],
        max_tokens=12,
        top_n=10,
        sparse_positions=None,
        save_sparse_for_mask=torch.tensor([False], dtype=torch.bool, device="cuda"),
        device="cuda",
    )
    np.testing.assert_allclose(res.beta_dec[0], 1.7, atol=1e-6)


def test_corrected_arm_has_beta_dec_times_beta_hat_eq_target(small_model) -> None:
    model, tok = small_model
    res = _generate_batch(
        model=model,
        tokenizer=tok,
        prompts=["hello world"],
        beta_target_per_traj=torch.tensor([1.7], dtype=torch.float32, device="cuda"),
        corrected_per_traj=torch.tensor([True], dtype=torch.bool, device="cuda"),
        sigma_0_per_traj=torch.tensor([0.5], dtype=torch.float32, device="cuda"),
        seeds_per_traj=[42],
        max_tokens=10,
        top_n=10,
        sparse_positions=None,
        save_sparse_for_mask=torch.tensor([False], dtype=torch.bool, device="cuda"),
        device="cuda",
    )
    # By construction of the corrector: β_dec_t · β̂_pre_t = β_target.
    eff = res.beta_dec[0] * res.beta_hat_pre[0]
    np.testing.assert_allclose(eff, 1.7, atol=1e-5)


def test_laplace_state_evolution_matches_offline(small_model) -> None:
    """Run the harness with a known prompt and seed, then re-run the
    Laplace update offline given the same per-step (logits topN, sampled token).

    The offline reconstruction can only use top-N logprobs which approximate
    full-vocab moments. We check that the harness's online β̂ trace matches an
    offline filter that uses the SAME full vocab moments — to do this we need
    the harness's recorded eta_hat trace AND a check that Var_q (fisher)
    matches what we'd compute from the saved logits.

    Specifically: J_pre[t+1] = J_pre[t] + fisher[t]. Verify this invariant.
    """
    model, tok = small_model
    res = _generate_batch(
        model=model,
        tokenizer=tok,
        prompts=["the cat sat"],
        beta_target_per_traj=torch.tensor([1.4], dtype=torch.float32, device="cuda"),
        corrected_per_traj=torch.tensor([True], dtype=torch.bool, device="cuda"),
        sigma_0_per_traj=torch.tensor([0.5], dtype=torch.float32, device="cuda"),
        seeds_per_traj=[42],
        max_tokens=15,
        top_n=10,
        sparse_positions=None,
        save_sparse_for_mask=torch.tensor([False], dtype=torch.bool, device="cuda"),
        device="cuda",
    )
    J_pre = res.J_pre[0]
    fisher = res.fisher[0]
    # J_pre[t+1] should equal J_pre[t] + fisher[t] for all t in [0, T-2].
    np.testing.assert_allclose(J_pre[1:], J_pre[:-1] + fisher[:-1], atol=1e-3, rtol=1e-3)
