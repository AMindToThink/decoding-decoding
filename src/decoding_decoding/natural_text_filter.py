"""Stream the Laplace β̂ filter through forward passes on real natural text.

This is a calibration diagnostic for the F2.1 calibration gap. Three arms,
all running the *same* streaming Laplace filter from `counter_decode.py`:

  1. ``filter_pass_self_gen``: feed the first ``warmup`` tokens of each
     prompt, then SAMPLE continuations at β_dec = 1 (no temperature
     manipulation). The filter sees (ℓ_t, x_t) where x_t ~ softmax(ℓ_t).
     This is the actual no-imprint null consistent with F2.1's sampling
     protocol. The filter should converge to log β̂ = 0.

  2. ``filter_pass_natural``: feed the full passage (prompt + real natural
     continuation). The filter sees (ℓ_t, x_t) where x_t is the actual next
     token in the corpus. If log β̂ drifts away from 0 here, that is the
     model's intrinsic calibration offset on the corpus — a confound for
     F2.1's interpretation, not "filter pathology" (the filter is doing
     correct MLE inference of β given the data).

Headline x-axis is cumulative Fisher Σ_τ Var_q(τ) = J(t) − J_0, not t —
different corpora accumulate evidence at different rates because of entropy
differences, and J is the right notion of "effective sample size" for the
filter.

The single ``PassageResult`` dataclass holds the per-step trajectory for
each passage so the two arms can be loaded and plotted symmetrically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from decoding_decoding.counter_decode import LaplaceState, init_laplace, laplace_update


MODEL_NAME = "Qwen/Qwen2.5-3B"
DEFAULT_SIGMA_0 = 0.5
DEFAULT_WARMUP = 20
DEFAULT_LENGTH = 220   # 20 prompt + 200 filter steps — matches F0's 200-step horizon


@dataclass(frozen=True)
class PassageResult:
    """Per-passage filter trajectory + diagnostics.

    All per-step arrays have length U = length - warmup, which is the number
    of filter updates run. Indexing: step k (k = 0..U-1) corresponds to using
    ℓ at sequence position (warmup-1+k) to predict the token at sequence
    position (warmup+k).

    log_beta_hat[k] and J[k] are recorded BEFORE the k-th update is applied.
    """

    corpus: str
    arm: str                    # "self_gen" or "natural"
    passage_id: int
    text_preview: str
    log_beta_hat: np.ndarray    # (U,)  η̂ at the START of step k
    J: np.ndarray               # (U,)  filter precision at the START of step k
    score: np.ndarray           # (U,)  per-step score (ℓ[x] − E_q)
    fisher: np.ndarray          # (U,)  per-step Fisher (Var_q)
    target_logprob: np.ndarray  # (U,)  log q[x_k] under model softmax
    model_entropy: np.ndarray   # (U,)  H(softmax(ℓ)) at each step
    sampled_token_ids: np.ndarray  # (U,) tokens fed to the filter (real or sampled)
    warmup: int
    length: int


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------


def load_wikitext_passages(
    *,
    n: int,
    min_tokens: int,
    tokenizer,
    seed: int = 0,
) -> list[str]:
    """Load `n` WikiText-103 test-split articles.

    The dataset is line-segmented with article headers " = TITLE = ". We join
    lines into articles, strip the leading title artifact, and return ``n``
    bodies that tokenize to at least ``min_tokens`` tokens.
    """
    from datasets import load_dataset

    header_re = re.compile(r"^ = [^=]+ = \n?$")
    articles: list[str] = []
    # Use both validation and test splits to get ~120 long articles.
    for split in ("validation", "test"):
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
        )
        cur: list[str] = []
        for row in ds:
            line = row["text"]
            if header_re.match(line):
                if cur:
                    articles.append("".join(cur).strip())
                cur = [line]
            else:
                cur.append(line)
        if cur:
            articles.append("".join(cur).strip())

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(articles))
    long_enough: list[str] = []
    for idx in order:
        art = articles[idx]
        if not art.strip():
            continue
        body = re.sub(r"^ = [^=]+ = \n", "", art, count=1)
        if not body.strip():
            continue
        toks = tokenizer(body, add_special_tokens=False)["input_ids"]
        if len(toks) < min_tokens:
            continue
        long_enough.append(body)
        if len(long_enough) >= n:
            break
    if len(long_enough) < n:
        raise RuntimeError(
            f"only found {len(long_enough)} long-enough WikiText articles; need {n}"
        )
    return long_enough


def load_writingprompts_passages(
    *,
    n: int,
    min_tokens: int,
    tokenizer,
    seed: int = 0,
) -> list[str]:
    """Load `n` WritingPrompts test-split stories long enough for the filter."""
    from datasets import load_dataset

    ds = load_dataset("euclaise/writingprompts", split="test")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    out: list[str] = []
    for idx in order:
        story = (ds[int(idx)]["story"] or "").strip()
        if not story:
            continue
        toks = tokenizer(story, add_special_tokens=False)["input_ids"]
        if len(toks) < min_tokens:
            continue
        out.append(story)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"only found {len(out)} long-enough WritingPrompts stories; need {n}"
        )
    return out


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------


def _tokenize_to_length(
    texts: Sequence[str],
    tokenizer,
    *,
    length: int,
) -> torch.Tensor:
    """Tokenize each text and truncate to exactly `length` tokens.

    Returns a (B, length) int64 tensor on CPU.
    """
    rows: list[list[int]] = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) < length:
            raise ValueError(
                f"passage tokenized to {len(ids)} tokens, need {length}"
            )
        rows.append(ids[:length])
    return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------------
# Natural-text arm
# ---------------------------------------------------------------------------


def filter_pass_natural(
    texts: Sequence[str],
    *,
    corpus: str,
    model,
    tokenizer,
    sigma_0: float = DEFAULT_SIGMA_0,
    warmup: int = DEFAULT_WARMUP,
    length: int = DEFAULT_LENGTH,
    device: str = "cuda",
    batch_size: int = 32,
) -> list[PassageResult]:
    """Run the streaming Laplace filter on each passage's real next-tokens.

    For each passage:
      1. Tokenize to exactly ``length`` tokens.
      2. One forward pass to get logits at every position.
      3. For each filter step k = 0..length-warmup-1:
           - ℓ = logits at position (warmup - 1 + k)
           - x_k = token at position (warmup + k)         ← the real corpus token
           - run ``laplace_update``.

    The filter state is INDEPENDENT per passage. Forward pass is batched.
    """
    B_total = len(texts)
    results: list[PassageResult] = []

    for chunk_start in range(0, B_total, batch_size):
        chunk_texts = list(texts[chunk_start:chunk_start + batch_size])
        ids = _tokenize_to_length(chunk_texts, tokenizer, length=length).to(device)
        B = ids.shape[0]
        attn = torch.ones_like(ids)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        all_logits = out.logits.float()  # (B, length, V)

        state = init_laplace(B, sigma_0=sigma_0, device=device, dtype=torch.float32)
        U = length - warmup
        log_beta_hat = np.zeros((B, U), dtype=np.float32)
        J_arr = np.zeros((B, U), dtype=np.float32)
        score_arr = np.zeros((B, U), dtype=np.float32)
        fisher_arr = np.zeros((B, U), dtype=np.float32)
        target_lp_arr = np.zeros((B, U), dtype=np.float32)
        entropy_arr = np.zeros((B, U), dtype=np.float32)
        sampled_arr = np.zeros((B, U), dtype=np.int64)

        for k in range(U):
            seq_pos = warmup - 1 + k                # logits position
            target_pos = warmup + k                  # token position (real)
            logits_t = all_logits[:, seq_pos, :]
            x_next = ids[:, target_pos]

            log_beta_hat[:, k] = state.eta_hat.detach().cpu().numpy()
            J_arr[:, k] = state.J.detach().cpu().numpy()
            log_q = F.log_softmax(logits_t, dim=-1)
            entropy_arr[:, k] = (-(log_q.exp() * log_q).sum(dim=-1)).detach().cpu().numpy()
            target_lp_arr[:, k] = (
                log_q.gather(-1, x_next.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
            )
            sampled_arr[:, k] = x_next.detach().cpu().numpy()

            state, score, fisher = laplace_update(state, logits_t, x_next)
            score_arr[:, k] = score.detach().cpu().numpy()
            fisher_arr[:, k] = fisher.detach().cpu().numpy()

        for b in range(B):
            results.append(
                PassageResult(
                    corpus=corpus,
                    arm="natural",
                    passage_id=chunk_start + b,
                    text_preview=chunk_texts[b][:120].replace("\n", " "),
                    log_beta_hat=log_beta_hat[b],
                    J=J_arr[b],
                    score=score_arr[b],
                    fisher=fisher_arr[b],
                    target_logprob=target_lp_arr[b],
                    model_entropy=entropy_arr[b],
                    sampled_token_ids=sampled_arr[b],
                    warmup=warmup,
                    length=length,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Self-generated arm (β_dec = 1, autoregressive)
# ---------------------------------------------------------------------------


def filter_pass_self_gen(
    texts: Sequence[str],
    *,
    corpus: str,
    model,
    tokenizer,
    sigma_0: float = DEFAULT_SIGMA_0,
    warmup: int = DEFAULT_WARMUP,
    length: int = DEFAULT_LENGTH,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 12345,
) -> list[PassageResult]:
    """Autoregressive self-generation at β_dec = 1, with the filter updated inline.

    For each text (used only for its first ``warmup`` tokens, which act as the
    prompt), generate ``length - warmup`` tokens by sampling x_t ~ softmax(ℓ_t)
    (no temperature scaling). The filter sees the same (ℓ_t, x_t) pairs as in
    the natural arm, but x_t is now sampled rather than read from the corpus.

    KV cache is reused across decode steps.
    """
    B_total = len(texts)
    results: list[PassageResult] = []
    gen = torch.Generator(device=device).manual_seed(seed)

    for chunk_start in range(0, B_total, batch_size):
        chunk_texts = list(texts[chunk_start:chunk_start + batch_size])
        prompt_ids = _tokenize_to_length(chunk_texts, tokenizer, length=warmup).to(
            device
        )
        B = prompt_ids.shape[0]
        attn = torch.ones_like(prompt_ids)

        # Prefill the prompt.
        with torch.no_grad():
            out = model(
                input_ids=prompt_ids, attention_mask=attn, use_cache=True
            )
        logits_t = out.logits[:, -1, :].float()      # logits used to predict token at index `warmup`
        past = out.past_key_values

        state = init_laplace(B, sigma_0=sigma_0, device=device, dtype=torch.float32)
        U = length - warmup
        log_beta_hat = np.zeros((B, U), dtype=np.float32)
        J_arr = np.zeros((B, U), dtype=np.float32)
        score_arr = np.zeros((B, U), dtype=np.float32)
        fisher_arr = np.zeros((B, U), dtype=np.float32)
        target_lp_arr = np.zeros((B, U), dtype=np.float32)
        entropy_arr = np.zeros((B, U), dtype=np.float32)
        sampled_arr = np.zeros((B, U), dtype=np.int64)

        for k in range(U):
            log_beta_hat[:, k] = state.eta_hat.detach().cpu().numpy()
            J_arr[:, k] = state.J.detach().cpu().numpy()
            log_q = F.log_softmax(logits_t, dim=-1)
            entropy_arr[:, k] = (-(log_q.exp() * log_q).sum(dim=-1)).detach().cpu().numpy()

            # Sample x_t ~ softmax(ℓ_t) (β_dec = 1).
            probs = log_q.exp()
            x_next = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

            target_lp_arr[:, k] = (
                log_q.gather(-1, x_next.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
            )
            sampled_arr[:, k] = x_next.detach().cpu().numpy()

            state, score, fisher = laplace_update(state, logits_t, x_next)
            score_arr[:, k] = score.detach().cpu().numpy()
            fisher_arr[:, k] = fisher.detach().cpu().numpy()

            if k < U - 1:
                # Next-token forward pass with KV cache.
                attn = torch.cat(
                    [attn, torch.ones(B, 1, device=device, dtype=attn.dtype)], dim=-1
                )
                with torch.no_grad():
                    out = model(
                        input_ids=x_next.unsqueeze(-1),
                        attention_mask=attn,
                        past_key_values=past,
                        use_cache=True,
                    )
                logits_t = out.logits[:, -1, :].float()
                past = out.past_key_values

        for b in range(B):
            results.append(
                PassageResult(
                    corpus=corpus,
                    arm="self_gen",
                    passage_id=chunk_start + b,
                    text_preview=chunk_texts[b][:120].replace("\n", " "),
                    log_beta_hat=log_beta_hat[b],
                    J=J_arr[b],
                    score=score_arr[b],
                    fisher=fisher_arr[b],
                    target_logprob=target_lp_arr[b],
                    model_entropy=entropy_arr[b],
                    sampled_token_ids=sampled_arr[b],
                    warmup=warmup,
                    length=length,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_results(
    results: Iterable[PassageResult],
    out_path: Path,
) -> None:
    """Save a list of PassageResult to a single .npz file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_list = list(results)
    if not res_list:
        raise ValueError("nothing to save")
    payload = {
        "corpus": np.array([r.corpus for r in res_list], dtype=object),
        "arm": np.array([r.arm for r in res_list], dtype=object),
        "passage_id": np.array([r.passage_id for r in res_list], dtype=np.int32),
        "text_preview": np.array([r.text_preview for r in res_list], dtype=object),
        "log_beta_hat": np.stack([r.log_beta_hat for r in res_list], axis=0),
        "J": np.stack([r.J for r in res_list], axis=0),
        "score": np.stack([r.score for r in res_list], axis=0),
        "fisher": np.stack([r.fisher for r in res_list], axis=0),
        "target_logprob": np.stack([r.target_logprob for r in res_list], axis=0),
        "model_entropy": np.stack([r.model_entropy for r in res_list], axis=0),
        "sampled_token_ids": np.stack([r.sampled_token_ids for r in res_list], axis=0),
        "warmup": np.array([r.warmup for r in res_list], dtype=np.int32),
        "length": np.array([r.length for r in res_list], dtype=np.int32),
    }
    np.savez_compressed(out_path, **payload, allow_pickle=True)


def load_results(in_path: Path) -> list[PassageResult]:
    npz = np.load(in_path, allow_pickle=True)
    n = npz["corpus"].shape[0]
    return [
        PassageResult(
            corpus=str(npz["corpus"][i]),
            arm=str(npz["arm"][i]),
            passage_id=int(npz["passage_id"][i]),
            text_preview=str(npz["text_preview"][i]),
            log_beta_hat=npz["log_beta_hat"][i],
            J=npz["J"][i],
            score=npz["score"][i],
            fisher=npz["fisher"][i],
            target_logprob=npz["target_logprob"][i],
            model_entropy=npz["model_entropy"][i],
            sampled_token_ids=npz["sampled_token_ids"][i],
            warmup=int(npz["warmup"][i]),
            length=int(npz["length"][i]),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    *, device: str = "cuda", dtype: torch.dtype = torch.float16
):
    """Load Qwen-2.5-3B (matches counter_decode_generate defaults)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    return model, tok
