"""Go/No-Go existence test for the full-sequence structured twist.

Pre-registered design: full-sequence/GO_NO_GO_DESIGN.md.

Question: does the sequence-level temperature correction F have enough structure
on real text — beyond what a global temperature retune already captures — for a
log-linear twist over the existing 1-D Laplace latent to recover a small but
statistically-significant fraction of it?

For each evaluation position (committed prefix x_<t):
  1. One model forward over the prefix -> base logits at the last position.
  2. Candidate set = top-K tokens by base probability (T-independent ranking).
  3. For each candidate v: ONE forward over prefix+v -> log P(.|prefix,v) ->
     exact depth-1 (optionally depth-2) future partition c_v(T) = log F(prefix,v).
     The leaf log-probs are T-INDEPENDENT, so they are computed once and reused
     across all temperatures.
  4. Candidate-conditioned Laplace latent theta(v) (vectorized peek), with the
     filter advanced over the prefix tokens first.

Then per (corpus, T):
  q_pt    ∝ exp(g_pt)                  # F ≡ 1  (per-token tempering)
  q_g     ∝ exp(g_pt + c)             # depth-h sequence-level target
  q_bestT ∝ exp(beta_eff * ell)       # best single global temperature
  q_filt  ∝ exp(g_pt + w_hat*theta)   # scalar-linear Laplace twist (distilled)
  q_rich  ∝ exp(g_pt + c_hat)         # flexible 1-D map of theta -> c (ceiling)

Metrics (bootstrap CIs over positions): D_total = E[KL(q_g||q_pt)],
recovered fractions vs q_pt, and the scrutiny-proof increments
frac_* - frac_bestT (correction recovered BEYOND any global temperature).

Outputs summary.json (+ a per-position JSONL) under the out dir; a companion
build-macros script turns summary.json into macros.tex (numbers-from-scripts).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from decoding_decoding.full_sequence.analyze import (
    PositionRecord,
    aggregate,
    go_no_go_decision,
)
from decoding_decoding.full_sequence.lookahead import (
    candidate_filter_latent,
    candidate_leaf_log_F,
)
from decoding_decoding.counter_decode import init_laplace, laplace_update
from decoding_decoding.natural_text_filter import (
    load_wikitext_passages,
    load_writingprompts_passages,
)

GPT2_LARGE = "gpt2-large"
QWEN = "Qwen/Qwen2.5-3B"


def load_model(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="sdpa"
    )
    model.to(device)
    model.eval()
    return model, tok


def make_forward_last_logits(model, device):
    """Return fn: (B, L) ids -> (B, V) logits at the last position."""

    def forward_last_logits(batch_ids: torch.Tensor) -> torch.Tensor:
        batch_ids = batch_ids.to(device)
        attn = torch.ones_like(batch_ids)
        with torch.no_grad():
            out = model(input_ids=batch_ids, attention_mask=attn, use_cache=False)
        return out.logits[:, -1, :].float().cpu()

    return forward_last_logits


def leaf_log_F_depth(
    forward_last_logits,
    prefix_ids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    T_list: list[float],
    depth: int,
    beam_k: int,
    leaf_batch: int,
) -> dict:
    """log F(prefix, v) for each candidate v, for each T, at the given depth.

    depth=1: exact one-step tempered partition at the child (T-independent leaf
    log-probs computed once, tempered per T).
    depth=2: expand each child over its own top-`beam_k` and recurse one level
    (a beam-truncated estimate; reported as a horizon-sensitivity check).
    Returns {T: np.ndarray[K]}.
    """
    from decoding_decoding.full_sequence.partition import tempered_log_partition

    K = candidates.shape[0]
    L = prefix_ids.shape[0]
    # Child leaf logits: one batched forward over [prefix + v].
    seqs = torch.empty((K, L + 1), dtype=torch.long)
    seqs[:, :L] = prefix_ids.unsqueeze(0).expand(K, L)
    seqs[:, L] = candidates
    child_log_p_list = []
    for s in range(0, K, leaf_batch):
        chunk = seqs[s:s + leaf_batch]
        last = forward_last_logits(chunk)
        child_log_p_list.append(torch.log_softmax(last.float(), dim=-1))
    child_log_p = torch.cat(child_log_p_list, dim=0)  # (K, V)

    out = {T: np.empty(K, dtype=np.float64) for T in T_list}

    if depth == 1:
        clp = child_log_p.numpy()
        for j in range(K):
            for T in T_list:
                out[T][j] = tempered_log_partition(clp[j], T)
        return out

    # depth >= 2: for each candidate v, for each T:
    #   log F = logsumexp_w [ (1/T) log P(w|prefix,v) + log F_{depth-1}(prefix,v,w) ]
    # approximated by expanding only the top-`beam_k` w (beam truncation).
    for j in range(K):
        v = int(candidates[j].item())
        # top beam_k children w of this candidate
        topw = torch.topk(child_log_p[j], k=beam_k).indices  # (beam_k,)
        grandchild = leaf_log_F_depth(
            forward_last_logits,
            torch.cat([prefix_ids, candidates[j:j + 1]]),
            topw,
            T_list=T_list,
            depth=depth - 1,
            beam_k=beam_k,
            leaf_batch=leaf_batch,
        )
        clp_j = child_log_p[j]
        for T in T_list:
            terms = clp_j[topw].numpy() / T + grandchild[T]
            m = float(terms.max())
            out[T][j] = m + float(np.log(np.exp(terms - m).sum()))
    return out


def evaluate_passage(
    text: str,
    *,
    corpus: str,
    passage_id: int,
    model,
    tok,
    forward_last_logits,
    device: str,
    T_list: list[float],
    top_k: int,
    depth: int,
    beam_k: int,
    leaf_batch: int,
    max_prefix_len: int,
    positions: list[int],
    sigma_0: float,
    evidence_weight: float,
    memory_decay: float,
) -> list[PositionRecord]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < max(positions) + 2:
        return []
    ids = ids[: max_prefix_len + 1]
    ids_t = torch.tensor(ids, dtype=torch.long)

    # Advance the Laplace filter over the prefix to get state at each position.
    # One forward over the whole window for the base logits at every position.
    with torch.no_grad():
        out = model(
            input_ids=ids_t.unsqueeze(0).to(device),
            attention_mask=torch.ones_like(ids_t).unsqueeze(0).to(device),
            use_cache=False,
        )
    all_logits = out.logits[0].float().cpu()  # (Lwin, V)

    state = init_laplace(1, sigma_0=sigma_0, dtype=torch.float32)
    records: list[PositionRecord] = []
    target_positions = sorted(p for p in positions if p < len(ids) - 1)

    for t in range(1, len(ids) - 1):
        logits_t = all_logits[t]  # predicts token t+1
        if t in target_positions:
            log_p_full = torch.log_softmax(logits_t, dim=-1)
            cand = torch.topk(log_p_full, k=min(top_k, logits_t.shape[-1])).indices
            cand = cand.sort().values
            prefix_ids = ids_t[: t + 1]  # tokens x_1..x_{t+1}? -> committed prefix
            c_by_T = leaf_log_F_depth(
                forward_last_logits,
                prefix_ids,
                cand,
                T_list=T_list,
                depth=depth,
                beam_k=beam_k,
                leaf_batch=leaf_batch,
            )
            theta = candidate_filter_latent(
                state,
                logits_t,
                cand,
                evidence_weight=evidence_weight,
                memory_decay=memory_decay,
            )
            # full-vocab E_q[ell] = sum_v softmax(ell)_v * ell_v (filter centering)
            q_full = torch.softmax(logits_t, dim=-1)
            E_q = float(torch.sum(q_full * logits_t).item())
            covered = {}
            for T in T_list:
                w = (log_p_full / T)
                wm = float(w.max())
                full_Z = wm + float(torch.log(torch.exp(w - wm).sum()))
                cw = w[cand]
                cwm = float(cw.max())
                cand_Z = cwm + float(torch.log(torch.exp(cw - cwm).sum()))
                covered[str(T)] = float(np.exp(cand_Z - full_Z))
            records.append(
                PositionRecord(
                    corpus=corpus,
                    passage_id=passage_id,
                    position=t,
                    candidates=[int(x) for x in cand.tolist()],
                    ell=[float(x) for x in logits_t[cand].tolist()],
                    log_p=[float(x) for x in log_p_full[cand].tolist()],
                    E_q=E_q,
                    covered_mass=covered,
                    c_by_T={str(T): [float(x) for x in c_by_T[T].tolist()] for T in T_list},
                    theta=[float(x) for x in theta.tolist()],
                )
            )
        # advance filter with the real next token
        x_next = ids_t[t + 1].unsqueeze(0)
        state, _s, _f = laplace_update(
            state,
            logits_t.unsqueeze(0),
            x_next,
            evidence_weight=evidence_weight,
            memory_decay=memory_decay,
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GPT2_LARGE)
    ap.add_argument("--temperatures", default="0.5,0.7,0.9")
    ap.add_argument("--corpora", default="wikitext,writingprompts")
    ap.add_argument("--n-passages", type=int, default=40)
    ap.add_argument("--positions", default="24,48,72,96,120")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--beam-k", type=int, default=8)
    ap.add_argument("--leaf-batch", type=int, default=32)
    ap.add_argument("--max-prefix-len", type=int, default=160)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--evidence-weight", type=float, default=1.0)
    ap.add_argument("--memory-decay", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--time-budget-end", type=float, default=0.0,
                    help="epoch seconds; stop launching new passages after this")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    T_list = [float(x) for x in args.temperatures.split(",")]
    corpora = args.corpora.split(",")
    positions = [int(x) for x in args.positions.split(",")]
    model_tag = args.model.split("/")[-1]
    out_dir = Path(args.out_dir or f"results/full_sequence/go_no_go/{model_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"

    def log(d):
        d["t"] = round(time.time(), 1)
        with progress.open("a") as f:
            f.write(json.dumps(d) + "\n")
            f.flush()

    torch.manual_seed(args.seed)
    dtype = torch.float32 if "gpt2" in args.model else torch.bfloat16
    log({"event": "load_model", "model": args.model, "dtype": str(dtype)})
    model, tok = load_model(args.model, args.device, dtype)
    forward_last_logits = make_forward_last_logits(model, args.device)
    log({"event": "model_loaded", "vocab": int(model.config.vocab_size)})

    min_tokens = args.max_prefix_len + 2
    all_records: list[PositionRecord] = []
    for corpus in corpora:
        if corpus == "wikitext":
            texts = load_wikitext_passages(n=args.n_passages, min_tokens=min_tokens, tokenizer=tok, seed=args.seed)
        elif corpus == "writingprompts":
            texts = load_writingprompts_passages(n=args.n_passages, min_tokens=min_tokens, tokenizer=tok, seed=args.seed)
        else:
            raise ValueError(f"unknown corpus {corpus}")
        log({"event": "corpus_loaded", "corpus": corpus, "n": len(texts)})
        for pid, text in enumerate(texts):
            if args.time_budget_end and time.time() > args.time_budget_end:
                log({"event": "time_budget_stop", "corpus": corpus, "done_passages": pid})
                break
            recs = evaluate_passage(
                text, corpus=corpus, passage_id=pid, model=model, tok=tok,
                forward_last_logits=forward_last_logits, device=args.device,
                T_list=T_list, top_k=args.top_k, depth=args.depth, beam_k=args.beam_k,
                leaf_batch=args.leaf_batch, max_prefix_len=args.max_prefix_len,
                positions=positions, sigma_0=args.sigma0,
                evidence_weight=args.evidence_weight, memory_decay=args.memory_decay,
            )
            all_records.extend(recs)
            if pid % 5 == 0:
                log({"event": "progress", "corpus": corpus, "passage": pid,
                     "n_records": len(all_records)})

    log({"event": "eval_done", "n_records": len(all_records)})

    # Aggregate overall and per corpus.
    summary = {
        "config": vars(args),
        "model": args.model,
        "temperatures": T_list,
        "n_records_total": len(all_records),
        "overall": aggregate(all_records, T_list, seed=args.seed),
        "by_corpus": {
            c: aggregate([r for r in all_records if r.corpus == c], T_list, seed=args.seed)
            for c in corpora
        },
    }
    summary["decision_overall"] = go_no_go_decision(summary["overall"])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # also dump raw per-position records for re-analysis
    with (out_dir / "records.jsonl").open("w") as f:
        for r in all_records:
            f.write(json.dumps(r.__dict__) + "\n")
    log({"event": "summary_written", "path": str(out_dir / "summary.json"),
         "ANY_GO": summary["decision_overall"]["ANY_GO"]})
    print(f"WROTE {out_dir / 'summary.json'}  ANY_GO={summary['decision_overall']['ANY_GO']}")


if __name__ == "__main__":
    main()
