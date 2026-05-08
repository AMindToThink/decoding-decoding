# Long-Context Sampling Capacity: Qwen2.5-3B on cs29824

## Hardware
- **2× Quadro RTX 8000, 48 GB each → 96 GB total VRAM**
- Compute capability **7.5 (Turing)**. Implications:
  - **FlashAttention-2 does not run** (needs SM 8.0+).
  - **bf16 has no tensor-core path.** Use **fp16** despite `torch.cuda.is_bf16_supported()` returning True — that flag is misleading on Turing.
  - The fast attention path you actually get is PyTorch's `mem_efficient_sdp` backend.

## Model: Qwen2.5-3B-Instruct
- 36 layers, hidden 2048, 16 query heads, **2 KV heads (GQA)**, head_dim 128.
- 3.09 B params (~6.2 GB in fp16).
- Native `max_position_embeddings = 32,768`.

## Memory-limited maximum context

KV cache per token (fp16):

```
2 (K+V) · 36 layers · 2 KV heads · 128 head_dim · 2 bytes = 36,864 bytes ≈ 36 KB/token
```

Budget after weights (~6.2 GB) and ~10 GB headroom for activations / chunked-prefill workspace:

```
(96 - 6.2 - 10) GB / 36 KB ≈ 2.16M tokens
```

**Memory alone permits ~2 M tokens of KV cache.** The GQA ratio (2 KV heads) makes this model unusually cache-efficient.

## Model-quality cap (the real binding constraint)
- **32 K** native (no config changes).
- **131 K (128 K)** with YaRN — set in the loaded config:
  ```json
  "rope_scaling": {
      "type": "yarn",
      "factor": 4.0,
      "original_max_position_embeddings": 32768
  }
  ```
- **Beyond 128 K the RoPE was never trained**; outputs degrade quickly. The 2 M memory ceiling is a ceiling for the *cache*, not for coherent sampling.

## Prefill time to reach each ceiling

FLOP model: `2·N_params·L + 4·n_layers·L²·d_model`.

- RTX 8000 peak: ~130 TFLOPS fp16 tensor.
- Realistic on Turing without FA2: **~40–60 TFLOPS effective single-GPU**.
- vLLM with `--tensor-parallel-size 2`: **~80–90 TFLOPS aggregate**.
- HF `device_map="auto"` with batch=1: the second GPU is mostly idle during a single forward — pipeline parallelism only helps with microbatches.

| Context | Total FLOPs | HF single-GPU effective (~50 TF) | vLLM TP=2 (~85 TF) |
|---|---|---|---|
| 32 K | ~0.5 PFLOP | ~10 s | ~6 s |
| 128 K (YaRN) | ~5.9 PFLOP | ~2 min | ~70 s |
| 1 M (memory only, broken outputs) | ~300 PFLOP | ~100 min | ~60 min |
| 2 M (memory cap, broken outputs) | ~1.2 EFLOP | ~6–7 hr | ~3–4 hr |

The L² attention term overtakes the linear term around L ≈ 21 K, so past ~30 K the cost roughly **quadruples each time you double the context**.

## Practical recommendations

1. **Use vLLM with `--tensor-parallel-size 2`** rather than HF `device_map="auto"` if you want the second GPU to actually pull weight during prefill. vLLM also does chunked prefill automatically, so activation memory at long L isn't a problem.
2. **Stay ≤ 128 K** for meaningful sampling. If you push further, treat it as an explicit failure-mode / calibration study, not a usage scenario.
3. **Don't trust the table numbers blindly** — once a model is loaded, measure prefill time at 8 K, 32 K, 128 K and fit a quadratic. Constants on Turing-without-FA2 vary more than on Ampere.
4. **Load weights in fp16, not bf16**, on this hardware.
