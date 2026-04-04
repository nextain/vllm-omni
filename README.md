# nextain/vllm-omni — MiniCPM-o 4.5 Contribution Fork

**Language / 언어**: English | [한국어](README.ko.md)

> **Fork of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)**  
> Upstream README: [README.upstream.md](README.upstream.md) · Upstream PR target: [vllm-omni#1182](https://github.com/vllm-project/vllm-omni/issues/1182)

---

## What This Fork Adds

This fork adds **MiniCPM-o 4.5** omni model support to vllm-omni as an upstream contribution.

**Hardware target**: 2× RTX 3090 (48 GB total, no NVLink) — consumer-grade multi-GPU setup powering [Naia OS](https://github.com/nextain/naia-os), an AI-native desktop OS.

**Methodology**: The entire contribution workflow — upstream pattern analysis, implementation, and adversarial code review using headless subagents — was driven by an AI coding agent. This is a proof of concept for **AI-native open-source contribution**. Full implementation history: [`.agents/context/contribution-journey.md`](.agents/context/contribution-journey.md)

---

## Architecture

MiniCPM-o 4.5 uses a fully disaggregated 3-stage pipeline:

```
Input (text + image / audio)
  → Stage 0: Thinker  (GPU 0) — Idefics2 vision + Whisper audio + Qwen3 LLM backbone
  → Stage 1: Talker   (GPU 1) — MiniCPMTTS Llama AR codec generator
  → Stage 2: Code2Wav (GPU 1) — CosyVoice2 flow model + HiFi-GAN vocoder
Output: PCM audio stream
```

Key differences from Qwen3-Omni: single RVQ layer (num_vq=1), inline TTS token boundary detection required, 1D codec token sequence.

---

## Quick Start

### Offline Inference

```bash
cd examples/offline_inference/minicpm_o

# Single prompt (24 GB single GPU)
bash run_single_prompt.sh

# Single prompt with async_chunk streaming (2× RTX 3090)
bash run_single_prompt_async_chunk.sh

# Multiple prompts
bash run_multiple_prompts.sh
```

### Online Serving

```bash
# 2× RTX 3090 — async_chunk streaming (recommended)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000

# Single 24 GB GPU
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --port 8000
```

> `NCCL_P2P_DISABLE=1` is required for RTX 3090 without NVLink.

See [`examples/online_serving/minicpm_o/`](examples/online_serving/minicpm_o/) for client scripts and evaluation.  
See [`examples/offline_inference/minicpm_o/`](examples/offline_inference/minicpm_o/) for offline inference.

---

## Stage Config Options

| Config | Use case | GPU |
|--------|---------|-----|
| [`minicpmo.yaml`](vllm_omni/model_executor/stage_configs/minicpmo.yaml) | Single 24 GB GPU | RTX 3090 × 1 |
| [`minicpmo_48gb_2gpu.yaml`](vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml) | 2× 24 GB GPU, sync | RTX 3090 × 2 |
| [`minicpmo_async_chunk.yaml`](vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml) | 2× 24 GB GPU, streaming (TTFP ~0.07s) | RTX 3090 × 2 |

---

## Benchmark Results

Tested on **2× RTX 3090** (48 GB, no NVLink), `minicpmo_async_chunk.yaml`.

### VoiceBench (EN, text-only)

| Category | Pass Rate |
|----------|----------:|
| Knowledge (20) | 100.0% |
| Instruction (20) | 95.0% |
| Robustness (20) | 100.0% |
| Safety (15) | 100.0% |
| **Overall (75)** | **98.7%** |

### Voice Conversation Quality

| Language | Accuracy | Latency | Status |
|----------|:---------|:--------|:------:|
| English | 92% word accuracy | 2.3s avg | ✅ Production ready |
| Chinese | 76.1% CER / 95% semantic | 1.5s avg | ✅ Working |
| Korean | Text OK / TTS garbled | — | ⚠️ CosyVoice2 not trained on KO |

**TTFP** (time to first audio packet, async_chunk): **~0.07s**

Full benchmark report: [`examples/online_serving/minicpm_o/BENCHMARK.md`](examples/online_serving/minicpm_o/BENCHMARK.md)

---

## Files Added

| Path | Purpose |
|------|---------|
| [`vllm_omni/model_executor/models/minicpm_o/`](vllm_omni/model_executor/models/minicpm_o/) | Model code (Thinker / Talker / Code2Wav + config) |
| [`vllm_omni/model_executor/stage_input_processors/minicpm_o.py`](vllm_omni/model_executor/stage_input_processors/minicpm_o.py) | Stage-to-stage data transfer (sync + async_chunk) |
| [`vllm_omni/model_executor/stage_configs/`](vllm_omni/model_executor/stage_configs/) | 3 stage configs (single-GPU / 2-GPU / async_chunk) |
| [`examples/offline_inference/minicpm_o/`](examples/offline_inference/minicpm_o/) | Offline inference scripts |
| [`examples/online_serving/minicpm_o/`](examples/online_serving/minicpm_o/) | Online serving scripts + benchmark suite |

Modified: [`models/registry.py`](vllm_omni/model_executor/models/registry.py) (6 model entries), [`pyproject.toml`](pyproject.toml) (optional deps).

---

## Known Limitations

| Issue | Impact | Workaround |
|-------|--------|-----------|
| Stop token 6561 rarely generated | Audio fixed at ~20s | [`_trim_silence()`](vllm_omni/model_executor/models/minicpm_o/minicpm_o_code2wav.py) post-processing |
| Korean TTS failure | Garbled audio | CosyVoice2 not trained on Korean |

See [`BENCHMARK.md § Known Issues`](examples/online_serving/minicpm_o/BENCHMARK.md#6-known-issues) for full details.

---

## Implementation Notes

- `NCCL_P2P_DISABLE=1` — required for RTX 3090 (no NVLink)
- `max_inflight: 1` — prevents OOM from concurrent stage memory (upstream [#1387](https://github.com/vllm-project/vllm-omni/issues/1387))
- [`_find_tts_bound()`](vllm_omni/model_executor/stage_input_processors/minicpm_o.py) — MiniCPM-o embeds TTS tokens inline in Thinker output; boundary detection required
- [`_ensure_list()`](vllm_omni/model_executor/stage_input_processors/minicpm_o.py) — vLLM `ConstantList` type does not iterate correctly via `list()`

---

## License

Apache License 2.0 — same as upstream vllm-omni. See [LICENSE](./LICENSE).
