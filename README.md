# nextain/vllm-omni — MiniCPM-o 4.5 Contribution Fork

> **Fork of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)**  
> Upstream README: [README.upstream.md](README.upstream.md)

---

## What This Fork Is

This repository adds **MiniCPM-o 4.5** omni model support to vllm-omni, developed as an upstream contribution.

**Practical goal**: Enable real-time voice conversation on consumer hardware (2× RTX 3090, no NVLink) — the hardware tier powering [Naia OS](https://github.com/nextain/naia-os), an AI-native desktop OS.

**Open-source goal**: Contribute MiniCPM-o 4.5 support to upstream vllm-omni so the broader community can serve this model efficiently on multi-GPU setups.

**Methodology experiment**: The entire workflow — upstream pattern analysis, implementation, and adversarial code review using headless subagents — was driven by an AI coding agent following an issue-driven development process. This is a proof of concept for **AI-native open-source contribution**: an AI agent that understands a complex framework, writes upstream-quality code, and self-reviews it to a production standard.

Upstream PR target: [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) · Related upstream issue: [#1182](https://github.com/vllm-project/vllm-omni/issues/1182)

---

## What We Added

### Model: 3-Stage Omni Pipeline

MiniCPM-o 4.5 uses a three-stage pipeline, each running on a dedicated GPU:

```
Input (text + image / audio)
  → Stage 0: Thinker  (GPU 0) — Multimodal LLM (Idefics2 vision + Whisper audio + Qwen3 backbone)
  → Stage 1: Talker   (GPU 1) — TTS AR codec generator (MiniCPMTTS Llama)
  → Stage 2: Code2Wav (GPU 1) — Audio synthesis (CosyVoice2 flow + HiFi-GAN vocoder)
Output: PCM audio stream
```

### Files Added

| File | Purpose |
|------|---------|
| `vllm_omni/model_executor/models/minicpm_o/configuration_minicpmo.py` | Model config class |
| `vllm_omni/model_executor/models/minicpm_o/minicpm_o.py` | Unified entry + talker preprocess |
| `vllm_omni/model_executor/models/minicpm_o/minicpm_o_thinker.py` | Thinker stage model |
| `vllm_omni/model_executor/models/minicpm_o/minicpm_o_talker.py` | Talker stage model |
| `vllm_omni/model_executor/models/minicpm_o/minicpm_o_code2wav.py` | Code2Wav stage model |
| `vllm_omni/model_executor/stage_input_processors/minicpm_o.py` | Stage-to-stage data transfer (sync + async_chunk) |
| `vllm_omni/model_executor/stage_configs/minicpmo.yaml` | Single 24 GB GPU config |
| `vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml` | 2× RTX 3090 sync config |
| `vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml` | 2× RTX 3090 streaming config (TTFP ~0.07s) |

### Files Modified

| File | Change |
|------|--------|
| `vllm_omni/model_executor/models/registry.py` | 6 MiniCPM-o model entries added |
| `pyproject.toml` | `sentence-transformers`, `scikit-learn` optional deps added |

---

## Benchmark Results

Tested on **2× RTX 3090** (48 GB total, no NVLink), `minicpmo_async_chunk.yaml`.

### VoiceBench (EN, text-only scoring)

| Category | Avg Score | Pass Rate |
|----------|----------:|----------:|
| Knowledge (20 MCQ) | 100.0% | 100.0% |
| Instruction (20) | 92.5% | 95.0% |
| Robustness (20) | 100.0% | 100.0% |
| Safety (15) | 100.0% | 100.0% |
| **Overall (75)** | **98.0%** | **98.7%** |

### Voice Conversation Quality

| Language | STT Accuracy | Latency | Status |
|----------|:------------|:--------|:------:|
| English | 92% word accuracy | 2.3s avg | ✅ Production ready |
| Chinese | 76.1% CER / 95% semantic | 1.5s avg | ✅ Working |
| Korean | Text OK / TTS garbled | — | ⚠️ CosyVoice2 not trained on KO |

### Latency (async_chunk mode)

| Stage | Time |
|-------|------|
| Thinker (text generation) | 1.5–2.5s |
| TTFP (time to first audio packet) | **~0.07s** |
| Total end-to-end | 2.0–4.0s |

Full results: [`examples/online_serving/minicpm_o/BENCHMARK.md`](examples/online_serving/minicpm_o/BENCHMARK.md)

---

## Quick Start

```bash
# Single 24 GB GPU
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --port 8000

# 2× RTX 3090 — sync mode
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml \
  --trust-remote-code --port 8000

# 2× RTX 3090 — streaming (recommended, TTFP ~0.07s)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --port 8000
```

> `NCCL_P2P_DISABLE=1` is required for RTX 3090 without NVLink.

Evaluation scripts and benchmark details: [`examples/online_serving/minicpm_o/`](examples/online_serving/minicpm_o/)

---

## Known Limitations

| Issue | Impact | Notes |
|-------|--------|-------|
| Stop token 6561 rarely generated | Audio fixed at ~20s | Talker training issue; `_trim_silence()` as workaround |
| Korean TTS | Garbled audio | CosyVoice2 not trained on Korean |

---

## Key Implementation Notes

- **`NCCL_P2P_DISABLE=1`** — required for RTX 3090 (no NVLink between GPUs)
- **`max_inflight: 1`** — prevents OOM from concurrent stage memory (upstream [#1387](https://github.com/vllm-project/vllm-omni/issues/1387))
- **`_find_tts_bound()`** — MiniCPM-o embeds TTS tokens inline in Thinker output; boundary detection required (unlike Qwen3-Omni)
- **`_ensure_list()`** — `ConstantList` vLLM internal type does not iterate correctly via `list()`; always convert explicitly
- **async_chunk hidden state accumulation** — `pooling_output["thinker_hidden_states"]` returns only the current decode step; must accumulate via `torch.cat` across all steps

See full implementation history and lessons learned: [`.agents/context/contribution-journey.md`](.agents/context/contribution-journey.md)

---

## License

Apache License 2.0 — same as upstream vllm-omni. See [LICENSE](./LICENSE).
