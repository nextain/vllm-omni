# MiniCPM-o 4.5 — vllm-omni Deployment Guide

## Overview

MiniCPM-o 4.5 (`openbmb/MiniCPM-o-4_5`) is an omni-modal model supporting
text + vision + audio input → text + audio output via a 3-stage pipeline:

- **Stage 0 (Thinker)**: SigLIP2 + Whisper + Qwen3-7B → text generation + hidden states
- **Stage 1 (Talker)**: MiniCPMTTS Llama AR → audio codec tokens (num_vq=1)
- **Stage 2 (Code2Wav)**: CosyVoice2 flow + HiFi-GAN → audio waveform

## Quick Start

### Prerequisites

- GPU: ≥24GB VRAM (RTX 3090, A5000, A6000, etc.)
- Python: 3.10–3.13
- Dependencies: `pip install minicpmo-utils hyperpyyaml onnx`

### Start Server

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
```

### API Usage

#### Text → Audio (Omni mode)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [
      {"role": "user", "content": "Say hello in a friendly tone."}
    ],
    "modalities": ["audio"],
    "max_tokens": 64
  }'
```

Response contains audio in `choices[0].message.audio.data` as base64-encoded WAV.

#### Text → Text only (Thinker only)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "modalities": ["text"],
    "max_tokens": 64
  }'
```

#### Image + Text → Audio

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "Describe this image."}
      ]}
    ],
    "modalities": ["audio"],
    "max_tokens": 128
  }'
```

#### Per-stage Sampling Parameters

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "Hello"}],
    "modalities": ["audio"],
    "sampling_params_list": [
      {"temperature": 0.4, "top_p": 0.9, "max_tokens": 256},
      {"temperature": 0.9, "top_k": 50, "max_tokens": 2048, "stop_token_ids": [6561]},
      {"temperature": 0.0, "max_tokens": 5000}
    ]
  }'
```

### Python Client

```python
import base64
import requests
import soundfile as sf
import io
import numpy as np

resp = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "Say hello."}],
    "modalities": ["audio"],
    "max_tokens": 64,
})
data = resp.json()

# Extract audio
audio_b64 = data["choices"][0]["message"]["audio"]["data"]
audio_bytes = base64.b64decode(audio_b64)

# Save as WAV
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

## Hardware Requirements

| Config | GPU | Memory Split |
|--------|-----|-------------|
| Single GPU | RTX 3090 24GB | Thinker 0.75 + Talker 0.07 + Code2Wav 0.02 |
| Single GPU | A6000 48GB | Thinker 0.50 + Talker 0.15 + Code2Wav 0.10 |
| Multi-GPU | 2× RTX 3090 | Use `--stage-id` for independent launch (RFC #937) |

### Single GPU Notes

- `enforce_eager: true` on all stages (saves CUDA graph memory)
- `max_model_len: 1400` on Thinker (tight fit for 24GB)
- `hf_config_name: tts_config` on Talker (critical for correct hidden_size=768)

## Stage Configuration

See `vllm_omni/model_executor/stage_configs/minicpmo.yaml` for the full config.

Key fields:
- `model_arch: MiniCPMOForConditionalGeneration` — unified entry for all stages
- `model_stage: thinker|talker|code2wav` — selects which sub-model to initialize
- `preserve_multimodal: true` — Stage 0 only (passes hidden states to Stage 1)
- `limit_mm_per_prompt: {image: 0, video: 0}` — Stage 1, 2 (no multimodal encoders)

## Supported Modalities

| Input | Supported | Notes |
|-------|:---------:|-------|
| Text | ✅ | Standard chat messages |
| Image | ✅ | Via `image_url` in message content |
| Audio | ✅ | Via `audio_url` — Whisper encoder processes input |
| Video | ❌ | Disabled (SigLIP expects batched tensors) |

| Output | Supported | Notes |
|--------|:---------:|-------|
| Text | ✅ | `modalities: ["text"]` |
| Audio | ✅ | `modalities: ["audio"]` — full WAV in response |
| Streaming audio | ❌ | `async_chunk: false` — full audio returned after generation |

## Known Limitations

- **Voice cloning**: Not supported (empty speaker embedding)
- **Video input**: Disabled (`limit_mm_per_prompt: {video: 0}`)
- **Code2Wav batch_size=1**: CosyVoice2 API constraint
- **Streaming audio**: Not yet (`async_chunk: false`); qwen3_omni has this feature
