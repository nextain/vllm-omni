# MiniCPM-o 4.5

**Language / 언어**: English | [한국어](README.ko.md)

## Setup

Please refer to the [stage configuration documentation](https://vllm-omni.readthedocs.io/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

### Prerequisites

```bash
pip install soundfile librosa
sudo apt-get install ffmpeg  # required for librosa MP3 backend
```

## Run Examples (MiniCPM-o 4.5)

### Launch the Server

```bash
# 2× RTX 3090 — async_chunk streaming (recommended, TTFP ~0.07s)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000

# 2× RTX 3090 — sync mode
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000

# Single 24 GB GPU
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000
```

> `NCCL_P2P_DISABLE=1` is required for RTX 3090 without NVLink.

Stage config options: [`minicpmo.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo.yaml) · [`minicpmo_48gb_2gpu.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml) · [`minicpmo_async_chunk.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml)

### Send Requests

Get into the example folder:
```bash
cd examples/online_serving/minicpm_o/
```

#### Text conversation

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

#### Audio output (text → speech)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "Describe vLLM in one sentence."}],
    "modalities": ["audio"]
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
audio = data['choices'][0]['message']['audio']['data']
open('output.wav', 'wb').write(base64.b64decode(audio))
print('Saved output.wav')
"
```

#### Image + text → audio

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"}},
        {"type": "text", "text": "What is in this image? Answer in one sentence."}
      ]
    }],
    "modalities": ["audio"]
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
audio = data['choices'][0]['message']['audio']['data']
open('output_image.wav', 'wb').write(base64.b64decode(audio))
print('Saved output_image.wav')
"
```

#### Audio input → audio output

```bash
# Encode local audio file to base64
AUDIO_B64=$(base64 -w 0 /path/to/audio.wav)

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"openbmb/MiniCPM-o-4_5\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"input_audio\", \"input_audio\": {\"data\": \"${AUDIO_B64}\", \"format\": \"wav\"}},
        {\"type\": \"text\", \"text\": \"What is said in this audio??\"}
      ]
    }],
    \"modalities\": [\"audio\"]
  }" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
audio = data['choices'][0]['message']['audio']['data']
open('output_audio.wav', 'wb').write(base64.b64decode(audio))
print('Saved output_audio.wav')
"
```

## Modality Control

You can control which output modalities the model generates.

### Supported modalities

| Modalities | Output |
|------------|--------|
| `["text"]` | Text only (skip audio generation — faster) |
| `["audio"]` | Audio only |
| `["text", "audio"]` | Text + Audio |
| Not specified | Text + Audio (default) |

### Text only (skip Talker + Code2Wav stages)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "Explain vLLM in brief."}],
    "modalities": ["text"]
  }'
```

### Using OpenAI Python SDK

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Text only
response = client.chat.completions.create(
    model="openbmb/MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    modalities=["text"],
)
print(response.choices[0].message.content)

# Audio output
response = client.chat.completions.create(
    model="openbmb/MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    modalities=["audio"],
)
audio_data = base64.b64decode(response.choices[0].message.audio.data)
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## Streaming Output

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "Tell me a short story."}],
    "modalities": ["text"],
    "stream": true
  }'
```

For audio streaming, use the `async_chunk` stage config (TTFP ~0.07s):
- [`minicpmo_async_chunk.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml)

## Evaluation Scripts

| Script | Purpose |
|--------|---------|
| [`conversation_benchmark.py`](conversation_benchmark.py) | Multi-turn EN conversation benchmark (6 scenarios) |
| [`language_test.py`](language_test.py) | EN / ZH / KO comparison (CER + semantic similarity) |
| [`voicebench_runner.py`](voicebench_runner.py) | VoiceBench (Knowledge / Instruction / Robustness / Safety) |
| [`e2e_conversation_test.py`](e2e_conversation_test.py) | Core Speaker / Monitor framework (imported by above) |
| [`metrics/cjk_metrics.py`](metrics/cjk_metrics.py) | CER, semantic similarity (sentence-transformers) |
| [`metrics/conversation_quality.py`](metrics/conversation_quality.py) | Relevance, coherence, knowledge retention |

```bash
# Run conversation benchmark
python conversation_benchmark.py --omni

# Run VoiceBench
python voicebench_runner.py

# Run language test (EN / ZH / KO)
python language_test.py
```

Full benchmark results: [BENCHMARK.md](BENCHMARK.md)

## FAQ

**librosa backend error:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**NCCL P2P error on RTX 3090:**
```bash
export NCCL_P2P_DISABLE=1
```

**Korean TTS outputs garbled audio:**  
CosyVoice2 (Code2Wav backbone) was not trained on Korean. Korean text generation works correctly, but audio synthesis requires fine-tuning. See [BENCHMARK.md § Korean](BENCHMARK.md#33-korean-ko).

**Audio always ~20 seconds:**  
The Talker stage rarely generates stop token 6561 under current training. `_trim_silence()` post-processing removes trailing silence. Tracked separately from the upstream PR.
