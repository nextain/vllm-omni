# MiniCPM-o 4.5 Offline Inference

## Setup

Please refer to the [stage configuration documentation](https://vllm-omni.readthedocs.io/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

```bash
pip install soundfile librosa
sudo apt-get install ffmpeg  # required for librosa MP3/audio backend
```

## Run Examples

### Single Prompt

Get into the example folder:
```bash
cd examples/offline_inference/minicpm_o
```

Then run:
```bash
bash run_single_prompt.sh
```

### Multiple Prompts

For processing larger batches, uses `py_generator` mode to avoid holding all outputs in memory:
```bash
bash run_multiple_prompts.sh
```

### Modality Control

To output text only (skip Talker and Code2Wav stages):
```bash
python end2end.py --output-dir output_audio \
                  --query-type use_audio \
                  --modalities text
```

### Using Local Media Files

`end2end.py` supports local image and audio files:

```bash
# Use local audio file
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav

# Use local image file
python end2end.py --query-type use_image --image-path /path/to/image.jpg

# Use both image and audio
python end2end.py --query-type use_image_audio \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav
```

If file paths are not provided, the script uses built-in vLLM sample assets.

Supported query types:
- `text` — text-only input
- `use_image` — image + text input
- `use_audio` — audio + text input
- `use_image_audio` — image + audio + text input

## Async-Chunk (Offline)

For true stage-level concurrency — where Talker and Code2Wav start **before**
Thinker finishes — use the async_chunk example. This reduces TTFP from ~6.5s
to ~0.07s on 2× RTX 3090.

Requirements:
1. A stage config YAML with `async_chunk: true` (e.g. [`minicpmo_async_chunk.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml))
2. 2× GPU setup (2× RTX 3090 or equivalent)
3. `NCCL_P2P_DISABLE=1` for RTX 3090 without NVLink

`end2end_async_chunk.py` uses `AsyncOmni` instead of the synchronous `Omni` class.

### Single prompt
```bash
cd examples/offline_inference/minicpm_o
bash run_single_prompt_async_chunk.sh
```

### Multiple prompts with concurrency control
```bash
bash run_multiple_prompts_async_chunk.sh --max-in-flight 2
```

### Text-only output (skip audio generation)
```bash
python end2end_async_chunk.py --query-type text --modalities text
```

### Custom stage config
```bash
python end2end_async_chunk.py \
    --query-type use_audio \
    --stage-configs-path /path/to/minicpmo_async_chunk.yaml
```

> **Note**: The synchronous `end2end.py` (using `Omni`) is the recommended entry
> point for single-GPU or sync workflows. Use `end2end_async_chunk.py` only when
> you need stage-level concurrency on a multi-GPU async_chunk setup.

## Stage Config Options

| Config | Use case | GPU |
|--------|---------|-----|
| [`minicpmo.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo.yaml) | Single 24 GB GPU | RTX 3090 × 1 |
| [`minicpmo_48gb_2gpu.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml) | 2× 24 GB GPU, sync | RTX 3090 × 2 |
| [`minicpmo_async_chunk.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml) | 2× 24 GB GPU, streaming (TTFP ~0.07s) | RTX 3090 × 2 |

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
CosyVoice2 (the Code2Wav backbone) was not trained on Korean. Korean text generation works correctly, but audio synthesis requires fine-tuning.

**Audio always ~20 seconds:**  
The Talker stage rarely generates stop token 6561 under current training. `_trim_silence()` post-processing in [`minicpm_o_code2wav.py`](../../../vllm_omni/model_executor/models/minicpm_o/minicpm_o_code2wav.py) removes trailing silence.
