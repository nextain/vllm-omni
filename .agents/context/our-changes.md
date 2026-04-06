# Our Changes — vllm-omni Fork (Nextain)

## Branch: main (fork of vllm-project/vllm-omni)

**Repo structure**: All work lives on `nextain/vllm-omni` main. When submitting upstream,
create a clean branch from main and open a PR to `vllm-project/vllm-omni` main.

## Purpose

This fork adds MiniCPM-o 4.5 omni model support to vllm-omni as an upstream contribution.

**Why**: Naia OS (nextain/naia-os) requires local omni model inference. MiniCPM-o 4.5 is the
primary target for Naia hardware tiers (48 GB dual-GPU, 32 GB single-GPU setups). By contributing
upstream, the work benefits the broader open-source community while keeping Naia OS aligned with
the official framework.

**AI-native OSS experiment**: This entire contribution — from upstream pattern analysis through
implementation to headless adversarial review — was produced by an AI coding agent (Claude)
following an issue-driven development workflow. The methodology, not just the code, is being
validated here.

---

## Files Added

All paths relative to `vllm_omni/model_executor/`:

| File | Purpose |
|------|---------|
| `models/minicpm_o/__init__.py` | Module exports |
| `models/minicpm_o/configuration_minicpmo.py` | Config class (custom — not in transformers) |
| `models/minicpm_o/minicpm_o.py` | Unified entry point + talker_preprocess |
| `models/minicpm_o/minicpm_o_thinker.py` | Thinker: Idefics2 vision + Whisper audio + Qwen3 LLM |
| `models/minicpm_o/minicpm_o_talker.py` | Talker: MiniCPMTTS Llama AR codec generator |
| `models/minicpm_o/minicpm_o_code2wav.py` | Code2Wav: CosyVoice2 flow + HiFi-GAN vocoder |
| `stage_input_processors/minicpm_o.py` | thinker2talker, talker2code2wav (sync + async_chunk) |
| `stage_configs/minicpmo.yaml` | Single-GPU 24 GB reference config |
| `stage_configs/minicpmo_48gb_2gpu.yaml` | 2× RTX 3090 (48 GB total) config |
| `stage_configs/minicpmo_async_chunk.yaml` | async_chunk streaming config (TTFP optimization) |

## Files Modified

| File | Change |
|------|--------|
| `models/registry.py` | Added 6 MiniCPM-o model entries |
| `pyproject.toml` | Added `sentence-transformers`, `scikit-learn` optional deps |

## Offline Inference Examples (examples/offline_inference/minicpm_o/)

Added complete offline inference example directory:

| File | Purpose |
|------|---------|
| `end2end.py` | Sync offline inference via `Omni` class; text/image/audio/image+audio query types |
| `end2end_async_chunk.py` | Async offline inference via `AsyncOmni`; stage-level concurrency |
| `run_single_prompt.sh` | Single prompt convenience script |
| `run_multiple_prompts.sh` | Multi-prompt batch with `--py-generator` |
| `run_single_prompt_async_chunk.sh` | Single prompt with async_chunk (TTFP ~0.07s) |
| `run_multiple_prompts_async_chunk.sh` | Multi-prompt async_chunk with `--max-in-flight` |
| `text_prompts_10.txt` | Sample AI-related prompts for batch testing |
| `README.md` | Usage guide matching qwen3_omni offline format |

**SamplingParams alignment** (from 8-pass adversarial review):
- Thinker: `temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048, detokenize=True, repetition_penalty=1.05`
- Talker: `temperature=0.9, top_k=50, max_tokens=4096, detokenize=False, repetition_penalty=1.05, stop_token_ids=[6561]`
- Code2Wav: `temperature=0.0, top_p=1.0, top_k=-1, max_tokens=65536, detokenize=True, repetition_penalty=1.1`

---

## Benchmark Scripts (examples/online_serving/minicpm_o/)

| File | Changes (multi-pass adversarial review) |
|------|-----------------------------------------|
| `metrics/cjk_metrics.py` | CER formula fixed (SequenceMatcher → edit_distance/len(ref)); `_character_overlap` set→Counter multiset; CER lower-bound `max(0.0,...)`; `_sentence_model` module-level cache |
| `metrics/conversation_quality.py` | `knowledge_retention` loop fixed (`range(i+1)`→`range(2,len)` + reverse same-role search); `_character_overlap` Counter multiset; `tts_speech_ratio` removed from `overall_quality` weights; L25 docstring corrected |
| `conversation_benchmark.py` | Print label: `"Avg STT CER (lower=better)"` |
| `language_test.py` | `stt_cer`/`stt_word_accuracy` key split (was mixed-direction); `evaluate_conversation` receives `stt_text` + `response_time`; opener preview ellipsis conditional |
| `voicebench_runner.py` | `SampleResult` field names fixed (`label`→`score_label`, `elapsed_s`→`response_time_s`); `list_count` empty response guard; API exception handling (score=0.0 + continue); `CATEGORIES` `deepcopy` to prevent global mutate; `list_count` comma filter strips empty tokens |

---

## Architecture: 3-Stage Pipeline

```
Input (text + image/audio)
  → Stage 0: Thinker  (GPU 0, LLM AR)  → hidden_states
  → Stage 1: Talker   (GPU 1, TTS AR)  → codec token IDs
  → Stage 2: Code2Wav (GPU 1, vocoder) → PCM audio
```

**Key differences from Qwen3-Omni / MIMO-Audio:**

| | MiniCPM-o | Qwen3-Omni | MIMO-Audio |
|--|-----------|-----------|-----------|
| num_vq (RVQ layers) | **1** | 8 | N/A |
| Hidden state key | `thinker_hidden_states` | `thinker_prefill_embeddings` + `thinker_decode_embeddings` | fused |
| Stage transfer | SharedMemoryConnector (0→1), direct injection (1→2) | SharedMemoryConnector (all) | SharedMemoryConnector |
| Codec token dim | 1D flat | 8×N matrix | N/A |
| TTS boundary detection | `_find_tts_bound()` required | no (full seq) | no |
| async_chunk support | ✅ implemented | ✅ upstream | N/A |

---

## stage_input_processors/minicpm_o.py — Key Functions

### sync path
- `thinker2talker_sync`: accumulates `thinker_hidden_states` into `request_payload`; applies `_find_tts_bound()` to slice TTS segment; returns hidden + token IDs as input to Talker.
- `talker2code2wav`: reads Talker's `output.token_ids`, filters stop token 6561 by **value** (not `[:-1]`), returns codec frame list to Code2Wav.

### async_chunk path
- `thinker2talker_async_chunk`: accumulates per-step hidden states via `torch.cat` into `transfer_manager.request_payload`; sends only when `is_finished=True`.
- `talker2code2wav_async_chunk`: cursor-based incremental codec reading (`_talker_token_cursor`, `_talker_emitted_len`); emits in `codec_chunk_frames=25` batches; returns finished sentinel `{code_predictor_codes: [], finished: True}` (not None) to trigger `cleanup_sender`.

### Helpers
- `_ensure_list(x)`: converts `ConstantList` / tensor-like to plain Python list. Based on `qwen3_omni._ensure_list`; differs in always converting to a true list (not returning non-list types as-is).
- `_find_tts_bound(thinker_token_ids)`: finds `<|tts_bos|>` ... `<|tts_eos|>` boundary within Thinker output tokens to extract the TTS segment.

---

## Benchmark Results (2× RTX 3090, async_chunk)

| Language | STT Accuracy | Latency | Grade |
|----------|:------------|:--------|:------|
| English | 92% (word) | 2.3s avg | A |
| Chinese | 76.1% CER / 95% semantic | 1.5s avg | B+ |
| Korean | text OK / TTS failed | — | F (TTS) |

**VoiceBench (EN, text-only scoring)**:
- Knowledge: 100% · Instruction: 95% · Robustness: 100% · Safety: 100%
- Overall: 98.0% avg, 98.7% pass rate

**Known limitations:**
- Stop token 6561 rarely generated by Talker → audio fixed at ~20s → tracked as separate issue
- Korean TTS: CosyVoice2 not trained on Korean → requires fine-tuning
- streaming TTFP measured: ~0.07s (first audio packet after Thinker finishes; baseline sync: 6.5s)

**OmniSpeaker client bug (fixed):**
- `getattr(chunk, "modality", "text")` default caused audio chunks to always fall into text branch
- Fixed: default changed to `None`, `elif modality == "text"` pattern (upstream gradio_demo.py aligned)
- Commit: `40576264`

---

## WebSocket Endpoint Status (as of 2026-04-06)

| Endpoint | Status | Notes |
|----------|:------:|-------|
| POST /v1/chat/completions (non-stream) | ✅ E2E working | text + audio returned together, ~28s |
| POST /v1/chat/completions (stream=True SSE) | ✅ confirmed | audio chunks as base64 WAV, 0.9s TTFP |
| GET /v1/realtime (WebSocket) | ⚠️ ASR only | transcription.delta/done only, no audio output |
| GET /v1/omni (WebSocket) | ✅ **implemented** (c3ad9985) | full-duplex: PCM16 in → WAV chunks out |

## /v1/omni Protocol

See `vllm_omni/entrypoints/openai/serving_omni_duplex.py` for full docs.

Client sends: session.config (text) -> binary PCM16 frames -> input.done (text)  
Server sends: turn.start -> transcript.delta -> audio.start -> binary WAV chunks -> audio.done -> turn.done

**Key design decisions:**
- Reuses `OmniOpenAIServingChat.create_chat_completion(raw_request=None)` - no new inference path
- `format: "wav_chunk"` - each binary frame is a self-contained WAV
- `abort_id = f"chatcmpl-{pre_uuid}"` pre-computed for safe early-disconnect abort
- Does NOT use `_add_streaming_input_request` (PR #2208) - MiniCPM-o needs full Thinker output; no TTFP benefit

**Known v1 limitations:** Unbounded conversation history; assistant audio not in history; E2E test pending.

---

## Trust State

| Item | Status | Notes |
|------|:------:|-------|
| Model code (thinker/talker/code2wav) | ✅ | 2× RTX 3090 E2E validated |
| minicpmo.yaml | ✅ | 10-pass adversarial review, 2 consecutive clean |
| minicpmo_48gb_2gpu.yaml | ✅ | Same param fixes applied, 10-pass review |
| minicpmo_async_chunk.yaml | ✅ | Same param fixes applied, 10-pass review |
| stage_input_processors/minicpm_o.py | ✅ | 10-pass adversarial review, 2 consecutive clean |
| registry.py | ✅ | Additive only |
| Audio input (MiniCPMO processor) | ⚠️ | Unvalidated on 2-GPU text→audio path |
| metrics/cjk_metrics.py | ✅ | Multi-pass review, metric formula fixes applied |
| metrics/conversation_quality.py | ✅ | Multi-pass review, knowledge_retention + weights fixed |
| conversation_benchmark.py | ✅ | Label correction |
| language_test.py | ✅ | stt_cer/word_accuracy separation, evaluate_conversation fix |
| voicebench_runner.py | ✅ | SampleResult fields, exception handling, CATEGORIES deepcopy |
| offline_inference/minicpm_o/end2end.py | ✅ | 8-pass adversarial review; SamplingParams aligned |
| offline_inference/minicpm_o/end2end_async_chunk.py | ✅ | 8-pass review; init_timeout, use_image_audio added |
| offline_inference/minicpm_o/run_*.sh | ✅ | File existence checks, PROMPTS_FILE variable reuse |
| serving_omni_duplex.py | ✅ | 3-pass adversarial review (3 CRITICALs fixed); /v1/omni full-duplex |
| api_server.py (/v1/omni route) | ✅ | Additive only; null guard for chat service |

---

## Lessons Learned

1. **Upstream patterns first** — read qwen3_omni.py, qwen2_5_omni.py, mimo_audio.py before writing
2. **ConstantList wrapping** — `list(request.output_token_ids)` unsafe; use `_ensure_list()`
3. **Per-step hidden states** — `pooling_output["thinker_hidden_states"]` = current-step slice only; must accumulate across all decode steps
4. **cleanup_sender contract** — only called when payload is truthy; return finished sentinel dict, not None, to avoid memory leak
5. **`[:-1]` trim is wrong** — when max_tokens reached without EOS, unconditional trim drops a real frame; filter by value instead
6. **NCCL_P2P_DISABLE=1** — required for RTX 3090 without NVLink
7. **Code2Wav GPU memory** — CosyVoice2 flow model ~15 GB; allocate accordingly
8. **max_inflight: 1** — prevents OOM from concurrent stage memory (vllm-omni#1387)
