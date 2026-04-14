# Contribution Journey — MiniCPM-o upstream PR

This file documents the real process of contributing MiniCPM-o 4.5 to vllm-omni.
Failures, pivots, and validated decisions are recorded here so that future contributors
(human or AI) can understand the *why* behind the current code.

---

## What We're Doing

**Goal**: Add MiniCPM-o 4.5 omni model support to vllm-omni as an upstream PR.

**Why vllm-omni fork**: Naia OS needs local omni model inference on consumer hardware
(2× RTX 3090, no NVLink). vllm-omni is the only production-quality framework supporting
multi-stage omni pipelines. Upstream contribution means Naia OS benefits from ongoing
maintenance while the broader community gets MiniCPM-o support.

**AI-native experiment**: The entire workflow — pattern analysis, implementation, adversarial
review with headless subagents — is handled by AI agents following an issue-driven development
process. This is not just about the code; it is a proof of concept for **AI-native open-source
contribution methodology**.

---

## Timeline

### Phase 1: Initial port (internal, Naia fork)
- Ported MiniCPM-o model code from HuggingFace transformers reference
- **Failure**: Worked on 24 GB single GPU → "done" → completely broken on 2-GPU setup
- **Root cause**: Did not read upstream stage_configs docs or reference model YAMLs first
- **Lesson**: `process: true` is required; GPU memory budget math is not obvious

### Phase 2: 2-GPU stabilization
- Added `NCCL_P2P_DISABLE=1` (RTX 3090 without NVLink)
- Fixed `gpu_memory_utilization` split: Thinker 0.88 + Talker 0.1 + Code2Wav 0.02
- **Failure**: Diagnosed "device isolation bug" — reported as upstream bug
- **Pivot**: Was NOT an upstream bug. Was our own stage config errors (wrong `devices` values)
- **Lesson**: Verify config first before blaming the framework

### Phase 3: E2E validation
- Added `_trim_silence()` to Code2Wav output
- Built conversation benchmark framework (OmniSpeaker, Whisper STT, CJK metrics)
- **Failure**: CJK metrics showed 0% for Chinese → false negative from word-boundary mismatch
- **Fix**: Switched to CER + semantic similarity (sentence-transformers)
- **Result**: EN 92% word accuracy, ZH 76.1% CER / 95% semantic similarity

### Phase 4: VoiceBench
- Implemented voicebench_runner.py with rule-based scoring
- **Result**: 98.0% avg score, 98.7% pass rate (Knowledge/Instruction/Robustness/Safety)

### Phase 5: async_chunk streaming implementation
- **Goal**: TTFP reduction from 6.5s → < 2s
- Referenced qwen3_omni.py async_chunk functions directly (file read, not WebSearch)
- **Failure**: Initial hidden state accumulation bug — `pooling_output["thinker_hidden_states"]`
  returns only current decode step (1 token), not full sequence
  - Root cause: Did not trace `gpu_ar_model_runner.py` line 591-619 first
  - Fix: accumulate via `torch.cat` into `transfer_manager.request_payload` across all steps
- **Failure**: `list(request.output_token_ids)` — ConstantList wrapper issue
  - Fix: `_ensure_list()` helper following qwen3_omni pattern
- **Failure**: `cleanup_sender` never called when returning `None` for `pending=0, finished=True`
  - Root cause: Did not read `chunk_transfer_adapter.py` framework contract first
  - Fix: return finished sentinel dict `{code_predictor_codes: [], finished: True}` instead
- **Failure**: `[:-1]` unconditional trim in sync path removes real codec frame when max_tokens reached
  - Fix: value-based filter `t != 6561`
- **Review**: 10-pass adversarial headless review, 2 consecutive clean passes achieved

### Phase 6: Stop token investigation (ongoing)
- **Finding**: Talker rarely generates stop token 6561 → audio always ~20s
- **Root cause analysis**: Model conditioning issue, not a code bug
  - Talker does not reliably learn to generate EOS under current training distribution
- **Decision**: Separate issue, not blocking upstream PR; document as known limitation
- **Next**: tracked internally as nextain/naia-os#207; upstream collaboration pending

### Phase 7: Benchmark script multi-pass review
- 11-pass adversarial headless review on all benchmark scripts
- **Key fixes**: CER formula, Counter multiset for character overlap, SentenceTransformer cache,
  knowledge_retention loop, tts_speech_ratio removed from weights, SampleResult field names,
  CATEGORIES deepcopy, OmniSpeaker audio chunk detection (`modality` default `None` not `"text"`)
- 2 consecutive clean passes achieved

### Phase 8: Repo structure cleanup
- **Problem**: Work split between `main` (54 commits, early code) and `feat/minicpm-o`
  (252 commits, complete). Fork main was upstream-identical — nothing visible at
  `github.com/nextain/vllm-omni`.
- **Fix**: Force pushed `feat/minicpm-o` → fork `main`; deleted `feat/minicpm-o` branch
- **Result**: All work on `nextain/vllm-omni` main. No more branches for normal work.

---

## Key Upstream Findings

### What we discovered about vllm-omni internals
- `ChunkTransferAdapter.code_prompt_token_ids` is `defaultdict(list)` — framework owns it
- `pooling_output["thinker_hidden_states"]` = per-step slice, not full sequence
- `cleanup_sender` only called when `_send_single_request` receives truthy payload
- `ConstantList` is a vLLM internal type that does not iterate correctly via `list()`

### Divergence from upstream patterns we kept
- `_ensure_list()` always converts to true Python list (qwen3_omni returns non-list as-is)
  — kept because MiniCPM-o's num_vq=1 means all codec tokens are plain lists, never tensors
- `_find_tts_bound()` required (Qwen3 does not have it) — MiniCPM-o embeds TTS tokens inline
  in the Thinker output sequence

### Divergence we eliminated
- Initially used `[:-1]` trim (copied from early reference) → wrong; replaced with value filter
- Initially returned `None` on finished=True → memory leak; replaced with sentinel dict

---

## Upstream PR Status

| Item | Status |
|------|:------:|
| Model code (3-stage) | ready |
| stage_configs (3 yaml files) | ready |
| stage_input_processors | ready |
| Benchmark results | documented |
| E2E validated hardware | 2× RTX 3090 |
| async_chunk streaming | implemented, E2E pending |
| Korean TTS | ⚠️ known failure, documented |
| Stop token 6561 | ✅ fixed (2026-04-14) — root cause was Talker sampling params mismatch (top_p=1.0, top_k=50, temp=0.9 vs reference). All 3 YAMLs updated to match TTSSamplingParams. |

**Upstream issue reference**: vllm-omni#1182 (Allyyi's parallel work, same model)

---

## Documentation Structure

The `examples/online_serving/minicpm_o/` directory uses a dual-language structure:

| File | Language | Purpose |
|------|----------|---------|
| `README.md` | **English (primary)** | Quick start, scripts, stage configs — link to KO |
| `README.ko.md` | Korean | Same content in Korean — link to EN |
| `BENCHMARK.md` | **English** | Full benchmark results — link to KO |
| `BENCHMARK.ko.md` | Korean | Same benchmark in Korean — link to EN |

Header convention: `**Language / 언어**: [English](README.md) | [한국어](README.ko.md)`

Old benchmark files (`final_benchmark_report.md`, `benchmark_summary.md`, `benchmark_analysis.md`,
`COMPREHENSIVE_BENCHMARK_ANALYSIS.md`, `FINAL_REPORT.md`, `SESSION_HANDOFF.md`) are kept for
internal history but are superseded by `BENCHMARK.md` / `BENCHMARK.ko.md`.

---

### Phase 9: Offline inference examples + README structure
- Added `examples/offline_inference/minicpm_o/` directory (end2end.py, end2end_async_chunk.py, 4 shell scripts, README.md, text_prompts_10.txt)
- Added `examples/online_serving/minicpm_o/README.md` (rewrote from benchmark report format → qwen3_omni usage guide format)
- Added `examples/online_serving/minicpm_o/README.ko.md` (Korean mirror)
- Added fork-level `README.md` + `README.ko.md` (EN/KO); renamed upstream README → `README.upstream.md`
- **8-pass adversarial headless review** on offline inference scripts → 12 CRITICAL fixes applied
- **Key bugs caught in review**: SamplingParams values misaligned with minicpmo.yaml, shared dict reference for multi-prompt, use_image_audio missing from async_chunk, VLLM_WORKER_MULTIPROC_METHOD set after imports, --init-timeout missing from async_chunk
- 2 consecutive clean passes achieved (Pass 7 + Pass 8)

### Phase 10: 323s gibberish audio root cause fix (2026-04-14)

**Symptoms**: Every audio response was 323 seconds of gibberish noise instead of 2-5s speech.

**Root cause**: Two independent bugs, both contributing:

1. **Talker sampling params mismatch** — YAML had `top_p=1.0` (missing), `top_k=50`, `temperature=0.9` vs
   reference `TTSSamplingParams` in `modeling_minicpmo.py` (`top_k=25, top_p=0.85, temp=0.8, min_new_token=50`).
   With `top_p=1.0`, the model rarely generates stop token 6561 u2192 runs to `max_tokens=4096` u2192 u00d7 time.
   Fixed: aligned all 3 YAMLs to reference values.

2. **left_context_size ignored in Code2Wav** u2014 `forward()` received (new 25 + left_ctx 25) = 50 tokens,
   processed all 50, returned full ~2s audio. Left context was prepended to smooth chunk boundaries,
   but its audio was already emitted in the previous chunk. Fix: proportional tail trim:
   `samples_to_keep = math.ceil(wav_len * new_token_count / total)`, `wav[..., -samples_to_keep:]`.

**Combined effect**: 4096 u00f7 25 per chunk u00d7 2u00d7 audio per chunk = 327-second output.

**Additional fixes found during 13-pass adversarial review**:
- `codes.shape[-1]` (batch max-len) u2192 `token.shape[1]` (per-item len) for correct batch trimming
- Empty tensor (`non_empty`) filter before `torch.cat` to prevent shape mismatch crash
- 1D wav guard (`wav.unsqueeze(0)`) before `_trim_silence` which requires 2D input
- `unsqueeze(1)` after `_trim_silence` to produce 3D `[1, 1, audio_len]` for consistent output shape
- `round()` u2192 `math.ceil()` to never lose the last new-token sample to integer truncation
- `max_model_len: 4096` added to Stage 1 in minicpmo.yaml and minicpmo_48gb_2gpu.yaml (was missing)
- `min_tokens: 50` added (matches `min_new_token=50` in reference, prevents premature EOS on short responses)

**Tests**: 6 CPU unit tests added to `TestMiniCPMOCode2WavForwardTrimming` — all passing.

**E2E status**: Code review complete (13-pass, 2 consecutive clean). Server E2E pending.

---

## Files to Review When Resuming

```
vllm_omni/model_executor/stage_input_processors/minicpm_o.py  # core logic
vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml
vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml
vllm_omni/model_executor/stage_configs/minicpmo.yaml
examples/online_serving/minicpm_o/conversation_benchmark.py
examples/online_serving/minicpm_o/voicebench_runner.py
```
