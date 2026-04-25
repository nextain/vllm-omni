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


### Phase 11: serving_chat.py duplicate index=0 fix (2026-04-14)

**Symptom**: `/v1/chat/completions` non-streaming response contains two choices both with `index=0`:
- `{index:0, message:{content:"text"}}` (text modality)
- `{index:0, message:{audio:{...}}}` (audio modality)

**Root cause**: `chat_completion_full_generator` loops over `OmniRequestOutput` items (one per
pipeline stage). Each calls `_create_text_choice` / `_create_audio_choice` which uses `output.index`
from `RequestOutput.outputs`. For n=1 requests, `output.index=0` for all modalities u2192 duplicates.

**Standard**: OpenAI Chat Completions spec requires unique sequential `index` per choice.

**Fix**: Post-loop sequential renumber u2014 one line after `choices.extend()`:
```python
for i, choice in enumerate(choices):
    choice.index = i
```

**Safety**: All demo clients (qwen3_omni, mimo_audio, minicpm_o) parse by field presence
(`choice.message.audio`, `choice.message.content`), not by `choice.index` value. No breakage.

**Bonus fix**: `final_res` is always `None` in this function scope u2014 output-logging token IDs
were silently always None. Added `final_res is not None` guard + explanatory comment.

**Recommended upstream**: Same bug affects all omni models. Upstream PR warranted.

**Review**: 2 consecutive clean passes achieved.

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

---

## Phase 11: naia-os /v1/realtime 연동 + multi-turn fix (2026-04-23)

### Background

naia-os `minicpm-o.ts`를 `/v1/omni` → `/v1/realtime` 마이그레이션 후 실제 테스트에서 발견된 서버 버그 수정.

### Findings

**Critical: `server_vad` 미구현**
- `session.update`의 `turn_detection: {type: "server_vad"}` 필드는 `_session_config`에 저장만 되고 VAD 로직 없음
- `_start_response()`는 `input_audio_buffer.commit` 또는 `response.create` 이벤트로만 트리거
- 검증: Python 테스트 (server_vad 설정 후 commit 없이 오디오 스트리밍 → 10초 타임아웃 무응답)
- 결론: naia-os 클라이언트는 반드시 explicit commit 사용해야 함

**Critical: multi-turn history 버그 (`fd273bdc`)**
- `_conversation_history`에 `{"role": "user", "content": "[audio input]"}` 저장
- Turn 2 요청 시 모델이 이 텍스트를 실제 사용자 발화로 오해 → 잘못된 응답 + TTS 노이즈 (RMS 0.0096 vs 정상 0.1290)
- Turn 2 응답 시간 0.78s (정상 5-6초) → 모델이 오디오 컨텍스트를 처리 못하고 단답 반환
- Fix: 히스토리 누적 비활성화. 각 턴 독립 처리. 올바른 멀티턴 구현은 ASR 기반 사용자 발화 텍스트 저장 필요 (future work)

### Lessons learned (추가)

15. **server_vad는 선언이지 구현이 아님** — `session.update`에서 `turn_detection: server_vad`를 수락해도 동작하지 않음. 항상 explicit commit 사용. qwen3_omni realtime client도 동일 패턴.
16. **history placeholder가 멀티모달 모델 혼란 유발** — 오디오 입력을 텍스트 플레이스홀더로 대체하면 모델이 이전 턴 컨텍스트를 잘못 해석. 오디오 히스토리 보존에는 ASR 선행 필요.
17. **omni_connection.py는 모델 API 측 배치 처리 전용** — `model.duplex.streaming_prefill / streaming_generate` 미사용, `create_chat_completion()` batch path 사용. 이는 모델 입력 경로의 batch 성격이지 WS audio output streaming 부재가 아님 (실제 WS는 매 SSE chunk를 `ResponseAudioDelta`로 즉시 forward — `omni_connection.py:191-195`). 진짜 incremental input 기반 full-duplex가 필요하면 model의 streaming API 직접 호출하는 별도 구현 필요. → 자세한 사항은 Phase 12 참조.

### naia-os side

브랜치 `issue-219-minicpm-realtime` (nextain/naia-os), 커밋:
- `0b822a16` feat(voice): migrate minicpm-o to OpenAI Realtime API
- `96c99375` refactor(voice): simplify audio delta path
- `7fc8765f` fix(voice): mic gate race + session.update gaps (#230)
- `6fd1b7c9` fix(voice): restore client VAD + explicit commit

naia-os #219 PR 대기 상태.

---

## Phase 12: archive `/v1/omni` 부활 시도와 revert (2026-04-25)

### Background

demo (`MiniCPM-o-Demo-forvLLM-omni`) 와 naia-os 모두 streaming audio output 필요.
이전 세션에서 "omni_connection.py는 batch-only이므로 audio output streaming 미지원"
이라 잘못 결론 → archive `/v1/omni` (`4b4f351e`로 폐기) 부활 결정.

### Attempt

- `ref/omni_duplex_v1/serving_omni_duplex.py`을 `vllm_omni/entrypoints/openai/`로 복사
- `api_server.py`에 `/v1/omni` WebSocket route 추가 (`/v1/realtime` 옆)
- 2× RTX 3090 + minicpmo_async_chunk로 시동 → Python WS smoke client 1-turn PASS
  (4.21s PCM16 input → 1.92MB output across 40 chunks + 21 transcript deltas)

### Cross-review에서 드러난 사실

1. **omni_connection.py가 chunk-by-chunk streaming** —
   line 191-195 매 SSE chunk를 즉시 `await self.send(ResponseAudioDelta(delta=content))`로
   forward. batch-only 아님.
2. **lesson 17의 "배치 처리 전용"은 model API 측 의미** —
   MiniCPM-o의 `model.duplex.streaming_prefill / streaming_generate` 미사용.
   WS audio output streaming 측이 아님.
3. **archive `/v1/omni`도 동일한 model batch path** —
   `create_chat_completion()` SSE를 base64 decode해서 binary frame으로 forward.
   즉 archive와 `/v1/realtime`은 model 측 foundation 동일, WS 측 둘 다 streaming.
4. **archive author 의도(WAV chunks)와 코드 동작(raw PCM forward) 처음부터 lapse** —
   `c3ad9985` docstring "Each binary output frame is a self-contained WAV"라 했으나
   audio output path에 RIFF wrap 코드 없음. SSE의 base64 PCM을 그대로 forward.
5. **OpenAI Realtime spec 호환성** —
   `/v1/realtime`이 base64 PCM16 24kHz mono in `ResponseAudioDelta` events로 spec 준수.
   `/v1/omni`은 fork-only proprietary protocol. naia-os production이 이미 `/v1/realtime` 사용.

### Decision

**Revert.** archive 부활 = 새로운 capability 없음, spec 호환 깨짐, redundant code path.
폐기 commit `4b4f351e`의 결정("Remove /v1/omni proprietary endpoint")이 옳음.

데모 측은 `/v1/realtime`로 client 작성 — naia-os `minicpm-o.ts` 패턴 + vllm-omni 자체
`examples/online_serving/minicpm_o/realtime_e2e_test.py` 참조.

### Lessons learned (추가)

18. **"batch-only" 같은 표현은 layer 명시 필수** — model API layer (입력의 batch 처리 성격)와
    transport layer (WS audio output streaming) 구분. 한 layer 사실이 다른 layer로 자동 전이
    안 됨. 컨텍스트 문서에 lesson 작성 시 "model API 측" / "WS 측" 명시.
19. **archive 부활 전 cross-review 강제** — archive author가 명시 목적으로 폐기한 코드는
    부활 전에 (a) 폐기 commit 본문, (b) ref/ 코드와 archive 직전 라이브 코드 diff,
    (c) live 측 후속 기능 진화 — 셋 모두 cross-check 필요. user가 plan에 의심 표명하면
    즉시 다중 agent cross-review.
20. **fork-only feature는 file docstring과 commit message 둘 다에 명시** —
    vllm-omni는 upstream contribution 목적의 fork. fork-only feature 도입 시 그 사실을
    file 자체에 markup해서 미래 maintainer/reviewer가 즉시 식별 가능해야.
