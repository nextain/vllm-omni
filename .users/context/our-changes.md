# 우리가 수정한 것 — 현재 상태

## 브랜치: main (vllm-project/vllm-omni fork)

**레포 구조**: 모든 작업은 `nextain/vllm-omni` main에 있습니다.
upstream PR 제출 시에는 main에서 별도 브랜치를 만들어 `vllm-project/vllm-omni` main으로 PR을 올립니다.

## 목적

이 fork는 MiniCPM-o 4.5 omni 모델 지원을 vllm-omni에 추가하고 upstream PR로 기여하기 위한 프로젝트입니다.

**왜 vllm-omni fork인가**: Naia OS는 로컬 omni 모델 추론이 필요합니다. MiniCPM-o 4.5는
Naia 하드웨어 티어(2× RTX 3090 48GB, 32GB 단일 GPU)의 주요 타깃입니다. upstream 기여를 통해
Naia OS는 공식 프레임워크의 유지보수 혜택을 받으면서, 오픈소스 커뮤니티 전체에도 MiniCPM-o 지원을 제공합니다.

**AI-native 오픈소스 실험**: 이 기여의 전 과정(upstream 패턴 분석, 구현, 헤드리스 서브에이전트를
이용한 적대적 코드 리뷰)이 모두 AI 에이전트(Claude)의 이슈 주도 개발 워크플로우로 수행되었습니다.
코드 결과물뿐 아니라 **AI-native 오픈소스 기여 방법론 자체를 검증**하는 실험입니다.

---

## 추가한 파일

경로 기준: `vllm_omni/model_executor/`

| 파일 | 목적 |
|------|------|
| `models/minicpm_o/__init__.py` | 모듈 export |
| `models/minicpm_o/configuration_minicpmo.py` | Config 클래스 (커스텀 — transformers에 없음) |
| `models/minicpm_o/minicpm_o.py` | 통합 엔트리 + talker_preprocess |
| `models/minicpm_o/minicpm_o_thinker.py` | Thinker: Idefics2 vision + Whisper audio + Qwen3 LLM |
| `models/minicpm_o/minicpm_o_talker.py` | Talker: MiniCPMTTS Llama AR 코덱 생성기 |
| `models/minicpm_o/minicpm_o_code2wav.py` | Code2Wav: CosyVoice2 flow + HiFi-GAN 보코더 |
| `stage_input_processors/minicpm_o.py` | thinker2talker, talker2code2wav (sync + async_chunk) |
| `stage_configs/minicpmo.yaml` | 단일 GPU 24GB 레퍼런스 config |
| `stage_configs/minicpmo_48gb_2gpu.yaml` | 2× RTX 3090 (48GB 합산) config |
| `stage_configs/minicpmo_async_chunk.yaml` | async_chunk 스트리밍 config (TTFP 최적화) |

## 수정한 파일

| 파일 | 변경 |
|------|------|
| `models/registry.py` | MiniCPM-o 6개 엔트리 추가 |
| `pyproject.toml` | `sentence-transformers`, `scikit-learn` 선택적 의존성 추가 |

## 오프라인 추론 예제 (examples/offline_inference/minicpm_o/)

오프라인 추론 예제 디렉토리 신규 추가:

| 파일 | 목적 |
|------|------|
| `end2end.py` | 동기 오프라인 추론 (`Omni` 클래스); text/image/audio/image+audio 쿼리 타입 |
| `end2end_async_chunk.py` | 비동기 오프라인 추론 (`AsyncOmni`); 스테이지 수준 동시성 |
| `run_single_prompt.sh` | 단일 프롬프트 편의 스크립트 |
| `run_multiple_prompts.sh` | 멀티 프롬프트 배치 (`--py-generator` 포함) |
| `run_single_prompt_async_chunk.sh` | async_chunk 단일 프롬프트 (TTFP ~0.07s) |
| `run_multiple_prompts_async_chunk.sh` | async_chunk 멀티 프롬프트 (`--max-in-flight` 제어) |
| `text_prompts_10.txt` | AI 관련 샘플 프롬프트 10개 |
| `README.md` | qwen3_omni offline 포맷에 맞춘 사용 가이드 |

**SamplingParams 정렬** (8패스 리뷰 + 2026-04-14 버그수정 후):
- Thinker: `temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048, detokenize=True, repetition_penalty=1.05`
- Talker: `temperature=0.8, top_p=0.85, top_k=25, max_tokens=1000, min_tokens=50, detokenize=False, repetition_penalty=1.05, stop_token_ids=[6561]` — `modeling_minicpmo.py`의 `TTSSamplingParams` 기준
- Code2Wav: `temperature=0.0, top_p=1.0, top_k=-1, max_tokens=65536, detokenize=True, repetition_penalty=1.1`

---

## 벤치마크 스크립트 (examples/online_serving/minicpm_o/)

| 파일 | 수정 내용 (멀티패스 적대적 리뷰) |
|------|--------------------------------|
| `metrics/cjk_metrics.py` | CER 공식 수정 (SequenceMatcher→edit_distance/len(ref)); `_character_overlap` set→Counter multiset; CER 하한 `max(0.0,...)`; `_sentence_model` 모듈 레벨 캐시 |
| `metrics/conversation_quality.py` | `knowledge_retention` 루프 수정; `_character_overlap` Counter multiset; `tts_speech_ratio` `overall_quality` 가중치 제거; L25 docstring 수정 |
| `conversation_benchmark.py` | 출력 레이블: `"Avg STT CER (lower=better)"` |
| `language_test.py` | `stt_cer`/`stt_word_accuracy` 키 분리; `evaluate_conversation`에 `stt_text` + `response_time` 전달; opener 미리보기 조건부 말줄임표 |
| `voicebench_runner.py` | `SampleResult` 필드명 수정; `list_count` 빈 응답 처리; API 예외 처리; `CATEGORIES` `deepcopy`로 전역 상태 보호; `list_count` 쉼표 필터 빈 토큰 제거 |

---

## 아키텍처: 3단계 파이프라인

```
입력 (텍스트 + 이미지/오디오)
  → Stage 0: Thinker  (GPU 0, LLM AR) → hidden_states
  → Stage 1: Talker   (GPU 1, TTS AR) → 코덱 토큰 ID
  → Stage 2: Code2Wav (GPU 1, 보코더) → PCM 오디오
```

**Qwen3-Omni / MIMO-Audio와의 주요 차이:**

| | MiniCPM-o | Qwen3-Omni | MIMO-Audio |
|--|-----------|-----------|-----------|
| num_vq (RVQ 레이어 수) | **1** | 8 | N/A |
| Hidden state 키 | `thinker_hidden_states` | `thinker_prefill_embeddings` + `thinker_decode_embeddings` | fused |
| 스테이지 전송 | SharedMemoryConnector (0→1), 직접 주입 (1→2) | SharedMemoryConnector (전체) | SharedMemoryConnector |
| 코덱 토큰 차원 | 1D flat | 8×N 행렬 | N/A |
| TTS 경계 감지 | `_find_tts_bound()` 필요 | 없음 (전체 시퀀스) | 없음 |
| async_chunk 지원 | ✅ 구현됨 | ✅ upstream | N/A |

---

## stage_input_processors/minicpm_o.py — 핵심 함수

### 동기(sync) 경로
- `thinker2talker_sync`: `thinker_hidden_states`를 `request_payload`에 누적; `_find_tts_bound()`로 TTS 세그먼트 슬라이싱; hidden + 토큰 ID를 Talker 입력으로 반환.
- `talker2code2wav`: Talker `output.token_ids`에서 stop token 6561을 **값 기반으로 필터**(`[:-1]` 아님); 코덱 프레임 리스트 반환.

### 비동기(async_chunk) 경로
- `thinker2talker_async_chunk`: 스텝별 hidden states를 `torch.cat`으로 `transfer_manager.request_payload`에 누적; `is_finished=True`일 때만 전송.
- `talker2code2wav_async_chunk`: cursor 방식 증분 코덱 읽기(`_talker_token_cursor`, `_talker_emitted_len`); `codec_chunk_frames=25` 단위로 청크 방출; `pending=0, finished=True`일 때 finished sentinel dict 반환 (`None` 아님 — cleanup_sender 트리거 필요).

### 헬퍼
- `_ensure_list(x)`: `ConstantList` / 텐서형을 plain Python list로 변환. `qwen3_omni._ensure_list` 기반; 차이: 항상 true list로 변환 (비리스트 타입을 그대로 반환하지 않음).
- `_find_tts_bound(thinker_token_ids)`: Thinker 출력 토큰에서 `<|tts_bos|>`..`<|tts_eos|>` 경계 탐색 → TTS 세그먼트 추출.

---

## 벤치마크 결과 (2× RTX 3090, async_chunk 모드)

| 언어 | STT 정확도 | 지연 | 등급 |
|------|:----------|:-----|:-----|
| 영어 | 92% (단어) | 평균 2.3s | A |
| 중국어 | CER 76.1% / 의미 유사도 95% | 평균 1.5s | B+ |
| 한국어 | 텍스트 OK / TTS 실패 | — | F (TTS) |

**VoiceBench (영어, 규칙 기반 채점)**:
- 지식: 100% · 지시: 95% · 견고성: 100% · 안전: 100%
- 종합: 98.0% 평균, 98.7% pass rate

**알려진 한계:**
- 한국어 TTS: CosyVoice2가 한국어 미지원 → 파인튜닝 필요
- 스트리밍 TTFP 실측: ~0.07s (Thinker 완료 후 첫 오디오 패킷; baseline sync: 6.5s)

**버그 수정 (2026-04-14):**
- ~~Stop token 6561 미생성~~ → 근본 원인: Talker 샘플링 파라미터 불일치 (`top_p=1.0` 누락, `top_k=50`, `temp=0.9`). `TTSSamplingParams` 기준값으로 수정
- ~~left_context_size 무시~~ → 50토큰 전체 처리 후 2배 오디오 출력. 비례 trimming으로 수정 (`math.ceil(wav_len * new_token_count / total)`)

**OmniSpeaker 클라이언트 버그 (수정 완료):**
- `getattr(chunk, "modality", "text")` 기본값으로 오디오 청크가 항상 텍스트 경로로 처리됨
- 수정: 기본값을 `None`으로 변경, `elif modality == "text"` 패턴 (upstream gradio_demo.py 정렬)
- 커밋: `40576264`

---

## WebSocket 엔드포인트 현황 (2026-04-07 기준)

| 엔드포인트 | 상태 | 비고 |
|-----------|:----:|------|
| POST /v1/chat/completions (non-stream) | ✅ E2E 작동 | 텍스트+오디오 함께 반환, ~28s |
| POST /v1/chat/completions (stream SSE) | ✅ 확인 | base64 WAV 청크, 0.9s TTFP |
| WebSocket /v1/realtime | ✅ E2E PASS | OpenAI Realtime API — 오디오 입력 → 트랜스크립트+오디오 출력, TTFP 0.61s |
| ~~WebSocket /v1/omni~~ | 🗑️ 제거됨 (#4) | 독자 규격 → ref/omni_duplex_v1/ 에 보관 |

## 완료된 작업 (2026-04-08)

| 이슈 | 내용 | 상태 |
|------|------|:----:|
| nextain/vllm-omni#4 | /v1/omni 제거, ref/omni_duplex_v1/ 이동 | ✅ |
| nextain/vllm-omni#5 | OpenAI Realtime API protocol 이벤트 | ✅ E2E PASS |
| nextain/vllm-omni#6 | OmniRealtime 오디오 출력 파이프라인 | ✅ E2E PASS |

## 대기 중 (naia-os)

| 이슈 | 내용 | 상태 |
|------|------|:----:|
| nextain/naia-os#219 | minicpm-o.ts /v1/realtime 전환 | 🔜 |
| nextain/naia-os#220 | Silero ONNX VAD 교체 | 🔜 |

## 방향 원칙 (2026-04-07 재확인)

- `/v1/realtime` OpenAI Realtime API 호환 경로만 사용
- 독자 엔드포인트/규격 절대 금지
- upstream vllm realtime 모듈은 서브클래싱으로만 확장 (직접 수정 불가)
- 클라이언트 책임(VAD 등)은 Naia에, 서버 책임은 vllm-omni에

---

## 신뢰도 상태

| 항목 | 상태 | 비고 |
|------|:----:|------|
| 모델 코드 (thinker/talker/code2wav) | ✅ | 2× RTX 3090 E2E 검증 |
| minicpmo.yaml | ✅ | 10패스 적대적 리뷰, 2연속 클린 |
| minicpmo_48gb_2gpu.yaml | ✅ | 동일 파라미터 수정 적용, 10패스 리뷰 |
| minicpmo_async_chunk.yaml | ✅ | 동일 파라미터 수정 적용, 10패스 리뷰 |
| stage_input_processors/minicpm_o.py | ✅ | 10패스 적대적 리뷰, 2연속 클린 |
| registry.py | ✅ | 추가만 |
| 오디오 입력 (MiniCPMO processor) | ⚠️ | 2-GPU 텍스트→오디오 경로 미검증 |
| metrics/cjk_metrics.py | ✅ | 멀티패스 리뷰, 메트릭 공식 수정 완료 |
| metrics/conversation_quality.py | ✅ | 멀티패스 리뷰, knowledge_retention + 가중치 수정 완료 |
| conversation_benchmark.py | ✅ | 레이블 수정 |
| language_test.py | ✅ | stt_cer/word_accuracy 분리, evaluate_conversation 수정 |
| voicebench_runner.py | ✅ | SampleResult 필드명, 예외 처리, CATEGORIES deepcopy |
| realtime/serving.py | ✅ | upstream pass-through 서브클래스 |
| realtime/protocol.py | ✅ | omni 이벤트, OpenAIBaseModel |
| realtime/connection.py | ✅ | upstream re-export |
| realtime/omni_connection.py | ✅ | 9-pass 적대적 리뷰, 2연속 클린 |
| offline_inference/minicpm_o/end2end.py | ✅ | 8패스 적대적 리뷰; SamplingParams 정렬 완료 |
| offline_inference/minicpm_o/end2end_async_chunk.py | ✅ | 8패스 리뷰; init_timeout, use_image_audio 추가 |
| offline_inference/minicpm_o/run_*.sh | ✅ | 파일 존재 검사, PROMPTS_FILE 변수 재사용 |
| entrypoints/openai/serving_chat.py | ✅ | 2026-04-14: 중복 index=0 수정(sequential renumber) + final_res None guard. upstream 코드 — 최소 변경, cross-review + E2E 검증 |

---

## 주요 교훈

1. **upstream 패턴 먼저** — qwen3_omni.py, qwen2_5_omni.py, mimo_audio.py 먼저 읽고 구현
2. **ConstantList 래핑** — `list(request.output_token_ids)` 안전하지 않음; `_ensure_list()` 사용
3. **스텝별 hidden states** — `pooling_output["thinker_hidden_states"]` = 현재 스텝 슬라이스만; 전체 디코딩 스텝 누적 필요
4. **cleanup_sender 계약** — payload가 truthy일 때만 호출됨; 메모리 누수 방지를 위해 None 대신 finished sentinel dict 반환
5. **`[:-1]` 트림은 잘못됨** — max_tokens 도달 시 EOS 없으면 실제 프레임 제거됨; 값 기반 필터 사용
6. **NCCL_P2P_DISABLE=1** — NVLink 없는 RTX 3090 필수
7. **Code2Wav GPU 메모리** — CosyVoice2 flow 모델 ~15 GB; 할당 계획 필수
8. **max_inflight: 1** — 동시 스테이지 메모리 OOM 방지 (vllm-omni#1387)


## 추가 교훈 (2026-04-14)

9. **serving_chat.py 중복 index=0** — omni 모델 텍스트+오디오 동시 출력 시 OpenAI 스펙 위반. post-loop enumerate로 수정.
10. **Talker sampling params 맞춰야 함** — top_p, top_k, temperature 멀슰 시 stop token 6561 생성 안 됨 → 323초 gibberish.
11. **final_res는 항상 None** — serving_chat.py 스코프에서 할당 안 됨. 로그 경로 None guard 필수.
12. **two-choice design은 OpenAI spec 위반** — n>1 시 choices 개수 != n. sequential renumber는 workaround; 근본은 upstream 설계 변경 필요.
13. **CosyVoice2 codec 프레임 레이트는 25fps (40ms/frame, 24kHz 출력)** — 25 new frames = 1초 오디오(left-context 트리밍 후). 트리밍 없으면 2초/chunk. 40 chunks × 1s = 40s는 정상 동작 (max_tokens=1000 도달).
14. **left_context_size async_chunk 데이터 흐름 확인** — Request.additional_information → OmniCachedRequestData → _update_additional_information → model_intermediate_buffer → runtime_additional_information → forward(). 전체 경로 검증 완료.
