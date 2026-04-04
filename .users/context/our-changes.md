# 우리가 수정한 것 — 현재 상태

## 브랜치: feat/minicpm-o (upstream/main 기반)

## 목적

이 fork는 MiniCPM-o 4.5 omni 모델 지원을 vllm-omni에 추가하고 upstream PR로 기여하기 위한 프로젝트입니다.

**왜 vllm-omni fork인가**: Naia OS는 로컬 omni 모델 추론이 필요합니다. MiniCPM-o 4.5는
Naia 하드웨어 티어(2× RTX 3090 48GB, 32GB 단일 GPU)의 주요 타깃입니다. upstream 기여를 통해
Naia OS는 공식 프레임워크의 유지보수 혜택을 받으면서, 오픈소스 커뮤니티 전체에도 MiniCPM-o 지원을 제공합니다.

**AI-native 오픈소스 실험**: 이 기여의 전 과정 — upstream 패턴 분석, 구현, 헤드리스 서브에이전트를
이용한 적대적 코드 리뷰 — 이 모두 AI 에이전트(Claude)의 이슈 주도 개발 워크플로우로 수행되었습니다.
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
- Stop token 6561 미생성 → 오디오 ~20s 고정 → 별도 이슈 #207로 추적
- 한국어 TTS: CosyVoice2가 한국어 미지원 → 파인튜닝 필요
- 스트리밍 TTFP 목표: < 2s (baseline sync: 6.5s)

---

## 신뢰도 상태

| 항목 | 상태 | 비고 |
|------|:----:|------|
| 모델 코드 (thinker/talker/code2wav) | ✅ | 2× RTX 3090 E2E 검증 |
| minicpmo.yaml | ✅ | 10패스 코드 리뷰 클린 |
| minicpmo_48gb_2gpu.yaml | ✅ | 동일 파라미터 수정 적용 |
| minicpmo_async_chunk.yaml | ✅ | 동일 파라미터 수정 적용 |
| stage_input_processors/minicpm_o.py | ✅ | 10패스 적대적 리뷰, 2연속 클린 |
| registry.py | ✅ | 추가만 |
| 오디오 입력 (MiniCPMO processor) | ⚠️ | 2-GPU 텍스트→오디오 경로 미검증 |

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
