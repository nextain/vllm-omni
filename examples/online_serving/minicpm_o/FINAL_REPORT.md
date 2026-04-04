# MiniCPM-o 4.5 vllm-omni 통합 최종 보고서

**작성일**: 2026-04-03  
**이슈**: nextain/naia-os#181, #72, #191, #205  
**브랜치**: feat/minicpm-o  

---

## 개요

MiniCPM-o 4.5 (Thinker→Talker→Code2Wav 3단계 파이프라인)를 vllm-omni 프레임워크에 통합하는 작업의 최종 결과물입니다. 텍스트 및 오디오 생성 E2E 검증부터 async_chunk 스트리밍 구현까지 완료했습니다.

---

## 완료된 작업

### Phase 1: 기반 구조 (이전 세션)

**구현 파일:**
- `vllm_omni/model_executor/models/minicpm_o/` — Thinker, Talker, Code2Wav 모델
- `vllm_omni/model_executor/stage_configs/minicpmo.yaml` — 동기 서버 설정 (2-GPU)
- `vllm_omni/model_executor/stage_input_processors/minicpm_o.py` — `thinker2talker()`, `talker2code2wav()` (동기)

**검증:**
- 2×RTX 3090 환경에서 E2E 텍스트+오디오 생성 PASS
- 영어/중국어 대화 10턴 STT 정확도 98% 이상

### Phase 2: 벤치마크 인프라

**신규 파일:**
- `examples/online_serving/minicpm_o/e2e_conversation_test.py` — 2 AI 화자 대화 E2E 프레임워크
- `examples/online_serving/minicpm_o/conversation_benchmark.py` — 6개 시나리오 대화 벤치마크
- `examples/online_serving/minicpm_o/metrics/cjk_metrics.py` — CJK 언어 메트릭 (CER, semantic similarity)
- `examples/online_serving/minicpm_o/metrics/conversation_quality.py` — 대화 품질 평가기

**결과 (sync 모드 기준선):**
| 메트릭 | 값 |
|--------|-----|
| STT CER (영어) | ~94–97% 정확도 |
| TTFP | ~6,459ms |
| 평균 오디오 길이 | ~20s (stop token 미생성) |

### Phase 3: async_chunk 스트리밍 (#72)

**핵심 구현:**

`vllm_omni/model_executor/stage_input_processors/minicpm_o.py`:
- `thinker2talker_async_chunk()` — Thinker 완료 시 tts_bound 슬라이싱 후 Talker 전달
- `talker2code2wav_async_chunk()` — `request.output_token_ids` cursor 방식 codec 토큰 스트리밍

`vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml`:
- `async_chunk: true`, SharedMemoryConnector, codec_chunk_frames: 25

**주요 버그 수정:**
1. `pooling_output.get("token_ids")` → `request.output_token_ids` + cursor 추적  
   (pooling_output에는 "hidden" 키만 존재 — codec 토큰은 AR 샘플링 결과)
2. `% chunk_size` 모듈로 → `< chunk_size` 임계값  
   (토큰이 경계를 건너뛸 때 영구 블로킹 방지)
3. `chunk_length = current_length` → `_talker_emitted_len` 트래커  
   (finished=True 시 전체 버퍼 재전송 → 중복 오디오 생성 방지)
4. 메모리 누수: per-request 딕셔너리 cleanup 추가

**벤치마크 결과 (async_chunk):**
| 메트릭 | sync | async_chunk | 개선 |
|--------|------|-------------|------|
| **TTFP** | 6,459ms | **56ms** | **99% 감소** |
| 응답시간 | 2.3s | 2.5s | 동등 |
| Partner STT CER | 97% | 97% | 유지 |

---

## 알려진 한계

### 1. Stop Token #207 (모델 weights 수준)
Talker가 stop token 6561을 안정적으로 생성하지 않아 max_tokens=512까지 생성.  
→ OmniSpeaker 오디오 길이가 고정(~20s), STT semantic similarity 낮음  
→ 별도 이슈 #207로 등록, upstream 협력 예정

### 2. thinker2talker_async_chunk는 여전히 Thinker 완료 후 전송
Thinker tts_bound 필터링이 전체 토큰 시퀀스 확정 후에만 가능 → Talker→Code2Wav만 스트리밍됨.  
실질적 TTFP 56ms는 Code2Wav 첫 청크 기준.

### 3. 한국어 TTS
한국어 입력 시 Talker codec 토큰 생성 불안정 → 오디오 무음  
(영어/중국어는 정상 작동)

---

## 파일 목록 (수정/신규)

### 핵심 구현
| 파일 | 변경 |
|------|------|
| `vllm_omni/model_executor/stage_input_processors/minicpm_o.py` | thinker2talker, talker2code2wav (동기+비동기) |
| `vllm_omni/model_executor/stage_configs/minicpmo.yaml` | 동기 서버 설정 |
| `vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml` | **신규** — 스트리밍 서버 설정 |

### 벤치마크 도구
| 파일 | 설명 |
|------|------|
| `examples/online_serving/minicpm_o/e2e_conversation_test.py` | E2E 대화 테스트 프레임워크 |
| `examples/online_serving/minicpm_o/conversation_benchmark.py` | 6개 시나리오 벤치마크 |
| `examples/online_serving/minicpm_o/metrics/cjk_metrics.py` | CJK 언어 메트릭 |
| `examples/online_serving/minicpm_o/metrics/conversation_quality.py` | 대화 품질 평가 |

---

## 서버 시작

```bash
# 동기 모드 (2-GPU RTX 3090)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --port 8000

# 스트리밍 모드 (async_chunk, TTFP 56ms)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --port 8000
```

---

## 벤치마크 실행

```bash
cd examples/online_serving/minicpm_o/

# 동기 모드 (edge-tts TTS)
python conversation_benchmark.py --scenarios greeting qa

# 스트리밍 모드 (MiniCPM-o 네이티브 오디오)
python conversation_benchmark.py --omni --scenarios greeting qa
```

---

## 커밋 목록

| 커밋 | 내용 |
|------|------|
| `e3ce7b05` | feat(voice): async_chunk streaming for MiniCPM-o (#72) |
| `f0d9f448` | fix(voice): talker2code2wav_async_chunk codec token fix |
| `6bbfd2a6` | fix: review-pass cleanups (TTFP falsy, deps, docstring) |
| `0a15d4ae` | chore: close #181 — MiniCPM-o E2E + VoiceBench complete |
| `5f3a5812` | feat: VoiceBench runner + final benchmark report |
| `5608138b` | fix: CJK metrics for speech conversation benchmarking |

---

## 관련 이슈

| 이슈 | 제목 | 상태 |
|------|------|------|
| nextain/naia-os#181 | MiniCPM-o 4.5 vllm-omni upstream PR | CLOSED ✓ |
| nextain/naia-os#72 | feat(voice): vLLM omni model support | CLOSED ✓ |
| nextain/naia-os#191 | Audio input 지원 | CLOSED ✓ |
| nextain/naia-os#205 | E2E 대화 테스트 프레임워크 | CLOSED ✓ |
| nextain/naia-os#207 | Talker stop token 6561 미생성 | OPEN — 별도 트랙 |
| vllm-omni#1182 | MiniCPM-o upstream contribution | Allyyi 담당 |
