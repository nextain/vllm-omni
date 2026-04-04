# 기여 여정 — MiniCPM-o upstream PR

이 파일은 vllm-omni에 MiniCPM-o 4.5를 기여하는 실제 과정을 기록합니다.
실패, 방향 전환, 검증된 결정을 모두 담아 — 다음 기여자(사람이든 AI든)가
현재 코드의 *왜*를 이해할 수 있도록 합니다.

---

## 우리가 하고 있는 것

**목표**: MiniCPM-o 4.5 omni 모델 지원을 vllm-omni upstream PR로 기여

**왜 vllm-omni fork인가**:
Naia OS는 소비자 하드웨어(2× RTX 3090, NVLink 없음)에서 로컬 omni 모델 추론이 필요합니다.
vllm-omni는 멀티스테이지 omni 파이프라인을 지원하는 유일한 프로덕션급 프레임워크입니다.
upstream 기여를 통해 Naia OS는 공식 유지보수 혜택을 누리면서,
오픈소스 커뮤니티 전체에 MiniCPM-o 지원을 제공합니다.

**AI-native 오픈소스 실험**:
이 기여의 전 과정(upstream 패턴 분석, 구현, 헤드리스 서브에이전트를 이용한 적대적 코드 리뷰)이
모두 AI 에이전트(Claude)의 이슈 주도 개발 워크플로우로 수행되었습니다.
코드 결과물뿐 아니라 **AI-native 오픈소스 기여 방법론 자체를 검증**하는 실험입니다:

- AI 에이전트가 오픈소스 프레임워크의 패턴을 이해하고 일관된 코드를 작성할 수 있는가?
- 헤드리스 서브에이전트 방식의 적대적 리뷰가 사람 코드 리뷰를 대체할 수 있는가?
- 컨텍스트 파일 기반 세션 인수인계로 장기 기여 작업이 가능한가?

---

## 타임라인

### Phase 1: 초기 포팅 (내부, Naia fork)
- HuggingFace transformers 레퍼런스에서 MiniCPM-o 모델 코드 포팅
- **실패**: 24GB 단일 GPU에서 동작 확인 → "완료" → 2-GPU 환경에서 완전히 깨짐
- **원인**: upstream stage_configs 문서와 레퍼런스 모델 YAML을 먼저 읽지 않음
- **교훈**: `process: true` 필수; GPU 메모리 예산 계산은 직관적이지 않음

### Phase 2: 2-GPU 안정화
- `NCCL_P2P_DISABLE=1` 추가 (NVLink 없는 RTX 3090)
- `gpu_memory_utilization` 분배 수정: Thinker 0.88 + Talker 0.1 + Code2Wav 0.02
- **실패**: "디바이스 격리 버그" 진단 → upstream 버그로 보고
- **방향 전환**: upstream 버그가 아님. 우리 stage config 오류(잘못된 `devices` 값)
- **교훈**: 프레임워크를 탓하기 전에 config 먼저 검증

### Phase 3: E2E 검증
- Code2Wav 출력에 `_trim_silence()` 추가
- 대화 벤치마크 프레임워크 구축 (OmniSpeaker, Whisper STT, CJK 메트릭)
- **실패**: CJK 메트릭에서 중국어 0% → 단어 경계 불일치에 의한 false negative
- **수정**: CER + 의미 유사도(sentence-transformers)로 전환
- **결과**: EN 92% 단어 정확도, ZH CER 76.1% / 의미 유사도 95%

### Phase 4: VoiceBench
- voicebench_runner.py 구현 (규칙 기반 채점)
- **결과**: 평균 점수 98.0%, pass rate 98.7% (지식/지시/견고성/안전)

### Phase 5: async_chunk 스트리밍 구현
- **목표**: TTFP 6.5s → 2s 미만 감소
- qwen3_omni.py async_chunk 함수 직접 파일 읽기 참조 (WebSearch 아님)
- **실패**: 초기 hidden state 누적 버그 — `pooling_output["thinker_hidden_states"]`가 현재 디코딩 스텝(토큰 1개)만 반환
  - 원인: `gpu_ar_model_runner.py` 591-619라인 먼저 추적하지 않음
  - 수정: `torch.cat`으로 `transfer_manager.request_payload`에 스텝별 누적
- **실패**: `list(request.output_token_ids)` — ConstantList 래퍼 문제
  - 수정: qwen3_omni 패턴 따라 `_ensure_list()` 헬퍼 추가
- **실패**: `pending=0, finished=True`일 때 `cleanup_sender` 미호출 → 메모리 누수
  - 원인: `chunk_transfer_adapter.py` 프레임워크 계약 먼저 읽지 않음
  - 수정: `None` 대신 finished sentinel dict `{code_predictor_codes: [], finished: True}` 반환
- **실패**: sync 경로의 `[:-1]` 무조건 트림이 max_tokens 도달 시 실제 코덱 프레임 제거
  - 수정: 값 기반 필터 `t != 6561`
- **리뷰**: 10패스 적대적 헤드리스 리뷰, 2연속 클린 패스 달성

### Phase 6: Stop token 조사 (진행 중)
- **발견**: Talker가 stop token 6561을 거의 생성하지 않음 → 오디오 항상 ~20s
- **원인 분석**: 모델 conditioning 문제, 코드 버그 아님
  - Talker가 현재 훈련 분포에서 EOS 생성을 학습하지 못함
- **결정**: 별도 이슈로 분리, upstream PR 차단 안 함; 알려진 한계로 문서화
- **다음**: 내부 이슈로 별도 추적 중; upstream 협력 예정

### Phase 7: 벤치마크 스크립트 멀티패스 리뷰
- 벤치마크 스크립트 전체 11패스 적대적 헤드리스 리뷰
- **주요 수정**: CER 공식, Counter multiset, SentenceTransformer 캐시,
  knowledge_retention 루프, tts_speech_ratio 가중치 제거, SampleResult 필드명,
  CATEGORIES deepcopy, OmniSpeaker 오디오 청크 감지 버그 (`modality` 기본값 `None`)
- 2연속 클린 패스 달성

### Phase 8: 레포 구조 정리
- **문제**: 작업이 `main` (54커밋, 초기 코드)과 `feat/minicpm-o` (252커밋, 완성본)로 분리됨.
  fork main이 upstream과 동일 → `github.com/nextain/vllm-omni`에서 아무것도 안 보임
- **수정**: `feat/minicpm-o` → fork `main` force push; `feat/minicpm-o` 브랜치 삭제
- **결과**: 모든 작업이 `nextain/vllm-omni` main에 있음. 브랜치 없이 main에서 작업.

---

## 핵심 upstream 발견사항

### vllm-omni 내부에 대해 발견한 것들
- `ChunkTransferAdapter.code_prompt_token_ids`는 `defaultdict(list)` — 프레임워크가 소유
- `pooling_output["thinker_hidden_states"]` = 스텝별 슬라이스, 전체 시퀀스 아님
- `cleanup_sender`는 `_send_single_request`가 truthy payload를 받을 때만 호출됨
- `ConstantList`는 vLLM 내부 타입으로 `list()`로 올바르게 이터레이션되지 않음

### 유지한 upstream 패턴 이탈
- `_ensure_list()` — 항상 true Python list로 변환 (qwen3_omni은 비리스트를 그대로 반환)
  — MiniCPM-o의 num_vq=1 때문에 모든 코덱 토큰이 plain list임; 텐서가 아님
- `_find_tts_bound()` 필요 (Qwen3는 없음) — MiniCPM-o는 Thinker 출력 시퀀스에 TTS 토큰을 인라인으로 삽입

### 제거한 이탈
- `[:-1]` 트림 (초기 레퍼런스에서 복사) → 잘못됨; 값 기반 필터로 교체
- `finished=True`일 때 `None` 반환 → 메모리 누수; sentinel dict로 교체

---

## Upstream PR 상태

| 항목 | 상태 |
|------|:----:|
| 모델 코드 (3단계) | 준비 완료 |
| stage_configs (3개 yaml) | 준비 완료 |
| stage_input_processors | 준비 완료 |
| 벤치마크 결과 | 문서화 완료 |
| E2E 검증 하드웨어 | 2× RTX 3090 |
| async_chunk 스트리밍 | 구현 완료, E2E 대기 |
| 한국어 TTS | ⚠️ 알려진 실패, 문서화 |
| Stop token 6561 | ⚠️ 알려진 이슈 → 별도 추적, upstream 협력 예정 |

**Upstream 이슈 참조**: vllm-omni#1182 (Allyyi의 병행 작업, 동일 모델)

---

## 문서 구조

`examples/online_serving/minicpm_o/` 디렉토리는 영/한 듀얼 언어 구조를 사용합니다:

| 파일 | 언어 | 목적 |
|------|------|------|
| `README.md` | **영어 (주 언어)** | 빠른 시작, 스크립트, stage config — KO 링크 포함 |
| `README.ko.md` | 한국어 | 동일 내용 한국어 — EN 링크 포함 |
| `BENCHMARK.md` | **영어** | 전체 벤치마크 결과 — KO 링크 포함 |
| `BENCHMARK.ko.md` | 한국어 | 동일 벤치마크 한국어 — EN 링크 포함 |

헤더 규칙: `**Language / 언어**: [English](README.md) | [한국어](README.ko.md)`

기존 파일들(`final_benchmark_report.md`, `benchmark_summary.md` 등)은 내부 히스토리로 보존되나
`BENCHMARK.md` / `BENCHMARK.ko.md`로 대체됨.

---

## 이어서 작업할 때 읽어야 할 파일

```
vllm_omni/model_executor/stage_input_processors/minicpm_o.py  # 핵심 로직
vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml
vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml
vllm_omni/model_executor/stage_configs/minicpmo.yaml
examples/online_serving/minicpm_o/conversation_benchmark.py
examples/online_serving/minicpm_o/voicebench_runner.py
```
