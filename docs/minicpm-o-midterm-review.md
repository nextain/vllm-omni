# MiniCPM-o 4.5 중간 점검 보고서

> **작성일**: 2026-03-28
> **상태**: E2E 재현 실패, 근본 재설계 필요

---

## 1. 현재 상황

- **이전 E2E** (3/25, 커밋 `edaf4d79`): 텍스트 + 오디오 응답 성공
- **현재**: 서버 시작은 되지만 요청 시 연쇄 에러 발생, E2E 재현 불가
- **소비**: RunPod ~$30, 이틀 삽질

## 2. 기존 문서의 문제점

### 2.1 구현 보고서 (`docs/minicpm-o-implementation-report.md`)

| 문서에 적힌 것 | 실제 |
|---------------|------|
| "코드 리뷰 완료, GPU E2E 재검증 대기" | E2E 재현 불가 |
| "vllm-omni 컨벤션 준수" | **핵심 컨벤션(stage_configs YAML) 미준수** — pipeline.yaml 사용 |
| "매직넘버 0개" | HF config에 없는 `tts_bos_token_id` 필드에 의존 → AttributeError |
| "AI slop 0건" | pipeline.yaml은 미검증 신기술 패턴 채택 — 다른 모델은 전부 stage_configs 사용 |
| 테스트 현황 "추가 필요" 표시 | fresh install 재현성 테스트 자체가 없음 |

### 2.2 PR Draft (`.agents/docs/pr-draft-minicpm-o.md`)

| 문제 | 상세 |
|------|------|
| `pipeline.yaml` 사용 명시 | 다른 모델(qwen3_omni, qwen2_5_omni 등)은 전부 `stage_configs/*.yaml` 사용. 유일하게 다른 방식 채택 |
| E2E 체크 표시 | 이전 pod 환경에서만 성공, 재현 방법 미기록 |
| `--trust-remote-code` 미언급 | 서버 시작에 필수인데 문서 어디에도 없음 |
| `--skip-mm-profiling` 미언급 | 동일 |
| `minicpmo-utils` 의존성 미언급 | requirements에도 없었음 |

### 2.3 수정 내용 설명 (`.agents/docs/changes-explained.ko.md`)

- 아키텍처 수정 12건 자체는 정확
- 하지만 **인프라 레벨 문제를 다루지 않음**: 서버 시작 방법, 필수 플래그, 의존성, 재현 가능성

## 3. 무엇이 틀렸는가

### 3.1 검증되지 않은 패턴 선택

```
qwen3_omni:     stage_configs/qwen3_omni_moe.yaml  ← 검증됨, 다른 모델도 사용
qwen2_5_omni:   stage_configs/qwen2_5_omni.yaml     ← 검증됨
fish_speech:    stage_configs/fish_speech_s2_pro.yaml ← 검증됨
minicpm_o:      pipeline.yaml + StageConfigFactory   ← 미검증, 유일하게 다름
```

"더 새로운 방식"이라는 이유만으로 선택. 실제 작동하는 모델이 어떤 패턴을 쓰는지 확인하지 않음.

### 3.2 upstream rebase 호환성 미고려

- `a90a7690` 커밋에서 vllm v0.18.0 rebase 발생
- `OpenAIServingResponses`, `OmniOpenAIServingChat` 등 새 클래스가 `model_config`를 요구
- omni 멀티스테이지 모드에서 `model_config=None` → 연쇄 크래시
- try/except 땜질로 대응했지만 근본 해결 아님

### 3.3 실제 작업 미기록

- `pip install minicpmo-utils` 수동 설치 → requirements 미반영
- `--trust-remote-code`, `--skip-mm-profiling` 필수 → 스크립트/문서 미반영
- pip freeze 미수행 → 환경 재현 불가
- workspace_setup.sh에 vllm 버전 잘못 기재 (0.17.1 vs 0.17.0)

### 3.4 리뷰의 한계

"6-pass 적대적 코드 리뷰"는 코드 로직만 봤고:
- **운영 완성도**: fresh install → 서버 시작 → E2E가 한번에 되는가? → 미검증
- **패턴 준수**: 다른 모델과 같은 방식인가? → 미확인
- **재현 가능성**: 다른 사람/환경에서 동일하게 돌아가는가? → 미확인

## 4. 어떻게 고쳐야 하는가

### 4.1 stage_configs/minicpmo.yaml 생성 (최우선)

qwen3_omni 패턴 그대로:

```yaml
# stage_configs/minicpmo.yaml
stage_args:
  - stage_id: 0
    stage_type: llm
    engine_args:
      model_stage: thinker
      model_arch: MiniCPMOForConditionalGeneration
      worker_type: ar
      trust_remote_code: true
      # ... qwen3_omni_moe.yaml 형식 그대로
  - stage_id: 1
    # talker ...
  - stage_id: 2
    # code2wav ...
```

이렇게 하면:
- `--trust-remote-code` CLI 의존 제거 (YAML에 명시)
- `StageConfigFactory` 우회 (legacy 경로로 직접 로드)
- `pipeline.yaml` 삭제 가능
- upstream rebase 영향 최소화

### 4.2 HF config 의존 제거

`tts_bos_token_id` 등 우리가 추가한 config 필드에 의존하지 말고, stage_configs YAML에 직접 명시하거나 코드에서 getattr fallback 사용.

### 4.3 try/except 땜질 제거

`api_server.py`의 `OpenAIServingResponses`, `OmniOpenAIServingChat` try/except는 stage_configs 방식으로 전환 후 불필요해지면 제거.

### 4.4 fresh install 검증 자동화

```bash
# 이 스크립트가 한번에 돌아가야 함
git clone → pip install → vllm serve → curl test → 성공
```

### 4.5 문서 전면 재작성

- 보고서: pipeline.yaml 방식 → stage_configs 방식으로 변경 반영
- PR draft: 실제 작동하는 서버 시작 명령 + 필수 의존성 포함
- workspace_setup.sh / workspace_start.sh: 검증된 명령만

---

## 5. 왜 적대적 반복 리뷰가 이걸 못 잡았는가

### 5.1 리뷰 스코프가 좁았음

6-pass 리뷰는 **코드 파일 내부 로직**만 봤음:
- 정확성: 변수명, 연산, 분기 로직
- 완전성: 누락된 엣지케이스, 빠진 파일
- 일관성: 파일 간 충돌

**안 본 것**:
- 이 코드가 **프로젝트의 다른 모델과 같은 패턴인가?** → stage_configs vs pipeline.yaml
- `pip install -e .` → 서버 시작 → curl이 **한번에 되는가?** → fresh install 테스트
- 의존성이 **전부 requirements에 있는가?** → minicpmo-utils 누락
- 서버 시작에 **어떤 플래그가 필요한가?** → --trust-remote-code, --skip-mm-profiling

### 5.2 하네스 실패인가, 컨텍스트 실패인가?

**둘 다.**

**하네스 실패**:
- `review-pass` 스킬의 렌즈(정확성/완전성/일관성)에 "프로젝트 패턴 준수" 렌즈가 없음
- "fresh install 재현성" 체크리스트가 없음
- 리뷰 스코프가 코드 파일로 한정 — 빌드/실행 환경을 안 봄

**컨텍스트 실패**:
- progress file에 "pipeline.yaml 방식 사용"이라고만 적고, "다른 모델은 stage_configs 사용"이라는 비교가 없음
- 이전 세션에서 수동 설치한 것(minicpmo-utils, ffmpeg 등)을 기록 안 함
- 서버 시작 필수 플래그를 기록 안 함

### 5.3 어떻게 하면 이런 실수를 안 하는가

**리뷰 렌즈 추가**:

| 새 렌즈 | 체크 항목 |
|---------|----------|
| 패턴 준수 | 같은 프로젝트의 다른 구현체와 동일한 방식인가? |
| 재현 가능성 | `git clone` → `pip install` → 실행이 한번에 되는가? |
| 의존성 완전성 | 코드의 모든 import가 requirements에 있는가? |
| 운영 완성도 | 서버 시작 명령이 문서화되어 있고, 필수 플래그가 명시되어 있는가? |

**세션 종료 체크리스트**:

1. 수동으로 한 작업(pip install, 설정 변경)이 전부 코드/스크립트에 반영됐는가?
2. pip freeze 했는가?
3. 서버 시작 명령이 스크립트에 있고, 그 스크립트로 실제 시작되는가?
4. 다른 모델과 다른 방식을 쓰고 있지는 않은가?

**근본 원칙**:
- 검증된 패턴을 따를 것. "더 새로운" ≠ "더 좋은"
- 코드 리뷰만으로는 운영 완성도를 검증할 수 없음. 실제 실행이 리뷰의 일부여야 함
- 기록하지 않은 작업은 없는 것과 같음
