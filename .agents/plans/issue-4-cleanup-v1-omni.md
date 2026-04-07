# Plan: Issue #4 — /v1/omni 제거 및 ref/ 이동

## 목적
독자 규격 `/v1/omni` 엔드포인트를 제거하고 참고용으로 보존.
업스트림 `/v1/realtime` 확장 경로로 전환하기 위한 선행 정리.

## 분석 요약

### 영향 파일 (총 2개)

**1. `vllm_omni/entrypoints/openai/serving_omni_duplex.py`**
- 단 1개 파일(`api_server.py`)에서만 import됨
- 제거해도 다른 코드에 영향 없음

**2. `vllm_omni/entrypoints/openai/api_server.py`**
- 줄 112: `from vllm_omni.entrypoints.openai.serving_omni_duplex import OmniFullDuplexHandler`
- 줄 822-826: `setup_app_state()` 내 `OmniFullDuplexHandler` 초기화
- 줄 1213-1228: `@router.websocket("/v1/omni")` 라우트 전체

## 작업 계획

### Step 1: ref/ 디렉토리에 복사
```
대상: vllm_omni/entrypoints/openai/serving_omni_duplex.py
목적지: ref/omni_duplex_v1/serving_omni_duplex.py
```
- ref 디렉토리는 git 추적은 하되 서버 실행에 관여하지 않음
- `ref/omni_duplex_v1/README.md` 추가 (제거 이유 + 대체 경로 설명)

### Step 2: api_server.py 수정 (3곳)

**제거 1 — Import (줄 112):**
```python
# 제거할 줄:
from vllm_omni.entrypoints.openai.serving_omni_duplex import OmniFullDuplexHandler
```

**제거 2 — 초기화 (줄 822-826):**
```python
# 제거할 블록:
if state.openai_serving_chat is not None:
    state.openai_omni_duplex = OmniFullDuplexHandler(
        chat_service=state.openai_serving_chat,
    )
else:
    state.openai_omni_duplex = None
```

**제거 3 — 라우트 (줄 1213-1228):**
```python
# 제거할 블록:
@router.websocket("/v1/omni")
async def omni_duplex_websocket(websocket: WebSocket):
    ...
```

### Step 3: serving_omni_duplex.py 원본 삭제
- `vllm_omni/entrypoints/openai/serving_omni_duplex.py` 삭제

## 검증 방법

1. **import 검증**: `python -c "from vllm_omni.entrypoints.openai.api_server import app"` — 오류 없어야 함
2. **소스 검증 (제거 완료 확인)**: 실제 제거됐는지 소스 grep으로 확인
   ```bash
   grep -r 'serving_omni_duplex\|OmniFullDuplexHandler\|v1/omni' vllm_omni/ -- 0건이어야 함
   ```
3. **서버 기동**: distrobox vllm-dev에서 서버 실행 — 정상 기동 확인
4. **라우트 검증**: `/v1/omni` WebSocket 연결 시도 → 404 (또는 연결 거부) 확인
5. **기존 기능 유지**: `/v1/realtime` ASR 동작 확인

## 주의사항
- api_server.py 수정 시 줄 번호 변동 주의 — 컨텍스트로 제거
- `setup_app_state()` 함수 내 다른 초기화 로직 건드리지 않도록 주의
- WebSocket route는 `/v1/realtime` 바로 위에 있음 — 인접 라우트 파악 후 제거

## 파일 최종 상태
```
삭제: vllm_omni/entrypoints/openai/serving_omni_duplex.py
수정: vllm_omni/entrypoints/openai/api_server.py (3곳 제거)
추가: ref/omni_duplex_v1/serving_omni_duplex.py (복사본)
추가: ref/omni_duplex_v1/README.md
```
