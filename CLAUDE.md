# vllm-omni (Nextain Fork)

Nextain's fork of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni).
MiniCPM-o 4.5 omni model support contribution.

## Mandatory Reads

1. `.agents/context/upstream-basics.md` — vllm-omni 프레임워크 기본 이해 (아키텍처, stage config, multi-GPU)
2. `.agents/context/our-changes.md` — 우리가 수정한 것과 현재 상태
3. `docs/configuration/stage_configs.md` — upstream 공식 stage config 문서
4. `docs/configuration/gpu_memory_utilization.md` — upstream GPU 메모리 가이드

## 핵심 원칙

### 1. Upstream 먼저
- 문제 발견 시 **코드 수정 전에** upstream 문서/이슈/PR 검색
- 유사 모델 구현 참조 (qwen3_omni, qwen2_5_omni, mimo_audio)
- 검증된 패턴만 사용, 자체 발명 금지

### 2. 기본 사용법 숙지
- 코드 수정 전에 **서버 기본 실행이 되는 상태**에서 시작
- stage_configs, multi-GPU, CLI 옵션을 정확히 이해한 후 작업
- E2E 테스트는 **실제 사용 환경**(2-GPU, Naia Shell)에서 수행

### 3. 개발 프로세스
- Issue-Driven Development (naia-os와 동일)
- 반복 리뷰 시 upstream 검증 렌즈 필수 (Pass 5)
- progress file로 세션 인수인계

## Upstream References

| 문서 | URL |
|------|-----|
| 공식 문서 | https://docs.vllm.ai/projects/vllm-omni/ |
| Stage Configs | docs/configuration/stage_configs.md |
| GPU Memory | docs/configuration/gpu_memory_utilization.md |
| Adding Omni Model | docs/contributing/model/adding_omni_model.md |
| Qwen3-Omni Example | examples/online_serving/qwen3_omni/ |
| Upstream Issues | https://github.com/vllm-project/vllm-omni/issues |

## 서버 시작

```bash
# 기본 (stage_configs 자동 로드)
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --port 8000

# 커스텀 stage_configs
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --port 8000

# RTX 3090 x2 (NCCL P2P 비활성 필수)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path minicpmo.yaml \
  --trust-remote-code --port 8000

# distrobox 환경
distrobox enter vllm-dev -- bash -c "source /home/luke/.venvs/vllm-omni/bin/activate && ..."
```

## 환경

- distrobox: vllm-dev (nvidia/cuda:12.8.0-devel-ubuntu24.04)
- venv: /home/luke/.venvs/vllm-omni (Python 3.12, vllm 0.18.0)
- GPU: RTX 3090 x2 (24GB each, NVLink 없음 → NCCL_P2P_DISABLE=1)
- 모델: ~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/ (18GB)

## 관련 이슈

| 이슈 | 제목 | 상태 |
|------|------|:----:|
| nextain/naia-os#181 | MiniCPM-o 기본 지원 | 진행중 |
| nextain/naia-os#191 | Audio input 지원 | 진행중 |
| nextain/naia-os#192 | Naia Shell E2E | 대기 |
| vllm-omni#1182 | MiniCPM-o upstream | Allyyi 담당 |
| vllm-omni#1387 | Multi-GPU OOM (알려진 버그) | OPEN |
