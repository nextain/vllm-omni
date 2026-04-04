# vllm-omni Upstream 기본 이해

## 아키텍처

### 핵심 컴포넌트
- **Orchestrator**: 멀티 스테이지 파이프라인 중앙 제어. 요청 라우팅, 스테이지 전환
- **Stage Engine**: 각 스테이지별 독립 엔진 (AR: LLMEngine, Generation: DiffusionEngine)
- **Worker**: GPU에서 실제 forward pass 실행 (GPUARWorker, GPUGenerationWorker)
- **OmniConnector**: 스테이지 간 데이터 전송 (shared memory/network)

### 스테이지 타입
- **AR (Autoregressive)**: 텍스트/멀티모달 이해 + 생성 (thinker, talker)
- **Generation (DiT)**: 비자기회귀 생성 (code2wav, image gen)

### 서버 시작 플로우
```
CLI (vllm serve --omni)
  → AsyncOmniEngine.__init__()
    → Load stage_configs YAML
    → Create Orchestrator
    → For each stage:
      → setup_stage_devices() (CUDA_VISIBLE_DEVICES 설정)
      → Subprocess spawn (process: true일 때)
      → Model load (init_vllm_registered_model)
      → Worker spawn (forward loop)
    → Orchestrator ready
  → Uvicorn serve (HTTP API)
```

## Stage Config 핵심 필드

### runtime
| 필드 | 기본값 | 의미 |
|------|--------|------|
| `process` | true | **별도 OS 프로세스에서 실행** (CUDA context 격리) |
| `devices` | "0" | CUDA_VISIBLE_DEVICES 설정. "0", "1", "0,1" |

### stage-level
| 필드 | 의미 |
|------|------|
| `stage_type` | "llm" (기본, AR+generation 모두) — 현재 유일한 값 |

### engine_args
| 필드 | 의미 |
|------|------|
| `model_stage` | "thinker", "talker", "code2wav" 등 |
| `model_arch` | registry.py 등록 클래스명 |
| `worker_type` | "ar" (자기회귀) 또는 "generation" (diffusion/vocoder) |
| `gpu_memory_utilization` | GPU VRAM 할당 비율 (0.0-1.0), GPU 당 적용 |
| `tensor_parallel_size` | TP GPU 수 (devices 수와 일치해야 함) |
| `enforce_eager` | CUDA graph 비활성. qwen2.5: 전부 true, qwen3: thinker/talker false + code2wav true |
| `engine_output_type` | "latent" (hidden states), "text", "audio" |
| `hf_config_name` | HF config 내 sub-config 이름 (qwen3: thinker_config, talker_config) |
| `max_num_seqs` | 최대 동시 시퀀스 수. qwen2.5: 1-3, qwen3: 32-64 (모델별 다름) |
| `async_scheduling` | 비동기 스케줄링. generation stage에서 false로 설정 |
| `max_model_len` | 최대 시퀀스 길이. 미설정 시 모델 기본값 |

### 메모리 규칙
- **같은 GPU에 동시 실행 stage**: `gpu_memory_utilization` 합 ≤ 1.0
- **같은 GPU에 순차 실행 stage**: 시간이 겹치지 않으면 독립 할당 가능 (예: Thinker 완료 후 Code2Wav 시작)
- **다른 GPU면**: 각각 독립적으로 1.0까지 가능
- **TP=N**: `devices: "0,1,...N-1"` → 각 GPU에 모델 1/N 로드, gpu_memory_utilization은 **GPU 당** 적용

## Multi-GPU

### 패턴 1: 스테이지별 GPU 분리 (Qwen2.5-Omni 방식)
```yaml
# qwen2_5_omni.yaml에서 발췌 — process: true 명시
- stage_id: 0
  runtime:
    process: true
    devices: "0"
- stage_id: 1
  runtime:
    process: true
    devices: "1"
```
**참고**: qwen3_omni_moe.yaml에서는 `process` 필드 생략 (기본값 true). 명시적으로 쓰는 것이 안전.

### 패턴 2: TP 분산 (GPU memory docs 예시)
```yaml
- stage_id: 0
  runtime:
    devices: "0,1"
  engine_args:
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.6  # GPU 당 60%
```

### 패턴 3: 순차 실행 강제 (Qwen2.5-Omni runtime section)
```yaml
# qwen2_5_omni.yaml의 top-level runtime — OOM 방지
runtime:
  defaults:
    window_size: -1      # 상위 stage 완전 완료 대기
    max_inflight: 1      # 동시 요청 1개만
```
**주의**: qwen3_omni_moe.yaml에는 이 runtime section이 없음.

### RTX 3090 x2 (NVLink 없음)
```bash
NCCL_P2P_DISABLE=1 vllm serve ... --omni
```

### 알려진 이슈
- **vllm-omni#1387**: multi-GPU OOM — OPEN 상태
  - 2xH100, 4xL40S에서도 보고됨
  - 원인: 스테이지 동시 실행 시 메모리 충돌 + KV cache 해제 타이밍
  - 완화: `runtime.defaults.window_size: -1` + `max_inflight: 1` (순차 실행)
  - qwen2.5에만 적용됨, qwen3에는 미적용

## 레퍼런스 모델 비교

| | Qwen3-Omni | Qwen2.5-Omni | MiMo-Audio |
|--|-----------|-------------|-----------|
| stages | 3 | 3 | 2 (fused) |
| process 명시 | 생략 (기본 true) | ✅ 명시 | ✅ 명시 |
| TP 필드 | 있음 (기본 1) | 없음 | 있음 (기본 1) |
| runtime section | 없음 | ✅ (window_size, max_inflight) | 없음 |
| eager 필수 | 일부 (code2wav만) | 전부 | 전부 |
| GPU mem | 0.9+0.6+0.1 | 0.8+0.8+0.15 | 0.3+0.2 |
| 검증 HW | 2xH100-80G | 1xH100-80G (stage_configs.md) | 1xH20-96G |
