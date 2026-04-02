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

### engine_args
| 필드 | 의미 |
|------|------|
| `model_stage` | "thinker", "talker", "code2wav" |
| `model_arch` | registry.py 등록 클래스명 |
| `worker_type` | "ar" 또는 "generation" |
| `gpu_memory_utilization` | GPU VRAM 할당 비율 (0.0-1.0) |
| `tensor_parallel_size` | TP GPU 수 (devices에 맞춰야 함) |
| `enforce_eager` | true=CUDA graph 비활성 (대부분 true) |
| `engine_output_type` | "latent", "text", "audio" |
| `hf_config_name` | HF config 내 sub-config 이름 |

### 메모리 규칙
- **같은 GPU에 여러 stage**: `gpu_memory_utilization` 합 ≤ 1.0
- **다른 GPU면**: 각각 독립적으로 1.0까지 가능
- **TP=2**: `devices: "0,1"` → 각 GPU에 모델 절반 로드

## Multi-GPU

### 패턴 1: 스테이지별 GPU 분리
```yaml
- stage_id: 0
  runtime: {devices: "0", process: true}  # GPU 0
- stage_id: 1
  runtime: {devices: "1", process: true}  # GPU 1
```

### 패턴 2: TP 분산
```yaml
- stage_id: 0
  runtime: {devices: "0,1", process: true}
  engine_args: {tensor_parallel_size: 2}
```

### RTX 3090 x2 (NVLink 없음)
```bash
NCCL_P2P_DISABLE=1 vllm serve ... --omni
```

### 알려진 이슈
- **#1387**: multi-GPU OOM — 스테이지 동시 실행 시 메모리 충돌
- **해결**: `runtime.defaults.window_size: -1` + `max_inflight: 1` (순차 실행)

## 레퍼런스 모델 비교

| | Qwen3-Omni | Qwen2.5-Omni | MiMo-Audio |
|--|-----------|-------------|-----------|
| stages | 3 | 3 | 2 (fused) |
| process: true | ✅ | ✅ | ✅ |
| TP 지원 | ✅ | ✗ | ✅ |
| eager 필수 | 일부 | 전부 | 전부 |
| GPU mem | 0.9+0.6+0.1 | 0.8+0.8+0.15 | 0.3+0.2 |
| 검증 HW | 2xH100-80G | 2xH100-80G | 1xH20-96G |
