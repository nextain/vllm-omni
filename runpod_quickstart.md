# RunPod Quick Start — MiniCPM-o 4.5 vllm-omni

E2E 검증 환경: A40 46GB, 2026-03-24

## 1. 새 Pod 설정

**권장 사양**: NVIDIA A40 46GB (On-Demand)
**이미지**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` 또는 유사

## 2. 초기 환경 구성 (최초 1회)

```bash
# 1. 모델 다운로드 (HF cache)
mkdir -p /workspace/.cache_hf
HF_HOME=/workspace/.cache_hf pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('openbmb/MiniCPM-o-4.5', cache_dir='/workspace/.cache_hf')
"

# 2. vllm-omni 코드 복사
cp -r /path/to/vllm-omni /workspace/vllm-omni

# 3. Python 환경 (venv)
python3 -m venv /workspace/venv
/workspace/venv/bin/pip install -e /workspace/vllm-omni[all]

# 4. 시작 스크립트 복사
cp /workspace/vllm-omni/start_server.sh /workspace/start_server.sh
chmod +x /workspace/start_server.sh
```

## 3. 서버 시작

```bash
# GPU 클린 확인 (중요!)
nvidia-smi
# Processes 항목에 아무것도 없어야 함. 있으면:
# nvidia-smi | grep VLLM | awk '{print $5}' | xargs kill -9

# 서버 시작
truncate -s 0 /workspace/server.log
nohup bash /workspace/start_server.sh &

# 로그 모니터링 (~10분 대기)
tail -f /workspace/server.log | grep -E "(Stage.*ready|ERROR|All.*stages)"
```

## 4. E2E 테스트

```bash
MODEL=/workspace/.cache_hf/models--openbmb--MiniCPM-o-4_5/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc
curl -s -X POST http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, say good morning.\"}], \"max_tokens\": 50}" \
  | python3 -m json.tool | head -40
# 기대: choices[0]=텍스트, choices[1]=오디오(base64 WAV)
```

## 5. 주요 파일 경로

| 파일 | 경로 |
|------|------|
| 서버 시작 스크립트 | `/workspace/start_server.sh` |
| 서버 로그 | `/workspace/server.log` |
| vllm-omni 코드 | `/workspace/vllm-omni/` |
| HF 모델 캐시 | `/workspace/.cache_hf/` |
| Python 환경 | `/workspace/venv/` |
| 모델 스냅샷 해시 | `44151b35f1b232a280bda5a87ea1a7575d5433fc` |

## 6. 트러블슈팅

### OOM / Stage-0 실패
```bash
# 이전 실행 좀비 프로세스 확인
nvidia-smi
# VLLM::EngineCore 프로세스 있으면 PID kill
kill -9 <PID1> <PID2> ...
# GPU 0 MiB 확인 후 재시작
```

### GPU 메모리 배분 (pipeline.yaml)
- Stage-0 (Thinker): 50% = 22.2 GiB
- Stage-1 (Talker): 15% + enforce_eager
- Stage-2 (Code2Wav): 20% + enforce_eager
- 합계 85% ≤ 90% (vllm-omni 권장)
