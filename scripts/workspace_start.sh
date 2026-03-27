#!/bin/bash
# /workspace/scripts/workspace_start.sh
# vllm-omni MiniCPM-o 4.5 서버 시작
# 사용: bash /workspace/scripts/workspace_start.sh
set -e

VENV=/workspace/venv
export HF_HOME=/workspace/.cache_hf
export TMPDIR=/workspace/tmp

MODEL_PATH="/workspace/.cache_hf/models--openbmb--MiniCPM-o-4_5/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc"
PORT=8091

# venv 확인
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[ERROR] venv 없음. 먼저 실행: bash /workspace/scripts/workspace_setup.sh"
  exit 1
fi
source "$VENV/bin/activate"

# 모델 확인
[ -d "$MODEL_PATH" ] || {
  echo "[ERROR] 모델 없음: $MODEL_PATH"
  exit 1
}

echo "[INFO] MiniCPM-o 4.5 vllm-omni 서버 시작"
echo "  모델: $MODEL_PATH"
echo "  포트: $PORT"

vllm serve "$MODEL_PATH" \
  --omni \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len 2048 \
  --skip-mm-profiling \
  2>&1 | tee /workspace/vllm_omni_server.log
