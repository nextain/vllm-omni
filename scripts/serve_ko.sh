#!/usr/bin/env bash
# Phase 6-A: KO 서버 기동
# 실행 위치: repo 루트 또는 worktree (스크립트가 절대 경로 자동 계산)
# 실행: bash scripts/serve_ko.sh
set -euo pipefail

# scripts/ 상위 디렉터리 = repo 루트
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
LORA_PATH="${REPO_ROOT}/tools/ko_finetune/output/naia_lora"

if [ ! -d "${LORA_PATH}" ]; then
    echo "❌ LoRA 어답터 없음: ${LORA_PATH}"
    echo "   train_qwen3_lora.py 먼저 실행 후 재시도"
    exit 1
fi

LORA_ABS="${LORA_PATH}"

distrobox enter vllm-dev -- bash -c "
set -e
export NCCL_P2P_DISABLE=1
source /home/luke/.venvs/vllm-omni/bin/activate
cd /var/home/luke/dev/vllm-omni
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --enable-lora \
  --lora-modules naia=${LORA_ABS} \
  --trust-remote-code --port 8000
"
