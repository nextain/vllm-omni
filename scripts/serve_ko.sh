#!/usr/bin/env bash
# Phase 6-A: KO 서버 기동
# 실행 위치: repo 루트 (tools/ko_finetune/output/naia_lora 상대 경로 기준)
# 실행: bash scripts/serve_ko.sh
set -euo pipefail

export NCCL_P2P_DISABLE=1

distrobox enter vllm-dev -- bash -c '
source /home/luke/.venvs/vllm-omni/bin/activate
cd /var/home/luke/dev/vllm-omni
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --enable-lora \
  --lora-modules naia=tools/ko_finetune/output/naia_lora \
  --trust-remote-code --port 8000
'
