#!/bin/bash
# MiniCPM-o 4.5 vllm-omni server startup script
# RunPod A40 46GB, BF16 mode
# E2E verified: 2026-03-24
MODEL=/workspace/.cache_hf/models--openbmb--MiniCPM-o-4_5/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc
PYTHONPATH=/workspace/vllm-omni PYTHONUNBUFFERED=1 \
  /workspace/venv/bin/vllm serve $MODEL \
  --omni \
  --port 8091 \
  --trust-remote-code \
  --max-model-len 4096 \
  --skip-mm-profiling \
  > /workspace/server.log 2>&1
