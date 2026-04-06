#!/bin/bash
# MiniCPM-o 4.5 — async_chunk streaming serve (2x RTX 3090)
# Run inside distrobox vllm-dev:
#   distrobox enter vllm-dev -- bash scripts/serve_async_chunk.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="/home/luke/.venvs/vllm-omni"
MODEL="openbmb/MiniCPM-o-4_5"
STAGE_CONFIG="$REPO_ROOT/vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml"

cd "$REPO_ROOT"

source "$VENV/bin/activate"

echo "=== Environment ==="
python --version
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')"

echo ""
echo "=== Starting vllm-omni serve (async_chunk, 2x RTX 3090) ==="
echo "Model: $MODEL"
echo "Stage config: $STAGE_CONFIG"
echo "Port: 8000"
echo "Ctrl+C to stop"
echo ""

export NCCL_P2P_DISABLE=1
"$VENV/bin/vllm" serve "$MODEL" --omni \
    --stage-configs-path "$STAGE_CONFIG" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
