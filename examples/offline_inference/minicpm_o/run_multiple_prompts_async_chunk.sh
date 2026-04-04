#!/bin/bash
# Run multiple MiniCPM-o 4.5 requests with async_chunk enabled.
#
# Uses AsyncOmni with --max-in-flight to control request-level concurrency.
# Each request still gets stage-level concurrency via async_chunk.
#
# Usage:
#   bash run_multiple_prompts_async_chunk.sh
#   bash run_multiple_prompts_async_chunk.sh --max-in-flight 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PROMPTS_FILE="${SCRIPT_DIR}/text_prompts_10.txt"
if [[ ! -f "${PROMPTS_FILE}" ]]; then
    echo "[Error] Prompts file not found: ${PROMPTS_FILE}" >&2
    exit 1
fi

export NCCL_P2P_DISABLE=1

python "${SCRIPT_DIR}/end2end_async_chunk.py" \
    --query-type text \
    --txt-prompts "${PROMPTS_FILE}" \
    --stage-configs-path "${REPO_ROOT}/vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml" \
    --output-dir output_audio_async_chunk \
    --max-in-flight 1 \
    "$@"
