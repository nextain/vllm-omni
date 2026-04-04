#!/bin/bash
# Run multiple MiniCPM-o 4.5 requests from a text file (sync mode).
#
# Uses py_generator mode for memory-efficient large batch processing.
#
# Usage:
#   bash run_multiple_prompts.sh
#   bash run_multiple_prompts.sh --num-prompts 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/end2end.py" \
    --query-type text \
    --txt-prompts "${SCRIPT_DIR}/text_prompts_10.txt" \
    --stage-configs-path "${REPO_ROOT}/vllm_omni/model_executor/stage_configs/minicpmo.yaml" \
    --output-dir output_audio \
    --py-generator \
    "$@"
