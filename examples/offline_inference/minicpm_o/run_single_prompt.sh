#!/bin/bash
# Run a single MiniCPM-o 4.5 request (sync mode).
#
# Usage:
#   bash run_single_prompt.sh
#   bash run_single_prompt.sh --query-type use_image --image-path /path/to/image.jpg
#   bash run_single_prompt.sh --query-type text --modalities text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/end2end.py" \
    --query-type use_audio \
    --stage-configs-path "${REPO_ROOT}/vllm_omni/model_executor/stage_configs/minicpmo.yaml" \
    --output-dir output_audio \
    "$@"
