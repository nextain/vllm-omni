#!/bin/bash
# Run a single MiniCPM-o 4.5 request with async_chunk enabled.
#
# Uses AsyncOmni so Talker and Code2Wav start *before* Thinker finishes,
# achieving true stage-level concurrency (TTFP ~0.07s vs ~6.5s sync).
#
# Prerequisites:
#   - minicpmo_async_chunk.yaml (has async_chunk: true)
#   - 2× RTX 3090 (48 GB total, no NVLink) — set NCCL_P2P_DISABLE=1
#
# Usage:
#   bash run_single_prompt_async_chunk.sh
#   bash run_single_prompt_async_chunk.sh --query-type text --modalities text
#   bash run_single_prompt_async_chunk.sh --stage-configs-path /path/to/custom.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export NCCL_P2P_DISABLE=1

python "${SCRIPT_DIR}/end2end_async_chunk.py" \
    --query-type use_audio \
    --stage-configs-path "${REPO_ROOT}/vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml" \
    --output-dir output_audio_async_chunk \
    "$@"
