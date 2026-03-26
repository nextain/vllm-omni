#!/bin/bash
# MiniCPM-o 4.5 Transformers Benchmark Evaluation Script
# Run from vllm-omni root directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

echo "Working directory: $(pwd)"

if [[ ! -f "benchmarks/minicpm-o/transformers/minicpm_o_transformers.py" ]]; then
    echo "Error: Not in vllm-omni root directory."
    exit 1
fi

cd benchmarks/minicpm-o/transformers

NUM_PROMPTS="${1:-10}"  # Default 10 prompts (use 100 for full benchmark)

python minicpm_o_transformers.py \
    --prompts-file ../../build_dataset/top100.txt \
    --num-prompts "$NUM_PROMPTS"

echo ""
echo "Results saved to $(pwd)/benchmark_results:"
echo "  - perf_stats.json    E2E latency, RTF, audio throughput (per-prompt + aggregated)"
echo "  - results.json       Per-prompt text output and audio paths"
echo "  - audio/             Generated wav files"
echo ""
echo "Key metrics: rtf_avg (< 1.0 = realtime), audio_throughput_ratio, total_time_s_p95"
