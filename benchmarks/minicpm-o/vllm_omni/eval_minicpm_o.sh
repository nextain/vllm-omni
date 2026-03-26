#!/bin/bash
# MiniCPM-o 4.5 vLLM-Omni Pipeline Benchmark
# Run from vllm-omni root directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

echo "Working directory: $(pwd)"

if [[ ! -d "benchmarks/minicpm-o/vllm_omni" ]]; then
    echo "Error: Not in vllm-omni root directory."
    exit 1
fi

log_dir=benchmarks/minicpm-o/vllm_omni/logs
outputs_dir=benchmarks/minicpm-o/vllm_omni/outputs
build_dataset_path=benchmarks/build_dataset/top100.txt

# MiniCPM-o uses the generic omni end2end runner
# Adjust the script path if a model-specific runner exists
end2end_script_path=examples/offline_inference/end2end.py

NUM_PROMPTS="${1:-10}"  # Default 10 prompts

python $end2end_script_path \
    --model openbmb/MiniCPM-o-4_5 \
    --output-wav $outputs_dir \
    --query-type text \
    --txt-prompts $build_dataset_path \
    --num-prompts "$NUM_PROMPTS" \
    --log-stats \
    --log-dir $log_dir

echo ""
echo "Logs:    ${log_dir}/"
echo "Outputs: ${outputs_dir}/"
echo ""
echo "Key files:"
echo "  - *.orchestrator.stats.jsonl   per-stage latency"
echo "  - *.overall.stats.jsonl        E2E latency/TPS"
echo "  - *.stage{0,1,2}.log           per-stage logs"
echo ""
echo "Key checks: overall RTF < 1.0, stable per-stage latency, no errors in stage logs"
