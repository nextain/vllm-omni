#!/usr/bin/env bash
# Phase 3: CosyVoice2 flow 파인튜닝
# 실행 위치: repo 루트 (data/ko_2h/ 등 상대 경로 기준)
# 실행: distrobox enter vllm-dev -- bash tools/ko_finetune/train_cosyvoice2_ko.sh
set -euo pipefail

source /home/luke/.venvs/vllm-omni/bin/activate

SNAPSHOT=~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5\
/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc
OUTPUT_DIR=tools/ko_finetune/output/ko_flow

mkdir -p "$OUTPUT_DIR"

echo "=== Phase 3: CosyVoice2 flow 학습 시작 ==="
echo "  데이터: data/ko_2h/train.list"
echo "  출력: $OUTPUT_DIR"

export NCCL_P2P_DISABLE=1

torchrun --nproc_per_node=2 \
  tools/ko_finetune/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model flow \
  --config tools/ko_finetune/config/cosyvoice2_ko.yaml \
  --checkpoint "$SNAPSHOT/assets/token2wav/flow.pt" \
  --train_data data/ko_2h/train.list \
  --cv_data data/ko_2h/dev.list \
  --model_dir "$OUTPUT_DIR"

echo "=== Phase 3 완료: $OUTPUT_DIR 확인 ==="
ls -la "$OUTPUT_DIR"
