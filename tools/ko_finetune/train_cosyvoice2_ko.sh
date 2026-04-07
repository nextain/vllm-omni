#!/usr/bin/env bash
# Phase 3: CosyVoice2 flow 파인튜닝
# 실행: distrobox enter vllm-dev -- bash tools/ko_finetune/train_cosyvoice2_ko.sh
set -euo pipefail

# 스크립트 위치 기반으로 repo 루트로 이동 (tools/ko_finetune/ 두 단계 위)
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source /home/luke/.venvs/vllm-omni/bin/activate

# 선결 조건 확인: Kaldi 포맷 파일 존재 여부
if [ ! -f data/ko_2h/train/wav.scp ]; then
    echo "❌ data/ko_2h/train/wav.scp 없음. preprocess_kspon.py 먼저 실행 후 재시도"
    exit 1
fi
if [ ! -f data/ko_2h/dev/wav.scp ]; then
    echo "❌ data/ko_2h/dev/wav.scp 없음. preprocess_kspon.py 먼저 실행 후 재시도"
    exit 1
fi

SNAPSHOT=~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc
OUTPUT_DIR=tools/ko_finetune/output/ko_flow

mkdir -p "$OUTPUT_DIR"

# CosyVoice make_parquet_list.py를 사용해 Kaldi 포맷 -> parquet 변환
# (CosyVoice train.py는 --train_data에 parquet 파일 목록 파일을 기대)
echo "=== Phase 3-A: Kaldi -> parquet 변환 ==="
COSYVOICE_TOOLS=tools/ko_finetune/CosyVoice/tools
mkdir -p data/ko_2h/train/parquet data/ko_2h/dev/parquet
python $COSYVOICE_TOOLS/make_parquet_list.py \
  --src_dir data/ko_2h/train \
  --des_dir data/ko_2h/train/parquet
python $COSYVOICE_TOOLS/make_parquet_list.py \
  --src_dir data/ko_2h/dev \
  --des_dir data/ko_2h/dev/parquet

# parquet 전환 결과 검증 (data.list 비어 있으면 학습 불가)
if [ ! -s data/ko_2h/train/parquet/data.list ]; then
    echo "❌ data/ko_2h/train/parquet/data.list 비어 있음. make_parquet_list.py 실패 가능성"
    exit 1
fi

echo "=== Phase 3-B: CosyVoice2 flow 학습 시작 ==="
echo "  데이터: data/ko_2h/train/parquet/data.list ($(wc -l < data/ko_2h/train/parquet/data.list)파켈)"

export NCCL_P2P_DISABLE=1

# CosyVoice + Matcha-TTS PYTHONPATH
# cosyvoice.* = FunAudioLLM/CosyVoice (학습 스크립트 + 데이터 파이프라인)
# stepaudio2.cosyvoice2.* = site-packages 설치 패키지 직접 참조
# 주의: STEPAUDIO2_ROOT를 PYTHONPATH에 넣으면 안 됨 — stepaudio2/ 디렉토리가 패키지를 새도잉하여
#              from stepaudio2.utils import ... 실패 발생
COSYVOICE_ROOT=tools/ko_finetune/CosyVoice
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

# --master_port: 포트 충돌 방지 (29500 기본포트 회피)
MASTER_PORT=$(shuf -i 29501-29599 -n 1)

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
  tools/ko_finetune/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model flow \
  --config tools/ko_finetune/config/cosyvoice2_ko.yaml \
  --checkpoint "$SNAPSHOT/assets/token2wav/flow.pt" \
  --onnx_path "$SNAPSHOT/assets/token2wav" \
  --qwen_pretrain_path "$SNAPSHOT" \
  --use_amp \
  --train_data data/ko_2h/train/parquet/data.list \
  --cv_data data/ko_2h/dev/parquet/data.list \
  --num_workers 4 \
  --prefetch 4 \
  --model_dir "$OUTPUT_DIR"

echo "=== Phase 3 완료: $OUTPUT_DIR 확인 ==="
ls -la "$OUTPUT_DIR"
