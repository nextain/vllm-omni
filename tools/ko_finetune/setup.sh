#!/usr/bin/env bash
# Phase 1: KO 파인튜닝 환경 설정
# 실행 순서:
#   1. bash tools/ko_finetune/setup.sh                (Phase 1 환경 설정)
#   2. python3 tools/ko_finetune/preprocess_kspon.py  (Phase 2 데이터 전처리)
#   3. bash tools/ko_finetune/train_cosyvoice2_ko.sh  (Phase 3 CosyVoice2 학습)
#   4. python3 tools/ko_finetune/export_flow.py       (Phase 4 flow.pt 내보내기)
#   5. python3 tools/ko_finetune/train_qwen3_lora.py  (Phase 5 Qwen3 LoRA)
#   6. bash scripts/ko_e2e_test.sh                    (Phase 6 E2E 테스트)
# 실행 위치: repo 루트에서 distrobox enter vllm-dev -- bash tools/ko_finetune/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 1-A: CosyVoice 공식 레포 클론 ==="
if [ -d "$SCRIPT_DIR/CosyVoice" ]; then
    echo "CosyVoice 이미 존재 — skip"
else
    git clone https://github.com/FunAudioLLM/CosyVoice "$SCRIPT_DIR/CosyVoice"
fi
# Matcha-TTS submodule (matcha.*) 이니셌엄: cosyvoice2.yaml 로드 시 필수
git -C "$SCRIPT_DIR/CosyVoice" submodule update --init --recursive

echo "=== Phase 1-B: 파이썬 패키지 설치 ==="
source /home/luke/.venvs/vllm-omni/bin/activate
pip install g2pk         # 한국어 G2P (g2pkk는 미사용)
pip install peft         # Qwen3 LoRA
pip install soundfile    # PCM→WAV 변환
pip install datasets     # train_qwen3_lora.py Dataset 구성
pip install transformers # AutoModelForCausalLM, Trainer (LoRA 학습 필수)
pip install pandas pyarrow  # make_parquet_list.py 파켈 변환
pip install deepspeed       # train.py 무조건 import (사용 안하더라도 필요)
pip install tensorboard     # train_utils.py에서 SummaryWriter import 필요
pip install pyworld         # cosyvoice/dataset/processor.py import 필요
pip install hydra-core       # Matcha-TTS matcha/utils/__init__.py import ud544uc694
pip install conformer diffusers gdown grpcio grpcio-tools hydra-core inflect librosa lightning omegaconf openai-whisper rich wetext x-transformers  # CosyVoice training deps (from requirements.txt)

echo "=== 검증 ==="
python3 -c "import g2pk, peft, soundfile, datasets, transformers, pandas, pyarrow, deepspeed, tensorboard, pyworld; print('OK: all deps')"
echo "=== Phase 1 완료 ==="
