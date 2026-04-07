#!/usr/bin/env bash
# Phase 1: KO 파인튜닝 환경 설정
# 실행: repo 루트에서 distrobox enter vllm-dev -- bash tools/ko_finetune/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 1-A: CosyVoice 공식 레포 클론 ==="
if [ -d "$SCRIPT_DIR/CosyVoice" ]; then
    echo "CosyVoice 이미 존재 — skip"
else
    git clone https://github.com/FunAudioLLM/CosyVoice "$SCRIPT_DIR/CosyVoice"
fi

echo "=== Phase 1-B: 파이썬 패키지 설치 ==="
source /home/luke/.venvs/vllm-omni/bin/activate
pip install g2pk g2pkk   # 한국어 G2P
pip install peft         # Qwen3 LoRA
pip install soundfile    # PCM→WAV 변환

echo "=== 검증 ==="
python3 -c "import g2pk, peft, soundfile; print('OK: g2pk, peft, soundfile')"
echo "=== Phase 1 완료 ==="
