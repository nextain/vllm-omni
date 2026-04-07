#!/usr/bin/env bash
# Phase 6-B: E2E 테스트
# 실행 위치: repo 루트
# 사전 준비: data/ko_test.wav 준비 (KsponSpeech eval 샘플 또는 마이크 녹음)
set -euo pipefail

# 한국어 음성 입력파일
TEST_WAV="data/ko_test.wav"
if [ ! -f "$TEST_WAV" ]; then
    echo "❌ $TEST_WAV 없음. KsponSpeech eval 샘플을 복사하거나 마이크 녹음 필요"
    exit 1
fi

# 1. vllm-omni 서버 기동 (KO flow.pt + Naia LoRA)
echo "=== 서버 기동 ==="
bash scripts/serve_ko.sh &
sleep 30  # 서버 준비 대기

# 2. 음성 입력 -> 응답
# scripts/test_omni_duplex.py 는 기존 repo에 이미 있는 스크립트
echo "=== E2E 테스트 실행 ==="
distrobox enter vllm-dev -- bash -c '
source /home/luke/.venvs/vllm-omni/bin/activate
cd /var/home/luke/dev/vllm-omni
python scripts/test_omni_duplex.py \
  --input data/ko_test.wav \
  --output /tmp/ko_response.wav
'

# 3. 예상 결과: 한국어로 답하는 Naia 음성 출력

# 4. 서버 종료
pkill -f "vllm serve" 2>/dev/null  # distrobox 프로세스는 호스트에서 가시적 - pkill 작동
echo "=== E2E 완료: /tmp/ko_response.wav 확인 ==="
