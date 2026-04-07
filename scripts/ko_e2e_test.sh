#!/usr/bin/env bash
# Phase 6-B: E2E 테스트
# 실행 위치: repo 루트 또는 worktree
set -euo pipefail

# scripts/ 상위 디렉토리 = repo 루트
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"

# 서버 종료 함수 (trap 연결)
# distrobox vllm-dev는 --pid=host 사용 — pkill이 컨테이너 내부 프로세스를 바라볼 수 있음
cleanup() {
    echo "[cleanup] vllm serve 종료..."
    pkill -f 'vllm serve' 2>/dev/null || true
}
trap cleanup EXIT

# 포트 8000이 이미 사용 중이면 정리
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "[warn] port 8000 occupied -- killing existing process"
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 1. vllm-omni 서버 기동 (KO flow.pt + Naia LoRA)
echo "=== 서버 기동 ==="
bash "${REPO_ROOT}/scripts/serve_ko.sh" &

# 서버 헬스체크 (최대 120초 대기)
echo "서버 시작 대기 중..."
for i in $(seq 1 24); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ 서버 준비 완료 (${i}회 시도, $((i * 5))초)"
        break
    fi
    if [ $i -eq 24 ]; then
        echo "❌ 서버 시작 실패 (120초 초과)"
        exit 1
    fi
    sleep 5
done

# 2. 음성 입력 -> 응답 (합성 1.5s PCM 톤, --audio 플래그)
echo "=== E2E 테스트 실행 ==="
distrobox enter vllm-dev -- bash -c "
set -e
source /home/luke/.venvs/vllm-omni/bin/activate
cd /var/home/luke/dev/vllm-omni
python scripts/test_omni_duplex.py --audio --model naia
"

echo "=== E2E 완료 ==="
