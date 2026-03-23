#!/bin/bash
# vllm-omni 코드를 RunPod으로 rsync
# 사용: bash scripts/sync_to_runpod.sh <SSH_PORT>
# 예:   bash scripts/sync_to_runpod.sh 22161

set -e

SSH_PORT="${1}"
if [ -z "$SSH_PORT" ]; then
  echo "Usage: $0 <SSH_PORT>"
  echo "  SSH 포트는 RunPod REST API로 확인:"
  echo "  curl -s https://rest.runpod.io/v1/pods/oy1teunjdptwrn -H 'Authorization: Bearer \$RUNPOD_API_KEY' | python3 -c \"import sys,json; print(json.load(sys.stdin)['portMappings'])\""
  exit 1
fi

RUNPOD_HOST="69.30.85.139"
REMOTE_DIR="/workspace/vllm-omni"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[INFO] 로컬: $LOCAL_DIR"
echo "[INFO] 원격: root@${RUNPOD_HOST}:${SSH_PORT} -> ${REMOTE_DIR}"

rsync -az --no-perms --no-owner --no-group \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'vllm_omni.egg-info' \
  --exclude 'uv.lock' \
  -e "ssh -p ${SSH_PORT} -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
  "${LOCAL_DIR}/" \
  "root@${RUNPOD_HOST}:${REMOTE_DIR}/"

echo "[OK] 동기화 완료"
echo ""
echo "다음 단계 (RunPod에서):"
echo "  1. source /workspace/venv/bin/activate  # venv 이미 설치됐으면"
echo "  2. pip install -e /workspace/vllm-omni --no-build-isolation --no-deps -q"
echo "  3. bash /workspace/scripts/workspace_start.sh"
