#!/bin/bash
# /workspace/scripts/workspace_setup.sh
# pod 재시작 후 실행 — /workspace/venv에 설치하므로 재시작 후에도 유지됨
# 사용: bash /workspace/scripts/workspace_setup.sh
# 설치 완료 후: bash /workspace/scripts/workspace_start.sh
set -e

VENV=/workspace/venv
export HF_HOME=/workspace/.cache_hf
export TMPDIR=/workspace/tmp
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

mkdir -p /workspace/tmp

# venv가 이미 있고 vllm_omni가 설치돼 있으면 스킵
if [ -f "$VENV/bin/python" ]; then
  if "$VENV/bin/python" -c "import vllm_omni" 2>/dev/null; then
    echo "[OK] /workspace/venv 이미 설치됨. 바로 서버 시작 가능:"
    echo "     bash /workspace/scripts/workspace_start.sh"
    exit 0
  else
    echo "[INFO] venv 존재하지만 vllm_omni 미설치 — 설치 재개..."
  fi
fi

echo "[0/4] /workspace/venv 생성..."
python3 -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"

echo "[1/4] vllm 설치..."
pip install vllm==0.17.1 -q 2>&1 | tail -3

# vllm 설치 후 torch 깨짐 방지
python3 -c "import torch; torch.Tensor" 2>/dev/null || {
  echo "  torch 깨짐 감지 → 재설치..."
  pip install --force-reinstall torch==2.10.0 --no-deps -q 2>&1 | tail -2
}
rm -rf "$VENV/lib/python3.11/site-packages/~orch" \
       "$VENV/lib/python3.11/site-packages/~orch-"*.dist-info \
       "$VENV/lib/python3.11/site-packages/~unctorch" 2>/dev/null || true
python3 -c "import torch; print('  torch:', torch.__version__, '/ CUDA:', torch.cuda.is_available())"
python3 -c "import vllm._C; print('  vllm C ext: OK')"

echo "[2/4] 의존성 설치..."
pip install setuptools_scm s3tokenizer soxr -q 2>&1 | tail -2
pip install "librosa>=0.11.0" "diffusers>=0.36.0" "accelerate==1.12.0" -q 2>&1 | tail -2
pip install "soundfile>=0.13.1" "torchsde>=0.2.6" "x-transformers>=2.12.2" -q 2>&1 | tail -2
pip install "openai-whisper>=20250625" "cache-dit==1.3.0" "fa3-fwd==0.0.2" -q 2>&1 | tail -2
pip install "aenum==3.1.16" omegaconf "imageio[ffmpeg]>=2.37.2" "onnxruntime>=1.23.2" \
    "prettytable>=3.8.0" "resampy>=0.4.3" sox -q 2>&1 | tail -2

echo "[3/4] vllm-omni 설치..."
cd /workspace/vllm-omni
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --no-build-isolation --no-deps -q 2>&1 | tail -3

echo "[4/4] 검증..."
python3 -c "
import vllm; print('  vllm:', vllm.__version__)
import vllm_omni; print('  vllm_omni: OK')
from vllm_omni.model_executor.models.minicpm_o import minicpm_o as m
print('  MiniCPM-o 모듈: OK')
"

echo ""
echo "=== 설치 완료 (재시작해도 유지됨) ==="
echo "서버 시작: bash /workspace/scripts/workspace_start.sh"
