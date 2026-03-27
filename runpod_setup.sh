#!/bin/bash
# RunPod setup script for MiniCPM-o 4.5 E2E testing
# Based on official quickstart: docs/getting_started/quickstart.md
set -e

echo "=== 1. Install uv ==="
pip install uv

echo "=== 2. Create venv with uv ==="
uv venv /workspace/venv --python 3.11 --seed
source /workspace/venv/bin/activate

echo "=== 3. Install vllm 0.17.0 (base) ==="
uv pip install vllm==0.17.0 --torch-backend=auto

echo "=== 4. Install vllm-omni on top ==="
cd /workspace/vllm-omni
uv pip install -e ".[all]"

echo "=== 5. Install ffmpeg (for audio) ==="
apt-get update && apt-get install -y ffmpeg espeak-ng sox libsox-fmt-all

echo "=== 6. Verify ==="
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_omni; print('vllm_omni: OK')"
python -c "from vllm.transformers_utils.config import get_config; c = get_config('/workspace/.cache_hf/models--openbmb--MiniCPM-o-4_5/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc', trust_remote_code=True); print(f'model_type: {c.model_type}')"

echo "=== Setup complete ==="
echo "Start server with: source /workspace/venv/bin/activate && bash /workspace/start_server.sh"
