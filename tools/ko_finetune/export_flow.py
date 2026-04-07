#!/usr/bin/env python3
# Phase 4: 학습 결과 flow.pt → HuggingFace assets/ 교체
# 실행: python3 tools/ko_finetune/export_flow.py
# CosyVoice train_utils.py save_model()는 스테이트딕을 평탄 dict으로 저장
# ({**model.state_dict(), 'epoch': N, 'step': N} 형식으로 저장 -> 'model' 키 없음)
import torch, shutil, os, sys, glob, re

# CWD 강제화
if not os.path.isdir('tools/ko_finetune'):
    sys.exit(f'ERROR: run from repo root (current: {os.getcwd()})')

SNAPSHOT = os.path.expanduser(
    '~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5'
    '/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc'
)
MODEL_ASSET = os.path.join(SNAPSHOT, 'assets/token2wav/flow.pt')
# 저장 시 제외할 메타 키 (state_dict에 포함하다 로드 오류 발생)
NON_PARAM_KEYS = {'epoch', 'step'}

# CosyVoice executor.py는 epoch_{N}_whole.pt 형식으로 저장
# 가장 높은 epoch 체크포인트 자동 선택
ckpt_glob = glob.glob('tools/ko_finetune/output/ko_flow/epoch_*_whole.pt')
if not ckpt_glob:
    print(f"❌ 체크포인트 없음: tools/ko_finetune/output/ko_flow/epoch_*_whole.pt")
    print("   train_cosyvoice2_ko.sh 먼저 실행 후 재시도")
    sys.exit(1)
TRAIN_OUT = sorted(
    ckpt_glob,
    key=lambda p: int(re.search(r'epoch_(\d+)_whole', p).group(1))
)[-1]
print(f"ℹ 선택된 체크포인트: {TRAIN_OUT}")

if not os.path.exists(MODEL_ASSET):
    print(f"❌ MODEL_ASSET 없음: {MODEL_ASSET}")
    sys.exit(1)

# 원본 백업
backup = MODEL_ASSET + '.backup'
if not os.path.exists(backup):
    shutil.copy(MODEL_ASSET, backup)
    print(f"✅ 원본 백업: {backup}")
else:
    print(f"ℹ 백업 이미 존재: {backup}")

# 체크포인트 로드
ckpt = torch.load(TRAIN_OUT, map_location='cpu', weights_only=False)  # CosyVoice saves non-tensor metadata
# CosyVoice save_model()은 {**state_dict, 'epoch': N, 'step': N} 평탄 dict으로 저장
# 'epoch'/'step' 두 메타 키를 제거해야 순수 state_dict이 됨
state_dict = {k: v for k, v in ckpt.items() if k not in NON_PARAM_KEYS}
removed = [k for k in ckpt if k in NON_PARAM_KEYS]
print(f"ℹ 체크포인트 정보: epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')}, 제거 키={removed}")
print(f"ℹ state_dict 파라미터 수: {len(state_dict)}")
torch.save(state_dict, MODEL_ASSET)
print(f'✅ Done: {TRAIN_OUT} -> {MODEL_ASSET}')
