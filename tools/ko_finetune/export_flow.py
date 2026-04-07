#!/usr/bin/env python3
# Phase 4: 학습 결과 flow.pt → HuggingFace assets/ 교체
# 실행: python3 tools/ko_finetune/export_flow.py
# CosyVoice train.py는 최종 체크포인트를 {{epoch}}.pt 형식으로 저장
# 실행 전 ls tools/ko_finetune/output/ko_flow/ 로 파일명 확인 후 필요 시 수정
import torch, shutil, os, sys

SNAPSHOT = os.path.expanduser(
    '~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5'
    '/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc'
)
TRAIN_OUT   = 'tools/ko_finetune/output/ko_flow/20.pt'
MODEL_ASSET = os.path.join(SNAPSHOT, 'assets/token2wav/flow.pt')

if not os.path.exists(TRAIN_OUT):
    print(f"❌ TRAIN_OUT 없음: {TRAIN_OUT}")
    sys.exit(1)
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

# 학습 결과 가중치 로드 후 덮어쓰기
ckpt = torch.load(TRAIN_OUT, map_location='cpu')
torch.save(ckpt, MODEL_ASSET)
print(f'✅ Done: {TRAIN_OUT} -> {MODEL_ASSET}')
