# Plan: KO Fine-tuning Pipeline (vllm-omni#1 + #3)

**Branch:** feat/ko-finetune  
**E2E 완료 기준:** 한국어 음성 입력 → (파인튜닝된 모델 경유) → 한국어 음성 출력

---

## 핵심 결정사항

| 항목 | 선택 | 이유 |
|------|------|------|
| CosyVoice2 학습 | FunAudioLLM/CosyVoice 공식 코드 활용 | minicpmo-utils는 Parquet+DeepSpeed 필수 — PoC 과대 |
| 데이터 형식 | CosyVoice list 형식 (TSV: id\tpath\ttext) | 공식 레포 연동이 편함 — 문서화 완비 |
| 학습 단위 | flow.pt 만 파인튜닝 | hift.pt(보코더)는 가중치 유지 |
| 통합 | flow.pt 교체 | 코드 수정 없이 assets/token2wav/ 교체만 |
| LoRA | peft + Qwen3 직접 | vllm-omni --lora-modules 옵션 활용 |

---

## Phase 1: 환경 설정 (tools/ko_finetune/setup.sh)

### 1-A. CosyVoice 공식 레포 클론
```bash
git clone https://github.com/FunAudioLLM/CosyVoice tools/ko_finetune/CosyVoice
```

### 1-B. 의존성 설치
```bash
distrobox enter vllm-dev -- bash -c '
source /home/luke/.venvs/vllm-omni/bin/activate
pip install g2pk g2pkk   # 한국어 G2P
pip install peft         # Qwen3 LoRA
'
```

**검증:** `python -c "import g2pk, peft; print('OK')"`

---

## Phase 2: KsponSpeech 2h 전처리 (tools/ko_finetune/preprocess_kspon.py)

### 입력
- `/var/home/luke/data/aihub/10.한국어음성/KsponSpeech_01.zip`
- 추출 대상: 앞 1,240개 (약 2h 분)

### 처리 단계
```
1. zip에서 선택적 추출 (1,240개 PCM + TXT)
2. PCM → WAV (16kHz, 16bit, mono) — soundfile
3. TXT cp949 디코딩 + 특수토큰 제거 (b/, n/, /, +)
4. 오디오 품질 필터링: 2~15초, SNR 기반 정렬
5. g2pk G2P: 한국어 텍스트 → 음소열
6. CosyVoice list 형식 변환
7. 80/20 분할: train.list + dev.list
```

### 출력 형식 (CosyVoice list 형식)
```
# data/ko_2h/train.list
KsponSpeech_000001	data/ko_2h/wavs/KsponSpeech_000001.wav	안녕하세요
```

**검증:** `wc -l data/ko_2h/train.list` — 필터링 후 유효 샘플 수 확인 (예상 1,000~1,240개; 2~15초 범위 이탈 발화만 제외)

---

## Phase 3: CosyVoice2 flow 파인튜닝 (tools/ko_finetune/train_cosyvoice2_ko.sh)

> **실행 위치:** repo 루트에서 실행 (`data/ko_2h/` 등 상대 경로 기준)

### 학습 대상
- `assets/token2wav/flow.pt` 의 flow-matching 모델
- hift.pt 제외 (보코더 가중치 고정)

### 스크립트 구조
```bash
# tools/ko_finetune/train_cosyvoice2_ko.sh
SNAPSHOT=~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5\
/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc
OUTPUT_DIR=tools/ko_finetune/output/ko_flow

torchrun --nproc_per_node=2 \
  tools/ko_finetune/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model flow \
  --config tools/ko_finetune/config/cosyvoice2_ko.yaml \
  --checkpoint $SNAPSHOT/assets/token2wav/flow.pt \
  --train_data data/ko_2h/train.list \
  --cv_data data/ko_2h/dev.list \
  --model_dir $OUTPUT_DIR
# dev.list: Phase 2에서 train 80%/dev 20% 스플릿으로 생성
```

### 학습 설정 (config/cosyvoice2_ko.yaml)
```yaml
train_conf:
  max_epoch: 20
  batch_size: 8  # global batch (2-GPU DDP: per-GPU 4)
  dtype: bf16
  grad_clip: 1.0
  accum_grad: 1
```

### 예상 시간 (2h 데이터, 20 epoch, RTX 3090 x2)
- 유효 샘플 1,000~1,240개 / global batch 8 = 125~155 steps/epoch
- 20 epoch = 2,500~3,100 steps @ 0.3s/step ≈ **약 12~15분**

---

## Phase 4: 가중치 내보내기 + 통합 (tools/ko_finetune/export_flow.py)

```python
# tools/ko_finetune/export_flow.py
import torch, shutil, os

SNAPSHOT = os.path.expanduser(
    '~/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5'
    '/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc'
)
# CosyVoice train.py는 최종 체크포인트를 {epoch}.pt 형식으로 저장
# 실행 전: ls tools/ko_finetune/output/ko_flow/ 으로 파일명 확인 후 수동 지정
TRAIN_OUT   = 'tools/ko_finetune/output/ko_flow/20.pt'         # max_epoch=20 최종 체크포인트
MODEL_ASSET = os.path.join(SNAPSHOT, 'assets/token2wav/flow.pt')  # 덮어쓸 대상

# 원본 백업
shutil.copy(MODEL_ASSET, MODEL_ASSET + '.backup')

# 학습 결과 가중치 로드 후 덮어쓰기
ckpt = torch.load(TRAIN_OUT, map_location='cpu')
torch.save(ckpt, MODEL_ASSET)
print(f'Done: {TRAIN_OUT} -> {MODEL_ASSET}')
```

---

## Phase 5: Qwen3 Thinker LoRA (tools/ko_finetune/train_qwen3_lora.py)

### 데이터 준비
> **실행 위치:** repo 루트에서 실행 (Phase 3과 동일, `data/` 경로가 정리됨)
```bash
# data/naia_persona.jsonl
# 300개 대화: Naia 페르소나 + Nextain 제품 + 한국어 자연스러운 대화
```

### LoRA 학습
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

# AutoModelForCausalLM로 로드 + target_modules 명시 → Thinker LLM 레이어만 LoRA 적용
# (audio/vision encoder는 q_proj 등 해당 이름 없음 — 자동 제외)
model = AutoModelForCausalLM.from_pretrained(
    'openbmb/MiniCPM-o-4_5',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',  # LoRA는 300개 데이터 + 소수 epoch으로 단일 GPU 충분; Phase 3과 동시 실행 안 함
)
config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=['q_proj','k_proj','v_proj','o_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, config)
# ... SFTTrainer 학습 루프
model.save_pretrained('tools/ko_finetune/output/naia_lora')
```

**예상 시간:** ~4시간 (300개 대화, 5 epoch)

---

## Phase 6: E2E 준비 + 테스트

### Phase 6-A: KO 서버 스크립트 작성 (scripts/serve_ko.sh)

> **실행 위치:** repo 루트에서 실행 (`tools/ko_finetune/output/naia_lora` 상대 경로 기준)

```bash
# scripts/serve_ko.sh
# KO flow.pt + Naia LoRA 적용 서버 기동
export NCCL_P2P_DISABLE=1

distrobox enter vllm-dev -- bash -c '
source /home/luke/.venvs/vllm-omni/bin/activate
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --enable-lora \
  --lora-modules naia=tools/ko_finetune/output/naia_lora \
  --trust-remote-code --port 8000
'
```

---

### Phase 6-B: E2E 테스트 실행 (scripts/ko_e2e_test.sh)

```bash
# 한국어 음성 입력파일 준비
TEST_WAV="data/ko_test.wav"  # 한국어 마이크 녹음 또는 KsponSpeech eval 샘플 복사

# 1. vllm-omni 서버 기동 (KO flow.pt + Naia LoRA)
bash scripts/serve_ko.sh &
sleep 30  # 서버 준비 대기

# 2. 음성 입력 → 응답
# scripts/test_omni_duplex.py 는 기존 repo에 이미 있는 스크립트
python scripts/test_omni_duplex.py \
  --input $TEST_WAV \
  --output /tmp/ko_response.wav

# 3. 예상 결과: 한국어로 답하는 Naia 음성 출력

# 4. 서버 종료
pkill -f "vllm serve" 2>/dev/null  # distrobox 프로세스는 호스트에서 가시적 — pkill 작동
```

**E2E 통과 기준:**
- [ ] 한국어 마이크 입력 → 한국어 텍스트 응답 (Whisper+Qwen3 정상)
- [ ] 한국어 텍스트 → 한국어 음성 파형 출력 (가비지 아님)
- [ ] Naia 페르소나 응답 확인 (이름 물으면 Naia라고 답)

---

## 파일 구조 (feat/ko-finetune)

```
tools/ko_finetune/
├── setup.sh                   # 의존성 + CosyVoice 클론
├── preprocess_kspon.py        # KsponSpeech 2h 전처리
├── train_cosyvoice2_ko.sh     # CosyVoice2 flow 학습
├── export_flow.py             # 체크포인트 → assets/ 교체
├── train_qwen3_lora.py        # Qwen3 LoRA 학습
├── config/
│   └── cosyvoice2_ko.yaml     # 학습 설정
└── output/                    # 학습 결과 (gitignore)
    ├── ko_flow/               # CosyVoice2 파인튜닝 결과
    └── naia_lora/             # Qwen3 LoRA 어답터
data/                              # repo 루트 기준 (scripts/와 동일 레벨)
├── ko_test.wav                # E2E 테스트용 한국어 음성 (다운로드 또는 녹음)
├── ko_2h/                     # KsponSpeech 2h 서브셋 (preprocess 출력)
│   ├── train.list
│   ├── dev.list
│   └── wavs/
└── naia_persona.jsonl         # Qwen3 LoRA 학습 데이터
scripts/
├── serve_ko.sh                # KO 모델 서버 기동
├── ko_e2e_test.sh             # E2E 테스트 (음성 입력 → 음성 출력)
# 기존 파일 (새로 생성 불필요):
# scripts/test_omni_duplex.py  # repo에 이미 존재하는 E2E 클라이언트
```

---

## 리스크 / 미확인 사항

| 위험 | 완화 방법 |
|------|----------|
| CosyVoice list 형식이 flow.pt 학습과 호환 안 될 수 있음 | Phase 3 시작 전 CosyVoice 공식 repo README 확인 후 진행 |
| AutoModelForCausalLM 로드 시 Thinker 외 레이어 LoRA 받을 수 있음 | target_modules 구체적 명시로 실제 적용 레이어 로그 확인 (Phase 5 실행 시) |
| GPU 메모리: CosyVoice2 학습 + LoRA 학습 동시 실행 시 OOM | 순차 학습 (CosyVoice2 먼저, 이후 LoRA) |

---

_작성일: 2026-04-07_
