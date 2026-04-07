#!/usr/bin/env python3
"""Phase 5: Qwen3 Thinker LoRA 학습 (Naia 페르소나 + Nextain 제품 지식)

실행 위치: repo 루트
실행: python3 tools/ko_finetune/train_qwen3_lora.py
"""
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch, json, os

DATA_PATH = 'data/naia_persona.jsonl'
OUTPUT_DIR = 'tools/ko_finetune/output/naia_lora'
MODEL_ID = 'openbmb/MiniCPM-o-4_5'

def load_dataset():
    conversations = []
    with open(DATA_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    return conversations

def format_conversation(conv):
    """conversations 리스트 -> 단일 텍스트"""
    text = ''
    for msg in conv.get('messages', []):
        role = msg['role']
        content = msg['content']
        if role == 'system':
            text += f'<|im_start|>system
{content}<|im_end|>\n'
        elif role == 'user':
            text += f'<|im_start|>user
{content}<|im_end|>\n'
        elif role == 'assistant':
            text += f'<|im_start|>assistant
{content}<|im_end|>\n'
    return text

def main():
    print(f'데이터 로드: {DATA_PATH}')
    raw_data = load_dataset()
    print(f'  대화 수: {len(raw_data)}개')

    print(f'모델 로드: {MODEL_ID}')
    # AutoModelForCausalLM으로 로드 + target_modules 명시 -> Thinker LLM 레이어만 LoRA 적용
    # (audio/vision encoder는 q_proj 등 해당 이름 없음 - 자동 제외)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',  # LoRA는 300개 데이터 + 소수 epoch으로 단일 GPU 충분; Phase 3과 동시 실행 안 함
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 데이터셋 포맷
    texts = [format_conversation(c) for c in raw_data]
    encodings = tokenizer(texts, truncation=True, max_length=1024, padding=True, return_tensors='pt')
    dataset = Dataset.from_dict({'input_ids': encodings['input_ids'], 'labels': encodings['input_ids']})

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy='epoch',
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print('학습 시작...')
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f'✅ LoRA 어답터 저장: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
