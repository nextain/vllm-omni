#!/usr/bin/env python3
"""Phase 5: Qwen3 Thinker LoRA 학습 (Naia 페르소나 + Nextain 제품 지식)

실행 위치: repo 루트
실행: python3 tools/ko_finetune/train_qwen3_lora.py
"""
import os, sys
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from datasets import Dataset
import torch, json

DATA_PATH = 'data/naia_persona.jsonl'
OUTPUT_DIR = 'tools/ko_finetune/output/naia_lora'
MODEL_ID = 'openbmb/MiniCPM-o-4_5'

# 실행 위치 확인: repo 루트에서 실행해야 데이터 경로가 잘링
if not os.path.exists(DATA_PATH):
    sys.exit(f"❌ {DATA_PATH} 없음. repo 루트(vllm-omni/)에서 실행할 것")

def load_conversations():
    conversations = []
    with open(DATA_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    return conversations

def main():
    print(f"데이터 로드: {DATA_PATH}")
    raw_data = load_conversations()
    print(f"  대화 수: {len(raw_data)}개")

    print(f"모델 로드: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
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

    # tokenizer.apply_chat_template() 사용 - 수동 템플릿 대신 모델 등록 테일릿 활용
    # MiniCPM-o-4.5의 tokenizer_config.json chat_template이 정확한 포맷으로 포맷팅
    texts = [
        tokenizer.apply_chat_template(
            conv['messages'],
            tokenize=False,
            add_generation_prompt=False,
        )
        for conv in raw_data
    ]

    # 데이터셋 포맷 - 개별 토크나이즈 (padding=False)
    # DataCollatorForLanguageModeling이 배치마다 패딩 + labels 마스킹(-100) 자동 수행
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding=False,  # collator가 패딩 처리
    )
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
    })

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        data_collator=data_collator,
    )

    print("학습 시작...")
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA 어답터 저장: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
