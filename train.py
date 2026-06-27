import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
)

# =========================
# 설정
# =========================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "./data/dataset.jsonl"
OUTPUT_DIR = "./outputs"

# =========================
# 토크나이저 및 모델 로드
# =========================
print("토크나이저 및 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, # 메모리 절약 및 학습 안정성
    device_map="auto"
)

# =========================
# LoRA 설정
# =========================
print("LoRA 설정 중...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()

# =========================
# 데이터셋 로드 및 안전한 토큰화 함수
# =========================
print("데이터셋 로드 및 전처리...")
dataset = load_dataset("json", data_files=DATASET_PATH)

def tokenize(example):
    messages = example["messages"]
    
    # Assistant 답변을 제외한 프롬프트 생성 (System + User)
    prompt_chat = messages[:-1]
    full_chat = messages

    prompt_str = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)
    full_str = tokenizer.apply_chat_template(full_chat, tokenize=False) + tokenizer.eos_token

    prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_tokens = tokenizer(full_str, add_special_tokens=False)["input_ids"]

    # 정밀 마스킹: 프롬프트 길이만큼 -100으로 채우기
    labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

    # 최대 길이 맞추기 (Truncation & Padding)
    max_length = 256
    if len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]
        labels = labels[:max_length]
    else:
        padding_len = max_length - len(full_tokens)
        full_tokens = full_tokens + [tokenizer.pad_token_id] * padding_len
        labels = labels + [-100] * padding_len

    return {
        "input_ids": full_tokens,
        "attention_mask": [1 if t != tokenizer.pad_token_id else 0 for t in full_tokens],
        "labels": labels
    }

# 💡 안전장치: .map() 적용 시 dataset["train"] 구조가 확실히 잡히도록 명시
tokenized_dataset = dataset["train"].map(tokenize)

# =========================
# 학습 설정 및 시작
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,               # 👈 50줄 데이터셋 특성상 확실한 각인을 위해 10 에포크로 상향
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,    
    learning_rate=1e-4,               
    weight_decay=0.01,
    logging_steps=5,                  # 👈 데이터가 적으므로 더 자주 로그를 보도록 수정
    save_strategy="no",               # 👈 데이터가 적을 때는 에포크마다 저장할 필요 없이 마지막에만 저장하는 게 빠릅니다
    bf16=True,                        
    report_to="none"                  # 불필요한 외부 완드비(Wandb) 연동 경고 방지
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("학습 시작")
trainer.train()

# 저장
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("완료")
