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
    # Qwen 계열은 아래 target_modules를 지정해줘야 싸가지없는 말투(스타일)가 전반적으로 잘 학습됩니다.
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
    
    # 1. Assistant 답변을 제외한 프롬프트 생성 (System + User)
    prompt_chat = messages[:-1]
    # 2. 전체 대화 생성 (System + User + Assistant)
    full_chat = messages

    prompt_str = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)
    full_str = tokenizer.apply_chat_template(full_chat, tokenize=False) + tokenizer.eos_token

    # 각각 토큰화
    prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_tokens = tokenizer(full_str, add_special_tokens=False)["input_ids"]

    # 🔥 정밀 마스킹: 프롬프트 길이만큼 -100으로 채우기
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

tokenized_dataset = dataset["train"].map(tokenize)

# =========================
# 학습 설정 및 시작
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,               # 말투 변화를 위해 에포크를 조금 더 늘리는 것을 추천합니다.
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,    # 배치가 작을 때 안정성 확보
    learning_rate=1e-4,               # LoRA는 일반 파인튜닝보다 lr을 조금 더 높게(1e-4 ~ 2e-4) 잡는 게 효과적입니다.
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True                         # GPU가 지원한다면 bf16 추천
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
