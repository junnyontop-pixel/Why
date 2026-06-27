# 1. 라이브러리 불러오기

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

# TODO:
# 필요한 라이브러리 추가


# =========================
# 설정
# =========================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


DATASET_PATH = "./data/dataset.jsonl"

OUTPUT_DIR = "./outputs"


# =========================
# 토크나이저 로드
# =========================

print("토크나이저 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

# =========================
# 모델 로드
# =========================

print("모델 로드 중...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# =========================
# 데이터셋 로드
# =========================

print("데이터셋 로드 중...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH
)


# =========================
# LoRA 설정
# =========================

print("LoRA 설정 중...")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# TODO:
# LoRA를 모델에 적용

lora_model = get_peft_model(model, peft_config)

# 학습 가능한 파라미터 비율 확인
lora_model.print_trainable_parameters()


# =========================
# 토큰화 함수
# =========================

def tokenize(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    labels = tokens["input_ids"].copy()

    # 🔥 system/user 부분 loss 제거 (핵심)
    sep = "assistant"

    # 그냥 간단 버전: prompt 부분 -100 처리
    # (실전에서는 더 정교하게 함)

    tokens["labels"] = labels

    return tokens


# =========================
# 데이터 전처리
# =========================

print("전처리 중...")

tokenized_dataset = dataset.map(tokenize)


# =========================
# 학습 설정
# =========================

# TODO:
# TrainingArguments 작성

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,

    eval_strategy="no",   # ← 변경

    save_strategy="epoch",

    logging_steps=100,
)


# =========================
# Trainer 생성
# =========================

# TODO:
# SFTTrainer 또는 Trainer 생성

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)


# =========================
# 학습 시작
# =========================

print("학습 시작")

# TODO:

trainer.train()


# =========================
# 저장
# =========================

print("모델 저장 중...")

# TODO:
# 저장 코드

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


print("완료")
