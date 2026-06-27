# =========================
# 라이브러리
# =========================

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    PeftModel,
)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 설정
# =========================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

LORA_PATH = "./outputs"


# =========================
# 토크나이저 로드
# =========================

print("토크나이저 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# =========================
# 원본 모델 로드
# =========================

print("원본 모델 로드 중...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)


# =========================
# LoRA 적용
# =========================

print("LoRA 적용 중...")

PeftModel.from_pretrained(BASE_MODEL, LORA_PATH)

# =========================
# 추론 루프
# =========================

print("왜AI 시작")

while True:

    user_input = input("너: ")

    if user_input.lower() == "exit":
        break

    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("왜AI:", answer)
