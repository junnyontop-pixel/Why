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
).to(device)


# =========================
# LoRA 적용
# =========================

print("LoRA 적용 중...")

model = PeftModel.from_pretrained(base_model, LORA_PATH).to(device)

# =========================
# 추론 루프
# =========================

print("왜AI 시작")

while True:
    user_input = input("너: ")
    if user_input.lower() == "exit":
        break

    # 💡 중요: 학습 데이터셋에 넣었던 것과 완전히 같은 형식의 System 프롬프트여야 합니다.
    messages = [
        {"role": "system", "content": "너는 짧고 무심하고 싸가지없게 대답한다."},
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,          # 👈 다양성 부여
        temperature=0.8,         # 👈 너무 높으면 헛소리하니 0.7~0.8 추천
        top_p=0.9,
        repetition_penalty=1.2,  # 👈 했던 말 반복하거나 상투적인 대답 방지
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    print("왜AI:", answer)

