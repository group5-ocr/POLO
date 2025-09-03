from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
app = FastAPI(title="easy-model", version="0.1.0")
class SimplifyReq(BaseModel):
    text: str
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
# 환경변수
BASE_MODEL = os.getenv("EASY_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_DIR = os.getenv("EASY_ADAPTER_DIR", "./checkpoints/qwen2.5-7b-easy")
# 모델/토크나이저 로드 (컨테이너 기동 시 1회)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
# 어댑터(LoRA) 붙이기 - 없으면 스킵
if os.path.isdir(ADAPTER_DIR):
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, device_map="auto")
model.eval()
def build_prompt(src: str) -> str:
    return (
        "다음 문장을 한국어로 쉽고 간결하게 바꿔줘. 의미는 보존하고, 전문용어는 쉬운 말로.\n"
        "### 원문:\n"
        f"{src}\n"
        "### 답변(쉬운 한국어):\n"
    )
@app.post("/simplify")
def simplify(req: SimplifyReq):
    prompt = build_prompt(req.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 프롬프트 부분 제거
    if full.startswith(prompt):
        full = full[len(prompt):]
    return {"simplified": full.strip()}