# models/easy/app.py
import os
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
app = FastAPI(title="easy-model", version="0.2.0")
class SimplifyReq(BaseModel):
    text: str
    max_new_tokens: int = 120          # 기본값을 짧게
    temperature: float = 0.3
    top_p: float = 0.9
# --- 환경 변수 ---
BASE_MODEL = os.getenv("EASY_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.getenv("EASY_ADAPTER_DIR", "")
# --- dtype 및 디바이스 안전 설정 ---
use_cuda = torch.cuda.is_available()
safe_dtype = torch.float16 if use_cuda else torch.float32
# --- 토크나이저/모델 로드 (메모리 안정화 옵션 포함) ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 생성시 left padding이 안전
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=safe_dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    # trust_remote_code=True,  # 필요 시만 사용
)
# LoRA 어댑터(있으면 붙이고, 없으면 스킵)
if os.path.isdir(ADAPTER_DIR):
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_DIR, device_map="auto")
    except Exception as e:
        print(f"[easy] WARN: Failed to load adapter from {ADAPTER_DIR}: {e}")
model.eval()
def build_prompt(src: str) -> str:
    return (
        "다음 문장을 한국어로 쉽고 간결하게 바꿔줘. 의미는 보존하고, 전문용어는 쉬운 말로.\n"
        "### 원문:\n"
        f"{src}\n"
        "### 답변(쉬운 한국어):\n"
    )
@app.get("/")
def root():
    return {"status": "ok", "model": BASE_MODEL}
@app.get("/health")
def health():
    return {"status": "healthy"}
@app.post("/simplify")
def simplify(req: SimplifyReq):
    try:
        # 과도한 길이 방지 (느림/메모리 이슈 예방)
        max_new = min(max(16, req.max_new_tokens), 160)
        prompt = build_prompt(req.text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 디바이스 일치
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=True,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if full.startswith(prompt):
            full = full[len(prompt):]
        return {"simplified": full.strip()}
    except Exception as e:
        # 내부 에러를 바로 확인할 수 있게 반환 (서버 500)
        return JSONResponse(status_code=500, content={"error": str(e)})
# 직접 실행 지원 (원하면 uvicorn 없이 python app.py 로도 기동)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("models.easy.app:app", host="127.0.0.1", port=8002, reload=False)