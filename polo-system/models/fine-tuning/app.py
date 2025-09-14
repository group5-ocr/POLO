import os
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login 
from pathlib import Path           
import re 

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Easy LLM Service", version="1.0.0")

# 모델 설정
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CHECKPOINT_PATH = "/app/outputs/llama32-3b-qlora/"
CHECKPOINT_ROOT = "/app/outputs/llama32-3b-qlora" 

# 전역 변수
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    text: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    generated_text: str
    input_text: str

def _get_latest_ckpt_dir(root: str) -> str:
    p = Path(root)
    cks = sorted(
        [d for d in p.glob("checkpoint-*") if d.is_dir()],
        key=lambda d: int(re.findall(r"\d+", d.name)[-1]) if re.findall(r"\d+", d.name) else -1
    )
    return str(cks[-1]) if cks else root  # checkpoint-* 없으면 루트 사용

def load_model():
    global model, tokenizer
    try:
        # ★ 1) 비대화식 로그인
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError("Hugging Face token not found in env.")
        login(token=hf_token, add_to_git_credential=True)

        logger.info(f"베이스 모델 로딩: {BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ★ 2) dtype 폴백
        dtype = torch.bfloat16
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            dtype = torch.float16

        logger.info("베이스 모델 로딩 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )

        # ★ 3) 최신 체크포인트 자동 선택
        ckpt_dir = _get_latest_ckpt_dir(CHECKPOINT_ROOT)
        logger.info(f"파인튜닝된 모델 로딩: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, ckpt_dir)
        model.eval()

        logger.info("모델 로딩 완료!")
        return True

    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    success = load_model()
    if not success:
        raise RuntimeError("모델 로딩에 실패했습니다.")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # ★ 4) 토큰 기준으로 생성분만 추출
        full_ids = outputs[0]
        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = full_ids[prompt_len:]
        generated_only = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return GenerateResponse(generated_text=generated_only, input_text=request.text)
    except Exception as e:
        logger.error(f"텍스트 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 생성 실패: {str(e)}")

@app.get("/model_info")
async def model_info():
    """모델 정보"""
    return {
        "base_model": BASE_MODEL_ID,
        "checkpoint_path": CHECKPOINT_PATH,
        "model_loaded": model is not None,
        "device": str(model.device) if model else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
