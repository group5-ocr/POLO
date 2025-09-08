import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Easy LLM Service", version="1.0.0")

# 모델 설정
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CHECKPOINT_PATH = "/app/outputs/llama32-3b-qlora/checkpoint-600"

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

def load_model():
    """파인튜닝된 모델 로드"""
    global model, tokenizer
    
    try:
        logger.info(f"베이스 모델 로딩: {BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("베이스 모델 로딩 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        logger.info(f"파인튜닝된 모델 로딩: {CHECKPOINT_PATH}")
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        
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
    """텍스트 생성"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    try:
        # 입력 텍스트 토크나이징
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # 텍스트 생성
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
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력 텍스트 제거 (생성된 부분만 반환)
        input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        generated_only = generated_text[input_length:].strip()
        
        return GenerateResponse(
            generated_text=generated_only,
            input_text=request.text
        )
        
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
