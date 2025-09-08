"""
POLO Easy Model - 논문을 쉽게 풀어 설명하는 LLM 서비스
"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from googletrans import Translator
from dotenv import load_dotenv

# --- 환경 변수 ---
BASE_MODEL = os.getenv("EASY_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

# 기본 어댑터 경로: fine-tuning 결과물(checkpoint-600)을 참조
_DEFAULT_ADAPTER_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "fine-tuning", "outputs", "llama32-3b-qlora", "checkpoint-600",
    )
)
ADAPTER_DIR = os.getenv("EASY_ADAPTER_DIR", _DEFAULT_ADAPTER_DIR)

# polo-system 루트의 .env 로드 (모델을 easy 디렉토리에서 실행해도 인식되도록)
_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(_ENV_PATH)

# Hugging Face 토큰 (가드 리포 접근용)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

app = FastAPI(title="POLO Easy Model", version="1.0.0")

# 전역 변수
model = None
tokenizer = None
translator = Translator()

class TextRequest(BaseModel):
    text: str
    translate: Optional[bool] = False

class TextResponse(BaseModel):
    simplified_text: str
    translated_text: Optional[str] = None

def load_model():
    """모델과 토크나이저 로드"""
    global model, tokenizer
    
    print(f"🔄 모델 로딩 중: {BASE_MODEL}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드: 로컬 서빙은 bitsandbytes 미사용 → bfloat16 고정(GPU), CPU는 float32
    use_cuda = torch.cuda.is_available()
    safe_dtype = torch.bfloat16 if use_cuda else torch.float32
    
    # accelerate/meta 텐서 경로를 피하기 위해 device_map/low_cpu_mem_usage 비활성화
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=safe_dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        token=HF_TOKEN,
    )
    if use_cuda:
        base.to("cuda")
    
    # 어댑터가 있으면 로드 (QLoRA 가중치)
    if ADAPTER_DIR and os.path.exists(ADAPTER_DIR):
        print(f"🔄 어댑터 로딩 중: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
        if use_cuda:
            model.to("cuda")
    else:
        print("⚠️ 어댑터 디렉토리를 찾지 못했습니다. 순수 베이스 모델로 동작합니다.")
    
    model.eval()
    print("✅ 모델 로딩 완료!")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_model()

@app.get("/")
async def root():
    return {"message": "POLO Easy Model API", "model": BASE_MODEL}

@app.post("/simplify", response_model=TextResponse)
async def simplify_text(request: TextRequest):
    """텍스트를 쉽게 풀어 설명"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
        
        # 프롬프트 구성
        prompt = f"""다음 논문 내용을 일반인이 이해하기 쉽게 풀어서 설명해주세요:

{request.text}

쉬운 설명:"""
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        simplified_text = generated_text[len(prompt):].strip()
        
        # 번역 (요청된 경우)
        translated_text = None
        if request.translate:
            try:
                translation = translator.translate(simplified_text, dest='ko')
                translated_text = translation.text
            except Exception as e:
                print(f"번역 오류: {e}")
        
        return TextResponse(
            simplified_text=simplified_text,
            translated_text=translated_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
