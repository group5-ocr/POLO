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

app = FastAPI(title="YOLO Easy LLM Service", version="1.0.0")

# 모델 설정
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CHECKPOINT_PATH = "/app/outputs/yolo-easy-qlora/"
CHECKPOINT_ROOT = "/app/outputs/yolo-easy-qlora" 

# 전역 변수
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    text: str
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    generated_text: str
    input_text: str

def _get_latest_ckpt_dir(root: str) -> str:
    """최신 체크포인트 디렉토리 찾기"""
    p = Path(root)
    if not p.exists():
        logger.warning(f"체크포인트 디렉토리가 없습니다: {root}")
        return root
        
    cks = sorted(
        [d for d in p.glob("checkpoint-*") if d.is_dir()],
        key=lambda d: int(re.findall(r"\d+", d.name)[-1]) if re.findall(r"\d+", d.name) else -1
    )
    return str(cks[-1]) if cks else root

def load_model():
    """YOLO 파인튜닝된 모델 로드"""
    global model, tokenizer
    try:
        # Hugging Face 토큰 확인
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError("Hugging Face token not found in env.")
        login(token=hf_token, add_to_git_credential=True)

        logger.info(f"베이스 모델 로딩: {BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # dtype 설정
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

        # YOLO 체크포인트 로드
        ckpt_dir = _get_latest_ckpt_dir(CHECKPOINT_ROOT)
        logger.info(f"YOLO 파인튜닝된 모델 로딩: {ckpt_dir}")
        
        if os.path.exists(ckpt_dir):
            model = PeftModel.from_pretrained(base_model, ckpt_dir)
        else:
            logger.warning(f"체크포인트를 찾을 수 없습니다: {ckpt_dir}")
            logger.info("베이스 모델을 사용합니다.")
            model = base_model
            
        model.eval()
        logger.info("YOLO 모델 로딩 완료!")
        return True

    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    success = load_model()
    if not success:
        raise RuntimeError("YOLO 모델 로딩에 실패했습니다.")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "model_loaded": model is not None, "service": "yolo-easy"}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """YOLO 논문 텍스트 간소화 생성"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    try:
        # YOLO 논문용 프롬프트 형식
        prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
너는 AI 논문 텍스트를 원문 의미를 유지한 채 한국어로 쉽게 풀어쓰는 전문가다.
규칙:
- 영문 전문용어는 영어 표기 그대로 두고, 바로 뒤에 (짧은 한국어 풀이)를 1회만 붙여라.
- 수식/코드/링크/토큰은 변형 금지.
- 숫자/기호/단위/약어를 바꾸지 마라.
- 3~5 문단, 각 2~4문장으로 자연스럽게 설명.
- 불필요한 반복/군더더기 제거.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
[TEXT]
{request.text}

[REWRITE in Korean]
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # 생성된 텍스트만 추출
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
        "device": str(model.device) if model else None,
        "service": "yolo-easy"
    }

@app.get("/yolo_test")
async def yolo_test():
    """YOLO 모델 테스트"""
    test_text = "Humans glance at an image and instantly know what objects are in the image, where they are, and how they interact."
    
    try:
        response = await generate_text(GenerateRequest(text=test_text))
        return {
            "input": test_text,
            "output": response.generated_text,
            "status": "success"
        }
    except Exception as e:
        return {
            "input": test_text,
            "output": None,
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)
