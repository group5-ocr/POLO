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
import json
import time
import logging
import re

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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
model = None
tokenizer = None
translator = Translator()
gpu_available = False

class TextRequest(BaseModel):
    text: str
    translate: Optional[bool] = False

class TextResponse(BaseModel):
    simplified_text: str
    translated_text: Optional[str] = None

def load_model():
    """모델과 토크나이저 로드"""
    global model, tokenizer, gpu_available
    
    logger.info(f"🔄 모델 로딩 중: {BASE_MODEL}")
    
    # GPU 상태 확인 - 강제로 GPU 사용 시도
    gpu_available = False
    try:
        # CUDA 사용 가능 여부 확인
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU 사용 가능: {device_name}")
            logger.info(f"💾 GPU 메모리: {device_memory:.1f}GB")
            logger.info(f"🎯 GPU 디바이스: cuda:0")
            gpu_available = True
        else:
            # CUDA가 감지되지 않아도 강제로 GPU 사용 시도
            logger.info("🚀 CUDA 감지 실패, GPU 강제 사용 시도...")
            try:
                # 간단한 텐서로 GPU 테스트
                test_tensor = torch.tensor([1.0]).cuda()
                logger.info("✅ GPU 강제 사용 성공!")
                gpu_available = True
            except Exception as e:
                logger.warning(f"⚠️ GPU 강제 사용 실패: {e}")
                gpu_available = False
    except Exception as e:
        logger.warning(f"⚠️ GPU 확인 중 오류: {e}")
        gpu_available = False
    
    if not gpu_available:
        logger.warning("⚠️ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")
    
    # 토크나이저 로드
    logger.info("📝 토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드: 로컬 서빙은 bitsandbytes 미사용 → bfloat16 고정(GPU), CPU는 float32
    safe_dtype = torch.bfloat16 if gpu_available else torch.float32
    logger.info(f"🧠 모델 로딩 중... (dtype: {safe_dtype})")
    
    # GPU 사용 시 device_map 설정
    if gpu_available:
        logger.info("🎯 GPU device_map으로 모델 로딩...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=safe_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="auto",
            token=HF_TOKEN,
        )
        logger.info("✅ GPU에 모델 로딩 완료")
    else:
        logger.info("💻 CPU로 모델 로딩...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=safe_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            token=HF_TOKEN,
        )
        logger.info("✅ CPU에 모델 로딩 완료")
    
    # 어댑터가 있으면 로드 (QLoRA 가중치)
    if ADAPTER_DIR and os.path.exists(ADAPTER_DIR):
        logger.info(f"🔄 어댑터 로딩 중: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
        if gpu_available:
            logger.info("🎯 어댑터를 GPU로 이동...")
            model = model.to("cuda")
        logger.info("✅ 어댑터 로딩 완료")
    else:
        logger.warning("⚠️ 어댑터 디렉토리를 찾지 못했습니다. 순수 베이스 모델로 동작합니다.")
        model = base
    
    model.eval()
    logger.info("✅ 모델 로딩 완료!")

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

@app.post("/generate")
async def generate_json(request: TextRequest):
    """논문 텍스트를 받아 섹션별로 쉽게 재해석한 JSON 생성"""
    start_time = time.time()
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

        logger.info("🚀 JSON 생성 시작")
        logger.info(f"📊 입력 텍스트 길이: {len(request.text)} 문자")
        logger.info(f"🎯 GPU 사용: {gpu_available}")

        # 1) 원문에서 주요 섹션 텍스트 미리 추출하여 'original' 채워두기
        def extract_sections(src: str) -> dict:
            sections = {
                "abstract": "",
                "introduction": "",
                "methods": "",
                "results": "",
                "discussion": "",
                "conclusion": "",
            }
            # 섹션 헤더 패턴 (대/소문자, 공백 포함, 콜론 허용)
            headers = [
                ("abstract", r"^\s*abstract\b[:\-]?"),
                ("introduction", r"^\s*introduction\b[:\-]?"),
                ("methods", r"^\s*methods?\b[:\-]?|^\s*materials?\s+and\s+methods\b[:\-]?"),
                ("results", r"^\s*results?\b[:\-]?"),
                ("discussion", r"^\s*discussion\b[:\-]?"),
                ("conclusion", r"^\s*conclusion[s]?\b[:\-]?|^\s*concluding\s+remarks\b[:\-]?")
            ]
            lines = src.splitlines()
            indices = []
            for idx, line in enumerate(lines):
                for key, pat in headers:
                    if re.match(pat, line.strip(), flags=re.IGNORECASE):
                        indices.append((idx, key))
                        break
            indices.sort()
            for i, (start_idx, key) in enumerate(indices):
                end_idx = indices[i+1][0] if i+1 < len(indices) else len(lines)
                chunk = "\n".join(lines[start_idx+1:end_idx]).strip()
                sections[key] = chunk[:2000]  # 원문은 길이 제한
            return sections

        extracted = extract_sections(request.text)

        json_schema = {
            "title": "",  # 논문 제목(원문 추출 불가 시 요약 기반 생성)
            "authors": [],  # 저자 목록(알 수 없으면 빈 배열)
            "abstract": {"original": extracted["abstract"], "easy": ""},
            "introduction": {"original": extracted["introduction"], "easy": ""},
            "methods": {"original": extracted["methods"], "easy": ""},
            "results": {"original": extracted["results"], "easy": ""},
            "discussion": {"original": extracted["discussion"], "easy": ""},
            "conclusion": {"original": extracted["conclusion"], "easy": ""},
            "keywords": [],
            "figures_tables": [],  # {label, caption, easy}
            "references": [],
            "contributions": [],  # 핵심 기여 포인트를 쉬운 문장으로
            "limitations": [],
            "glossary": [],  # 중요 용어 {term, definition}
            "plain_summary": ""  # 전체를 일반인도 이해할 수 있게 5-7문장 요약
        }

        instruction = (
            "너는 과학 커뮤니케이터다. 아래 스키마의 키와 구조를 절대 변경하지 말고, 값만 채워라. "
            "출력은 오직 JSON 하나만 허용된다(마크다운, 설명, 코드블록 금지). "
            "각 섹션의 'easy'에는 중학생도 이해할 수 있게 4-6문장으로 풀어쓰고, 과장/추측 금지. "
            "모를 정보는 빈 문자열이나 빈 배열로 둔다. 'figures_tables'는 있으면 {label, caption, easy}로 목록화. "
            "'plain_summary'는 전체를 일반어로 5-7문장 요약." 
        )

        schema_str = json.dumps(json_schema, ensure_ascii=False, indent=2)

        prompt = f"""{instruction}

다음은 출력해야 할 JSON 스키마(미리 일부 original이 채워짐)이다. 키/구조는 그대로 두고 값만 채워라. 반드시 순수 JSON만 출력:
{schema_str}

참고용 전체 원문 텍스트(섹션 추출이 부정확할 수 있으므로 보조로만 사용):\n\n"""

        logger.info("📝 토크나이징 시작...")
        # 토크나이징
        inputs = tokenizer(
            prompt + request.text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        if gpu_available:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logger.info("🎯 입력을 GPU로 이동 완료")

        logger.info("🧠 모델 추론 시작...")
        inference_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1600,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        inference_time = time.time() - inference_start
        logger.info(f"⚡ 추론 완료: {inference_time:.2f}초")

        logger.info("📄 디코딩 시작...")
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw = generated[len(prompt):].strip()

        # JSON 강제 파싱: 첫 { 부터 마지막 } 까지 자르고 파싱 시도
        def coerce_json(text: str):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]
            return json.loads(text)

        def is_meaningful(d: dict) -> bool:
            try:
                # 섹션 easy 중 하나라도 내용이 있으면 의미 있다고 간주
                sections = ["abstract","introduction","methods","results","discussion","conclusion"]
                return any(len((d.get(s,{}) or {}).get("easy","")) > 10 for s in sections)
            except Exception:
                return False

        try:
            data = coerce_json(raw)
            if not is_meaningful(data):
                raise ValueError("empty_json")
            logger.info("✅ JSON 파싱/검증 성공")
        except Exception as e:
            logger.warning(f"⚠️ 1차 파싱 실패: {e}. 재생성 시도")
            # 2차 시도: 더 엄격한 지시문과 샘플링/길이 조정
            strict_instruction = (
                "위 스키마를 기준으로 값을 채워 '유효한 JSON'만 출력하라. 반드시 '{' 로 시작하고 '}' 로 끝내라. "
                "코드블록, 주석, 설명, 키 변경 일절 금지. 응답은 순수 JSON 문자열 하나만 허용."
            )
            strict_prompt = f"{strict_instruction}\n\n스키마:\n{schema_str}\n\n참고 원문(요약/재해석에만 사용):\n\n"
            inputs2 = tokenizer(
                strict_prompt + request.text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            if gpu_available:
                inputs2 = {k: v.cuda() for k, v in inputs2.items()}
            with torch.no_grad():
                outputs2 = model.generate(
                    **inputs2,
                    max_new_tokens=1400,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
            raw2 = gen2[len(strict_prompt):].strip()
            try:
                data = coerce_json(raw2)
                if not is_meaningful(data):
                    raise ValueError("empty_json_retry")
                logger.info("✅ 2차 JSON 파싱/검증 성공")
            except Exception as e2:
                logger.warning(f"⚠️ 2차 파싱 실패: {e2}. 기본 스키마 반환")
                data = json_schema
                data["plain_summary"] = ""

        total_time = time.time() - start_time
        logger.info(f"🎉 전체 처리 완료: {total_time:.2f}초")
        
        # 메타데이터 추가
        data["processing_info"] = {
            "gpu_used": gpu_available,
            "inference_time": inference_time,
            "total_time": total_time,
            "input_length": len(request.text),
            "output_length": len(str(data))
        }

        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ JSON 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JSON 생성 중 오류: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": gpu_available,
        "gpu_device": torch.cuda.get_device_name(0) if gpu_available else None,
        "model_name": BASE_MODEL
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
