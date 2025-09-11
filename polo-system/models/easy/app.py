"""
POLO Easy Model - Grounded JSON Generator (GPU-forced, stable)
- GPU 필수: CUDA 없으면 기동 중단
- 어텐션 백엔드 자동 선택: flash_attn > sdpa(가능 시) > eager 폴백
- 섹션별 original만 컨텍스트로 제공하여 'easy'를 재서술(추측 금지)
- Greedy 디코딩, 낮은 토큰 수(기본 600)로 속도/안정성 확보
- JSON 강제 파싱 + 재시도, 실패 시 빈 값 유지
"""
import os
import uvicorn
import time
import json
import re
import logging
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# -------------------- 환경/로깅 --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polo.easy")

# polo-system 루트의 .env 로드
_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(_ENV_PATH)

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
BASE_MODEL = os.getenv("EASY_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

_DEFAULT_ADAPTER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fine-tuning", "outputs", "llama32-3b-qlora", "checkpoint-600")
)
ADAPTER_DIR = os.getenv("EASY_ADAPTER_DIR", _DEFAULT_ADAPTER_DIR)

MAX_NEW_TOKENS = int(os.getenv("EASY_MAX_NEW_TOKENS", "4000"))  # 가중치 4000으로 설정

# -------------------- FastAPI --------------------
app = FastAPI(title="POLO Easy Model", version="1.2.0")

# -------------------- 전역 상태 --------------------
model = None
tokenizer = None
device = "cuda"
gpu_available = False
safe_dtype = torch.float16

# -------------------- 모델/유틸 --------------------
def _pick_attn_impl() -> str:
    # 1) flash_attn 가능하면 최우선
    try:
        import flash_attn  # noqa: F401
        logger.info("✅ flash_attn 사용: flash_attention_2")
        return "flash_attention_2"
    except Exception:
        pass
    # 2) sdpa 가능 여부(없으면 transformers가 ImportError 던질 수 있음)
    try:
        from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
        logger.info("ℹ️ sdpa 사용 가능")
        return "sdpa"
    except Exception:
        logger.info("ℹ️ sdpa 불가 → eager로 진행")
        return "eager"

def _coerce_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

def _is_meaningful(d: dict) -> bool:
    try:
        sections = ["abstract","introduction","methods","results","discussion","conclusion"]
        return any(len((d.get(s, {}) or {}).get("easy", "")) > 10 for s in sections)
    except Exception:
        return False

def _extract_sections(src: str) -> dict:
    sections = {
        "abstract": "",
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
    }
    headers = [
        ("abstract", r"^\s*abstract\b[:\-]?"),
        ("introduction", r"^\s*introduction\b[:\-]?"),
        ("methods", r"^\s*methods?\b[:\-]?|^\s*materials?\s+and\s+methods\b[:\-]?"),
        ("results", r"^\s*results?\b[:\-]?"),
        ("discussion", r"^\s*discussion\b[:\-]?"),
        ("conclusion", r"^\s*conclusion[s]?\b[:\-]?|^\s*concluding\s+remarks\b[:\-]?"),
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
        sections[key] = chunk[:2000]  # 섹션 original은 길이 제한
    return sections

# -------------------- 스키마 --------------------
GROUND_SCHEMA = {
    "title": "",
    "authors": [],
    "abstract": {"original": "", "easy": ""},
    "introduction": {"original": "", "easy": ""},
    "methods": {"original": "", "easy": ""},
    "results": {"original": "", "easy": ""},
    "discussion": {"original": "", "easy": ""},
    "conclusion": {"original": "", "easy": ""},
    "keywords": [],
    "figures_tables": [],
    "references": [],
    "contributions": [],
    "limitations": [],
    "glossary": [],
    "plain_summary": "",
}

# -------------------- 요청/응답 모델 --------------------
class TextRequest(BaseModel):
    text: str
    translate: Optional[bool] = False
    model_config = ConfigDict(extra="allow")  # 추가 필드 허용(422 방지)

class TextResponse(BaseModel):
    simplified_text: str
    translated_text: Optional[str] = None

# -------------------- 모델 로드 --------------------
def load_model():
    global model, tokenizer, gpu_available, device, safe_dtype

    logger.info(f"🔄 모델 로딩 시작: {BASE_MODEL}")

    # ✅ GPU 강제
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA(GPU)를 사용할 수 없습니다. GPU 환경에서만 실행하도록 강제되었습니다.")
    gpu_available = True
    device = "cuda"
    safe_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = _pick_attn_impl()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1차: 선택된 어텐션으로 로드, 실패 시 eager로 폴백
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=safe_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
            device_map=None,  # 명시적 .to(device) 사용
        )
    except Exception as e:
        logger.warning(f"attn='{attn_impl}' 로딩 실패({e}) → eager로 폴백")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=safe_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
            device_map=None,
        )

    # LoRA 어댑터 (실패해도 베이스 모델로 계속 진행)
    m = base  # 기본값은 베이스 모델
    
    if ADAPTER_DIR and os.path.exists(ADAPTER_DIR):
        logger.info(f"🔄 어댑터 로딩 시도: {ADAPTER_DIR}")
        try:
            # Windows 경로 문제 해결을 위해 절대 경로 사용
            adapter_path = os.path.abspath(ADAPTER_DIR)
            # Hugging Face가 로컬 경로를 인식하도록 처리
            m = PeftModel.from_pretrained(base, adapter_path, is_trainable=False, local_files_only=True)
            logger.info("✅ 어댑터 로딩 성공")
        except Exception as e:
            logger.error(f"❌ 어댑터 로딩 실패: {e}")
            logger.warning("⚠️ 베이스 모델로 계속 진행")
            m = base
    else:
        logger.warning("⚠️ 어댑터 경로 없음 → 베이스 모델로 동작")

    m.eval()
    m = m.to(safe_dtype).to(device)

    p = next(m.parameters())
    logger.info(f"🧠 MODEL DEVICE: {p.device}, DTYPE: {p.dtype}")

    model = m
    logger.info("✅ 모델 로딩 완료")

# -------------------- 스타트업 --------------------
@app.on_event("startup")
async def startup_event():
    load_model()

# -------------------- 엔드포인트 --------------------
@app.get("/")
async def root():
    return {"message": "POLO Easy Model API", "model": BASE_MODEL}

@app.get("/health")
async def health():
    return {
        "status": "healthy" if (model is not None and tokenizer is not None and gpu_available) else "unavailable",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": gpu_available,
        "gpu_device": torch.cuda.get_device_name(0) if gpu_available else None,
        "model_name": BASE_MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

@app.get("/healthz")
async def healthz():
    return await health()

@app.post("/simplify", response_model=TextResponse)
async def simplify_text(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

    prompt = (
        "아래 텍스트를 일반인이 이해하기 쉽게 풀어 써라. "
        "추가 가정이나 외부 지식 사용 금지. 제공된 문장만 재서술하며, 모르면 생략:\n\n"
        f"{request.text}\n\n쉬운 설명:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4000,          # ✅ 가중치 4000으로 설정
            do_sample=False,              # ✅ 그리디(추측 억제)
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    simplified_text = generated[len(prompt):].strip()
    return TextResponse(simplified_text=simplified_text, translated_text=None)

@app.post("/generate")
async def generate_json(request: TextRequest):
    """
    논문 텍스트를 받아 섹션별 original만 바탕으로 'easy'를 재서술하여 JSON을 생성.
    - 외부지식/추측 금지
    - original이 비어 있으면 easy도 빈 문자열 유지
    """
    start_time = time.time()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

    logger.info("🚀 JSON 생성 시작")
    logger.info(f"📊 입력 길이: {len(request.text)}")

    # 1) 섹션 추출
    extracted = _extract_sections(request.text)

    # 2) 스키마 준비 + original 주입
    data_schema = json.loads(json.dumps(GROUND_SCHEMA))  # deepcopy
    for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
        data_schema[k]["original"] = extracted.get(k, "")

    # 3) 프롬프트(강한 제약) — 섹션별 원문만 참고하도록 명시
    instruction = (
        "너는 과학 커뮤니케이터다. 다음 JSON 스키마의 '키와 구조'를 절대 변경하지 말고, "
        "'값'만 채워라. 출력은 오직 '유효한 JSON' 하나만 허용된다(코드블록/설명/주석 금지). "
        "각 섹션의 'easy'는 해당 섹션의 'original'에 있는 문장만 재서술하여 4-6문장으로 작성하라. "
        "original이 비어 있으면 해당 'easy'는 빈 문자열로 남겨라. "
        "외부 지식/추측/일반 상식/개발자 상상 금지. 사실이 불확실하면 빈 값으로 둔다. "
        "'title', 'authors', 'keywords', 'references' 등 원문에서 명확히 알 수 없는 정보는 빈 값으로 유지한다. "
        "'figures_tables'가 보이면 {label, caption, easy} 형식으로만 작성하며, 없으면 빈 배열. "
        "'plain_summary'는 위 섹션 'easy'에 포함된 내용만 바탕으로 5-7문장으로 요약하라. "
        "JSON은 반드시 '{'로 시작해 '}'로 끝나야 하며, 키 순서/이름을 바꾸지 마라."
    )

    schema_str = json.dumps(data_schema, ensure_ascii=False, indent=2)
    context_only = {
        "abstract": extracted["abstract"],
        "introduction": extracted["introduction"],
        "methods": extracted["methods"],
        "results": extracted["results"],
        "discussion": extracted["discussion"],
        "conclusion": extracted["conclusion"],
    }
    context_str = json.dumps(context_only, ensure_ascii=False, indent=2)

    prompt = (
        f"{instruction}\n\n"
        "=== 출력 스키마(값만 채워라) ===\n"
        f"{schema_str}\n\n"
        "=== 섹션별 original (이 텍스트만 근거로 사용) ===\n"
        f"{context_str}\n\n"
        "위 지시를 따라 JSON만 출력:"
    )

    # 4) 토크나이즈 + 디바이스 이동
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.info(f"INPUT -> device={inputs['input_ids'].device}, shape={tuple(inputs['input_ids'].shape)}")

    # 5) 생성(그리디, 제한된 길이)
    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,              # ✅ 그리디(추측 억제)
            use_cache=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - t0
    logger.info(f"⚡ 추론 완료: {inference_time:.2f}s")

    # 6) 파싱/검증 (+재시도)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = generated[len(prompt):].strip()

    try:
        data = _coerce_json(raw)
        if not _is_meaningful(data):
            raise ValueError("empty_json")
        logger.info("✅ 1차 JSON 파싱/검증 성공")
    except Exception as e:
        logger.warning(f"⚠️ 1차 파싱 실패: {e}. 재시도")
        strict_instruction = (
            "스키마의 키/구조를 유지하고 값만 채운 '유효한 JSON'만 출력하라. "
            "반드시 '{'로 시작해 '}'로 끝나야 한다. 외부지식/추측 금지, 섹션 original만 근거로 사용."
        )
        strict_prompt = f"{strict_instruction}\n\n스키마:\n{schema_str}\n\n섹션 original:\n{context_str}\n\nJSON만 출력:"
        inputs2 = tokenizer(strict_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        with torch.inference_mode():
            outputs2 = model.generate(
                **inputs2,
                max_new_tokens=min(MAX_NEW_TOKENS, 800),
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
        raw2 = gen2[len(strict_prompt):].strip()
        try:
            data = _coerce_json(raw2)
            if not _is_meaningful(data):
                raise ValueError("empty_json_retry")
            logger.info("✅ 2차 JSON 파싱/검증 성공")
        except Exception as e2:
            logger.warning(f"⚠️ 2차 파싱 실패: {e2}. 빈 값 유지하여 스키마 반환")
            data = data_schema  # 원본 스키마(빈 값)로 반환

    total_time = time.time() - start_time
    logger.info(f"🎉 전체 처리 완료: {total_time:.2f}s")

    data["processing_info"] = {
        "gpu_used": gpu_available,
        "inference_time": inference_time,
        "total_time": total_time,
        "input_length": len(request.text),
        "output_length": len(str(data)),
    }
    return data

# -------------------- main --------------------
if __name__ == "__main__":
    # GPU 강제 모드: CUDA 없으면 즉시 종료
    if not torch.cuda.is_available():
        raise SystemExit("CUDA(GPU)가 필요합니다. GPU 환경에서 실행하세요.")
    uvicorn.run(app, host="0.0.0.0", port=5003)