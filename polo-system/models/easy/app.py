# -*- coding: utf-8 -*-
"""
POLO Easy Model - Grounded JSON Generator

- CUDA 우선(없으면 CPU 폴백)
- 어텐션 백엔드: flash_attn > sdpa > eager
- '쉬운 한국어 재해석'에 최적화
- /easy, /generate, /batch 제공
"""
from __future__ import annotations

import os
import re
import json
import time
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import anyio
import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

# -------------------- .env 로드 (루트만) --------------------
logger = logging.getLogger("polo.easy")
logging.basicConfig(level=logging.INFO)

ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
if ROOT_ENV.exists():
    load_dotenv(dotenv_path=str(ROOT_ENV), override=True)
    logger.info(f"[dotenv] loaded: {ROOT_ENV}")
else:
    logger.info("[dotenv] no .env at repo root")

HF_TOKEN   = os.getenv("허깅페이스 토큰")
BASE_MODEL = os.getenv("EASY_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
ADAPTER_DIR = os.getenv(
    "EASY_ADAPTER_DIR",
    str(Path(__file__).resolve().parent.parent / "fine-tuning" / "outputs" / "llama32-3b-qlora" / "checkpoint-4000")
)
MAX_NEW_TOKENS      = int(os.getenv("EASY_MAX_NEW_TOKENS", "1200"))
VIZ_MODEL_URL       = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
EASY_CONCURRENCY    = int(os.getenv("EASY_CONCURRENCY", "8"))
EASY_BATCH_TIMEOUT  = int(os.getenv("EASY_BATCH_TIMEOUT", "600"))

# -------------------- HF 캐시 경로 '무조건' 안전 폴더로 고정 --------------------
SAFE_CACHE_DIR = Path(__file__).resolve().parent / "hf_cache"
SAFE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def force_safe_hf_cache():
    # 시스템 전역/사용자 환경변수에 이상한 경로(D:\...)가 있어도 여기로 통일
    for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        os.environ[k] = str(SAFE_CACHE_DIR)
    logger.info(f"[hf_cache] forced cache dir → {SAFE_CACHE_DIR}")

force_safe_hf_cache()
CACHE_DIR = os.environ["HF_HOME"]  # 동일 경로 사용

# ⛔ transformers/peft는 캐시 경로가 import 시점에 굳어질 수 있음
#    반드시 캐시 세팅 이후에 import
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

# -------------------- FastAPI --------------------
app = FastAPI(title="POLO Easy Model", version="1.3.2")

# -------------------- 전역 상태 --------------------
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_available = torch.cuda.is_available()
safe_dtype = torch.float16 if gpu_available else torch.float32

# -------------------- 유틸 --------------------
def _pick_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        logger.info("✅ flash_attn 사용: flash_attention_2")
        return "flash_attention_2"
    except Exception:
        pass
    try:
        from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
        logger.info("ℹ️ sdpa 사용 가능")
        return "sdpa"
    except Exception:
        logger.info("ℹ️ sdpa 불가 → eager로 진행")
        return "eager"

def _coerce_json(text: str) -> Dict[str, Any]:
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s:e+1]
    return json.loads(text)

def _is_meaningful(d: dict) -> bool:
    try:
        sections = ["abstract","introduction","methods","results","discussion","conclusion"]
        return any(len((d.get(s, {}) or {}).get("easy", "")) > 10 for s in sections)
    except Exception:
        return False

def _extract_sections(src: str) -> dict:
    sections = {k: "" for k in ["abstract","introduction","methods","results","discussion","conclusion"]}
    headers = [
        ("abstract", r"^\s*abstract\b[:\-]?"),
        ("introduction", r"^\s*introduction\b[:\-]?"),
        ("methods", r"^\s*methods?\b[:\-]?|^\s*materials?\s+and\s+methods\b[:\-]?"),
        ("results", r"^\s*results?\b[:\-]?"),
        ("discussion", r"^\s*discussion\b[:\-]?"),
        ("conclusion", r"^\s*conclusion[s]?\b[:\-]?|^\s*concluding\s+remarks\b[:\-]?"),
    ]
    lines = src.splitlines()
    idxs = []
    for i, line in enumerate(lines):
        for key, pat in headers:
            if re.match(pat, line.strip(), flags=re.IGNORECASE):
                idxs.append((i, key))
                break
    idxs.sort()
    for j, (start_i, key) in enumerate(idxs):
        end_i = idxs[j+1][0] if j+1 < len(idxs) else len(lines)
        sections[key] = "\n".join(lines[start_i+1:end_i]).strip()[:2000]
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

# -------------------- I/O 모델 --------------------
class TextRequest(BaseModel):
    text: str
    translate: Optional[bool] = False
    model_config = ConfigDict(extra="allow")

class TextResponse(BaseModel):
    simplified_text: str
    translated_text: Optional[str] = None

class BatchRequest(BaseModel):
    paper_id: str = Field(..., description="결과 파일/경로 식별자")
    chunks_jsonl: str = Field(..., description="각 라인에 {'text': ...} 형태의 JSONL")
    output_dir: str = Field(..., description="이미지/결과 저장 루트")

class VizResult(BaseModel):
    ok: bool = True
    index: int
    image_path: Optional[str] = None
    error: Optional[str] = None

class BatchResult(BaseModel):
    ok: bool
    paper_id: str
    count: int
    success: int
    failed: int
    out_dir: str
    images: List[VizResult]

# -------------------- 모델 로드 --------------------
def load_model():
    global model, tokenizer, gpu_available, device, safe_dtype

    logger.info(f"🔄 모델 로딩 시작: {BASE_MODEL}")
    logger.info(f"EASY_ADAPTER_DIR={ADAPTER_DIR}")
    logger.info(f"HF_HOME={os.getenv('HF_HOME')}")

    if torch.cuda.is_available():
        gpu_available = True
        device = "cuda"
        safe_dtype = torch.float16
        logger.info(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        gpu_available = False
        device = "cpu"
        safe_dtype = torch.float32
        logger.info("⚠️ GPU 미사용 → CPU 모드")

    # 토크나이저 (캐시 고정)
    tokenizer_local = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    if tokenizer_local.pad_token_id is None:
        tokenizer_local.pad_token = tokenizer_local.eos_token

    attn_impl = _pick_attn_impl()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 베이스 모델
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=safe_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
            device_map=None,
            cache_dir=CACHE_DIR,
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
            cache_dir=CACHE_DIR,
        )

    # LoRA 어댑터(선택)
    m = base
    if ADAPTER_DIR and os.path.exists(ADAPTER_DIR):
        try:
            m = PeftModel.from_pretrained(
                base,
                os.path.abspath(ADAPTER_DIR),
                is_trainable=False,
                local_files_only=True,
            )
            logger.info("✅ 어댑터 로딩 성공")
        except Exception as e:
            logger.error(f"❌ 어댑터 로딩 실패: {e} → 베이스로 진행")
            m = base
    else:
        logger.info("ℹ️ 어댑터 경로 없음(베이스만 사용)")

    m.eval()
    m = m.to(safe_dtype).to(device)
    logger.info(f"🧠 MODEL DEVICE: {next(m.parameters()).device}, DTYPE: {next(m.parameters()).dtype}")

    # 전역 주입
    globals()["model"] = m
    globals()["tokenizer"] = tokenizer_local
    logger.info("✅ 모델 로딩 완료")

# -------------------- 스타트업 --------------------
@app.on_event("startup")
async def startup_event():
    load_model()

# -------------------- 내부 유틸 (재해석) --------------------
def _build_easy_prompt(text: str) -> str:
    return (
        "다음 텍스트를 **중학생도 이해할 수 있게 쉽고 재미있게** 변환해라.\n\n"
        "🎯 변환 원칙:\n"
        "- 전문 용어는 일상 용어로 바꾸기 (예: '알고리즘' → '문제 해결 방법')\n"
        "- 복잡한 문장은 짧고 명확하게 나누기\n"
        "- 추상적인 개념은 구체적인 비유로 설명하기\n"
        "- 수식이나 기호는 '이것은 ~을 의미해요'로 풀어쓰기\n"
        "- 핵심 내용은 '요약하면'으로 정리하기\n"
        "- 어려운 부분은 '쉽게 말하면'으로 다시 설명하기\n\n"
        "📝 작성 스타일:\n"
        "- 친근하고 대화하는 톤으로 작성\n"
        "- '~합니다', '~입니다' 같은 존댓말 사용 (단, '~요'로 끝나지 않게)\n"
        "- 중요한 내용은 **굵게** 표시\n"
        "- 단계별 설명은 1), 2), 3) 형태로 정리\n\n"
        f"[원문]\n{text}\n\n[쉬운 설명]\n"
    )

async def _rewrite_text(text: str) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("모델이 로드되지 않았습니다")

    prompt = _build_easy_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()

# -------------------- Viz 호출 --------------------
async def _send_to_viz(paper_id: str, index: int, text_ko: str, out_dir: Path) -> VizResult:
    try:
        async with httpx.AsyncClient(timeout=EASY_BATCH_TIMEOUT) as client:
            r = await client.post(
                f"{VIZ_MODEL_URL.rstrip('/')}/viz",
                json={
                    "paper_id": paper_id,
                    "index": index,
                    "rewritten_text": text_ko,
                    "target_lang": "ko",
                    "bilingual": "missing",
                    "text_type": "easy_korean",  # 쉽게 변환된 한국어임을 명시
                },
            )
            r.raise_for_status()
            data = r.json()

        img_path = data.get("image_path")

        if not img_path and data.get("image_base64"):
            out_path = out_dir / f"{index:06d}.png"
            out_path.write_bytes(base64.b64decode(data["image_base64"]))
            return VizResult(ok=True, index=index, image_path=str(out_path))

        if not img_path and data.get("image_url"):
            out_path = out_dir / f"{index:06d}.png"
            async with httpx.AsyncClient(timeout=60) as client:
                rr = await client.get(data["image_url"])
                rr.raise_for_status()
                out_path.write_bytes(rr.content)
            return VizResult(ok=True, index=index, image_path=str(out_path))

        if img_path:
            return VizResult(ok=True, index=index, image_path=str(img_path))

        return VizResult(ok=False, index=index, error="No image_path from viz")
    except Exception as e:
        return VizResult(ok=False, index=index, error=str(e))

# -------------------- 엔드포인트 --------------------
@app.get("/")
async def root():
    return {"message": "POLO Easy Model API", "model": BASE_MODEL}

@app.get("/health")
async def health():
    return {
        "status": "healthy" if (model is not None and tokenizer is not None) else "unavailable",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": gpu_available,
        "gpu_device": torch.cuda.get_device_name(0) if gpu_available else None,
        "model_name": BASE_MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "cache_dir": str(CACHE_DIR),
    }

@app.get("/healthz")
async def healthz():
    return await health()

@app.post("/easy", response_model=TextResponse)
async def simplify_text(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    simplified_text = await _rewrite_text(request.text)
    return TextResponse(simplified_text=simplified_text, translated_text=None)

@app.post("/generate")
async def generate_json(request: TextRequest):
    start_time = time.time()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

    extracted = _extract_sections(request.text)
    data_schema = json.loads(json.dumps(GROUND_SCHEMA))  # deepcopy
    for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
        data_schema[k]["original"] = extracted.get(k, "")

    instruction = (
        "너는 친근한 과학 선생님이다. 다음 JSON 스키마의 '키와 구조'를 절대 변경하지 말고 "
        "'값'만 채워라. 출력은 오직 '유효한 JSON' 하나만 허용된다(코드블록/설명/주석 금지).\n\n"
        "🎯 각 섹션의 'easy' 작성 방법:\n"
        "- 중학생도 이해할 수 있게 쉽고 재미있게 변환\n"
        "- 전문 용어는 일상 용어로 바꾸기\n"
        "- 복잡한 내용은 단계별로 나누어 설명\n"
        "- 구체적인 비유와 예시 사용\n"
        "- '요약하면', '쉽게 말하면' 같은 정리 문구 활용\n"
        "- 친근한 톤으로 4-8문장 작성 (존댓말 '~합니다', '~입니다' 사용, '~요'로 끝나지 않게)\n\n"
        "original이 비어 있으면 해당 'easy'는 빈 문자열로 남겨라. 외부 지식/추측 금지."
    )

    schema_str = json.dumps(data_schema, ensure_ascii=False, indent=2)
    context_only = {k: extracted[k] for k in ["abstract","introduction","methods","results","discussion","conclusion"]}
    context_str = json.dumps(context_only, ensure_ascii=False, indent=2)

    prompt = (
        f"{instruction}\n\n=== 출력 스키마(값만 채워라) ===\n{schema_str}\n\n"
        f"=== 섹션별 original (이 텍스트만 근거로 사용) ===\n{context_str}\n\n"
        "JSON만 출력:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - t0

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = generated[len(prompt):].strip()

    try:
        data = _coerce_json(raw)
        if not _is_meaningful(data):
            raise ValueError("empty_json")
    except Exception:
        strict_instruction = (
            "스키마의 키/구조를 유지하고 값만 채운 '유효한 JSON'만 출력하라. "
            "반드시 '{'로 시작해 '}'로 끝나야 한다. 외부지식/추측 금지."
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
        except Exception:
            data = data_schema

    total_time = time.time() - start_time
    data["processing_info"] = {
        "gpu_used": gpu_available,
        "inference_time": inference_time,
        "total_time": total_time,
        "input_length": len(request.text),
        "output_length": len(str(data)),
    }
    return data

@app.post("/batch", response_model=BatchResult)
async def batch_generate(req: BatchRequest):
    jsonl_path = Path(req.chunks_jsonl).resolve()
    out_dir = Path(req.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        raise HTTPException(status_code=400, detail=f"JSONL not found: {jsonl_path}")

    # JSONL 로드
    items: List[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    logger.warning(f"[batch] 라인 {i}: text 비어있음 → skip")
                    continue
                items.append({"index": i, "text": text})
            except Exception as e:
                logger.warning(f"[batch] 라인 {i}: JSON 파싱 실패 → skip ({e})")

    sem = anyio.Semaphore(EASY_CONCURRENCY)
    results: List[VizResult] = []

    async def worker(item: dict):
        async with sem:
            idx = item["index"]
            try:
                ko = await _rewrite_text(item["text"])
                vz = await _send_to_viz(req.paper_id, idx, ko, out_dir)
                results.append(vz)
            except Exception as e:
                results.append(VizResult(ok=False, index=idx, error=str(e)))

    async with anyio.create_task_group() as tg:
        for item in items:
            tg.start_soon(worker, item)

    ok_cnt = sum(1 for r in results if r.ok)
    fail_cnt = len(results) - ok_cnt
    return BatchResult(
        ok=fail_cnt == 0,
        paper_id=req.paper_id,
        count=len(items),
        success=ok_cnt,
        failed=fail_cnt,
        out_dir=str(out_dir),
        images=sorted(results, key=lambda r: r.index),
    )

# -------------------- main --------------------
if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            print("🔧 디바이스: cuda, dtype: float16")
        else:
            print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
            print("🔧 디바이스: cpu, dtype: float32")

        print("🚀 Easy Model 서버 시작 중...")
        uvicorn.run(app, host="0.0.0.0", port=5003)
    except Exception as e:
        print(f"❌ Easy Model 시작 실패: {e}")
        import traceback
        traceback.print_exc()
