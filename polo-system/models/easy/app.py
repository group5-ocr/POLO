# -*- coding: utf-8 -*-
"""
POLO Easy Model - Grounded JSON Generator (GPU-forced, stable)
- GPU 필수: CUDA 없으면 기동 중단(단, 코드상 CPU 폴백은 허용)
- 어텐션 백엔드 자동 선택: flash_attn > sdpa(가능 시) > eager 폴백
- 섹션별 original만 컨텍스트로 제공하여 'easy'를 재서술(추측 금지)
- Greedy 디코딩, 낮은 토큰 수로 속도/안정성 확보
- JSON 강제 파싱 + 재시도, 실패 시 빈 값 유지
- 배치 엔드포인트(/batch): JSONL → (재서술) → Viz 렌더(/viz) → PNG 저장
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

MAX_NEW_TOKENS = int(os.getenv("EASY_MAX_NEW_TOKENS", "800"))  # VRAM 보호 기본 800
VIZ_MODEL_URL = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
EASY_CONCURRENCY = int(os.getenv("EASY_CONCURRENCY", "8"))
EASY_BATCH_TIMEOUT = int(os.getenv("EASY_BATCH_TIMEOUT", "600"))  # seconds

# -------------------- FastAPI --------------------
app = FastAPI(title="POLO Easy Model", version="1.3.0")

# -------------------- 전역 상태 --------------------
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_available = torch.cuda.is_available()
safe_dtype = torch.float16 if gpu_available else torch.float32

# -------------------- 모델/유틸 --------------------
def _pick_attn_impl() -> str:
    # 1) flash_attn 가능하면 최우선
    try:
        import flash_attn  # noqa: F401
        logger.info("✅ flash_attn 사용: flash_attention_2")
        return "flash_attention_2"
    except Exception:
        pass
    # 2) sdpa 가능 여부
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
        sections[key] = chunk[:2000]  # 섹션 original 길이 제한
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
    if torch.cuda.is_available():
        gpu_available = True
        device = "cuda"
        safe_dtype = torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU 사용 가능: {gpu_name}")
        logger.info(f"🔧 디바이스: {device}, dtype: {safe_dtype}")
    else:
        gpu_available = False
        device = "cpu"
        safe_dtype = torch.float32
        logger.info("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        logger.info(f"🔧 디바이스: {device}, dtype: {safe_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
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
            device_map=None,
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
    m = base
    if ADAPTER_DIR and os.path.exists(ADAPTER_DIR):
        logger.info(f"🔄 어댑터 로딩 시도: {ADAPTER_DIR}")
        try:
            adapter_path = os.path.abspath(ADAPTER_DIR)
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

# -------------------- 내부 유틸 (단일 문장 재서술) --------------------
async def _rewrite_text(text: str) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("모델이 로드되지 않았습니다")

    prompt = (
        "아래 텍스트를 일반인이 이해하기 쉽게 풀어 써라. "
        "추가 가정이나 외부 지식 사용 금지. 제공된 문장만 재서술하며, 모르면 생략:\n\n"
        f"{text}\n\n쉬운 설명:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(MAX_NEW_TOKENS, 800),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()

# -------------------- Viz 호출 (렌더 → PNG 저장) --------------------
async def _send_to_viz(paper_id: str, index: int, text_ko: str, out_dir: Path) -> VizResult:
    """
    Viz API (너의 viz 서비스 스펙에 맞춤):
      POST {VIZ_MODEL_URL}/viz
      body = {"paper_id": "...", "index": 0, "rewritten_text": "한국어 문장", "target_lang":"ko","bilingual":"missing"}
      응답: {"paper_id": "...", "index": 0, "image_path": "...", "success": true}
    """
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
                },
            )
            r.raise_for_status()
            data = r.json()

        img_path = data.get("image_path")

        # 안전망: 혹시 base64나 url이 올 수도 있으니 최소 처리 추가(옵션)
        if not img_path and data.get("image_base64"):
            out_path = out_dir / f"{index:06d}.png"
            raw = base64.b64decode(data["image_base64"])
            out_path.write_bytes(raw)
            return VizResult(ok=True, index=index, image_path=str(out_path))
        if not img_path and data.get("image_url"):
            out_path = out_dir / f"{index:06d}.png"
            async with httpx.AsyncClient(timeout=60) as client:
                rr = await client.get(data["image_url"])
                rr.raise_for_status()
                out_path.write_bytes(rr.content)
            return VizResult(ok=True, index=index, image_path=str(out_path))

        if img_path:
            # viz가 만들어둔 로컬 경로를 그대로 기록
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

@app.post("/easy", response_model=TextResponse)
async def simplify_text(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    simplified_text = await _rewrite_text(request.text)
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

    extracted = _extract_sections(request.text)

    data_schema = json.loads(json.dumps(GROUND_SCHEMA))  # deepcopy
    for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
        data_schema[k]["original"] = extracted.get(k, "")

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
    context_only = {k: extracted[k] for k in ["abstract","introduction","methods","results","discussion","conclusion"]}
    context_str = json.dumps(context_only, ensure_ascii=False, indent=2)

    prompt = (
        f"{instruction}\n\n"
        "=== 출력 스키마(값만 채워라) ===\n"
        f"{schema_str}\n\n"
        "=== 섹션별 original (이 텍스트만 근거로 사용) ===\n"
        f"{context_str}\n\n"
        "위 지시를 따라 JSON만 출력:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.info(f"INPUT -> device={inputs['input_ids'].device}, shape={tuple(inputs['input_ids'].shape)}")

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
    logger.info(f"⚡ 추론 완료: {inference_time:.2f}s")

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

@app.post("/batch", response_model=BatchResult)
async def batch_generate(req: BatchRequest):
    """
    JSONL의 각 라인: {"text": "..."}을
      1) Easy로 한국어 재서술
      2) Viz에 (paper_id, index, rewritten_text) 전달 (/viz)
      3) Viz가 반환한 이미지 경로를 수집 (없으면 에러로 기록)
    """
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
                vz = await _send_to_viz(req.p
aper_id, idx, ko, out_dir)
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
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 사용 가능: {gpu_name}")
            print(f"🔧 디바이스: cuda, dtype: float16")
        else:
            print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
            print(f"🔧 디바이스: cpu, dtype: float32")

        print("🚀 Easy Model 서버 시작 중...")
        uvicorn.run(app, host="0.0.0.0", port=5003)
    except Exception as e:
        print(f"❌ Easy Model 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
