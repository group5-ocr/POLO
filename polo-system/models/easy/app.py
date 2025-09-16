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
import gzip
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from googletrans import Translator

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
translator = Translator()

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

def _translate_to_korean(text: str) -> str:
    """Google Translator를 사용해서 한국어로 번역"""
    try:
        if not text or not text.strip():
            return ""
        
        # 텍스트가 너무 길면 잘라서 번역
        if len(text) > 4000:  # Google Translator 제한
            text = text[:4000] + "..."
        
        result = translator.translate(text, dest='ko', src='en')
        return result.text
    except Exception as e:
        logger.warning(f"번역 실패: {e}")
        return text  # 번역 실패 시 원본 반환

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
    easy_text: Optional[str] = None
    section_title: Optional[str] = None

class BatchResult(BaseModel):
    ok: bool
    paper_id: str
    count: int
    success: int
    failed: int
    out_dir: str
    images: List[VizResult]

class TransportRequest(BaseModel):
    paper_id: str
    transport_path: str
    output_dir: Optional[str] = None

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
        "다음 논문 텍스트를 **일반인도 쉽게 이해할 수 있게** 재해석해주세요.\n\n"
        "🎯 변환 원칙:\n"
        "- 논문의 핵심 내용을 그대로 유지하되, 전문 용어를 쉬운 말로 바꿔주세요\n"
        "- 복잡한 문장은 여러 개의 짧은 문장으로 나누어 설명해주세요\n"
        "- 수식이나 기호는 '이것은 ~을 의미합니다'로 풀어쓰세요\n"
        "- 논문에서 설명하는 방법이나 과정을 단계별로 명확하게 설명해주세요\n"
        "- 논문의 결론이나 핵심 아이디어를 강조해주세요\n"
        "- LaTeX 명령어(\\begin, \\end, \\ref 등)는 무시하고 실제 내용만 설명해주세요\n"
        "- 논문에 없는 내용을 추가하거나 추측하지 마세요\n"
        "- 논문의 원래 의미를 정확히 전달해주세요\n"
        "- 반복적인 내용은 한 번만 설명해주세요\n"
        "- 2-3문장으로 간결하게 설명해주세요\n\n"
        "📝 작성 스타일:\n"
        "- 친근하고 이해하기 쉬운 톤으로 작성해주세요\n"
        "- '~합니다', '~입니다' 같은 존댓말을 사용해주세요 (단, '~요'로 끝나지 않게)\n"
        "- 중요한 내용은 **굵게** 표시해주세요\n"
        "- 논문의 논리적 흐름을 따라 설명해주세요\n"
        "- 구체적인 예시나 비유를 사용해서 설명해주세요\n\n"
        f"[논문 원문]\n{text}\n\n[쉬운 재해석]\n"
    )

def _clean_latex_text(text: str) -> str:
    """LaTeX 명령어를 정리하고 읽기 쉽게 만듭니다"""
    import re
    
    # LaTeX 명령어를 의미있는 텍스트로 변환
    text = re.sub(r'\\title\{([^}]*)\}', r'제목: \1', text)  # \title{content} → 제목: content
    text = re.sub(r'\\author\{([^}]*)\}', r'저자: \1', text)  # \author{content} → 저자: content
    text = re.sub(r'\\section\{([^}]*)\}', r'섹션: \1', text)  # \section{content} → 섹션: content
    text = re.sub(r'\\subsection\{([^}]*)\}', r'하위섹션: \1', text)  # \subsection{content} → 하위섹션: content
    text = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', text)  # \textbf{content} → **content**
    text = re.sub(r'\\textit\{([^}]*)\}', r'*\1*', text)  # \textit{content} → *content*
    
    # 수식 환경을 의미있는 텍스트로 변환
    text = re.sub(r'\$([^$]*)\$', r'수식: \1', text)  # $수식$ → 수식: 수식
    text = re.sub(r'\$\$([^$]*)\$\$', r'수식: \1', text)  # $$수식$$ → 수식: 수식
    
    # LaTeX 환경을 의미있는 텍스트로 변환
    text = re.sub(r'\\begin\{itemize\}', '목록:', text)
    text = re.sub(r'\\item\s*', '• ', text)
    text = re.sub(r'\\end\{itemize\}', '', text)
    
    text = re.sub(r'\\begin\{enumerate\}', '번호목록:', text)
    text = re.sub(r'\\end\{enumerate\}', '', text)
    
    # 특수 문자 정리 (LRB, RRB 등)
    text = re.sub(r'LRB', '(', text)  # LRB → (
    text = re.sub(r'RRB', ')', text)  # RRB → )
    text = re.sub(r'\\ref\{([^}]*)\}', r'그림 \1', text)  # \ref{system} → 그림 system
    text = re.sub(r'\\cite\{([^}]*)\}', '', text)  # \cite{paper} → 제거 (참고문헌)
    
    # 나머지 LaTeX 명령어 제거
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # \command{content}
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # \command
    text = re.sub(r'\\[^a-zA-Z]', '', text)  # \특수문자
    
    # 특수 문자 정리
    text = re.sub(r'[{}]', '', text)  # 중괄호 제거
    text = re.sub(r'\\[a-zA-Z]', '', text)  # 남은 백슬래시 명령어
    
    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def _parse_latex_sections(tex_path: Path) -> List[dict]:
    """LaTeX 파일을 섹션별로 파싱합니다"""
    import re
    
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    current_section = None
    current_content = []
    
    lines = content.split('\n')
    
    for line in lines:
        # 섹션 시작 감지
        if re.match(r'\\section\{([^}]*)\}', line):
            # 이전 섹션 저장
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip()
                })
            
            # 새 섹션 시작
            title_match = re.match(r'\\section\{([^}]*)\}', line)
            current_section = title_match.group(1) if title_match else "Unknown Section"
            current_content = [line]
            
        elif re.match(r'\\subsection\{([^}]*)\}', line):
            # 서브섹션도 섹션으로 처리
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip()
                })
            
            title_match = re.match(r'\\subsection\{([^}]*)\}', line)
            current_section = title_match.group(1) if title_match else "Unknown Subsection"
            current_content = [line]
            
        elif re.match(r'\\begin\{abstract\}', line):
            # Abstract 섹션
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip()
                })
            
            current_section = "Abstract"
            current_content = [line]
            
        elif re.match(r'\\begin\{document\}', line):
            # Document 시작
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip()
                })
            
            current_section = "Introduction"
            current_content = [line]
            
        else:
            # 일반 내용
            if current_section:
                current_content.append(line)
    
    # 마지막 섹션 저장
    if current_section and current_content:
        sections.append({
            "index": len(sections),
            "title": current_section,
            "content": '\n'.join(current_content).strip()
        })
    
    # 빈 섹션 제거
    sections = [s for s in sections if s["content"].strip()]
    
    return sections

async def _rewrite_text(text: str) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("모델이 로드되지 않았습니다")

    # LaTeX 텍스트 정리 (의미있는 텍스트로 변환)
    cleaned_text = _clean_latex_text(text)

    prompt = _build_easy_prompt(cleaned_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=float(os.getenv("EASY_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("EASY_TOP_P", "0.9")),
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # 반복 방지 강화
            no_repeat_ngram_size=3,  # 3-gram 반복 방지
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()

# -------------------- Viz 호출 --------------------
async def _send_to_viz(paper_id: str, index: int, text_ko: str, out_dir: Path) -> VizResult:
    try:
        print(f"🔍 [DEBUG] Viz 모델 호출: {VIZ_MODEL_URL}/viz")
        print(f"🔍 [DEBUG] 전송 데이터: paper_id={paper_id}, index={index}, text_length={len(text_ko)}")
        print(f"🔍 [DEBUG] 전송 텍스트 미리보기: {text_ko[:200]}...")
        
        # Viz 모델이 실행 중인지 먼저 확인
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                health_response = await client.get(f"{VIZ_MODEL_URL.rstrip('/')}/health")
                if health_response.status_code != 200:
                    print(f"❌ [ERROR] Viz 모델 헬스체크 실패: {health_response.status_code}")
                    return VizResult(ok=False, index=index, error="Viz 모델이 실행되지 않음")
                print(f"✅ [SUCCESS] Viz 모델 헬스체크 성공")
        except Exception as e:
            print(f"❌ [ERROR] Viz 모델 헬스체크 실패: {e}")
            return VizResult(ok=False, index=index, error=f"Viz 모델 연결 불가: {e}")
        
        # 실제 Viz 요청
        async with httpx.AsyncClient(timeout=60) as client:  # 타임아웃 증가
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
            print(f"🔍 [DEBUG] Viz 모델 응답: {r.status_code}")
            
            if r.status_code != 200:
                print(f"❌ [ERROR] Viz 모델 응답 실패: {r.status_code} - {r.text}")
                return VizResult(ok=False, index=index, error=f"Viz 모델 응답 실패: {r.status_code}")
            
            try:
                data = r.json()
                print(f"🔍 [DEBUG] Viz 모델 응답 데이터: {data}")
            except Exception as json_error:
                print(f"❌ [ERROR] Viz 모델 응답 JSON 파싱 실패: {json_error}")
                return VizResult(ok=False, index=index, error=f"Viz 모델 응답 파싱 실패: {json_error}")

        img_path = data.get("image_path")

        if not img_path and data.get("image_base64"):
            out_path = out_dir / f"{index:06d}.png"
            out_path.write_bytes(base64.b64decode(data["image_base64"]))
            print(f"✅ [SUCCESS] 이미지 저장: {out_path}")
            return VizResult(ok=True, index=index, image_path=str(out_path))

        if not img_path and data.get("image_url"):
            out_path = out_dir / f"{index:06d}.png"
            async with httpx.AsyncClient(timeout=60) as client:
                rr = await client.get(data["image_url"])
                rr.raise_for_status()
                out_path.write_bytes(rr.content)
            print(f"✅ [SUCCESS] 이미지 저장: {out_path}")
            return VizResult(ok=True, index=index, image_path=str(out_path))

        if img_path:
            print(f"✅ [SUCCESS] 이미지 경로: {img_path}")
            return VizResult(ok=True, index=index, image_path=str(img_path))

        print(f"❌ [ERROR] 이미지 경로 없음: {data}")
        return VizResult(ok=False, index=index, error="No image_path from viz")
    except httpx.ConnectError as e:
        print(f"❌ [ERROR] Viz 모델 연결 실패: {e}")
        return VizResult(ok=False, index=index, error=f"Viz 모델 연결 실패: {e}")
    except httpx.TimeoutException as e:
        print(f"❌ [ERROR] Viz 모델 타임아웃: {e}")
        return VizResult(ok=False, index=index, error=f"Viz 모델 타임아웃: {e}")
    except Exception as e:
        print(f"❌ [ERROR] Viz 모델 처리 실패: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"🔍 [DEBUG] Easy /batch 엔드포인트 호출됨")
    print(f"🔍 [DEBUG] 요청 데이터:")
    print(f"  - paper_id: {req.paper_id}")
    print(f"  - chunks_jsonl: {req.chunks_jsonl}")
    print(f"  - output_dir: {req.output_dir}")
    
    # merged_body.tex 파일 경로로 변경
    tex_path = Path(req.chunks_jsonl).parent / "merged_body.tex"
    out_dir = Path(req.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 [DEBUG] 파일 경로 확인:")
    print(f"  - tex_path: {tex_path}")
    print(f"  - tex_path 존재: {tex_path.exists()}")
    print(f"  - out_dir: {out_dir}")
    print(f"  - out_dir 생성됨: {out_dir.exists()}")

    if not tex_path.exists():
        print(f"❌ [ERROR] merged_body.tex 파일이 존재하지 않음: {tex_path}")
        raise HTTPException(status_code=400, detail=f"merged_body.tex not found: {tex_path}")

    # LaTeX 파일을 섹션별로 분할
    sections = _parse_latex_sections(tex_path)
    print(f"🔍 [DEBUG] 총 {len(sections)}개 섹션 파싱됨")
    
    if not sections:
        print(f"❌ [ERROR] 유효한 섹션이 없음")
        raise HTTPException(status_code=400, detail="No valid sections found in merged_body.tex")

    # 모델 상태 확인
    if model is None or tokenizer is None:
        print(f"❌ [ERROR] 모델이 로드되지 않음")
        raise HTTPException(status_code=500, detail="Model not loaded")

    print(f"🔍 [DEBUG] 모델 상태: model={model is not None}, tokenizer={tokenizer is not None}")
    print(f"🔍 [DEBUG] 디바이스: {device}, GPU 사용: {gpu_available}")

    sem = anyio.Semaphore(EASY_CONCURRENCY)
    results: List[VizResult] = []

    async def worker(section: dict):
        async with sem:
            idx = section["index"]
            try:
                print(f"🔍 [DEBUG] 섹션 {idx}/{len(sections)} 처리 시작: {section['title']}")
                ko = await _rewrite_text(section["content"])
                print(f"🔍 [DEBUG] 섹션 {idx}/{len(sections)} 변환 완료: {ko[:100]}...")
                
                # Google Translator로 한국어 번역
                print(f"🔍 [DEBUG] 섹션 {idx}/{len(sections)} 한국어 번역 시작...")
                ko_translated = _translate_to_korean(ko)
                print(f"🔍 [DEBUG] 섹션 {idx}/{len(sections)} 한국어 번역 완료: {ko_translated[:100]}...")
                
                # 한국어 번역본으로 Viz 처리
                vz = await _send_to_viz(req.paper_id, idx, ko_translated, out_dir)
                print(f"🔍 [DEBUG] 섹션 {idx}/{len(sections)} Viz 완료: {vz.ok}")
                
                # 결과에 번역된 텍스트 저장
                vz.easy_text = ko_translated
                vz.section_title = section["title"]
                results.append(vz)
                
                # 진행률 표시
                completed = len(results)
                progress = (completed / len(sections)) * 100
                print(f"📊 [PROGRESS] {completed}/{len(sections)} ({progress:.1f}%) 완료")
                
            except Exception as e:
                print(f"❌ [ERROR] 섹션 {idx}/{len(sections)} 처리 실패: {e}")
                results.append(VizResult(ok=False, index=idx, error=str(e)))

    print(f"🔍 [DEBUG] 배치 처리 시작...")
    async with anyio.create_task_group() as tg:
        for section in sections:
            tg.start_soon(worker, section)

    ok_cnt = sum(1 for r in results if r.ok)
    fail_cnt = len(results) - ok_cnt
    
    print(f"🔍 [DEBUG] 배치 처리 완료:")
    print(f"  - 총 섹션: {len(sections)}")
    print(f"  - 성공: {ok_cnt}")
    print(f"  - 실패: {fail_cnt}")
    
    result = BatchResult(
        ok=fail_cnt == 0,
        paper_id=req.paper_id,
        count=len(sections),
        success=ok_cnt,
        failed=fail_cnt,
        out_dir=str(out_dir),
        images=sorted(results, key=lambda r: r.index),
    )
    
    # JSON 결과 파일 생성 (프론트엔드용)
    json_result = {
        "paper_id": req.paper_id,
        "total_sections": len(sections),
        "success_count": ok_cnt,
        "failed_count": fail_cnt,
        "sections": []
    }
    
    # 각 섹션별 결과 추가
    for i, section in enumerate(sections):
        section_result = {
            "index": i,
            "title": section["title"],
            "original_content": section["content"][:200] + "..." if len(section["content"]) > 200 else section["content"],
            "easy_text": "",
            "korean_translation": "",
            "image_path": "",
            "status": "failed"
        }
        
        # 해당 인덱스의 결과 찾기
        for r in results:
            if r.index == i:
                section_result["status"] = "success" if r.ok else "failed"
                if r.ok and r.image_path:
                    section_result["image_path"] = r.image_path
                if hasattr(r, 'easy_text') and r.easy_text:
                    section_result["korean_translation"] = r.easy_text
                break
        
        json_result["sections"].append(section_result)
    
    # JSON 파일 저장
    json_file_path = out_dir / "easy_results.json"
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    
    print(f"📄 [JSON] 결과 파일 저장: {json_file_path}")
    print(f"✅ [SUCCESS] Easy 모델 배치 처리 완료: {result}")
    return result

@app.post("/from-transport", response_model=BatchResult)
async def generate_from_transport(req: TransportRequest):
    tp = Path(req.transport_path)
    if not tp.exists():
        raise HTTPException(status_code=400, detail=f"transport.json not found: {tp}")

    try:
        data = json.loads(tp.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid transport.json: {e}")

    # chunks 경로 우선: artifacts.chunks.path → tp.parent/chunks.jsonl → tp.parent/chunks.jsonl.gz
    chunks_path: Optional[Path] = None
    try:
        chunks_path_str = (((data.get("artifacts", {}) or {}).get("chunks", {}) or {}).get("path"))
        if chunks_path_str:
            chunks_path = Path(chunks_path_str)
    except Exception:
        chunks_path = None
    if chunks_path is None:
        base_dir = tp.parent
        cand1 = base_dir / "chunks.jsonl"
        cand2 = base_dir / "chunks.jsonl.gz"
        if cand1.exists():
            chunks_path = cand1
        elif cand2.exists():
            chunks_path = cand2

    if chunks_path is None or not chunks_path.exists():
        raise HTTPException(status_code=400, detail=f"chunks file not found near transport: {tp}")

    # 출력 경로
    out_dir = Path(req.output_dir).resolve() if req.output_dir else (tp.parent / "easy_outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 기존 배치 로직 재사용
    items: List[dict] = []
    open_fn = gzip.open if str(chunks_path).endswith(".gz") else open
    with open_fn(chunks_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                items.append({"index": i, "text": text})
            except Exception:
                continue

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
