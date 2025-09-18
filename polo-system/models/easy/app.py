# -*- coding: utf-8 -*-
"""
POLO Easy Model - Grounded JSON/HTML Generator (Final, KR-optimized)
- 한글 결과 강제, 환각 방지, 스키마/형식 보호, 반복/잡음 정리, 안전 토크나이즈
"""

from __future__ import annotations
import os, re, json, time, base64, gzip, sys, asyncio, logging, html
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import anyio, httpx, torch, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

# -------------------- .env --------------------
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
if ROOT_ENV.exists():
    load_dotenv(dotenv_path=str(ROOT_ENV), override=True)

# -------------------- ENV / 기본값 --------------------
HF_TOKEN                 = os.getenv("HUGGINGFACE_TOKEN", "")
BASE_MODEL               = os.getenv("EASY_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
ADAPTER_DIR              = os.getenv("EASY_ADAPTER_DIR", str(Path(__file__).resolve().parent.parent / "fine-tuning" / "outputs" / "llama32-3b-qlora" / "checkpoint-4000"))
MAX_NEW_TOKENS           = int(os.getenv("EASY_MAX_NEW_TOKENS", "1200"))
EASY_CONCURRENCY         = int(os.getenv("EASY_CONCURRENCY", "2"))
EASY_BATCH_TIMEOUT       = int(os.getenv("EASY_BATCH_TIMEOUT", "1800"))
EASY_VIZ_TIMEOUT         = float(os.getenv("EASY_VIZ_TIMEOUT", "1800"))
EASY_VIZ_HEALTH_TIMEOUT  = float(os.getenv("EASY_VIZ_HEALTH_TIMEOUT", "5"))
EASY_VIZ_ENABLED         = os.getenv("EASY_VIZ_ENABLED", "0").lower() in ("1","true","yes")
VIZ_MODEL_URL            = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")

EASY_STRIP_MATH          = os.getenv("EASY_STRIP_MATH", "1").lower() in ("1","true","yes")
EASY_FORCE_KO            = os.getenv("EASY_FORCE_KO", "1").lower() in ("1","true","yes")
EASY_AUTO_BOLD           = os.getenv("EASY_AUTO_BOLD", "1").lower() in ("1","true","yes")
EASY_HILITE              = os.getenv("EASY_HILITE", "1").lower() in ("1","true","yes")

# 외부 번역기 항상 우선 (Google→Papago→LLM)
EASY_FORCE_TRANSLATE     = os.getenv("EASY_FORCE_TRANSLATE", "1").lower() in ("1","true","yes")
GOOGLE_PROJECT           = (os.getenv("GOOGLE_PROJECT", "polo-472507") or "").strip()

MAX_INPUT_TOKENS         = 2048
_RETRY_TOKENS            = (1536, 1024, 768)

# -------------------- HF cache pin --------------------
SAFE_CACHE_DIR = Path(__file__).resolve().parent / "hf_cache"
SAFE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
for k in ("HF_HOME","TRANSFORMERS_CACHE","HF_DATASETS_CACHE","HF_HUB_CACHE"):
    os.environ[k] = str(SAFE_CACHE_DIR)
CACHE_DIR = os.environ["HF_HOME"]

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

# -------------------- Logger --------------------
logger = logging.getLogger("polo.easy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Windows 소켓 종료 관련 경고/에러 완화
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logging.getLogger("asyncio").setLevel(logging.WARNING)

# -------------------- Global state --------------------
app = FastAPI(title="POLO Easy Model", version="2.2.0-final-kr")
model: Optional[torch.nn.Module] = None
tokenizer: Optional[AutoTokenizer] = None
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_available = torch.cuda.is_available()
safe_dtype = torch.float16 if gpu_available else torch.float32

# -------------------- Text Utils --------------------
def _pick_attn_impl() -> str:
    try:
        import flash_attn  # noqa
        logger.info("✅ flash_attn 사용: flash_attention_2")
        return "flash_attention_2"
    except Exception:
        pass
    try:
        from torch.nn.functional import scaled_dot_product_attention  # noqa
        logger.info("ℹ️ sdpa 사용 가능")
        return "sdpa"
    except Exception:
        logger.info("ℹ️ sdpa 불가 → eager")
        return "eager"

def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))

def _detect_lang_safe(text: str) -> str:
    if not text or not text.strip(): return "en"
    if _contains_hangul(text): return "ko"
    latin = len(re.findall(r"[A-Za-z]", text)); total = len(re.findall(r"[A-Za-z가-힣]", text))
    if total == 0: return "en"
    return "ko" if (latin/total) < 0.5 else "en"

def _normalize_bracket_tokens(t: str) -> str:
    if not t: return t
    t = re.sub(r"(?i)\bL\s*R\s*B\b|\blrb\b|\bl\s*r\s*b\b", "(", t)
    t = re.sub(r"(?i)\bR\s*R\s*B\b|\brrb\b|\br\s*r\s*b\b", ")", t)
    return t

def _strip_math_all(s: str) -> str:
    if not s: return s
    rep = " . "
    s = re.sub(r"\$\$[\s\S]*?\$\$", rep, s)
    s = re.sub(r"\\\[[\s\S]*?\\\]", rep, s)
    s = re.sub(r"\\\([\s\S]*?\\\)", rep, s)
    s = re.sub(r"\$(?!\$)(?:[^$\\]|\\.)+\$", rep, s)
    s = re.sub(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}[\s\S]*?\\end\{\1\}", rep, s)
    return s

def _strip_tables_figures_all(s: str) -> str:
    if not s: return s
    rep = " . "
    s = re.sub(r"\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}", rep, s)
    s = re.sub(r"\\begin\{tabularx?\*?\}[\s\S]*?\\end\{tabularx?\*?\}", rep, s)
    s = re.sub(r"\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}", rep, s)
    s = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", rep, s)
    s = re.sub(r"\\caption\{[^}]*\}", rep, s)
    s = re.sub(r"(?i)\b(?:figure|table)\s*\d+\s*[:.]\s*", rep, s)
    s = re.sub(r"(?:표|그림)\s*\d+\s*[:.]\s*", rep, s)
    return s

def _postprocess_terms(t: str) -> str:
    if not t: return t
    t = re.sub(r"(?i)\bm\s*A\s*P\b", "mAP", t)
    t = re.sub(r"(?i)\bR\s*-?\s*CNN\b", "R-CNN", t)
    t = re.sub(r"(?i)\bFast\s*-?\s*R\s*-?\s*CNN\b", "Fast R-CNN", t)
    t = re.sub(r"(?i)\bFaster\s*-?\s*R\s*-?\s*CNN\b", "Faster R-CNN", t)
    return t

def _auto_bold_terms(t: str) -> str:
    if not t or not EASY_AUTO_BOLD: return t
    pats = [
        r"\b(?:BLEU|ROUGE|BERTScore|BLEURT|mAP|IoU|F1|AUC|ROC|PSNR|SSIM|CER|WER)\b",
        r"\b(?:Transformer|BERT|GPT(?:-\d+)?|Llama|LLaMA|Mistral|Qwen|Gemma|T5|Whisper)\b",
        r"\b\d+\s?(?:x|×)\s?\d+\b",
        r"\b(?:Top-1|Top-5|SOTA|FPS|GFLOPs?|Params?)\b",
    ]
    for pat in pats:
        t = re.sub(pat, lambda m: f"**{m.group(0)}**" if not (m.group(0).startswith("**") and m.group(0).endswith("**")) else m.group(0), t)
    return t

def _hilite_sentences(t: str, max_marks: int = 2) -> str:
    if not t or not EASY_HILITE: return t
    parts = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+", t.strip()); marks=0; out=[]
    key = re.compile(r"(핵심|요약|결론|성능|개선|향상|정확도|우수|달성|증가|감소|한계|제안|우리\s*모델|SOTA|improv|achiev|propos|contribut|BER|SNR)", re.I)
    has_num = re.compile(r"\d+(?:\.\d+)?\s*(%|점|배|ms|s|fps|GFLOPs?|M|K|dB)?", re.I)
    for s in parts:
        s=s.strip()
        if not s: continue
        if marks < max_marks and (key.search(s) or has_num.search(s)):
            out.append(f"=={s}=="); marks += 1
        else:
            out.append(s)
    return " ".join(out)

def _sanitize_repeats(t: str) -> str:
    if not t: return t
    t = re.sub(r"\b(\w{2,})\b(?:\s+\1\b){3,}", r"\1 \1 \1", t, flags=re.I)
    t = re.sub(r"(?:\b[\w\-]*flops\b[\s,.;:]*){6,}", " ", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _need_ko(t: str) -> bool:
    if not t or not t.strip(): return True
    hangul = len(re.findall(r"[가-힣]", t)); latin = len(re.findall(r"[A-Za-z]", t))
    total = hangul + latin
    if hangul < 10: return True
    if total > 0 and hangul / total < 0.35: return True
    return False

# -------------------- 안전 토크나이즈 --------------------
def _safe_tokenize(prompt: str, max_len: int = MAX_INPUT_TOKENS):
    assert tokenizer is not None, "Tokenizer not loaded"
    for L in (max_len, *_RETRY_TOKENS):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=L)
        if enc["input_ids"].shape[1] <= L:
            return {k: v.to(device) for k, v in enc.items()}
    enc = tokenizer(prompt[-8000:], return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()}

# -------------------- 모델 로드 --------------------
def load_model():
    global model, tokenizer, gpu_available, device, safe_dtype
    logger.info(f"🔄 모델 로딩: {BASE_MODEL}")
    logger.info(f"EASY_ADAPTER_DIR={ADAPTER_DIR}")
    logger.info(f"HF_HOME={os.getenv('HF_HOME')}")
    if torch.cuda.is_available():
        gpu_available = True; device = "cuda"; safe_dtype = torch.float16
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        gpu_available = False; device = "cpu"; safe_dtype = torch.float32
        logger.info("⚠️ CPU 모드")

    tokenizer_local = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tokenizer_local.pad_token_id is None:
        tokenizer_local.pad_token = tokenizer_local.eos_token

    attn_impl = _pick_attn_impl()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=safe_dtype, trust_remote_code=True,
            attn_implementation=attn_impl, low_cpu_mem_usage=True, token=HF_TOKEN, cache_dir=CACHE_DIR,
        )
    except Exception as e:
        logger.warning(f"attn='{attn_impl}' 실패({e}) → eager")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=safe_dtype, trust_remote_code=True,
            attn_implementation="eager", low_cpu_mem_usage=True, token=HF_TOKEN, cache_dir=CACHE_DIR,
        )

    m = base
    if ADAPTER_DIR and os.path.exists(ADAPTER_DIR):
        try:
            m = PeftModel.from_pretrained(base, os.path.abspath(ADAPTER_DIR), is_trainable=False, local_files_only=True)
            logger.info("✅ 어댑터 로딩 OK")
        except Exception as e:
            logger.error(f"❌ 어댑터 로딩 실패: {e} → 베이스 사용")
            m = base

    m.eval(); m = m.to(safe_dtype).to(device)
    globals()["model"] = m; globals()["tokenizer"] = tokenizer_local
    logger.info("✅ 모델 준비 완료")

# -------------------- 번역 파이프라인 (v3 우선, 프로젝트 고정) --------------------
PROJECT_FIXED = "polo-472507"  # ← 고정 프로젝트 ID
GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT", PROJECT_FIXED).strip() or PROJECT_FIXED

def _translate_to_korean(text: str) -> str:
    """
    외부 번역 우선: Google v3(서비스계정) → v2(API Key/OAuth) → (옵션) Papago → 실패 시 LLM
    """
    try:
        if not text or not text.strip():
            return ""
        out = _translate_with_google_api(text)
        if out:
            return out
        out = _translate_with_papago(text)  # env 없으면 자동 skip
        if out:
            return out
        logger.warning("⚠️ 모든 외부 번역 실패 → LLM 백업 사용")
        return _translate_with_llm(text)
    except Exception as e:
        logger.warning(f"번역 실패: {e}")
        return text

def _translate_with_google_api(text: str) -> str:
    """
    Google Translation 호출:
      1) v3 Advanced (OAuth/Service Account, 프로젝트 고정)
      2) v2 (API Key) → 3) v2 (OAuth/Service Account)
    """
    try:
        import requests
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request as GARequest

        # ----- 공통 유틸 -----
        def _find_service_account_path() -> Optional[Path]:
            base = Path(__file__).resolve().parent
            primary = base / "polo-472507-c5db0ee4a109.json"  # 주키
            if primary.exists():
                return primary
            for p in base.glob("*.json"):
                try:
                    info = json.loads(p.read_text(encoding="utf-8"))
                    if str(info.get("type", "")) == "service_account":
                        return p
                except Exception:
                    continue
            return None

        def _load_google_api_key() -> str:
            try:
                base = Path(__file__).resolve().parent
                primary = base / "polo-472507-c5db0ee4a109.json"
                candidates = [primary] + [p for p in base.glob("*.json") if p != primary][:5]
                for jf in candidates:
                    if jf.exists():
                        d = json.loads(jf.read_text(encoding="utf-8"))
                        for key_field in ("GOOGLE_API_KEY", "api_key", "apiKey", "key"):
                            v = str(d.get(key_field, "")).strip()
                            if v:
                                return v
            except Exception:
                pass
            return ""

        # ===== 1) v3 Advanced (프로젝트 고정) =====
        sa_path = _find_service_account_path()
        if sa_path:
            try:
                info = json.loads(sa_path.read_text(encoding="utf-8"))
                if str(info.get("type", "")) == "service_account":
                    scopes = [
                        "https://www.googleapis.com/auth/cloud-translation",
                        "https://www.googleapis.com/auth/cloud-platform",
                    ]
                    creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
                    creds.refresh(GARequest())
                    headers = {
                        "Authorization": f"Bearer {creds.token}",
                        "Content-Type": "application/json",
                    }
                    url_v3 = f"https://translation.googleapis.com/v3/projects/{GOOGLE_PROJECT}/locations/global:translateText"
                    body_v3 = {
                        "contents": [text[:4000]],
                        "mimeType": "text/plain",
                        "sourceLanguageCode": "en",
                        "targetLanguageCode": "ko",
                    }
                    r3 = requests.post(url_v3, headers=headers, data=json.dumps(body_v3, ensure_ascii=False), timeout=25)
                    if r3.status_code == 200:
                        js3 = r3.json()
                        trans = (js3.get("translations") or [])
                        if trans:
                            return trans[0].get("translatedText", "") or ""
                    else:
                        # 403 (미사용/비활성) 등은 바로 로그 남기고 v2로 폴백
                        msg = r3.text[:500]
                        logger.warning(f"[TR] v3 HTTP {r3.status_code}: {msg}")
                        if r3.status_code == 403 and ("not been used" in msg or "disabled" in msg or "PERMISSION_DENIED" in msg):
                            logger.warning("🔒 v3 권한/활성화 이슈 → v2로 폴백")
                        # 다른 코드도 v2로 폴백
                else:
                    logger.warning("[TR] v3: 잘못된 service_account JSON (type!=service_account)")
            except Exception as e:
                logger.warning(f"[TR] v3 예외: {e}")
        else:
            logger.warning("[TR] v3: service_account JSON 없음 → v2로 폴백")

        # ===== 2) v2 (API Key) =====
        try:
            api_key = _load_google_api_key()
            if api_key:
                url_v2 = "https://translation.googleapis.com/language/translate/v2"
                data_v2 = {"q": text[:4000], "source": "en", "target": "ko", "format": "text"}
                r = requests.post(url_v2, params={"key": api_key}, data=data_v2, timeout=20)
                if r.status_code == 200:
                    js = r.json()
                    translations = (((js or {}).get("data") or {}).get("translations") or [])
                    if translations:
                        return translations[0].get("translatedText", "") or ""
                else:
                    logger.warning(f"[TR] v2(API key) HTTP {r.status_code}: {r.text[:400]}")
        except Exception as e:
            logger.warning(f"[TR] v2(API key) 예외: {e}")

        # ===== 3) v2 (OAuth/Service Account) =====
        if sa_path:
            try:
                info = json.loads(sa_path.read_text(encoding="utf-8"))
                if str(info.get("type", "")) == "service_account":
                    scopes = [
                        "https://www.googleapis.com/auth/cloud-translation",
                        "https://www.googleapis.com/auth/cloud-platform",
                    ]
                    creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
                    creds.refresh(GARequest())
                    headers = {"Authorization": f"Bearer {creds.token}"}
                    url_v2 = "https://translation.googleapis.com/language/translate/v2"
                    data_v2 = {"q": text[:4000], "source": "en", "target": "ko", "format": "text"}
                    r2 = requests.post(url_v2, headers=headers, data=data_v2, timeout=20)
                    if r2.status_code == 200:
                        js2 = r2.json()
                        translations2 = (((js2 or {}).get("data") or {}).get("translations") or [])
                        if translations2:
                            return translations2[0].get("translatedText", "") or ""
                    else:
                        logger.warning(f"[TR] v2(OAuth) HTTP {r2.status_code}: {r2.text[:400]}")
                else:
                    logger.warning("[TR] v2(OAuth): 잘못된 service_account JSON")
            except Exception as e:
                logger.warning(f"[TR] v2(OAuth) 예외: {e}")

        return ""
    except Exception as e:
        logger.warning(f"[TR] 예외: {e}")
        return ""

def _translate_with_papago(text: str) -> str:
    """Papago (env 없으면 빈 문자열 반환)"""
    try:
        import requests
        cid = os.getenv("PAPAGO_CLIENT_ID", "").strip()
        sec = os.getenv("PAPAGO_CLIENT_SECRET", "").strip()
        if not cid or not sec:
            return ""
        url = "https://openapi.naver.com/v1/papago/n2mt"
        headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": sec}
        data = {"source": "en", "target": "ko", "text": text[:4000]}
        r = requests.post(url, headers=headers, data=data, timeout=15)
        if r.status_code == 200:
            return r.json().get("message", {}).get("result", {}).get("translatedText", "") or ""
        logger.warning(f"[TR] Papago HTTP {r.status_code}: {r.text[:300]}")
        return ""
    except Exception:
        return ""

def _translate_with_llm(text: str) -> str:
    try:
        if model is None or tokenizer is None:
            return text
        translate_prompt = (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            "너는 학술 논문 전문 번역가다. 영어 문장을 정확하고 자연스러운 한국어로 번역하라.\n"
            "규칙: (1) 의미 정확 (2) 도메인 용어 보존 (3) 숫자/고유명사/기호 왜곡 금지 (4) 한국어 문장부호\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{text}\n\n[한국어 번역]\n"
            "<|eot_id|>\n"
        )
        inputs = tokenizer(translate_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(MAX_NEW_TOKENS, 600),
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        seq = outputs[0]
        inp_len = inputs["input_ids"].shape[1]
        gen = seq[inp_len:]
        result = tokenizer.decode(gen, skip_special_tokens=True).strip()
        for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
            p = result.find(stop)
            if p != -1:
                result = result[:p].strip()
                break
        result = _postprocess_terms(_normalize_bracket_tokens(result))
        if EASY_STRIP_MATH:
            result = _strip_math_all(result)
        result = _strip_tables_figures_all(result)
        result = _sanitize_repeats(result)
        return result or text
    except Exception as e:
        logger.warning(f"LLM 번역 예외: {e}")
        return text

def _ensure_korean(text: str, *, force_external: bool = False) -> str:
    """한글이 부족하면 외부 번역 사용(옵션), 아니면 원문 유지"""
    if not text or not text.strip():
        return text
    if force_external:
        return _translate_to_korean(text)
    lang = _detect_lang_safe(text)
    if lang == "ko" and not _need_ko(text):
        return text
    return _translate_to_korean(text)


# -------------------- Startup --------------------
@app.on_event("startup")
async def _startup_event():
    load_model()
    # 번역 워밍업 (로그만)
    try:
        probe = "health check"
        ok = _translate_with_google_api(probe) or _translate_with_papago(probe)
        if ok:
            logger.info("🈯 외부 번역 경로 활성(OK)")
        else:
            logger.warning("⚠️ 외부 번역 비활성 → LLM 백업 사용 예정")
    except Exception as e:
        logger.warning(f"번역 워밍업 실패: {e}")

# app.py — Part 3/6
# -------------------- Pydantic I/O --------------------
class TextRequest(BaseModel):
    text: str
    translate: Optional[bool] = False
    style: Optional[str] = Field(
        default="three_para_ko",
        description="easy 스타일: 'three_para_ko' (기본, 한글 3문단) | 'one_sentence_en' (영어 1문장)"
    )
    model_config = ConfigDict(extra="allow")

class TextResponse(BaseModel):
    simplified_text: str
    translated_text: Optional[str] = None  # 호환성

class BatchRequest(BaseModel):
    paper_id: str = Field(..., description="결과 파일/경로 식별자")
    chunks_jsonl: str = Field(..., description="JSONL 내용 문자열 또는 경로(파일/디렉토리/tex) — Part 4~6에서 처리")
    output_dir: str = Field(..., description="결과 저장 루트 (JSON+HTML 출력)")
    style: Optional[str] = Field(default="three_para_ko", description="easy 스타일 (three_para_ko 권장)")

class VizResult(BaseModel):
    ok: bool = True
    index: int
    image_path: Optional[str] = None
    error: Optional[str] = None
    easy_text: Optional[str] = None
    section_title: Optional[str] = None
    section_type: Optional[str] = None
    original_images: Optional[List[Dict[str, Any]]] = None

class BatchResult(BaseModel):
    ok: bool
    paper_id: str
    count: int
    success: int
    failed: int
    out_dir: str
    images: List[VizResult]

# -------------------- 루트/헬스 --------------------
@app.get("/")
async def root():
    return {"message": "POLO Easy Model API (Final KR)", "model": BASE_MODEL, "mode": "JSON+HTML"}

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
        "torch": torch.__version__,
        "cuda": (torch.version.cuda if torch.cuda.is_available() else None),
        "dtype": str(getattr(getattr(model, "dtype", None), "name", getattr(model, "dtype", None))),
    }

@app.get("/healthz")
async def healthz():
    return await health()

# -------------------- /easy (단일 섹션 변환) --------------------
@app.post("/easy", response_model=TextResponse)
async def simplify_text(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    style = (request.style or "three_para_ko")
    simplified_text = await _rewrite_text(request.text, style=style)
    if style != "one_sentence_en" and _need_ko(simplified_text):
        simplified_text = _ensure_korean(simplified_text)
    return TextResponse(simplified_text=simplified_text, translated_text=None)

# -------------------- /generate (섹션 추출 + 스키마 JSON 생성) --------------------
def _coerce_json(text: str) -> Dict[str, Any]:
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s: text = text[s:e+1]
    return json.loads(text)

def _merge_with_schema(data: dict, schema: dict) -> dict:
    if not isinstance(data, dict): return json.loads(json.dumps(schema))
    out = json.loads(json.dumps(schema))  # deep copy
    for k, v in data.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_with_schema(v, out[k])
        else:
            out[k] = v
    return out

# app.py — Part 4/6
# ---------- LaTeX 표 간단 추출 ----------
def _extract_table_data(text: str) -> List[dict]:
    tables = []
    pat = r'\\begin\{tabular\}[^{]*\{([^}]+)\}(.*?)\\end\{tabular\}'
    for m in re.finditer(pat, text, re.DOTALL):
        content = m.group(2)
        lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith('\\hline')]
        if len(lines) < 2:
            continue
        headers = [h.strip().rstrip('\\') for h in lines[0].split('&')]
        rows = []
        for ln in lines[1:]:
            if '&' in ln:
                row = [c.strip().rstrip('\\') for c in ln.split('&')]
                if len(row) == len(headers):
                    rows.append(row)
        if rows:
            tables.append({"type": "metric_table", "headers": headers, "rows": rows})
    return tables

# ---------- LaTeX 섹션 파서 ----------
def _parse_latex_sections(tex_path: Path) -> List[dict]:
    content = tex_path.read_text(encoding="utf-8", errors="ignore")
    sections: List[dict] = []
    cur_title, cur_buf, cur_raw = None, [], []
    section_type = "section"

    def _flush():
        if cur_title is None:
            return
        raw_txt = "\n".join(cur_raw).strip()
        clean = _clean_latex_text("\n".join(cur_buf))
        sections.append({
            "index": len(sections),
            "title": cur_title,
            "content": clean,
            "raw_content": raw_txt,
            "table_data": _extract_table_data(raw_txt),
            "section_type": section_type,
        })

    sec_pat  = re.compile(r"^\s*\\section\*?\{([^}]+)\}\s*$")
    sub_pat  = re.compile(r"^\s*\\subsection\*?\{([^}]+)\}\s*$")
    abs_beg  = re.compile(r"^\s*\\begin\{abstract\}")
    abs_end  = re.compile(r"^\s*\\end\{abstract\}")

    for ln in content.splitlines():
        if abs_beg.match(ln):
            _flush(); cur_title="Abstract"; cur_buf, cur_raw=[], []; section_type="section"; continue
        if abs_end.match(ln):
            _flush(); cur_title=None; cur_buf, cur_raw=[], []; continue
        m1 = sec_pat.match(ln); m2 = sub_pat.match(ln)
        if m1:
            _flush(); cur_title=m1.group(1).strip(); cur_buf, cur_raw=[], []; section_type="section"; continue
        if m2:
            _flush(); cur_title=m2.group(1).strip(); cur_buf, cur_raw=[], []; section_type="subsection"; continue
        if cur_title is None:
            if not any(k in ln.lower() for k in ["\\title","\\author","\\date","\\maketitle"]):
                cur_title="Introduction"; section_type="section"
        if cur_title is not None:
            cur_buf.append(ln); cur_raw.append(ln)

    _flush()
    if not sections:
        clean = _clean_latex_text(content)
        sections = [{
            "index": 0, "title": "Full Document", "content": clean, "raw_content": content,
            "table_data": _extract_table_data(content), "section_type": "section",
        }]
    logger.info(f"[EASY] Parsed {len(sections)} sections from LaTeX")
    return sections

# ---------- assets.jsonl 로더 ----------
def _load_assets_metadata(source_dir: Path) -> Dict[str, Any]:
    assets_file = source_dir / "assets.jsonl"
    if not assets_file.exists():
        return {}
    assets: Dict[str, Any] = {}
    try:
        with open(assets_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    asset = json.loads(line)
                    if asset.get("kind") == "figure" and asset.get("graphics"):
                        assets[asset["id"]] = asset
    except Exception as e:
        logger.warning(f"[EASY] assets.jsonl 로드 실패: {e}")
    return assets

# ---------- JSONL → 섹션 ----------
def _append_from_jsonl_lines(lines: List[str]) -> List[dict]:
    sections: List[dict] = []
    for idx, ln in enumerate(lines):
        try:
            obj = json.loads(ln)
        except Exception as e:
            obj = {"title": f"Section {idx+1}", "text": f"[JSONL parse error] {e}"}
        sections.append({
            "index": idx,
            "title": obj.get("title") or f"Section {idx+1}",
            "content": obj.get("text") or obj.get("content") or "",
            "raw_content": obj.get("text") or obj.get("content") or "",
            "table_data": [],
            "section_type": "section",
        })
    return sections

# ---------- HTML utils ----------
def _get_current_datetime() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")

def _slugify(s: str, fallback: str) -> str:
    s = re.sub(r"[^0-9A-Za-z가-힣\- ]", "", s or "")
    s = s.strip().replace(" ", "-")
    return s if s else fallback

def _dedup_titles(sections: List[dict]) -> List[dict]:
    seen: Dict[str, int] = {}
    out: List[dict] = []
    for s in sections:
        s = dict(s)
        t = (s.get("title") or "").strip() or "Section"
        c = seen.get(t, 0) + 1
        seen[t] = c
        s["title"] = f"{t} ({c})" if c > 1 else t
        out.append(s)
    return out

def _slugify_unique(titles: List[str]) -> List[str]:
    used: set = set()
    slugs: List[str] = []
    for t in titles:
        base = _slugify(t, "sec")
        cand = base; k = 2
        while cand in used:
            cand = f"{base}-{k}"; k += 1
        used.add(cand); slugs.append(cand)
    return slugs

def _md_to_html(md: str) -> str:
    if not md: return ""
    def _codeblock_repl(m): return f"<pre><code>{html.escape(m.group(1))}</code></pre>"
    md = re.sub(r"```([\s\S]*?)```", _codeblock_repl, md)
    md = re.sub(r"^###\s*(.+)$", r"<h3>\1</h3>", md, flags=re.MULTILINE)
    md = re.sub(r"^##\s*(.+)$",  r"<h2>\1</h2>", md, flags=re.MULTILINE)
    md = re.sub(r"^#\s*(.+)$",   r"<h1>\1</h1>", md, flags=re.MULTILINE)
    md = re.sub(r"(?<!\$)\*\*([^*$]+?)\*\*(?!\$)", r"<strong>\1</strong>", md)
    md = re.sub(r"(?<!\$)\*([^*$]+)\*(?!\$)",     r"<em>\1</em>", md)
    md = re.sub(r"`([^`]+?)`",                    r"<code>\1</code>", md)
    md = re.sub(r"==([^=]+)==",                   r"<mark>\1</mark>", md)
    md = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" target="_blank">\1</a>', md)
    md = re.sub(r"(https?://[^\s<>\"']+)", r'<a href="\1" target="_blank">\1</a>', md)
    lines = md.splitlines(); out, in_ul = [], False
    for ln in lines:
        if re.match(r"^\s*[-•]\s+.+", ln):
            if not in_ul: out.append("<ul>")
            in_ul = True; out.append("<li>"+re.sub(r"^\s*[-•]\s+", "", ln).strip()+"</li>")
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            out.append(ln)
    if in_ul: out.append("</ul>")
    html_txt = "\n".join(out)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", html_txt) if b.strip()]
    wrapped = []
    for b in blocks:
        if b.startswith(("<h1","<h2","<h3","<ul","<pre","<table","<blockquote")):
            wrapped.append(b)
        else:
            wrapped.append(f"<p>{b}</p>")
    return "\n".join(wrapped)

def _render_rich_html(text: str) -> str:
    if not text: return ""
    html_text = _md_to_html(text)
    html_text = re.sub(r'\\textbf\{([^}]+)\}', r'<strong>\1</strong>', html_text)
    html_text = re.sub(r'\\textit\{([^}]+)\}', r'<em>\1</em>', html_text)
    html_text = re.sub(r'\\(sub)*section\{([^}]+)\}', r'<h2>\2</h2>', html_text)
    html_text = re.sub(r' +', ' ', html_text)
    return html_text

def _starts_with_same_heading(html_txt: str, title: str) -> bool:
    if not title or not html_txt: return False
    plain = re.sub(r"<[^>]+>", "", html_txt).strip().lower()
    t = (title or "").strip().lower()
    return plain.startswith(t) or plain[:120].startswith(t + ":")

def _split_paragraphs_ko(text: str, min_s: int = 3, max_s: int = 5, max_chars: int = 700) -> str:
    if not text: return ""
    s = re.sub(r"\s+", " ", str(text).strip())
    parts = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+", s)
    parts = [p.strip() for p in parts if p and p.strip()]
    paras: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for sent in parts:
        start_new = (len(cur) >= min_s and (len(cur) >= max_s or cur_len + len(sent) > max_chars))
        if start_new and cur:
            paras.append(" ".join(cur))
            cur = [sent]; cur_len = len(sent)
        else:
            cur.append(sent); cur_len += len(sent)
    if cur: paras.append(" ".join(cur))
    out: List[str] = []
    for p in paras:
        if out and len(p) < 80:
            out[-1] = (out[-1] + " " + p).strip()
        else:
            out.append(p)
    return "\n\n".join(out)

def _save_html_results(sections: List[dict], results: List['VizResult'], output_path: Path, paper_id: str):
    sections = _dedup_titles(list(sections))
    titles = [(sec.get("title") or f"Section {i+1}").strip() for i, sec in enumerate(sections)]
    slugs = _slugify_unique(titles)
    gen_at = _get_current_datetime()

    html_header = """<!DOCTYPE html><html lang="ko"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>POLO - 쉬운 논문 설명</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
<style>/* (스타일 생략 없이 그대로) */</style>
<script>/*(스크립트 생략 없이 그대로)*/</script>
</head><body><div class="wrapper">
  <div class="header">
    <div class="title">POLO 논문 이해 도우미 변환 결과</div>
    <div class="subtitle">복잡한 논문을 쉽게 이해할 수 있도록 재구성한 결과</div>
    <div class="meta">논문 ID: __PAPER_ID__ | 생성: __GEN_AT__</div>
  </div>
"""
    html_parts: List[str] = [html_header.replace("__PAPER_ID__", paper_id).replace("__GEN_AT__", gen_at)]
    html_parts.append('<div class="layout">')

    # TOC
    html_parts.append('<aside class="toc-sidebar"><div class="toc-title">목차</div><ol class="toc-list">')
    section_num = 0; open_sub = False; subsection_num = 0
    for i, sec in enumerate(sections):
        title = (sec.get("title") or f"Section {i+1}").strip()
        sid = slugs[i]
        section_type = sec.get("section_type", "section")
        if section_type == "section":
            if open_sub:
                html_parts.append('</ol></li>'); open_sub = False
            section_num += 1; subsection_num = 0
            html_parts.append(f'<li class="toc-item"><a class="toc-link" href="#{sid}"><span class="num">{section_num}.</span>{html.escape(title)}</a>')
            html_parts.append('<ol class="toc-sublist">'); open_sub = True
        else:
            subsection_num += 1
            html_parts.append(f'<li class="toc-item"><a class="toc-link" href="#{sid}"><span class="num">{section_num}.{subsection_num}</span>{html.escape(title)}</a></li>')
    if open_sub:
        html_parts.append('</ol></li>')
    html_parts.append('</ol></aside>')

    # 본문
    html_parts.append('<main class="content-area">')
    for i, sec in enumerate(sections):
        title = (sec.get("title") or f"Section {i+1}").strip()
        sid = slugs[i]
        section_type = sec.get("section_type", "section")
        res = results[i]
        raw_text = (getattr(res, "easy_text", None) or sec.get("content") or "").strip()
        processed_text = _split_paragraphs_ko(raw_text)
        content_html = _render_rich_html(processed_text)
        header_html = "" if _starts_with_same_heading(content_html, title) else f"<h2>{html.escape(title)}</h2>"
        html_parts.append(f'<div class="section-card {section_type}" id="{sid}">{header_html}<div class="content">{content_html}</div>')

        # 생성된 시각화 이미지
        if res.ok and res.image_path and Path(res.image_path).exists():
            src_path = Path(res.image_path)
            dst_path = output_path.parent / src_path.name
            try:
                import shutil; shutil.copy2(src_path, dst_path)
                logger.info(f"📊 [EASY] 시각화 이미지 복사 완료: {src_path.name}")
            except Exception as e:
                logger.warning(f"📊 [EASY] 시각화 이미지 복사 실패: {e}"); dst_path = src_path
            html_parts.append(f"""
<div class="image-container">
  <img src="{dst_path.name}" alt="{html.escape(title)} 관련 시각화" style="max-width:100%; height:auto; border-radius:8px;" />
  <div class="image-caption">그림 {i+1}: {html.escape(title)} 관련 시각화</div>
</div>""")

        # 원본 논문 이미지
        if res.original_images:
            for img_idx, img_info in enumerate(res.original_images):
                try:
                    src_path = Path(img_info["path"])
                    dst_path = output_path.parent / img_info["filename"]
                    import shutil; shutil.copy2(src_path, dst_path)
                    logger.info(f"📊 [EASY] 원본 이미지 복사 완료: {img_info['filename']}")
                    caption = img_info.get("caption", f"원본 논문 그림 {img_idx+1}")
                    html_parts.append(f"""
<div class="image-container">
  <img src="{img_info['filename']}" alt="{html.escape(caption)}" style="max-width:100%; height:auto; border-radius:8px;" />
  <div class="image-caption">원본 논문 그림 {img_idx+1}: {html.escape(caption)}</div>
</div>""")
                except Exception as e:
                    logger.warning(f"📊 [EASY] 원본 이미지 복사 실패: {e}")
        html_parts.append("</div>")  # section-card

    html_parts.append("""
<div class="footer-actions">
  <button class="btn" onclick="downloadHTML()">HTML 저장</button>
  <button class="btn secondary" id="toggleHi" onclick="toggleHighlights()">형광펜 끄기</button>
</div>
""")
    html_parts.append('</main></div>')
    html_parts.append("<script>if(window.__attachObserver) window.__attachObserver();</script>")
    html_parts.append("</div></body></html>")

    output_path.write_text("".join(html_parts), encoding="utf-8")

# app.py — Part 5/6
# ---------- 시각화 엔진 호출 ----------
def _httpx_client(timeout_total: float = 1200.0) -> httpx.AsyncClient:
    try:
        to = httpx.Timeout(timeout_total)
    except TypeError:
        to = timeout_total
    return httpx.AsyncClient(
        timeout=to,
        http2=False,
        headers={"Connection": "close"},
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=20),
        follow_redirects=False,
    )

class _VizRes(VizResult):
    pass

async def _send_to_viz(
    paper_id: str,
    index: int,
    easy_text_ko: str,
    out_dir: Path,
    table_data: List[dict] = None,
    *,
    health_checked: bool = False
) -> VizResult:
    if not EASY_VIZ_ENABLED:
        return _VizRes(ok=True, index=index, image_path=None, easy_text=easy_text_ko)

    if not health_checked:
        try:
            async with _httpx_client(EASY_VIZ_HEALTH_TIMEOUT) as client:
                r = await client.get(f"{VIZ_MODEL_URL.rstrip('/')}/health")
                if r.status_code != 200:
                    return _VizRes(ok=True, index=index, image_path=None, easy_text=easy_text_ko)
        except Exception:
            return _VizRes(ok=True, index=index, image_path=None, easy_text=easy_text_ko)

    payload = {
        "paper_id": paper_id,
        "index": index,
        "rewritten_text": easy_text_ko,
        "target_lang": "ko",
        "bilingual": "missing",
        "text_type": "easy_korean",
    }
    if table_data:
        payload["tables"] = table_data

    try:
        async with _httpx_client(EASY_VIZ_TIMEOUT) as client:
            r = await client.post(f"{VIZ_MODEL_URL.rstrip('/')}/viz", json=payload)
            if r.status_code != 200:
                return _VizRes(ok=True, index=index, image_path=None, easy_text=easy_text_ko)
            data = r.json()
            if data.get("image_base64"):
                img_path = out_dir / f"{index:06d}.png"
                img_path.write_bytes(base64.b64decode(data["image_base64"]))
                return _VizRes(ok=True, index=index, image_path=str(img_path), easy_text=easy_text_ko)
            if data.get("image_path"):
                return _VizRes(ok=True, index=index, image_path=data["image_path"], easy_text=easy_text_ko)
            return _VizRes(ok=True, index=index, image_path=None, easy_text=easy_text_ko)
    except Exception as e:
        logger.warning(f"[EASY] viz error: {e}")
        return _VizRes(ok=False, index=index, image_path=None, easy_text=easy_text_ko, error=str(e))

# ---------- 배치 엔드포인트 ----------
@app.post("/batch", response_model=BatchResult)
async def run_batch(request: BatchRequest, assets_metadata: Optional[Dict[str, Any]] = None):
    t_batch_start = time.time()
    paper_id = request.paper_id.strip()
    base_out = Path(request.output_dir).expanduser().resolve()
    out_dir  = base_out / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[EASY] output dir: {out_dir}")

    # 입력 소스 파싱
    src = (request.chunks_jsonl or "").strip()
    src_path: Optional[Path] = None
    if src:
        p = Path(src)
        if p.exists(): src_path = p.resolve()

    sections: List[dict] = []
    try:
        if src_path is None:
            lines = [ln for ln in src.splitlines() if ln.strip()]
            sections = _append_from_jsonl_lines(lines)
        else:
            if src_path.is_dir():
                tex = src_path / "merged_body.tex"
                if tex.exists():
                    sections = _parse_latex_sections(tex)
                else:
                    hits = sorted(src_path.rglob("*.jsonl"))
                    if not hits:
                        raise ValueError("디렉토리에 merged_body.tex 또는 *.jsonl 없음")
                    lines = hits[0].read_text(encoding="utf-8", errors="ignore").splitlines()
                    sections = _append_from_jsonl_lines([ln for ln in lines if ln.strip()])
            else:
                suf = src_path.suffix.lower()
                if suf in (".jsonl", ".ndjson"):
                    lines = src_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    sections = _append_from_jsonl_lines([ln for ln in lines if ln.strip()])
                elif suf == ".gz":
                    with gzip.open(src_path, "rt", encoding="utf-8") as f:
                        lines = [ln for ln in f if ln.strip()]
                    sections = _append_from_jsonl_lines(lines)
                elif suf == ".tex":
                    sections = _parse_latex_sections(src_path)
                else:
                    raise ValueError("지원하지 않는 입력 형식 (jsonl/jsonl.gz/tex/dir)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"입력 파싱 실패: {e}")

    if not sections:
        raise HTTPException(status_code=400, detail="처리할 섹션 없음")

    # 이미지 자산 로드
    if assets_metadata is None:
        assets_metadata = {}
        if src_path and src_path.is_dir():
            assets_metadata = _load_assets_metadata(src_path)
        elif src_path and src_path.suffix.lower() == ".tex":
            assets_metadata = _load_assets_metadata(src_path.parent)
    logger.info(f"[EASY] 이미지 자산 수: {len(assets_metadata)}")

    # viz health 1회 확인
    viz_health_ok = False
    if EASY_VIZ_ENABLED:
        try:
            async with _httpx_client(EASY_VIZ_HEALTH_TIMEOUT) as client:
                r = await client.get(f"{VIZ_MODEL_URL.rstrip('/')}/health")
                viz_health_ok = (r.status_code == 200)
        except Exception:
            viz_health_ok = False

    sem = anyio.Semaphore(EASY_CONCURRENCY)
    results: List[VizResult] = [VizResult(ok=False, index=i) for i in range(len(sections))]
    section_times_ms: List[float] = [0.0] * len(sections)
    error_briefs: List[str] = []

    def _find_related_images(section_content: str, assets_metadata: Dict[str, Any], source_dir: Path) -> List[Dict[str, Any]]:
        related_images = []
        fig_refs = []
        fig_refs.extend(re.findall(r'\\ref\{([^}]+)\}', section_content))
        fig_refs.extend(re.findall(r'⟨FIG:([^⟩]+)⟩', section_content))
        fig_refs.extend(re.findall(r'[?쭲]IG:([^?]+)', section_content))
        fig_refs.extend(re.findall(r'[?쭲]IG:fig:([^?]+)', section_content))
        fig_refs.extend(re.findall(r'[?쭲]IG:([^?]+)??', section_content))

        try:
            server_dir = Path(__file__).resolve().parents[2] / "server"
        except Exception:
            server_dir = Path(".")
        extra_roots: List[Path] = [
            source_dir,
            server_dir / "data" / "out" / "transformer" / "source",
            server_dir / "data" / "out" / "transformer",
            server_dir / "data" / "arxiv" / "2008.01686" / "source",
            server_dir / "data" / "arxiv",
        ]
        allowed_exts = ['.pdf', '.jpg', '.jpeg', '.png', '.eps']

        for ref in fig_refs:
            if ref in assets_metadata:
                asset = assets_metadata[ref]
                for graphic_path in asset.get("graphics", []):
                    candidates: List[Path] = []
                    for root in extra_roots:
                        p = root / graphic_path
                        candidates.append(p)
                        if not p.suffix:
                            for ext in allowed_exts:
                                candidates.append(root / f"{graphic_path}{ext}")
                    base = Path(graphic_path).name
                    base_wo = base.rsplit('.', 1)[0] if '.' in base else base
                    for root in extra_roots:
                        for ext in allowed_exts:
                            candidates.extend(root.rglob(f"{base_wo}{ext}"))

                    full_path = next((c for c in candidates if isinstance(c, Path) and c.exists()), None)
                    if full_path and full_path.exists():
                        related_images.append({
                            "id": asset["id"],
                            "path": str(full_path),
                            "caption": asset.get("caption", ""),
                            "label": asset.get("label", ""),
                            "filename": full_path.name
                        })
        return related_images

    async def _work(i: int, section: dict):
        async with sem:
            total = len(sections)
            title = section.get("title") or f"Section {i+1}"
            logger.info(f"[EASY] ▶ [{i+1}/{total}] section START: {title}")
            sec_t0 = time.time()
            try:
                context_info = f"이 섹션은 전체 {total}개 중 {i+1}번째입니다. "
                if i > 0:
                    prev_title = sections[i-1].get("title", "이전 섹션")
                    context_info += f"이전: {prev_title}. "
                if i < total - 1:
                    next_title = sections[i+1].get("title", "다음 섹션")
                    context_info += f"다음: {next_title}. "
                if section.get("section_type","section") == "subsection":
                    context_info += "이것은 서브섹션으로, 상위 내용을 세부적으로 다룹니다. "
                else:
                    context_info += "이것은 메인 섹션입니다. "

                easy_text = await _rewrite_text(
                    section.get("content",""),
                    title,
                    context_info,
                    style=(request.style or "three_para_ko")
                )
                easy_text = _ensure_korean(easy_text)
                logger.info(f"[EASY] ✔ text rewritten for section {i+1}")
            except Exception as e:
                msg = f"rewrite 실패: {e}"
                logger.error(f"[EASY] ❌ 섹션 변환 실패 idx={i}: {e}")
                results[i] = VizResult(ok=False, index=i, error=msg, section_title=title)
                error_briefs.append(f"[{i+1}] {title}: {e}")
                section_times_ms[i] = (time.time() - sec_t0) * 1000.0
                return

            section_table_data = sections[i].get('table_data', []) if i < total else []
            related_images = []
            if src_path:
                source_dir = src_path if src_path.is_dir() else src_path.parent
                related_images = _find_related_images(section.get("content",""), assets_metadata, source_dir)
                if related_images:
                    logger.info(f"[EASY] 섹션 {i+1}에서 {len(related_images)}개 이미지 발견")

            viz_res = await _send_to_viz(
                paper_id, i, easy_text, out_dir, section_table_data,
                health_checked=viz_health_ok
            )
            viz_res.section_title = title
            viz_res.section_type = section.get("section_type","section")
            if viz_res.ok and not viz_res.easy_text:
                viz_res.easy_text = easy_text
            viz_res.original_images = related_images
            results[i] = viz_res

            sec_elapsed = time.time() - sec_t0
            section_times_ms[i] = sec_elapsed * 1000.0
            has_img = "img" if (viz_res.ok and viz_res.image_path) else "no-img"
            logger.info(f"[EASY] ⏹ [{i+1}/{total}] section DONE: {title} elapsed={sec_elapsed:.2f}s {has_img}")

    timed_out = False
    try:
        if EASY_BATCH_TIMEOUT and EASY_BATCH_TIMEOUT > 0:
            with anyio.fail_after(EASY_BATCH_TIMEOUT):
                for i, sec in enumerate(sections):
                    await _work(i, sec)
        else:
            for i, sec in enumerate(sections):
                await _work(i, sec)
    except (anyio.exceptions.TimeoutError, TimeoutError):
        timed_out = True
        logger.error(f"[EASY] ⏱ batch timed out after {EASY_BATCH_TIMEOUT}s — partial results will be saved")

    success = sum(1 for r in results if r.ok)
    failed  = len(sections) - success
    success_rate = (success / max(1, len(sections))) * 100.0
    status_str = "ok" if (not timed_out and failed == 0) else "partial"

    easy_json = {
        "paper_id": paper_id,
        "status": status_str,
        "count": len(sections),
        "success": success,
        "failed": failed,
        "success_rate": round(success_rate, 2),
        "sections": [
            {
                "index": i,
                "title": sections[i].get("title"),
                "original_content": sections[i].get("content"),
                "korean_translation": (results[i].easy_text or ""),
                "image_path": (results[i].image_path or ""),
                "original_images": (results[i].original_images or []),
                "status": "ok" if results[i].ok else f"error: {results[i].error}",
                "section_type": results[i].section_type or sections[i].get("section_type","section"),
            }
            for i in range(len(sections))
        ],
        "generated_at": _get_current_datetime(),
        "timed_out": timed_out,
        "elapsed_ms": round((time.time() - t_batch_start) * 1000.0, 2),
    }
    (out_dir / "easy_results.json").write_text(json.dumps(easy_json, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[EASY] saved JSON: {out_dir / 'easy_results.json'}")

    # HTML 저장(비치명적 실패 허용)
    try:
        html_path = out_dir / "easy_results.html"
        _save_html_results(sections, results, html_path, paper_id)
        (out_dir / "index.html").write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"[EASY] saved HTML: {html_path}")
    except Exception as e:
        logger.warning(f"[EASY] HTML save skipped (non-fatal): {e}")

    return BatchResult(
        ok=(status_str == "ok"),
        paper_id=paper_id,
        count=len(sections),
        success=success,
        failed=failed,
        out_dir=str(out_dir),
        images=results,
    )
# app.py — Part 6/6

# ---------- JSON 스키마 요약 (/generate) ----------
@app.post("/generate")
async def generate_json(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

    # 섹션 간단 추출
    def _extract_sections(src: str) -> dict:
        sections = {k: "" for k in ["abstract","introduction","methods","results","discussion","conclusion"]}
        headers = [
            ("abstract", r"^\s*abstract\b"),
            ("introduction", r"^\s*introduction\b"),
            ("methods", r"^\s*methods?\b|^\s*materials?\s+and\s+methods\b"),
            ("results", r"^\s*results?\b"),
            ("discussion", r"^\s*discussion\b"),
            ("conclusion", r"^\s*conclusion[s]?\b|^\s*concluding\s+remarks\b"),
        ]
        lines = src.splitlines(); idxs = []
        for i, line in enumerate(lines):
            for key, pat in headers:
                if re.match(pat, line.strip(), flags=re.IGNORECASE):
                    idxs.append((i, key)); break
        idxs.sort()
        for j, (start_i, key) in enumerate(idxs):
            end_i = idxs[j+1][0] if j+1 < len(idxs) else len(lines)
            chunk = "\n".join(lines[start_i+1:end_i]).strip()[:2000]
            if EASY_STRIP_MATH: chunk = _strip_math_all(chunk)
            chunk = _strip_tables_figures_all(chunk)
            sections[key] = chunk
        return sections

    extracted = _extract_sections(request.text)

    # 기본 스키마
    SCHEMA = {
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
    schema_copy = json.loads(json.dumps(SCHEMA))
    for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
        schema_copy[k]["original"] = extracted.get(k, "")

    instruction = (
        "너는 과학 선생님이다. 아래 JSON 스키마의 '키/구조'를 그대로 두고 '값'만 채워라. "
        "외부 지식 금지. 원문에 없으면 '원문에 없음'이라고 적어라. "
        "출력은 반드시 '유효한 JSON' 하나만."
    )
    schema_str = json.dumps(schema_copy, ensure_ascii=False, indent=2)
    context_str = json.dumps(extracted, ensure_ascii=False, indent=2)
    prompt = f"{instruction}\n\n=== 스키마 ===\n{schema_str}\n\n=== 원문 ===\n{context_str}\n\n[OUTPUT]\n"

    # LLM 실행
    inputs = _safe_tokenize(prompt, max_len=MAX_INPUT_TOKENS)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    seq = outputs[0]
    inp_len = inputs["input_ids"].shape[1]
    gen_tokens = seq[inp_len:]
    raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
        p = raw.find(stop)
        if p != -1: raw = raw[:p].strip(); break

    # JSON 보정
    try:
        data = _coerce_json(raw)
    except Exception:
        data = schema_copy
    data = _merge_with_schema(data, schema_copy)

    # 후처리 + 한글 강제
    for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
        try:
            easy_val = (data.get(k, {}) or {}).get("easy", "")
            if isinstance(easy_val, str):
                t = _postprocess_terms(_normalize_bracket_tokens(easy_val))
                if EASY_STRIP_MATH: t = _strip_math_all(t)
                t = _strip_tables_figures_all(t)
                t = _auto_bold_terms(t)
                t = _hilite_sentences(t, max_marks=2)
                t = _sanitize_repeats(t)
                if _need_ko(t): t = _ensure_korean(t)
                data[k]["easy"] = t
        except Exception:
            pass

    data["processing_info"] = {"gpu_used": gpu_available, "max_new_tokens": MAX_NEW_TOKENS}
    return data

# ---------- 파일 경유 실행 (/from-transport) ----------
class TransportRequest(BaseModel):
    paper_id: str
    transport_path: str
    output_dir: Optional[str] = None
    style: Optional[str] = Field(default="three_para_ko")

ALLOWED_DATA_ROOT = Path(os.getenv("EASY_ALLOWED_ROOT", ".")).resolve()
TRANSPORT_MAX_MB  = int(os.getenv("EASY_TRANSPORT_MAX_MB", "50"))

def _is_under_allowed_root(p: Path) -> bool:
    try: return str(p.resolve()).startswith(str(ALLOWED_DATA_ROOT))
    except Exception: return False

def _read_text_guarded(p: Path, encoding="utf-8") -> str:
    if p.stat().st_size > TRANSPORT_MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다(>{TRANSPORT_MAX_MB}MB): {p.name}")
    return p.read_text(encoding=encoding, errors="ignore")

def _json_to_jsonl(s: str) -> str:
    try: obj = json.loads(s)
    except Exception as e: raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {e}")
    lines = []
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                lines.append(json.dumps({"title": item.get("title", f"Section {i+1}"), "text": item.get("text","")}, ensure_ascii=False))
            else:
                lines.append(json.dumps({"title": f"Section {i+1}", "text": str(item)}, ensure_ascii=False))
    elif isinstance(obj, dict):
        lines.append(json.dumps({"title": obj.get("title","Section 1"), "text": obj.get("text","")}, ensure_ascii=False))
    else:
        lines.append(json.dumps({"title": "Section 1", "text": str(obj)}, ensure_ascii=False))
    return "\n".join(lines)

@app.post("/from-transport", response_model=BatchResult)
async def from_transport(request: TransportRequest):
    tpath = Path(request.transport_path).expanduser().resolve()
    if not _is_under_allowed_root(tpath):
        raise HTTPException(status_code=403, detail="허용되지 않은 경로입니다")
    if not tpath.exists():
        raise HTTPException(status_code=404, detail=f"파일 없음: {tpath}")

    base_out = Path(request.output_dir).expanduser().resolve() if request.output_dir else tpath.parent
    assets_metadata = {}
    if tpath.is_dir(): assets_metadata = _load_assets_metadata(tpath)
    elif tpath.suffix.lower() == ".tex": assets_metadata = _load_assets_metadata(tpath.parent)

    content: Optional[str] = None
    if tpath.is_dir():
        tex = tpath / "merged_body.tex"
        if tex.exists():
            secs = _parse_latex_sections(tex)
            content = "\n".join(json.dumps({"title": s["title"], "text": s["content"]}, ensure_ascii=False) for s in secs)
        else:
            hits = sorted(tpath.rglob("*.jsonl")) or sorted(tpath.rglob("*.jsonl.gz"))
            if not hits: raise HTTPException(status_code=400, detail="처리 가능한 파일 없음")
            if hits[0].suffix == ".gz":
                with gzip.open(hits[0], "rt", encoding="utf-8") as f: content = f.read()
            else:
                content = _read_text_guarded(hits[0])
    else:
        ext = tpath.suffix.lower()
        if ext in (".jsonl",".ndjson"): content = _read_text_guarded(tpath)
        elif ext == ".gz": 
            with gzip.open(tpath, "rt", encoding="utf-8") as f: content = f.read()
        elif ext == ".tex":
            secs = _parse_latex_sections(tpath)
            content = "\n".join(json.dumps({"title": s["title"], "text": s["content"]}, ensure_ascii=False) for s in secs)
        elif ext == ".json": content = _json_to_jsonl(_read_text_guarded(tpath))
        else: raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {tpath.name}")

    req = BatchRequest(paper_id=request.paper_id, chunks_jsonl=content or "", output_dir=str(base_out), style=request.style)
    return await run_batch(req, assets_metadata=assets_metadata)

# ---------- Run ----------
if __name__ == "__main__":
    host = os.getenv("EASY_HOST", "0.0.0.0")
    port = int(os.getenv("EASY_PORT", "5003"))
    reload_flag = os.getenv("EASY_RELOAD", "false").lower() in ("1","true","yes")
    uvicorn.run("app:app", host=host, port=port, reload=reload_flag, timeout_keep_alive=120)
