# -*- coding: utf-8 -*-
"""
POLO Easy Model - Grounded JSON/HTML Generator (Final, KR-optimized + AI Glossary)
- 한글 결과 강제, 환각 방지, 스키마/형식 보호, 반복/잡음 정리, 안전 토크나이즈
- 영문 용어 유지 + (쉬운 한국어 풀이) 자동 주석 (glossary_ai.json 기반)
- 수식/코드/URL 고유 마스킹 후 복원 → 수식 절대 파손 금지
- 선로딩/진행 로그 강화, 구버전 httpx 호환, Windows HF 캐시 안전화
"""

from __future__ import annotations
import os, re, json, time, base64, gzip, sys, asyncio, logging, html
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Pattern

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
ADAPTER_DIR              = os.getenv("EASY_ADAPTER_DIR", str(Path(__file__).resolve().parent.parent / "fine-tuning" / "outputs" / "yolo-easy-qlora" / "checkpoint-200"))
MAX_NEW_TOKENS           = int(os.getenv("EASY_MAX_NEW_TOKENS", "1200"))  # 600 → 1200으로 복원
EASY_CONCURRENCY         = int(os.getenv("EASY_CONCURRENCY", "4"))  # 2 → 4로 증가
EASY_BATCH_TIMEOUT       = int(os.getenv("EASY_BATCH_TIMEOUT", "1800"))
EASY_VIZ_TIMEOUT         = float(os.getenv("EASY_VIZ_TIMEOUT", "1800"))
EASY_VIZ_HEALTH_TIMEOUT  = float(os.getenv("EASY_VIZ_HEALTH_TIMEOUT", "5"))
EASY_VIZ_ENABLED         = os.getenv("EASY_VIZ_ENABLED", "0").lower() in ("1","true","yes")
VIZ_MODEL_URL            = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")

# ✅ 기본을 "수식 보존"으로 변경 (원하면 .env 에서 EASY_STRIP_MATH=1 로 테이블/수식 제거 가능)
EASY_STRIP_MATH          = os.getenv("EASY_STRIP_MATH", "0").lower() in ("1","true","yes")
EASY_FORCE_KO            = os.getenv("EASY_FORCE_KO", "1").lower() in ("1","true","yes")
EASY_AUTO_BOLD           = os.getenv("EASY_AUTO_BOLD", "1").lower() in ("1","true","yes")
EASY_HILITE              = os.getenv("EASY_HILITE", "1").lower() in ("1","true","yes")
EASY_CONSERVATIVE        = os.getenv("EASY_CONSERVATIVE", "1").lower() in ("1","true","yes")

# 번역 경로(실패해도 무시, 크래시 금지)
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

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402
from peft import PeftModel  # noqa: E402

# -------------------- Logger --------------------
logger = logging.getLogger("polo.easy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Windows 소켓 종료 관련 경고/에러 완화
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logging.getLogger("asyncio").setLevel(logging.WARNING)

# -------------------- Global state --------------------
app = FastAPI(title="POLO Easy Model", version="2.4.0-final-kr+glossary")
model: Optional[torch.nn.Module] = None
tokenizer: Optional[AutoTokenizer] = None
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_available = torch.cuda.is_available()
safe_dtype = torch.float16 if gpu_available else torch.float32

# ======================================================================
# AI Glossary annotator (English kept + Korean gloss in parenthesis)
# ======================================================================
# 기본 경로: 사용자 지정(ENV) > 고정 Windows 경로 > 로컬 파일
WIN_GLOSS = Path(r"C:\POLO\POLO\polo-system\models\easy\glossary_ai.json")
GLOSSARY_PATH = (
    Path(os.getenv("EASY_GLOSSARY_PATH", "")) if os.getenv("EASY_GLOSSARY_PATH")
    else (WIN_GLOSS if WIN_GLOSS.exists() else Path(__file__).resolve().parent / "glossary_ai.json")
)

import functools

# 수식 마스킹 패턴 (개선된 버전)
_MATH_PATTERNS = [
    (re.compile(r"\$\$[\s\S]*?\$\$"), "MATH_DBL"),
    (re.compile(r"\\\[[\s\S]*?\\\]"), "MATH_BRK"),
    (re.compile(r"\\\([\s\S]*?\\\)"), "MATH_PAR"),
    (re.compile(r"\$(?:\\.|[^\$\\])+\$"), "MATH_SNG"),
    (re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}[\s\S]*?\\end\{\1\}"), "MATH_ENV"),
]

# 기존 호환성을 위한 레거시 패턴
_MASK_MATH = [
    (r"\$\$[\s\S]*?\$\$", "__MATH_DBL__"),
    (r"\\\[[\s\S]*?\\\]", "__MATH_BRK__"),
    (r"\\\([\s\S]*?\\\)", "__MATH_PRT__"),
    (r"\$(?:\\.|[^\$\\])+\$", "__MATH_SNG__"),
    (r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}[\s\S]*?\\end\{\1\}", "__MATH_ENV__"),
]
_MASK_LINK_CODE = [
    (r"https?://[^\s)]+", "__URL__"),
    (r"`[^`]+`", "__CODE__"),
]

# 용어 사전 단어들을 번역 전에 마스킹
_MASK_GLOSSARY_TERMS = []

def _compile_glossary_masking_patterns() -> List[Tuple[str, str]]:
    """용어 사전 단어들을 번역 전에 마스킹하기 위한 패턴 생성"""
    terms = _load_glossary()
    if not terms:
        return []
    
    patterns = []
    # 긴 용어 우선으로 정렬
    ordered = sorted(terms.items(), key=lambda kv: (-len(kv[0]), kv[0].lower()))
    
    for term, gloss in ordered:
        safe = re.escape(term).replace(r"\ ", r"\s+")
        # 단어 경계로 정확히 매칭
        pattern = rf"\b({safe})\b"
        token = f"__GLOSSARY_{hash(term)}__"
        patterns.append((pattern, token))
    
    return patterns

def _mask_math_blocks(s: str) -> Tuple[str, List[str]]:
    """수식 블록을 마스킹하고 플레이스홀더 반환"""
    placeholders, out = [], s
    for pat, tag in _MATH_PATTERNS:
        def _sub(m):
            idx = len(placeholders)
            placeholders.append(m.group(0))
            return f"«M{idx}»"
        out = pat.sub(_sub, out)
    return out, placeholders

def _restore_math_blocks(s: str, placeholders: List[str]) -> str:
    """마스킹된 수식 블록을 복원"""
    for i, val in enumerate(placeholders):
        s = s.replace(f"«M{i}»", val)
    return s

def _mask_math_blocks_full(s: str) -> Tuple[str, List[str]]:
    pats = [
        (re.compile(r"\$\$[\s\S]*?\$\$"),),
        (re.compile(r"\\\[[\s\S]*?\\\]"),),
        (re.compile(r"\\\([\s\S]*?\\\)"),),
        (re.compile(r"\$(?:\\.|[^\$\\])+\$"),),
        (re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}[\s\S]*?\\end\{\1\}"),),
    ]
    slots: List[str] = []
    out = s
    for (pat,) in pats:
        def sub(m):
            idx = len(slots)
            slots.append(m.group(0))
            return f"«M{idx}»"
        out = pat.sub(sub, out)
    return out, slots

def _restore_math_blocks_full(s: str, slots: List[str]) -> str:
    for i, val in enumerate(slots):
        s = s.replace(f"«M{i}»", val)
    return s

def _mask_blocks(text: str, spans: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    spans: [(regex_pattern, token_base), ...]
    각 매치마다 __<BASE><idx>__ 형태의 '유일 토큰'을 부여해 언마스킹 충돌을 막는다.
    """
    if not text:
        return text, []
    out = text
    masks: List[Tuple[str, str]] = []
    idx = 0
    for pat, token_base in spans:
        base = re.sub(r"\W+", "", token_base or "MASK")  # 토큰 base 정규화
        regex = re.compile(pat, re.DOTALL)
        def repl(m):
            nonlocal idx
            tok = f"⟦{base.upper()}_{idx}⟧"
            masks.append((tok, m.group(0)))
            idx += 1
            return tok
        out = regex.sub(repl, out)
    return out, masks

def _unmask_blocks(text: str, masks: List[Tuple[str, str]]) -> str:
    """
    Robust unmask: (token, original) 2-튜플이 아닐 때도 안전하게 처리
    """
    out = text
    for item in masks or []:
        try:
            token, original = item[0], item[1]
        except Exception:
            continue
        if token and original:
            out = out.replace(token, original)
    return out

def _unmask_glossary_terms(text: str, glossary_masks: List[Tuple[str, str]]) -> str:
    """마스킹된 용어들을 복원하면서 괄호 설명 추가"""
    if not glossary_masks:
        return text
    
    terms = _load_glossary()
    out = text
    
    for token, original_term in glossary_masks:
        if original_term in terms:
            gloss = terms[original_term]
            # 원어(설명) 형태로 복원
            replacement = f"{original_term}({gloss})"
        else:
            # 용어 사전에 없으면 원어만 복원
            replacement = original_term
        
        out = out.replace(token, replacement)
    
    return out

@functools.lru_cache(maxsize=1)
def _load_glossary() -> Dict[str, str]:
    if not GLOSSARY_PATH.exists():
        logger.info(f"[GLOSSARY] file not found: {GLOSSARY_PATH}")
        return {}
    try:
        obj = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8", errors="ignore"))
        terms = obj.get("terms", {}) if isinstance(obj, dict) else {}
        out = {}
        for k, v in terms.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                out[k.strip()] = v.strip()
        logger.info(f"[GLOSSARY] loaded {len(out)} terms from {GLOSSARY_PATH}")
        return out
    except Exception as e:
        logger.warning(f"[GLOSSARY] load failed: {e}")
        return {}

@functools.lru_cache(maxsize=1)
def _compile_gloss_patterns() -> List[Tuple[re.Pattern, str, str]]:
    terms = _load_glossary()
    ordered = sorted(terms.items(), key=lambda kv: (-len(kv[0]), kv[0].lower()))
    patterns: List[Tuple[re.Pattern, str, str]] = []
    for term, gloss in ordered:
        safe = re.escape(term).replace(r"\ ", r"\s+")
        # 단어 경계 엄격 + 이미 괄호가 뒤따르면 skip는 replace에서 처리
        pat = re.compile(rf"(?i)(?<![A-Za-z0-9_])({safe})(?![A-Za-z0-9_])")
        patterns.append((pat, gloss, term))
    return patterns

# === PATCH: glossary annotator (drop-in replace) ===========================
def annotate_terms_with_glossary(text: str) -> str:
    """
    - 수식/링크/코드는 마스킹해서 보호
    - 같은 문단 내 같은 용어는 1회만 주석
    """
    if not text or not text.strip():
        return text
    masked, m_math = _mask_blocks(text, _MASK_MATH)
    masked, m_misc = _mask_blocks(masked, _MASK_LINK_CODE)

    patterns = _compile_gloss_patterns()
    paragraphs = re.split(r"(\n{2,})", masked)  # 구분자 포함 분할 유지

    out_parts: List[str] = []
    for part in paragraphs:
        if not part or part.isspace() or re.match(r"\n{2,}$", part):
            out_parts.append(part)
            continue
        used = set(); count = 0
        limit = int(os.getenv("EASY_GLOSS_PER_PARA", "4"))
        txt = part
        for pat, gloss, term in patterns:
            if count >= limit: break
            term_key = term.lower()

            def repl(m):
                nonlocal count
                if count >= limit: return m.group(1)
                end = m.end(1)
                tail = txt[end:end+64]
                if re.match(r"\s*\(", tail):
                    return m.group(1)
                if term_key in used:
                    return m.group(1)
                used.add(term_key)
                count += 1
                return f"{m.group(1)} ({gloss})"

            txt = pat.sub(repl, txt, count=1)
        out_parts.append(txt)

    masked2 = "".join(out_parts)
    masked2 = _unmask_blocks(masked2, m_misc)
    masked2 = _unmask_blocks(masked2, m_math)
    return _squash_duplicate_parens(masked2)
# ========================================================================

def reload_glossary_cache() -> int:
    _load_glossary.cache_clear()
    _compile_gloss_patterns.cache_clear()
    _load_glossary(); _compile_gloss_patterns()
    return len(_load_glossary())

# === Unique masking helpers (LLM용: 고유 토큰으로 정확 복원) ===
def _mask_unique(text: str, patterns: List[Pattern], tag: str) -> Tuple[str, List[str]]:
    """각 매치별로 __TAG_0000__ 형태의 고유 토큰으로 바꾸고, 원문 리스트를 반환"""
    saved: List[str] = []
    out = text

    def _make_repl(m):
        idx = len(saved)
        token = f"__{tag}_{idx:04d}__"
        saved.append(m.group(0))
        return token

    for pat in patterns:
        out = pat.sub(_make_repl, out)
    return out, saved

def _unmask_unique(text: str, tag: str, saved: List[str]) -> str:
    out = text
    for idx, original in enumerate(saved):
        out = out.replace(f"__{tag}_{idx:04d}__", original)
    return out

# 고유 마스킹용 패턴(수식/코드/URL)
_MATH_PATTERNS = [
    re.compile(r"\$\$[\s\S]*?\$\$"),
    re.compile(r"\\\[[\s\S]*?\\\]"),
    re.compile(r"\\\([\s\S]*?\\\)"),
    re.compile(r"\$(?:\\.|[^\$\\])+\$"),
    re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}[\s\S]*?\\end\{\1\}"),
]
_CODE_URL_PATTERNS = [
    re.compile(r"https?://[^\s)]+"),
    re.compile(r"`[^`]+`"),
]
# -------------------- Text Utils --------------------
def _pick_attn_impl() -> str:
    # GPU만 허용 (CPU 우회 금지)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다. CUDA_VISIBLE_DEVICES 설정을 확인하세요.")
    # Windows/드라이버 이슈 회피: eager 우선
    try:
        import flash_attn  # noqa: F401
        logger.info("✅ flash_attn 사용 시도 → flash_attention_2")
        return "flash_attention_2"
    except Exception:
        pass
    try:
        from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
        logger.info("ℹ️ sdpa 사용")
        return "sdpa"
    except Exception:
        logger.info("⚙️ attention backend: 'eager'")
        return "eager"

def _gen_cfg() -> GenerationConfig:
    if model is None:
        # 모델이 로드되지 않았을 때 기본 설정
        cfg = GenerationConfig()
        cfg.do_sample = False
        cfg.top_p = 1.0
        cfg.temperature = 1.0
        cfg.no_repeat_ngram_size = 4
        return cfg
    
    cfg = GenerationConfig.from_model_config(model.config)
    cfg.do_sample = False
    # 샘플링 파라미터 해제(경고 억제). 일부 버전은 None 미지원 → fallback
    try:
        cfg.top_p = None
        cfg.temperature = None
    except Exception:
        cfg.top_p = 1.0
        cfg.temperature = 1.0
    cfg.no_repeat_ngram_size = 4
    # repetition_penalty는 generate 인자로 넘기는 편이 안전
    return cfg

def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))

def _detect_lang_safe(text: str) -> str:
    if not text or not text.strip(): return "en"
    if _contains_hangul(text): return "ko"
    latin = len(re.findall(r"[A-Za-z]", text)); total = len(re.findall(r"[A-Za-z가-힣]", text))
    if total == 0: return "en"
    return "ko" if (latin/total) < 0.5 else "en"

# === PATCH: bracket/ocr noise normalization (drop-in replace) =============
def _normalize_bracket_tokens(t: str) -> str:
    if not t: return t
    # () : LRB/RRB가 '단어 경계 + 공백 0~1 + 영문/숫자' 앞일 때만 치환
    t = re.sub(r"(?i)\bL\s*R\s*B\b(?=\s*[A-Za-z0-9])|\bLRB\b(?=\s*[A-Za-z0-9])", "(", t)
    t = re.sub(r"(?i)\bR\s*R\s*B\b(?=\s*[A-Za-z0-9])|\bRRB\b(?=\s*[A-Za-z0-9])", ")", t)
    # [] : LSB/RSB도 동일
    t = re.sub(r"(?i)\bL\s*S\s*B\b(?=\s*[A-Za-z0-9])|\bLSB\b(?=\s*[A-Za-z0-9])|\bLRSB\b(?=\s*[A-Za-z0-9])", "[", t)
    t = re.sub(r"(?i)\bR\s*S\s*B\b(?=\s*[A-Za-z0-9])|\bRSB\b(?=\s*[A-Za-z0-9])|\bRRSB\b(?=\s*[A-Za-z0-9])", "]", t)
    # ⟨⟩ 등
    t = re.sub(r"(?i)\blangle\b|⟨", "<", t)
    t = re.sub(r"(?i)\brangle\b|⟩", ">", t)
    return t

_DUP_PARENS = re.compile(r"\(([^()]{1,80})\)\s*\(\1\)")

def _squash_duplicate_parens(t: str) -> str:
    if not t: return t
    # (같은내용)(같은내용) → (같은내용)
    while _DUP_PARENS.search(t):
        t = _DUP_PARENS.sub(r"(\1)", t)
    # 여분 괄호/공백 수축
    t = re.sub(r"\)\s*\)+", ")", t)
    t = re.sub(r"\(\s*\(+", "(", t)
    t = re.sub(r"\(\s+\)", "", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    return t

def _drop_latex_curly_blocks(t: str) -> str:
    """LaTeX 중괄호 블록 정리 (보수 모드)"""
    if not t: return t
    
    def repl(m):
        inner = m.group(1).strip()
        
        # 수식/코드/URL 의심되면 건드리지 않음
        if re.search(r"(\\[a-zA-Z]+|[$]{1,2}.*?[$]{1,2}|https?://|`[^`]+`)", inner):
            return "{" + inner + "}"  # 그대로 보존
        
        # 보수 모드: 더 엄격하게 '쓸모없는 토막'만 제거
        if EASY_CONSERVATIVE:
            if len(inner) <= 2: return ""
            if re.fullmatch(r"[\s.,;:~_=+\-*/\\#\[\]()|^%]+", inner): return ""
            # 짧은 토막은 ( )로 감싸지 말고 그냥 제거
            if len(inner.split()) <= 2 and len(re.findall(r"[A-Za-z가-힣0-9]", inner)) <= 4:
                return ""
            # 그 외는 아예 보존
            return "{" + inner + "}"
        
        # 공격 모드(기존 동작): 의미 있으면 ( )로 변환
        if re.fullmatch(r"[\s\d.,;:~_=+\-*/\\#\[\]()|^%]+", inner): return ""
        if len(inner.split()) <= 3 and len(re.findall(r"[A-Za-z가-힣0-9]", inner)) <= 6: return ""
        return f"({inner})"
    
    t = re.sub(r"\{+\s*([^{}]{1,200})\s*\}+", repl, t)
    # 남은 중괄호는 깔끔히 제거(보수 모드에서는 보존한 블록도 있으니 다시 한 번 정리)
    t = t.replace("{", "").replace("}", "")
    return t

def _strip_debug_markers(s: str) -> str:
    """CHUNK/마커 제거"""
    if not s: return s
    s = re.sub(r"\[/?(CHUNK|TEXT|CONCLU\w*|CHANK|CHUMP)[^\]]*\]", " ", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# TikZ/PGF/단위 쓰레기 제거
_TIKZ_ENV_PAT = (
    r"\\begin\{(tikzpicture|axis|scope)\}[\s\S]*?\\end\{\1\}"
)

def _strip_tikz_len_garbage(s: str) -> str:
    if not s: return s
    t = s
    # 1) tikz/axis/scope 환경 통째로 제거
    t = re.sub(_TIKZ_ENV_PAT, " ", t, flags=re.IGNORECASE)
    # 2) {0.25mm}, {.75cm}, {14mm} 같은 길이 토큰 제거
    t = re.sub(r"\{\s*\d+(?:\.\d+)?\s*(?:mm|cm|pt|in|bp)\s*\}", " ", t, flags=re.I)
    # 3) { .25cm .75cm 1.5cm ... } 같은 길이 리스트 제거
    t = re.sub(r"\{\s*(?:\d+(?:\.\d+)?\s*(?:mm|cm|pt|in|bp)\s+){1,}\d*(?:\.\d+)?\s*(?:mm|cm|pt|in|bp)\s*\}", " ", t, flags=re.I)
    # 4) line cap=round, line join=round 등 스타일 잔여
    t = re.sub(r"(?:^|[\s,])(?:line\s+cap|line\s+join)\s*=\s*(?:round|miter|bevel)\b", " ", t, flags=re.I)
    # 5) \pgf..., \tikz... 명령 토막
    t = re.sub(r"\\(?:pgf|tikz)[A-Za-z@]*\b(?:\[[^\]]*\])?", " ", t)
    # 6) 빈 중괄호 반복 "{ } { } ..." 제거
    t = re.sub(r"(?:\{\s*\}\s*){2,}", " ", t)
    # 7) 공백 정리
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _strip_tikz_dims_and_styles(s: str) -> str:
    """
    TikZ/PGF 그림 설정 잔여물과 치수 토큰 블록을 강하게 제거
    예: { } {0mm} {.25cm} {.75cm} ... , line cap=round,line join=round, postaction={decorate}
    """
    if not s: return s
    # {}나 {0mm} {.25cm} 등 치수 토큰 블록 연속 제거
    s = re.sub(r"(?:\{\s*(?:-?\d+(?:\.\d+)?\s*(?:mm|cm|pt|in)|\.?)\s*\}\s*){1,}", " ", s, flags=re.I)
    s = re.sub(r"\{\s*\}", " ", s)

    # TikZ 스타일 키 제거
    s = re.sub(r"(?:line\s+cap|line\s+join|postaction|decorate|dash\s*pattern|draw\s*=\s*|fill\s*=\s*|node\s*|dsparrow\w*)[^{}\n]*", " ", s, flags=re.I)

    # 스타일 블록 {key=value,...} 제거 (간단형)
    s = re.sub(r"\{[^{}=]*=[^{}]*\}", " ", s)

    # 과도 공백 정리
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _purge_ocr_bracey_noise(t: str) -> str:
    """중괄호·제어토큰 덩어리(티크즈/그림 잔여) 과감히 정리"""
    if not t:
        return t
    # {..}{..}{..}가 연속 3회 이상이면 노이즈로 보고 제거
    t = re.sub(r"(?:\{[^{}]{0,40}\}\s*){3,}", " ", t)
    # 괄호만 잔뜩 있거나 기호 비중이 60% 넘는 라인 정리
    lines = []
    for ln in t.splitlines():
        if not ln.strip():
            continue
        sym = len(re.findall(r"[{}\\_/^$~]", ln))
        alnum = len(re.findall(r"[A-Za-z0-9가-힣]", ln))
        if alnum == 0 and sym > 10:
            continue
        if sym > 0 and (sym / max(1, sym + alnum)) > 0.6:
            continue
        lines.append(ln)
    t = "\n".join(lines)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t
# ========================================================================

def _strip_math_all(s: str) -> str:
    if not s: return s
    rep = " . "
    s = re.sub(r"\$\$[\s\S]*?\$\$", rep, s)
    s = re.sub(r"\\\[[\s\S]*?\\\]", rep, s)
    s = re.sub(r"\\\([\s\S]*?\\\)", rep, s)
    s = re.sub(r"\$(?!\$)(?:[^$\\]|\\.)+\$", rep, s)
    s = re.sub(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}[\\s\S]*?\\end\{\1\}", rep, s)
    return s

def _strip_tables_figures_all(s: str) -> str:
    if not s: return s
    rep = " . "
    t = s
    t = re.sub(r"\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}", rep, t)
    t = re.sub(r"\\begin\{tabularx?\*?\}[\s\S]*?\\end\{tabularx?\*?\}", rep, t)
    t = re.sub(r"\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}", rep, t)
    t = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", rep, t)
    t = re.sub(r"\\caption\{[^}]*\}", rep, t)
    t = re.sub(r"(?i)\b(?:figure|table)\s*\d+\s*[:.]\s*", rep, t)
    t = re.sub(r"(?:표|그림)\s*\d+\s*[:.]\s*", rep, t)
    # tikz/axis/scope 환경도 제거
    t = re.sub(_TIKZ_ENV_PAT, rep, t, flags=re.IGNORECASE)
    return t

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
        r"\b(?:Top-1|Top-5|SOTA|F1|AUC|ROC|mAP|IoU|PSNR|SSIM|CER|WER)\b",
        r"\b\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?(?:ms|s|fps|GFLOPs?|M|K|dB)\b",
        r"\b(?:Transformer|BERT|GPT(?:-\d+)?|Llama|Mistral|Qwen|Gemma|T5|Whisper)\b",
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
    logger.info("==============================================")
    logger.info(f"🔄 모델 로딩 시작: {BASE_MODEL}")
    logger.info(f"    EASY_ADAPTER_DIR = {ADAPTER_DIR}")
    logger.info(f"    HF_HOME          = {os.getenv('HF_HOME')}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다. torch.cuda.is_available() == False")
    gpu_available = True
    device = "cuda"
    safe_dtype = torch.float16
    logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    logger.info("==============================================")

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
        logger.warning(f"attn='{attn_impl}' 실패({e}) → eager 재시도")
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

# -------------------- 번역 파이프라인 --------------------
PROJECT_FIXED = "polo-472507"
GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT", PROJECT_FIXED).strip() or PROJECT_FIXED

def _translate_to_korean(text: str) -> str:
    try:
        if not text or not text.strip():
            return ""
        out = _translate_with_google_api(text)
        if out:
            return out
        # 파파고 사용하지 않음
        # out = _translate_with_papago(text)
        # if out:
        #     return out
        logger.warning("⚠️ 외부 번역 실패 → LLM 백업 사용")
        return _translate_with_llm(text)
    except Exception as e:
        logger.warning(f"번역 실패: {e}")
        return text

def _translate_with_google_api(text: str) -> str:
    try:
        import requests
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request as GARequest

        def _find_service_account_path() -> Optional[Path]:
            base = Path(__file__).resolve().parent
            primary = base / "polo-472507-c5db0ee4a109.json"
            if primary.exists():
                return primary
            for p in base.glob("*.json"):
                try:
                    info = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
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
                        d = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
                        for key_field in ("GOOGLE_API_KEY", "api_key", "apiKey", "key"):
                            v = str(d.get(key_field, "")).strip()
                            if v:
                                return v
            except Exception:
                pass
            return ""

        sa_path = _find_service_account_path()
        if sa_path:
            try:
                info = json.loads(sa_path.read_text(encoding="utf-8", errors="ignore"))
                if str(info.get("type", "")) == "service_account":
                    scopes = [
                        "https://www.googleapis.com/auth/cloud-translation",
                        "https://www.googleapis.com/auth/cloud-platform",
                    ]
                    creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
                    creds.refresh(GARequest())
                    headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}
                    url_v3 = f"https://translation.googleapis.com/v3/projects/{GOOGLE_PROJECT}/locations/global:translateText"
                    body_v3 = {"contents": [text[:4000]], "mimeType": "text/plain", "sourceLanguageCode": "en", "targetLanguageCode": "ko"}
                    r3 = requests.post(url_v3, headers=headers, data=json.dumps(body_v3, ensure_ascii=False), timeout=25)
                    if r3.status_code == 200:
                        js3 = r3.json()
                        trans = (js3.get("translations") or [])
                        if trans:
                            return trans[0].get("translatedText", "") or ""
                    else:
                        msg = r3.text[:500]
                        logger.warning(f"[TR] v3 HTTP {r3.status_code}: {msg}")
                else:
                    logger.warning("[TR] v3: 잘못된 service_account JSON")
            except Exception as e:
                logger.warning(f"[TR] v3 예외: {e}")
        # v2 (API key)
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
        # v2 (OAuth with SA)
        if sa_path:
            try:
                info = json.loads(sa_path.read_text(encoding="utf-8", errors="ignore"))
                if str(info.get("type", "")) == "service_account":
                    from google.oauth2 import service_account
                    from google.auth.transport.requests import Request as GARequest
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
            except Exception as e:
                logger.warning(f"[TR] v2(OAuth) 예외: {e}")
        return ""
    except Exception as e:
        logger.warning(f"[TR] 예외: {e}")
        return ""

def _translate_with_papago(text: str) -> str:
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
        
        # 용어 사전 단어들을 마스킹하여 번역되지 않도록 함
        glossary_patterns = _compile_glossary_masking_patterns()
        masked_text, glossary_masks = _mask_blocks(text, glossary_patterns)
        
        translate_prompt = (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            "너는 YOLO 객체 탐지 논문 전문 번역가다. 영어 문장을 정확하고 자연스러운 한국어로 번역하라.\n"
            "규칙:\n"
            "- YOLO 관련 용어(YOLO, object detection, bounding box, IoU, mAP, COCO, NMS, backbone, neck, head, FPN, Darknet, ResNet 등)는 영어 그대로 두고 (한국어 풀이) 추가\n"
            "- 의미 정확, 도메인 용어 보존\n"
            "- 숫자/고유명사/기호 왜곡 금지\n"
            "- 한국어 문장부호 사용\n"
            "- 마스킹된 토큰(⟦GLOSSARY_xxx⟧)은 절대 번역하지 말고 그대로 유지\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{masked_text}\n\n[한국어 번역]\n"
            "<|eot_id|>\n"
        )
        inputs = tokenizer(translate_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(MAX_NEW_TOKENS, 600),
                do_sample=False,  # 결정적 디코딩
                use_cache=True,
                repetition_penalty=1.2,  # 반복 억제
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
        # 용어는 '그대로'만 복원 (주석은 이후 단계에서 1회만 적용)
        result = _unmask_blocks(result, glossary_masks)
        
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
    if not text or not text.strip():
        return text
    if force_external:
        return _translate_to_korean(text)
    lang = _detect_lang_safe(text)
    if lang == "ko" and not _need_ko(text):
        return text
    return _translate_to_korean(text)

# -------------------- 간단 LaTeX 클리너 --------------------
def _clean_latex_text(s: str) -> str:
    if not s: return ""
    t = s
    # 주석 제거
    t = re.sub(r"(?m)^%.*?$", " ", t)
    # 수식/표/그림 제거 또는 마스킹
    if EASY_STRIP_MATH:
        t = _strip_math_all(t)
    t = _strip_tables_figures_all(t)
    # 일반 LaTeX 명령 제거
    t = re.sub(r"\\(emph|textbf|textit|underline)\{([^}]+)\}", r"\2", t)
    t = re.sub(r"\\(cite|ref|label|url)\{[^}]+\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    # TikZ/치수 잔여물 추가 제거
    t = _strip_tikz_dims_and_styles(t)
    # 공백 정리
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# === PATCH: _rewrite_text (drop-in replace) ===============================
def _smart_chunks(text: str, max_chars: int = 3500) -> List[str]:
    """문단 기준으로 잘라 전체 본문을 빠짐없이 처리"""
    s = text.replace("\r\n", "\n")
    paras = re.split(r"\n\s*\n", s)
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if cur_len + len(p) + 2 > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks if chunks else [text]

# -------------------- 핵심: 쉬운 한국어 리라이터 (수식 고유 토큰 보호) --------------------
async def _rewrite_text(content: str, title: Optional[str] = None, context_hint: Optional[str] = "", *, style: str = "three_para_ko") -> str:
    if not content or not content.strip():
        return ""
    
    start_time = time.time()
    logger.info(f"[PERF] _rewrite_text 시작: {title or 'Unknown'}")
    
    # 수식 마스킹
    content_masked, math_masks = _mask_blocks(content, _MASK_MATH)
    
    sys_msg = (
        "너는 YOLO(You Only Look Once) 객체 탐지 논문을 쉽게 풀어쓰는 전문가다.\n"
        "YOLO 논문의 핵심 개념들을 정확하고 이해하기 쉽게 설명하라.\n"
        "\n"
        "규칙:\n"
        "- YOLO 관련 용어(YOLO, object detection, bounding box, IoU, mAP, COCO, NMS, backbone, neck, head, FPN, Darknet, ResNet 등)는 영어 그대로 두고 (한국어 풀이)를 1회만 붙여라.\n"
        "- 수식/코드/링크/토큰은 변형 금지. 입력에 ⟦MATH_i⟧, ⟦CODE_i⟧ 같은 마스크가 오면 그대로 보존하고, 출력에도 동일하게 남겨라.\n"
        "- 숫자/기호/단위/약어를 바꾸지 마라. 추측/과장은 금지.\n"
        "- 3~5 문단, 각 2~4문장. 매 문단은 연결어(먼저/다음/마지막 등)로 자연스럽게 잇는다.\n"
        "- YOLO의 핵심 아이디어(실시간 탐지, 단일 네트워크, 격자 기반 예측)를 강조하라.\n"
        "- 불필요한 반복/군더더기 제거. \"방법/코딩방식\" 같은 보조풀이를 과도하게 넣지 마라.\n"
        "\n"
        "형식:\n"
        "- 한국어 본문만 출력. 머리말/꼬리말/설명 금지.\n"
        "\n"
        "주의: ⟦MATH_i⟧, ⟦CODE_i⟧, ⟦URL_i⟧ 같은 토큰은 **그대로** 출력에 복사하라. 내용/순서/공백을 바꾸지 마라.\n"
    )
    title_line = f"[SECTION] {title}\n" if title else ""
    user_msg = f"{title_line}{context_hint}\n[TEXT]\n{content_masked[:6000]}\n\n[REWRITE in Korean]\n"

    prompt = (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{sys_msg}"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_msg}"
        "<|eot_id|>\n"
    )
    inputs = _safe_tokenize(prompt, max_len=MAX_INPUT_TOKENS)
    
    # 디버깅: 입력 프롬프트 로그
    logger.info(f"[DEBUG] 입력 토큰 수: {inputs['input_ids'].shape[1]}")
    logger.info(f"[DEBUG] 프롬프트 미리보기: {prompt[:300]}...")
    
    # LLM 생성 시간 측정
    gen_start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=_gen_cfg(),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # 🔧 샘플링 OFF
            use_cache=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - gen_start
    logger.info(f"[PERF] LLM 생성 시간: {gen_time:.2f}초")
    
    seq = outputs[0]
    inp_len = inputs["input_ids"].shape[1]
    gen_tokens = seq[inp_len:]
    
    # 디버깅: 토큰 정보
    logger.info(f"[DEBUG] 입력 길이: {inp_len}, 출력 길이: {len(seq)}")
    logger.info(f"[DEBUG] 생성된 토큰 수: {len(gen_tokens)}")
    logger.info(f"[DEBUG] 생성된 토큰 ID들: {gen_tokens.tolist()[:20]}...")
    
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # 디버깅: 생성된 텍스트 로그
    logger.info(f"[DEBUG] 생성된 텍스트 길이: {len(text)}")
    logger.info(f"[DEBUG] 생성된 텍스트 미리보기: {text[:200]}...")
    
    # skip_special_tokens=False로도 시도해보기
    text_raw = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
    logger.info(f"[DEBUG] skip_special_tokens=False 결과: {text_raw[:200]}...")
    
    for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
        p = text.find(stop)
        if p != -1:
            text = text[:p].strip()
            break

    # 후처리
    text = _unmask_blocks(text, math_masks)  # 수식 언마스킹
    text = _normalize_bracket_tokens(text)
    text = _strip_tikz_dims_and_styles(text)  # TikZ/치수 제거
    if EASY_STRIP_MATH:
        text = _strip_math_all(text)
    text = _strip_tables_figures_all(text)
    text = _postprocess_terms(text)

    # 🔧 추가 클리너
    text = _strip_debug_markers(text)
    text = _drop_latex_curly_blocks(text)
    text = _squash_duplicate_parens(text)

    text = _auto_bold_terms(text)
    text = _hilite_sentences(text, max_marks=2)
    text = _sanitize_repeats(text)

    # 용어 주석 — 이 자리에서 1회만
    text = annotate_terms_with_glossary(text)

    if _need_ko(text):
        text = _ensure_korean(text, force_external=EASY_FORCE_TRANSLATE)
    
    # JSON 전달용: 마크다운 형식으로 변환
    text = _convert_to_markdown(text, math_masks)
    
    total_time = time.time() - start_time
    logger.info(f"[PERF] _rewrite_text 완료: {total_time:.2f}초 (LLM: {gen_time:.2f}초)")
    return text
# -------------------- Startup --------------------
@app.on_event("startup")
async def _startup_event():
    load_model()
    # 사전 워밍업
    reload_glossary_cache()
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
    chunks_jsonl: str = Field(..., description="JSONL 내용 문자열 또는 경로(파일/디렉토리/tex)")
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
    return {"message": "POLO Easy Model API (Final KR + Glossary)", "model": BASE_MODEL, "mode": "JSON+HTML"}

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
        "glossary_path": str(GLOSSARY_PATH),
        "glossary_terms": len(_load_glossary()),
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
    # 주석은 _rewrite_text 내에서 1회 적용됨 — 중복 방지
    return TextResponse(simplified_text=simplified_text, translated_text=None)

# -------------------- /generate (요약 JSON 생성) --------------------
def _coerce_json(text: str) -> Dict[str, Any]:
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s: text = text[s:e+1]
    return json.loads(text)

def _merge_with_schema(data: dict, schema: dict) -> dict:
    if not isinstance(data, dict):
        return json.loads(json.dumps(schema))
    out = json.loads(json.dumps(schema))  # deep copy
    for k, v in data.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_with_schema(v, out[k])
        else:
            out[k] = v
    return out

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
        lines = (src or "").splitlines(); idxs = []
        for i, line in enumerate(lines):
            for key, pat in headers:
                if re.match(pat, line.strip(), flags=re.IGNORECASE):
                    idxs.append((i, key)); break
        idxs.sort()
        for j, (start_i, key) in enumerate(idxs):
            end_i = idxs[j+1][0] if j+1 < len(idxs) else len(lines)
            chunk = "\n".join(lines[start_i+1:end_i]).strip()[:4000]
            # 수식 보존을 원하면 EASY_STRIP_MATH=0 (기본값 0)
            if EASY_STRIP_MATH:
                chunk = _strip_math_all(chunk)
            chunk = _strip_tables_figures_all(chunk)
            sections[key] = chunk
        
        # 섹션 추출 실패 대비 Fallback
        for key in ["introduction", "methods", "results", "discussion", "conclusion"]:
            if not sections[key] or len(sections[key].strip()) < 40:
                # Introduction이 비면 강제로 기본 내용 생성
                if key == "introduction" and not sections[key]:
                    sections[key] = "This paper presents a novel approach to the problem. The main contributions include theoretical analysis and experimental validation."
                    logger.warning(f"[EASY] {key} 섹션이 비어있어 기본 내용으로 대체")
                    continue
                
                # 바로 직후 섹션과 묶기
                next_key = None
                if key == "introduction": next_key = "methods"
                elif key == "methods": next_key = "results"
                elif key == "results": next_key = "discussion"
                elif key == "discussion": next_key = "conclusion"
                
                if next_key and sections[next_key]:
                    # 현재 섹션이 비어있으면 다음 섹션 내용을 가져옴
                    sections[key] = sections[next_key][:2000]  # 일부만 가져옴
                    logger.warning(f"[EASY] {key} 섹션이 비어있어 {next_key} 내용으로 대체")
                else:
                    # JSONL 파편에서 문단 단위로 재조합
                    paragraphs = re.split(r'\n\s*\n', src)
                    for para in paragraphs:
                        if len(para.strip()) > 40 and not any(symbol in para for symbol in ['{', '}', '\\', '$']):
                            sections[key] = para.strip()[:2000]
                            break
                    logger.warning(f"[EASY] {key} 섹션을 문단 단위로 재조합")
        
        return sections

    extracted = _extract_sections(request.text or "")

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
        "너는 YOLO 객체 탐지 논문을 분석하는 과학 선생님이다. 아래 JSON 스키마의 '키/구조'를 그대로 두고 '값'만 채워라. "
        "YOLO 관련 용어는 영어 그대로 두고 (한국어 풀이)를 붙여라. "
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
            generation_config=_gen_cfg(),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    seq = outputs[0]
    inp_len = inputs["input_ids"].shape[1]
    gen_tokens = seq[inp_len:]
    raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
        p = raw.find(stop)
        if p != -1:
            raw = raw[:p].strip()
            break

    # JSON 보정
    try:
        data = _coerce_json(raw)
    except Exception:
        data = schema_copy
    data = _merge_with_schema(data, schema_copy)

    # 후처리 + 한글 강제 + 용어 주석 (수식은 보호)
    for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
        try:
            easy_val = (data.get(k, {}) or {}).get("easy", "")
            if isinstance(easy_val, str) and easy_val.strip():
                # 수식 보호
                masked, m_math = _mask_blocks(easy_val, _MASK_MATH)

                t = _postprocess_terms(_normalize_bracket_tokens(masked))
                if EASY_STRIP_MATH:
                    t = _strip_math_all(t)
                t = _strip_tables_figures_all(t)
                t = _auto_bold_terms(t)
                t = _hilite_sentences(t, max_marks=2)
                t = _sanitize_repeats(t)
                if _need_ko(t):
                    t = _ensure_korean(t)
                t = annotate_terms_with_glossary(t)  # ← 중복 괄호 방지 포함된 최신 함수로 교체
                
                # JSON 전달용: 마크다운 형식으로 변환
                t = _convert_to_markdown(t, m_math)
                data[k]["easy"] = t
        except Exception:
            pass

    # plain_summary 생성(간단히 이어 붙인 뒤 정돈) — 수식 보존 + 주석 절제
    plain = " ".join(
        (data.get("abstract", {}).get("easy") or "",
         data.get("introduction", {}).get("easy") or "",
         data.get("conclusion", {}).get("easy") or "")
    ).strip()
    if plain:
        masked, m_math = _mask_blocks(plain, _MASK_MATH)
        plain = re.sub(r"\s+", " ", masked)
        plain = _purge_ocr_bracey_noise(plain)
        plain = _unmask_blocks(plain, m_math)
        # 요약에는 주석을 더 절제(전체 1회) — 가독성
        plain = annotate_terms_with_glossary(plain)
        data["plain_summary"] = plain[:2000]
        
# === PART 4/7 =============================================================
# ---------- LaTeX 표/섹션/자산 로더 ----------
def _extract_table_data(text: str) -> List[dict]:
    """아주 단순한 tabular 파서 (헤더 1줄 + 데이터 여러 줄)"""
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

    in_abstract = False

    for ln in content.splitlines():
        # --- Abstract 시작/끝 ---
        if abs_beg.match(ln):
            _flush()
            cur_title = "Abstract"
            cur_buf, cur_raw = [], []
            section_type = "section"
            in_abstract = True
            continue
        if abs_end.match(ln):
            # abstract 본문 누적 끝
            in_abstract = False
            _flush()
            cur_title, cur_buf, cur_raw = None, [], []
            section_type = "section"
            continue

        # --- Section / Subsection 헤더 ---
        m1 = sec_pat.match(ln); m2 = sub_pat.match(ln)
        if m1:
            _flush()
            cur_title = m1.group(1).strip()
            cur_buf, cur_raw = [], []
            section_type = "section"
            continue
        if m2:
            _flush()
            cur_title = m2.group(1).strip()
            cur_buf, cur_raw = [], []
            section_type = "subsection"
            continue

        # --- 본문 누적 (✨ 기존 버그: 누적이 전혀 안 되고 있었습니다) ---
        if cur_title is not None or in_abstract:
            cur_buf.append(ln)
            cur_raw.append(ln)
        else:
            # \section 이전의 프리앰블/머릿말은 버림
            continue

    _flush()

    if not sections:
        clean = _clean_latex_text(content)
        sections = [{
            "index": 0, "title": "Full Document", "content": clean, "raw_content": content,
            "table_data": _extract_table_data(content), "section_type": "section",
        }]

    # 초단문/노이즈 섹션 제거 로직 (조건은 기존과 동일)
    def _len_clean(s: str) -> int:
        return len(_clean_latex_text(s or ""))

    pruned: List[dict] = []
    for s in sections:
        clen = _len_clean(s.get("content",""))
        if s.get("table_data"):
            pruned.append(s); continue
        tlow = (s.get("title","").strip().lower())
        if (tlow.startswith("abstract") and clen < 20) or (not tlow.startswith("abstract") and clen < 90):
            logger.info(f"[EASY] drop tiny section: '{s.get('title')}' (len={clen})")
            continue
        pruned.append(s)

    if not pruned and sections:
        pruned = sections

    for i, s in enumerate(pruned):
        s["index"] = i

    logger.info(f"[EASY] Parsed {len(pruned)} sections from LaTeX (pruned)")
    return pruned

def _load_assets_metadata(source_dir: Path) -> Dict[str, Any]:
    """arXiv 변환 파이프라인에서 생성한 assets.jsonl 포맷 로드(없으면 빈 dict)"""
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

# ---------- HTML 유틸 ----------
def _get_current_datetime() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def _convert_to_markdown(text: str, math_masks: List[Tuple[str, str]] = None) -> str:
    """
    JSON 전달용: 수식 마스킹 토큰을 마크다운 형식으로 변환
    """
    if not text: return ""
    
    # 수식 마스킹 토큰을 마크다운 수식으로 변환
    if math_masks:
        for token, original in math_masks:
            # LaTeX 수식을 마크다운 형식으로 변환
            if original.startswith('$') and original.endswith('$'):
                # 인라인 수식: $...$ → $...$
                text = text.replace(token, original)
            elif original.startswith('$$') and original.endswith('$$'):
                # 블록 수식: $$...$$ → $$...$$
                text = text.replace(token, original)
            elif original.startswith('\\[') and original.endswith('\\]'):
                # 블록 수식: \[...\] → $$...$$
                math_content = original[2:-2]
                text = text.replace(token, f"$${math_content}$$")
            elif original.startswith('\\(') and original.endswith('\\)'):
                # 인라인 수식: \(...\) → $...$
                math_content = original[2:-2]
                text = text.replace(token, f"${math_content}$")
            else:
                # 기타 수식 환경
                text = text.replace(token, f"$${original}$$")
    
    # LaTeX 수식 명령을 마크다운으로 변환
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', text)
    
    # LaTeX 섹션 명령을 마크다운 헤더로 변환
    text = re.sub(r'\\section\{([^}]+)\}', r'## \1', text)
    text = re.sub(r'\\subsection\{([^}]+)\}', r'### \1', text)
    text = re.sub(r'\\subsubsection\{([^}]+)\}', r'#### \1', text)
    
    return text

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
# === PART 5/7 =============================================================
# ---------- HTML 저장 ----------
def _save_html_results(sections: List[dict], results: List['VizResult'], output_path: Path, paper_id: str):
    sections = _dedup_titles(list(sections))
    titles = [(sec.get("title") or f"Section {i+1}").strip() for i, sec in enumerate(sections)]
    slugs = _slugify_unique(titles)
    gen_at = _get_current_datetime()

    css = """
    :root{
      --bg: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --brand: #2563eb;
      --hi: #fff4ae;
    }

    html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
    body{
      font-family: system-ui, -apple-system, "Apple SD Gothic Neo", "Malgun Gothic",
                   "Pretendard", "Noto Sans KR", Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg); color: var(--text);
      font-size: 17px; line-height: 1.85; letter-spacing: -0.005em;
      word-break: keep-all; line-break: loose;
      margin:0;
    }

    .wrapper{max-width: 1120px; margin: 0 auto; padding: 28px;}
    .layout{display:grid; grid-template-columns: 300px 1fr; gap:28px;}
    @media (max-width: 1100px){ .layout{grid-template-columns: 1fr;} }

    .header{margin-bottom:28px; padding:36px 22px;
      border-radius:16px; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
      color:#fff; box-shadow:0 6px 24px rgba(0,0,0,.12)}
    .title{font-weight:800; letter-spacing:-0.01em; font-size: clamp(24px, 3vw, 34px); margin:0 0 6px}
    .subtitle{opacity:.95; font-weight:600; margin-top:6px}
    .meta{opacity:.9; font-size:14px; margin-top:10px}

    .toc-sidebar{background:#f8fafc; border:1px solid var(--border); border-radius:12px; padding:18px; position:sticky; top:24px}
    .toc-title{font-weight:800; color:#0b1220; font-size:18px; margin-bottom:10px}
    .toc-link{color:#0b1220; text-decoration:none; display:block; padding:8px 10px; border-radius:8px}
    .toc-link:hover{background:#eef2ff}
    .num{opacity:.65; font-weight:700; margin-right:6px}

    .section-card{
      border:1px solid var(--border); border-radius:16px; padding:22px 22px;
      box-shadow:0 1px 8px rgba(0,0,0,.05); transition: box-shadow .18s, border-color .18s
    }
    .section-card:hover{box-shadow:0 6px 18px rgba(0,0,0,.08); border-color:#d7e0ea}

    .content p, .content ul, .content ol, .content pre, .content blockquote { max-width: 70ch; }

    h2{font-size: clamp(20px, 2.2vw, 26px); font-weight: 800; letter-spacing:-0.01em;
       margin: 2px 0 14px; line-height:1.35}
    h3{font-size: clamp(17px, 1.8vw, 20px); font-weight: 700; margin: 18px 0 8px; line-height:1.45}

    p{margin: 0 0 14px; color: var(--text)}
    p strong{font-weight: 650}
    p:first-of-type{font-weight: 500}
    mark{background: var(--hi); padding:.12em .25em; border-radius:.25em}

    a{color: var(--brand); text-decoration: underline; text-underline-offset: 2px}
    ul{padding-left: 1.25em; margin: 0 0 14px}
    li{margin: .2em 0}
    code{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
         background:#0f172a; color:#e5e7eb; padding:2px 6px; border-radius:6px; font-size: 85%}
    pre{background:#0f172a; color:#e5e7eb; padding:14px; border-radius:10px; overflow:auto}

    blockquote{
      border-left: 4px solid #c7d2fe; padding: 10px 14px; background:#f5f7ff; border-radius: 8px;
      color:#1f2a44; margin: 8px 0 14px; font-weight:500
    }

    .image-container{margin:18px 0}
    .image-container img{max-width:100%; height:auto; border-radius:12px; box-shadow:0 4px 16px rgba(0,0,0,.08)}
    .image-caption{font-size: 13px; color:#64748b; margin-top:6px; font-style: italic}

    .footer-actions{display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin-top: 22px; padding: 16px 0}
    .btn{background: var(--brand); color:#fff; border:0; padding:11px 18px; border-radius:10px;
         font-weight:700; box-shadow:0 2px 6px rgba(37,99,235,.25); cursor:pointer}
    .btn:hover{transform: translateY(-1px); box-shadow:0 6px 12px rgba(37,99,235,.32)}
    .btn.secondary{background:#64748b}

    :focus-visible{outline: 3px solid #a5b4fc; outline-offset: 3px}

    @media (prefers-color-scheme: dark){
      :root{ --bg:#0b1220; --text:#e6edf6; --muted:#9fb0c5; --border:#213047; --hi:#534900;}
      .toc-sidebar{background:#0f172a}
      .section-card{background:#0b1220}
      a{color:#8ab4ff}
      blockquote{background:#0e1a33; border-left-color:#5b7bff; color:#cdd7e5}
    }
    """
    js = """
    function downloadHTML(){
      const blob=new Blob([document.documentElement.outerHTML],{type:'text/html;charset=utf-8'});
      const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='easy_results.html';a.click();
      setTimeout(()=>URL.revokeObjectURL(a.href),1500);
    }
    function toggleHighlights(){
      document.querySelectorAll('mark').forEach(m=>{m.style.display=(m.style.display==='none'?'inline':'none')});
      const t=document.getElementById('toggleHi');
      if(t){t.textContent=t.textContent.includes('끄기')?'형광펜 켜기':'형광펜 끄기'}
    }
    """

    html_header = f"""<!DOCTYPE html><html lang="ko"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>POLO - 쉬운 논문 설명</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>{css}</style>
<script>{js}</script>
</head><body><div class="wrapper">
  <div class="header">
    <div class="title">POLO 논문 이해 도우미 변환 결과</div>
    <div class="subtitle">복잡한 논문을 쉽게 이해할 수 있도록 재구성한 결과</div>
    <div class="meta">논문 ID: {html.escape(paper_id)} | 생성: {html.escape(gen_at)}</div>
  </div>
  <div class="layout">
    <aside class="toc-sidebar"><div class="toc-title">목차</div><ol class="toc-list">
"""
    parts: List[str] = [html_header]
    # TOC
    section_num = 0; open_sub = False; subsection_num = 0
    for i, sec in enumerate(sections):
        title = (sec.get("title") or f"Section {i+1}").strip()
        sid = slugs[i]
        section_type = sec.get("section_type", "section")
        if section_type == "section":
            if open_sub:
                parts.append('</ol></li>'); open_sub = False
            section_num += 1; subsection_num = 0
            parts.append(f'<li class="toc-item"><a class="toc-link" href="#{sid}"><span class="num">{section_num}.</span>{html.escape(title)}</a>')
            parts.append('<ol class="toc-sublist">'); open_sub = True
        else:
            subsection_num += 1
            parts.append(f'<li class="toc-item"><a class="toc-link" href="#{sid}"><span class="num">{section_num}.{subsection_num}</span>{html.escape(title)}</a></li>')
    if open_sub:
        parts.append('</ol></li>')
    parts.append('</ol></aside><main class="content-area">')

    # 본문
    for i, sec in enumerate(sections):
        title = (sec.get("title") or f"Section {i+1}").strip()
        sid = slugs[i]
        section_type = sec.get("section_type", "section")
        res = results[i]
        raw_text = (getattr(res, "easy_text", None) or sec.get("content") or "").strip()
        processed_text = _split_paragraphs_ko(raw_text)
        content_html = _render_rich_html(processed_text)
        header_html = "" if _starts_with_same_heading(content_html, title) else f"<h2>{html.escape(title)}</h2>"
        parts.append(f'<div class="section-card {section_type}" id="{sid}">{header_html}<div class="content">{content_html}</div>')

        # 생성된 시각화 이미지
        if res.ok and res.image_path and Path(res.image_path).exists():
            src_path = Path(res.image_path)
            dst_path = output_path.parent / src_path.name
            try:
                import shutil; shutil.copy2(src_path, dst_path)
                logger.info(f"📊 [EASY] 시각화 이미지 복사 완료: {src_path.name}")
            except Exception as e:
                logger.warning(f"📊 [EASY] 시각화 이미지 복사 실패: {e}"); dst_path = src_path
            parts.append(f"""
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
                    parts.append(f"""
<div class="image-container">
  <img src="{img_info['filename']}" alt="{html.escape(caption)}" style="max-width:100%; height:auto; border-radius:8px;" />
  <div class="image-caption">원본 논문 그림 {img_idx+1}: {html.escape(caption)}</div>
</div>""")
                except Exception as e:
                    logger.warning(f"📊 [EASY] 원본 이미지 복사 실패: {e}")
        parts.append("</div>")  # section-card

    parts.append("""
<div class="footer-actions">
  <button class="btn" onclick="downloadHTML()">HTML 저장</button>
  <button class="btn secondary" id="toggleHi" onclick="toggleHighlights()">형광펜 끄기</button>
</div>""")
    parts.append('</main></div>')
    parts.append(f"<script>{js}</script>")
    parts.append("</div></body></html>")

    output_path.write_text("".join(parts), encoding="utf-8")

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
# === PART 6/7 =============================================================
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
            # 전달된 문자열이 실제 JSONL 컨텐츠일 때
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

    def _find_related_images(section_content: str, assets_metadata: Dict[str, Any], source_dir: Path, *, fallback_top_k: int = 3) -> List[Dict[str, Any]]:
        related_images = []
        if not assets_metadata:
            return related_images

        # 1) 섹션 안에서 \ref{...} 찾기
        fig_refs = []
        fig_refs.extend(re.findall(r'\\ref\{([^}]+)\}', section_content))
        fig_refs.extend(re.findall(r'\\label\{([^}]+)\}', section_content))

        try:
            server_dir = Path(__file__).resolve().parents[2] / "server"
        except Exception:
            server_dir = Path(".")
        extra_roots: List[Path] = [
            source_dir,
            server_dir / "data" / "out" / "transformer" / "source",
            server_dir / "data" / "out" / "transformer",
            server_dir / "data" / "arxiv",
            server_dir,
        ]
        img_exts = ['.png', '.jpg', '.jpeg', '.webp']  # pdf는 <img>가 못 그림

        def _resolve_one(graphic_path: str) -> Optional[Path]:
            p = Path(graphic_path)
            cand: List[Path] = []
            if p.is_absolute():
                cand.append(p)
            else:
                for root in extra_roots:
                    cand.append(root / p)
            # 확장자 없으면 이미지 확장자로 시도
            out = next((c for c in cand if c.exists() and c.suffix.lower() in img_exts), None)
            if out:
                return out
            # 확장자가 pdf 등인 경우, 같은 이름의 .png/.jpg 찾기
            base = p.stem
            for root in extra_roots:
                for ext in img_exts:
                    q = root / f"{base}{ext}"
                    if q.exists():
                        return q
            return None

        # 2) 참조 기반 우선 매칭
        for ref in fig_refs:
            if ref in assets_metadata:
                asset = assets_metadata[ref]
                for graphic_path in asset.get("graphics", []):
                    full = _resolve_one(graphic_path)
                    if full:
                        related_images.append({
                            "id": asset.get("id", ref),
                            "path": str(full),
                            "caption": asset.get("caption", ""),
                            "label": asset.get("label", ref),
                            "filename": full.name
                        })

        # 3) 아무것도 못 찾으면 상위 N개 자산을 자동 첨부
        if not related_images:
            for _, asset in list(assets_metadata.items())[:fallback_top_k]:
                for graphic_path in asset.get("graphics", []):
                    full = _resolve_one(graphic_path)
                    if full:
                        related_images.append({
                            "id": asset.get("id", ""),
                            "path": str(full),
                            "caption": asset.get("caption", ""),
                            "label": asset.get("label", ""),
                            "filename": full.name
                        })
                        break  # 한 자산당 1장

        # 4) 아무것도 못 찾으면 상위 N개 자산을 자동 첨부
        if not related_images and assets_metadata:
            for _, asset in list(assets_metadata.items())[:fallback_top_k]:
                for graphic_path in asset.get("graphics", []):
                    full = _resolve_one(graphic_path)
                    if full:
                        related_images.append({
                            "id": asset.get("id", ""),
                            "path": str(full),
                            "caption": asset.get("caption", ""),
                            "label": asset.get("label", ""),
                            "filename": full.name
                        })
                        break  # 한 자산당 1장

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
                # 한국어 보장 (주석은 _rewrite_text에서 1회 적용됨)
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

    # easy.json 구조에 맞게 변환
    easy_sections = []
    for i, section in enumerate(sections):
        result = results[i]
        section_title = section.get("title", f"Section {i+1}")
        
        # 문단들을 쉬운 설명에서 추출 (간단한 분할)
        easy_text = result.easy_text or ""
        paragraphs = []
        if easy_text:
            # 문단별로 분할 (간단한 휴리스틱)
            para_texts = [p.strip() for p in easy_text.split('\n\n') if p.strip()]
            for j, para_text in enumerate(para_texts):
                paragraphs.append({
                    "easy_paragraph_id": f"easy_paragraph_{i+1}_{j+1}",
                    "easy_paragraph_text": para_text,
                    "easy_paragraph_order": j + 1,
                    "easy_visualization_trigger": bool(result.image_path) and j == 0  # 첫 번째 문단에만 시각화 트리거
                })
        
        # 시각화 정보
        visualizations = []
        if result.image_path:
            visualizations.append({
                "easy_viz_id": f"easy_viz_{i+1}_1",
                "easy_viz_title": f"{section_title} 시각화",
                "easy_viz_description": f"{section_title}에 대한 시각적 설명",
                "easy_viz_image_path": f"/api/viz/{Path(result.image_path).name}",
                "easy_viz_type": "diagram"
            })
        
        easy_section = {
            "easy_section_id": f"easy_section_{i+1}",
            "easy_section_title": section_title,
            "easy_section_type": result.section_type or section.get("section_type", "section"),
            "easy_section_order": i + 1,
            "easy_section_level": 1,
            "easy_content": easy_text[:200] + "..." if len(easy_text) > 200 else easy_text,
            "easy_paragraphs": paragraphs,
            "easy_visualizations": visualizations
        }
        easy_sections.append(easy_section)
    
    # easy.json 구조로 변환
    easy_json = {
        "paper_info": {
            "paper_id": paper_id,
            "paper_title": f"논문 {paper_id}",
            "paper_authors": "Unknown",
            "paper_venue": "Unknown",
            "total_sections": len(sections)
        },
        "easy_sections": easy_sections,
        "metadata": {
            "generated_at": _get_current_datetime(),
            "easy_model_version": "yolo-easy-qlora-checkpoint-45",
            "total_processing_time": round((time.time() - t_batch_start), 2),
            "visualization_triggers": sum(1 for r in results if r.image_path),
            "total_paragraphs": sum(len(sec["easy_paragraphs"]) for sec in easy_sections),
            "section_types": {
                "section": len([s for s in easy_sections if s["easy_section_type"] == "section"]),
                "subsection": len([s for s in easy_sections if s["easy_section_type"] == "subsection"])
            }
        }
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

# === PART 7/7 =============================================================
# ---------- Glossary 관리/미리보기 ----------
@app.get("/glossary")
async def get_glossary_info():
    terms = _load_glossary()
    return {"path": str(GLOSSARY_PATH), "count": len(terms)}

@app.post("/glossary/reload")
async def reload_glossary():
    n = reload_glossary_cache()
    return {"reloaded": True, "count": n, "path": str(GLOSSARY_PATH)}

class GlossaryPreview(BaseModel):
    text: str

@app.post("/glossary/preview")
async def preview_glossary(req: GlossaryPreview):
    return {"annotated": annotate_terms_with_glossary(req.text or "")}

# ---------- 파일 경유 실행 (/from-transport) ----------
class TransportRequest(BaseModel):
    paper_id: str
    transport_path: str
    output_dir: Optional[str] = None
    style: Optional[str] = Field(default="three_para_ko")

ALLOWED_DATA_ROOT = Path(os.getenv("EASY_ALLOWED_ROOT", ".")).resolve()
TRANSPORT_MAX_MB  = int(os.getenv("EASY_TRANSPORT_MAX_MB", "50"))

def _is_under_allowed_root(p: Path) -> bool:
    try:
        return str(p.resolve()).startswith(str(ALLOWED_DATA_ROOT))
    except Exception:
        return False

def _read_text_guarded(p: Path, encoding="utf-8") -> str:
    if p.stat().st_size > TRANSPORT_MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다(>{TRANSPORT_MAX_MB}MB): {p.name}")
    return p.read_text(encoding=encoding, errors="ignore")

def _json_to_jsonl(s: str) -> str:
    try:
        obj = json.loads(s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {e}")
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
        elif ext == ".json":
            content = _json_to_jsonl(_read_text_guarded(tpath))
        else: raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {tpath.name}")

    req = BatchRequest(
        paper_id=request.paper_id,
        chunks_jsonl=content or "",
        output_dir=str(base_out),
        style=request.style,
    )
    return await run_batch(req, assets_metadata=assets_metadata)

# ---------- Run ----------
if __name__ == "__main__":
    host = os.getenv("EASY_HOST", "0.0.0.0")
    port = int(os.getenv("EASY_PORT", "5003"))
    reload_flag = os.getenv("EASY_RELOAD", "false").lower() in ("1","true","yes")
    uvicorn.run("app:app", host=host, port=port, reload=reload_flag, timeout_keep_alive=120)

