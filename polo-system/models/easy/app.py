# === PART 1/4 START ===
# -*- coding: utf-8 -*-
"""
POLO Easy Model - Grounded JSON Generator (Patched)
- Fix(1): token-based slicing (첫 글자 잘림 방지)
- Fix(2): decoding for stability (do_sample=False, no_repeat_ngram_size=4, repetition_penalty=1.2)
- Fix(3): explicit stop markers + safe cut
- Fix(4): LaTeX/table strip keeps sentence boundaries (" . ")
- Fix(5): sanitize repeats (e.g., *flops spam*)
- Fix(6): prompt policy (no invention; 'not stated in the original text'), + style=three_para_ko
"""

from __future__ import annotations
import os, re, json, time, base64, gzip, logging
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
HF_TOKEN           = os.getenv("HUGGINGFACE_TOKEN", "")
BASE_MODEL         = os.getenv("EASY_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
ADAPTER_DIR        = os.getenv("EASY_ADAPTER_DIR", str(Path(__file__).resolve().parent.parent / "fine-tuning" / "outputs" / "llama32-3b-qlora" / "checkpoint-4000"))
MAX_NEW_TOKENS     = int(os.getenv("EASY_MAX_NEW_TOKENS", "1200"))
VIZ_MODEL_URL      = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
EASY_CONCURRENCY   = int(os.getenv("EASY_CONCURRENCY", "8"))
EASY_BATCH_TIMEOUT = int(os.getenv("EASY_BATCH_TIMEOUT", "600"))

EASY_STRIP_MATH = os.getenv("EASY_STRIP_MATH", "1").lower() in ("1", "true", "yes")
EASY_FORCE_KO   = os.getenv("EASY_FORCE_KO", "1").lower() in ("1", "true", "yes")
EASY_AUTO_BOLD  = os.getenv("EASY_AUTO_BOLD", "1").lower() in ("1", "true", "yes")
EASY_HILITE     = os.getenv("EASY_HILITE", "1").lower() in ("1", "true", "yes")

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# -------------------- FastAPI --------------------
app = FastAPI(title="POLO Easy Model", version="1.5.0-patched")

# -------------------- Global state --------------------
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_available = torch.cuda.is_available()
safe_dtype = torch.float16 if gpu_available else torch.float32

# -------------------- Utils --------------------
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

def _coerce_json(text: str) -> Dict[str, Any]:
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s:e+1]
    return json.loads(text)

def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))

def _detect_lang_safe(text: str) -> str:
    if not text or not text.strip():
        return "en"
    if _contains_hangul(text): return "ko"
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total_chars = len(re.findall(r"[A-Za-z가-힣]", text))
    if total_chars == 0: return "en"
    return "ko" if (latin_chars/total_chars) < 0.5 else "en"

# === Fix(4): Keep sentence boundaries when stripping ===
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

def _normalize_bracket_tokens(text: str) -> str:
    if not text: return text
    text = re.sub(r"(?i)\bL\s*R\s*B\b", "(", text)
    text = re.sub(r"(?i)\bL\s*L\s*B\b", "(", text)
    text = re.sub(r"(?i)\bR\s*R\s*B\b", ")", text)
    return text

def _postprocess_terms(text: str) -> str:
    if not text: return text
    text = re.sub(r"(?i)\bY\s*O\s*L\s*O\b", "YOLO", text)
    text = re.sub(r"(?i)\bO\s*L\s*O\b", "YOLO", text)
    text = re.sub(r"(?i)\bm\s*A\s*P\b", "mAP", text)
    text = re.sub(r"(?i)\bR\s*-?\s*CNN\b", "R-CNN", text)
    text = re.sub(r"(?i)\bFast\s*-?\s*R\s*-?\s*CNN\b", "Fast R-CNN", text)
    text = re.sub(r"(?i)\bFaster\s*-?\s*R\s*-?\s*CNN\b", "Faster R-CNN", text)
    return text

def _auto_bold_terms(text: str) -> str:
    if not text or not EASY_AUTO_BOLD: return text
    patterns = [
        r"\b(?:YOLOv?\d+|YOLO|RetinaNet|Faster\s*R-?CNN|Fast\s*R-?CNN|R-?CNN|ResNet-?\d+|EfficientNet[-A-Z0-9]*)\b",
        r"\b(?:Transformer|ViT|CLIP|BERT|GPT(?:-\d+)?)\b",
        r"\b(?:mAP|IoU|F1|AP50|AP75|AUC|ROC|BLEU|ROUGE|PSNR|SSIM|CER|WER)\b",
        r"\b\d+\s?[x×]\s?\d+\b",
        r"\b(?:Top-1|Top-5|SOTA|SotA|FPS|GFLOPs?|Params?)\b",
        r"\b(?:Llama|LLaMA|Mistral|Phi|Qwen|Gemma|Whisper)\b",
    ]
    def wrap(m: re.Match) -> str:
        g = m.group(0)
        if g.startswith("**") and g.endswith("**"): return g
        return f"**{g}**"
    for pat in patterns:
        text = re.sub(pat, wrap, text)
    return text

def _hilite_sentences(text: str, max_marks: int = 2) -> str:
    if not text or not EASY_HILITE: return text
    parts = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+", text.strip())
    marks = 0; out = []
    key = re.compile(r"(?:기여|핵심|요약|결론|성능|개선|향상|정확도|우수|달성|증가|감소|한계|제안|우리\s*모델|본\s*연구|SOTA|improv|achiev|propos|contribut)", re.IGNORECASE)
    has_number = re.compile(r"\d+(?:\.\d+)?\s*(?:%|점|배|ms|s|fps|GFLOPs?|M|K)?", re.IGNORECASE)
    for sent in parts:
        s = sent.strip()
        if not s: continue
        if marks < max_marks and (key.search(s) or has_number.search(s)):
            out.append(f"=={s}=="); marks += 1
        else:
            out.append(s)
    return " ".join(out)

# === Fix(5): sanitize pathological repeats (e.g., flops spam) ===
def _sanitize_repeats(text: str) -> str:
    if not text: return text
    # 같은 토큰 4회 이상 반복 → 3회로 축소
    text = re.sub(r"(?:\b[\w\-가-힣]{2,}\b)(?:\s+\1){3,}", r"\1 \1 \1", text, flags=re.IGNORECASE)
    # flops 도배 제거 (10회 이상 연속)
    text = re.sub(r"(?:\b[\w\-]*flops\b[\s,.;:]*){10,}", "", text, flags=re.IGNORECASE)
    return text
# === PART 1/4 END ===

# === PART 2/4 START ===
# -------------------- Translation (LLM) --------------------
def _ensure_korean(text: str) -> str:
    if not text or not text.strip(): return text
    lang = _detect_lang_safe(text)
    latin = len(re.findall(r"[A-Za-z]", text))
    hangul = len(re.findall(r"[가-힣]", text))
    high_latin_ratio = latin > 0 and (hangul == 0 or latin / max(1, latin + hangul) >= 0.4)
    if lang == "ko" and not (EASY_FORCE_KO and high_latin_ratio): return text
    if lang == "en" or (EASY_FORCE_KO and high_latin_ratio): return _translate_to_korean(text)
    if EASY_FORCE_KO and hangul < latin: return _translate_to_korean(text)
    return text

# Translation prompt uses the model itself
def _translate_to_korean(text: str) -> str:
    try:
        if not text or not text.strip(): return ""
        if model is None or tokenizer is None:
            logger.warning("모델 미로드 → 번역 스킵")
            return text
        translate_prompt = (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            "너는 전문 번역가다. 영어를 자연스럽고 정확한 한국어로 번역하되, "
            "전문 용어는 원어 유지 + 괄호 해설을 덧붙인다. 수식/표/그림 언급은 제거.\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"[INPUT]\n{text}\n\n[OUTPUT]\n"
            "<|eot_id|>\n"
        )
        inputs = tokenizer(translate_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(MAX_NEW_TOKENS, 700),
                do_sample=False,                    # 안정 우선
                use_cache=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # === Fix(1): token-based slicing ===
        seq = outputs[0]
        inp_len = inputs["input_ids"].shape[1]
        gen_tokens = seq[inp_len:]
        result = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        # === Fix(3): Stop cut ===
        for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
            p = result.find(stop)
            if p != -1:
                result = result[:p].strip()
                break
        result = _postprocess_terms(result)
        result = _strip_math_all(result); result = _strip_tables_figures_all(result)
        result = _sanitize_repeats(result)
        return result or text
    except Exception as e:
        logger.warning(f"LLM 번역 실패: {e}")
        return text

# -------------------- I/O Models --------------------
class TextRequest(BaseModel):
    text: str
    translate: Optional[bool] = False
    style: Optional[str] = Field(default="default", description="easy 스타일: 'default' | 'three_para_ko'")
    model_config = ConfigDict(extra="allow")

class TextResponse(BaseModel):
    simplified_text: str
    translated_text: Optional[str] = None

class BatchRequest(BaseModel):
    paper_id: str = Field(..., description="결과 파일/경로 식별자")
    chunks_jsonl: str = Field(..., description="각 라인에 {'text': ...} 형태의 JSONL 또는 경로/폴더")
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

# -------------------- Model Load --------------------
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

@app.on_event("startup")
async def startup_event():
    load_model()

# -------------------- Prompts --------------------
def _clean_latex_text(text: str) -> str:
    if not text: return ""
    s = text
    if EASY_STRIP_MATH: s = _strip_math_all(s)
    s = _strip_tables_figures_all(s)
    s = re.sub(r"\\cite\{[^}]*\}|\\label\{[^}]*\}|\\ref\{[^}]*\}", " ", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"(\1)", s)
    s = re.sub(r"\\(textbf|textit|emph)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"^\\(sub)*section\{[^}]*\}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s+", " ", s); s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

# === Fix(6): strict policy to avoid invention; style support ===
def _build_easy_prompt(text: str, section_title: str | None = None) -> str:
    title_line = f"[Section] {section_title}\n\n" if section_title else ""
    system_block = (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a professional science communicator.\n"
        "Rewrite to simple, readable English for beginners.\n\n"
        "Policy (DO NOT VIOLATE):\n"
        "- Use ONLY information from [Original Text].\n"
        "- Never invent numbers, tables, versions, or comparisons not present.\n"
        "- If something is not in [Original Text], write: 'not stated in the original text'.\n"
        "- Exclude equations/tables/figures.\n"
        "<|eot_id|>\n"
    )
    user_block = (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{title_line}[Original Text]\n{text}\n\n[OUTPUT]\n"
        "<|eot_id|>\n"
    )
    return system_block + user_block

def _build_easy_prompt_three_para_ko(text: str, section_title: str | None = None) -> str:
    title_line = f"[섹션] {section_title}\n\n" if section_title else ""
    system = (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "너는 과학/AI 내용을 중학생도 이해하게 한국어로 아주 쉽게 설명한다.\n"
        "정확히 3단락(각 1~2문장). 1)무엇 2)왜(장점/빠름) 3)어디에 쓰나.\n"
        "원문 밖 정보 금지. 원문에 없으면 '원문에 없음'이라고 적는다.\n"
        "<|eot_id|>\n"
    )
    user = (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{title_line}[원문]\n{text}\n\n[OUTPUT]\n"
        "<|eot_id|>\n"
    )
    return system + user
# === PART 2/4 END ===

# === PART 3/4 START ===
# -------------------- Core Rewriter --------------------
async def _rewrite_text(text: str, section_title: str = None, context_info: str = None, style: str = "default") -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("모델이 로드되지 않았습니다")
    cleaned_text = _clean_latex_text(text)
    if context_info:
        cleaned_text = f"[문맥] {context_info}\n\n" + cleaned_text

    if style == "three_para_ko":
        prompt = _build_easy_prompt_three_para_ko(cleaned_text, section_title)
    else:
        prompt = _build_easy_prompt(cleaned_text, section_title)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(MAX_NEW_TOKENS, 700),
            do_sample=False,                 # === Fix(2): 안정 모드
            use_cache=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # === Fix(1): token-based slicing ===
    seq = outputs[0]
    inp_len = inputs["input_ids"].shape[1]
    gen_tokens = seq[inp_len:]
    result = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # === Fix(3): stop marker cut ===
    for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
        p = result.find(stop)
        if p != -1:
            result = result[:p].strip()
            break

    # post
    result = _normalize_bracket_tokens(result)
    result = _postprocess_terms(result)
    if EASY_STRIP_MATH: result = _strip_math_all(result)
    result = _strip_tables_figures_all(result)
    result = _auto_bold_terms(result)
    result = _hilite_sentences(result, max_marks=2)
    result = _sanitize_repeats(result)
    return result

# -------------------- HTML helpers (unchanged behavior) --------------------
def _get_current_datetime() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")

def _slugify(s: str, fallback: str) -> str:
    s = re.sub(r"[^0-9A-Za-z가-힣\- ]", "", s or "")
    s = s.strip().replace(" ", "-")
    return s if s else fallback

def _md_to_html(md: str) -> str:
    if not md: return ""
    def _codeblock_repl(m): return f"<pre><code>{m.group(1)}</code></pre>"
    md = re.sub(r"```([\s\S]*?)```", _codeblock_repl, md)
    md = re.sub(r"^###\s*(.+)$", r"<h3>\1</h3>", md, flags=re.MULTILINE)
    md = re.sub(r"^##\s*(.+)$",  r"<h2>\1</h2>", md, flags=re.MULTILINE)
    md = re.sub(r"^#\s*(.+)$",   r"<h1>\1</h1>", md, flags=re.MULTILINE)
    md = re.sub(r"(?<!\$)\*\*([^*$]+?)\*\*(?!\$)", r"<strong>\1</strong>", md)
    md = re.sub(r"(?<!\$)\*([^*$]+?)\*(?!\$)",     r"<em>\1</em>", md)
    md = re.sub(r"`([^`]+?)`",                    r"<code>\1</code>", md)
    md = re.sub(r"==([^=]+?)==",                  r"<mark>\1</mark>", md)
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
    html = "\n".join(out)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", html) if b.strip()]
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

def _starts_with_same_heading(html: str, title: str) -> bool:
    if not title or not html: return False
    plain = re.sub(r"<[^>]+>", "", html).strip().lower()
    t = (title or "").strip().lower()
    return plain.startswith(t) or plain[:120].startswith(t + ":")

def _save_html_results(sections: List[dict], results: List[VizResult], output_path: Path, paper_id: str):
    # TOC 준비
    toc_items: List[Tuple[str, str]] = []  # (id, title)

    def _split_paragraphs_ko(text: str, min_s: int = 3, max_s: int = 5, max_chars: int = 700) -> str:
        import re as _re
        if not text:
            return ""
        s = _re.sub(r"\s+", " ", str(text).strip())
        parts = _re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+", s)
        parts = [p.strip() for p in parts if p and p.strip()]
        paras: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for sent in parts:
            start_new = (len(cur) >= min_s and (len(cur) >= max_s or cur_len + len(sent) > max_chars))
            if start_new and cur:
                paras.append(" ".join(cur))
                cur = [sent]
                cur_len = len(sent)
            else:
                cur.append(sent)
                cur_len += len(sent)
        if cur:
            paras.append(" ".join(cur))
        out: List[str] = []
        for p in paras:
            if out and len(p) < 80:
                out[-1] = (out[-1] + " " + p).strip()
            else:
                out.append(p)
        return "\n\n".join(out)

    # -------- HTML 헤더(일반 문자열 + 토큰치환) --------
    gen_at = _get_current_datetime()
    html_header = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>POLO - 쉬운 논문 설명</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#f7fafc; --card:#ffffff; --text:#111827; --muted:#6b7280; --brand:#2563eb; --hi:#fff7a1;
  --code-bg:#0f172a; --code-fg:#e5e7eb;
  --sidebar-w:260px;
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body {
  font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo','Segoe UI',Roboto,Arial,sans-serif;
  background:var(--bg); color:var(--text); line-height:1.85; margin:0; padding:24px;
}
.wrapper { max-width:1100px; margin:0 auto; }
.header {
  background:var(--card); border:1px solid #e5e7eb; border-radius:12px;
  padding:22px 24px; margin-bottom:16px;
}
.title { font-weight:700; font-size:22pt; margin:0 0 6px; }
.subtitle { color:var(--muted); font-size:12.5pt; }
.meta { color:var(--muted); font-size:11pt; margin-top:6px; }

/* 레이아웃: 좌측 고정 TOC + 우측 본문 */
.layout { display:grid; grid-template-columns:minmax(200px,var(--sidebar-w)) 1fr; gap:24px; align-items:start; }
.toc-sidebar {
  position:sticky; top:20px; max-height:calc(100vh - 40px); overflow:auto;
  background:var(--card); border:1px solid #e5e7eb; border-radius:12px; padding:16px 14px;
}
.toc-title { font-weight:700; font-size:13.5pt; margin:2px 0 10px; }
.toc-list, .toc-sublist { list-style:none; margin:0; padding:0; }
.toc-item { margin:6px 0; }
.toc-link {
  display:block; text-decoration:none; color:#1f2937; padding:2px 4px; border-radius:6px;
  font-weight:600; font-size:12.5pt;
}
.toc-link .num { color:var(--muted); margin-right:6px; font-weight:700; }
.toc-sublist .toc-link { font-weight:500; font-size:11.5pt; color:#374151; padding-left:6px; }
.toc-link:hover { background:#f3f4f6; }
.toc-link.active { background:#eef2ff; color:#111827; box-shadow:0 0 0 2px rgba(37,99,235,.15) inset; }

.content-area { min-width:0; }

.section-card {
  background:var(--card); border:1px solid #e5e7eb; border-radius:12px;
  padding:22px 22px; margin-bottom:18px;
}
.section-card.section { border-left:4px solid var(--brand); box-shadow:0 2px 8px rgba(37,99,235,.08); }
.section-card.subsection { border-left:4px solid #10b981; box-shadow:0 2px 8px rgba(16,185,129,.08); margin-left:0; margin-right:0; }
.section-card h2 { font-size:19pt; margin:2px 0 12px; font-weight:700; letter-spacing:.2px; }
.section-card.subsection h2 { font-size:16pt; color:#059669; position:relative; padding-left:0; }
.content p { margin:10px 0 12px; }
.content ul { margin:8px 0 12px 18px; }
.content li { margin:4px 0; }
.content code {
  background:#f6f8fa; padding:2px 6px; border-radius:4px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace; font-size:12pt;
}
.content pre { background:var(--code-bg); color:var(--code-fg); padding:14px; border-radius:8px; overflow:auto; }
.content strong { font-weight:700; }
.content mark { background:var(--hi); padding:0 .2em; border-radius:3px; box-shadow:0 0 0 2px rgba(255,247,161,.4) inset; }

.image-container { text-align:center; margin:18px 0 6px; padding:12px; border:1px dashed #e5e7eb; border-radius:10px; }
.image-caption { margin-top:6px; color:var(--muted); font-size:10.5pt; font-style:italic; }

.footer-actions { position:fixed; bottom:16px; right:16px; display:flex; gap:8px; z-index:5; }
.btn { appearance:none; border:none; padding:9px 14px; border-radius:10px; font-weight:700; color:white; background:#2563eb; cursor:pointer; box-shadow:0 6px 18px rgba(37,99,235,.24); }
.btn.secondary { background:#111827; }

@media print {
  .footer-actions, .toc-sidebar { display:none; }
  body { background:white; padding:0; }
  .header, .section-card { border:none; box-shadow:none; }
}
@media (max-width:980px){
  .layout { grid-template-columns:1fr; }
  .toc-sidebar { position:relative; top:auto; max-height:none; }
}
/* 형광펜 토글 */
.hide-hilite mark { background:transparent; box-shadow:none; }
</style>
<script>
function downloadHTML(){
  const b=new Blob([document.documentElement.outerHTML],{type:'text/html'});
  const u=URL.createObjectURL(b); const a=document.createElement('a');
  a.href=u; a.download='polo_easy_explanation___PAPER_ID__.html';
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(u);
}
function toggleHighlights(){
  document.body.classList.toggle('hide-hilite');
  const on=!document.body.classList.contains('hide-hilite');
  document.getElementById('toggleHi').innerText=on?'형광펜 끄기':'형광펜 켜기';
}
document.addEventListener('DOMContentLoaded',()=>{
  const links=[...document.querySelectorAll('.toc-link')];
  const map=new Map(links.map(a=>[a.getAttribute('href').slice(1),a]));
  const obs=new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      const id=e.target.id, a=map.get(id);
      if(!a) return;
      if(e.isIntersecting){
        links.forEach(x=>x.classList.remove('active'));
        a.classList.add('active');
      }
    });
  },{rootMargin:'0px 0px -70% 0px',threshold:0.1});
  window.__attachObserver=()=>document.querySelectorAll('.section-card[id]').forEach(sec=>obs.observe(sec));
});
</script>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="title">POLO 논문 이해 도우미 변환 결과</div>
    <div class="subtitle">복잡한 논문을 쉽게 이해할 수 있도록 재구성한 결과</div>
    <div class="meta">논문 ID: __PAPER_ID__ | 생성: __GEN_AT__</div>
  </div>
"""

    html: List[str] = []
    html.append(html_header.replace("__PAPER_ID__", paper_id).replace("__GEN_AT__", gen_at))

    # 레이아웃 시작: 좌측 TOC + 우측 본문
    html.append('<div class="layout">')

    # ===== 좌측 고정 TOC =====
    html.append('<aside class="toc-sidebar">')
    html.append('<div class="toc-title">목차</div>')
    html.append('<ol class="toc-list">')

    section_num = 0
    open_sub = False
    subsection_num = 0

    for i, sec in enumerate(sections):
        title = (sec.get("title") or f"Section {i+1}").strip()
        sid = _slugify(title, f"sec-{i+1}")
        section_type = sec.get("section_type", "section")
        toc_items.append((sid, title))

        if section_type == "section":
            if open_sub:
                html.append('</ol></li>')  # 이전 섹션 서브리스트 닫기
                open_sub = False
            section_num += 1
            subsection_num = 0
            html.append(
                f'<li class="toc-item"><a class="toc-link" href="#{sid}">'
                f'<span class="num">{section_num}.</span>{title}</a>'
            )
            html.append('<ol class="toc-sublist">')  # 서브섹션 리스트 시작
            open_sub = True
        else:
            subsection_num += 1
            html.append(
                f'<li class="toc-item"><a class="toc-link" href="#{sid}">'
                f'<span class="num">{section_num}.{subsection_num}</span>{title}</a></li>'
            )

    if open_sub:
        html.append('</ol></li>')  # 마지막 섹션 서브리스트 닫기
    html.append('</ol>')
    html.append('</aside>')  # /toc-sidebar

    # ===== 우측 본문 시작 =====
    html.append('<main class="content-area">')

    # 섹션들
    for i, (sec, res) in enumerate(zip(sections, results)):
        title = (sec.get("title") or f"Section {i+1}").strip()
        sid = toc_items[i][0]
        section_type = sec.get("section_type", "section")

        # 우선순위: viz.easy_text -> sec.content
        raw_text = (getattr(res, "easy_text", None) or sec.get("content") or "").strip()

        def _split_ko_local(tx: str) -> str:
            return _split_paragraphs_ko(tx)

        processed_text = _split_ko_local(raw_text)
        content_html = _render_rich_html(processed_text)

        header_html = "" if _starts_with_same_heading(content_html, title) else f"<h2>{title}</h2>"

        html.append(
            f'<div class="section-card {section_type}" id="{sid}">{header_html}'
            f'<div class="content">{content_html}</div>'
        )

        # (옵션) 이미지
        if res.ok and res.image_path and Path(res.image_path).exists():
            src_path = Path(res.image_path)
            dst_path = output_path.parent / src_path.name
            try:
                import shutil
                shutil.copy2(src_path, dst_path)
                logger.info(f"📊 [EASY] 이미지 복사 완료: {src_path.name}")
            except Exception as e:
                logger.warning(f"📊 [EASY] 이미지 복사 실패: {e}")
                dst_path = Path("../../viz") / paper_id / src_path.name

            html.append(f"""
<div class="image-container">
  <img src="{dst_path.name}" alt="{title} 관련 시각화" style="max-width:100%; height:auto; border-radius:8px;" />
  <div class="image-caption">그림 {i+1}: {title} 관련 시각화</div>
</div>""")

        html.append("</div>")  # /section-card

    # 플로팅 액션
    html.append("""
<div class="footer-actions">
  <button class="btn" onclick="downloadHTML()">HTML 저장</button>
  <button class="btn secondary" id="toggleHi" onclick="toggleHighlights()">형광펜 끄기</button>
</div>
""")

    # 레이아웃/본문 닫기 + Observer attach
    html.append('</main></div>')  # </main></div.layout>
    html.append("<script>if(window.__attachObserver) window.__attachObserver();</script>")
    html.append("</div></body></html>")

    output_path.write_text("".join(html), encoding="utf-8")


# === PART 3/4 END ===

# === PART 4/4 START ===
# -------------------- Endpoints --------------------
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
    simplified_text = await _rewrite_text(request.text, style=request.style or "default")

    # style==three_para_ko면 이미 한국어 → 번역 생략
    if (request.style or "default") == "three_para_ko":
        out_text = simplified_text
    else:
        out_text = _ensure_korean(simplified_text) if request.translate or EASY_FORCE_KO else simplified_text
    return TextResponse(simplified_text=out_text, translated_text=None)

@app.post("/generate")
async def generate_json(request: TextRequest):
    start_total = time.time()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

    # 간이 섹션 추출(원본 코드 유지)
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

    data_schema = json.loads(json.dumps(GROUND_SCHEMA))
    for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
        data_schema[k]["original"] = extracted.get(k, "")

    instruction = (
        "너는 친근한 과학 선생님이다. 아래 JSON 스키마의 '키/구조'를 그대로 두고 '값'만 채워라. "
        "외부 지식/추측 금지. 원문에 없으면 '원문에 없음'이라고 적어라. "
        "출력은 오직 '유효한 JSON' 하나만 허용된다."
    )
    schema_str = json.dumps(data_schema, ensure_ascii=False, indent=2)
    context_only = {k: extracted[k] for k in ["abstract","introduction","methods","results","discussion","conclusion"]}
    context_str = json.dumps(context_only, ensure_ascii=False, indent=2)

    prompt = f"{instruction}\n\n=== 출력 스키마 ===\n{schema_str}\n\n=== 섹션 원문 ===\n{context_str}\n\n[OUTPUT]\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.2,         # === Fix(2)
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - t0

    # === Fix(1): token-based slicing ===
    seq = outputs[0]
    inp_len = inputs["input_ids"].shape[1]
    gen_tokens = seq[inp_len:]
    raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # === Fix(3): stop cut ===
    for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
        p = raw.find(stop)
        if p != -1:
            raw = raw[:p].strip()
            break

    def _is_meaningful(d: dict) -> bool:
        try:
            sections = ["abstract","introduction","methods","results","discussion","conclusion"]
            return any(len((d.get(s, {}) or {}).get("easy", "")) > 10 for s in sections)
        except Exception:
            return False

    try:
        data = _coerce_json(raw)
        if not _is_meaningful(data):
            raise ValueError("empty_json")
    except Exception:
        strict_instruction = (
            "스키마의 키/구조를 유지하고 값만 채운 '유효한 JSON'만 출력하라. 반드시 '{'로 시작해 '}'로 끝나야 한다."
        )
        strict_prompt = f"{strict_instruction}\n\n스키마:\n{schema_str}\n\n섹션 원문:\n{context_str}\n\n[OUTPUT]\n"
        inputs2 = tokenizer(strict_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        with torch.inference_mode():
            outputs2 = model.generate(
                **inputs2,
                max_new_tokens=min(MAX_NEW_TOKENS, 800),
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        seq2 = outputs2[0]
        inp_len2 = inputs2["input_ids"].shape[1]
        gen_tokens2 = seq2[inp_len2:]
        raw2 = tokenizer.decode(gen_tokens2, skip_special_tokens=True).strip()
        for stop in ("[/OUTPUT]", "[/output]", "<|eot_id|>"):
            p = raw2.find(stop)
            if p != -1:
                raw2 = raw2[:p].strip()
                break
        try:
            data = _coerce_json(raw2)
        except Exception:
            data = data_schema

    total_time = time.time() - start_total   # ← 경과시간으로 수정
    data["processing_info"] = {
        "gpu_used": gpu_available,
        "inference_time": inference_time,
        "total_time": total_time,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    # postprocess easy fields
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
                data[k]["easy"] = t
        except Exception:
            pass

    return data

# -------------------- (옵션) batch/transport/viz 파트 --------------------
# ====================== BATCH / TRANSPORT / VIZ (Optimized for KO-Easy) ======================

# ENV defaults (if not defined above)
EASY_VIZ_ENABLED = os.getenv("EASY_VIZ_ENABLED", "0").lower() in ("1", "true", "yes")

# ---------- LaTeX 파서: section/subsection 추출 + 표 라이트 파싱 ----------
def _extract_table_data(text: str) -> List[dict]:
    """LaTeX tabular에서 간단 메트릭 테이블을 추출 (옵션)"""
    tables = []
    pat = r'\\begin\{tabular\}[^{]*\{([^}]+)\}(.*?)\\end\{tabular\}'
    for m in re.finditer(pat, text, re.DOTALL):
        content = m.group(2)
        lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith('\\hline')]
        if len(lines) < 2:  # header + at least 1 row
            continue
        headers = [h.strip().rstrip('\\') for h in lines[0].split('&')]
        rows = []
        for ln in lines[1:]:
            if '&' in ln:
                row = [c.strip().rstrip('\\') for c in ln.split('&')]
                if len(row) == len(headers):
                    rows.append(row)
        if not rows: 
            continue
        tables.append({
            "type": "metric_table",
            "headers": headers,
            "rows": rows,
        })
    return tables

def _parse_latex_sections(tex_path: Path) -> List[dict]:
    """
    merged_body.tex 같은 LaTeX 본문을 읽어
    - \\section{}, \\subsection{} 단위로 분할
    - section_type: section | subsection
    - 각 섹션 raw_content에서 표 정보(옵션) 추출
    """
    content = tex_path.read_text(encoding="utf-8", errors="ignore")

    sections: List[dict] = []
    cur_title = None
    cur_buf: List[str] = []
    cur_raw: List[str] = []
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

    # header regex
    sec_pat  = re.compile(r"^\s*\\section\*?\{([^}]+)\}\s*$")
    sub_pat  = re.compile(r"^\s*\\subsection\*?\{([^}]+)\}\s*$")
    abs_beg  = re.compile(r"^\s*\\begin\{abstract\}")
    abs_end  = re.compile(r"^\s*\\end\{abstract\}")

    in_abstract = False
    lines = content.splitlines()
    for ln in lines:
        # abstract
        if abs_beg.match(ln):
            _flush()
            cur_title = "Abstract"
            cur_buf, cur_raw = [], []
            section_type = "section"
            in_abstract = True
            continue
        if abs_end.match(ln):
            _flush()
            cur_title = None
            cur_buf, cur_raw = [], []
            in_abstract = False
            continue

        # section/subsection
        m1 = sec_pat.match(ln)
        m2 = sub_pat.match(ln)
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

        # collect
        if cur_title is None:
            # 문서 시작부(타이틀/저자) 스킵
            if not any(k in ln.lower() for k in ["\\title", "\\author", "\\date", "\\maketitle"]):
                # 첫 섹션 없으면 초반부를 Introduction으로 승격
                cur_title = "Introduction"
                section_type = "section"
        if cur_title is not None:
            cur_buf.append(ln)
            cur_raw.append(ln)

    _flush()

    if not sections:
        # fallback: 전체 문서를 하나의 섹션으로
        clean = _clean_latex_text(content)
        sections = [{
            "index": 0,
            "title": "Full Document",
            "content": clean,
            "raw_content": content,
            "table_data": _extract_table_data(content),
            "section_type": "section",
        }]

    logger.info(f"[EASY] Parsed {len(sections)} sections from LaTeX")
    return sections

# ---------- (옵션) 시각화 엔진 호출 ----------
async def _send_to_viz(paper_id: str, index: int, easy_text_ko: str, out_dir: Path, table_data: List[dict] = None) -> Tuple[bool, Optional[str]]:
    """
    시각화 서버(VIZ_MODEL_URL)로 텍스트를 보내 이미지 생성 (옵션).
    - EASY_VIZ_ENABLED=0이면 텍스트만.
    - 실패해도 실패로 막지 않고 (ok=False, image=None) 로 리턴.
    """
    if not EASY_VIZ_ENABLED:
        return True, None

    # 간단 health probe (실패시 텍스트만)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(f"{VIZ_MODEL_URL.rstrip('/')}/health")
            if r.status_code != 200:
                return True, None
    except Exception:
        return True, None

    # payload 구성
    payload = {
        "paper_id": paper_id,
        "index": index,
        "rewritten_text": easy_text_ko,
        "target_lang": "ko",
        "bilingual": "missing",
        "text_type": "easy_korean",
    }
    # 표 정보가 있으면 간단히 실어보내 힌트 제공
    if table_data:
        payload["tables"] = table_data

    try:
        timeout = httpx.Timeout(60.0, connect=10.0)
        limits  = httpx.Limits(max_keepalive_connections=2, max_connections=5)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            r = await client.post(f"{VIZ_MODEL_URL.rstrip('/')}/viz", json=payload)
            if r.status_code != 200:
                return True, None
            data = r.json()
            # image 저장
            img_path = None
            if data.get("image_base64"):
                img_path = out_dir / f"{index:06d}.png"
                img_path.write_bytes(base64.b64decode(data["image_base64"]))
                return True, str(img_path)
            if data.get("image_path"):
                # 원경로 복사 대신 HTML 옆 폴더로 링크해도 됨
                return True, data["image_path"]
            return True, None
    except Exception as e:
        logger.warning(f"[EASY] viz error: {e}")
        return False, None

# ---------- 배치 실행 (/batch) ----------
class BatchRequest(BaseModel):
    paper_id: str = Field(..., description="결과 파일/경로 식별자")
    chunks_jsonl: str = Field(..., description="JSONL 내용 문자열 또는 경로(파일/디렉토리/tex)")
    output_dir: str = Field(..., description="결과 저장 루트")
    style: Optional[str] = Field(default="three_para_ko", description="easy 스타일 (default|three_para_ko)")

class VizResult(BaseModel):
    ok: bool = True
    index: int
    image_path: Optional[str] = None
    error: Optional[str] = None
    easy_text: Optional[str] = None
    section_title: Optional[str] = None
    section_type: Optional[str] = None

class BatchResult(BaseModel):
    ok: bool
    paper_id: str
    count: int
    success: int
    failed: int
    out_dir: str
    images: List[VizResult]

@app.post("/batch", response_model=BatchResult)
async def run_batch(request: BatchRequest):
    """
    입력:
      - request.chunks_jsonl:
          * JSONL 문자열(각 라인 {"title":..., "text":...})
          * 파일 경로(.jsonl/.jsonl.gz/.tex)
          * 디렉토리(내부에 merged_body.tex 또는 *.jsonl 탐색)
    처리:
      1) 섹션 파싱(문단 클렌징)
      2) 섹션별 쉬운 한국어 변환(_rewrite_text, style 기본 three_para_ko)
      3) (옵션) 시각화 호출
      4) easy_results.json / easy_results.html 저장
    """
    paper_id = request.paper_id.strip()
    base_out = Path(request.output_dir).expanduser().resolve()
    out_dir  = base_out / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 입력 소스 로딩
    src = (request.chunks_jsonl or "").strip()
    src_path: Optional[Path] = None
    if src:
        p = Path(src)
        if p.exists():
            src_path = p.resolve()

    sections: List[dict] = []

    def _append_from_jsonl_lines(lines: List[str]):
        for idx, ln in enumerate(lines):
            try:
                obj = json.loads(ln)
            except Exception as e:
                obj = {"title": f"Section {idx+1}", "text": f"[JSONL parse error] {e}"}
            sections.append({
                "index": idx,
                "title": obj.get("title") or f"Section {idx+1}",
                "content": obj.get("text") or "",
                "raw_content": obj.get("text") or "",
                "table_data": [],
                "section_type": "section",
            })

    try:
        if src_path is None:
            # JSONL 내용을 직접 넘긴 케이스
            lines = [ln for ln in src.splitlines() if ln.strip()]
            _append_from_jsonl_lines(lines)
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
                    _append_from_jsonl_lines([ln for ln in lines if ln.strip()])
            else:
                if src_path.suffix.lower() in (".jsonl", ".ndjson"):
                    lines = src_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    _append_from_jsonl_lines([ln for ln in lines if ln.strip()])
                elif src_path.suffix.lower() == ".gz":
                    with gzip.open(src_path, "rt", encoding="utf-8") as f:
                        lines = [ln for ln in f if ln.strip()]
                    _append_from_jsonl_lines(lines)
                elif src_path.suffix.lower() == ".tex":
                    sections = _parse_latex_sections(src_path)
                else:
                    raise ValueError("지원하지 않는 입력 형식 (jsonl/jsonl.gz/tex/dir)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"입력 파싱 실패: {e}")

    if not sections:
        raise HTTPException(status_code=400, detail="처리할 섹션 없음")

    logger.info(f"[EASY] batch sections = {len(sections)} (paper_id={paper_id})")

    # 2) 섹션별 변환 + (3) 시각화
    sem = anyio.Semaphore(EASY_CONCURRENCY)
    results: List[VizResult] = [VizResult(ok=False, index=i) for i in range(len(sections))]

    async def _work(i: int, sec: dict):
        async with sem:
            title = sec.get("title") or f"Section {i+1}"
            stype = sec.get("section_type", "section")
            try:
                context_info = f"{len(sections)}개 중 {i+1}번째 섹션. 유형: {stype}."
                easy_text = await _rewrite_text(sec.get("content", ""), title, context_info, style=request.style or "three_para_ko")
            except Exception as e:
                logger.exception(f"[EASY] 섹션 변환 실패 idx={i}: {e}")
                results[i] = VizResult(ok=False, index=i, error=str(e), section_title=title, section_type=stype)
                return

            # viz
            ok_viz, img_path = await _send_to_viz(paper_id, i, easy_text, out_dir, sec.get("table_data", []))
            results[i] = VizResult(
                ok=True, index=i, image_path=img_path or None, error=None,
                easy_text=easy_text, section_title=title, section_type=stype
            )

    async with anyio.create_task_group() as tg:
        for i, sec in enumerate(sections):
            tg.start_soon(_work, i, sec)

    success = sum(1 for r in results if r.ok)
    failed  = len(sections) - success

    # 4) JSON 저장
    easy_json = {
        "paper_id": paper_id,
        "count": len(sections),
        "sections": [
            {
                "index": i,
                "title": sections[i].get("title"),
                "original_content": sections[i].get("content"),
                "korean_translation": (results[i].easy_text or ""),
                "image_path": (results[i].image_path or ""),
                "status": "ok" if results[i].ok else f"error: {results[i].error}",
                "section_type": results[i].section_type or sections[i].get("section_type", "section"),
            }
            for i in range(len(sections))
        ],
        "generated_at": _get_current_datetime(),
    }
    json_path = out_dir / "easy_results.json"
    json_path.write_text(json.dumps(easy_json, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[EASY] saved: {json_path}")

    # 5) HTML 저장 (사이드바 TOC 버전 - 네가 이미 정의한 _save_html_results 사용)
    html_path = out_dir / "easy_results.html"
    _save_html_results(sections, results, html_path, paper_id)
    (out_dir / "index.html").write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")  # index.html 복제

    return BatchResult(
        ok=True,
        paper_id=paper_id,
        count=len(sections),
        success=success,
        failed=failed,
        out_dir=str(out_dir),
        images=results,
    )

# ---------- 파일 경유 실행 (/from-transport) ----------
class TransportRequest(BaseModel):
    paper_id: str
    transport_path: str
    output_dir: Optional[str] = None
    style: Optional[str] = Field(default="three_para_ko")

@app.post("/from-transport", response_model=BatchResult)
async def from_transport(request: TransportRequest):
    """
    transport_path 경로(파일: .jsonl/.jsonl.gz/.tex)를 읽어 /batch 로 위임
    """
    tpath = Path(request.transport_path).expanduser().resolve()
    if not tpath.exists():
        raise HTTPException(status_code=404, detail=f"transport 파일 없음: {tpath}")

    base_out = Path(request.output_dir).expanduser().resolve() if request.output_dir else tpath.parent
    content = None

    try:
        ext = tpath.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            content = tpath.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".gz":
            with gzip.open(tpath, "rt", encoding="utf-8") as f:
                content = f.read()
        elif ext == ".tex":
            # .tex → 섹션 파싱 후 JSONL로 변환해 전달
            secs = _parse_latex_sections(tpath)
            lines = [json.dumps({"title": s.get("title") or f"Section {i+1}", "text": s.get("content") or ""}, ensure_ascii=False)
                     for i, s in enumerate(secs)]
            content = "\n".join(lines)
        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {tpath.name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 읽기/파싱 실패: {e}")

    # /batch 위임
    req = BatchRequest(
        paper_id=request.paper_id,
        chunks_jsonl=content,
        output_dir=str(base_out),
        style=request.style or "three_para_ko",
    )
    return await run_batch(req)
# ====================== /END BATCH / TRANSPORT / VIZ ======================

# -------------------- Run --------------------
if __name__ == "__main__":
    host = os.getenv("EASY_HOST", "0.0.0.0")
    port = int(os.getenv("EASY_PORT", "5003"))
    reload_flag = os.getenv("EASY_RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run("app:app", host=host, port=port, reload=reload_flag)
# === PART 4/4 END ===

