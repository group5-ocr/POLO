# -*- coding: utf-8 -*-
"""
LaTeX 수식 해설 API (FastAPI) + Google Cloud Translation v3 번역(ko) 후처리

출력물(총 4개)
- 원본 JSON:  equations_explained.json
- 원본 TeX :  yolo_math_report.tex
- 번역 JSON:  equations_explained.ko.json
- 번역 TeX :  yolo_math_report.ko.tex
"""

# === 기본 환경 ===
import os, sys, json, re, textwrap, datetime, torch
from typing import List, Dict, Tuple
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# dotenv 없이 간단하게 환경변수 로드
def load_env_file(env_path):
    """간단한 .env 파일 로드"""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

# [NEW] Google Cloud Translation v3
try:
    from google.cloud import translate
except Exception:
    translate = None

# [권장] 콘솔 출력 flush
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

VERSION = "POLO-Math-API v4 + GCP-Translate (EN->KO)"
print(VERSION, flush=True)

# ----- 경로/상수 -----
INPUT_TEX_PATH = r"C:\\POLO\\polo-system\\models\\math\\yolo.tex"
OUT_DIR        = "C:/POLO/polo-system/models/math/_build"
os.makedirs(OUT_DIR, exist_ok=True)

# ----- 모델/토크나이저 -----
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
USE_4BIT = False
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python: {sys.version.split()[0]}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"🔧 Device: {DEVICE}, dtype: float16", flush=True)
else:
    print("⚠️ GPU 없음 → CPU 모드", flush=True)
    print(f"🔧 Device: {DEVICE}, dtype: float32", flush=True)

# --- 간단한 .env 로드 ---
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_env_file(str(ROOT_ENV))
print(f"[env] .env loaded from: {ROOT_ENV}", flush=True)

SAFE_CACHE_DIR = Path(__file__).resolve().parent / "hf_cache"
SAFE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _force_safe_hf_cache():
    for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        os.environ[k] = str(SAFE_CACHE_DIR)
    print(f"[hf_cache] forced → {SAFE_CACHE_DIR}", flush=True)

_force_safe_hf_cache()
# Hugging Face 토큰 설정 (여러 가능한 이름으로 시도)
HF_TOKEN = os.getenv("허깅페이스 토큰") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
print(f"HF_TOKEN={'설정됨' if HF_TOKEN else '없음'} (환경변수: '허깅페이스 토큰' 또는 'HUGGINGFACE_TOKEN')", flush=True)

def load_model():
    """모델 로드 함수"""
    global tokenizer, model, GEN_KW
    
    try:
        print(f"🔄 Math 모델 로딩 시작: {MODEL_ID}", flush=True)
        print(f"HF_HOME={os.getenv('HF_HOME')}", flush=True)
        print(f"HF_TOKEN={'설정됨' if HF_TOKEN else '없음'} (환경변수: '허깅페이스 토큰')", flush=True)
        
        # 1) 토크나이저 (캐시/토큰 명시)
        print("📝 토크나이저 로딩 중...", flush=True)
        print(f"📝 MODEL_ID: {MODEL_ID}", flush=True)
        print(f"📝 CACHE_DIR: {SAFE_CACHE_DIR}", flush=True)
        print(f"📝 HF_TOKEN: {'설정됨' if HF_TOKEN else '없음'} (환경변수: '허깅페이스 토큰')", flush=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=HF_TOKEN,
            cache_dir=str(SAFE_CACHE_DIR),
        )
        print("✅ 토크나이저 로딩 완료", flush=True)

        # 2) pad 토큰 보정
        PAD_ADDED = False
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            PAD_ADDED = True
            print("🔧 PAD 토큰 추가됨", flush=True)

        # 3) 모델 로드 (필요 시 4bit)
        print("🧠 모델 로딩 중...", flush=True)
        print(f"🧠 DEVICE: {DEVICE}", flush=True)
        print(f"🧠 USE_4BIT: {USE_4BIT}", flush=True)
        
        bnb_config = None
        if USE_4BIT:
            print("🔧 4bit 양자화 설정 적용", flush=True)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    if PAD_ADDED:
        model.resize_token_embeddings(len(tokenizer))
    GEN_KW = dict(max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=True)
    print("Model & tokenizer loaded.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=HF_TOKEN,
            cache_dir=str(SAFE_CACHE_DIR),
        )

        if PAD_ADDED:
            model.resize_token_embeddings(len(tokenizer))
            print("🔧 토큰 임베딩 크기 조정됨", flush=True)

        GEN_KW = dict(
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )
        print("✅ Math 모델 로딩 완료", flush=True)
        print(f"✅ 모델 디바이스: {next(model.parameters()).device}", flush=True)
        print(f"✅ 모델 dtype: {next(model.parameters()).dtype}", flush=True)
        return True

    except Exception as e:
        print(f"❌ Math 모델 로딩 실패: {e}", flush=True)
        print(f"❌ 에러 타입: {type(e).__name__}", flush=True)
        print(f"❌ 에러 상세: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        tokenizer = None
        model = None
        GEN_KW = {}
        return False

# 모델 로드 시도
print("🚀 Math 모델 로딩 시작...", flush=True)
model_loaded = load_model()
if not model_loaded:
    print("⚠️ 모델 로딩 실패 - 서버는 시작되지만 기능이 제한됩니다.", flush=True)
    print("⚠️ 가능한 원인:", flush=True)
    print("  - '허깅페이스 토큰' 환경변수가 설정되지 않음", flush=True)
    print("  - 인터넷 연결 문제", flush=True)
    print("  - 모델 다운로드 실패", flush=True)
    print("  - 메모리 부족", flush=True)
    print("  - CUDA/GPU 문제", flush=True)
else:
    print("🎉 Math 모델 로딩 성공!", flush=True)

# === [NEW] GCP Translation 초기화 ===
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION   = os.getenv("GOOGLE_CLOUD_TRANSLATE_LOCATION", "global")
gcp_translate_client = None
GCP_PARENT = None
if translate is not None and GCP_PROJECT_ID:
    try:
        gcp_translate_client = translate.TranslationServiceClient()
        GCP_PARENT = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
        print(f"GCP Translation ready: parent={GCP_PARENT}", flush=True)
    except Exception as e:
        print("[Warn] GCP Translation init failed:", e, flush=True)
else:
    print("[Warn] GCP Translation not configured (missing lib or env)", flush=True)


# === 공통 유틸 ===
def make_line_offsets(text: str) -> List[int]:
    lines = text.splitlines()
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
        pos += len(ln) + 1  # '\n' 포함
    return offsets

def build_pos_to_line(offsets: List[int]):
    def pos_to_line(p: int) -> int:
        lo, hi = 0, len(offsets)-1
        while lo <= hi:
            mid = (lo+hi)//2
            if offsets[mid] <= p:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi + 1
    return pos_to_line


# === 수식 추출 ===
def extract_equations(tex: str, pos_to_line) -> List[Dict]:
    matches: List[Dict] = []

    def add(kind, start, end, body, env=""):
        matches.append({
            "kind": kind, "env": env, "start": start, "end": end,
            "line_start": pos_to_line(start), "line_end": pos_to_line(end),
            "body": body.strip()
        })

    # $$ ... $$
    for m in re.finditer(r"\$\$(.+?)\$\$", tex, flags=re.DOTALL):
        add("display($$ $$)", m.start(), m.end(), m.group(1))
    # \[ ... \]
    for m in re.finditer(r"\\\[(.+?)\\\]", tex, flags=re.DOTALL):
        add("display(\\[ \\])", m.start(), m.end(), m.group(1))
    # \( ... \)
    for m in re.finditer(r"\\\((.+?)\\\)", tex, flags=re.DOTALL):
        add("inline(\\( \\))", m.start(), m.end(), m.group(1))
    # inline $...$ (but not $$)
    for m in re.finditer(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", tex, flags=re.DOTALL):
        add("inline($ $)", m.start(), m.end(), m.group(1))

    # environments
    envs = ["equation","equation*","align","align*","multline","multline*",
            "gather","gather*","flalign","flalign*","eqnarray","eqnarray*","split","cases"]
    for env in envs:
        pattern = rf"\\begin{{{re.escape(env)}}}(.+?)\\end{{{re.escape(env)}}}"
        for m in re.finditer(pattern, tex, flags=re.DOTALL):
            add("env", m.start(), m.end(), m.group(1), env=env)

    uniq = {}
    for it in matches:
        key = (it["start"], it["end"])
        if key not in uniq:
            uniq[key] = it

    out = list(uniq.values())
    out.sort(key=lambda x: x["start"])
    return out


# === 난이도 휴리스틱 ===
# === 셀 3: 난이도 휴리스틱 ===
ADV_TOKENS = [
    r"\\sum", r"\\prod", r"\\int", r"\\lim", r"\\nabla", r"\\partial",
    r"\\mathbb", r"\\mathcal", r"\\mathbf", r"\\boldsymbol",
    r"\\argmax", r"\\argmin", r"\\operatorname", r"\\mathrm\{KL\}",
    r"\\mathbb\{E\}", r"\\Pr", r"\\sigma", r"\\mu", r"\\Sigma", r"\\theta",
    r"\\frac\{[^{}]*\{[^{}]*\}[^{}]*\}",
    r"\\frac\{[^{}]*\{[^{}]*\}[^{}]*\}",  # nested fraction
    r"\\hat\{", r"\\tilde\{", r"\\bar\{", r"\\widehat\{", r"\\widetilde\{",
    r"\\sqrt\{[^{}]*\{",
    r"\\sqrt\{[^{}]*\{",                  # nested sqrt
    r"\\left", r"\\right",
    r"\\in", r"\\subset", r"\\forall", r"\\exists",
    r"\\cdot", r"\\times", r"\\otimes",
    r"IoU", r"\\log", r"\\exp",
    r"\\mathbb\{R\}", r"\\mathbb\{N\}", r"\\mathbb\{Z\}",
    r"\\Delta", r"\\delta", r"\\epsilon", r"\\varepsilon",
]
ADV_RE = re.compile("|".join(ADV_TOKENS))

def count_subscripts(expr: str) -> int:
    return len(re.findall(r"_[a-zA-Z0-9{\\]", expr))

def is_advanced(eq: str) -> bool:
    if ADV_RE.search(eq):
        return True
    if len(eq) > 40 and count_subscripts(eq) >= 2:
        return True
    if "\n" in eq and len(eq) > 30:
        return True
    return False


# === 셀 4: 문서 개요 LLM ===
def take_slices(text: str, head_chars=4000, mid_chars=2000, tail_chars=4000):
    n = len(text)
    head = text[:min(head_chars, n)]
    mid_start = max((n // 2) - (mid_chars // 2), 0)
    mid = text[mid_start: mid_start + min(mid_chars, n)]
    tail = text[max(0, n - tail_chars):]
    return head, mid, tail

def _generate_with_mask_from_messages(messages: List[Dict]) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", padding=True
    )
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.to(model.device),
            attention_mask=attention_mask.to(model.device),
            **GEN_KW
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def chat_overview(prompt: str) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded.")
    messages = [
        {"role": "system", "content":
         "You are a clear, concise technical writer who summarizes LaTeX-based AI papers for a general technical audience."},
        {"role": "user", "content": prompt}
    ]
    text = _generate_with_mask_from_messages(messages)
    return text.split(messages[-1]["content"])[-1].strip()


# === 셀 5: 수식 해설 LLM ===
EXPLAIN_SYSTEM = (
    "You are a teacher who explains math/AI research equations in clear, simple English. "
    "Always be precise, polite, and easy to understand."
)
EXPLAIN_TEMPLATE = """Please explain the following equation so that it can be understood by someone at least at a middle school level.
Follow this exact order in your output: Example → Explanation → Conclusion

- Example: Show the equation exactly as LaTeX in a single block (do not modify or add anything).
- Explanation: Provide bullet points explaining the meaning of symbols (∑, 𝟙, ^, _, √, \\, etc.) and the role of each term, in a clear and concise way.
- Conclusion: Summarize in one sentence the core purpose of this equation in the context of the paper (e.g., loss composition, normalization, coordinate error, probability/log-likelihood, etc.).
- (Important) Do not change the symbols or the order of the equation, and do not invent new symbols.
- (Important) Write only in English.

[Equation]
{EQUATION}
"""

def explain_equation_with_llm(eq_latex: str) -> str:
    if not model_loaded or tokenizer is None or model is None:
        raise RuntimeError("Math model is not loaded")
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user",   "content": EXPLAIN_TEMPLATE.format(EQUATION=eq_latex)}
    ]
    text = _generate_with_mask_from_messages(messages)
    return text.split(messages[-1]["content"])[-1].strip()


# === TeX 리포트 생성 ===
# === 셀 6: LaTeX 리포트(.tex) ===
def latex_escape_verbatim(s: str) -> str:
    s = s.replace("\\", r"\\")
    s = s.replace("#", r"\#").replace("$", r"\$")
    s = s.replace("%", r"\%").replace("&", r"\&")
    s = s.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
    s = s.replace("^", r"\^{}").replace("~", r"\~{}")
    return s

def build_report(overview: str, items: List[Dict]) -> str:
    header = (r"""\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{amsmath, amssymb, amsfonts}
\\usepackage{hyperref}
\\usepackage{kotex}
\\setlength{\\parskip}{6pt}
\\setlength{\\parindent}{0pt}
\\title{LaTeX Equation Explanation Report (Middle-School Level+)}
\\author{Automatic Pipeline}
\\date{""" + datetime.date.today().isoformat() + r"""}
\\begin{document}
\\maketitle
\\tableofcontents
\\newpage
""")
    parts = [header]
    parts.append(r"\\section*{Document Overview}")
    parts.append(latex_escape_verbatim(overview))
    parts.append("\n\\newpage\n")

    for it in items:
        title = f"Lines {it['line_start']}–{it['line_end']} / {it['kind']} {('['+it['env']+']') if it['env'] else ''}"
        parts.append(f"\\section*{{{latex_escape_verbatim(title)}}}")
        parts.append(it["explanation"])
        parts.append("\n")

    parts.append("\\end{document}\n")
    return "\n".join(parts)


# === 보조: 수식 개수 ===
def count_equations_only(input_tex_path: str) -> Dict[str, int]:
    p = Path(input_tex_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find TeX file: {input_tex_path}")

    src = p.read_text(encoding="utf-8", errors="ignore")
    offsets = make_line_offsets(src)
    pos_to_line = build_pos_to_line(offsets)
    equations_all = extract_equations(src, pos_to_line)
    equations_advanced = [e for e in equations_all if is_advanced(e["body"])]

    print(f"총 수식: {len(equations_all)}", flush=True)
    print(f"중학생 수준 이상: {len(equations_advanced)} / {len(equations_all)}", flush=True)

    return {"총 수식": len(equations_all),
            "중학생 수준 이상": len(equations_advanced)}


# ======================================================================
# ======================= [NEW] 번역 유틸 섹션 ==========================
# ======================================================================

# 수식 보호 (디스플레이/인라인/환경)
_MATH_ENV_NAMES = r"(?:equation|align|gather|multline|eqnarray|cases|split)\*?"
_MATH_PATTERN = re.compile(
    r"(?P<D2>\${2}[\s\S]*?\${2})"      # $$ ... $$
    r"|(?P<D1>(?<!\\)\$[\s\S]*?(?<!\\)\$)"  # $ ... $
    r"|(?P<LB>\\\[[\s\S]*?\\\])"       # \[ ... \]
    r"|(?P<LP>\\\([\s\S]*?\\\))"       # \( ... \)
    r"|(?P<ENV>\\begin\{" + _MATH_ENV_NAMES + r"\}[\s\S]*?\\end\{" + _MATH_ENV_NAMES + r"\})",
    re.MULTILINE
)

def protect_math(text: str) -> Tuple[str, Dict[str, str]]:
    holders = {}
    def _repl(m):
        key = f"⟦MATH{len(holders)}⟧"
        holders[key] = m.group(0)
        return key
    return _MATH_PATTERN.sub(_repl, text), holders

def restore_math(text: str, holders: Dict[str, str]) -> str:
    for k, v in holders.items():
        text = text.replace(k, v)
    return text

# 괄호 보호: 중첩 대응을 위해 반복 적용
def protect_parens(text: str) -> Tuple[str, Dict[str, str]]:
    holders = {}
    idx = 0
    pat = re.compile(r"\([^()\n]*\)")
    out = text
    while True:
        changed = False
        def _repl(m):
            nonlocal idx, changed
            key = f"⟦P{idx}⟧"
            holders[key] = m.group(0)
            idx += 1
            changed = True
            return key
        out2 = pat.sub(_repl, out)
        out = out2
        if not changed:
            break
    return out, holders

def restore_parens(text: str, holders: Dict[str, str]) -> str:
    for k, v in holders.items():
        text = text.replace(k, v)
    return text

def split_into_paragraphs(s: str) -> List[str]:
    # 빈 줄(두 줄 이상) 기준으로 문단 분리, 앞뒤 공백은 보존
    # 번역 정확도를 위해 문단 단위로 번역
    parts = re.split(r"\n\s*\n", s)
    return parts

def join_paragraphs(paragraphs: List[str]) -> str:
    return "\n\n".join(paragraphs)

def translate_paragraphs_gcp(paragraphs: List[str], target_lang="ko") -> List[str]:
    """
    - 수식/괄호 보호 후 번역
    - 빈 문단은 API 호출에서 제외
    - 번역 실패 시 원문 유지
    """
    if gcp_translate_client is None or GCP_PARENT is None:
        print("[Warn] GCP Translation not ready; return original.", flush=True)
        return paragraphs[:]

    prot_list, holders_math, holders_paren = [], [], []
    for para in paragraphs:
        if not para.strip():
            prot_list.append("")
            holders_math.append({})
            holders_paren.append({})
            continue
        p1, h_m = protect_math(para)
        p2, h_p = protect_parens(p1)
        prot_list.append(p2)
        holders_math.append(h_m)
        holders_paren.append(h_p)

    out_list = [""] * len(paragraphs)

    BATCH = 32
    def _flush_batch(idxs: List[int]):
        if not idxs:
            return
        nonempty = [i for i in idxs if prot_list[i].strip() != ""]
        if not nonempty:
            for i in idxs:
                out_list[i] = prot_list[i]
            return
        contents = [prot_list[i] for i in nonempty]
        try:
            resp = gcp_translate_client.translate_text(
                request={
                    "parent": GCP_PARENT,
                    "contents": contents,
                    "mime_type": "text/plain",
                    "target_language_code": target_lang,
                }
            )
            translated = [t.translated_text for t in resp.translations]
        except Exception as e:
            print("[Translate Error][GCP]", e, file=sys.stderr)
            translated = contents  # 실패 시 원문

        for j, idx in enumerate(nonempty):
            t = translated[j]
            t = restore_parens(t, holders_paren[idx])
            t = restore_math(t, holders_math[idx])
            out_list[idx] = t

        for i in set(idxs) - set(nonempty):
            out_list[i] = prot_list[i]

    buf = []
    for i in range(len(paragraphs)):
        buf.append(i)
        if len(buf) >= BATCH:
            _flush_batch(buf); buf.clear()
    _flush_batch(buf)
    return out_list

def translate_text_gcp(text: str, target_lang="ko") -> str:
    """단일 문자열 번역: 문단 분리 → 배치 번역 → 병합"""
    paras = split_into_paragraphs(text)
    out_paras = translate_paragraphs_gcp(paras, target_lang=target_lang)
    return join_paragraphs(out_paras)

def translate_tex_file(in_path: str, out_path: str, target_lang="ko") -> None:
    s = Path(in_path).read_text(encoding="utf-8", errors="ignore")
    paras = split_into_paragraphs(s)
    out_paras = translate_paragraphs_gcp(paras, target_lang=target_lang)
    out_text = join_paragraphs(out_paras)
    Path(out_path).write_text(out_text, encoding="utf-8")
    print(f"[OK] 번역본 TeX 저장: {out_path}", flush=True)

def translate_json_payload(in_json_path: str, out_json_path: str, target_lang="ko") -> None:
    data = json.loads(Path(in_json_path).read_text(encoding="utf-8"))
    # overview 번역
    overview_en = data.get("overview", "")
    overview_ko = translate_text_gcp(overview_en, target_lang=target_lang) if overview_en else ""

    items_ko = []
    for it in data.get("items", []):
        exp_en = it.get("explanation", "")
        # 수식(it["equation"])은 보호 → 번역 대상이 아님 (영문 그대로 유지), 설명만 번역
        exp_ko = translate_text_gcp(exp_en, target_lang=target_lang) if exp_en else ""
        it_ko = dict(it)
        it_ko["explanation"] = exp_ko
        items_ko.append(it_ko)

    out_obj = {"overview": overview_ko, "items": items_ko}
    Path(out_json_path).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 번역본 JSON 저장: {out_json_path}", flush=True)


# ======================================================================
# ========================== 메인 파이프라인 ============================
# ======================================================================
def run_pipeline(input_tex_path: str) -> Dict:
    p = Path(input_tex_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find TeX file: {input_tex_path}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    src = p.read_text(encoding="utf-8", errors="ignore")
    offsets = make_line_offsets(src)
    pos_to_line = build_pos_to_line(offsets)

    equations_all = extract_equations(src, pos_to_line)
    equations_advanced = [e for e in equations_all if is_advanced(e["body"])]

    print(f"총 수식: {len(equations_all)}", flush=True)
    print(f"중학생 수준 이상: {len(equations_advanced)} / {len(equations_all)}", flush=True)

    head, mid, tail = take_slices(src)
    overview_prompt = textwrap.dedent(f"""
    You will be given three slices of a LaTeX document (head / middle / tail).
    Please produce a concise English overview with:
    - One short paragraph describing the core topic and objective of the paper.
    - A few bullet points listing the main sections or ideas (if discernible).
    - Focus on how mathematical notation and symbols are used to support the overall flow.

    [HEAD]
    {head}

    [MIDDLE]
    {mid}

    [TAIL]
    {tail}
    """).strip()
    doc_overview = chat_overview(overview_prompt)

    explanations: List[Dict] = []
    for idx, item in enumerate(equations_advanced, start=1):
        print(f"[{idx}/{len(equations_advanced)}] 라인 {item['line_start']}–{item['line_end']}", flush=True)
        exp = explain_equation_with_llm(item["body"])
        explanations.append({
            "index": idx,
            "line_start": item["line_start"],
            "line_end": item["line_end"],
            "kind": item["kind"],
            "env": item["env"],
            "equation": item["body"],
            "explanation": exp
        })

    # === 원본 저장 ===
    json_path = os.path.join(OUT_DIR, "equations_explained.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overview": doc_overview, "items": explanations}, f, ensure_ascii=False, indent=2)
    print(f"저장된 JSON: {json_path}", flush=True)

    report_tex_path = os.path.join(OUT_DIR, "yolo_math_report.tex")
    report_tex = build_report(doc_overview, explanations)
    Path(report_tex_path).write_text(report_tex, encoding="utf-8")
    print(f"저장된 TeX: {report_tex_path}", flush=True)

    # === [NEW] 번역본 생성 ===
    json_ko_path  = os.path.join(OUT_DIR, "equations_explained.ko.json")
    tex_ko_path   = os.path.join(OUT_DIR, "yolo_math_report.ko.tex")

    try:
        translate_json_payload(json_path, json_ko_path, target_lang="ko")
    except Exception as e:
        print("[Translate JSON Error]", e, file=sys.stderr)

    try:
        translate_tex_file(report_tex_path, tex_ko_path, target_lang="ko")
    except Exception as e:
        print("[Translate TeX Error]", e, file=sys.stderr)

    # === 반환 ===
    return {
        "input": str(p),
        "counts": {
            "총 수식": len(equations_all),
            "중학생 수준 이상": len(equations_advanced)
        },
        "outputs": {
            "json": json_path,
            "report_tex": report_tex_path,
            "json_ko": json_ko_path,
            "report_tex_ko": tex_ko_path,
            "out_dir": OUT_DIR
        }
    }


# === FastAPI ===
app = FastAPI(title="POLO Math Explainer API", version="1.1.0")

class MathRequest(BaseModel):
    path: str

@app.get("/health")
async def health():
    return {
        "status": "ok" if model_loaded else "degraded",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": DEVICE,
        "model_loaded": (tokenizer is not None and model is not None),
        "gcp_translate_ready": (gcp_translate_client is not None and GCP_PARENT is not None),
        "gcp_parent": GCP_PARENT
    }

@app.get("/count/{file_path:path}")
async def count_get(file_path: str):
    try:
        return count_equations_only(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/count")
async def count_post(req: MathRequest):
    try:
        return count_equations_only(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.get("/math/{file_path:path}")
async def math_get(file_path: str):
    try:
        return run_pipeline(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/math")
async def math_post(req: MathRequest):
    try:
        return run_pipeline(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# 직접 실행
# 직접 실행
if __name__ == "__main__":
    try:
        import uvicorn
        print("🚀 Math Model 서버 시작...")
        uvicorn.run("app:app", host="0.0.0.0", port=5004, reload=False)
    except Exception as e:
        print(f"❌ 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
