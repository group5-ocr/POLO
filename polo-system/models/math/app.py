# -*- coding: utf-8 -*-
"""
LaTeX 수식 해설 API (FastAPI) + GCP Translation(ko) 번역본 저장
(버그픽스: MathJax에서 \mathlarger, \mathbbm 깨짐 방지)
(업데이트: 논문 매크로 정의 파싱 & 프롬프트 컨텍스트 주입)
(추가수정: env별 원래 환경 감싸기 -> MathJax 'Misplaced &' 제거)
"""

# === 셀 1: 환경 준비 & 모델 로드 ===
import os, sys, json, re, textwrap, datetime, torch
from typing import List, Dict, Tuple
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---- GCP Translation v3 ----
try:
    from google.cloud import translate
    from google.oauth2 import service_account
except Exception:
    translate = None
    service_account = None

# dotenv 없이 간단하게 환경변수 로드
def load_env_file(env_path):
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

# stdout flush
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

VERSION = "POLO-Math-API v7.1 (macro-parsing + context-aware explain + env-wrap)"  # ★ 변경
print(VERSION, flush=True)

# ----- 경로 설정 -----
INPUT_TEX_PATH = r"C:\\POLO\\polo-system\\models\\math\\yolo.tex"
# OUT_DIR = "C:/POLO/polo-system/models/math/\_build"

OUT_DIR = "C:/POLO/POLO/polo-system/models/math/\_build"
os.makedirs(OUT_DIR, exist_ok=True)

# ----- 모델/토크나이저 설정 -----
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
USE_4BIT = False
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python: {sys.version.split()[0]}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU 사용 가능: {gpu_name}", flush=True)
    print(f"🔧 디바이스: {DEVICE}, 데이터 타입: float16", flush=True)
else:
    print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.", flush=True)
    print(f"🔧 디바이스: {DEVICE}, 데이터 타입: float32", flush=True)

# --- .env 로드 ---
try:
    ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
except Exception:
    ROOT_ENV = Path(".") / ".env"
load_env_file(str(ROOT_ENV))
print(f"[env] .env loaded from: {ROOT_ENV}", flush=True)

SAFE_CACHE_DIR = Path(__file__).resolve().parent / "hf_cache"
SAFE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _force_safe_hf_cache():
    for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        os.environ[k] = str(SAFE_CACHE_DIR)
    print(f"[hf_cache] forced → {SAFE_CACHE_DIR}", flush=True)

_force_safe_hf_cache()
HF_TOKEN = os.getenv("허깅페이스 토큰") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
print(f"HF_TOKEN={'설정됨' if HF_TOKEN else '없음'}", flush=True)

def load_model():
    global tokenizer, model
    try:
        print(f"🔄 Math 모델 로딩 시작: {MODEL_ID}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True, token=HF_TOKEN, cache_dir=str(SAFE_CACHE_DIR)
        )
        print("✅ 토크나이저 로딩 완료", flush=True)

        PAD_ADDED = False
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            PAD_ADDED = True
            print("🔧 PAD 토큰 추가됨", flush=True)

        bnb_config = None
        if USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            quantization_config=bnb_config, low_cpu_mem_usage=True,
            trust_remote_code=True, token=HF_TOKEN, cache_dir=str(SAFE_CACHE_DIR),
        )
        if PAD_ADDED:
            model.resize_token_embeddings(len(tokenizer))
            print("🔧 토큰 임베딩 크기 조정됨", flush=True)

        print("🎉 Math 모델 로딩 성공!", flush=True)
        return True
    except Exception as e:
        print(f"❌ Math 모델 로딩 실패: {e}", flush=True)
        import traceback; traceback.print_exc()
        tokenizer = None; model = None
        return False

print("🚀 Math 모델 로딩 시작...", flush=True)
model_loaded = load_model()

# === 공통 유틸: 라인 오프셋 인덱스 ===
def make_line_offsets(text: str) -> List[int]:
    lines = text.splitlines()
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
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

# === 셀 2: LaTeX 수식 추출 ===
def extract_equations(tex: str, pos_to_line) -> List[Dict]:
    matches: List[Dict] = []
    def add(kind, start, end, body, env=""):
        matches.append({
            "kind": kind, "env": env, "start": start, "end": end,
            "line_start": pos_to_line(start), "line_end": pos_to_line(end),
            "body": body.strip()
        })
    for m in re.finditer(r"\$\$(.+?)\$\$", tex, flags=re.DOTALL):
        add("display($$ $$)", m.start(), m.end(), m.group(1))
    for m in re.finditer(r"\\\[(.+?)\\\]", tex, flags=re.DOTALL):
        add("display(\\[ \\])", m.start(), m.end(), m.group(1))
    for m in re.finditer(r"\\\((.+?)\\\)", tex, flags=re.DOTALL):
        add("inline(\\( \\))", m.start(), m.end(), m.group(1))
    for m in re.finditer(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", tex, flags=re.DOTALL):
        add("inline($ $)", m.start(), m.end(), m.group(1))
    envs = ["equation","equation*","align","align*","multline","multline*",
            "gather","gather*","flalign","flalign*","eqnarray","eqnarray*","split","cases","cases*","aligned"]
    for env in envs:
        pattern = rf"\\begin{{{re.escape(env)}}}(.+?)\\end{{{re.escape(env)}}}"
        for m in re.finditer(pattern, tex, flags=re.DOTALL):
            add("env", m.start(), m.end(), m.group(1), env=env)
    uniq = {}
    for it in matches:
        key = (it["start"], it["end"])
        if key not in uniq:
            uniq[key] = it
    out = list(uniq.values()); out.sort(key=lambda x: x["start"])
    return out

# === 셀 3: 난이도 휴리스틱 (점수제)
_NUMERIC_ONLY_RE = re.compile(r"^[\s\d\.\,\+\-\*/×x\(\)\[\]\{\}:;=]+$")

def numeric_only(eq: str) -> bool:
    s = re.sub(r"\\times|\\cdot|\\left|\\right|\\,|\\\\|\\:", "", eq)
    s = re.sub(r"\s+", "", s)
    return bool(_NUMERIC_ONLY_RE.match(s))

_GREEK_RE = re.compile(r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|theta|mu|sigma|lambda|phi|psi|omega)\\b")

def complexity_score(eq: str) -> int:
    score = 0
    score += 3 * len(re.findall(r"\\sum|\\prod|\\int|\\lim|\\Pr|\\mathbb\{E\}", eq))
    score += 2 * len(re.findall(r"\\frac\{.+?\}\{.+?\}", eq))
    score += 2 * len(re.findall(r"\\sqrt\{", eq))
    score += 2 if "\n" in eq else 0
    score += 2 * len(re.findall(r"\\begin\{(align|multline|cases|split|aligned)\*?\}", eq))
    score += 1 * (len(re.findall(r"_[A-Za-z0-9{\\]", eq)) >= 2)
    score += 1 * bool(_GREEK_RE.search(eq))
    score -= 1 * len(re.findall(r"\\times|\\cdot", eq))
    return score

MIN_COMPLEXITY = 1.3
def is_advanced(eq: str) -> bool:
    if numeric_only(eq):
        return False
    return complexity_score(eq) >= MIN_COMPLEXITY

# === 셀 3.5: 논문 매크로/연산자 파싱 =========================================
_MACRO_NEWCOMMAND_RE = re.compile(
    r"""\\newcommand\s*\{\\([A-Za-z@]+)\}\s*(?:\[[^\]]*\])?\s*\{([\s\S]*?)\}"""
)
_MACRO_DEF_RE = re.compile(
    r"""\\def\s*\\([A-Za-z@]+)\s*\{([\s\S]*?)\}"""
)
_DECL_OP_RE = re.compile(
    r"""\\DeclareMathOperator\*?\s*\{\\([A-Za-z@]+)\}\s*\{([\s\S]*?)\}"""
)

HUMAN_MEANING_MAP = {
    "snr": "Signal-to-Noise Ratio (feedforward): P/σ²",
    "bsnr": "Feedback SNR: \\widetilde{P}/\\widetilde{σ}² (paper formats S N~ R)",
    "dsnr": "Delta SNR: ΔSNR (e.g., feedback minus feedforward)",
    "dB": "decibel unit",
    "Pe": "error probability",
    "qfunc": "Q-function (tail probability of standard normal)",
    "wt": "widetilde (tilde accent)",
    "wh": "widehat (hat accent)",
    "dfn": "triangle equals (definition)",
    "argmin": "argument that minimizes a function",
    "argmax": "argument that maximizes a function"
}

def extract_macro_definitions(tex: str) -> Dict[str, Dict[str, str]]:
    latex_defs: Dict[str, str] = {}
    for m in _MACRO_NEWCOMMAND_RE.finditer(tex):
        name, body = m.group(1), m.group(2).strip()
        latex_defs[name] = body
    for m in _MACRO_DEF_RE.finditer(tex):
        name, body = m.group(1), m.group(2).strip()
        latex_defs[name] = body
    for m in _DECL_OP_RE.finditer(tex):
        name, body = m.group(1), m.group(2).strip()
        latex_defs[name] = body

    human_defs: Dict[str, str] = {}
    for k in latex_defs.keys():
        if k in HUMAN_MEANING_MAP:
            human_defs[k] = HUMAN_MEANING_MAP[k]
    for k, v in HUMAN_MEANING_MAP.items():
        if k not in human_defs:
            human_defs[k] = v
    return {"latex": latex_defs, "human": human_defs}

def collect_used_macros(eq_latex: str, latex_defs: Dict[str, str]) -> Dict[str, str]:
    used = {}
    for name in latex_defs.keys():
        if re.search(rf"\\{name}(?![A-Za-z])", eq_latex):
            used[name] = latex_defs[name]
    for name in ("snr", "bsnr", "dsnr", "dfn", "wt", "wh", "dB", "Pe", "qfunc"):
        if name in latex_defs and name not in used:
            if name in ("snr", "bsnr", "dsnr"):
                used[name] = latex_defs[name]
    return used

def build_paper_context(eq_item: Dict, full_text: str, macros: Dict[str, Dict[str, str]]) -> str:
    start = max(0, eq_item.get("start", 0) - 360)
    end   = min(len(full_text), eq_item.get("end", 0) + 360)
    local_ctx = full_text[start:end]
    local_ctx = re.sub(r"\s+", " ", local_ctx).strip()

    used = collect_used_macros(eq_item["body"], macros["latex"])
    if not used:
        used = {k: macros["latex"][k] for k in macros["latex"].keys() if k in ("snr","bsnr","dsnr","dfn","wt","wh","dB")}

    macro_lines = []
    for k, v in sorted(used.items()):
        human = macros["human"].get(k, "")
        macro_lines.append(f"\\{k} := {v}  {(' // ' + human) if human else ''}")

    ctx = [
        "Paper context:",
        "- Local snippet around the equation (verbatim from the LaTeX source):",
        f"  {local_ctx}",
        "- Macro definitions from this paper (LaTeX and human meaning):"
    ] + [f"  - {line}" for line in macro_lines]

    return "\n".join(ctx)

# === 셀 4: 문서 개요 LLM ===
def take_slices(text: str, head_chars=4000, mid_chars=2000, tail_chars=4000):
    n = len(text)
    head = text[:min(head_chars, n)]
    mid_start = max((n // 2) - (mid_chars // 2), 0)
    mid = text[mid_start: mid_start + min(mid_chars, n)]
    tail = text[max(0, n - tail_chars):]
    return head, mid, tail

def _generate_with_mask_from_messages(messages: List[Dict], gen_kw: dict) -> str:
    if not model_loaded or tokenizer is None or model is None:
        raise RuntimeError("Math model is not loaded")
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", padding=True
    )
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.to(model.device),
            attention_mask=attention_mask.to(model.device),
            **gen_kw
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def chat_overview(prompt: str) -> str:
    messages = [
        {"role": "system", "content":
         "You are a clear, concise technical writer who summarizes LaTeX-based AI papers for a general technical audience."},
        {"role": "user", "content": prompt}
    ]
    gen_kw = dict(max_new_tokens=256, do_sample=False, early_stopping=True,
                  eos_token_id=tokenizer.eos_token_id)
    text = _generate_with_mask_from_messages(messages, gen_kw)
    return text.split(messages[-1]["content"])[-1].strip()

# === 셀 5: 수식 해설 LLM (논문 맥락 주입 + 기호/용어 고정 강화) ===============
EXPLAIN_SYSTEM = (
    "You are an AI research equation explanation expert. "
    "You receive equations extracted from a LaTeX research paper, ALONG WITH the paper context and macro definitions. "
    "Use ONLY the provided paper context/macros to resolve symbol meanings; do not invent meanings. "
    "Ignore trivial arithmetic. "
    "English only — do NOT use CJK characters. "
    "Always follow the required output sections in order and keep LaTeX symbols verbatim."
)

EXPLAIN_TEMPLATE = """You are given:
1) Paper context (local snippet + macro definitions for this paper).
2) One LaTeX equation from the paper.

Explain the equation grounded in the paper context/macros. If a symbol is not defined in the context, say "not defined in the provided context" instead of making assumptions.

Follow this exact order and headings:

### Explanation
- Briefly describe what the equation computes or states, referencing the context when relevant.
- Mention any assumptions or conditions stated in the local snippet.

### Variable Glossary
- Bullet-list each symbol appearing in the equation with a short meaning.
- Use the paper's macro definitions. If a symbol lacks a definition in context, mark it as "not defined in the provided context".

### Conclusion
- One sentence on the equation’s purpose/role in the paper.

Keep symbols verbatim. English only.

[Paper Context]
{CONTEXT}

[Equation]
{EQUATION}
"""

def _calc_gen_kwargs(eq_latex: str, ctx_text: str = "") -> dict:
    eq_tokens = len(tokenizer(eq_latex, add_special_tokens=False).input_ids)
    ctx_tokens = len(tokenizer(ctx_text, add_special_tokens=False).input_ids) if ctx_text else 0
    base = 384 + int(eq_tokens * 2.0) + int(0.5 * ctx_tokens)
    max_new = max(512, min(1792, base))
    return dict(
        max_new_tokens=max_new,
        do_sample=False,
        top_p=None, temperature=None,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
    )

def explain_equation_with_llm(eq_latex: str, paper_ctx: str) -> str:
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user",   "content": EXPLAIN_TEMPLATE.format(CONTEXT=paper_ctx, EQUATION=eq_latex)}
    ]
    gen_kw = _calc_gen_kwargs(eq_latex, paper_ctx)
    text = _generate_with_mask_from_messages(messages, gen_kw)
    return text.split(messages[-1]["content"])[-1].strip()

CONC_PROMPT = "Continue the previous answer. Output ONLY the missing section:\n\n### Conclusion\n"

def ensure_conclusion(text: str, eq_latex: str, paper_ctx: str) -> str:
    if re.search(r"###\s*Conclusion", text, flags=re.I):
        return text
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user",   "content": EXPLAIN_TEMPLATE.format(CONTEXT=paper_ctx, EQUATION=eq_latex)},
        {"role": "assistant", "content": text},
        {"role": "user", "content": CONC_PROMPT}
    ]
    gen_kw = dict(max_new_tokens=160, do_sample=False, early_stopping=True,
                  eos_token_id=tokenizer.eos_token_id)
    add = _generate_with_mask_from_messages(messages, gen_kw)
    add = add.split(CONC_PROMPT)[-1].strip()
    return (text.rstrip() + ("\n\n" if not text.endswith("\n") else "") + add).strip()

# === [NEW] 수식/해설 Sanitizer ===============================================
CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7A3]")
MATH_BLOCK_RE = re.compile(
    r"(?P<D2>\$\$[\s\S]*?\$\$)|(?P<LB>\\\[[\s\S]*?\\\])|(?P<LP>\\\([\s\S]*?\\\))",
    flags=re.MULTILINE
)

def _normalize_example_section(text: str, eq_body: str) -> str:
    sec_pat = re.compile(
        r"###\s*Example[\s\S]*?(?=(###\s*Explanation|###\s*Variable Glossary|###\s*Conclusion|\Z))",
        re.I
    )
    text = sec_pat.sub("", text)
    return text.lstrip()

def _drop_cjk_math_blocks(text: str) -> str:
    def repl(m: re.Match) -> str:
        block = m.group(0)
        return "" if CJK_RE.search(block) else block
    return MATH_BLOCK_RE.sub(repl, text)

def sanitize_explanation(exp_text: str, eq_body: str) -> str:
    if not isinstance(exp_text, str):
        exp_text = str(exp_text)
    exp_text = _normalize_example_section(exp_text, eq_body)
    exp_text = _drop_cjk_math_blocks(exp_text)
    exp_text = re.sub(r"\n{3,}", "\n\n", exp_text).strip()
    return exp_text

# === [NEW] MathJax 친화적 수식 정규화 =======================================
_MATHBBM_ONE_RE = re.compile(r"\\mathbbm\s*\{\s*1\s*\}")
_MATHBBM_ANY_RE = re.compile(r"\\mathbbm\s*\{")

def normalize_for_mathjax(eq: str) -> str:
    s = eq
    s = _MATHBBM_ONE_RE.sub(r"\\mathbf{1}", s)
    s = _MATHBBM_ANY_RE.sub(r"\\mathbb{", s)
    return s

# === 렌더링: env 저장 시 원래 환경으로 감싸기 ===============================
_ENV_WRAP_SET = {"cases","cases*","split","aligned","align","align*","gather","gather*","multline","multline*","eqnarray","eqnarray*","flalign","flalign*"}
def wrap_for_render(eq_body: str, env: str) -> str:
    """
    MathJax에서 '&' 정렬 오류를 막기 위해, 정렬형/케이스형 env는 원래 환경으로 감싸서 렌더.
    그 외에는 $$ ... $$ 로만 감싸도 OK.
    """
    if env and env in _ENV_WRAP_SET:
        return f"\\begin{{{env}}}\n{eq_body}\n\\end{{{env}}}"
    return eq_body

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

# === 보조: 수식 개수만 ===
def count_equations_only(input_tex_path: str) -> Dict[str, int]:
    p = Path(input_tex_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find TeX file: {input_tex_path}")
    src = p.read_text(encoding="utf-8", errors="ignore")
    offsets = make_line_offsets(src); pos_to_line = build_pos_to_line(offsets)
    equations_all = extract_equations(src, pos_to_line)
    equations_advanced = [e for e in equations_all if is_advanced(e["body"])]
    print(f"총 수식: {len(equations_all)}", flush=True)
    print(f"중학생 수준 이상: {len(equations_advanced)} / {len(equations_all)}", flush=True)
    return {"총 수식": len(equations_all), "중학생 수준 이상": len(equations_advanced)}

# ======================================================================
# ======================= 번역 유틸 섹션 ===============================
# ======================================================================

# SERVICE_ACCOUNT_PATH = Path(r"C:\POLO\polo-system\models\math\stone-booking-466716-n6-f6fff7380e05.json")

SERVICE_ACCOUNT_PATH = Path(r"C:\POLO\POLO\polo-system\models\math\stone-booking-466716-n6-f6fff7380e05.json")
GCP_LOCATION = "global"

gcp_translate_client = None
GCP_PARENT = None
GCP_PROJECT_ID = None

def init_gcp_local():
    global gcp_translate_client, GCP_PARENT, GCP_PROJECT_ID
    if translate is None or service_account is None:
        print("[Warn] google-cloud-translate 또는 oauth2 패키지가 없습니다. 번역 기능을 사용할 수 없습니다.", flush=True)
        return
    if not SERVICE_ACCOUNT_PATH.exists():
        print(f"[Warn] 서비스 계정 키 파일이 없습니다: {SERVICE_ACCOUNT_PATH}", flush=True)
        print("[Warn] 번역 기능을 사용할 수 없습니다. Math 모델은 계속 실행됩니다.", flush=True)
        return
    try:
        creds = service_account.Credentials.from_service_account_file(str(SERVICE_ACCOUNT_PATH))
        GCP_PROJECT_ID = getattr(creds, "project_id", None)
        if not GCP_PROJECT_ID:
            print("[Warn] 키 파일에서 project_id를 찾지 못했습니다. 번역 기능을 사용할 수 없습니다.", flush=True)
            return
        gcp_translate_client = translate.TranslationServiceClient(credentials=creds)
        GCP_PARENT = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
        print(f"GCP Translation ready (local creds): parent={GCP_PARENT}", flush=True)
    except Exception as e:
        print("[Warn] GCP Translation init failed (local creds):", e, flush=True)
        print("[Warn] 번역 기능을 사용할 수 없습니다. Math 모델은 계속 실행됩니다.", flush=True)

def init_gcp_from_env():
    global gcp_translate_client, GCP_PARENT, GCP_PROJECT_ID
    if translate is None:
        return
    proj = os.getenv("GOOGLE_CLOUD_PROJECT")
    loc  = os.getenv("GOOGLE_CLOUD_TRANSLATE_LOCATION", "global")
    if not proj:
        return
    try:
        gcp_translate_client = translate.TranslationServiceClient()
        GCP_PROJECT_ID = proj
        GCP_PARENT = f"projects/{proj}/locations/{loc}"
        print(f"GCP Translation ready (env): parent={GCP_PARENT}", flush=True)
    except Exception as e:
        print("[Warn] GCP Translation init failed (env):", e, flush=True)

init_gcp_from_env()
if gcp_translate_client is None:
    init_gcp_local()

_MATH_ENV_NAMES = r"(?:equation|align|gather|multline|eqnarray|cases|split|aligned)\*?"
_MATH_PATTERN = re.compile(
    r"(?P<D2>\${2}[\s\S]*?\${2})"
    r"|(?P<D1>(?<!\\)\$[\s\S]*?(?<!\\)\$)"
    r"|(?P<LB>\\\[[\s\S]*?\\\])"
    r"|(?P<LP>\\\([\s\S]*?\\\))"
    r"|(?P<ENV>\\begin\{" + _MATH_ENV_NAMES + r"\}[\s\S]*?\\end\{" + _MATH_ENV_NAMES + r"\})",
    re.MULTILINE
)

_KEEP_TERMS = [
    r"\bIOU\b", r"\bIoU\b", r"\bNMS\b", r"\bmAP\b", r"\bAP50\b",
    r"\bBCE\b", r"\bCE\b", r"\bMSE\b", r"\bSGD\b",
    r"\bReLU\b", r"\bGELU\b", r"\bSoftmax\b", r"\bSigmoid\b",
    r"\bobjectness\b", r"\blogits\b", r"\bbbox(?:es)?\b", r"\bone-hot\b",
    r"\bClass_[A-Za-z0-9]+\b", r"\bObject\b", r"\bIoU_pred\b", r"\bIoU_truth\b"
]
_KEEP_RE = re.compile("|".join(_KEEP_TERMS))

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

def protect_parens(text: str) -> Tuple[str, Dict[str, str]]:
    holders = {}; idx = 0
    pat = re.compile(r"\([^()\n]*\)")
    out = text
    while True:
        changed = False
        def _repl(m):
            nonlocal idx, changed
            key = f"⟦P{idx}⟧"; holders[key] = m.group(0); idx += 1; changed = True
            return key
        out2 = pat.sub(_repl, out); out = out2
        if not changed: break
    return out, holders

def restore_parens(text: str, holders: Dict[str, str]) -> str:
    for k, v in holders.items():
        text = text.replace(k, v)
    return text

def protect_terms(text: str) -> Tuple[str, Dict[str, str]]:
    holders = {}
    def repl(m):
        key = f"⟦KEEP{len(holders)}⟧"
        holders[key] = m.group(0)
        return key
    return _KEEP_RE.sub(repl, text), holders

def restore_terms(text: str, holders: Dict[str, str]) -> str:
    for k, v in holders.items():
        text = text.replace(k, v)
    return text

def split_into_paragraphs(s: str) -> List[str]:
    return re.split(r"\n\s*\n", s)

def join_paragraphs(paragraphs: List[str]) -> str:
    return "\n\n".join(paragraphs)

def translate_paragraphs_gcp(paragraphs: List[str], target_lang="ko") -> List[str]:
    if gcp_translate_client is None or GCP_PARENT is None:
        print("[Warn] GCP Translation not ready; return original.", flush=True)
        return paragraphs[:]

    prot_list, holders_math, holders_paren, holders_terms = [], [], [], []
    for para in paragraphs:
        if not para.strip():
            prot_list.append(""); holders_math.append({}); holders_paren.append({}); holders_terms.append({})
            continue
        p1, h_m = protect_math(para)
        p2, h_p = protect_parens(p1)
        p3, h_t = protect_terms(p2)
        prot_list.append(p3); holders_math.append(h_m); holders_paren.append(h_p); holders_terms.append(h_t)

    out_list = [""] * len(paragraphs)

    BATCH = 32
    def _flush_batch(idxs: List[int]):
        if not idxs: return
        nonempty = [i for i in idxs if prot_list[i].strip() != ""]
        if not nonempty:
            for i in idxs: out_list[i] = prot_list[i]
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
            translated = contents
        for j, idx in enumerate(nonempty):
            t = translated[j]
            t = restore_parens(t, holders_paren[idx])
            t = restore_math(t, holders_math[idx])
            t = restore_terms(t, holders_terms[idx])
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
    overview_en = data.get("overview", "")
    overview_ko = translate_text_gcp(overview_en, target_lang=target_lang) if overview_en else ""
    items_ko = []
    for it in data.get("items", []):
        exp_en = it.get("explanation", "")
        exp_ko = translate_text_gcp(exp_en, target_lang=target_lang) if exp_en else ""
        it_ko = dict(it); it_ko["explanation"] = exp_ko
        items_ko.append(it_ko)
    out_obj = {"overview": overview_ko, "items": items_ko}
    Path(out_json_path).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 번역본 JSON 저장: {out_json_path}", flush=True)

# === 메인 파이프라인 ===
MAX_EXPLAINS = 40

def run_pipeline(input_tex_path: str) -> Dict:
    p = Path(input_tex_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find TeX file: {input_tex_path}")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    src = p.read_text(encoding="utf-8", errors="ignore")
    offsets = make_line_offsets(src); pos_to_line = build_pos_to_line(offsets)
    equations_all = extract_equations(src, pos_to_line)
    equations_advanced = [e for e in equations_all if is_advanced(e["body"])]
    print(f"총 수식: {len(equations_all)}", flush=True)
    print(f"중학생 수준 이상: {len(equations_advanced)} / {len(equations_all)}", flush=True)

    # 논문 매크로 파싱
    macro_defs = extract_macro_definitions(src)

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
    target_items = equations_advanced[:MAX_EXPLAINS]
    for idx, item in enumerate(target_items, start=1):
        print(f"[{idx}/{len(target_items)}] 라인 {item['line_start']}–{item['line_end']}", flush=True)
        paper_ctx = build_paper_context(item, src, macro_defs)
        raw_exp = explain_equation_with_llm(item["body"], paper_ctx)
        raw_exp = ensure_conclusion(raw_exp, item["body"], paper_ctx)
        exp = sanitize_explanation(raw_exp, item["body"])
        explanations.append({
            "index": idx, "line_start": item["line_start"], "line_end": item["line_end"],
            "kind": item["kind"], "env": item["env"],
            "equation": item["body"], "explanation": exp
        })

    json_path = os.path.join(OUT_DIR, "equations_explained.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overview": doc_overview, "items": explanations}, f, ensure_ascii=False, indent=2)
    print(f"저장된 JSON: {json_path}", flush=True)

    report_tex_path = os.path.join(OUT_DIR, "yolo_math_report.tex")
    report_tex = build_report(doc_overview, explanations)
    Path(report_tex_path).write_text(report_tex, encoding="utf-8")
    print(f"저장된 TeX: {report_tex_path}", flush=True)

    json_ko_path = os.path.join(OUT_DIR, "equations_explained.ko.json")
    tex_ko_path  = os.path.join(OUT_DIR, "yolo_math_report.ko.tex")
    try:
        translate_json_payload(json_path, json_ko_path, target_lang="ko")
    except Exception as e:
        print("[Translate JSON Error]", e, file=sys.stderr)
    try:
        translate_tex_file(report_tex_path, tex_ko_path, target_lang="ko")
    except Exception as e:
        print("[Translate TeX Error]", e, file=sys.stderr)

    return {
        "input": str(p),
        "counts": {"총 수식": len(equations_all), "중학생 수준 이상": len(equations_advanced)},
        "outputs": {
            "json": json_path, "report_tex": report_tex_path,
            "json_ko": json_ko_path, "report_tex_ko": tex_ko_path,
            "out_dir": OUT_DIR
        }
    }

# === HTML 미리보기(개요 + 수식 + 해설/번역) ===
from urllib.parse import quote

def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot read JSON: {p} ({e})")

def _mathjax_macros_block() -> str:
    return """
      macros: {
        mathlarger: ['{\\\\large #1}', 1],
        mathbbm:    ['{\\\\mathbb{#1}}', 1],
        wt:         ['{\\\\widetilde{#1}}', 1],
        wh:         ['{\\\\widehat{#1}}', 1],
        dfn:        '{\\\\triangleq}',
        dB:         '{\\\\mathrm{dB}}',

        snr:        '{\\\\mathrm{SNR}}',
        bsnr:       '{\\\\mathrm{S}\\\\widetilde{\\\\mathrm{N}}\\\\mathrm{R}}',
        dsnr:       '{\\\\Delta\\\\snr}'
      }
    """.strip()

# 1) 새로 추가: 따뜻한 테마 CSS 공통 함수
def _warm_styles() -> str:
    return r"""
  :root {
    --bg: #f8f5f0;            /* 크림 */
    --fg: #3b2f2f;            /* 다크 브라운 */
    --muted: #6b4f4f;         /* 브라운 톤 보조 */
    --panel: #fff7ed;         /* 살구빛 패널 */
    --panel-2: #fff3e0;       /* 조금 더 진한 살구 */
    --bd: #f2e8de;            /* 따뜻한 테두리 */
    --acc: #ea580c;           /* 오렌지(포인트) */
    --acc-2: #f59e0b;         /* 앰버(보조 포인트) */
    --ring: rgba(234, 88, 12, .25);
  }

  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, 'Noto Sans KR', sans-serif;
    color: var(--fg);
    background:
      radial-gradient(1000px 600px at 10% -10%, #fffaf3 0%, transparent 60%),
      radial-gradient(900px 500px at 100% 0%, #fff5ea 0%, transparent 60%),
      var(--bg);
    margin: 0;
    line-height: 1.75;
  }

  .wrap { max-width: 960px; margin: 0 auto; padding: 28px; }

  h1, h2, h3 { color: #2c2424; margin: 0 0 12px; letter-spacing: .2px; }
  .muted { color: var(--muted); }

  .topbar {
    display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:18px;
  }

  .badge {
    font-size:12px; color:#5b3a1f;
    background: linear-gradient(180deg, #fff0e0, #ffe7d1);
    border:1px solid var(--bd);
    padding:6px 10px; border-radius:999px;
    box-shadow: 0 2px 8px rgba(234, 88, 12, .08);
  }

  .card {
    background: linear-gradient(180deg, var(--panel), #fffaf4);
    border:1px solid var(--bd);
    border-radius:16px;
    padding:18px 20px; margin: 18px 0;
    box-shadow:
      0 10px 30px rgba(234, 88, 12, .08),
      0 2px 6px rgba(60, 32, 8, .04);
  }

  .eq {
    background: linear-gradient(180deg, #fff9ef, #fff5e7);
    border:1px dashed #eddfce;
    border-radius:14px;
    padding:14px; overflow:auto; margin: 10px 0 14px;
  }

  .tabs, .ovtabs { display:flex; gap:8px; margin-bottom:8px; flex-wrap: wrap; }
  .tab {
    background: #fff3e6;
    color:#5b3a1f;
    border:1px solid #f0dfcf;
    border-radius:999px;
    padding:6px 12px; cursor:pointer;
    transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
  }
  .tab:hover { transform: translateY(-1px); box-shadow: 0 3px 10px rgba(245, 158, 11, .12); }
  .tab.active {
    background: linear-gradient(180deg, #ffe7d1, #ffdcb9);
    border-color: var(--acc-2);
    color:#3b2512;
    box-shadow: 0 4px 14px rgba(245, 158, 11, .18);
  }

  .pane { display:none; white-space:pre-wrap; }
  .pane.show { display:block; }

  a.btn {
    text-decoration:none;
    background: linear-gradient(180deg, #ff8a3d, #ea580c);
    color:#fff; padding:10px 14px; border-radius:12px;
    box-shadow: 0 8px 18px rgba(234, 88, 12, .25);
    border: 1px solid rgba(0,0,0,.05);
  }
  a.btn:focus, .tab:focus { outline: 3px solid var(--ring); outline-offset: 1px; }

  /* 코드/리스트 가독성 */
  code, pre { background:#fff3e6; border:1px solid #f0dfcf; border-radius:8px; padding:.2em .4em; }
  ul, ol { padding-left: 1.2rem; }
    """

def _render_html(doc_en: dict, doc_ko: dict) -> str:
    items_en = doc_en.get("items", [])
    items_ko = doc_ko.get("items", [])
    ko_by_idx = {it.get("index"): it for it in items_ko}

    def sec(it_en):
        idx = it_en.get("index")
        it_ko = ko_by_idx.get(idx, {})
        title = f"Lines {it_en.get('line_start')}–{it_en.get('line_end')} / {it_en.get('kind')} {('['+it_en.get('env','')+']') if it_en.get('env') else ''}"

        eq_raw = normalize_for_mathjax(it_en.get("equation", ""))
        eq_wrapped = wrap_for_render(eq_raw, it_en.get("env","") or "")
        eq_block = f"<div class='eq'>$$\n{eq_wrapped}\n$$</div>"

        exp_en = it_en.get("explanation", "").strip()
        exp_ko = (it_ko.get("explanation", "") or "").strip() or "<em>번역 없음</em>"

        return f"""
        <section class="card">
          <h3>{title}</h3>
          {eq_block}
          <div class="tabs">
            <button class="tab active" data-for="en-{idx}">EN</button>
            <button class="tab" data-for="ko-{idx}">KR</button>
          </div>
          <div id="en-{idx}" class="pane show">{exp_en}</div>
          <div id="ko-{idx}" class="pane">{exp_ko}</div>
        </section>
        """

    overview_en = doc_en.get("overview","").strip() or "<em>No overview</em>"
    overview_ko = doc_ko.get("overview","").strip() or "<em>개요 번역 없음</em>"
    body_sections = "\n".join(sec(it) for it in items_en)
    macros_block = _mathjax_macros_block()

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>POLO – LaTeX Math HTML Preview</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{_warm_styles()}</style>
<script>
  addEventListener('click', (e) => {{
    if (!e.target.classList.contains('tab')) return;
    const btn = e.target;
    const paneId = btn.dataset.for;
    btn.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    const sec = btn.closest('section');
    sec.querySelectorAll('.pane').forEach(p => p.classList.remove('show'));
    sec.querySelector('#'+paneId).classList.add('show');
  }});
</script>
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['\\\\(','\\\\)'], ['$', '$']],
      displayMath: [['\\\\[','\\\\]'], ['$$','$$']],
      {macros_block}
    }},
    svg: {{ fontCache: 'global' }}
  }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <h1>POLO – Equation Overview & Explanations</h1>
      <span class="badge">HTML Preview</span>
    </div>

    <div class="card">
      <h2>Document Overview</h2>
      <div class="ovtabs">
        <button class="tab active" data-for="ov-en">EN</button>
        <button class="tab" data-for="ov-ko">KR</button>
      </div>
      <div id="ov-en" class="pane show">{overview_en}</div>
      <div id="ov-ko" class="pane">{overview_ko}</div>
    </div>

    {body_sections}

    <div class="muted" style="margin-top:24px;">Rendered with MathJax • POLO</div>
  </div>
</body>
</html>"""

# === HTML 즉시 렌더(디스크 저장 없이) ===
def _render_live_html(overview_en: str, overview_ko: str, items: list) -> str:
    sections = []
    for it in items:
        eq_raw = normalize_for_mathjax(it['equation'])
        eq_wrapped = wrap_for_render(eq_raw, it.get("env","") if isinstance(it, dict) else "")
        eq_block = f"<div class='eq'>$$\n{eq_wrapped}\n$$</div>"
        sections.append(f"""
        <section class="card">
          <h3>{it['title']}</h3>
          {eq_block}
          <div class="tabs">
            <button class="tab active" data-for="en-{it['index']}">EN</button>
            <button class="tab" data-for="ko-{it['index']}">KR</button>
          </div>
          <div id="en-{it['index']}" class="pane show">{it['exp_en']}</div>
          <div id="ko-{it['index']}" class="pane">{it['exp_ko'] or '<em>번역 없음</em>'}</div>
        </section>
        """)

    body_sections = "\n".join(sections)
    macros_block = _mathjax_macros_block()

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>POLO – Live Math Preview</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{_warm_styles()}</style>
<script>
  addEventListener('click', (e) => {{
    if (!e.target.classList.contains('tab')) return;
    const btn = e.target;
    btn.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    const sec = btn.closest('section');
    sec.querySelectorAll('.pane').forEach(p => p.classList.remove('show'));
    sec.querySelector('#'+btn.dataset.for).classList.add('show');
  }});
</script>
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['\\\\(','\\\\)'], ['$', '$']],
      displayMath: [['\\\\[','\\\\]'], ['$$','$$']],
      {macros_block}
    }},
    svg: {{ fontCache: 'global' }}
  }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <h1>POLO – Equation Overview & Explanations (Live)</h1>
      <span class="badge">No disk writes</span>
    </div>

    <div class="card">
      <h2>Document Overview</h2>
      <div class="tabs">
        <button class="tab active" data-for="ov-en">EN</button>
        <button class="tab" data-for="ov-ko">KR</button>
      </div>
      <div id="ov-en" class="pane show">{overview_en or '<em>No overview</em>'}</div>
      <div id="ov-ko" class="pane">{overview_ko or '<em>개요 번역 없음</em>'}</div>
    </div>

    {body_sections}
    <div class="muted" style="margin-top:24px;">Rendered with MathJax • POLO</div>
  </div>
</body>
</html>"""

# === FastAPI 앱 ===
app = FastAPI(title="POLO Math Explainer API", version="1.4.1")  # 그대로

@app.get("/html/{file_path:path}", response_class=HTMLResponse)
async def html_preview(file_path: str):
    try:
        result = run_pipeline(file_path)
        out_dir = Path(result["outputs"]["out_dir"])
        en_json = out_dir / "equations_explained.json"
        ko_json = out_dir / "equations_explained.ko.json"
        doc_en = _read_json(en_json)
        doc_ko = _read_json(ko_json) if ko_json.exists() else {"overview":"", "items":[]}
        html = _render_html(doc_en, doc_ko)
        return HTMLResponse(html)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.get("/html-live/{file_path:path}", response_class=HTMLResponse)
async def html_live(file_path: str):
    try:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Cannot find TeX file: {file_path}")

        src = p.read_text(encoding="utf-8", errors="ignore")
        offsets = make_line_offsets(src)
        pos_to_line = build_pos_to_line(offsets)
        equations_all = extract_equations(src, pos_to_line)
        equations_adv = [e for e in equations_all if is_advanced(e["body"])]

        # 논문 매크로 파싱
        macro_defs = extract_macro_definitions(src)

        head, mid, tail = take_slices(src)
        overview_prompt = f"""
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
""".strip()
        overview_en = chat_overview(overview_prompt)

        items = []
        for idx, it in enumerate(equations_adv[:MAX_EXPLAINS], start=1):
            paper_ctx = build_paper_context(it, src, macro_defs)
            raw_exp_en = explain_equation_with_llm(it["body"], paper_ctx)
            raw_exp_en = ensure_conclusion(raw_exp_en, it["body"], paper_ctx)
            exp_en = sanitize_explanation(raw_exp_en, it["body"])
            exp_ko = translate_text_gcp(exp_en, target_lang="ko") if (gcp_translate_client and GCP_PARENT) else ""
            title = f"Lines {it['line_start']}–{it['line_end']} / {it['kind']} {('['+it['env']+']') if it['env'] else ''}"
            # env도 넘겨서 라이브 미리보기에서도 감싸기 동작
            items.append({
                "index": idx,
                "title": title,
                "equation": it["body"],
                "env": it.get("env",""),
                "exp_en": exp_en,
                "exp_ko": exp_ko
            })

        overview_ko = translate_text_gcp(overview_en, target_lang="ko") if (gcp_translate_client and GCP_PARENT) else ""
        html = _render_live_html(overview_en, overview_ko, items)
        return HTMLResponse(html)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
@app.get("/")
async def root():
    return RedirectResponse(url="/health")

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
        "model_loaded": model_loaded,
        "tokenizer_loaded": tokenizer is not None,
        "model_name": MODEL_ID,
        "cache_dir": str(SAFE_CACHE_DIR),
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

if __name__ == "__main__":
    try:
        import uvicorn
        print("🚀 Math Model 서버 시작 중...")
        uvicorn.run("app:app", host="127.0.0.1", port=5004, reload=False)
    except Exception as e:
        print(f"❌ Math Model 시작 실패: {e}")
        import traceback; traceback.print_exc()
        input("Press Enter to exit...")
