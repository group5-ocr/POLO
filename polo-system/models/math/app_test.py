# -*- coding: utf-8 -*-
"""
LaTeX 수식 해설 API (FastAPI)

실행 방법
- 개발 모드(핫 리로드): uvicorn --reload app:app
- 프로덕션 모드(권장):   uvicorn app:app

제공 엔드포인트
- GET  /health
- GET  /count/{file_path:path}
- POST /count
- GET  /math/{file_path:path}
- POST /math

주의/특징
- stdout 라인 버퍼링
- pad_token 보정 + attention_mask 명시 전달
- LLM 프롬프트(개요/해설)는 영어
- (추가) googletrans로 한국어 번역본 JSON/TeX도 생성
"""

# === 셀 1: 환경 준비 & 모델 로드 ===
import os, sys, json, re, textwrap, datetime, torch
from typing import List, Dict, Tuple
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# (추가) 구글 번역기
try:
    # pip install googletrans==4.0.0rc1 권장
    from googletrans import Translator
    _GT_AVAILABLE = True
except Exception:
    Translator = None
    _GT_AVAILABLE = False

# [권장] 콘솔 출력 즉시화
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

VERSION = "POLO-Math-API v5 (EN->KO translate + math-protect)"
print(VERSION, flush=True)

# ----- 경로 설정 -----
INPUT_TEX_PATH = r"C:\\POLO\\polo-system\\models\\math\\yolo.tex"
OUT_DIR        = "C:/POLO/polo-system/models/math/_build"
os.makedirs(OUT_DIR, exist_ok=True)

# ----- 모델/토크나이저 설정 -----
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
USE_4BIT = False
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python: {sys.version.split()[0]}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Device selected: {DEVICE}", flush=True)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    PAD_ADDED = False
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        PAD_ADDED = True

    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
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

except Exception as e:
    tokenizer = None
    model = None
    GEN_KW = {}
    print("[Model Load Error]", e, flush=True)

# (추가) 번역기 인스턴스
translator = None
if _GT_AVAILABLE:
    try:
        # googletrans는 기본 엔드포인트 이슈가 있을 수 있어 보조 URL을 지정
        translator = Translator(service_urls=["translate.googleapis.com", "translate.google.com"])
        print("Google Translator initialized.", flush=True)
    except Exception as e:
        print("[Translator Init Error]", e, flush=True)
        translator = None

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
            "gather","gather*","flalign","flalign*","eqnarray","eqnarray*","split"]
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

# === 셀 3: 난이도 휴리스틱 ===
ADV_TOKENS = [
    r"\\sum", r"\\prod", r"\\int", r"\\lim", r"\\nabla", r"\\partial",
    r"\\mathbb", r"\\mathcal", r"\\mathbf", r"\\boldsymbol",
    r"\\argmax", r"\\argmin", r"\\operatorname", r"\\mathrm\{KL\}",
    r"\\mathbb\{E\}", r"\\Pr", r"\\sigma", r"\\mu", r"\\Sigma", r"\\theta",
    r"\\frac\{[^{}]*\{[^{}]*\}[^{}]*\}",
    r"\\hat\{", r"\\tilde\{", r"\\bar\{", r"\\widehat\{", r"\\widetilde\{",
    r"\\sqrt\{[^{}]*\{",
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

# === 셀 4: 개요 생성 ===
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

# === 셀 5: 수식 해설 생성 ===
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
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded.")
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user",   "content": EXPLAIN_TEMPLATE.format(EQUATION=eq_latex)}
    ]
    text = _generate_with_mask_from_messages(messages)
    return text.split(messages[-1]["content"])[-1].strip()

# === 셀 6: LaTeX 리포트(.tex) 생성 ===
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
        # 설명 텍스트는 (의도적으로) verbatim 이스케이프하지 않음:
        # - 수식 블록(예: \[...\])이 그대로 LaTeX로 렌더되도록
        parts.append(it["explanation"])
        parts.append("\n")

    parts.append("\\end{document}\n")
    return "\n".join(parts)

# (추가) 한국어 리포트 빌더
def build_report_ko(overview_ko: str, items_ko: List[Dict]) -> str:
    header = (r"""\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{amsmath, amssymb, amsfonts}
\\usepackage{hyperref}
\\usepackage{kotex}
\\setlength{\\parskip}{6pt}
\\setlength{\\parindent}{0pt}
\\title{LaTeX 수식 해설 리포트 (중학생 이상)}
\\author{자동 생성 파이프라인}
\\date{""" + datetime.date.today().isoformat() + r"""}
\\begin{document}
\\maketitle
\\tableofcontents
\\newpage
""")
    parts = [header]
    parts.append(r"\\section*{문서 개요}")
    parts.append(latex_escape_verbatim(overview_ko))
    parts.append("\n\\newpage\n")

    for it in items_ko:
        title = f"라인 {it['line_start']}–{it['line_end']} / {it['kind']} {('['+it['env']+']') if it['env'] else ''}"
        parts.append(f"\\section*{{{latex_escape_verbatim(title)}}}")
        parts.append(it["explanation"])  # 수식은 보호/복원되어 LaTeX 유지
        parts.append("\n")

    parts.append("\\end{document}\n")
    return "\n".join(parts)

# === (추가) 번역 유틸: 수식 보호/복원 + 번역 ===
# === (수정) 번역 유틸: 수식 보호/복원 + 번역 ===
_MATH_ENV_NAMES = r"(?:equation|align|gather|multline|eqnarray|cases|split)\*?"
_MATH_PATTERN = re.compile(
    r"(?P<D2>\${2}[\s\S]*?\${2})"           # $$ ... $$
    r"|(?P<D1>(?<!\\)\$[\s\S]*?(?<!\\)\$)"  # $ ... $ (이스케이프 제외)
    r"|(?P<LB>\\\[[\s\S]*?\\\])"            # \[ ... \]
    r"|(?P<LP>\\\([\s\S]*?\\\))"            # \( ... \)
    r"|(?P<ENV>\\begin\{" + _MATH_ENV_NAMES + r"\}[\s\S]*?\\end\{" + _MATH_ENV_NAMES + r"\})",
    re.MULTILINE
)


def protect_math(text: str) -> Tuple[str, Dict[str, str]]:
    """
    수식 블록을 보호 토큰으로 치환하여 번역 시 변형을 방지합니다.
    """
    placeholders = {}
    def _repl(m):
        key = f"⟦MATH{len(placeholders)}⟧"
        placeholders[key] = m.group(0)
        return key
    protected = _MATH_PATTERN.sub(_repl, text)
    return protected, placeholders

def restore_math(text: str, placeholders: Dict[str, str]) -> str:
    for k, v in placeholders.items():
        text = text.replace(k, v)
    return text

def translate_text_ko(text: str) -> str:
    """
    영문 설명/개요를 한국어로 번역합니다.
    - 수식 보호 후 번역 → 복원
    - googletrans 사용 불가 시 원문 그대로 반환(로그만 출력)
    """
    if translator is None:
        print("[Translate] Translator unavailable; returning original.", flush=True)
        return text

    # 문단 단위로 쪼개 번역(googletrans 길이 제한/안정성 보완)
    paras = text.split("\n\n")
    out_paras = []
    for para in paras:
        prot, ph = protect_math(para)
        try:
            t = translator.translate(prot, dest="ko").text
        except Exception as e:
            print("[Translate Error]", e, flush=True)
            t = prot  # 실패 시 원문 유지
        out_paras.append(restore_math(t, ph))
    return "\n\n".join(out_paras)

# === 보조: 수식 개수만 빠르게 세기 ===
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

# === 메인 파이프라인 ===
def run_pipeline(input_tex_path: str) -> Dict:
    p = Path(input_tex_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find TeX file: {input_tex_path}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1) 파일 읽기
    src = p.read_text(encoding="utf-8", errors="ignore")

    # 2) 라인 인덱스
    offsets = make_line_offsets(src)
    pos_to_line = build_pos_to_line(offsets)

    # 3) 수식 추출 & 고난도 분류
    equations_all = extract_equations(src, pos_to_line)
    equations_advanced = [e for e in equations_all if is_advanced(e["body"])]

    print(f"총 수식: {len(equations_all)}", flush=True)
    print(f"중학생 수준 이상: {len(equations_advanced)} / {len(equations_all)}", flush=True)

    # 4) 문서 개요(영어)
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

    # 5) 고난도 수식 해설(영어)
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

    # 6) JSON 저장 (영문)
    json_path = os.path.join(OUT_DIR, "equations_explained.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overview": doc_overview, "items": explanations}, f, ensure_ascii=False, indent=2)
    print(f"저장된 JSON: {json_path}", flush=True)

    # 7) LaTeX 리포트(.tex) 저장 (영문)
    report_tex_path = os.path.join(OUT_DIR, "yolo_math_report.tex")
    report_tex = build_report(doc_overview, explanations)
    Path(report_tex_path).write_text(report_tex, encoding="utf-8")
    print(f"저장된 TeX: {report_tex_path}", flush=True)

    # 8) (추가) 한국어 번역본 생성 및 저장
    # 8-1) 개요 번역
    overview_ko = translate_text_ko(doc_overview)

    # 8-2) 각 해설 번역 (수식 보호)
    ko_items: List[Dict] = []
    for it in explanations:
        exp_ko = translate_text_ko(it["explanation"])
        ko_items.append({
            **{k: it[k] for k in ["index","line_start","line_end","kind","env","equation"]},
            "explanation": exp_ko
        })

    # 8-3) JSON 저장 (한국어)
    json_ko_path = os.path.join(OUT_DIR, "equations_explained.ko.json")
    with open(json_ko_path, "w", encoding="utf-8") as f:
        json.dump({"overview": overview_ko, "items": ko_items}, f, ensure_ascii=False, indent=2)
    print(f"저장된 한국어 JSON: {json_ko_path}", flush=True)

    # 8-4) LaTeX 리포트(.tex) 저장 (한국어)
    report_ko_tex_path = os.path.join(OUT_DIR, "yolo_math_report.ko.tex")
    report_ko_tex = build_report_ko(overview_ko, ko_items)
    Path(report_ko_tex_path).write_text(report_ko_tex, encoding="utf-8")
    print(f"저장된 한국어 TeX: {report_ko_tex_path}", flush=True)

    # 9) 처리 요약 반환
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
            "report_tex_ko": report_ko_tex_path,
            "out_dir": OUT_DIR
        },
        "translate": {
            "googletrans_available": (_GT_AVAILABLE and translator is not None)
        }
    }

# === FastAPI 앱 정의 ===
app = FastAPI(title="POLO Math Explainer API", version="1.0.0")

class MathRequest(BaseModel):
    path: str

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": DEVICE,
        "model_loaded": (tokenizer is not None and model is not None),
        "googletrans": (_GT_AVAILABLE and translator is not None)
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
