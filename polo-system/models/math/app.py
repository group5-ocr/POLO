# -*- coding: utf-8 -*-
"""
LaTeX 수식 해설 API (FastAPI)

실행 방법
- 개발 모드(핫 리로드): uvicorn --reload app:app
- 프로덕션 모드(권장):   uvicorn app:app

제공 엔드포인트
- GET  /health
  : 서버/모델 상태 점검용 엔드포인트입니다.

- GET  /count/{file_path:path}
  : 특정 TeX 파일의 수식을 추출해 "총 수식 개수"와 "고난도(중학생 이상) 수식 개수"만 빠르게 계산합니다.
    콘솔에도 즉시 출력(printf)되며, JSON으로 개수만 반환합니다.

- POST /count
  : {"path": "C:\\...\\yolo.tex"} 형식의 JSON으로 위와 동일한 기능을 제공합니다.

- GET  /math/{file_path:path}
  : 전체 파이프라인 실행(수식 추출/분류 → 문서 개요(영어) 생성 → 고난도 수식에 대한 해설(영어) 생성 → JSON/TeX 저장)

- POST /math
  : {"path": "C:\\...\\yolo.tex"} 형식의 JSON으로 위와 동일한 기능을 제공합니다.

주의/특징
- 콘솔 출력이 지연되지 않도록 stdout 라인 버퍼링 + print(..., flush=True)를 사용합니다.
- 일부 모델에서 pad_token과 eos_token이 같을 때 뜨는 경고를 피하기 위해,
  pad 토큰이 없거나 eos와 같으면 [PAD] 토큰을 추가하고, generate 시 attention_mask를 명시적으로 전달합니다.
- LLM 프롬프트(문서 개요/수식 해설)는 요청에 따라 영어로 작성되어 있습니다.
"""

# === 셀 1: 환경 준비 & 모델 로드 ===
import os, sys, json, re, textwrap, datetime, torch
from typing import List, Dict
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# [권장] 콘솔 출력이 바로 보이도록 stdout을 줄 단위로 버퍼링합니다.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

VERSION = "POLO-Math-API v4 (EN-prompts + flush + mask + pad)"
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
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU 사용 가능: {gpu_name}", flush=True)
    print(f"🔧 디바이스: {DEVICE}, 데이터 타입: float16", flush=True)
else:
    print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.", flush=True)
    print(f"🔧 디바이스: {DEVICE}, 데이터 타입: float32", flush=True)

# --- 추가: 안전 HF 캐시 + .env ---
try:
    from dotenv import load_dotenv
    ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
    if ROOT_ENV.exists():
        load_dotenv(dotenv_path=str(ROOT_ENV), override=True)
        print(f"[dotenv] loaded: {ROOT_ENV}", flush=True)
except Exception:
    pass

SAFE_CACHE_DIR = Path(__file__).resolve().parent / "hf_cache"
SAFE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _force_safe_hf_cache():
    for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        os.environ[k] = str(SAFE_CACHE_DIR)
    print(f"[hf_cache] forced → {SAFE_CACHE_DIR}", flush=True)

_force_safe_hf_cache()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

try:
    # 1) 토크나이저 (캐시/토큰 명시)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
        cache_dir=str(SAFE_CACHE_DIR),
    )

    # 2) pad 토큰 보정
    PAD_ADDED = False
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        PAD_ADDED = True

    # 3) 모델 로드 (필요 시 4bit)
    bnb_config = None
    if USE_4BIT:
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
        trust_remote_code=True,
        token=HF_TOKEN,
        cache_dir=str(SAFE_CACHE_DIR),
    )

    if PAD_ADDED:
        model.resize_token_embeddings(len(tokenizer))

    GEN_KW = dict(
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )
    print("Model & tokenizer loaded.", flush=True)

except Exception as e:
    tokenizer = None
    model = None
    GEN_KW = {}
    print("[Model Load Error]", e, flush=True)


# === 공통 유틸: 라인 오프셋 인덱스 ===
def make_line_offsets(text: str) -> List[int]:
    lines = text.splitlines()
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
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
        return hi + 1  # 1-based
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
    r"\\frac\{[^{}]*\{[^{}]*\}[^{}]*\}",  # nested fraction
    r"\\hat\{", r"\\tilde\{", r"\\bar\{", r"\\widehat\{", r"\\widetilde\{",
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
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded.")
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user",   "content": EXPLAIN_TEMPLATE.format(EQUATION=eq_latex)}
    ]
    text = _generate_with_mask_from_messages(messages)
    return text.split(messages[-1]["content"])[-1].strip()


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

    json_path = os.path.join(OUT_DIR, "equations_explained.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overview": doc_overview, "items": explanations}, f, ensure_ascii=False, indent=2)
    print(f"저장된 JSON: {json_path}", flush=True)

    report_tex_path = os.path.join(OUT_DIR, "yolo_math_report.tex")
    report_tex = build_report(doc_overview, explanations)
    Path(report_tex_path).write_text(report_tex, encoding="utf-8")
    print(f"저장된 TeX: {report_tex_path}", flush=True)

    return {
        "input": str(p),
        "counts": {
            "총 수식": len(equations_all),
            "중학생 수준 이상": len(equations_advanced)
        },
        "outputs": {
            "json": json_path,
            "report_tex": report_tex_path,
            "out_dir": OUT_DIR
        }
    }


# === FastAPI 앱 ===
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
        "model_loaded": (tokenizer is not None and model is not None)
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
    try:
        import uvicorn
        print("🚀 Math Model 서버 시작 중...")
        uvicorn.run("app:app", host="0.0.0.0", port=5004, reload=False)
    except Exception as e:
        print(f"❌ Math Model 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
