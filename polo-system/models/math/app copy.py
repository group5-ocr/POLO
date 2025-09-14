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
# 일부 Windows + uvicorn --reload 환경에서 print 출력이 늦게 보이는 문제를 완화합니다.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

VERSION = "POLO-Math-API v4 (EN-prompts + flush + mask + pad)"
print(VERSION, flush=True)

# ----- 경로 설정 -----
# INPUT_TEX_PATH: 기본 예시 경로 (엔드포인트에서 별도로 파일 경로를 넘기는 경우 이 값은 사용되지 않을 수 있습니다)
INPUT_TEX_PATH = r"C:\\POLO\\polo-system\\models\\math\\yolo.tex"
# OUT_DIR: 산출물 저장 폴더(JSON/TeX). 존재하지 않으면 생성합니다.
OUT_DIR        = "C:/POLO/polo-system/models/math/_build"
os.makedirs(OUT_DIR, exist_ok=True)

# ----- 모델/토크나이저 설정 -----
# MODEL_ID: 사용할 허깅페이스 모델 이름(Qwen 수학 특화 지시형 모델).
# USE_4BIT: VRAM 절약 목적의 4bit 양자화 사용 여부(필요 시 True로 설정).
# DEVICE: CUDA 사용 가능하면 'cuda', 아니면 'cpu'.
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
USE_4BIT = False
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# 현재 파이썬/파이토치/장치 상태를 콘솔에 출력합니다(디버깅 편의).
print(f"Python: {sys.version.split()[0]}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Device selected: {DEVICE}", flush=True)

try:
    # 1) 토크나이저 로드
    #    trust_remote_code=True는 모델 저장소의 커스텀 코드를 신뢰하고 로드한다는 의미입니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2) pad 토큰 보정
    #    - 일부 모델은 pad_token이 없거나 eos_token과 같은 값으로 설정되어 있습니다.
    #    - 이런 경우 attention_mask 자동 생성이 모호하여 경고/오동작이 발생할 수 있으므로
    #      [PAD]를 추가하여 pad_token_id를 명확히 만듭니다.
    PAD_ADDED = False
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        PAD_ADDED = True

    # 3) 모델 로드 (필요 시 4bit 양자화 설정)
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
        device_map="auto",  # 가용 장치에 자동 매핑
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 4) pad 토큰을 추가했다면 임베딩 테이블 크기 리사이즈가 필요합니다.
    if PAD_ADDED:
        model.resize_token_embeddings(len(tokenizer))

    # 5) 텍스트 생성 시 공통 하이퍼파라미터(필요 시 조절)
    GEN_KW = dict(
        max_new_tokens=512,  # 생성 최대 토큰 수
        temperature=0.2,     # 샘플링 온도(낮을수록 결정적)
        top_p=0.9,           # 누적 확률 상위 p만 샘플링
        do_sample=True       # 샘플링 사용(온도/탑P 적용)
    )
    print("Model & tokenizer loaded.", flush=True)

except Exception as e:
    # 모델 로드 실패 시 이후 요청에서 에러가 발생하도록 None 처리
    tokenizer = None
    model = None
    GEN_KW = {}
    print("[Model Load Error]", e, flush=True)


# === 공통 유틸: 라인 오프셋 인덱스 ===
def make_line_offsets(text: str) -> List[int]:
    """
    전체 텍스트를 줄 단위로 분할하여, 각 줄의 시작 인덱스(오프셋)를 리스트로 만듭니다.
    추후 특정 문자 위치(index)를 '몇 번째 줄'인지 빠르게 역매핑하는 데 사용합니다.
    """
    lines = text.splitlines()
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1  # 줄바꿈 문자('\n') 고려
    return offsets

def build_pos_to_line(offsets: List[int]):
    """
    문자 위치 p(0-based index)를 받아 1-based 라인 번호로 변환하는 함수를 반환합니다.
    이진 탐색을 사용해 빠르게 계산합니다.
    """
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
    """
    LaTeX 소스 문자열에서 다양한 수식 표기들을 탐지하여 추출합니다.
    - $$ ... $$, \[ ... \], \( ... \), inline $ ... $, 그리고 수식 환경(equation, align 등)
    - 각 수식에 대해 시작/끝 인덱스, 라인 번호, 원문(body) 등을 기록합니다.
    - 중복(동일 범위) 결과는 제거하고, 문서 내 등장 순서대로 정렬하여 반환합니다.
    """
    matches: List[Dict] = []

    def add(kind, start, end, body, env=""):
        matches.append({
            "kind": kind, "env": env, "start": start, "end": end,
            "line_start": pos_to_line(start), "line_end": pos_to_line(end),
            "body": body.strip()
        })

    # $$ ... $$ 디스플레이 수식
    for m in re.finditer(r"\$\$(.+?)\$\$", tex, flags=re.DOTALL):
        add("display($$ $$)", m.start(), m.end(), m.group(1))

    # \[ ... \] 디스플레이 수식
    for m in re.finditer(r"\\\[(.+?)\\\]", tex, flags=re.DOTALL):
        add("display(\\[ \\])", m.start(), m.end(), m.group(1))

    # \( ... \) 인라인 수식(별도 구분)
    for m in re.finditer(r"\\\((.+?)\\\)", tex, flags=re.DOTALL):
        add("inline(\\( \\))", m.start(), m.end(), m.group(1))

    # inline $...$ (단, $$ 제외)
    for m in re.finditer(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", tex, flags=re.DOTALL):
        add("inline($ $)", m.start(), m.end(), m.group(1))

    # 수식 환경들 (equation, align, gather 등)
    envs = ["equation","equation*","align","align*","multline","multline*",
            "gather","gather*","flalign","flalign*","eqnarray","eqnarray*","split"]
    for env in envs:
        pattern = rf"\\begin{{{re.escape(env)}}}(.+?)\\end{{{re.escape(env)}}}"
        for m in re.finditer(pattern, tex, flags=re.DOTALL):
            add("env", m.start(), m.end(), m.group(1), env=env)

    # 동일 범위 중복 제거
    uniq = {}
    for it in matches:
        key = (it["start"], it["end"])
        if key not in uniq:
            uniq[key] = it

    # 등장 순서대로 정렬
    out = list(uniq.values())
    out.sort(key=lambda x: x["start"])
    return out


# === 셀 3: 난이도 휴리스틱 정의 ===
# 아래 휴리스틱은 "중학생 이상" 난이도로 판단할 수식의 특징을 간단히 체크합니다.
# - 특정 수학 기호/명령어 포함(∑, ∫, 𝔼, KL 등)
# - 수식 길이, 첨자/윗첨자 개수, 줄바꿈 포함여부 등
ADV_TOKENS = [
    r"\\sum", r"\\prod", r"\\int", r"\\lim", r"\\nabla", r"\\partial",
    r"\\mathbb", r"\\mathcal", r"\\mathbf", r"\\boldsymbol",
    r"\\argmax", r"\\argmin", r"\\operatorname", r"\\mathrm\{KL\}",
    r"\\mathbb\{E\}", r"\\Pr", r"\\sigma", r"\\mu", r"\\Sigma", r"\\theta",
    r"\\frac\{[^{}]*\{[^{}]*\}[^{}]*\}",  # 중첩 분수 패턴
    r"\\hat\{", r"\\tilde\{", r"\\bar\{", r"\\widehat\{", r"\\widetilde\{",
    r"\\sqrt\{[^{}]*\{",                  # 중첩 제곱근
    r"\\left", r"\\right",
    r"\\in", r"\\subset", r"\\forall", r"\\exists",
    r"\\cdot", r"\\times", r"\\otimes",
    r"IoU", r"\\log", r"\\exp",
    r"\\mathbb\{R\}", r"\\mathbb\{N\}", r"\\mathbb\{Z\}",
    r"\\Delta", r"\\delta", r"\\epsilon", r"\\varepsilon",
]
ADV_RE = re.compile("|".join(ADV_TOKENS))

def count_subscripts(expr: str) -> int:
    """
    첨자('_') 사용 횟수를 대략적으로 셉니다.
    '{' 또는 '\' 다음에 오는 '_'도 감안하기 위해 단순한 정규식으로 세어 줍니다.
    """
    return len(re.findall(r"_[a-zA-Z0-9{\\]", expr))

def is_advanced(eq: str) -> bool:
    """
    수식 문자열 eq가 고난도(중학생 이상)인지 간단한 휴리스틱으로 판정합니다.
    - ADV_RE 키워드 존재
    - 길이가 길고 첨자가 일정 개수 이상
    - 여러 줄 수식(줄바꿈 포함) + 일정 길이 이상
    """
    if ADV_RE.search(eq):
        return True
    if len(eq) > 40 and count_subscripts(eq) >= 2:
        return True
    if "\n" in eq and len(eq) > 30:
        return True
    return False


# === 셀 4: 문서 개요(Overview) LLM 생성 ===
def take_slices(text: str, head_chars=4000, mid_chars=2000, tail_chars=4000):
    """
    긴 LaTeX 문서 전체를 모두 모델에 넣기 어렵기 때문에,
    앞/중/뒤 일부만 슬라이스하여 요약에 사용합니다.
    - head: 문서 앞부분
    - mid : 문서 중간의 중간(Middle 중심)
    - tail: 문서 끝부분
    """
    n = len(text)
    head = text[:min(head_chars, n)]
    mid_start = max((n // 2) - (mid_chars // 2), 0)
    mid = text[mid_start: mid_start + min(mid_chars, n)]
    tail = text[max(0, n - tail_chars):]
    return head, mid, tail

def _generate_with_mask_from_messages(messages: List[Dict]) -> str:
    """
    Qwen 지시형 포맷에 맞춰 chat 템플릿을 적용합니다.
    - padding=True로 패딩을 강제하고,
    - attention_mask를 직접 생성하여 generate에 명시적으로 전달합니다.
      (pad==eos 경고 및 비일관 동작 방지)
    """
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
    """
    문서 앞/중/뒤 슬라이스를 입력 받아, 영어로 간결한 문서 개요를 생성합니다.
    - system: 기술 블로그 스타일의 간결/명확 요약자 역할 부여
    - user  : 실제 프롬프트(영어)
    """
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded.")
    messages = [
        {"role": "system", "content":
         "You are a clear, concise technical writer who summarizes LaTeX-based AI papers for a general technical audience."},
        {"role": "user", "content": prompt}
    ]
    text = _generate_with_mask_from_messages(messages)
    # 모델 출력에서 사용자 프롬프트 이후 부분만 잘라 반환
    return text.split(messages[-1]["content"])[-1].strip()


# === 셀 5: 수식 해설(Explanation) LLM 생성 ===
# 시스템/템플릿은 영어로 유지(요청사항). 예시/설명/결론 순서로 설명하게 합니다.
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
    """
    단일 수식(LaTeX 문자열)에 대해 영어 해설을 생성합니다.
    - system: 설명자 역할 지시
    - user  : 템플릿 + 실제 수식 삽입
    """
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded.")
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user",   "content": EXPLAIN_TEMPLATE.format(EQUATION=eq_latex)}
    ]
    text = _generate_with_mask_from_messages(messages)
    # 사용자 프롬프트 이후 부분만 반환
    return text.split(messages[-1]["content"])[-1].strip()


# === 셀 6: LaTeX 리포트(.tex) 생성 ===
def latex_escape_verbatim(s: str) -> str:
    """
    LaTeX에서 의미 있는 특수문자들을 이스케이프합니다.
    - 리포트에 원문 텍스트(개요 등)를 안전하게 삽입하기 위한 유틸입니다.
    """
    s = s.replace("\\", r"\\")
    s = s.replace("#", r"\#").replace("$", r"\$")
    s = s.replace("%", r"\%").replace("&", r"\&")
    s = s.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
    s = s.replace("^", r"\^{}").replace("~", r"\~{}")
    return s

def build_report(overview: str, items: List[Dict]) -> str:
    """
    최종 LaTeX 리포트(.tex) 내용을 문자열로 생성합니다.
    - 개요(Overview) 섹션 1개
    - 각 수식 해설 섹션(라인/종류/환경 표기 + 모델 출력 텍스트)
    """
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
        # 해설은 이미 LaTeX이 아닌 일반 텍스트(영어)일 가능성이 크므로 그대로 삽입
        parts.append(it["explanation"])
        parts.append("\n")

    parts.append("\\end{document}\n")
    return "\n".join(parts)


# === 보조: 수식 개수만 빠르게 세기 ===
def count_equations_only(input_tex_path: str) -> Dict[str, int]:
    """
    파일을 읽어 수식을 추출하고, 고난도 수식 분류까지 수행하여 개수만 반환합니다.
    - 콘솔에도 합계를 즉시 출력합니다(디버깅/확인 편의).
    """
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
    """
    전체 처리 파이프라인:
    1) 파일 읽기
    2) 수식 추출 및 고난도 분류
    3) 문서 개요(영어) 생성
    4) 고난도 수식 각각에 대한 해설(영어) 생성
    5) JSON/TeX 산출물 저장
    6) 처리 요약(개수/경로) 반환
    """
    p = Path(input_tex_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find TeX file: {input_tex_path}")

    # 산출 폴더 보장
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1) 파일 읽기
    src = p.read_text(encoding="utf-8", errors="ignore")

    # 2) 라인 오프셋 준비
    offsets = make_line_offsets(src)
    pos_to_line = build_pos_to_line(offsets)

    # 3) 수식 추출 & 고난도 분류
    equations_all = extract_equations(src, pos_to_line)
    equations_advanced = [e for e in equations_all if is_advanced(e["body"])]

    # 콘솔에 합계 즉시 출력
    print(f"총 수식: {len(equations_all)}", flush=True)
    print(f"중학생 수준 이상: {len(equations_advanced)} / {len(equations_all)}", flush=True)

    # 4) 문서 개요 생성(영어 프롬프트)
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

    # 5) 고난도 수식 해설(영어) 생성
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

    # 6) JSON 저장
    json_path = os.path.join(OUT_DIR, "equations_explained.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overview": doc_overview, "items": explanations}, f, ensure_ascii=False, indent=2)
    print(f"저장된 JSON: {json_path}", flush=True)

    # 7) LaTeX 리포트(.tex) 저장
    report_tex_path = os.path.join(OUT_DIR, "yolo_math_report.tex")
    report_tex = build_report(doc_overview, explanations)
    Path(report_tex_path).write_text(report_tex, encoding="utf-8")
    print(f"저장된 TeX: {report_tex_path}", flush=True)

    # 8) 처리 요약 반환(엔드포인트 응답)
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


# === FastAPI 앱 정의 ===
app = FastAPI(title="POLO Math Explainer API", version="1.0.0")

# 요청 바디 유효성 검사용 모델
class MathRequest(BaseModel):
    path: str

@app.get("/health")
async def health():
    """
    서버 상태/환경 점검용 엔드포인트입니다.
    - 파이썬/파이토치 버전, CUDA 사용 가능 여부, 장치, 모델 로드 여부를 반환합니다.
    """
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
    """
    경로를 URL path로 넘겨 수식 개수만 빠르게 계산합니다.
    - 파일이 없으면 404를 반환합니다.
    - 그 외 예외는 500으로 감싸서 반환합니다.
    """
    try:
        return count_equations_only(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/count")
async def count_post(req: MathRequest):
    """
    JSON 바디 {"path": "..."}로 수식 개수만 빠르게 계산합니다.
    - 파일이 없으면 404를 반환합니다.
    - 그 외 예외는 500으로 감싸서 반환합니다.
    """
    try:
        return count_equations_only(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.get("/math/{file_path:path}")
async def math_get(file_path: str):
    """
    경로를 URL path로 넘겨 전체 파이프라인을 실행합니다.
    - 파일이 없으면 404를 반환합니다.
    - 그 외 예외는 500으로 감싸서 반환합니다.
    """
    try:
        return run_pipeline(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/math")
async def math_post(req: MathRequest):
    """
    JSON 바디 {"path": "..."}로 전체 파이프라인을 실행합니다.
    - 파일이 없으면 404를 반환합니다.
    - 그 외 예외는 500으로 감싸서 반환합니다.
    """
    try:
        return run_pipeline(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# 직접 실행 진입점 (python app.py)
if __name__ == "__main__":
    import uvicorn
    # --reload는 개발 편의용(코드 변경 시 자동 재시작). 배포 시에는 제거하세요.
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
