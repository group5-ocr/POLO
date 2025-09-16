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
def _build_easy_prompt(text: str, section_title: str | None = None) -> str:
    title_line = f"[섹션] {section_title}\n\n" if section_title else ""
    return (
        title_line +
        "아래 학술 텍스트를 '쉽게말해,'로 시작하여 고등학생도 이해할 수 있게 한국어로 재서술하라.\n"
        "- 새로운 정보 추가 금지, 원문의 의미를 정확히 보존\n"
        "- 문장을 짧고 명확하게 분할, 문단을 논리적으로 구분\n"
        "- 존댓말로 서술(입니다/합니다/한다), ~요 금지\n"
        "- 수식/기호/그림 내용은 설명만 하고 텍스트에 재삽입하지 말 것(수식은 별도로 복원됨)\n"
        "- 목록이 자연스러우면 간단한 불릿을 사용\n"
        "- 라텍스 명령어/표/그림 코드는 생성하지 말고 순수 텍스트로만 작성\n\n"
        "출력 형식:\n"
        "- 1~3개의 짧은 문단으로 나눠 작성\n"
        "- 첫 문장은 반드시 '쉽게말해,'로 시작\n\n"
        f"[원문]\n{text}\n\n[출력]\n"
    )

def _build_verify_prompt(first_pass_text: str, section_title: str | None = None) -> str:
    title_line = f"[섹션] {section_title}\n\n" if section_title else ""
    return (
        title_line +
        "너는 고등학생 독자이자 Llama LLM의 검토자이다. 아래 재서술 결과를 읽고 가독성을 높여라.\n"
        "- 의미 왜곡, 정보 추가 금지\n"
        "- 문단이 길면 2~3문단으로 분할\n"
        "- 반복/군더더기 제거, 문장 간 연결을 자연스럽게 조정\n"
        "- 존댓말(입니다/합니다/한다) 유지, ~요 금지\n"
        "- 라텍스 코드는 생성하지 말 것, 순수 텍스트만 출력\n"
        "- 첫 문장은 가능하면 '쉽게말해,'로 시작\n\n"
        f"[검토 대상]\n{first_pass_text}\n\n[개선된 출력]\n"
    )

def _extract_math_placeholders(text: str):
    """수식을 플레이스홀더로 치환하여 반환합니다.
    반환: (치환된_텍스트, inline_map, block_map)
    """
    import re

    # 디스플레이 수식 (우선 처리)
    block_map = {}
    block_idx = 0

    def _sub_block_dollar(m):
        nonlocal block_idx
        key = f"[MATH_BLOCK_{block_idx}]"
        block_map[key] = m.group(0)
        block_idx += 1
        return key

    text = re.sub(r"\$\$[\s\S]*?\$\$", _sub_block_dollar)

    def _sub_equation_env(m):
        nonlocal block_idx
        key = f"[MATH_BLOCK_{block_idx}]"
        block_map[key] = m.group(0)
        block_idx += 1
        return key

    text = re.sub(r"\\begin\{(equation\*?|align\*?|eqnarray\*?)\}[\s\S]*?\\end\{\1\}", _sub_equation_env)

    # 인라인 수식
    inline_map = {}
    inline_idx = 0

    def _sub_inline(m):
        nonlocal inline_idx
        key = f"[MATH_INLINE_{inline_idx}]"
        inline_map[key] = m.group(0)
        inline_idx += 1
        return key

    text = re.sub(r"\$(?!\$)(?:[^$\\]|\\.)+\$", _sub_inline)

    return text, inline_map, block_map


def _clean_latex_text(text: str) -> str:
    """LLM 입력용으로 LaTeX 노이즈를 최대한 제거합니다(구조 파싱은 별도로 수행)."""
    import re

    # LRB, RRB 변환 (괄호)
    text = re.sub(r"LRB", "(", text)
    text = re.sub(r"RRB", ")", text)

    # 인용/라벨/참조 제거
    text = re.sub(r"\\cite\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\ref\{[^}]*\}", "", text)
    text = re.sub(r"\\footnote\{[\s\S]*?\}", "", text)

    # URL은 텍스트로만 남김
    text = re.sub(r"\\url\{([^}]*)\}", r"(\1)", text)

    # 그림/표 환경 제거
    text = re.sub(r"\\begin\{figure\}[\s\S]*?\\end\{figure\}", "", text)
    text = re.sub(r"\\begin\{table\}[\s\S]*?\\end\{table\}", "", text)
    text = re.sub(r"\\begin\{tabular\}[\s\S]*?\\end\{tabular\}", "", text)

    # 서식 명령 내용만 남김
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)

    # 섹션/소제목 명령은 파싱 단계에서 관리하므로 본문에서는 제거
    text = re.sub(r"^\\section\{[^}]*\}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\\subsection\{[^}]*\}\s*", "", text, flags=re.MULTILINE)

    # 공백 정리
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()

def _extract_technical_terms(text: str) -> List[str]:
    """텍스트에서 전문 용어를 추출합니다"""
    import re
    
    # 일반적인 컴퓨터 비전/딥러닝 전문 용어 패턴
    technical_patterns = [
        r'\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b',  # CNN, R-CNN, YOLO 등
        r'\b(?:fast|faster|fastest)\s+rcnn\b',  # fast rcnn
        r'\b(?:anchor|anchors)\b',  # anchor
        r'\b(?:feature|features)\b',  # feature
        r'\b(?:detection|detector)\b',  # detection
        r'\b(?:classification|classifier)\b',  # classification
        r'\b(?:backbone|neck|head)\b',  # 네트워크 구조
        r'\b(?:convolutional|conv)\b',  # convolutional
        r'\b(?:neural|network)\b',  # neural network
        r'\b(?:multi[-\s]?scale|multiscale)\b',  # multi-scale
        r'\b(?:object|objects)\b',  # object
        r'\b(?:bounding|box|boxes)\b',  # bounding box
        r'\b(?:IoU|mAP|AP)\b',  # 평가 지표
    ]
    
    terms = set()
    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        terms.update([match.lower() for match in matches])
    
    return list(terms)

def _generate_term_explanations(terms: List[str]) -> Dict[str, str]:
    """전문 용어에 대한 설명을 생성합니다"""
    explanations = {
        'cnn': '합성곱 신경망(Convolutional Neural Network): 이미지 인식에 특화된 딥러닝 모델',
        'rcnn': 'R-CNN(Region-based CNN): 객체 검출을 위한 딥러닝 모델',
        'fast rcnn': 'Fast R-CNN: R-CNN의 속도를 개선한 객체 검출 모델',
        'faster rcnn': 'Faster R-CNN: Fast R-CNN을 더욱 빠르게 만든 모델',
        'yolo': 'YOLO(You Only Look Once): 실시간 객체 검출을 위한 딥러닝 모델',
        'anchor': '앵커(Anchor): 객체 검출에서 사용하는 참조 박스',
        'feature': '특징(Feature): 이미지에서 추출한 의미 있는 정보',
        'detection': '검출(Detection): 이미지에서 객체를 찾아내는 과정',
        'classification': '분류(Classification): 객체의 종류를 구분하는 과정',
        'backbone': '백본(Backbone): 딥러닝 모델의 주요 특징 추출 부분',
        'convolutional': '합성곱(Convolutional): 이미지 처리에 사용하는 수학적 연산',
        'neural network': '신경망(Neural Network): 인간의 뇌를 모방한 인공지능 모델',
        'multi-scale': '다중 스케일(Multi-scale): 다양한 크기의 객체를 처리하는 방법',
        'object': '객체(Object): 이미지에서 인식하고자 하는 대상',
        'bounding box': '바운딩 박스(Bounding Box): 객체의 위치를 나타내는 사각형',
        'iou': 'IoU(Intersection over Union): 객체 검출 성능을 측정하는 지표',
        'map': 'mAP(mean Average Precision): 객체 검출 모델의 전체적인 성능 지표',
    }
    
    result = {}
    for term in terms:
        if term in explanations:
            result[term] = explanations[term]
        else:
            # 간단한 설명 생성
            result[term] = f"{term.upper()}: 관련 전문 용어"
    
    return result

def _parse_latex_sections(tex_path: Path) -> List[dict]:
    """LaTeX 파일을 섹션별로 파싱합니다"""
    import re
    
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    current_section = None
    current_content = []
    subsections = []  # subsection 정보 저장
    
    lines = content.split('\n')
    
    for line in lines:
        # 섹션 시작 감지
        if re.match(r'\\section\{([^}]*)\}', line):
            # 이전 섹션 저장
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip(),
                    "subsections": subsections.copy()
                })
            
            # 새 섹션 시작
            title_match = re.match(r'\\section\{([^}]*)\}', line)
            current_section = title_match.group(1) if title_match else "Unknown Section"
            current_content = [line]
            subsections = []  # 새 섹션의 subsection 리스트 초기화
            
        elif re.match(r'\\subsection\{([^}]*)\}', line):
            # subsection 정보 추출
            title_match = re.match(r'\\subsection\{([^}]*)\}', line)
            subsection_title = title_match.group(1) if title_match else "Unknown Subsection"
            subsections.append(subsection_title)
            
            # 서브섹션도 섹션으로 처리
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip(),
                    "subsections": subsections.copy()
                })
            
            current_section = subsection_title
            current_content = [line]
            
        elif re.match(r'\\begin\{abstract\}', line):
            # Abstract 섹션
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip(),
                    "subsections": subsections.copy()
                })
            
            current_section = "Abstract"
            current_content = [line]
            subsections = []
            
        elif re.match(r'\\begin\{document\}', line):
            # Document 시작
            if current_section and current_content:
                sections.append({
                    "index": len(sections),
                    "title": current_section,
                    "content": '\n'.join(current_content).strip(),
                    "subsections": subsections.copy()
                })
            
            current_section = "Introduction"
            current_content = [line]
            subsections = []
            
        else:
            # 일반 내용
            if current_section:
                current_content.append(line)
    
    # 마지막 섹션 저장
    if current_section and current_content:
        sections.append({
            "index": len(sections),
            "title": current_section,
            "content": '\n'.join(current_content).strip(),
            "subsections": subsections.copy()
        })
    
    # 빈 섹션 제거
    sections = [s for s in sections if s["content"].strip()]
    
    return sections

async def _rewrite_text(text: str) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("모델이 로드되지 않았습니다")

    # 수식 플레이스홀더 치환 → 비수식 LaTeX 정리
    text_no_math, inline_map, block_map = _extract_math_placeholders(text)
    cleaned_text = _clean_latex_text(text_no_math)
    print(f"🔍 [DEBUG] 정리된 텍스트 미리보기: {cleaned_text[:200]}...")

    # 섹션 제목 힌트가 있으면 전달(없으면 None)
    section_title = None
    prompt = _build_easy_prompt(cleaned_text, section_title)
    print(f"🔍 [DEBUG] 프롬프트 미리보기: {prompt[:300]}...")
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
    print(f"🔍 [DEBUG] 생성된 전체 텍스트: {generated[:500]}...")
    
    result = generated[len(prompt):].strip()
    print(f"🔍 [DEBUG] 1차 결과: {result[:300]}...")

    # 2단계 검증/개선 (옵션, 기본 활성화)
    use_verify = os.getenv("EASY_VERIFY", "true").lower() in ("1","true","yes")
    if use_verify:
        verify_prompt = _build_verify_prompt(result, section_title)
        v_inputs = tokenizer(verify_prompt, return_tensors="pt", truncation=True, max_length=2048)
        v_inputs = {k: v.to(device) for k, v in v_inputs.items()}
        with torch.inference_mode():
            v_out = model.generate(
                **v_inputs,
                max_new_tokens=MAX_NEW_TOKENS // 2,
                do_sample=True,
                temperature=float(os.getenv("EASY_TEMPERATURE", "0.6")),
                top_p=float(os.getenv("EASY_TOP_P", "0.9")),
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        v_text = tokenizer.decode(v_out[0], skip_special_tokens=True)
        result = v_text[len(verify_prompt):].strip() or result
        print(f"🔍 [DEBUG] 2차 검증 결과: {result[:300]}...")
    
    # 플레이스홀더 복원(블록→인라인 순서)
    for key, val in block_map.items():
        result = result.replace(key, val)
    for key, val in inline_map.items():
        result = result.replace(key, val)

    print(f"🔍 [DEBUG] 최종 결과(복원 후) 미리보기: {result[:300]}...")
    return result

def _format_latex_output(text: str) -> str:
    """생성된 텍스트를 LaTeX 형태로 정리하되 원본 구조는 최대한 보존합니다"""
    import re
    
    # 섹션 제목 정리 (마크다운 스타일이 있는 경우만)
    text = re.sub(r'^#+\s*(.+)$', r'\\section{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^##+\s*(.+)$', r'\\subsection{\1}', text, flags=re.MULTILINE)
    
    # 굵은 글씨 정리 (마크다운 스타일이 있는 경우만)
    text = re.sub(r'(?<!\$)\*\*([^$]+?)\*\*(?!\$)', r'\\textbf{\1}', text)
    
    # 기울임 글씨 정리 (마크다운 스타일이 있는 경우만)
    text = re.sub(r'(?<!\$)\*([^$*]+?)\*(?!\$)', r'\\textit{\1}', text)
    
    # 목록 정리 (마크다운 스타일이 있는 경우만)
    text = re.sub(r'^[-•]\s*(.+)$', r'\\item \1', text, flags=re.MULTILINE)
    
    # 연속된 item을 itemize로 감싸기
    lines = text.split('\n')
    result_lines = []
    in_itemize = False
    
    for line in lines:
        if line.strip().startswith('\\item'):
            if not in_itemize:
                result_lines.append('\\begin{itemize}')
                in_itemize = True
            result_lines.append(line)
        else:
            if in_itemize:
                result_lines.append('\\end{itemize}')
                in_itemize = False
            result_lines.append(line)
    
    if in_itemize:
        result_lines.append('\\end{itemize}')
    
    return '\n'.join(result_lines)

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
            # Viz 모델에서 생성된 이미지가 다른 경로에 있으면 복사
            img_path_obj = Path(img_path)
            if img_path_obj.exists() and not img_path_obj.parent.samefile(out_dir):
                # 다른 경로에 있으면 easy_outputs로 복사
                out_path = out_dir / f"{index:06d}.png"
                import shutil
                shutil.copy2(img_path_obj, out_path)
                print(f"✅ [SUCCESS] 이미지 복사: {img_path} -> {out_path}")
                return VizResult(ok=True, index=index, image_path=str(out_path))
            else:
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

    results: List[VizResult] = []

    print(f"🔍 [DEBUG] 배치 처리 시작...")
    # 순차적으로 처리 (병렬 처리로 인한 메모리 부족 방지)
    for idx, section in enumerate(sections):
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
    
    # LaTeX 결과 파일 생성
    latex_result_path = out_dir / "easy_results.tex"
    _save_latex_results(sections, results, latex_result_path)
    
    # HTML 결과 파일 생성
    html_result_path = out_dir / "easy_results.html"
    _save_html_results(sections, results, html_result_path, req.paper_id)
    
    # JSON 파일 저장
    json_file_path = out_dir / "easy_results.json"
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    
    print(f"📄 [JSON] 결과 파일 저장: {json_file_path}")
    print(f"📄 [LaTeX] 결과 파일 저장: {latex_result_path}")
    print(f"📄 [HTML] 결과 파일 저장: {html_result_path}")
    print(f"✅ [SUCCESS] Easy 모델 배치 처리 완료: {result}")
    return result

def _save_latex_results(sections: List[dict], results: List[VizResult], output_path: Path):
    """LaTeX 형태로 결과를 저장합니다"""
    latex_content = []
    latex_content.append("\\documentclass{article}")
    latex_content.append("\\usepackage[utf8]{inputenc}")
    latex_content.append("\\usepackage{korean}")
    latex_content.append("\\usepackage{graphicx}")
    latex_content.append("\\usepackage{amsmath}")
    latex_content.append("\\usepackage{amsfonts}")
    latex_content.append("\\begin{document}")
    latex_content.append("")
    
    # 섹션별 결과 추가
    for i, (section, result) in enumerate(zip(sections, results)):
        if result.ok and result.easy_text:
            # 섹션 제목
            latex_content.append(f"\\section{{{section['title']}}}")
            latex_content.append("")
            
            # 변환된 텍스트 (LaTeX 형태)
            latex_content.append(result.easy_text)
            latex_content.append("")
            
            # 이미지가 있으면 추가
            if result.image_path and Path(result.image_path).exists():
                latex_content.append("\\begin{figure}[h]")
                latex_content.append("\\centering")
                latex_content.append(f"\\includegraphics[width=0.8\\textwidth]{{{result.image_path}}}")
                latex_content.append(f"\\caption{{{section['title']} 관련 시각화}}")
                latex_content.append("\\end{figure}")
                latex_content.append("")
    
    latex_content.append("\\end{document}")
    
    # 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_content))

def _get_current_datetime() -> str:
    """현재 날짜와 시간을 문자열로 반환합니다"""
    from datetime import datetime
    return datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")

def _save_html_results(sections: List[dict], results: List[VizResult], output_path: Path, paper_id: str):
    """HTML 형태로 결과를 저장합니다"""
    html_content = []
    
    # HTML 헤더 (ArXiv 스타일)
    html_content.append("""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>POLO - 쉬운 논문 설명</title>
    <style>
        body {
            font-family: 'Times New Roman', 'Noto Serif KR', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            color: #000;
            font-size: 12pt;
        }
        .paper-container {
            background: white;
            padding: 40px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border: 1px solid #ddd;
        }
        .copyright-notice {
            background: #f0f0f0;
            border: 2px solid #333;
            padding: 15px;
            margin-bottom: 30px;
            text-align: center;
            font-weight: bold;
            font-size: 11pt;
        }
        .copyright-notice .title {
            font-size: 14pt;
            margin-bottom: 10px;
            color: #d32f2f;
        }
        .copyright-notice .content {
            font-size: 10pt;
            line-height: 1.4;
        }
        .paper-header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #000;
            padding-bottom: 20px;
        }
        .paper-title {
            font-size: 18pt;
            font-weight: bold;
            margin-bottom: 15px;
            line-height: 1.3;
        }
        .paper-subtitle {
            font-size: 14pt;
            color: #666;
            margin-bottom: 20px;
            font-style: italic;
        }
        .paper-meta {
            font-size: 10pt;
            color: #666;
            margin-bottom: 10px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            font-size: 16pt;
            font-weight: bold;
            margin: 30px 0 15px 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .section h3 {
            font-size: 14pt;
            font-weight: bold;
            margin: 20px 0 10px 0;
        }
        .subsection-title {
            font-size: 13pt;
            font-weight: bold;
            margin: 15px 0 8px 0;
            color: #333;
            border-left: 3px solid #1976d2;
            padding-left: 10px;
        }
        .subsection-section {
            margin-bottom: 25px;
            padding: 15px;
            background: #fafafa;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .content {
            font-size: 12pt;
            line-height: 1.6;
            text-align: justify;
            margin-bottom: 15px;
        }
        .content p {
            margin-bottom: 12px;
            text-indent: 1.5em;
        }
        .content strong {
            font-weight: bold;
        }
        .content em {
            font-style: italic;
        }
        .content ul, .content ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        .content li {
            margin-bottom: 6px;
        }
        .math {
            background: #f9f9f9;
            padding: 10px;
            border: 1px solid #ddd;
            margin: 15px 0;
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            text-align: center;
            overflow-x: auto;
        }
        .image-container {
            text-align: center;
            margin: 25px 0;
            padding: 15px;
            border: 1px solid #ddd;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
        }
        .image-caption {
            margin-top: 10px;
            font-style: italic;
            font-size: 10pt;
            color: #666;
        }
        .download-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #1976d2;
            color: white;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 11pt;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        .download-btn:hover {
            background: #1565c0;
            transform: translateY(-1px);
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            font-size: 10pt;
            color: #666;
            border-top: 1px solid #ddd;
        }
        .abstract {
            background: #f8f8f8;
            padding: 20px;
            border-left: 4px solid #1976d2;
            margin: 20px 0;
        }
        .abstract h3 {
            font-size: 12pt;
            font-weight: bold;
            margin: 0 0 10px 0;
            text-transform: uppercase;
        }
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            .paper-container {
                padding: 20px;
            }
            .paper-title {
                font-size: 16pt;
            }
        }
    </style>
</head>
<body>""")
    
    # 저작권 표시 및 헤더 섹션
    html_content.append(f"""
    <div class="paper-container">
        <div class="copyright-notice">
            <div class="title">⚠️ 저작권 고지</div>
            <div class="content">
                이 문서는 POLO AI 논문 이해 도우미에 의해 원본 논문을 고등학생도 이해할 수 있게 쉽게 변환한 것입니다.<br>
                원본 논문의 저작권은 원 저자에게 있으며, 이 변환된 문서는 교육 목적으로만 사용되어야 합니다.<br>
                상업적 이용이나 재배포 시에는 원본 논문의 저작권 정책을 준수해야 합니다.
            </div>
        </div>
        
        <div class="paper-header">
            <div class="paper-title">POLO 논문 이해 도우미 변환 결과</div>
            <div class="paper-subtitle">복잡한 논문을 쉽게 이해할 수 있도록 변환한 결과</div>
            <div class="paper-meta">
                논문 ID: {paper_id} | 변환 일시: {_get_current_datetime()}
            </div>
        </div>
    """)
    
    # 섹션별 결과 추가
    for i, (section, result) in enumerate(zip(sections, results)):
        if result.ok and result.easy_text:
            # Abstract 섹션은 특별한 스타일 적용
            if section['title'].lower() in ['abstract', '요약']:
                html_content.append(f"""
        <div class="abstract">
            <h3>{section['title']}</h3>
            <div class="content">
                {_latex_to_html(result.easy_text)}
            </div>
        </div>
                """)
            else:
                # subsection이 있으면 각각 분리하여 표시
                if 'subsections' in section and section['subsections']:
                    # 메인 섹션 제목
                    html_content.append(f"""
        <div class="section">
            <h2>{section['title']}</h2>
        </div>
                    """)
                    
                    # 각 subsection과 내용을 분리하여 표시
                    for sub_idx, subsection in enumerate(section['subsections']):
                        html_content.append(f"""
        <div class="subsection-section">
            <div class="subsection-title">{subsection}</div>
            <div class="content">
                {_latex_to_html(result.easy_text)}
            </div>
        </div>
                        """)
                else:
                    # subsection이 없으면 일반 섹션으로 표시
                    html_content.append(f"""
        <div class="section">
            <h2>{section['title']}</h2>
            <div class="content">
                {_latex_to_html(result.easy_text)}
            </div>
        </div>
                    """)
            
            # 이미지가 있으면 추가
            if result.image_path and Path(result.image_path).exists():
                image_name = Path(result.image_path).name
                html_content.append(f"""
        <div class="image-container">
            <img src="{image_name}" alt="{section['title']} 관련 시각화">
            <div class="image-caption">Figure {i+1}: {section['title']} 관련 시각화</div>
        </div>
                """)
    
    # 다운로드 버튼과 푸터
    html_content.append(f"""
        <button class="download-btn" onclick="downloadHTML()">📥 HTML 다운로드</button>
        
        <div class="footer">
            <p><strong>POLO AI 논문 이해 도우미</strong></p>
            <p>이 문서는 원본 논문을 고등학생도 이해할 수 있게 쉽게 변환한 것입니다</p>
            <p>변환 일시: {_get_current_datetime()} | 논문 ID: {paper_id}</p>
            <p style="font-size: 9pt; color: #999; margin-top: 20px;">
                본 문서는 교육 목적으로만 사용되어야 하며, 상업적 이용 시 원본 논문의 저작권 정책을 준수해야 합니다.
            </p>
        </div>
    </div>
        
        <script>
            function downloadHTML() {{
                const element = document.documentElement.outerHTML;
                const blob = new Blob([element], {{type: 'text/html'}});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'polo_easy_explanation_{paper_id}.html';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}
        </script>
    </body>
</html>""")
    
    # 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

def _latex_to_html(latex_text: str, technical_terms: Dict[str, str] = None) -> str:
    """LaTeX 텍스트를 HTML로 변환합니다 (ArXiv 스타일)"""
    import re
    
    # LaTeX 명령어를 HTML로 변환
    html_text = latex_text
    
    # 섹션 제목 (ArXiv 스타일로 대문자 변환)
    html_text = re.sub(r'\\section\{([^}]+)\}', r'<h2>\1</h2>', html_text)
    html_text = re.sub(r'\\subsection\{([^}]+)\}', r'<h3>\1</h3>', html_text)
    
    # 텍스트 스타일
    html_text = re.sub(r'\\textbf\{([^}]+)\}', r'<strong>\1</strong>', html_text)
    html_text = re.sub(r'\\textit\{([^}]+)\}', r'<em>\1</em>', html_text)
    
    # 수식 처리 (ArXiv 스타일)
    html_text = re.sub(r'\$([^$]+)\$', r'<span class="math">\1</span>', html_text)
    html_text = re.sub(r'\$\$([^$]+)\$\$', r'<div class="math">\1</div>', html_text)
    
    # 목록 처리
    html_text = re.sub(r'\\begin\{itemize\}', '<ul>', html_text)
    html_text = re.sub(r'\\end\{itemize\}', '</ul>', html_text)
    html_text = re.sub(r'\\begin\{enumerate\}', '<ol>', html_text)
    html_text = re.sub(r'\\end\{enumerate\}', '</ol>', html_text)
    html_text = re.sub(r'\\item\s*', '<li>', html_text)
    
    # 링크 처리 (LaTeX \href 명령어)
    html_text = re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}', r'<a href="\1" target="_blank">\2</a>', html_text)
    
    # URL 자동 링크 처리
    html_text = re.sub(r'(https?://[^\s<>"]+)', r'<a href="\1" target="_blank">\1</a>', html_text)
    
    # 문단 처리 (ArXiv 스타일 - 들여쓰기 적용)
    html_text = re.sub(r'\n\n+', '</p><p>', html_text)
    html_text = '<p>' + html_text + '</p>'
    
    # 빈 문단 제거
    html_text = re.sub(r'<p>\s*</p>', '', html_text)
    
    # 연속된 공백 정리
    html_text = re.sub(r' +', ' ', html_text)
    
    # 용어 설명 출력은 더 이상 사용하지 않음(요청에 따라 제거)
    
    return html_text

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
