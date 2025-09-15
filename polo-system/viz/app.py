# -*- coding: utf-8 -*-
# 스펙(JSON 유사 딕셔너리) 리스트를 받아 각 항목을 PNG로 렌더링
import os
from copy import deepcopy
from registry import get as gram_get
from switch import make_opts, resolve_label, merge_caption
import importlib, pkgutil
from pathlib import Path
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties
from pathlib import Path
import matplotlib as mpl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import torch

# GPU/CPU 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_AVAILABLE = torch.cuda.is_available()

# matplotlib 설정
mpl.rcParams["savefig.dpi"] = 220
mpl.rcParams["figure.dpi"]  = 220
mpl.rcParams["axes.unicode_minus"] = False

# GPU 가속 설정 (가능한 경우)
if GPU_AVAILABLE:
    try:
        # GPU 백엔드 시도 (cudf, cupy 등이 설치된 경우)
        import matplotlib.pyplot as plt
        # GPU 메모리 최적화
        torch.cuda.empty_cache()
        print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"🔧 Viz 디바이스: {DEVICE}")
    except Exception as e:
        print(f"⚠️ GPU 백엔드 설정 실패, CPU 사용: {e}")
        DEVICE = "cpu"
        GPU_AVAILABLE = False
else:
    print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
    print(f"🔧 Viz 디바이스: {DEVICE}")

_MT_MAP = {
    "≈": r"$\approx$", "×": r"$\times$", "∈": r"$\in$",
    "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$", "−": "-"
}
def _mt(s):
    if not isinstance(s, str):
        return s
    out = s
    for k, v in _MT_MAP.items():
        out = out.replace(k, v)
    return out

def _mtexify_value(v):
    if isinstance(v, str):
        return _mt(v)
    if isinstance(v, dict):
        return {k: _mtexify_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_mtexify_value(x) for x in v]
    return v

def _present_families(candidates):
    """설치돼 있는 폰트만 필터링해서 family 이름 리스트로 반환."""
    present = []
    for name in candidates:
        if not name:
            continue
        try:
            path = font_manager.findfont(
                FontProperties(family=name),
                fallback_to_default=False,  # ← 없으면 실패하게
            )
        except Exception:
            path = ""
        if path and os.path.exists(path):
            # 시스템폰트면 addfont 없어도 되지만, 경로로 family 이름을 정확히 가져오자
            try:
                fam = FontProperties(fname=path).get_name()
            except Exception:
                fam = name
            present.append(fam)
    return present

def _setup_matplotlib_fonts():
    # (1) 한글 본문 후보 (환경변수 우선)
    kr_candidates = (
        [os.getenv("FONT_KR_FAMILY")] if os.getenv("FONT_KR_FAMILY") else []
    ) + [
        "Noto Sans KR", "Noto Sans CJK KR", "Apple SD Gothic Neo",
        "Malgun Gothic", "NanumGothic", "Source Han Sans K", "Source Han Sans KR",
    ]

    # (2) 기호 폴백 후보 (Windows에 흔한 'Segoe UI Symbol'도 포함)
    sym_candidates = ["Noto Sans Symbols 2", "Segoe UI Symbol", "DejaVu Sans"]

    kr_present  = _present_families(kr_candidates)
    sym_present = _present_families(sym_candidates)

    # 최종 우선순위: [한글 본문 1개] + [존재하는 기호 폴백들] (최소 DejaVu Sans 보장)
    family = []
    if kr_present:
        family.append(kr_present[0])
    family += (sym_present or ["DejaVu Sans"])

    rcParams["font.family"] = family
    rcParams["font.sans-serif"] = family
    rcParams["axes.unicode_minus"] = False
    # 수학기호는 mathtext로 렌더 → DejaVu Sans 기반이라 기호가 안전함
    rcParams["mathtext.fontset"] = "dejavusans"

def _prepare_outdir(outdir, clear=False, patterns=("*.png", "*.json")):
    os.makedirs(outdir, exist_ok=True)
    if not clear:
        return
    p = Path(outdir)
    for pat in patterns:
        for f in p.glob(pat):
            try:
                f.unlink()
            except Exception:
                pass


_GRAMMARS_LOADED = False
def _ensure_grammars_loaded():
    global _GRAMMARS_LOADED
    if _GRAMMARS_LOADED:
        return
    pkg_dir = Path(__file__).parent / "templates" / "grammars"

    # 안전: 현재 폴더(viz)를 sys.path에 보장
    import sys
    viz_dir = str(Path(__file__).parent.resolve())
    if viz_dir not in sys.path:
        sys.path.insert(0, viz_dir)

    for m in pkgutil.iter_modules([str(pkg_dir)]):
        importlib.import_module(f"templates.grammars.{m.name}")
    _GRAMMARS_LOADED = True

def _localize_value(v, opts):
    # {'ko':..., 'en':...} 구조는 문자열로 변환, 나머지는 재귀
    if isinstance(v, dict) and (("ko" in v) or ("en" in v)):
        return resolve_label(v, opts)
    if isinstance(v, dict):
        return {k: _localize_value(sub, opts) for k, sub in v.items()}
    if isinstance(v, list):
        return [_localize_value(x, opts) for x in v]
    return v

def _inject_labels_into_inputs(item, inputs, opts):
    # item.labels / item.caption_labels → inputs.title/label/caption에 주입
    if "labels" in item:
        resolved = resolve_label(item["labels"] or {}, opts)
        placed = False
        for k in ["title","label","name","text"]:
            if not str(inputs.get(k, "")).strip():
                inputs[k] = resolved; placed = True; break
        if not placed:
            inputs["title"] = resolved
    if "caption_labels" in item:
        cap = merge_caption(item["caption_labels"] or {}, opts)
        if not str(inputs.get("caption","")).strip():
            inputs["caption"] = cap
    return inputs

def _localize_inputs(inputs, opts):
    return _localize_value(inputs, opts)

# 랜더링 할때 결과 출력물 세팅
def render_from_spec(spec_list, outdir, target_lang: str = "ko", bilingual: str = "missing", clear_outdir: bool = True):
    """
    spec_list: [{ id, type, labels?, caption_labels?, inputs: {...} }, ...]
    """
    _ensure_grammars_loaded() # 시각화 기법 로드
    _setup_matplotlib_fonts() # 폰트 한글화
    _prepare_outdir(outdir, clear=clear_outdir) # 이전 출력물 제거
    opts = make_opts(target_lang=target_lang, bilingual=bilingual)
    os.makedirs(outdir, exist_ok=True)

    outputs = []
    for item in spec_list:
        g = gram_get(item["type"])                         # 문법 조회 (기존 그대로)
        raw_inputs = deepcopy(item.get("inputs", {}))      # 원본 보존
        raw_inputs = _inject_labels_into_inputs(item, raw_inputs, opts)
        inputs = _localize_inputs(raw_inputs, opts)        # ko/en 딕셔너리를 문자열로 변환
        inputs = _mtexify_value(inputs)   

        # 필수 입력 자동 채움(기존 로직 유지)
        for need in getattr(g, "needs", []):
            if need not in inputs:
                inputs[need] = "__MISSING__"

        out_path = os.path.join(outdir, f"{item['id']}_{item['type']}.png")
        g.renderer(inputs, out_path)                       # 실제 렌더 (기존 그대로)
        outputs.append({"id": item["id"], "type": item["type"], "path": out_path})
    return outputs

# FastAPI 앱 생성
app = FastAPI(title="POLO Viz Service", version="1.0.0")

# 요청/응답 모델
class VizRequest(BaseModel):
    paper_id: str
    index: int
    rewritten_text: str
    target_lang: str = "ko"
    bilingual: str = "missing"

class VizResponse(BaseModel):
    paper_id: str
    index: int
    image_path: str
    success: bool

@app.get("/health")
async def health():
    return {"status": "ok", "service": "viz"}

@app.post("/viz", response_model=VizResponse)
async def generate_viz(request: VizRequest):
    """
    텍스트를 받아서 시각화 이미지를 생성합니다.
    """
    try:
        # 그래머 로드 보장
        _ensure_grammars_loaded()
        
        # 텍스트에서 스펙 자동 생성
        from text_to_spec import auto_build_spec_from_text
        spec = auto_build_spec_from_text(request.rewritten_text)
        
        # 출력 디렉토리 설정
        # 절대 경로로 viz 디렉토리 설정
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent / "server"  # polo-system/server
        outdir = server_dir / "data" / "viz" / request.paper_id
        outdir.mkdir(parents=True, exist_ok=True)
        
        # 렌더링 실행
        outputs = render_from_spec(
            spec, 
            str(outdir), 
            target_lang=request.target_lang, 
            bilingual=request.bilingual,
            clear_outdir=False
        )
        
        # 첫 번째 이미지 경로 반환 (여러 개 생성될 수 있음)
        image_path = outputs[0]["path"] if outputs else None
        
        if not image_path:
            raise HTTPException(status_code=500, detail="이미지 생성 실패")
        
        return VizResponse(
            paper_id=request.paper_id,
            index=request.index,
            image_path=image_path,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시각화 생성 실패: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.getenv("VIZ_PORT", "5005"))
    
    # 디바이스 상태 출력
    print("🎨 POLO Viz Service 시작")
    if GPU_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU 사용 가능: {gpu_name}")
        print(f"🔧 디바이스: {DEVICE} (GPU 가속 시각화)")
    else:
        print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        print(f"🔧 디바이스: {DEVICE} (CPU 시각화)")
    print(f"📊 포트: {port}")
    
    # 논문 텍스트 → 스펙 자동 생성 → 렌더 (개발용)
    try:
        from pathlib import Path
        from text_to_spec import auto_build_spec_from_text

        _ensure_grammars_loaded()  # 그래머 로드 보장

        root = Path(__file__).parent
        text_path = root / "paper.txt"   # 같은 폴더의 논문 텍스트
        if text_path.exists():
            print("📄 개발용 paper.txt 발견, 테스트 렌더링 실행...")
            text = text_path.read_text(encoding="utf-8")
            spec = auto_build_spec_from_text(text)       # glossary_hybrid.json 자동 탐색

            outdir = root / "charts"
            outs = render_from_spec(spec, str(outdir), target_lang="ko", bilingual="missing")
            for o in outs:
                print(f"✅ 생성됨: {o['path']}")
        else:
            print("ℹ️ paper.txt 없음, API 서버만 실행")
    except Exception as e:
        print(f"⚠️ 테스트 렌더링 실패: {e}")
        print("ℹ️ API 서버는 정상 실행됩니다.")
    
    # FastAPI 서버 실행
    print("🚀 Viz API 서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=port)

