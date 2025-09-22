# 스펙(JSON 유사 딕셔너리) 리스트를 받아 각 항목을 PNG로 렌더링
import os
from copy import deepcopy
from registry import get as gram_get
from switch import make_opts, resolve_label, merge_caption
import importlib, pkgutil, time
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

# GPU/CPU 디바이스 설정 (GPU 메모리 절약을 위해 CPU 강제 사용)
DEVICE = "cpu"  # GPU 메모리 절약을 위해 CPU 강제 사용
GPU_AVAILABLE = False  # GPU 사용 안함


# matplotlib 설정
mpl.rcParams["savefig.dpi"] = 220
mpl.rcParams["figure.dpi"]  = 220
mpl.rcParams["axes.unicode_minus"] = False

# CPU 모드 강제 설정 (GPU 메모리 절약)
print("🔧 Viz 서비스: CPU 모드로 실행 (GPU 메모리 절약)")
print(f"🔧 디바이스: {DEVICE}")

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
    """
    한글 폰트를 '설치되어 있든/없든' 최대한 자동으로 잡아준다.
    1) 환경변수 FONT_KR_PATH, 2) ./fonts/*, 3) OS 공용 경로 순서로 폰트 파일을 찾아
    font_manager.addfont 로 런타임 등록 후 family 우선순위를 세팅한다.
    """
    # 0) 런타임 등록 후보 경로 (존재하는 첫 파일 1개면 충분)
    here = Path(__file__).parent
    font_file_candidates = [
        os.getenv("FONT_KR_PATH"),
        str(here / "fonts" / "NotoSansKR-Regular.otf"),
        str(here / "fonts" / "NanumGothic.ttf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "C:\\Windows\\Fonts\\malgun.ttf",              # Windows
    ]
    for p in filter(None, font_file_candidates):
        try:
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)  # ← 런타임 등록 (없던 폰트도 사용 가능)
                break
        except Exception:
            pass

    # 1) 한글 본문 후보 (환경변수로 family 강제 시 최우선)
    kr_candidates = (
        [os.getenv("FONT_KR_FAMILY")] if os.getenv("FONT_KR_FAMILY") else []
    ) + [
        "Noto Sans KR", "Noto Sans CJK KR", "Apple SD Gothic Neo",
        "Malgun Gothic", "NanumGothic", "Source Han Sans K", "Source Han Sans KR",
    ]

    # 2) 기호 폴백
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
    _ensure_grammars_loaded() # 시각화 기법 로드
    _setup_matplotlib_fonts() # 폰트 한글화
    _prepare_outdir(outdir, clear=clear_outdir) # 이전 출력물 제거
    opts = make_opts(target_lang=target_lang, bilingual=bilingual)
    os.makedirs(outdir, exist_ok=True)

    outputs = []
    for item in spec_list:
        g = gram_get(item["type"])                         # 문법 조회
        raw_inputs = deepcopy(item.get("inputs", {}))      # 원본 보존
        raw_inputs = _inject_labels_into_inputs(item, raw_inputs, opts)
        inputs = _localize_inputs(raw_inputs, opts)        # ko/en 딕셔너리를 문자열로 변환
        inputs = _mtexify_value(inputs)   

        # 필수 입력 자동 채움(기존 로직 유지)
        for need in getattr(g, "needs", []):
            if need not in inputs:
                inputs[need] = "__MISSING__"

        # 항상 같은 파일명으로 저장 (브라우저 캐시 우회는 응답에서 처리)
        out_path = os.path.join(outdir, f"{item['id']}_{item['type']}.png")

        # 원자적 교체: 임시 파일로 먼저 저장 후 교체(부분 쓰기 노출 방지)
        tmp_path = os.path.join(outdir, f".~{item['id']}_{item['type']}.png")
        g.renderer(inputs, tmp_path)
        os.replace(tmp_path, out_path)  # 같은 이름으로 교체

        now = time.time()
        os.utime(out_path, (now, now))

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

class GenerateVisualizationsRequest(BaseModel):
    paper_id: str
    easy_results: Dict[str, Any]
    output_dir: str

class GenerateVisualizationsResponse(BaseModel):
    paper_id: str
    viz_results: List[Dict[str, Any]]
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
            clear_outdir=True
        )
        
        # 첫 번째 이미지 경로 반환 (여러 개 생성될 수 있음)
        image_path = outputs[0]["path"] if outputs else None
        
        if not image_path:
            raise HTTPException(status_code=500, detail="이미지 생성 실패")
        
        rev = str(time.time_ns())  # 항상 새로운 값 → 브라우저/뷰어 캐시 완전 우회
        image_path_versioned = f"{image_path}?rev={rev}"
        
        return VizResponse(
            paper_id=request.paper_id,
            index=request.index,
            image_path=image_path_versioned,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시각화 생성 실패: {str(e)}")

@app.post("/generate-visualizations", response_model=GenerateVisualizationsResponse)
async def generate_visualizations(request: GenerateVisualizationsRequest):
    """
    Easy 결과를 받아서 시각화 이미지들을 생성합니다.
    """
    try:
        # 그래머 로드 보장
        _ensure_grammars_loaded()
        
        paper_id = request.paper_id
        easy_results = request.easy_results
        output_dir = request.output_dir
        
        print(f"🎨 [VIZ] 시각화 생성 시작: paper_id={paper_id}")
        
        # Easy 결과에서 섹션들 추출
        easy_sections = easy_results.get("easy_sections", [])
        if not easy_sections:
            print(f"⚠️ [VIZ] Easy 섹션이 없습니다")
            return GenerateVisualizationsResponse(
                paper_id=paper_id,
                viz_results=[],
                success=True
            )
        
        viz_results = []
        
        # 각 섹션에 대해 시각화 생성
        for i, section in enumerate(easy_sections):
            try:
                # 섹션의 텍스트 추출
                section_text = section.get("easy_text", "")
                if not section_text:
                    print(f"⚠️ [VIZ] 섹션 {i+1}에 텍스트가 없습니다")
                    continue
                
                print(f"🎨 [VIZ] 섹션 {i+1} 시각화 생성 중...")
                
                # 텍스트에서 스펙 자동 생성
                from text_to_spec import auto_build_spec_from_text
                spec = auto_build_spec_from_text(section_text)
                
                if not spec:
                    print(f"⚠️ [VIZ] 섹션 {i+1}에서 스펙을 생성할 수 없습니다")
                    continue
                
                # 출력 디렉토리 설정
                section_outdir = Path(output_dir) / f"section_{i+1}"
                section_outdir.mkdir(parents=True, exist_ok=True)
                
                # 렌더링 실행
                outputs = render_from_spec(
                    spec, 
                    str(section_outdir), 
                    target_lang="ko", 
                    bilingual="missing",
                    clear_outdir=True
                )
                
                # 결과 저장
                for j, output in enumerate(outputs):
                    viz_result = {
                        "section_index": i + 1,
                        "image_index": j + 1,
                        "image_path": output["path"],
                        "image_type": output["type"],
                        "section_id": section.get("easy_section_id", f"section_{i+1}"),
                        "section_title": section.get("easy_section_title", f"섹션 {i+1}")
                    }
                    viz_results.append(viz_result)
                    print(f"✅ [VIZ] 섹션 {i+1} 이미지 {j+1} 생성 완료: {output['path']}")
                
            except Exception as e:
                print(f"❌ [VIZ] 섹션 {i+1} 시각화 생성 실패: {e}")
                continue
        
        print(f"✅ [VIZ] 시각화 생성 완료: {len(viz_results)}개 이미지")
        
        return GenerateVisualizationsResponse(
            paper_id=paper_id,
            viz_results=viz_results,
            success=True
        )
        
    except Exception as e:
        print(f"❌ [VIZ] 시각화 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시각화 생성 실패: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.getenv("VIZ_PORT", "5005"))
    
    # 디바이스 상태 출력
    print("🎨 POLO Viz Service 시작")
    print(f"🔧 디바이스: {DEVICE} (CPU 모드 - GPU 메모리 절약)")
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

