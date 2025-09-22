# 스펙(JSON 유사 딕셔너리) 리스트를 받아 각 항목을 PNG로 렌더링
import os
from copy import deepcopy
from registry import get as gram_get
from switch import make_opts, resolve_label, merge_caption
import importlib, pkgutil, time, json, sys
from pathlib import Path
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# GPU/CPU 디바이스 설정 (GPU 메모리 절약을 위해 CPU 강제 사용)
DEVICE = "cpu"
GPU_AVAILABLE = False

# matplotlib 설정
mpl.rcParams["savefig.dpi"] = 220
mpl.rcParams["figure.dpi"]  = 220
mpl.rcParams["axes.unicode_minus"] = False

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
    present = []
    for name in candidates:
        if not name:
            continue
        try:
            path = font_manager.findfont(
                FontProperties(family=name),
                fallback_to_default=False,
            )
        except Exception:
            path = ""
        if path and os.path.exists(path):
            try:
                fam = FontProperties(fname=path).get_name()
            except Exception:
                fam = name
            present.append(fam)
    return present

def _setup_matplotlib_fonts():
    here = Path(__file__).parent
    font_file_candidates = [
        os.getenv("FONT_KR_PATH"),
        str(here / "fonts" / "NotoSansKR-Regular.otf"),
        str(here / "fonts" / "NanumGothic.ttf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:\\Windows\\Fonts\\malgun.ttf",
    ]
    for p in filter(None, font_file_candidates):
        try:
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
                break
        except Exception:
            pass

    kr_candidates = (
        [os.getenv("FONT_KR_FAMILY")] if os.getenv("FONT_KR_FAMILY") else []
    ) + [
        "Noto Sans KR", "Noto Sans CJK KR", "Apple SD Gothic Neo",
        "Malgun Gothic", "NanumGothic", "Source Han Sans K", "Source Han Sans KR",
    ]
    sym_candidates = ["Noto Sans Symbols 2", "Segoe UI Symbol", "DejaVu Sans"]

    kr_present  = _present_families(kr_candidates)
    sym_present = _present_families(sym_candidates)

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
    viz_dir = str(Path(__file__).parent.resolve())
    if viz_dir not in sys.path:
        sys.path.insert(0, viz_dir)
    for m in pkgutil.iter_modules([str(pkg_dir)]):
        importlib.import_module(f"templates.grammars.{m.name}")
    _GRAMMARS_LOADED = True

def _localize_value(v, opts):
    if isinstance(v, dict) and (("ko" in v) or ("en" in v)):
        return resolve_label(v, opts)
    if isinstance(v, dict):
        return {k: _localize_value(sub, opts) for k, sub in v.items()}
    if isinstance(v, list):
        return [_localize_value(x, opts) for x in v]
    return v

def _inject_labels_into_inputs(item, inputs, opts):
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

def render_from_spec(spec_list, outdir, target_lang: str = "ko", bilingual: str = "missing", clear_outdir: bool = True):
    _ensure_grammars_loaded()
    _setup_matplotlib_fonts()
    _prepare_outdir(outdir, clear=clear_outdir)
    opts = make_opts(target_lang=target_lang, bilingual=bilingual)
    os.makedirs(outdir, exist_ok=True)

    outputs = []
    for item in spec_list:
        if not isinstance(item, dict) or "type" not in item:
            continue
        g = gram_get(item["type"])
        raw_inputs = deepcopy(item.get("inputs", {}))
        raw_inputs = _inject_labels_into_inputs(item, raw_inputs, opts)
        inputs = _localize_inputs(raw_inputs, opts)
        inputs = _mtexify_value(inputs)

        for need in getattr(g, "needs", []):
            if need not in inputs:
                inputs[need] = "__MISSING__"

        out_path = os.path.join(outdir, f"{item['id']}.png")
        tmp_path = os.path.join(outdir, f".~{item['id']}.png")
        g.renderer(inputs, tmp_path)
        os.replace(tmp_path, out_path)
        now = time.time()
        os.utime(out_path, (now, now))
        outputs.append({"id": item["id"], "type": item["type"], "path": out_path})
    return outputs

# easy_results.json 파서/실행

def _iter_easy_paragraphs(easy_json_path: str):
    """
    easy_results.json에서 (paper_id, section_id, paragraph_id, text) 순회.
    """
    data = json.loads(Path(easy_json_path).read_text(encoding="utf-8"))
    paper_id = ((data.get("paper_info") or {}).get("paper_id")) or "paper"
    for sec in data.get("easy_sections", []) or []:
        sec_id = sec.get("easy_section_id") or "section"
        for p in (sec.get("easy_paragraphs") or []):
            pid = p.get("easy_paragraph_id") or "p"
            txt = p.get("easy_paragraph_text") or ""
            yield paper_id, sec_id, pid, txt
        for sub in (sec.get("easy_subsections") or []):
            sub_id = sub.get("easy_section_id") or sec_id
            for p in (sub.get("easy_paragraphs") or []):
                pid = p.get("easy_paragraph_id") or "p"
                txt = p.get("easy_paragraph_text") or ""
                yield paper_id, sub_id, pid, txt


def sanitize_and_shorten_spec(spec_list, paragraph_id: str) -> list[dict]:
            """None/결측 스펙 제거 + 짧은 id로 재지정"""
            clean = []
            for i, s in enumerate(spec_list or []):
                if not s or not isinstance(s, dict):  # None 방지
                    continue
                if not s.get("type"):                # type 없으면 스킵
                    continue
                # 짧은 파일명: 02__bar_group
                short_id = f"{i:02d}__{s['type']}"
                s2 = deepcopy(s)
                s2["id"] = short_id
                clean.append(s2)
            return clean

# FastAPI 앱 생성
app = FastAPI(title="POLO Viz Service", version="1.1.0")

# 요청/응답 모델 (기존 엔드포인트 유지)
class VizRequest(BaseModel):
    paper_id: str
    index: int
    rewritten_text: Optional[str] = None
    target_lang: str = "ko"
    bilingual: str = "missing"

class VizResponse(BaseModel):
    paper_id: str
    index: int
    paragraph_id: str  
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

# 렌더 엔드포인트
@app.post("/viz", response_model=VizResponse)
async def generate_viz(request: VizRequest):
    """
    (JSON 전용) easy_results.json을 읽어서 index번째 문단을 시각화.
    - 결과 파일명: <easy_paragraph_id>__<spec-id>.png
    - 출력 경로: server/data/viz/<paper_id>/<section_id>/<paragraph_id>/
    """
    try:
        _ensure_grammars_loaded()
        from text_to_spec import auto_build_spec_from_text

        # easy_results.json 경로: app.py와 같은 폴더
        here = Path(__file__).parent
        easy_path = here / "easy_results.json"
        if not easy_path.exists():
            raise HTTPException(status_code=400, detail=f"입력 JSON이 없습니다: {easy_path}")

        # 문단 평탄화 리스트 구성
        paras = list(_iter_easy_paragraphs(str(easy_path)))
        if not paras:
            raise HTTPException(status_code=400, detail="easy_results.json에 문단이 없습니다.")
        
        # 인덱스 검사: -1(전체 처리) 허용하도록 수정
        if request.index != -1 and not (0 <= request.index < len(paras)):
            raise HTTPException(status_code=400, detail=f"index 범위를 벗어났습니다(0~{len(paras)-1} 또는 -1).")
        
        # index = -1 이면 전체 처리
        if request.index == -1:
            all_outs = []
            for paper_id, section_id, paragraph_id, text in paras:
                if not text or not text.strip():
                    continue
                spec = auto_build_spec_from_text(text)
                spec = sanitize_and_shorten_spec(spec, paragraph_id)
                if not spec:
                    continue
                outdir = (Path(__file__).resolve().parent.parent / "server" /
                        "data" / "viz" / paper_id / section_id / paragraph_id)
                outdir.mkdir(parents=True, exist_ok=True)
                outs = render_from_spec(
                    spec, str(outdir),
                    target_lang=request.target_lang,
                    bilingual=request.bilingual,
                    clear_outdir=True
                )
                all_outs.extend(outs)

            if not all_outs:
                raise HTTPException(status_code=500, detail="생성된 이미지가 없습니다.")

            # 대표 1장만 응답(파이프라인 호환). 필요하면 리스트 반환으로 바꿔도 됨.
            image_path = all_outs[0]["path"] + f"?rev={time.time_ns()}"
            return VizResponse(
                paper_id=paras[0][0],
                index=request.index,
                paragraph_id=paras[0][2],
                image_path=image_path,
                success=True
            )

        # 대상 문단 선택
        paper_id, section_id, paragraph_id, text = paras[request.index]
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"선택한 문단({paragraph_id})에 텍스트가 없습니다.")

        # 스펙 생성
        spec = auto_build_spec_from_text(text)
        spec = sanitize_and_shorten_spec(spec, paragraph_id)
        if not spec:
            raise HTTPException(status_code=500, detail="이미지 생성 실패 (스펙 없음)")

        # 출력 디렉토리 (문단별)
        server_dir = Path(__file__).resolve().parent.parent / "server"  # polo-system/server
        outdir = server_dir / "data" / "viz" / paper_id / section_id / paragraph_id
        outdir.mkdir(parents=True, exist_ok=True)

        # 렌더
        outputs = render_from_spec(
            spec,
            str(outdir),
            target_lang=request.target_lang,
            bilingual=request.bilingual,
            clear_outdir=True  # 문단 폴더는 덮어써도 안전
        )

        if not outputs:
            raise HTTPException(status_code=500, detail="이미지 생성 실패 (스펙 없음)")

        # 첫 이미지 반환 (여러 개면 프론트에서 리스트화를 원할 수도 있음)
        image_path = outputs[0]["path"]
        rev = str(time.time_ns())  # 캐시 우회
        image_path_versioned = f"{image_path}?rev={rev}"

        return VizResponse(
            paper_id=paper_id,
            index=request.index,
            paragraph_id=paragraph_id,
            image_path=image_path_versioned,
            success=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/viz 실행 실패: {str(e)}")

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
    port = int(os.getenv("VIZ_PORT", "5005"))
    print("🎨 POLO Viz Service 시작")
    print(f"🔧 디바이스: {DEVICE} (CPU 모드)")
    print(f"📊 포트: {port}")

    # 부트 시 easy_results.json 자동 테스트 (있을 때만)
    try:
        _ensure_grammars_loaded()
        from text_to_spec import auto_build_spec_from_text

        here = Path(__file__).parent
        easy_path = here / "easy_results.json"
        if easy_path.exists():
            print(f"easy_results.json 발견 → 문단별 테스트 렌더링 실행...")
            # 서버 경로
            server_dir = Path(__file__).resolve().parent.parent / "server"
            cnt = 0
            for paper_id, section_id, paragraph_id, text in _iter_easy_paragraphs(str(easy_path)):
                if not text.strip():
                    print(f"[SKIP] {section_id}/{paragraph_id} (empty)")
                    continue
                try:
                    spec = auto_build_spec_from_text(text)
                    print(f"[DEBUG] {section_id}/{paragraph_id} text_len={len(text)} raw_spec={len(spec or [])}")
                    spec = sanitize_and_shorten_spec(spec, paragraph_id)
                    print(f"[DEBUG] {section_id}/{paragraph_id} clean_spec={len(spec)}")
                    if not spec:
                        print(f"[SKIP] {section_id}/{paragraph_id} (no spec after sanitize)")
                        continue

                    spec = auto_build_spec_from_text(text)
                    spec = sanitize_and_shorten_spec(spec, paragraph_id)
                    outdir = server_dir / "data" / "viz" / paper_id / section_id / paragraph_id
                    outdir.mkdir(parents=True, exist_ok=True)
                    outs = render_from_spec(spec, str(outdir), target_lang="ko", bilingual="missing", clear_outdir=True)
                    for o in outs:
                        print(f" [{section_id}/{paragraph_id}] → {o['path']}")
                        cnt += 1

                except Exception as e:
                    import traceback
                    print(f"[ERR] {section_id}/{paragraph_id}: {e}")
                    traceback.print_exc()
            print(f"✅ 테스트 렌더 완료: {cnt} 개")
        else:
            print("ℹ️ easy_results.json 없음, API 서버만 실행")
    except Exception as e:
        print(f"⚠️ 부트 테스트 실패: {e} (서버는 계속 실행)")

    print("🚀 Viz API 서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=port)
