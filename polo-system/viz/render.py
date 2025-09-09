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
mpl.rcParams["savefig.dpi"] = 220
mpl.rcParams["figure.dpi"]  = 220
mpl.rcParams["axes.unicode_minus"] = False

def _setup_matplotlib_fonts():
    candidates = ([os.getenv("FONT_KR_FAMILY")] if os.getenv("FONT_KR_FAMILY") else []) + [
        "Noto Sans KR","Noto Sans CJK KR","Apple SD Gothic Neo",
        "Malgun Gothic","NanumGothic","Source Han Sans K","Source Han Sans KR"
    ]
    chosen = None
    for name in candidates:
        if not name: continue
        try:
            path = font_manager.findfont(FontProperties(family=name), fallback_to_default=False)
        except Exception:
            path = ""
        if path and os.path.exists(path):
            font_manager.fontManager.addfont(path)
            chosen = FontProperties(fname=path).get_name()
            break
    if chosen:
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
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

        # 필수 입력 자동 채움(기존 로직 유지)
        for need in getattr(g, "needs", []):
            if need not in inputs:
                inputs[need] = "__MISSING__"

        out_path = os.path.join(outdir, f"{item['id']}_{item['type']}.png")
        g.renderer(inputs, out_path)                       # 실제 렌더 (기존 그대로)
        outputs.append({"id": item["id"], "type": item["type"], "path": out_path})
    return outputs

if __name__ == "__main__":
    # 논문 텍스트 → 스펙 자동 생성 → 렌더
    from pathlib import Path
    from text_to_spec import auto_build_spec_from_text, ensure_minimum_charts

    _ensure_grammars_loaded()  # 그래머 로드 보장

    root = Path(__file__).parent
    text_path = root / "paper.txt"   # 같은 폴더의 논문 텍스트
    if not text_path.exists():
        raise FileNotFoundError("paper.txt 를 같은 폴더에 두세요.")

    text = text_path.read_text(encoding="utf-8")
    spec = auto_build_spec_from_text(text)       # glossary_hybrid.json 자동 탐색
    spec = ensure_minimum_charts(spec)

    outdir = root / "charts"
    outs = render_from_spec(spec, str(outdir), target_lang="ko", bilingual="missing")
    for o in outs:
        print(o["path"])

