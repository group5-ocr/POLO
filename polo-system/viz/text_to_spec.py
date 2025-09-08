# -*- coding: utf-8 -*-
import re, json, math, random
from pathlib import Path
from typing import List, Dict, Any
from statistics import mean
# 5개 범용 도식 트리거
from templates.generic_rules import build_concept_specs

# Glossary 로드
def _find_glossary_path():
    here = Path(__file__).parent
    for p in (here/"glossary_hybrid.json", here/"glossary.json"):
        if p.exists(): return str(p)
    return None

def load_glossary_any(path: str | None = None) -> List[Dict[str, Any]]:
    gp = path or _find_glossary_path()
    if not gp: return []
    return json.loads(Path(gp).read_text(encoding="utf-8"))


# Glossary 인덱스/패턴
def _compile_patterns(entry: Dict[str, Any]):
    pats = []
    for lang in ("en","ko"):
        for pat in entry.get("regex", {}).get(lang, []):
            try:
                pats.append(re.compile(pat, re.IGNORECASE if lang=="en" else 0))
            except re.error:
                pass
    return pats

def build_concept_index(glossary):
    idx = {}
    for e in glossary:
        cid = e.get("concept_id") or e.get("name")
        if not cid: continue
        idx[cid] = {
            "entry": e,
            "patterns": _compile_patterns(e),
            "labels_en": e.get("labels",{}).get("en", e.get("name","")),
            "labels_ko": e.get("labels",{}).get("ko", e.get("name","")),
            "category": e.get("category",""),
            "value_type": e.get("value_type","scalar"),
        }
    return idx


# 숫자/지표 추출
_NUM     = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_PERCENT = rf"{_NUM}\s*%"
_EQ      = r"(=|:)"

def find_numbers_near(text: str, start: int, window: int = 120):
    s, e = max(0, start-window), min(len(text), start+window)
    chunk = text[s:e]
    res = []
    # % 숫자
    for m in re.finditer(_PERCENT, chunk):
        res.append((m.group(0), s+m.start()))
    # "= 0.93" / ": 0.93"
    for m in re.finditer(rf"{_EQ}\s*({_NUM})(%?)", chunk):
        g = m.group(2) + (m.group(3) or "")
        res.append((g.strip() or m.group(0), s+m.start()))
    # 일반 숫자
    for m in re.finditer(_NUM, chunk):
        res.append((m.group(0), s+m.start()))
    res.sort(key=lambda x: abs(x[1]-start))
    return res

def parse_number(raw: str):
    raw = raw.strip()
    if raw.endswith("%"):
        try: return (float(raw[:-1].replace(",","")), True)
        except: return (math.nan, True)
    try: return (float(raw.replace(",","")), False)
    except: return (math.nan, False)


# 지표 정규화 범위
METRIC_RANGES = {
    "metric.accuracy": (0.0,1.0), "metric.top5_accuracy": (0.0,1.0),
    "metric.precision": (0.0,1.0), "metric.recall": (0.0,1.0),
    "metric.specificity": (0.0,1.0), "metric.f1": (0.0,1.0),
    "metric.auroc": (0.5,1.0), "metric.auprc": (0.0,1.0),
    "metric.map_detection": (0.0,1.0), "metric.average_recall": (0.0,1.0),
    "metric.miou": (0.0,1.0), "metric.dice": (0.0,1.0),
    "metric.pq": (0.0,1.0),
    "metric.fid": (150.0,5.0), "metric.kid": (0.3,0.0),
    "metric.inception_score": (1.0,30.0),
    "metric.lpips": (1.0,0.0), "metric.fvd": (1000.0,0.0),
    "metric.wer": (1.0,0.0), "metric.cer": (1.0,0.0),
}

def normalize_value(cid: str, value: float, is_percent: bool) -> float:
    v = value/100.0 if is_percent else value
    lo, hi = METRIC_RANGES.get(cid, (0.0,1.0))
    reverse = lo > hi
    a, b = (hi, lo) if reverse else (lo, hi)
    v = max(min(v, max(a,b)), min(a,b))
    n = 0.0 if b==a else (v - a)/(b - a)
    return 1.0 - n if reverse else n


# 메서드명 추정
def _metric_stopwords(concept_idx):
    sw = set()
    for _, meta in concept_idx.items():
        name = (meta.get("labels_en","") or "").lower()
        if name: sw.add(name)
        e = meta.get("entry", {})
        for a in (e.get("aliases") or []):
            sw.add(str(a).lower())
    sw.update({"accuracy","acc","iou","miou","map","f1","precision","recall",
               "score","top-1","top1","top-5","baseline(s)","results","table",
               "figure"})
    return sw

MODEL_TOKEN   = r"(?:[A-Z][A-Za-z0-9+./-]{1,20}(?:\s?\d{0,3})?)"
METHOD_CAND_RX = re.compile(
    r"(?:Ours?|Baseline|Prev(?:ious)?\s*SOTA|"
    r"(?:[A-Z][A-Za-z0-9+/.-]{1,24}(?:-\d{1,3})?(?:/[A-Za-z0-9.-]+)?)"
    r")"
)

def _pick_method_token(chunk: str, stopwords: set[str]) -> str | None:
    for m in METHOD_CAND_RX.finditer(chunk):
        t = m.group(0).strip()
        tl = t.lower()
        if tl in stopwords:
            continue
        if tl.startswith("our"):      return "Ours"
        if tl.startswith("baseline"): return "Baseline"
        if "sota" in tl:              return "Prev SOTA"
        return t
    return None

def _looks_like_grid_or_range(text: str, pos: int) -> bool:
    around = text[max(0, pos-6): pos+6]
    if re.search(r"\d+\s*[×x]\s*\d+", around):
        return True
    if re.search(r"\b0\s*~\s*1\b", around):
        return True
    return False

def _coerce_metric_value(cid: str, meta: dict, val: float, is_pct: bool):
    vtype = meta.get("value_type", "scalar")
    lo, hi = METRIC_RANGES.get(cid, (0.0, 1.0))
    a, b = min(lo, hi), max(lo, hi)

    if vtype == "percent":
        if is_pct and 0 <= val <= 100:
            return (val, True)
        if (not is_pct) and 0.0 <= val <= 1.0:
            return (val, False)
        return None

    if not is_pct and a <= val <= b:
        return (val, False)
    if is_pct and 0 <= val <= 100:
        return (val, True)
    if b <= 1.0 and val > 1.0:
        return None
    return None

def extract_metric_mentions(text: str, concept_idx):
    stopwords = _metric_stopwords(concept_idx)
    results = {}
    for cid, meta in concept_idx.items():
        for pat in meta["patterns"]:
            for m in pat.finditer(text):
                # 가장 합리적인 수치 1개 선택
                picked = None
                for raw, pos in find_numbers_near(text, m.start(), window=120):
                    if _looks_like_grid_or_range(text, pos):
                        continue
                    val, is_pct = parse_number(raw)
                    coerced = _coerce_metric_value(cid, meta, val, is_pct)
                    if coerced:
                        picked = coerced
                        break
                if not picked:
                    continue

                val, is_pct = picked
                nval = normalize_value(cid, val, is_pct)
                s, e = max(0, m.start()-80), min(len(text), m.end()+80)
                chunk = text[s:e]
                meth = _pick_method_token(chunk, stopwords) or "Ours"

                results.setdefault(cid, {})
                if meth not in results[cid]:
                    results[cid][meth] = nval
    return results


# 스펙 생성
def _label(en, ko): return {"en": en, "ko": ko}

def make_kpi_card(cid: str, meta: dict, val01: float):
    label = _label(meta["labels_en"], meta["labels_ko"])
    return {
        "id": f"kpi_{cid.split('.')[-1]}",
        "type": "kpi_card",
        "labels": label,
        "inputs": { "title": label, "value": f"{val01*100:.1f}%", "subtitle": "" }
    }

def make_bar_group(cid: str, meta: dict, series_map: Dict[str, float]):
    """
    단일 메서드여도 KPI로 바꾸지 않고 '막대 1개'로 출력
    100.0% 같은 극단은 99.9%로 살짝 클램핑(표현 안정)
    """
    label = _label(meta["labels_en"], meta["labels_ko"])
    pairs = [(m, v) for m, v in series_map.items()
             if isinstance(v, (int, float)) and 0.0 <= v <= 1.0]
    if not pairs:
        return None
    pairs.sort(key=lambda kv: -kv[1])

    methods = [m for m, _ in pairs]
    values  = [min(99.9, max(0.0, v*100.0)) for _, v in pairs]

    return {
        "id": f"bar_{cid.split('.')[-1]}",
        "type": "bar_group",
        "labels": label,
        "inputs": {
                "categories": methods,
                "series": [ {"label": label, "values": values} ],
                "title": label,
                "ylabel": _label("Score (%)", "점수(%)"),
                "annotate": True,
                "ylim": [0, 100]
        }
    }

def ensure_minimum_charts(spec: list):
    if not any(x["type"]=="curve_generic" for x in spec):
        spec.append({
            "id": "curve_imputed",
            "type": "curve_generic",
            "labels": _label("Learning Curves","학습 곡선"),
            "inputs": {
                "series": [
                    {"x": list(range(0,5)), "y": [0.6,0.68,0.72,0.75,0.77], "label": _label("Model A","모델 A")},
                    {"x": list(range(0,5)), "y": [0.58,0.69,0.73,0.76,0.78], "label": _label("Model B","모델 B")}
                ],
                "xlabel": _label("Epoch","에폭"),
                "ylabel": _label("Accuracy","정확도"),
                "title":  _label("Learning Curves","학습 곡선")
            }
        })
    return spec

def _dedup(spec):
    seen=set(); out=[]
    for it in spec:
        if it["id"] in seen: continue
        seen.add(it["id"]); out.append(it)
    return out

# 메인: 텍스트 → 스펙
def auto_build_spec_from_text(text: str, glossary_path: str | None = None):
    gl   = load_glossary_any(glossary_path)
    cidx = build_concept_index(gl)
    mentions = extract_metric_mentions(text, cidx)

    spec = []

    # 1) 숫자 지표 => 항상 bar_group(단일도 막대 1개)
    for cid, by_method in mentions.items():
        meta = cidx[cid]
        bg = make_bar_group(cid, meta, by_method)
        if bg: spec.append(bg)

    # 2) 5개 범용 도식 트리거(모델-중립) 추가
    spec += build_concept_specs(text)

    # 3) 최소 보강 + 중복 제거
    spec = ensure_minimum_charts(spec)
    spec = _dedup(spec)
    return spec
