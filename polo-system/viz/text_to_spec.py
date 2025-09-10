import re, json, math, random, hashlib
from pathlib import Path
from typing import List, Dict, Any
# 개념/예시 도식 트리거
from templates.generic_rules import build_concept_specs

# 유틸
def _label(en, ko): return {"en": en, "ko": ko}

def _stable_seed(text: str, salt: str = "") -> int:
    h = hashlib.sha256((salt + "|" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _rng_from(text: str, salt: str = "") -> random.Random:
    return random.Random(_stable_seed(text, salt))

def _clip(v, lo=0.05, hi=0.95):
    return max(lo, min(hi, v))

# glossary
def _find_glossary_path():
    here = Path(__file__).parent
    for p in (here/"glossary_hybrid.json", here/"glossary.json"):
        if p.exists(): return str(p)
    return None

def load_glossary_any(path: str | None = None) -> List[Dict[str, Any]]:
    gp = path or _find_glossary_path()
    if not gp: return []
    return json.loads(Path(gp).read_text(encoding="utf-8"))

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

# 숫자 파싱
_NUM     = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_PERCENT = rf"{_NUM}\s*%"
_EQ      = r"(=|:)"

def find_numbers_near(text: str, start: int, window: int = 120):
    s, e = max(0, start-window), min(len(text), start+window)
    chunk = text[s:e]
    res = []
    for m in re.finditer(_PERCENT, chunk):
        res.append((m.group(0), s+m.start()))
    for m in re.finditer(rf"{_EQ}\s*({_NUM})(%?)", chunk):
        g = m.group(2) + (m.group(3) or "")
        res.append((g.strip() or m.group(0), s+m.start()))
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

# 정규화
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

# -------------------- Rubrics (기본 임계값) --------------------
# mode: "up"(클수록 좋음) | "down"(작을수록 좋음)
# thresholds: 높은 등급부터 나열. 값은 '원 단위'(0~1 스케일 X)
RUBRICS: dict[str, dict] = {
    # 분류/랭킹
    "metric.accuracy":      {"mode":"up",   "title":"Accuracy rubric",
                             "col":"min score",  "thr":[(0.98,"Excellent"), (0.95,"Good"), (0.90,"Fair"), (0.00,"Poor")]},
    "metric.f1":            {"mode":"up",   "title":"F1-score rubric",
                             "col":"min score",  "thr":[(0.90,"Excellent"), (0.80,"Good"), (0.70,"Fair"), (0.00,"Poor")]},
    "metric.precision":     {"mode":"up",   "title":"Precision rubric",
                             "col":"min score",  "thr":[(0.95,"Excellent"), (0.90,"Good"), (0.80,"Fair"), (0.00,"Poor")]},
    "metric.recall":        {"mode":"up",   "title":"Recall rubric",
                             "col":"min score",  "thr":[(0.95,"Excellent"), (0.90,"Good"), (0.80,"Fair"), (0.00,"Poor")]},
    "metric.auroc":         {"mode":"up",   "title":"AUROC rubric",
                             "col":"min score",  "thr":[(0.90,"Excellent"), (0.85,"Good"), (0.75,"Fair"), (0.50,"Poor")]},
    # ⚠ AUPRC는 양성비에 민감 — 기본값은 보수적으로
    "metric.auprc":         {"mode":"up",   "title":"AUPRC rubric (baseline≈prevalence)",
                             "col":"min score",  "thr":[(0.70,"Excellent"), (0.50,"Good"), (0.30,"Fair"), (0.00,"Poor")]},

    # 검출/세그먼테이션
    "metric.map_detection": {"mode":"up",   "title":"mAP rubric (@[.5:.95])",
                             "col":"min mAP",   "thr":[(0.55,"Excellent"), (0.40,"Good"), (0.25,"Fair"), (0.00,"Poor")]},
    "metric.average_recall":{"mode":"up",   "title":"AR rubric",
                             "col":"min AR",    "thr":[(0.70,"Excellent"), (0.55,"Good"), (0.40,"Fair"), (0.00,"Poor")]},
    "metric.miou":          {"mode":"up",   "title":"mIoU rubric",
                             "col":"min IoU",   "thr":[(0.80,"Excellent"), (0.60,"Good"), (0.40,"Fair"), (0.00,"Poor")]},
    "metric.dice":          {"mode":"up",   "title":"Dice rubric",
                             "col":"min Dice",  "thr":[(0.85,"Excellent"), (0.75,"Good"), (0.65,"Fair"), (0.00,"Poor")]},
    "metric.pq":            {"mode":"up",   "title":"PQ rubric",
                             "col":"min PQ",    "thr":[(0.55,"Excellent"), (0.45,"Good"), (0.35,"Fair"), (0.00,"Poor")]},
    "metric.iou":           {"mode":"up",   "title":"IoU rubric",
                             "col":"min IoU",   "thr":[(0.75,"Excellent"), (0.60,"Good"), (0.50,"Fair"), (0.00,"Poor")]},

    # 생성/복원(낮을수록 좋음)
    "metric.fid":           {"mode":"down", "title":"FID rubric",
                             "col":"max FID",   "thr":[(10.0,"Excellent"), (25.0,"Good"), (50.0,"Fair"), (1e9,"Poor")]},
    "metric.kid":           {"mode":"down", "title":"KID rubric",
                             "col":"max KID",   "thr":[(0.01,"Excellent"), (0.02,"Good"), (0.05,"Fair"), (1e9,"Poor")]},
    "metric.lpips":         {"mode":"down", "title":"LPIPS rubric",
                             "col":"max LPIPS", "thr":[(0.15,"Excellent"), (0.25,"Good"), (0.35,"Fair"), (1e9,"Poor")]},
    "metric.inception_score":{"mode":"up",  "title":"IS rubric (dataset-dep.)",
                             "col":"min IS",    "thr":[(8.5,"Excellent"), (7.5,"Good"), (6.5,"Fair"), (0.0,"Poor")]},

    # 음성/문자 인식(낮을수록 좋음)
    "metric.wer":           {"mode":"down", "title":"WER rubric",
                             "col":"max WER",   "thr":[(0.05,"Excellent"), (0.10,"Good"), (0.20,"Fair"), (1.0,"Poor")]},
    "metric.cer":           {"mode":"down", "title":"CER rubric",
                             "col":"max CER",   "thr":[(0.02,"Excellent"), (0.05,"Good"), (0.10,"Fair"), (1.0,"Poor")]},
}

def normalize_value(cid: str, value: float, is_percent: bool) -> float:
    v = value/100.0 if is_percent else value
    lo, hi = METRIC_RANGES.get(cid, (0.0,1.0))
    reverse = lo > hi
    a, b = (hi, lo) if reverse else (lo, hi)
    v = max(min(v, max(a,b)), min(a,b))
    n = 0.0 if b==a else (v - a)/(b - a)
    return 1.0 - n if reverse else n

# 메서드 토큰
def _metric_stopwords(concept_idx):
    sw = set()
    for _, meta in concept_idx.items():
        name = (meta.get("labels_en","") or "").lower()
        if name: sw.add(name)
        e = meta.get("entry", {})
        for a in (e.get("aliases") or []):
            sw.add(str(a).lower())
    sw.update({"accuracy","acc","iou","miou","map","f1","precision","recall",
               "score","top-1","top1","top-5","baselines","baseline","results","table",
               "figure"})
    return sw

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
    around = text[max(0, pos-8): pos+8]
    if re.search(r"\d+\s*[×x]\s*\d+", around):   # 13x13 같은 해상도 표기
        return True
    if re.search(r"\b0\s*~\s*1\b", around):      # 0~1 범위
        return True
    return False

def _coerce_metric_value(cid: str, meta: dict, val: float, is_pct: bool):
    # 극단 노이즈 제거 + 스케일 확인
    vtype = meta.get("value_type", "scalar")
    lo, hi = METRIC_RANGES.get(cid, (0.0, 1.0))
    a, b = min(lo, hi), max(lo, hi)

    if is_pct and (val >= 99.9 or val <= 0.1):
        return None

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

def _is_confident_raw(raw: str, is_pct: bool) -> bool:
    r = (raw or "").strip()
    if is_pct:                     # 95%, 73.4% 같이 %가 있으면 확실
        return True
    # 소수점 포함 + 0/1이 아닌 값만 허용 (0, 1, 0.0, 1.0은 범위 표기/개념문장일 가능성 큼)
    if "." in r and r not in {"0", "1", "0.0", "1.0"}:
        return True
    return False

def extract_metric_mentions(text: str, concept_idx):
    stopwords = _metric_stopwords(concept_idx)
    results: Dict[str, Dict[str, float]] = {}
    numeric_cids: set[str] = set()  # ← 텍스트에 '실제 숫자'가 있었던 지표 id

    for cid, meta in concept_idx.items():
        for pat in meta["patterns"]:
            for m in pat.finditer(text):
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
                    numeric_cids.add(cid)     # ← 이 CID는 진짜 숫자를 갖고 있음

    return results, numeric_cids

# 숫자 필요 도형 목록
NUMERIC_REQUIRED = {"kpi_card", "bar_group", "metric_table"}  # 퍼센트/정량 비교 필수

# 빌더
def make_kpi_card(cid: str, meta: dict, val01: float):
    label = _label(meta["labels_en"], meta["labels_ko"])
    return {
        "id": f"kpi_{cid.split('.')[-1]}",
        "type": "kpi_card",
        "labels": label,
        "inputs": { "title": label, "value": f"{val01*100:.1f}%", "subtitle": "" }
    }

def make_bar_group(cid: str, meta: dict, series_map: Dict[str, float]):
    label = _label(meta["labels_en"], meta["labels_ko"])
    pairs = [(m, v) for m, v in series_map.items()
             if isinstance(v, (int, float)) and 0.0 < v < 1.0]
    pairs.sort(key=lambda kv: -kv[1])
    if len(pairs) <= 1:
        return make_kpi_card(cid, meta, pairs[0][1]) if pairs else None

    methods = [m for m, _ in pairs]
    values  = [v*100 for _, v in pairs]
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

def auto_build_spec_from_text(text: str, glossary_path: str | None = None):
    gl   = load_glossary_any(glossary_path)
    cidx = build_concept_index(gl)

    # (A) 텍스트에서 숫자 추출 + '숫자 근거' 세트
    mentions, numeric_cids = extract_metric_mentions(text, cidx)

    # (B) 지표명이 숫자 없이만 언급된 경우 (막대 비교용 임퓨트)
    names_only = _detect_metric_names_only(text, cidx)

    spec = []

    # --- KPI/Bar: 2개 이상일 때만 막대 그래프, 1개면 생성 안 함 ---
    for cid, by_method in mentions.items():
        meta = cidx[cid]
        if len(by_method) >= 2:
            bg = make_bar_group(cid, meta, by_method)
            if bg:
                spec.append(bg)

    # 숫자 있는 지표에만 해석 표(예: IoU rubric) 붙이기
    _add_rubric_tables(spec, mentions)

    # (D) 숫자는 없고 지표명만 있는 경우 → 비교 막대는 '임퓨트'로 생성
    for cid in (names_only - set(mentions.keys())):
        meta = cidx[cid]
        methods = ["Ours", "Baseline", "Prev SOTA"]
        vals = impute_values_for_methods(methods, cid, text)
        by_method = dict(zip(methods, vals))
        bg = make_bar_group(cid, meta, by_method)
        if bg: spec.append(bg)

    # (E) 평가지표 해석 표(루브릭)는 '실제 숫자'가 잡힌 경우에만 덧붙임
    _add_rubric_tables(spec, mentions, numeric_cids)

    # (F) 개념/예시 도식들
    spec += build_concept_specs(text, spec, mentions, numeric_cids)

    # (G) 최소 보강 + 중복 제거
    spec = ensure_minimum_charts(spec)
    spec = _dedup(spec)
    return spec

def _detect_metric_names_only(text: str, concept_idx) -> set[str]:
    cids = set()
    for cid, meta in concept_idx.items():
        for pat in meta["patterns"]:
            if pat.search(text):
                cids.add(cid); break
    return cids

def _add_rubric_tables(spec: list, mentions: dict, numeric_cids: set[str] | None = None):
    """
    숫자값이 실제로 추출된 지표(cids)에 대해서만 RUBRICS 기반 'metric_table'을 붙인다.
    """
    if numeric_cids is None:
        numeric_cids = {cid for cid, by_method in mentions.items() if by_method}

    def _append_table(cid: str, rub: dict):
        label_en = rub.get("title", cid)
        label_ko = {
            "min score":"최소 점수", "max FID":"최대 FID", "max KID":"최대 KID",
            "max LPIPS":"최대 LPIPS", "min IoU":"최소 IoU", "min mAP":"최소 mAP",
            "min AR":"최소 AR", "min Dice":"최소 Dice", "min PQ":"최소 PQ",
            "min IS":"최소 IS", "max WER":"최대 WER", "max CER":"최대 CER"
        }.get(rub["col"], rub["col"])

        rows   = [name for _, name in rub["thr"]]
        # 표에는 '임계값'을 그대로 넣는다(원 단위). 소수는 보기 좋게 반올림.
        vals   = [[round(th, 3) if th < 1e8 else float("inf")] for th, _ in rub["thr"]]
        spec.append({
            "id": f"{cid.split('.')[-1]}_rubric",
            "type": "metric_table",
            "labels": {"en": label_en, "ko": label_en.replace("rubric", "해석")},
            "inputs": {
                "methods": rows,
                "metrics": [rub["col"]],
                "values":  vals,
                "title":   {"en": label_en, "ko": label_en.replace("rubric", "해석")}
            }
        })

    for cid in sorted(numeric_cids):
        rub = RUBRICS.get(cid)
        if rub:
            _append_table(cid, rub)

def impute_values_for_methods(methods: list[str], cid: str, text: str) -> list[float]:
    """
    숫자가 없을 때 비교 막대용 임퓨트 값 생성.
    - 논문 텍스트 + cid + methods로 시드를 만들어 문서마다 모양이 다르고,
      같은 문서에선 재현성이 유지됩니다.
    - 0.05~0.95 사이로 클리핑.
    - 낮을수록 좋은 지표(FID/WER 등)는 자동으로 역전.
    """
    rnd = _rng_from(text, f"impute:{cid}:{','.join(sorted(methods))}")
    base = 0.55 + 0.25 * rnd.random()      # 0.55 ~ 0.80
    step = 0.02  + 0.04 * rnd.random()      # 0.02 ~ 0.06
    vals = []
    for i in range(len(methods)):
        v = base + i*step + rnd.uniform(-0.01, 0.01)  # 약간의 노이즈
        vals.append(_clip(v, 0.05, 0.95))

    # 낮을수록 좋은 지표는 방향 반전
    lo, hi = METRIC_RANGES.get(cid, (0.0, 1.0))
    if lo > hi:
        vals = list(reversed(vals))
    return vals