"""
텍스트 → spec 자동 빌더 (glossary 기반)
- 지표 탐지/정규화/바차트/루브릭 테이블
- 히스토그램은 glossary의 viz.trigger.histogram로만 생성
- 나머지 개념 도식은 templates/generic_rules.build_concept_specs 에서 추가
"""
from __future__ import annotations
import re, json, math, random, hashlib, statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 개념/예시 도식 트리거
from templates.generic_rules import build_concept_specs

# 스펙을 값 없이 템플릿으로 추가하지 않음
ALLOW_FALLBACK_TEMPLATES = False

# 히스토그램 필터
_NUM = r"-?\d+(?:\.\d+)?"
_LIST_LINE = re.compile(
    rf"(?:values?|data|list|측정값|데이터)\s*[:：]\s*(?P<body>(?:{_NUM}(?:\s*[,，]\s*{_NUM})+))",
    re.I,
)
_INLINE_LIST = re.compile(rf"\[\s*(?P<body>{_NUM}(?:\s*[,，]\s*{_NUM})+)\s*\]")

def _to_floats(s): 
    return [float(x) for x in re.findall(_NUM, s)]
def _looks_like_year(x): 
    return 1900 <= x <= 2100 and float(x).is_integer()
def _is_percent_ctx(s): 
    return "%" in s or re.search(r"(비율|퍼센트|percentage|ratio)", s, re.I)

def parse_hist_values(text, min_n=20, min_unique=8):
    cands=[]
    for m in _LIST_LINE.finditer(text):
        if _is_percent_ctx(m.group(0)): continue
        cands.append(_to_floats(m.group("body")))
    for m in _INLINE_LIST.finditer(text):
        ctx = text[max(0, m.start()-30): m.end()+30]
        if _is_percent_ctx(ctx): continue
        cands.append(_to_floats(m.group("body")))
    for vals in cands:
        vals = [v for v in vals if not _looks_like_year(v)]
        if len(vals) < min_n or len(set(vals)) < min_unique: continue
        try:
            if statistics.pstdev(vals) == 0: continue
        except statistics.StatisticsError:
            continue
        return vals
    return []


# 구성/비율: "Label 12%" 여러 개 → 100±5%면 parts로 채택
_COMPOSE_PAT = re.compile(r'(?P<label>[A-Za-z가-힣0-9/+\-_. ]{2,30})\s*[:=]?\s*(?P<pct>\d{1,3})\s*%')
def parse_composition_parts(text, max_labels=10):
    items = [(m.group('label').strip(), int(m.group('pct'))) for m in _COMPOSE_PAT.finditer(text)]
    if not items: return []
    best, run, last_end = [], [], -10
    for m in _COMPOSE_PAT.finditer(text):
        pair = (m.group('label').strip(), int(m.group('pct')))
        if m.start() - last_end < 50: run.append(pair)
        else:
            if len(run) >= 2 and 95 <= sum(p for _,p in run) <= 105: best = run; break
            run = [pair]
        last_end = m.end()
    if not best:
        items.sort(key=lambda x: -x[1])
        if 95 <= sum(p for _,p in items[:max_labels]) <= 105: best = items[:max_labels]
    return best

# 비교/벤치: "Task: A=80, B=83" 라인 모음 → (categories, series)
_TASK = re.compile(r'(?P<task>[A-Za-z가-힣0-9._/\-+ ]{2,40})\s*[:：]\s*(?P<body>.+)')
_PAIR = re.compile(r'(?P<name>[A-Za-z가-힣0-9._/\-+ ]{1,30})\s*[=:\(]\s*(?P<val>-?\d+(?:\.\d+)?)')
def parse_benchmark(text, max_tasks=12, max_series=8):
    cats, vals = [], {}
    for line in text.splitlines():
        m = _TASK.match(line.strip())
        if not m: continue
        cats.append(m.group('task').strip())
        for n,v in _PAIR.findall(m.group('body')):
            vals.setdefault(n.strip(), []).append(float(v))
    cats = cats[:max_tasks]
    series = [{"label": k, "values": (v + [0.0]*(len(cats)-len(v)))[:len(cats)]}
              for k,v in list(vals.items())[:max_series]]
    return (cats, series) if cats and series else ([], [])

# 카테고리 분해(스택): "Dataset A: Span 20%, ..." 라인 다수 → (categories, series)
def parse_breakdown(text, max_cats=10, max_parts=10):
    cats, row_parts, part_names = [], [], []
    for line in text.splitlines():
        m = _TASK.match(line.strip())
        if not m: continue
        cat = m.group('task').strip()
        pairs = [(l.strip(), int(p)) for l,p in _COMPOSE_PAT.findall(m.group('body'))]
        if len(pairs) < 2: continue
        cats.append(cat); row_parts.append(pairs)
        for l,_ in pairs:
            if l not in part_names and len(part_names) < max_parts: part_names.append(l)
    if not cats: return ([], [])
    series = []
    for name in part_names:
        col = [next((p for l,p in row if l==name), 0) for row in row_parts]
        series.append({"label": name, "values": col})
    return (cats[:max_cats], series[:max_parts])

# 특수 토큰/시퀀스: 문서에 실제 등장하면만 토큰 배열 생성
def parse_special_tokens(text):
    tl = text.lower()
    has_cls = "[cls]" in tl or "<s>" in text
    has_sep = "[sep]" in tl or "</s>" in text
    if not (has_cls or has_sep): return []
    toks = ["[CLS]"] if has_cls else []
    toks += ["sentA"]
    if has_sep: toks += ["[SEP]", "sentB", "[SEP]"]
    return toks

# 유틸리티(라벨 한글화, 값 정규화)
def _label(en, ko): return {"en": en, "ko": ko}

def _stable_seed(text: str, salt: str = "") -> int:
    h = hashlib.sha256((salt + "|" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _rng_from(text: str, salt: str = "") -> random.Random:
    return random.Random(_stable_seed(text, salt))

def _clip(v, lo=0.05, hi=0.95):
    return max(lo, min(hi, v))

# 용어 IO
def _find_glossary_path():
    here = Path(__file__).parent
    for p in (here / "glossary_hybrid.json", here / "glossary.json"):
        if p.exists():
            return str(p)
    return None

def load_glossary_any(path: str | None = None) -> List[Dict[str, Any]]:
    gp = path or _find_glossary_path()
    if not gp:
        return []
    return json.loads(Path(gp).read_text(encoding="utf-8"))

def _compile_patterns(entry: Dict[str, Any]):
    pats = []
    for lang in ("en", "ko"):
        for pat in entry.get("regex", {}).get(lang, []):
            try:
                pats.append(re.compile(pat, re.IGNORECASE if lang == "en" else 0))
            except re.error:
                pass
    return pats

def build_concept_index(glossary):
    """glossary 레코드를 concept_id → 메타 dict로 인덱싱"""
    idx = {}
    for e in glossary:
        cid = e.get("concept_id") or e.get("name")
        if not cid:
            continue
        idx[cid] = {
            "entry": e,
            "patterns": _compile_patterns(e),
            "labels_en": e.get("labels", {}).get("en", e.get("name", "")),
            "labels_ko": e.get("labels", {}).get("ko", e.get("name", "")),
            "category": e.get("category", ""),
            "value_type": e.get("value_type", "scalar"),
        }
    return idx

# 숫자 파서
_NUM     = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_PERCENT = rf"{_NUM}\s*%"
_EQ      = r"(=|:)"

def find_numbers_near(text: str, start: int, window: int = 120):
    s, e = max(0, start - window), min(len(text), start + window)
    chunk = text[s:e]
    res: List[Tuple[str, int]] = []
    for m in re.finditer(_PERCENT, chunk):
        res.append((m.group(0), s + m.start()))
    for m in re.finditer(rf"{_EQ}\s*({_NUM})(%?)", chunk):
        g = m.group(2) + (m.group(3) or "")
        res.append((g.strip() or m.group(0), s + m.start()))
    for m in re.finditer(_NUM, chunk):
        res.append((m.group(0), s + m.start()))
    res.sort(key=lambda x: abs(x[1] - start))
    return res

def parse_number(raw: str):
    raw = raw.strip()
    if raw.endswith("%"):
        try:
            return (float(raw[:-1].replace(",", "")), True)
        except:
            return (math.nan, True)
    try:
        return (float(raw.replace(",", "")), False)
    except:
        return (math.nan, False)

def _looks_like_grid_or_range(text: str, pos: int) -> bool:
    around = text[max(0, pos - 8): pos + 8]
    if re.search(r"\d+\s*[×x]\s*\d+", around):  # 13x13 같은 해상도 표기
        return True
    if re.search(r"\b0\s*~\s*1\b", around):     # 0~1 범위
        return True
    return False

# 평가지표 범위
# (glossary에 norm_range/min_reasonable가 없을 때만 사용)
METRIC_RANGES = {
    "metric.accuracy": (0.0, 1.0), "metric.top5_accuracy": (0.0, 1.0),
    "metric.precision": (0.0, 1.0), "metric.recall": (0.0, 1.0),
    "metric.specificity": (0.0, 1.0), "metric.f1": (0.0, 1.0),
    "metric.auroc": (0.5, 1.0), "metric.auprc": (0.0, 1.0),
    "metric.map_detection": (0.0, 1.0), "metric.average_recall": (0.0, 1.0),
    "metric.miou": (0.0, 1.0), "metric.dice": (0.0, 1.0),
    "metric.pq": (0.0, 1.0),
    "metric.fid": (150.0, 5.0), "metric.kid": (0.3, 0.0),
    "metric.inception_score": (1.0, 30.0),
    "metric.lpips": (1.0, 0.0), "metric.fvd": (1000.0, 0.0),
    "metric.wer": (1.0, 0.0), "metric.cer": (1.0, 0.0),
}
MIN_REASONABLE = {
    "metric.accuracy": 0.5, "metric.precision": 0.5, "metric.recall": 0.5,
    "metric.f1": 0.5, "metric.auroc": 0.6, "metric.auprc": 0.3,
    "metric.map_detection": 0.2, "metric.miou": 0.3, "metric.iou": 0.3,
}
BAR_ALLOWED_CIDS = set(METRIC_RANGES.keys())

# Glossary-driven 핼퍼
def _g_aliases(meta, key="aliases"):
    al = meta.get("entry", {}).get(key, {})
    out = []
    for lang in ("en", "ko"):
        out += [a.lower() for a in al.get(lang, [])]
    return out

def _g_label_tokens(meta):
    lab = meta.get("entry", {}).get("labels", {})
    return [lab.get("en", "").lower(), lab.get("ko", "").lower()]

def _stopwords_from_glossary(concept_idx) -> set[str]:
    sw, cats = set(), {"dataset", "section", "stopword", "forbidden.method_token"}
    for _, meta in concept_idx.items():
        if meta.get("category") in cats:
            sw.update([t for t in _g_label_tokens(meta) if t])
            sw.update(_g_aliases(meta))
    sw.update({"results", "table", "figure", "supplementary", "appendix"})
    return sw

def _candidate_method_regex(concept_idx):
    wl = []
    for _, meta in concept_idx.items():
        if meta.get("category") == "method.whitelist":
            wl += _g_aliases(meta) + [t for t in _g_label_tokens(meta) if t]
    core = r"(?:Ours?|Baseline|Prev(?:ious)?\s*SOTA)"
    wlrx = ("|" + "|".join(sorted(set(re.escape(w) for w in wl)))) if wl else ""
    generic = r"|(?:[A-Z][A-Za-z0-9+/.-]*\d[A-Za-z0-9+/.-]*)"
    return re.compile(f"(?:{core}{wlrx}{generic})")

def _forbidden_tokens(concept_idx) -> set[str]:
    out = set()
    for _, meta in concept_idx.items():
        if meta.get("category") == "forbidden.method_token":
            out.update(_g_aliases(meta))
            out.update([t for t in _g_label_tokens(meta) if t])
    return out

def _has_trigger(concept_idx, category: str, text: str) -> bool:
    for _, meta in concept_idx.items():
        if meta.get("category") == category:
            for pat in meta["patterns"]:
                if pat.search(text):
                    return True
    return False

def _norm_range_from_meta(cid: str, meta: dict) -> tuple[float, float]:
    rng = meta.get("entry", {}).get("norm_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return float(rng[0]), float(rng[1])
    return METRIC_RANGES.get(cid, (0.0, 1.0))

def _min_reasonable_from_meta(cid: str, meta: dict) -> float | None:
    mr = meta.get("entry", {}).get("min_reasonable")
    if mr is not None:
        try:
            return float(mr)
        except:
            pass
    return MIN_REASONABLE.get(cid)

# Normalization / coercion
def normalize_value(cid: str, value: float, is_percent: bool, meta: dict | None = None) -> float:
    v = value / 100.0 if is_percent else value
    lo, hi = _norm_range_from_meta(cid, meta or {"entry": {}})
    reverse = lo > hi
    a, b = (hi, lo) if reverse else (lo, hi)
    v = max(min(v, max(a, b)), min(a, b))
    n = 0.0 if b == a else (v - a) / (b - a)
    return 1.0 - n if reverse else n

def _coerce_metric_value(cid: str, meta: dict, val: float, is_pct: bool):
    vtype = meta.get("value_type", "scalar")
    lo, hi = _norm_range_from_meta(cid, meta)
    a, b = min(lo, hi), max(lo, hi)

    if is_pct and (val >= 99.9 or val <= 0.1):     # 너무 극단적인 % 값 노이즈 컷
        return None

    if vtype == "percent":
        if is_pct and 0 <= val <= 100:  return (val, True)
        if (not is_pct) and 0.0 <= val <= 1.0: return (val, False)
        return None

    if (not is_pct) and a <= val <= b:  return (val, False)
    if is_pct and 0 <= val <= 100:      return (val, True)
    if b <= 1.0 and val > 1.0:          return None
    return None

# 특수 토큰 필터
def _pick_method_token(chunk: str, stopwords: set[str],
                    cand_rx: re.Pattern, forbid: set[str]) -> str | None:
    def _in_parens_near(idx: int) -> bool:
        left = chunk.rfind("(", 0, idx)
        right = chunk.find(")", idx)
        return left != -1 and (right == -1 or (right - left) < 40)

    for m in cand_rx.finditer(chunk):
        t = (m.group(0) or "").strip()
        tl = t.lower()
        if not t:
            continue
        if tl in stopwords or tl in forbid:
            continue
        if _in_parens_near(m.start()):
            continue
        if tl.startswith("our"):      return "Ours"
        if tl.startswith("baseline"): return "Baseline"
        if "sota" in tl:              return "Prev SOTA"
        return t
    return None

# Metric extraction core 
def extract_metric_mentions(text: str, concept_idx):
    stopwords = _stopwords_from_glossary(concept_idx)
    cand_rx   = _candidate_method_regex(concept_idx)
    forbid    = _forbidden_tokens(concept_idx)

    results: Dict[str, Dict[str, float]] = {}
    numeric_cids: set[str] = set()

    for cid, meta in concept_idx.items():
        if meta.get("category") != "metric":
            continue

        by_method: Dict[str, float] = {}

        for pat in meta["patterns"]:
            for m in pat.finditer(text):
                picked = None  # (val, is_percent)
                for raw, pos in find_numbers_near(text, m.start(), window=120):
                    if _looks_like_grid_or_range(text, pos):
                        continue
                    v, is_pct = parse_number(raw)
                    coerced = _coerce_metric_value(cid, meta, v, is_pct)
                    if coerced:
                        picked = coerced
                        break  # 가장 가까운 ‘유효 숫자’만 사용
                if not picked:
                    continue

                val, is_pct = picked
                v01 = normalize_value(cid, val, is_pct, meta)

                thr = _min_reasonable_from_meta(cid, meta)
                if thr is not None and v01 < thr:
                    continue

                s, e = max(0, m.start() - 80), min(len(text), m.end() + 80)
                chunk = text[s:e]
                meth = _pick_method_token(chunk, stopwords, cand_rx, forbid) or "Ours"
                if meth not in by_method:
                    by_method[meth] = v01

        if by_method:
            results[cid] = by_method
            numeric_cids.add(cid)

    return results, numeric_cids

# 차트 빌더 (bars/kpi)
def make_kpi_card(cid: str, meta: dict, val01: float):
    label = _label(meta["labels_en"], meta["labels_ko"])
    return {
        "id": f"kpi_{cid.split('.')[-1]}",
        "type": "kpi_card",
        "labels": label,
        "inputs": {"title": label, "value": f"{val01*100:.1f}%", "subtitle": ""}
    }

def make_bar_group(cid: str, meta: dict, series_map: Dict[str, float]):
    label = _label(meta["labels_en"], meta["labels_ko"])
    pairs = [(m, v) for m, v in series_map.items() if isinstance(v, (int, float)) and 0.0 < v < 1.0]
    pairs.sort(key=lambda kv: -kv[1])
    if len(pairs) <= 1:
        return make_kpi_card(cid, meta, pairs[0][1]) if pairs else None

    methods = [m for m, _ in pairs]
    values  = [round(v * 100, 1) for _, v in pairs]
    return {
        "id": f"bar_{cid.split('.')[-1]}",
        "type": "bar_group",
        "labels": label,
        "inputs": {
            "categories": methods,
            "series": [{"label": label, "values": values}],
            "title": label,
            "ylabel": _label("Normalized score (%)", "정규화 점수(%)"),
            "annotate": True,
            "ylim": [0, 100]
        }
    }

# 루브릭 테이블
RUBRICS: dict[str, dict] = {
    "metric.accuracy": {"mode": "up", "title": "Accuracy rubric",
                        "col": "min score",
                        "thr": [(0.90, "Excellent"), (0.80, "Good"), (0.70, "Fair"), (0.60, "Poor")]},
    "metric.f1": {"mode": "up", "title": "F1-score rubric",
                "col": "min score",
                "thr": [(0.90, "Excellent"), (0.80, "Good"), (0.70, "Fair"), (0.60, "Poor")]},
    "metric.precision": {"mode": "up", "title": "Precision rubric",
                        "col": "min score",
                        "thr": [(0.90, "Excellent"), (0.80, "Good"), (0.70, "Fair"), (0.60, "Poor")]},
    "metric.recall": {"mode": "up", "title": "Recall rubric",
                    "col": "min score",
                    "thr": [(0.90, "Excellent"), (0.80, "Good"), (0.70, "Fair"), (0.60, "Poor")]},
    "metric.auroc": {"mode": "up", "title": "AUROC rubric",
                    "col": "min score",
                    "thr": [(0.90, "Excellent"), (0.80, "Good"), (0.70, "Fair"), (0.60, "Poor")]},
    "metric.auprc": {"mode": "up", "title": "AUPRC rubric (baseline≈prevalence)",
                    "col": "min score",
                    "thr": [(0.70, "Excellent"), (0.50, "Good"), (0.30, "Fair"), (0.00, "Poor")]},
    "metric.map_detection": {"mode": "up", "title": "mAP rubric (@[.5:.95])",
                            "col": "min mAP",
                            "thr": [(0.55, "Excellent"), (0.40, "Good"), (0.25, "Fair"), (0.00, "Poor")]},
    "metric.average_recall": {"mode": "up", "title": "AR rubric",
                            "col": "min AR",
                            "thr": [(0.70, "Excellent"), (0.55, "Good"), (0.40, "Fair"), (0.00, "Poor")]},
    "metric.miou": {"mode": "up", "title": "mIoU rubric",
                    "col": "min IoU",
                    "thr": [(0.80, "Excellent"), (0.60, "Good"), (0.40, "Fair"), (0.00, "Poor")]},
    "metric.dice": {"mode": "up", "title": "Dice rubric",
                    "col": "min Dice",
                    "thr": [(0.85, "Excellent"), (0.75, "Good"), (0.65, "Fair"), (0.00, "Poor")]},
    "metric.pq": {"mode": "up", "title": "PQ rubric",
                "col": "min PQ",
                "thr": [(0.55, "Excellent"), (0.45, "Good"), (0.35, "Fair"), (0.00, "Poor")]},
    "metric.iou": {"mode": "up", "title": "IoU rubric",
                "col": "min IoU",
                "thr": [(0.75, "Excellent"), (0.60, "Good"), (0.50, "Fair"), (0.00, "Poor")]},
    "metric.fid": {"mode": "down", "title": "FID rubric",
                "col": "max FID",
                "thr": [(10.0, "Excellent"), (25.0, "Good"), (50.0, "Fair"), (1e9, "Poor")]},
    "metric.kid": {"mode": "down", "title": "KID rubric",
                "col": "max KID",
                "thr": [(0.01, "Excellent"), (0.02, "Good"), (0.05, "Fair"), (1e9, "Poor")]},
    "metric.lpips": {"mode": "down", "title": "LPIPS rubric",
                    "col": "max LPIPS",
                    "thr": [(0.15, "Excellent"), (0.25, "Good"), (0.35, "Fair"), (1e9, "Poor")]},
    "metric.inception_score": {"mode": "up", "title": "IS rubric (dataset-dep.)",
                            "col": "min IS",
                            "thr": [(8.5, "Excellent"), (7.5, "Good"), (6.5, "Fair"), (0.0, "Poor")]},
    "metric.wer": {"mode": "down", "title": "WER rubric",
                "col": "max WER",
                "thr": [(0.05, "Excellent"), (0.10, "Good"), (0.20, "Fair"), (1.0, "Poor")]},
    "metric.cer": {"mode": "down", "title": "CER rubric",
                "col": "max CER",
                "thr": [(0.02, "Excellent"), (0.05, "Good"), (0.10, "Fair"), (1.0, "Poor")]},
}

def _add_rubric_tables(spec: list, mentions: dict, numeric_cids: set[str] | None = None):
    if numeric_cids is None:
        numeric_cids = {cid for cid, by_method in mentions.items() if by_method}

    def _append_table(cid: str, rub: dict):
        title_en = rub.get("title", cid)
        title_ko = title_en.replace("rubric", "해석")
        rows   = [name for _, name in rub["thr"]]
        values = [[round(th, 3)] for th, _ in rub["thr"]]
        spec.append({
            "id": f"{cid.split('.')[-1]}_rubric",
            "type": "metric_table",
            "labels": {"en": title_en, "ko": title_ko},
            "inputs": {
                "methods": rows,
                "metrics": [rub["col"]],
                "values":  values,
                "title":   {"en": title_en, "ko": title_ko},
            }
        })

    for cid in sorted(numeric_cids):
        rub = RUBRICS.get(cid)
        if rub:
            _append_table(cid, rub)

# 학습 곡선 빌더
ALLOW_IMPUTED_CURVES = False
ALLOW_CURVE_FRONT_IMPUTE = True
MIN_POINTS_FOR_CURVE = 3

def _build_series_with_front_impute(points, label, text, ylabel):
    rnd = _rng_from(text, f"train-curve:{label}:{ylabel}")
    points = sorted(points, key=lambda t: t[0])
    xs_obs = [p[0] for p in points]
    ys_obs = [p[1] for p in points]

    x0, y0 = xs_obs[0], ys_obs[0]
    x_last = xs_obs[-1]
    x_min = 0 if x0 > 0 else x0

    xs = list(range(int(x_min), int(x_last) + 1))
    ys, mask = [], []
    y = max(0.0, min(1.0, y0 - (0.04 + 0.03 * rnd.random())))
    slope = (ys_obs[-1] - y) / max(1, (x_last - x_min))
    slope = max(0.002, min(0.08, slope))
    for x in xs:
        if x < x0:
            y = y + slope * 0.6 + rnd.uniform(-0.004, 0.004)
            ys.append(_clip(y)); mask.append(0)
        else:
            if x in xs_obs:
                y = ys_obs[xs_obs.index(x)]
                ys.append(_clip(y)); mask.append(1)
            else:
                y = y + slope + rnd.uniform(-0.004, 0.004)
                ys.append(_clip(y)); mask.append(0)
    return xs, ys, mask, any(m == 0 for m in mask)

def _dedup(spec):
    seen, out = set(), []
    for it in spec:
        if it["id"] in seen:
            continue
        seen.add(it["id"]); out.append(it)
    return out

# 빌더
def auto_build_spec_from_text(text: str, glossary_path: str | None = None):
    gl   = load_glossary_any(glossary_path) # glossary 매칭/인덱스 생성
    cidx = build_concept_index(gl)

    spec: list[dict] = []

    # 텍스트에서 숫자 추출 + '숫자값 있는 지표' 집합
    mentions, numeric_cids = extract_metric_mentions(text, cidx)

    # Bar: 같은 지표에 '방법이 2개 이상'일 때만 생성 + 허용 지표만
    for cid, by_method in mentions.items():
        if cid not in BAR_ALLOWED_CIDS:
            continue
        if len(by_method) >= 2:
            meta = cidx[cid]
            bg = make_bar_group(cid, meta, by_method)
            if bg:
                spec.append(bg)

    # 히스토그램: glossary 트리거 + 충분한 숫자일 때만
    # 추가: 새 viz 트리거 처리 (bar_group / donut_pct / embedding_sum / token_sequence / stack_bar / curve_generic)
    def _append_if(cond, item):
        if cond:
            spec.append(item)

    # 루브릭 표: 실제 숫자가 잡힌 지표에만 추가
    _add_rubric_tables(spec, mentions, numeric_cids)

    # 곡선(학습/ROC/PR 템플릿) — ★모든 매칭은 glossary 기반★
    if _has_trigger(cidx, "viz.trigger.curve_generic", text):
        # glossary 힌트: diag/kind
        diag_true  = _has_trigger(cidx, "viz.hint.curve.diag.roc", text)
        diag_false = _has_trigger(cidx, "viz.hint.curve.diag.pr",  text)
        kind = None
        if _has_trigger(cidx, "viz.hint.curve.kind.threshold_sweep", text):
            kind = "threshold_sweep"
        elif _has_trigger(cidx, "viz.hint.curve.kind.focal_vs_ce", text):
            kind = "focal_vs_ce"
        elif _has_trigger(cidx, "viz.hint.curve.kind.map_vs_iou", text):
            kind = "map_vs_iou"
        elif _has_trigger(cidx, "viz.hint.curve.kind.calibration", text):
            kind = "calibration"

        inputs = {
            "title": "Learning Curve", "xlabel": "epoch", "ylabel": "metric",
            "series": [
                {"label": "train", "x": [1,2,3,4,5], "y": [0.9,0.7,0.6,0.5,0.45]},
                {"label": "val",   "x": [1,2,3,4,5], "y": [1.0,0.8,0.7,0.62,0.60]}
            ],
            "legend_loc": "upper right",
            "annotate_last": True,
            # caption 위치 기본값(프로젝트 공통 스타일)
            "caption_bottom": 0.10,
            "caption_y": 0.005,
        }
        # diag 우선순위: ROC 힌트가 있으면 True, PR 힌트가 있으면 False
        if diag_true:
            inputs["diag"] = True
        elif diag_false:
            inputs["diag"] = False
        # kind 힌트
        if kind:
            inputs["kind"] = kind

        spec.append({
            "id": "curve_auto",
            "type": "curve_generic",
            "labels": {"ko": "곡선(예시)"},
            "inputs": inputs
        })

    # 비교/벤치마크 → 그룹 막대
    if _has_trigger(cidx, "viz.intent.comparison", text) or _has_trigger(cidx, "viz.trigger.bar_group", text):
        cats, series = parse_benchmark(text)
        if cats and series:
            spec.append({
                "id":"benchmark_bars",
                "type":"bar_group",
                "labels":{"ko":"성능 비교"},
                "inputs":{
                    "title":{"ko":"성능 비교"},
                    "ylabel":{"ko":"점수/지표"},
                    "categories": cats,
                    "series": series,
                    "legend": True
                }
            })

    # 구성/비율 → 도넛
    if _has_trigger(cidx, "viz.intent.composition", text) or _has_trigger(cidx, "viz.trigger.donut_pct", text):
        parts = parse_composition_parts(text)
        if parts:
            spec.append({
                "id":"composition_donut",
                "type":"donut_pct",
                "labels":{"ko":"구성 비율"},
                "inputs":{"title":{"ko":"구성 비율"}, "parts": parts}
            })

    # 카테고리 분해/누적 → 스택 바
    if _has_trigger(cidx, "viz.intent.breakdown", text) or _has_trigger(cidx, "viz.trigger.stack_bar", text):
        cats, series = parse_breakdown(text)
        if cats and series:
            spec.append({
                "id":"breakdown_stack",
                "type":"stack_bar",
                "labels":{"ko":"분해 누적"},
                "inputs":{
                    "title":{"ko":"분해 누적"},
                    "ylabel":{"ko":"비율(%)"},
                    "categories": cats,
                    "series": series,
                    "normalize": True,
                    "legend_out": True
                }
            })

    # 임베딩/특징 결합 → 도식 (값 파싱 불필요)
    if _has_trigger(cidx, "viz.intent.fusion", text) or _has_trigger(cidx, "viz.trigger.embedding_sum", text):
        spec.append({
            "id":"fusion_sum",
            "type":"embedding_sum",
            "labels":{"ko":"특징/임베딩 결합"},
            "inputs":{
                "title":{"ko":"특징/임베딩 결합"},
                "rows":["Feature/Embedding A","Feature/Embedding B","(optional) C"],
                "right":"Encoder / Fusion"
            }
        })

    # 시퀀스 포맷/특수 토큰 → 도식 (문서에 실제 토큰 있을 때만)
    if _has_trigger(cidx, "viz.intent.sequence_format", text) or _has_trigger(cidx, "viz.trigger.token_sequence", text):
        tokens = parse_special_tokens(text)
        if tokens:
            spec.append({
                "id":"seq_format",
                "type":"token_sequence",
                "labels":{"ko":"시퀀스 포맷"},
                "inputs":{
                    "title":{"ko":"시퀀스 포맷"},
                    "tokens": tokens,
                    "notes":{"[CLS]":"문서 대표","[SEP]":"경계"}
                }
            })

    # 곡선(학습/ROC/PR 템플릿)
    _append_if(_has_trigger(cidx, "viz.trigger.curve_generic", text), {
        "id":"curve_auto",
        "type":"curve_generic",
        "labels":{"ko":"곡선(예시)"},
        "inputs":{"title":"Learning Curve","xlabel":"epoch","ylabel":"metric",
                "series":[
                    {"label":"train","x":[1,2,3,4,5],"y":[0.9,0.7,0.6,0.5,0.45]},
                    {"label":"val","x":[1,2,3,4,5],"y":[1.0,0.8,0.7,0.62,0.60]}
                ],
                "legend_loc":"upper right","annotate_last":True}
    })

    # 분포/히스토그램: 의도 트리거 + 값 파싱 성공 시에만
    vals = parse_hist_values(text)
    _append_if(
        _has_trigger(cidx, "viz.intent.distribution", text) and bool(vals),
        {
            "id": "dist_hist",
            "type": "histogram",
            "labels": _label("Histogram", "히스토그램"),
            "inputs": {
                "values": vals,
                "bins": "fd",  # Freedman–Diaconis
                "xlabel": _label("value", "값"),
                "title":  _label("Histogram", "히스토그램")
            }
        }
    )
        # values가 없으면 아무 것도 append 하지 않음 (예시/템플릿 금지)

    # (E) 개념/예시 도식들
    spec += build_concept_specs(text, spec, mentions, numeric_cids)

    # (F) 단일 포인트 학습곡선(옵션: 앞단 보정)
    m = re.search(
        r"(?:epoch|에폭)\s*(\d+)[^\n]{0,40}?"
        r"(?:acc(?:uracy)?|정확도)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(%)?",
        text, re.I
    )
    if m:
        ep   = int(m.group(1))
        val  = float(m.group(2))
        y    = val / 100.0 if m.group(3) else val
        points = [(ep, y)]
        xs, ys, mask, imputed = _build_series_with_front_impute(points, "Model A", text, "정확도")
        spec.append({
            "id": "train_curve",
            "type": "curve_generic",
            "labels": _label("Learning curve", "학습 곡선"),
            "inputs": {
                "series": [{"x": xs, "y": ys, "label": _label("Model A", "모델 A"),
                            "observed_mask": mask}],
                "xlabel": _label("Epoch", "에폭"),
                "ylabel": _label("Accuracy", "정확도"),
                "title": _label("Learning curve" + (" (예시 보정)" if imputed else ""),
                                "학습 곡선" + (" (예시 보정)" if imputed else ""))
            }
        })

    return _dedup(spec)
