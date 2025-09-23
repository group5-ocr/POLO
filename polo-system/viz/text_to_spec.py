"""
텍스트 → spec 자동 빌더 (glossary 기반)
- 지표 탐지/정규화/바차트/루브릭 테이블
- 히스토그램은 glossary의 viz.trigger.histogram로만 생성
- 나머지 개념 도식은 templates/generic_rules.build_concept_specs 에서 추가
"""
from __future__ import annotations
import re, json, math, statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 개념/예시 도식 트리거
from templates.generic_rules import build_concept_specs

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
    # glossary 레코드를 concept_id → 메타 dict로 인덱싱
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

# 유니코드 지수/마이너스 정규화
_SUP_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

def _normalize_superscript(expr: str) -> str:
    # 10⁻⁶ → 10^-6 , 유니코드 마이너스(−) → -, 슈퍼스크립트 숫자 → 일반 숫자
    s = expr.replace("−", "-").replace("⁻", "^-")
    return s.translate(_SUP_DIGITS)

def _parse_ber_value(token: str) -> float | None:
    t = _normalize_superscript(token.strip())
    # 10^-6
    m = re.search(r"10\s*\^\s*(-?\d+)", t)
    if m:
        try:
            return 10.0 ** (int(m.group(1)))
        except Exception:
            pass
    # 1e-6, 3.2e-5
    m = re.search(r"(\d+(?:\.\d+)?)\s*[eE]\s*(-?\d+)", t)
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except Exception:
            pass
    # 소수 직접
    try:
        v = float(t)
        return v if 0 < v < 1 else None
    except Exception:
        return None

# BER vs SNR spec 빌더
def build_spec_ber_snr(text: str) -> dict:
    T = text
    pairs_dc, pairs_mod = [], []

    rx1 = re.finditer(
        r"(?P<snr>\d+(?:\.\d+)?)\s*dB.{0,600}?(?:BER|오류\s*확률)\s*[:=]?\s*(?P<ber>10\s*\^\s*-?\d+|10⁻\d+|\d+(?:\.\d+)?[eE]-?\d+|\d*(?:\.\d+)?)",
        T, re.I | re.S
    )
    rx2 = re.finditer(
        r"(?:BER|오류\s*확률)\s*[:=]?\s*(?P<ber>10\s*\^\s*-?\d+|10⁻\d+|\d+(?:\.\d+)?[eE]-?\d+|\d*(?:\.\d+)?)\s*(?:at|@|에서)?.{0,600}?SNR\s*[:=]?\s*(?P<snr>\d+(?:\.\d+)?)\s*dB",
        T, re.I | re.S
    )

    for m in list(rx1) + list(rx2):
        snr = float(m.group("snr"))
        ber = _parse_ber_value(m.group("ber"))
        if not ber:
            continue
        ctx = T[max(0, m.start()-120): min(len(T), m.end()+120)].lower()
        if "deepcode" in ctx:
            pairs_dc.append([snr, ber])
        elif ("modulo-sk" in ctx) or ("모듈로" in ctx):
            pairs_mod.append([snr, ber])
        else:
            pairs_mod.append([snr, ber])

    def _uniq_sorted(pairs):
        seen, out = set(), []
        for x, y in pairs:
            k = (round(x, 3), round(y, 12))
            if k in seen:
                continue
            seen.add(k)
            out.append([x, y])
        return sorted(out, key=lambda t: t[0])

    pairs_dc, pairs_mod = _uniq_sorted(pairs_dc), _uniq_sorted(pairs_mod)
    if not (pairs_dc or pairs_mod):
        return {}
    series = []
    if pairs_dc:
        series.append({"name": "Deepcode", "points": pairs_dc})
    if pairs_mod:
        series.append({"name": "Modulo-SK", "points": pairs_mod})
    return {
        "series": series,
        "title": "BER vs SNR",
        "x_label": "SNR (dB)",
        "y_label": "BER",
        "y_log": True
    }

# BER vs Rounds spec 빌더
def build_spec_ber_rounds(text: str) -> dict:
    T = text
    pairs_dc, pairs_mod = [], []

    N_PAT = r"(?:N\s*=\s*(?P<n1>\d+)|(?P<n2>\d+)\s*(?:rounds?|라운드|회))"
    B_PAT = r"(?:BER|오류\s*확률)\s*[:=]?\s*(?P<ber>10\s*\^\s*-?\d+|10⁻\d+|\d+(?:\.\d+)?[eE]-?\d+|\d*(?:\.\d+)?)"

    rx1 = re.finditer(fr"{N_PAT}.{{0,600}}?{B_PAT}", T, re.I | re.S)
    rx2 = re.finditer(fr"{B_PAT}.{{0,600}}?{N_PAT}", T, re.I | re.S)

    for m in list(rx1) + list(rx2):
        n = int((m.group("n1") or m.group("n2") or "0"))
        ber = _parse_ber_value(m.group("ber"))
        if not ber:
            continue
        ctx = T[max(0, m.start()-120): min(len(T), m.end()+120)].lower()
        if "deepcode" in ctx:
            pairs_dc.append([n, ber])
        elif ("modulo-sk" in ctx) or ("모듈로" in ctx):
            pairs_mod.append([n, ber])
        else:
            pairs_mod.append([n, ber])

    def _uniq_sorted(pairs):
        seen, out = set(), []
        for x, y in pairs:
            k = (int(x), round(y, 12))
            if k in seen:
                continue
            seen.add(k)
            out.append([x, y])
        return sorted(out, key=lambda t: t[0])

    pairs_dc, pairs_mod = _uniq_sorted(pairs_dc), _uniq_sorted(pairs_mod)
    if not (pairs_dc or pairs_mod):
        return {}
    series = []
    if pairs_dc:
        series.append({"name": "Deepcode", "points": pairs_dc})
    if pairs_mod:
        series.append({"name": "Modulo-SK", "points": pairs_mod})
    return {
        "series": series,
        "title": "BER vs Rounds",
        "x_label": "Rounds (N)",
        "y_label": "BER",
        "y_log": True
    }

# --- 트리거 체크 유틸 (그대로 사용) ---
def _has_trigger(concept_idx, key: str, text: str) -> bool:
    meta = concept_idx.get(key)
    if meta and meta.get("patterns"):
        return any(p.search(text) for p in meta["patterns"])
    for _, m in concept_idx.items():
        if m.get("category") == key:
            for p in m.get("patterns", []):
                if p.search(text):
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

# 산점도 전용 숫자 파서
_SCAT_PAIR = re.compile(r'\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)')
_SCAT_X    = re.compile(r'(?:^|\b)x\s*[:=]\s*\[\s*([^\]]+)\s*\]', re.I)
_SCAT_Y    = re.compile(r'(?:^|\b)y\s*[:=]\s*\[\s*([^\]]+)\s*\]', re.I)

def _to_float_list_csv(s: str):
    return [float(t) for t in re.findall(r'[-+]?\d+(?:\.\d+)?', s or '')]

def parse_scatter_points(text: str, min_n: int = 10):
    """
    (x,y) 쌍 또는 x:[..], y:[..] 병렬 리스트를 찾아 반환.
    - 최소 포인트 미만이면 빈 리스트 반환
    - 수평/수직/단일점(변동 없음) 형태면 빈 리스트 반환
    """
    xs, ys = [], []

    # (x, y) 쌍
    for m in _SCAT_PAIR.finditer(text):
        xs.append(float(m.group(1))); ys.append(float(m.group(2)))

    # x:[..], y:[..] 병렬 리스트
    if len(xs) < min_n:
        mx = _SCAT_X.search(text); my = _SCAT_Y.search(text)
        if mx and my:
            xlist = _to_float_list_csv(mx.group(1))
            ylist = _to_float_list_csv(my.group(1))
            if len(xlist) == len(ylist):
                xs, ys = xlist, ylist

    # 기본 유효성 체크
    if len(xs) >= min_n:
        try:
            if len(set(xs)) <= 1 or len(set(ys)) <= 1:
                return [], []
        except Exception:
            return [], []
        return xs, ys

    return [], []

def _L(en, ko):
    return {"en": en, "ko": ko}

def pick_scatter_axis_labels_from_glossary(text: str, concept_idx):
    # 매칭되는 첫 쌍을 적용. 없으면 x/y 기본값을 반환.
    xlab = _L("x", "x")
    ylab = _L("y", "y")

    for cid, meta in concept_idx.items():
        if meta.get("category") != "viz.trigger.scatter":
            continue
        entry = meta.get("entry", {})
        pats  = meta.get("patterns", [])
        for p in pats:
            if p.search(text):
                al = entry.get("axis_labels", {})
                if isinstance(al.get("x"), dict) and isinstance(al.get("y"), dict):
                    return al["x"], al["y"]
                # axis_labels가 없으면 다음 매칭 후보로 넘어감
                break
    return xlab, ylab

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

# 평가지표 테이블
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
    "metric.map_detection": {"mode": "up", "title": "mAP rubric",
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

def collect_metric_keyword_hits(text: str, concept_idx) -> set[str]:
    """숫자 없이 '지표 키워드'만 매칭된 CID 집합을 수집."""
    hits = set()
    for cid, meta in concept_idx.items():
        if meta.get("category") != "metric":
            continue
        for pat in meta["patterns"]:
            if pat.search(text):
                hits.add(cid); break
    return hits

# 평가지표 하단 설명
RUBRIC_CAPTIONS = {
    "metric.accuracy": "정확도=(TP+TN)/전체. 클래스 불균형 상황에선 보조 지표와 함께 해석.",
    "metric.precision": "Precision=TP/(TP+FP). 적중의 순도. 낮으면 오탐이 많음.",
    "metric.recall": "Recall=TP/(TP+FN). 놓침 방지. 낮으면 미탐이 많음.",
    "metric.f1": "F1=2·(Prec·Rec)/(Prec+Rec). 정밀도·재현율의 균형.",
    "metric.auroc": "AUROC: 임계 전 범위에서 TPR 대 FPR 면적. 0.5는 무작위.",
    "metric.auprc": "AUPRC: 클래스 희소 시 해석 유리. 기준선≈양성 비율.",
    "metric.map_detection": "COCO mAP: IoU 0.50–0.95(0.05 간격) AP 평균.",
    "metric.average_recall": "AR: 다수 임계에서의 평균 재현율.",
    "metric.miou": "mIoU: 교집합/합집합의 클래스 평균. 세그멘테이션 품질.",
    "metric.iou": "IoU: 교집합/합집합. 1에 가까울수록 정확.",
    "metric.dice": "Dice=2·|∩|/(|A|+|B|). IoU와 유사, 민감도 높음.",
    "metric.pq": "PQ: Panoptic Quality. 세그멘트 매칭+정확도 결합.",
    "metric.fid": "FID↓: 생성 분포와 실제 분포의 거리. 낮을수록 좋음.",
    "metric.kid": "KID↓: 커널 기반 분포 거리. 낮을수록 좋음.",
    "metric.lpips": "LPIPS↓: 지각적 거리. 낮을수록 유사.",
    "metric.inception_score": "IS: 클래스 다양성과 확신을 함께 반영.",
    "metric.wer": "WER↓: 단어 오류율. 낮을수록 좋음.",
    "metric.cer": "CER↓: 문자 오류율. 낮을수록 좋음.",
}

def _add_rubric_tables(spec: list, mentions: dict,
                    numeric_cids: set[str] | None = None):
    """
    숫자 유무와 무관하게 '언급된 지표'에 대해 1회만 표 생성.
    - 상단 헤드라인/수치(예: VOC2012: …)는 넣지 않음
    - 캡션은 지표별 설명을 붙이고, 간격을 좁히기 위한 파라미터 전달
    """
    if numeric_cids is None:
        # 기본: 숫자 없이 키워드만 매칭된 CID 들
        numeric_cids = set(mentions.keys())

    def _append_table(cid: str, rub: dict):
        title_en = rub.get("title", cid)
        if cid == "metric.map_detection":
            title_en = "mAP rubric"
        title_ko = title_en.replace("rubric", "해석")

        rows   = [name for _, name in rub["thr"]]
        values = [[round(th, 3)] for th, _ in rub["thr"]]

        item = {
            "id": f"{cid.split('.')[-1]}_rubric",
            "type": "metric_table",
            "labels": {"en": title_en, "ko": title_ko},
            "caption_labels": {"ko": RUBRIC_CAPTIONS.get(cid, "")},
            "inputs": {
                "methods": rows,
                "metrics": [rub["col"]],
                "values":  values,
                "title":   {"en": title_en, "ko": title_ko},
                # 간격
                "caption_bottom": 0.14,
                "caption_y": 0.02,      
            }
        }
        spec.append(item)

    for cid in sorted(numeric_cids):
        rub = RUBRICS.get(cid)
        if rub:
            _append_table(cid, rub)

# 학습 곡선 빌더
MIN_POINTS_FOR_CURVE = 3

def _dedup(spec):
    seen, out = set(), []
    for it in spec:
        if it["id"] in seen:
            continue
        seen.add(it["id"]); out.append(it)
    return out

def _to_id_set(mentions) -> set:
    """mentions가 dict/list/tuple 어느 형태든 안전하게 id set으로 변환."""
    if isinstance(mentions, dict):
        return set(map(str, mentions.keys()))
    if isinstance(mentions, (list, tuple, set)):
        return set(map(str, mentions))
    try:
        return set(map(str, mentions or []))
    except Exception:
        return set()

def _norm_value(v):
    """중복 판정을 위한 정규화(더 강하게)."""
    if isinstance(v, (int, float)):
        return round(float(v), 3)              # 소수 3자리
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        return tuple(_norm_value(x) for x in v)
    if isinstance(v, dict):
        # 키 순서 영향 제거
        return tuple(sorted((str(k), _norm_value(x)) for k, x in v.items()))
    return v

def _canonical_bar_series(series):
    """
    bar_group의 series를 시그니처 계산/스케일 보정이 쉬운 통일 구조로 변환.
    반환: ("dict", {name: [values...]}) 또는 ("list", [{"name":..., "values":[...]}])
    """
    if isinstance(series, dict):
        # {"A":[...], "B":[...]}
        return "dict", {str(k): list(v) if isinstance(v,(list,tuple)) else v for k, v in series.items()}
    if isinstance(series, list):
        # [{"name":"A","values":[...]} ...]
        out = []
        for s in series:
            if isinstance(s, dict):
                name = str(s.get("name", ""))
                vals = s.get("values")
                if isinstance(vals, (list, tuple)):
                    out.append({"name": name, "values": list(vals)})
        return "list", out
    return "unknown", series

def _looks_like_ratio_0to1(vals):
    try:
        nums = [float(x) for x in vals if isinstance(x, (int, float))]
        if not nums:
            return False
        mn, mx = min(nums), max(nums)
        # (0,1) 사이 값이 하나라도 있고 전체가 0~1 사이면 비율로 간주
        return (0 <= mn <= 1) and (0 <= mx <= 1) and any(0 < x < 1 for x in nums)
    except Exception:
        return False

def _scale_bar_group_percent_if_needed(item: dict):
    """bar_group 값이 0~1이면 100배 + y라벨 (%) 보정. dict/list series 모두 지원."""
    if item.get("type") != "bar_group":
        return item
    inp = item.get("inputs", {}) or {}
    mode, canon = _canonical_bar_series(inp.get("series"))

    # 모든 값 수집
    all_vals = []
    if mode == "dict":
        for vals in canon.values():
            if isinstance(vals, list):
                all_vals += [x for x in vals if isinstance(x, (int, float))]
    elif mode == "list":
        for s in canon:
            all_vals += [x for x in s.get("values", []) if isinstance(x, (int, float))]
    else:
        return item

    if not all_vals or not _looks_like_ratio_0to1(all_vals):
        return item

    # 100배 보정
    if mode == "dict":
        new_series = {}
        for name, vals in canon.items():
            if isinstance(vals, list):
                new_series[name] = [(float(x) * 100.0) for x in vals]
            else:
                new_series[name] = vals
        inp["series"] = new_series
    elif mode == "list":
        new_series = []
        for s in canon:
            vals = s.get("values", [])
            new_series.append({"name": s.get("name",""), "values": [(float(x) * 100.0) for x in vals]})
        inp["series"] = new_series

    yl = inp.get("y_label") or inp.get("ylabel") or ""
    if "%" not in str(yl):
        inp["y_label"] = (str(yl) + " (%)").strip() if yl else "정규화 점수(%)"
    item["inputs"] = inp
    return item

def _ensure_activation_fallback(mentions_set: set, spec_list: list):
    """
    activation_panel 트리거가 '실제'로 잡혔는데, 활성화 관련 스펙이 하나도 없으면
    안전한 curve_generic 1장을 추가. (트리거가 없으면 절대 추가 안 함)
    """
    if "viz.trigger.activation_panel" not in mentions_set:
        return spec_list
    if any(s.get("type") in {"activation_curve", "curve_generic"} for s in (spec_list or [])):
        return spec_list

    xs = list(range(-6, 7))
    def _sigmoid(x): 
        import math
        return 1.0 / (1.0 + math.e**(-x))

    spec_list = list(spec_list or [])
    spec_list.append({
        "type": "curve_generic",
        "labels": {"ko": "활성화 함수 예시", "en": "Activation examples"},
        "inputs": {
            "series": [
                {"name": "Sigmoid", "x": xs, "y": [_sigmoid(x) for x in xs]}
            ],
            "ylabel": {"ko": "출력", "en": "Output"},
            # caption은 grammar가 없으면 무시해도 됨
            "caption": {"ko": "여러 시리즈 비교는 범례 참고", "en": "Multiple series: see legend"}
        }
    })
    return spec_list

SEEN_GLOBAL_TYPES = set()
SEEN_METRIC_SIGNATURES = set()   # (type, inputs 시그니처) 전역 중복 제거

def postprocess_specs(text: str, spec_list: list, mentions) -> list:
    global SEEN_GLOBAL_TYPES, SEEN_METRIC_SIGNATURES

    ids = _to_id_set(mentions)

    allow_cell_scale = True

    EXAMPLE_ONE_SHOT_TYPES = {
        # 예시/개념으로만 한정 (필요 최소만 유지)
        "cell_scale",        # 같은 예시 반복 방지
        # "activation_curve", "curve_generic" 는 실제 데이터 기반일 수 있어 제외
        # generic concept 도식 타입이 따로 있다면 여기에 추가
    }

    METRIC_CONTENT_TYPES = {
        "metric_table", "bar_group", "donut_pct",
        "stack_bar", "histogram", "dist_scatter",
        "ber_vs_snr", "ber_vs_rounds",
        "curve_generic", "iou_overlap"
    }

    seen_sig = set()  # (문단 내) 동일 시그니처 중복 제거
    out = []

    for it in (spec_list or []):
        if not isinstance(it, dict):
            continue

        t = it.get("type")
        inp = it.get("inputs", {}) or {}

        # cell_scale 트리거 없으면 제외
        if t == "cell_scale" and not allow_cell_scale:
            continue

        # bar_group 0~1 → % 보정
        if t == "bar_group":
            it = _scale_bar_group_percent_if_needed(it)
            inp = it.get("inputs", {}) or {}

        # 예시/개념: 타입 단위 전역 1회
        if t in EXAMPLE_ONE_SHOT_TYPES:
            if t in SEEN_GLOBAL_TYPES:
                continue
            SEEN_GLOBAL_TYPES.add(t)

        # 지표/표/그래프: 내용이 같으면 전역 1회
        if t in METRIC_CONTENT_TYPES:
            sig_global = (t, _norm_value(inp))  # 내용 시그니처(정규화 입력값)
            if sig_global in SEEN_METRIC_SIGNATURES:
                continue
            SEEN_METRIC_SIGNATURES.add(sig_global)

        # (문단 범위) 동일 시그니처 중복 제거
        sig_local = (t, _norm_value(inp))
        if sig_local in seen_sig:
            continue
        seen_sig.add(sig_local)

        out.append(it)

    # activation fallback (트리거 있을 때만, 이미 있으면 추가 X)
    out = _ensure_activation_fallback(ids, out)
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

    # 평가지표
    mentioned_cids = collect_metric_keyword_hits(text, cidx)
    _add_rubric_tables(spec, mentions, numeric_cids=mentioned_cids)

    # 산점도: 트리거가 있어도 '실데이터'가 없으면 생성하지 않음(예시/더미 금지)
    if _has_trigger(cidx, "viz.trigger.scatter", text):
        xs, ys = parse_scatter_points(text, min_n=10)
        if xs and ys:
            # 축 라벨: 사전에서 별도 축-키를 안 쓰는 경우 기본값 (x/y)로만
            xlab = {"en": "x", "ko": "x"}
            ylab = {"en": "y", "ko": "y"}

            spec.append({
                "id": "ratios_scatter",       
                "type": "dist_scatter",
                "labels": {"ko": "산점도", "en": "Scatter"},
                "inputs": {
                    "x": xs,
                    "y": ys,
                    "xlabel": xlab,
                    "ylabel": ylab,
                    "title": {"ko": "산점도", "en": "Scatter"}
                }
            })
        # else: 데이터 부족 → 추가하지 않음

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

    # 임베딩/특징 결합 → 도식 (흐름이 있을 때만)
    if _has_trigger(cidx, "viz.intent.fusion", text) or _has_trigger(cidx, "viz.trigger.embedding_sum", text):
        # 결합/합치기 의사어휘가 실제로 등장?
        has_action = bool(re.search(r"(합치|결합|sum|add|concat|merge|fuse)", text, re.I))
        # 따옴표/대괄호/백틱 안의 명칭을 후보로 수집
        feats = re.findall(r"[‘'\"`]{1}([^‘'\"`]{1,40})[’'\"`]{1}", text)
        if not feats:
            feats = re.findall(r"\[([^\[\]\n]{1,40})\]", text)
        feats = [f.strip() for f in feats if f.strip()]

        if has_action and len(feats) >= 2:
            spec.append({
                "id":"fusion_sum",
                "type":"embedding_sum",
                "labels":{"ko":"특징/임베딩 결합"},
                "inputs":{
                    "title":{"ko":"특징/임베딩 결합"},
                    "rows": feats[:3],
                    "right":"Encoder / Fusion"
                }
            })

    # 시퀀스 포맷/특수 토큰 → 도식 (흐름이 있을 때만)
    if _has_trigger(cidx, "viz.intent.sequence_format", text) or _has_trigger(cidx, "viz.trigger.token_sequence", text):
        tokens = parse_special_tokens(text)
        # 경계/순서 기호가 실제 텍스트에 있는지 확인
        has_flow = any(sym in text for sym in ["[CLS]","[SEP]","<bos>","<eos>","→","->","|"])
        if tokens and len(tokens) >= 3 and has_flow:
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

    # 개념/예시 도식들
    try:
        spec += build_concept_specs(text, spec, mentions, numeric_cids, concept_idx=cidx)
    except Exception as e:
        print(f"[WARN] build_concept_specs skipped: {e}")

    # 학습곡선/ROC/PR 등 curve_generic — 실제 수치가 여러 개 있을 때만
    # 더미/임퓨트 금지
    # 용어사전 트리거(의도) + 최소 3개 포인트 필요
    if _has_trigger(cidx, "viz.trigger.curve_generic", text) or _has_trigger(cidx, "viz.intent.curve_generic", text):

        # (epoch, accuracy) 페어 추출 — 주변이 '예시/샘플/example/dummy'면 버림
        pts = []
        for m in re.finditer(
            r"(?:epoch|에폭)\s*(\d+)[^\n]{0,40}?(?:acc(?:uracy)?|정확도)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(%)?",
            text, re.I
        ):
            ep   = int(m.group(1))
            val  = float(m.group(2))
            y    = val / 100.0 if m.group(3) else val

            ctx = text[max(0, m.start()-40): min(len(text), m.end()+40)].lower()
            if re.search(r"예시|샘플|example|e\.g\.|dummy", ctx):
                continue  # 문맥이 예시면 제외

            pts.append((ep, y))

        # 중복 epoch 제거 + 정렬 + 기본 유효성(포인트≥3, epoch 증가)
        pts_dict = {}
        for ep, y in pts:
            pts_dict[ep] = y
        points = sorted(pts_dict.items(), key=lambda t: t[0])

        if len(points) >= 3 and all(points[i][0] < points[i+1][0] for i in range(len(points)-1)):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            spec.append({
                "id": "train_curve_points",
                "type": "curve_generic",
                "labels": _label("Learning curve", "학습 곡선"),
                "inputs": {
                    "series": [{"x": xs, "y": ys, "label": _label("Model", "모델")}],
                    "xlabel": _label("Epoch", "에폭"),
                    "ylabel": _label("Accuracy", "정확도"),
                    "title":  _label("Learning curve", "학습 곡선"),
                    # 캡션/기타 옵션은 glossary 힌트로만 추가(예: diag/kind)
                }
            })
            # 조건을 못 만족하면 아무 것도 추가하지 않음 (더미/예시 차트 생성 금지)

    # BER vs SNR (trigger: viz.trigger.ber_snr)
    if _has_trigger(cidx, "viz.trigger.ber_snr", text):
        snr_spec = build_spec_ber_snr(text)
        if snr_spec.get("series"):
            spec.append({
                "id": "ber_vs_snr",
                "type": "ber_vs_snr",
                "labels": {"ko": "BER 대 SNR", "en": "BER vs SNR"},
                "inputs": snr_spec
            })

    # BER vs Rounds
    if _has_trigger(cidx, "viz.trigger.ber_rounds", text):
        r_spec = build_spec_ber_rounds(text)
        if r_spec.get("series"):
            spec.append({
                "id": "ber_vs_rounds",
                "type": "ber_vs_rounds",
                "labels": {"ko": "BER 대 라운드수", "en": "BER vs Rounds"},
                "inputs": r_spec
            })
    # postprocess에서 트리거/중복/보정 처리
    misc = set()
    if _has_trigger(cidx, "viz.trigger.cell_scale", text):
        misc.add("viz.trigger.cell_scale")
    if _has_trigger(cidx, "viz.trigger.activation_panel", text):
        misc.add("viz.trigger.activation_panel")

    return postprocess_specs(text, spec, mentions)
