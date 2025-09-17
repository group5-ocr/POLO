# 개념/예시 도식 빌더
# 텍스트에 '수치'가 없어도, 의미 있는 트리거가 있으면 예시 도식 생성

from __future__ import annotations
import re, math, random, hashlib
from typing import List, Dict, Any

def _label(en, ko):
    return {"en": en, "ko": ko}

# 시드 고정 랜덤
import hashlib
def _stable_seed(text: str, salt: str = "") -> int:
    h = hashlib.sha256((salt + "|" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)
def _rng_from(text: str, salt: str = "") -> random.Random:
    return random.Random(_stable_seed(text, salt))
def _clip(v, lo=0.05, hi=0.95): return max(lo, min(hi, v))

# 헬퍼
def _nums(text):
    return [float(x.replace(',', '')) for x in re.findall(r'-?\d+(?:\.\d+)?', text)]

def _parse_arrow_chain(text: str) -> list[str]:
    seps = r'(?:->|→|⇒|⟶)'
    best = []
    for line in text.splitlines():
        if re.search(seps, line):
            s = re.sub(r'\s*(->|→|⇒|⟶)\s*', '->', line)
            toks = [t.strip(' []()') for t in s.split('->') if t.strip()]
            toks = [t for t in toks if 2 <= len(t) <= 40]
            if len(toks) >= 3 and len(toks) > len(best):
                best = toks
    return best

def _extract_confusion_2x2(text: str):
    m = re.search(r'(?:confusion\s*matrix|혼동\s*행렬)(.{0,120}?)(-?\d+(?:\.\d+)?)[^\d]+'
                r'(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)',
                text, re.I | re.S)
    if not m:
        return None
    a, b, c, d = map(float, m.groups()[-4:])
    return [[a, b], [c, d]]

def _extract_points_2d(text: str):
    pts = []
    for m in re.finditer(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', text):
        x, y = float(m.group(1)), float(m.group(2))
        pts.append({"x": x, "y": y})
    return pts

def _has_wh_evidence(text: str) -> bool:
    rx = [
        r"\b\d+\s*[×x]\s*\d+\b",                    # 640x480
        r"\b\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?\b",# 0.42×0.31
        r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)", # (0.6,0.4)
        r"\b(?:aspect|비율)\s*\d+\s*:\s*\d+\b",
        r"width\s*[:=]\s*\d+(?:\.\d+)?",
        r"height\s*[:=]\s*\d+(?:\.\d+)?",
        r"폭\s*[:=]\s*\d+(?:\.\d+)?|높이\s*[:=]\s*\d+(?:\.\d+)?",
    ]
    t = text
    return any(re.search(p, t, re.I) for p in rx)

def _ratios_from_text(text: str, k_range=(2,5), n_total=300):
    rnd = _rng_from(text, "wh-mixture")
    k = rnd.randint(k_range[0], k_range[1])
    pts = []
    centers = [(rnd.uniform(0.15,0.85), rnd.uniform(0.15,0.85)) for _ in range(k)]
    for (cx,cy) in centers:
        angle = rnd.uniform(0, math.pi)
        a, b  = rnd.uniform(0.03,0.08), rnd.uniform(0.02,0.06)
        ca, sa = math.cos(angle), math.sin(angle)
        for _ in range(max(40, n_total//k)):
            u, v = rnd.gauss(0,1), rnd.gauss(0,1)
            x = cx + a*ca*u - b*sa*v
            y = cy + a*sa*u + b*ca*v
            pts.append([_clip(round(x,3)), _clip(round(y,3))])
    for _ in range(max(5, int(0.05*n_total))):
        pts.append([_clip(rnd.uniform(0.05,0.95)), _clip(rnd.uniform(0.05,0.95))])
    rnd.shuffle(pts)
    return pts[:n_total]

def _read_k_target(text: str, default: int = 5):
    pats = [
        r"\bk\s*[:=]\s*(\d+)",
        r"\bk\s*[≈~]\s*(\d+)",
        r"\bk\s*(?:값|value)\s*(?:은|=)\s*(\d+)",
        r"\bk\s*는\s*(\d+)",
    ]
    for p in pats:
        m = re.search(p, text, re.I)
        if m:
            k = int(m.group(1))
            return max(1, min(10, k)), True
    return default, False

def _k_curve_from_text(text: str):
    rnd   = _rng_from(text, "kcurve")
    ks    = list(range(1, 11))
    kstar, has_hint = _read_k_target(text, 5)

    y0         = 0.45 + 0.04 * rnd.random()
    y_plateau  = 0.80 + 0.06 * rnd.random()
    pre_slope  = 0.025 + 0.008 * rnd.random()
    post_slope = 0.006 + 0.004 * rnd.random()

    ys, y = [], y0
    for k in ks:
        if k < kstar:
            y += pre_slope + rnd.uniform(-0.004, 0.004)
        elif k == kstar:
            y += (pre_slope * 2.5) + rnd.uniform(0.01, 0.02)
        else:
            y += post_slope + rnd.uniform(-0.003, 0.003)
        y = min(y, y_plateau + rnd.uniform(-0.01, 0.01))
        ys.append(round(_clip(y), 3))
    return ks, ys, kstar, has_hint

def _extract_grids(text: str) -> list[tuple[int,int]]:
    grids = []
    for m in re.finditer(r"(\d{1,3})\s*[×x]\s*(\d{1,3})", text, re.I):
        gw, gh = int(m.group(1)), int(m.group(2))
        if 1 <= gw <= 256 and 1 <= gh <= 256:
            grids.append((gw, gh))
    grids = list(dict.fromkeys(grids))
    if len(grids) >= 2:
        grids.sort(key=lambda g: g[0]*g[1])
        return [grids[0], grids[-1]]
    return grids

def _extract_img_wh(text: str):
    m = re.search(r"(\d{2,4})\s*[×x]\s*(\d{2,4})(?:\s*px|\s*픽셀|\s*pixels)?", text, re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return [640, 640]

def _activation_mentions(text: str) -> list[dict]:
    text_l = text.lower()
    RX = {
        "ReLU":       [r"\brelu\b", r"렐루"],
        "LeakyReLU":  [r"leaky\s*relu|leakyrelu", r"리키\s*렐루|리키렐루", r"누수\s*relu"],
        "PReLU":      [r"\bprelu\b", r"parametric\s*relu", r"파라메트릭\s*relu|프리루"],
        "Tanh":       [r"\btanh\b", r"탄슈|탄흐|탄하"],
        "Sigmoid":    [r"\bsigmoid\b", r"시그모이드"],
        "ELU":        [r"\belu\b", r"엘루"],
        "SELU":       [r"\bselu\b", r"세루"],
        "GELU":       [r"\bgelu\b", r"겔루|젤루"],
        "SiLU":       [r"\bsilu\b|\bswish\b", r"스위시|시루"],
        "Softplus":   [r"\bsoft\s*plus\b|\bsoftplus\b", r"소프트\s*플러스"],
        "Mish":       [r"\bmish\b", r"미시"],
        "HardSigmoid":[r"hard\s*sigmoid|hardsigmoid", r"하드\s*시그모이드"],
        "HardSwish":  [r"hard\s*swish|hardswish", r"하드\s*스위시"],
        "HardTanh":   [r"hard\s*tanh|hardtanh", r"하드\s*(tanh|탄하|탄흐|탄슈)"],
        "ReLU6":      [r"\brelu6\b", r"렐루6"],
    }
    NUM = r"[-+]?\d+(?:\.\d+)?"
    def _param_near(pos: int, keys=("alpha","β","beta","slope","negative\s*slope")) -> dict:
        win = 48
        s = max(0, pos-win); e = min(len(text), pos+win)
        chunk = text[s:e]
        out = {}
        for k in keys:
            m = re.search(fr"{k}\s*[:=]?\s*({NUM})", chunk, re.I)
            if m:
                val = float(m.group(1))
                if "beta" in k or "β" in k:
                    out["beta"] = val
                elif "slope" in k:
                    out["alpha"] = val
                else:
                    out["alpha"] = val
        return out

    funcs: list[dict] = []
    used = set()
    for name, pats in RX.items():
        for pat in pats:
            m = re.search(pat, text_l, re.I)
            if m:
                if name in used: break
                params = _param_near(m.start())
                if name == "LeakyReLU" and "alpha" not in params: params["alpha"] = 0.2
                if name == "PReLU"     and "alpha" not in params: params["alpha"] = 0.25
                if name == "ELU"       and "alpha" not in params: params["alpha"] = 1.0
                if name == "SiLU"      and "beta"  not in params: params["beta"]  = 1.0
                funcs.append({"name": name, **params})
                used.add(name)
                break
    return funcs

def _parse_taus_from_text(text: str) -> list[float]:
    NUM = r"[0-9]+(?:\.[0-9]+)?"
    pats = [
        rf"τ\s*[:=]\s*({NUM})",
        rf"\btau\b\s*[:=]\s*({NUM})",
        rf"\bT\b\s*[:=]\s*({NUM})",
        rf"temperature\s*[:=]?\s*({NUM})",
    ]
    vals = []
    for p in pats:
        for m in re.finditer(p, text, flags=re.I):
            try:
                v = float(m.group(1))
                if v > 0: vals.append(v)
            except:
                pass
    return sorted(set(round(v, 3) for v in vals))

# 빌더
def build_concept_specs(text: str, spec: list,
                        mentions: dict,
                        numeric_cids: set[str] | None = None) -> list:
    numeric_cids = numeric_cids or set()
    T  = lambda rx: re.search(rx, text, re.I | re.S) is not None

    # (w,h) 산점
    if T(r"width.*height|w[, ]?h|가로.*세로|비율"):
        has_num = _has_wh_evidence(text)
        spec.append({
            "id": "ratios_scatter",
            "type": "dist_scatter",
            "labels": _label("(w,h) Scatter", "(w,h) 산점"),
            "inputs": {
                "points": _ratios_from_text(text, k_range=(2,5), n_total=300),
                "xlabel": _label("width (norm)", "폭 (정규화)"),
                "ylabel": _label("height (norm)", "높이 (정규화)"),
                "title":  _label("(w,h) Scatter" + ("" if has_num else " (예시)"),
                                "(w,h) 산점" + ("" if has_num else " (예시)")),
                "example_badge": (not has_num)
            }
        })

    # k 튜닝 곡선
    if T(r"\bk[-\s]*means|\bk\s*(?:=|≈|~)\s*\d+|cluster|클러스터|튜닝"):
        ks, ys, kstar, has_hint = _k_curve_from_text(text)
        suffix = "" if has_hint else " (예시)"
        spec.append({
            "id": "k_vs_quality",
            "type": "curve_generic",
            "labels": _label("k tuning curve", "k 튜닝 곡선"),
            "inputs": {
                "series": [{"x": ks, "y": ys, "label": _label("avg quality", "평균 품질")}],
                "xlabel": _label("k (clusters)", "k (클러스터 수)"),
                "ylabel": _label("avg overlap/IoU", "평균 겹침/IoU"),
                "title":  _label(f"k tuning curve (k≈{kstar}){suffix}",
                                f"k 튜닝 곡선 (k≈{kstar}){suffix}")
            }
        })

    # 셀 스케일/앵커 개념
    if T(r"(cell|셀|격자|grid).*(0\s*~\s*1|0~1|sigmoid|시그모이드)"):
        grids = _extract_grids(text)
        img_wh = _extract_img_wh(text)
        inputs = {"img_wh": img_wh}
        if grids:
            inputs["grids"] = grids
            inputs["example_badge"] = False
        else:
            inputs["grids"] = [(13,13), (26,26)]
            inputs["example_badge"] = True
        m = re.search(r"anchor[s]?\s*[:=]?\s*\(?\s*([0-9.]+)\s*[,x×]\s*([0-9.]+)\s*\)?",
                    text, re.I)
        if m:
            inputs["anchor_rel_cell"] = [float(m.group(1)), float(m.group(2))]
        spec.append({
            "id": "cell_scale",
            "type": "cell_scale",
            "labels": _label("Cell→Pixel scale", "셀 내부 → 픽셀 스케일"),
            "inputs": {"title": _label("Cell→Pixel scale", "셀 내부 → 픽셀 스케일"), **inputs}
        })

    # 고해상도 특징 결합(예시 허용, 데이터셋 소개문에서 제외)
    if T(r"(concat|skip|pass\s?-?\s?through|스킵|업샘플).*(feature|특징|채널)") and not T(r"celeba|lsun|cifar|imagenet"):
        spec.append({
            "id": "highres_fusion",
            "type": "flow_arch",
            "labels": _label("High-res fusion", "특징 결합(개념)"),
            "inputs": {
                "title": _label("High-res fusion (concept)", "특징 결합(개념)"),
                "nodes": [
                    {"id":"in26","label":_label("26×26 features","고해상도 특성")},
                    {"id":"reshape","label":_label("reshape/slice","리셰이프")},
                    {"id":"concat","label":_label("concat along C","채널 방향 결합")},
                    {"id":"out13","label":_label("13×13 head","검출 헤드")},
                ],
                "edges": [{"src":"in26","dst":"reshape"},
                        {"src":"reshape","dst":"concat"},
                        {"src":"concat","dst":"out13"}],
            }
        })

    # IoU 겹침 도식(실수 값이 있으면 표시)
    if T(r"\bIoU\b|Intersection\s*over\s*Union|겹치(는|ㅁ)"):
        iou_val = None
        for key in ("metric.iou", "metric.miou"):
            if key in mentions and mentions[key]:
                iou_val = float(list(mentions[key].values())[0])  # 0~1
                break
        inputs = {}
        if iou_val is not None and 0.01 < iou_val < 0.99:
            inputs["iou"] = round(iou_val, 2)
        spec.append({
            "id": "iou_overlap_demo",
            "type": "iou_overlap",
            "labels": _label("IoU concept", "IoU(겹침) 예시"),
            "inputs": inputs
        })

    # 활성화 함수 패널(언급된 것만)
    if T(r"활성화\s*함수|ReLU|Leaky\s*ReLU|Tanh|Sigmoid|ELU|GELU|PReLU|SELU|SiLU|Swish|Softplus|Mish|Hard\s*(Sigmoid|Swish|Tanh)|ReLU6"):
        funcs = _activation_mentions(text)
        if funcs:
            spec.append({
                "id": "activations_panel",
                "type": "activations_panel",
                "labels": _label("Activation functions", "활성화 함수"),
                "inputs": {
                    "funcs": funcs,
                    "title": _label("Activation functions", "활성화 함수"),
                    "xlim": [-6, 6], "ylim": [-1.3, 1.3]
                }
            })

    # Softmax (temperature/τ) — 수치가 없으면 예시곡선(0.5,1,2)
    if re.search(r"\bsoft\s*max\b|소프트\s*맥스|소프트맥스", text, re.I):
        taus = _parse_taus_from_text(text)
        is_example = False
        if not taus:
            taus = [0.5, 1.0, 2.0]
            is_example = True
        title = _label("Softmax temperature (τ)", "소프트맥스 온도(τ)")
        if is_example:
            title = {"en": title["en"] + " (example)", "ko": title["ko"] + " (예시)"}
        spec.append({
            "id": "softmax_temp",
            "type": "softmax",
            "labels": title,
            "inputs": {"title": title, "taus": taus, "logit_range": [-6, 6], "example_badge": is_example}
        })

    # "A -> B -> C" 체인이 있으면 그대로 파이프라인
    chain = _parse_arrow_chain(text)
    if chain:
        nodes = [{"id": f"n{i}", "label": _label(chain[i], chain[i])} for i in range(len(chain))]
        edges = [{"src": f"n{i}", "dst": f"n{i+1}"} for i in range(len(chain)-1)]
        spec.append({
            "id": "flow_from_text",
            "type": "flow_arch",
            "labels": _label("Pipeline", "파이프라인"),
            "inputs": {"title": _label("Pipeline", "파이프라인"), "nodes": nodes, "edges": edges}
        })

    # Transformer 간단 개념 흐름
    if T(r"transformer") and T(r"encoder") and T(r"decoder"):
        nodes = [
            {"id":"tok","label":_label("Tokens","토큰")},
            {"id":"emb","label":_label("Embed + PosEnc","임베딩 + 위치부호화")},
            {"id":"enc","label":_label("Encoder ×6","인코더 ×6")},
            {"id":"dec","label":_label("Decoder ×6","디코더 ×6")},
            {"id":"smx","label":_label("Softmax","소프트맥스")}
        ]
        edges = [{"src":"tok","dst":"emb"},{"src":"emb","dst":"enc"},
                {"src":"enc","dst":"dec"},{"src":"dec","dst":"smx"}]
        spec.append({
            "id":"transformer_flow",
            "type":"flow_arch",
            "labels": _label("Transformer (concept)","Transformer (개념)"),
            "inputs":{"title":_label("Transformer (concept)","Transformer (개념)"),
                    "nodes":nodes,"edges":edges}
        })

    # Confusion Matrix 2×2 (본문에 네 수치가 있을 때만)
    if T(r"confusion\s*matrix|혼동\s*행렬"):
        M = _extract_confusion_2x2(text)
        if M:
            spec.append({
                "id":"conf_mat",
                "type":"confusion_matrix",
                "labels": _label("Confusion Matrix","혼동행렬"),
                "inputs":{"matrix": M, "labels": ["Neg","Pos"],
                        "title": _label("Confusion Matrix","혼동행렬")}
            })

    return spec
