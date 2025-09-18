# 개념/예시 도식 빌더 — glossary 트리거만 사용 (정규식 하드코딩 제거)

from __future__ import annotations
import re, math, random, hashlib
from typing import List, Dict, Any, Callable

def _label(en, ko):
    return {"en": en, "ko": ko}

# ---------- 공통 유틸 ----------
def _stable_seed(text: str, salt: str = "") -> int:
    h = hashlib.sha256((salt + "|" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)
def _rng_from(text: str, salt: str = "") -> random.Random:
    return random.Random(_stable_seed(text, salt))
def _clip(v, lo=0.05, hi=0.95): return max(lo, min(hi, v))

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
    m = re.search(
        r'(?:confusion\s*matrix|혼동\s*행렬)(.{0,120}?)(-?\d+(?:\.\d+)?)[^\d]+'
        r'(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)',
        text, re.I | re.S
    )
    if not m: return None
    a, b, c, d = map(float, m.groups()[-4:])
    return [[a, b], [c, d]]

def _extract_grids(text: str) -> list[tuple[int,int]]:
    grids = []
    for m in re.finditer(r"(\d{1,3})\s*[×x]\s*(\d{1,3})", text, re.I):
        gw, gh = int(m.group(1)), int(m.group(2))
        if 1 <= gw <= 256 and 1 <= gh <= 256:
            grids.append((gw, gh))
    grids = list(dict.fromkeys(grids))
    if 4 <= gw <= 256 and 4 <= gh <= 256:
        grids.append((gw, gh))
    return grids

def _extract_img_wh(text: str):
    m = re.search(r"(\d{2,4})\s*[×x]\s*(\d{2,4})(?:\s*px|\s*픽셀|\s*pixels)?", text, re.I)
    if m: return [int(m.group(1)), int(m.group(2))]
    return [640, 640]

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

    y0, y_plateau  = 0.45 + 0.04*rnd.random(), 0.80 + 0.06*rnd.random()
    pre_slope, post_slope = 0.025 + 0.008*rnd.random(), 0.006 + 0.004*rnd.random()

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
                if "beta" in k or "β" in k: out["beta"] = val
                elif "slope" in k:          out["alpha"] = val
                else:                       out["alpha"] = val
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

# 빌더: glossary 트리거 기반
def build_concept_specs(
    text: str,
    spec: list,
    mentions: dict,
    numeric_cids: set[str] | None = None,
    has_trigger=None,   # ← 추가: glossary 트리거 콜백
) -> list:

    numeric_cids = numeric_cids or set()
    if has_trigger is None:
        # 필수: JSON 트리거를 쓰기 위해 반드시 콜백이 필요
        has_trigger = lambda cid: False

    # k 튜닝 곡선 — 숫자 없어도 트리거면 그린다 (기본 k★=5)
    if has_trigger and has_trigger("viz.trigger.k_tuning"):
        ks, ys, kstar, has_hint = _k_curve_from_text(text)
        suffix = "" if has_hint else " (예시)"
        spec.append({
            "id": "k_vs_quality",
            "type": "curve_generic",
            "labels": {"en":"k tuning curve","ko":"k 튜닝 곡선"},
            "inputs": {
                "series": [{"x": ks, "y": ys, "label": {"en":"avg quality","ko":"평균 품질"}}],
                "xlabel": {"en":"k (clusters)","ko":"k (클러스터 수)"},
                "ylabel": {"en":"avg overlap/IoU","ko":"평균 겹침/IoU"},
                "title":  {"en": f"k tuning curve (k≈{kstar}){suffix}",
                        "ko": f"k 튜닝 곡선 (k≈{kstar}){suffix}"}
            }
        })

    # 셀→픽셀 스케일
    if has_trigger("viz.trigger.cell_scale"):
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

    # IoU 개념 (숫자 있으면 반영)
    if has_trigger("viz.trigger.iou_concept"):
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

    # 활성화 함수 패널
    if has_trigger("viz.trigger.activation_panel"):
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

    # 소프트맥스 온도
    if has_trigger("viz.trigger.softmax_temp"):
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

    # 파이프라인 화살표
    if has_trigger("viz.trigger.pipeline_arrows"):
        # 원본 로직: 텍스트의 "A -> B -> C" 그대로 쓰는 버전이었다면 거기 함수 사용
        seps = r'(?:->|→|⇒|⟶)'
        best = []
        for line in text.splitlines():
            if re.search(seps, line):
                s = re.sub(r'\s*(->|→|⇒|⟶)\s*', '->', line)
                toks = [t.strip(' []()') for t in s.split('->') if t.strip()]
                toks = [t for t in toks if 2 <= len(t) <= 40]
                if len(toks) >= 3 and len(toks) > len(best):
                    best = toks
        if best:
            nodes = [{"id": f"n{i}", "label": _label(best[i], best[i])} for i in range(len(best))]
            edges = [{"src": f"n{i}", "dst": f"n{i+1}"} for i in range(len(best)-1)]
            spec.append({
                "id": "flow_from_text",
                "type": "flow_arch",
                "labels": _label("Pipeline", "파이프라인"),
                "inputs": {"title": _label("Pipeline", "파이프라인"), "nodes": nodes, "edges": edges}
            })

    # 트랜스포머 개념
    if has_trigger("viz.trigger.transformer_concept"):
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

    # 혼동행렬
    if has_trigger("viz.trigger.confusion_matrix"):
        m = re.search(r'(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)',
                    text, re.I)
        if m:
            a, b, c, d = map(float, m.groups())
            spec.append({
                "id":"conf_mat",
                "type":"confusion_matrix",
                "labels": _label("Confusion Matrix","혼동행렬"),
                "inputs":{"matrix": [[a,b],[c,d]], "labels": ["Neg","Pos"],
                        "title": _label("Confusion Matrix","혼동행렬")}
            })

    return spec