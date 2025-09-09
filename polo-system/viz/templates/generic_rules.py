import re, random, math
from typing import List, Dict, Any

def _label(en, ko): 
    return {"en": en, "ko": ko}

# 텍스트 시드 기반 난수 (논문마다 다른 모양)
import hashlib
def _stable_seed(text: str, salt: str = "") -> int:
    h = hashlib.sha256((salt + "|" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)
def _rng_from(text: str, salt: str = "") -> random.Random:
    return random.Random(_stable_seed(text, salt))
def _clip(v, lo=0.05, hi=0.95): return max(lo, min(hi, v))

def _ratios_from_text(text: str, k=5, n_per=120):
    rnd = _rng_from(text, "wh")
    centers=[]
    for _ in range(k):
        cx = 0.15 + 0.7*rnd.random()
        cy = 0.15 + 0.7*rnd.random()
        centers.append((cx,cy))
    pts=[]
    for cx,cy in centers:
        for _ in range(n_per):
            x = max(0.05, min(0.95, cx + rnd.uniform(-0.1, 0.1)))
            y = max(0.05, min(0.95, cy + rnd.uniform(-0.1, 0.1)))
            pts.append([round(x,3), round(y,3)])
    return pts

def _read_k_target(text: str, default: int = 5) -> int:
    # 다양한 표현 지원: k=5, k ≈ 5, k 는 5, k 값은 5 ...
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
            return max(1, min(10, k))
    return default

def _k_curve_from_text(text: str):
    rnd   = _rng_from(text, "kcurve")
    ks    = list(range(1, 11))              # 1..10
    kstar = _read_k_target(text, 5)          # 엘보 위치

    y_lo  = 0.45 + 0.04 * rnd.random()       # 하한
    y_hi  = 0.78 + 0.08 * rnd.random()       # 상한
    slope = 0.9  + 1.0  * rnd.random()       # 굽힘(로지스틱 가파름)

    ys = []
    for k in ks:
        t = 1.0 / (1.0 + math.exp(-slope * (k - kstar)))  # k*에서 급격히 증가
        y = y_lo + (y_hi - y_lo) * t
        y += rnd.uniform(-0.008, 0.008)                   # 미세 노이즈
        ys.append(round(_clip(y), 3))
    return ks, ys, kstar



def build_concept_specs(text: str, spec: list, mentions: dict, numeric_cids: set[str] | None = None) -> list:
    numeric_cids = numeric_cids or set()
    T  = lambda rx: re.search(rx, text, re.I | re.S) is not None

    # (w,h) 산점 — 예시 허용
    if T(r"width.*height|w[, ]?h|가로.*세로|비율"):
        spec.append({
            "id": "ratios_scatter",
            "type": "dist_scatter",
            "labels": _label("(w,h) Scatter", "(w,h) 산점"),
            "inputs": {
                "points": _ratios_from_text(text, k=5, n_per=80),
                "xlabel": _label("width (norm)", "폭 (정규화)"),
                "ylabel": _label("height (norm)", "높이 (정규화)"),
                "title":  _label("(w,h) Scatter", "(w,h) 산점"),
            }
        })

    # k 튜닝 곡선
    if T(r"\bk[-\s]*means|\bk\s*=\s*\d+|cluster|클러스터|튜닝"):
        ks, ys, kstar = _k_curve_from_text(text)
        spec.append({
            "id": "k_vs_quality",
            "type": "curve_generic",
            "labels": _label("k tuning curve","k 튜닝 곡선"),
            "inputs": {
                "series": [{"x": ks, "y": ys, "label": _label("avg quality","평균 품질")}],
                "xlabel": _label("k (clusters)", "k (클러스터 수)"),
                "ylabel": _label("avg overlap/IoU", "평균 겹침/IoU"),
                "title":  _label(f"k tuning curve (k≈{kstar})", f"k 튜닝 곡선 (k≈{kstar})")
            }
        })

    # 셀 내부 0~1 / 시그모이드 — 예시 허용
    if T(r"(cell|셀|격자|grid).*(0\s*~\s*1|0~1|sigmoid|시그모이드)"):
        spec.append({
            "id": "cell_relative_flow",
            "type": "flow_arch",
            "labels": _label("Cell-relative prediction", "셀 내부(0~1) 위치 예측"),
            "inputs": {
                "title": _label("Cell-relative prediction", "셀 내부(0~1) 위치 예측"),
                "nodes": [
                    {"id":"a","label":_label("Grid cell","격자 셀")},
                    {"id":"b","label":_label("sigmoid(cx,cy)∈[0,1]","시그모이드(cx,cy)∈[0,1]")},
                    {"id":"c","label":_label("scale by anchors","프라이어/앵커 반영")},
                    {"id":"d","label":_label("box center","박스 중심")},
                ],
                "edges": [{"src":"a","dst":"b"},{"src":"b","dst":"c"},{"src":"c","dst":"d"}],
            }
        })

    # 고해상도 특징 결합(패스스루/concat) — 예시 허용
    if T(r"pass\s?-?\s?through|concat|채널.*(합|결합)|고해상도|26\s*[×x]\s*26|13\s*[×x]\s*13"):
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

    # IoU 겹침 도식: 숫자 있으면 그 값, 없으면 '(예시)' 표시로만
    if T(r"\bIoU\b|Intersection\s*over\s*Union|겹치(는|ㅁ)"):
        iou_val = None
        for key in ("metric.iou", "metric.miou"):
            if key in mentions and mentions[key]:
                # mentions[...]는 0~1 정규화 값. 첫 값만 사용
                iou_val = float(list(mentions[key].values())[0])
                break

        inputs = {}
        if iou_val is not None:
            inputs["iou"] = round(iou_val, 2)  # 숫자 있을 때만 명시적으로 전달

        spec.append({
            "id": "iou_overlap_demo",
            "type": "iou_overlap",
            "labels": _label("IoU concept", "IoU(겹침) 예시"),
            "inputs": inputs
        })
    return spec