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
    ks    = list(range(1, 11))
    kstar = _read_k_target(text, 5)

    y0        = 0.45 + 0.04 * rnd.random()
    y_plateau = 0.80 + 0.06 * rnd.random()
    pre_slope  = 0.025 + 0.008 * rnd.random()  # k* 전
    post_slope = 0.006 + 0.004 * rnd.random()  # k* 후(느리게 증가)

    ys = []
    y  = y0
    for k in ks:
        if k < kstar:
            y += pre_slope + rnd.uniform(-0.004, 0.004)
        elif k == kstar:
            # 엘보 부스트 (눈에 띄게 꺾임)
            y += (pre_slope * 2.5) + rnd.uniform(0.01, 0.02)
        else:
            y += post_slope + rnd.uniform(-0.003, 0.003)

        y = min(y, y_plateau + rnd.uniform(-0.01, 0.01))
        ys.append(round(_clip(y), 3))

    return ks, ys, kstar

def _extract_grids(text: str) -> list[tuple[int,int]]:
    # 13x13, 26×26, 80x80 등 찾기 (최대 두 개)
    grids = []
    for m in re.finditer(r"(\d{1,3})\s*[×x]\s*(\d{1,3})", text, re.I):
        gw, gh = int(m.group(1)), int(m.group(2))
        if 1 <= gw <= 256 and 1 <= gh <= 256:
            grids.append((gw, gh))
    # 중복 제거 + 작은/큰 것 2개만
    grids = list(dict.fromkeys(grids))
    if len(grids) >= 2:
        # 가장 작은 면적과 가장 큰 면적 선택
        grids.sort(key=lambda g: g[0]*g[1])
        return [grids[0], grids[-1]]
    return grids

def _extract_img_wh(text: str):
    # 640x640, 1280×720 px 등
    m = re.search(r"(\d{2,4})\s*[×x]\s*(\d{2,4})(?:\s*px|\s*픽셀|\s*pixels)?", text, re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return [640, 640]



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
        subtitle = f" (k≈{kstar})"
        spec.append({
            "id":"k_vs_quality",
            "type":"curve_generic",
            "labels": _label("k tuning curve","k 튜닝 곡선"),
            "inputs":{
                "series":[{"x":ks,"y":ys,"label":_label("avg quality","평균 품질")}],
                "xlabel": _label("k (clusters)","k (클러스터 수)"),
                "ylabel": _label("avg overlap/IoU","평균 겹침/IoU"),
                "title":  _label(f"k tuning curve{subtitle}", f"k 튜닝 곡선{subtitle}")
            }
        })

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

        # norm 값/앵커 비도 텍스트에서 찾아 넣을 수 있으면 넣기 (옵션)
        m = re.search(r"anchor[s]?\s*[:=]?\s*\(?\s*([0-9.]+)\s*[,x×]\s*([0-9.]+)\s*\)?", text, re.I)
        if m:
            inputs["anchor_rel_cell"] = [float(m.group(1)), float(m.group(2))]

        spec.append({
            "id": "cell_scale_demo",
            "type": "cell_scale_demo",
            "labels": _label("Cell→Pixel scale", "셀 내부(0~1) → 픽셀 스케일"),
            "inputs": inputs
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

    # IoU 겹침 도식: 숫자 있으면 그 값(극단 제외), 없으면 예시 배지만
    if T(r"\bIoU\b|Intersection\s*over\s*Union|겹치(는|ㅁ)"):
        iou_val = None
        for key in ("metric.iou", "metric.miou"):
            if key in mentions and mentions[key]:
                iou_val = float(list(mentions[key].values())[0])  # 0~1
                break

        inputs = {}
        # 0.01 < IoU < 0.99 인 경우에만 명시적으로 전달
        if iou_val is not None and 0.01 < iou_val < 0.99:
            inputs["iou"] = round(iou_val, 2)

        spec.append({
            "id": "iou_overlap_demo",
            "type": "iou_overlap",
            "labels": _label("IoU concept", "IoU(겹침) 예시"),
            "inputs": inputs
        })
    return spec