# 개념/예시 도식 빌더 (트리거는 100% glossary JSON 기준)
from __future__ import annotations
import re, math, random, json, hashlib
from pathlib import Path
from typing import List, Dict, Any

# 공용 라벨 헬퍼
def _label(en, ko):
    return {"en": en, "ko": ko}

# 규칙 파일 가드
def _safe_wh(img_wh, default=(800, 600)):
    try:
        if isinstance(img_wh, (list, tuple)) and len(img_wh) >= 2:
            return int(img_wh[0]), int(img_wh[1])
    except Exception:
        pass
    return default

# RNG (예시 도식 일관성용)
def _stable_seed(text: str, salt: str = "") -> int:
    h = hashlib.sha256((salt + "|" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)
def _rng_from(text: str, salt: str = "") -> random.Random:
    return random.Random(_stable_seed(text, salt))
def _clip(v, lo=0.05, hi=0.95): 
    return max(lo, min(hi, v))

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

# 혼동행렬 유틸
def _extract_confusion_2x2(text: str):
    m = re.search(r'(?:confusion\s*matrix|혼동\s*행렬)(.{0,120}?)(-?\d+(?:\.\d+)?)[^\d]+'
                r'(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)[^\d]+(-?\d+(?:\.\d+)?)',
                text, re.I | re.S)
    if not m:
        return None
    a, b, c, d = map(float, m.groups()[-4:])
    return [[a, b], [c, d]]

# 셀 그리드 유틸
def _extract_grids(text: str) -> list[tuple[int,int]]:
    grids = []
    # 13×13, 7x7 같은 표기들 회수
    for m in re.finditer(r"(\d{1,3})\s*[×x]\s*(\d{1,3})", text, re.I):
        gw, gh = int(m.group(1)), int(m.group(2))
        if 1 <= gw <= 256 and 1 <= gh <= 256:
            grids.append((gw, gh))
    # 중복 제거/정렬
    grids = list(dict.fromkeys(grids))
    return grids

def _extract_img_wh(text: str):
    m = re.search(r"(\d{2,4})\s*[×x]\s*(\d{2,4})(?:\s*px|\s*픽셀|\s*pixels)?", text, re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    # 값이 없으면 None -> 렌더러가 내부 기본값(예: 640×640)을 사용
    return None

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

# k-means 유틸
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


# Glossary 로딩 & 트리거 헬퍼
# (caller 시그니처 유지 위해 여기서 자체 인덱스 구성)
def _find_glossary_path_here():
    here = Path(__file__).parent
    for p in (here / "glossary_hybrid.json", here / "glossary.json"):
        if p.exists():
            return str(p)
    return None

def _has_trigger(concept_idx: dict, key: str, text: str) -> bool:
    """key가 concept_id면 그 패턴만, category면 해당 카테고리 아무 패턴 매칭되면 True"""
    meta = concept_idx.get(key)
    if meta and meta.get("patterns"):
        return any(p.search(text) for p in meta["patterns"])
    for _, m in concept_idx.items():
        if m.get("category") == key:
            for p in m.get("patterns", []):
                if p.search(text):
                    return True
    return False

# 메인: 텍스트 → 개념 도식 스펙
def build_concept_specs(text: str, spec: list,
                        mentions: dict,
                        numeric_cids: set[str] | None = None,
                        concept_idx: dict | None = None) -> list:
    numeric_cids = numeric_cids or set()
    cidx = concept_idx or {} 

    _has = (lambda key: _has_trigger(cidx, key, text)) if cidx else (lambda key: False)

    # 1) k 튜닝 곡선 (viz.trigger.k_tuning)
    if _has("viz.trigger.k_tuning"):
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
                                f"k 튜닝 곡선 (k≈{kstar}){suffix}"),
                "example_badge": not has_hint
            }
        })

    # 2) Cell→Pixel 스케일 (viz.trigger.cell_scale)
    if _has("viz.trigger.cell_scale"):
        # 텍스트에서 이미지 해상도와 S×S 그리드 추출
        img_wh = _extract_img_wh(text)   
        grids  = _extract_grids(text)  

        # 왼쪽: S×S(텍스트에서 첫 번째로 발견된 작은 그리드)
        # 오른쪽: '픽셀 그리드' = (img_w, img_h)
        if grids:
            left = grids[0]
        else:
            # 하드코딩 최소화: YOLO류 텍스트에서 S=숫자 패턴도 시도
            mS = re.search(r"\bS\s*=\s*(\d{1,3})", text, re.I)
            if mS:
                s = int(mS.group(1))
                left = (s, s)
            else:
                left = (7, 7)  # 마지막 안전 디폴트(텍스트에 아무 단서 없을 때만)

        right = _safe_wh(img_wh, default=(800, 600))  # 픽셀 레벨 그리드

        inputs = {
            "title":  _label("Cell→Pixel scale", "셀 → 픽셀 스케일"),
            "img_wh": img_wh,
            "grids":  [left, right],   # ← 핵심: [S×S, 픽셀 그리드]
            "norm_center":  [0.5, 0.5],
            "norm_wh_cell": [0.6, 0.6],
        }

        # anchor 크기(셀 대비 비율)가 텍스트에 있으면 반영 (예: anchor: 1.5×1.2)
        m = re.search(r"anchor[s]?\s*[:=]?\s*\(?\s*([0-9.]+)\s*[,x×]\s*([0-9.]+)\s*\)?", text, re.I)
        if m:
            inputs["anchor_rel_cell"] = [float(m.group(1)), float(m.group(2))]

        spec.append({
            "id": "cell_scale",
            "type": "cell_scale",
            "labels": _label("Cell→Pixel scale", "셀 내부 → 픽셀 스케일"),
            "inputs": inputs
        })

    # 3) IoU 개념 (viz.trigger.iou_concept)
    if _has("viz.trigger.iou_concept"):
        iou_val = None
        for key in ("metric.iou", "metric.miou"):
            if key in mentions and mentions[key]:
                try:
                    iou_val = float(list(mentions[key].values())[0])
                except:
                    pass
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

    # 4) 활성화 함수 패널 (viz.trigger.activation_panel)
    if _has("viz.trigger.activation_panel"):
        funcs = _activation_mentions(text)
        if funcs:
            spec.append({
                "id": "activations_panel",
                "type": "activations_panel",
                "labels": _label("Activation functions", "활성화 함수"),
                "inputs": {
                    "funcs": funcs,
                    "title": _label("Activation functions", "활성화 함수"),
                    "xlim": [-6, 6],
                    "ylim": [-1.3, 1.3]
                }
            })

    # 5) Softmax 온도 (viz.trigger.softmax_temp)
    if _has("viz.trigger.softmax_temp"):
        NUM = r"[0-9]+(?:\.[0-9]+)?"
        pats = [rf"τ\s*[:=]\s*({NUM})",
                rf"\btau\b\s*[:=]\s*({NUM})",
                rf"\bT\b\s*[:=]\s*({NUM})",
                rf"temperature\s*[:=]?\s*({NUM})"]
        taus = []
        for p in pats:
            for m in re.finditer(p, text, flags=re.I):
                try:
                    v = float(m.group(1))
                    if v > 0: taus.append(v)
                except:
                    pass
        taus = sorted(set(round(v, 3) for v in taus))
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

    # 6) 파이프라인 화살표 (viz.trigger.pipeline_arrows)
    if _has("viz.trigger.pipeline_arrows"):
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

    # 7) Transformer 개념 (viz.trigger.transformer_concept)
    if _has("viz.trigger.transformer_concept"):
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

    # 8) 혼동행렬 (viz.trigger.confusion_matrix)
    if _has("viz.trigger.confusion_matrix"):
        M = _extract_confusion_2x2(text)
        if M:
            spec.append({
                "id":"conf_mat",
                "type":"confusion_matrix",
                "labels": _label("Confusion Matrix","혼동행렬"),
                "inputs":{"matrix": M, "labels": ["Neg","Pos"],
                        "title": _label("Confusion Matrix","혼동행렬")}
            })

    # YOLO 학습/추론 플로우 (viz.trigger.yolo_training_flow)
    if _has("viz.trigger.yolo_training_flow"):
        # [한글 주석] 텍스트에 '책임 박스', 'λ_coord/λ_noobj', 'NMS' 등 키워드가 있을 때 glossary 트리거
        nodes = [
            {"id":"img","label":_label("Input Image","입력 이미지")},
            {"id":"grid","label":_label("Grid cells (S×S)","그리드 셀 (S×S)")},
            {"id":"pred","label":_label("Boxes + Scores + Class prob.","박스+점수+클래스확률")},
            {"id":"assign","label":_label("Max IoU → responsible box","최대 IoU → 책임 박스")},
            {"id":"loss","label":_label("Loss (coord/size/conf/cls)","손실(좌표/크기/신뢰도/분류)")},
            {"id":"nms","label":_label("NMS","NMS")}
        ]
        edges = [
            {"src":"img","dst":"grid"},
            {"src":"grid","dst":"pred"},
            {"src":"pred","dst":"assign"},
            {"src":"assign","dst":"loss"},
            {"src":"pred","dst":"nms"}
        ]
        spec.append({
            "id":"yolo_train_flow",
            "type":"flow_arch",
            "labels":_label("YOLO training/inference flow","YOLO 학습/추론 플로우"),
            "inputs":{"title":_label("YOLO training/inference flow","YOLO 학습/추론 플로우"),
                    "nodes":nodes,"edges":edges}
        })

    # 손실 가중치 비교 바 (viz.trigger.loss_weights)
    if _has("viz.trigger.loss_weights"):
        # [한글 주석] 텍스트에서 λ_coord, λ_noobj 값을 추출 (표기 다양성 커버)
        m1 = re.search(r"λ\s*coord\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        m2 = re.search(r"λ\s*noobj\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if not (m1 and m2):
            m1 = re.search(r"lambda[_\s]*coord\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I) or m1
            m2 = re.search(r"lambda[_\s]*noobj\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I) or m2
        if m1 and m2:
            lam_coord = float(m1.group(1))
            lam_noobj = float(m2.group(1))
            spec.append({
                "id":"loss_weights_bar",
                "type":"bar_group",
                "labels":{"ko":"손실 가중치 비교","en":"Loss weights"},
                "inputs":{
                    "title":{"ko":"손실 가중치 (λ)","en":"Loss weights (λ)"},
                    "ylabel":{"ko":"가중치 값","en":"weight"},
                    "categories":["λ_coord","λ_noobj"],
                    "series":[{"label":{"ko":"가중치","en":"weight"},
                            "values":[lam_coord, lam_noobj]}],
                    "legend": False
                }
            })

    # 7×7×30 텐서 캡션 보강 (viz.trigger.yolo_tensor_7x7x30)
    if _has("viz.trigger.yolo_tensor_7x7x30"):
        # [한글 주석] 렌더러가 캡션을 지원하므로 작은 더미 테이블 형태로 캡션만 노출
        spec.append({
            "id":"tensor_caption_7x7x30",
            "type":"metric_table",
            "labels":{"ko":"출력 텐서 설명","en":"Output tensor"},
            "caption_labels":{"ko":"7×7×30: 셀 7×7, 각 셀당 30차원(2×박스(5) + 20×클래스)","en":"7×7×30: 7×7 cells, 30 dims per cell (2 boxes×5 + 20 classes)"},
            "inputs":{
                "methods": [" "], "metrics": [" "], "values": [[0]],
                "title":{"ko":"출력 텐서 설명","en":"Output tensor"},
                "caption_bottom": 0.10, "caption_y": 0.02
            }
        })

    return spec