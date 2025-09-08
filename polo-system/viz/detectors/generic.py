import re, random
from typing import List, Dict, Any

def _extract_classes(text: str):
    m = re.search(r"[Cc]lasses?\s*[:=]\s*([A-Za-z0-9_,\s\"']{3,})", text)
    if not m: return []
    raw = m.group(1)
    parts = re.findall(r"[A-Za-z0-9._+-]+", raw)
    parts = [p for p in parts if not p.isdigit() and 1 <= len(p) <= 20]
    seen, out = set(), []
    for p in parts:
        q = p.strip(" '\"").lower()
        if q and q not in seen:
            seen.add(q); out.append(p.strip(" '\""))
        if len(out) >= 8: break
    return out

def _synth_confusion(labels):
    rnd = random.Random(17)
    n = len(labels)
    mat = [[0]*n for _ in range(n)]
    for i in range(n):
        total = rnd.randint(45, 65)
        diag  = rnd.randint(int(total*0.6), int(total*0.8))
        rest  = total - diag
        shares = [rnd.random() for _ in range(n-1)]
        s = sum(shares) or 1.0
        k = 0
        for j in range(n):
            mat[i][j] = diag if j==i else int(rest * shares[k] / s + 0.5); k += (j!=i)
    return mat

def _synth_hist(n=250):
    rnd = random.Random(23)
    vals = []
    for _ in range(n):
        v = min(1, max(0, 0.5 + rnd.gauss(0, 0.15)))
        vals.append(round(v, 4))
    return vals

def _synth_embedding_points():
    rnd = random.Random(29)
    labs = list("ABCDE")
    pts = []
    for i,l in enumerate(labs):
        cx, cy = 0.2 + 0.15*i, 0.8 - 0.12*i
        for _ in range(30):
            x = max(0.05, min(0.95, cx + rnd.uniform(-0.08, 0.08)))
            y = max(0.05, min(0.95, cy + rnd.uniform(-0.08, 0.08)))
            pts.append({"x": round(x,3), "y": round(y,3), "label": l})
    return pts

def _build_metric_table_from_mentions(mentions: Dict[str, Dict[str, float]]):
    if not mentions: return None
    metric_ids = list(mentions.keys())
    metric_names = [cid.split(".")[-1].upper() for cid in metric_ids]
    by_method={}
    for cid in metric_ids:
        for m,v in mentions[cid].items():
            by_method.setdefault(m,{})[cid]=v
    def avg(m):
        xs=[by_method[m].get(cid) for cid in metric_ids if by_method[m].get(cid) is not None]
        return sum(xs)/len(xs) if xs else 0.0
    methods = sorted(by_method.keys(), key=avg, reverse=True)[:3]
    if not methods: return None
    values=[]
    for m in methods:
        row=[]
        for cid in metric_ids:
            v=by_method[m].get(cid)
            row.append(round(v,3) if v is not None else None)
        values.append(row)
    return {
        "id": "metric_table_auto",
        "type": "metric_table",
        "labels": {"en":"SOTA Comparison","ko":"SOTA 비교 표"},
        "inputs": {
            "methods": methods,
            "metrics": metric_names,
            "values": values,
            "title": {"en":"SOTA Comparison","ko":"SOTA 비교"}
        }
    }

def detect(text: str, mentions: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    T  = lambda rx: re.search(rx, text, re.I | re.S) is not None
    rnd = random.Random(41)
    spec=[]

    # 1) Confusion Matrix
    if T(r"confusion\s*matrix|오\s?분류\s?행렬|혼동\s?행렬"):
        labels = _extract_classes(text) or ["cat","dog","bird"]
        mat = _synth_confusion(labels)
        spec.append({
            "id":"cm_auto","type":"confusion_matrix",
            "labels":{"en":"Confusion Matrix","ko":"혼동 행렬"},
            "inputs":{"labels":labels,"matrix":mat,"title":{"en":"Confusion Matrix","ko":"혼동 행렬"}}
        })

    # 2) Histogram (confidence/uncertainty distribution)
    if T(r"confidence\s*distribution|uncertainty|확률\s*분포|점수\s*분포|히스토그램"):
        spec.append({
            "id":"hist_auto","type":"histogram",
            "labels":{"en":"Confidence Distribution","ko":"신뢰도(확률) 분포"},
            "inputs":{"values":_synth_hist(240),"bins":20,
                      "xlabel":{"en":"confidence","ko":"신뢰도"},
                      "title":{"en":"Confidence Distribution","ko":"신뢰도 분포"}}
        })

    # 3) (w,h) 산점
    if T(r"width.*height|w[, ]?h|비율|가로.*세로"):
        pts=[]
        for cx,cy in [(0.25,0.65),(0.6,0.45),(0.8,0.25)]:
            for _ in range(60):
                x=max(0.05,min(0.95,cx+rnd.uniform(-0.1,0.1)))
                y=max(0.05,min(0.95,cy+rnd.uniform(-0.1,0.1)))
                pts.append([round(x,3),round(y,3)])
        spec.append({
            "id":"wh_scatter","type":"dist_scatter",
            "labels":{"en":"(w,h) scatter","ko":"(w,h) 산점"},
            "inputs":{"points":pts,
                      "xlabel":{"en":"width (norm)","ko":"폭 (정규화)"},
                      "ylabel":{"en":"height (norm)","ko":"높이 (정규화)"},
                      "title":{"en":"(w,h) Scatter","ko":"(w,h) 산점"}}
        })

    # 4) Embedding map
    if T(r"\b(t-?sne|umap|pca|embedding|임베딩|특징\s*공간)\b"):
        spec.append({
            "id":"embed_map","type":"embedding_map",
            "labels":{"en":"Embedding (2D)","ko":"임베딩(2D)"},
            "inputs":{"points":_synth_embedding_points(),
                      "title":{"en":"Embedding (2D)","ko":"임베딩(2D)"}}
        })

    # 5) Metric table
    if T(r"SOTA|baseline|비교|table\s*\d+|표\s*\d+") or len(mentions) >= 2:
        mt = _build_metric_table_from_mentions(mentions)
        if mt: spec.append(mt)

    # 6) Flow / Module
    if T(r"pipeline|flow|architecture|아키텍처|백본|backbone|neck|head"):
        spec.append({
            "id":"flow_auto","type":"flow_arch",
            "labels":{"en":"Pipeline","ko":"파이프라인"},
            "inputs":{"title":{"en":"Pipeline","ko":"파이프라인"},
                      "nodes":[{"id":"in","label":{"en":"Input","ko":"입력"}},
                               {"id":"bb","label":{"en":"Backbone","ko":"백본"}},
                               {"id":"nk","label":{"en":"Neck","ko":"넥"}},
                               {"id":"hd","label":{"en":"Head","ko":"헤드"}},
                               {"id":"out","label":{"en":"Output","ko":"출력"}}],
                      "edges":[{"src":"in","dst":"bb"},{"src":"bb","dst":"nk"},
                               {"src":"nk","dst":"hd"},{"src":"hd","dst":"out"}]}
        })
        spec.append({
            "id":"module_stack","type":"module_block",
            "labels":{"en":"Model Stack","ko":"모델 스택"},
            "inputs":{"blocks":["Input","Backbone","Neck","Head","Output"],
                      "title":{"en":"Model Stack","ko":"모델 스택"}}
        })
    return spec