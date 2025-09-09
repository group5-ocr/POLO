import re, random

def build_concept_specs(text: str) -> list:
    """
    논문 요약 텍스트에서 일반 키워드를 잡아 다음 도식을 생성:
      - (w,h) 산점  → dist_scatter
      - k 튜닝 곡선 → curve_generic
      - 셀 0~1 좌표 → flow_arch
      - 고해상도 결합 → flow_arch
      - IoU 겹침     → iou_overlap
    """
    T = lambda rx: re.search(rx, text, re.I | re.S) is not None
    spec = []
    rnd = random.Random(42)

    # 1) 박스 비율/클러스터/앵커/프로토타입
    if T(r"\b(k-?\s?means|cluster|anchor|prior|prototype|aspect\s*ratio)\b|가로.*세로|비율"):
        centers = [(0.2,0.65),(0.3,0.3),(0.55,0.45),(0.7,0.65),(0.85,0.25)]
        pts=[]
        for cx, cy in centers:
            for _ in range(100):
                x=max(0.05,min(0.95,cx+rnd.uniform(-0.07,0.07)))
                y=max(0.05,min(0.95,cy+rnd.uniform(-0.07,0.07)))
                pts.append([round(x,3),round(y,3)])
        spec.append({
            "id":"ratios_scatter","type":"dist_scatter",
            "labels":{"en":"(w,h) scatter","ko":"(w,h) 산점"},
            "inputs":{
                "points":pts,
                "xlabel":{"en":"width (norm)","ko":"폭 (정규화)"},
                "ylabel":{"en":"height (norm)","ko":"높이 (정규화)"},
                "title":{"en":"(w,h) scatter","ko":"(w,h) 산점"}
            }
        })

    # 2) k vs 품질(평균 겹침/IoU 등)
    if T(r"\bk\s*(=|\d)|cluster|클러스터|평균\s*(IoU|overlap|겹침)"):
        ks=list(range(1,11))
        ys=[0.47,0.58,0.62,0.67,0.71,0.73,0.745,0.768,0.785,0.792]
        spec.append({
            "id":"k_vs_quality","type":"curve_generic",
            "labels":{"en":"k vs quality","ko":"k 변화에 따른 품질"},
            "inputs":{
                "series":[{"x":ks,"y":ys,"label":{"en":"avg quality","ko":"평균 품질"}}],
                "xlabel":{"en":"k (clusters)","ko":"k (클러스터 수)"},
                "ylabel":{"en":"avg overlap/IoU","ko":"평균 겹침/IoU"},
                "title":{"en":"k tuning curve","ko":"k 튜닝 곡선"}
            }
        })

    # 3) 셀/격자 & 0~1/시그모이드
    if T(r"(cell|셀|격자|grid).*(0\s*~\s*1|0~1|sigmoid|시그모이드)"):
        spec.append({
            "id":"cell_relative","type":"flow_arch",
            "labels":{"en":"Cell-relative prediction","ko":"셀 내부(0~1) 위치 예측"},
            "inputs":{
                "title":{"en":"Cell-relative prediction","ko":"셀 내부(0~1) 위치 예측"},
                "nodes":[
                    {"id":"a","label":{"en":"Grid cell","ko":"격자 셀"}},
                    {"id":"b","label":{"en":"sigmoid(cx,cy)∈[0,1]","ko":"시그모이드(cx,cy)∈[0,1]"}},
                    {"id":"c","label":{"en":"scale by prior/anchor","ko":"프라이어/앵커 반영"}},
                    {"id":"d","label":{"en":"box center","ko":"박스 중심"}},
                ],
                "edges":[{"src":"a","dst":"b"},{"src":"b","dst":"c"},{"src":"c","dst":"d"}]
            }
        })

    # 4) 고해상도 특징 결합(업샘플/concat/패스스루 등)
    if T(r"(pass\s?-?\s?through|concat|업샘플|업샘플링|고해상도|feature\s*fusion)"):
        spec.append({
            "id":"highres_fusion","type":"flow_arch",
            "labels":{"en":"High-res feature fusion","ko":"고해상도 특징 결합"},
            "inputs":{
                "title":{"en":"Feature fusion (concept)","ko":"특징 결합(개념)"},
                "nodes":[
                    {"id":"inH","label":{"en":"High-res feat.","ko":"고해상도 특성"}},
                    {"id":"reshape","label":{"en":"reshape/slice","ko":"리셰이프"}},
                    {"id":"concat","label":{"en":"concat along C","ko":"채널 방향 결합"}},
                    {"id":"head","label":{"en":"Detection head","ko":"검출 헤드"}},
                ],
                "edges":[{"src":"inH","dst":"reshape"},{"src":"reshape","dst":"concat"},{"src":"concat","dst":"head"}]
            }
        })

    # 5) IoU/겹침 개념도
    if T(r"\bIoU\b|Intersection\s*over\s*Union|overlap|겹침"):
        spec.append({
            "id":"iou_overlap_demo","type":"iou_overlap",
            "labels":{"en":"IoU overlap","ko":"IoU(겹침) 개념"},
            "inputs":{
                "A":[0.15,0.15,0.55,0.55],
                "B":[0.35,0.25,0.55,0.55],
                "title":{"en":"IoU overlap","ko":"IoU(겹침) 개념"},
                "show_value": True
            }
        })
    return spec