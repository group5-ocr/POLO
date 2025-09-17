# AI Grammars

## 파일 구성

- `glossary_hybrid.json` — 한글, 영어 용어 사전 (지표·리소스 등 키워드 + 정규식) - 그림을 그릴지 고려할 의도 신호
- `text_to_spec.py` — 사전을 읽어 텍스트를 스펙(JSON)으로 변환 - 정말 그려도 되는지 증거(값)를 확인
- `registry.py` — 문법(템플릿) 레지스트리
- `app.py` — 스펙을 받아 PNG로 렌더 -> 실행파일
- `grammars/` — 도형 문법 모듈
- `charts/` — 샘플 출력 이미지가 생성되는 폴더(결과 이미지지)
- `switch.py` — 시각화 라벨 한/영 병기 스위치 유틸
- `generic_rules.py` — 개념/예시 도식 빌더 문법 모듈과 연결
- `patch_glossary.py` — 용어 사전 보강 패처(패치 끝나면 지워도 됨)

## 평가지표

### 루브릭 표·간단 설명 제공

> **기호 안내**: ↑ 높을수록 좋음, ↓ 낮을수록 좋음

- **Accuracy (↑)** — (TP+TN)/전체. 클래스 불균형에선 보조 지표와 함께 해석.
- **Precision (↑)** — TP/(TP+FP). 적중의 순도, 낮으면 오탐이 많음.
- **Recall (↑)** — TP/(TP+FN). 검출 민감도, 낮으면 미탐이 많음.
- **F1-score (↑)** — Precision·Recall의 조화평균 \(2PR/(P+R)\).
- **AUROC (↑)** — 임계값 전 범위 성능. **무작위 기준선 0.5**.
- **AUPRC (↑)** — 불균형 데이터에 유리. **베이스라인≈양성 비율**.
- **mAP (↑)** — 객체검출. **COCO mAP@[.5:.95]**: IoU 0.50–0.95(0.05 간격)에서 AP 평균.
- **Average Recall / AR (↑)** — 여러 IoU/임계에서의 평균 재현율.
- **mIoU (↑)** — 세그멘테이션 품질: 클래스별 IoU 평균.
- **IoU (↑)** — 두 영역의 겹침 비율.
- **Dice (↑)** — \(2|A∩B|/(|A|+|B|)\). IoU와 유사하나 값 스케일이 다름.
- **PQ (Panoptic Quality, ↑)** — 판옵틱 품질: **PQ = SQ × RQ**.
- **FID (↓)** — Frechet Inception Distance. 생성 이미지 분포 거리.
- **KID (↓)** — Kernel Inception Distance. 샘플 수에 덜 민감.
- **LPIPS (↓)** — 지각적 거리. 낮을수록 원본과 유사.
- **Inception Score / IS (↑)** — 다양성과 품질 지표(데이터셋 의존).
- **WER (↓)** — Word Error Rate.
- **CER (↓)** — Character Error Rate.

### 추출 지원(루브릭 없음)

- **Top-5 Accuracy (↑)** — 정답이 상위 5위 내 포함되는 비율.
- **Specificity (↑)** — TN/(TN+FP). 음성 판별력.
- **FVD (↓)** — Fréchet Video Distance.

## 문법(Grammar) 퀵 레퍼런스

### metric_table — 평가지표 루브릭 표

필수:

methods: ["Excellent","Good","Fair","Poor"]

metrics: ["min mAP"] (대개 1개)

values: [[0.55],[0.40],[0.25],[0.00]]

주요 옵션:

title(문자/라벨), caption(문자/라벨)

배치: title_y(기본 0.92), title_gap(제목↔표, 기본 0.030), caption_gap(표↔설명, 기본 0.010)

설명: 제목/설명은 표 bbox 기준으로 근접 배치 → 숫자만 바꿔도 즉시 반영.

2. ### curve_generic — ROC/PR/학습곡선 등 범용 선 그래프

필수: series: [{"x":[...], "y":[...], "label":...}, ...]

주요 옵션:

style("line"|"step"), legend_loc, xlim/ylim

annotate_last(마지막 점 값 표시)

diag(ROC 무작위 기준선)

하단 캡션: caption_bottom(기본 0.10), caption_y(기본 0.005)

3. ### bar_group — 방법 비교 막대

필수:

categories: 방법/조건 라벨들

series: [{"label":"mAP","values":[...]}]

주요 옵션: ylabel, annotate, ylim

4. ### donut_pct — 구성비 도넛

필수: parts: [("A", 30), ("B", 70)]

5. ### stack_bar — 분해/누적 막대

필수:

categories: 상단 축 라벨

series: [{"label":"Span","values":[...]}, ...]

주요 옵션: normalize(100% 스택), legend_out

6. ### histogram — 분포

필수: values: [..충분히 많은 수치..]

주요 옵션: bins: "fd"(기본), xlabel, title

7. ### iou_overlap — 박스 IoU 예시

필수 없음 (기본 A/B 박스)

주요 옵션: A, B, title, show_value, iou(직접값)

8. ### softmax — 2-class softmax(=sigmoid) 곡선

필수: taus: [1.0, 2.0]

주요 옵션: logit_range, title

9. ### token_sequence — 시퀀스/토큰 도식

필수: tokens: ["[CLS]","sentA","[SEP]","sentB","[SEP]"]

10. ### embedding_sum — 임베딩/특징 결합 도식

필수: rows: ["feat1","feat2", ...], right: "Encoder / Fusion"

11. ### kpi_card — 단일 KPI 카드

필수: title, value (문자열)

## 트리거(용어사전 카테고리) 예시

viz.trigger.curve_generic, viz.intent.curve_generic

viz.trigger.bar_group, viz.intent.comparison

viz.trigger.donut_pct, viz.intent.composition

viz.trigger.stack_bar, viz.intent.breakdown

viz.intent.distribution (히스토그램)

viz.trigger.embedding_sum, viz.intent.fusion

viz.trigger.token_sequence, viz.intent.sequence_format

중복 트리거는 용어사전 단계에서 합쳐 매칭합니다(코드도 해당 방식으로 변경).

## 폰트 & 환경

런타임 폰트 등록 + 패밀리 우선순위 자동 결정

FONT_KR_PATH / FONT_KR_FAMILY 지원

DPI: savefig.dpi = 220

디바이스: CPU 강제 (DEVICE="cpu")

포트: VIZ_PORT (기본 5005)

## 실행

```bash
python app.py
```
