# YOLOv1 논문 통합 분석 목업 데이터

이 폴더는 YOLOv1 논문을 기반으로 한 통합 분석 기능의 목업 데이터와 독립 실행 환경을 포함합니다.

## 폴더 구조

```
mockup_data/
├── data/                       # 데이터 폴더
│   └── integrated_result.jsonl # 통합 결과 (생성됨)
├── src/                        # 소스 코드 폴더
│   ├── integrated_generator.py # 통합 JSONL 생성 코드
│   └── config.py              # 설정 파일
├── frontend/                   # 프론트엔드 폴더
│   ├── index.html             # 메인 페이지
│   ├── script.js              # JavaScript 로직
│   ├── styles.css             # 스타일시트
│   ├── data/                  # 프론트엔드용 데이터 폴더 (자동 생성)
│   │   └── integrated_result.jsonl # 통합 결과 (자동 생성됨)
│   └── charts/                # 시각화 이미지 폴더
│       ├── cell_scale_cell_scale.png
│       ├── activations_panel_activations_panel.png
│       └── ... (기타 시각화 이미지들)
├── run.py                     # 실행 스크립트
└── README.md                  # 이 파일
```

## 사용법

### 1. 통합 JSONL 생성

```bash
cd polo-system/mockup_data/
python run.py
```

이 명령어는 다음을 수행합니다:

- `data/integrated_result.jsonl` 생성 (원본 백업용)
- `frontend/data/integrated_result.jsonl` 생성 (프론트엔드에서 사용)
- `frontend/data/` 폴더 자동 생성
- 프론트엔드가 독립적으로 동작할 수 있도록 필요한 파일들을 모두 준비

### 2. 프론트엔드 실행

```bash
cd polo-system/mockup_data/frontend
python -m http.server 8000
```

### 3. 브라우저에서 확인

```
http://localhost:8000
```

## 데이터 설명

### 통합 결과 (integrated_result.jsonl)

- 논문의 자연스러운 흐름에 따라 섹션별로 통합
- 각 섹션에 쉬운 설명 + 관련 수식 해설 + 시각화 포함
- 프론트엔드에서 표시할 최종 데이터
- 첫 번째 줄: 논문 정보 (제목, 저자, 발표 등)
- 나머지 줄들: 섹션별 상세 내용

**데이터 위치:**

- `data/integrated_result.jsonl`: 원본 백업용
- `frontend/data/integrated_result.jsonl`: 프론트엔드에서 실제 사용

### 시각화 이미지 (charts/)

- YOLO 격자 구조 (cell_scale_cell_scale.png)
- 네트워크 활성화 패턴 (activations_panel_activations_panel.png)
- 성능 비교 차트들 (accuracy_rubric_metric_table.png 등)

## 주요 기능

### 1. 논문 정보 동적 로딩

- 제목, 저자, 발표, 날짜, DOI 등 논문 메타데이터
- 통합 JSONL에서 자동으로 로드

### 2. 목차 네비게이션

- 왼쪽 사이드바에 목차 표시
- 클릭하면 해당 섹션으로 스크롤
- 현재 섹션 하이라이트

### 3. 섹션별 통합 내용

- **쉬운 설명**: 중학생도 이해할 수 있는 논문 내용
- **수식 해설**: 핵심 수식들의 상세한 설명 (MathJax 렌더링)
- **시각화**: 용어 기반 자동 생성된 차트와 그래프

### 4. 반응형 디자인

- 데스크톱과 모바일 모두 지원
- 오렌지/노란색 테마 (기존 프로젝트와 일치)

## 기존 프로젝트와의 통합

이 목업 데이터로 개발 완료 후, 기존 프로젝트와 통합할 수 있습니다:

### 1. HTML 파일을 public 폴더에 복사

```bash
# React 프로젝트에 목업 HTML 복사
cp polo-system/mockup_data/frontend/index.html polo-system/polo-front/public/integrated-result.html
cp polo-system/mockup_data/frontend/styles.css polo-system/polo-front/public/integrated-result.css
cp polo-system/mockup_data/frontend/script.js polo-system/polo-front/public/integrated-result.js

# 데이터 파일도 함께 복사
cp -r polo-system/mockup_data/frontend/data polo-system/polo-front/public/
cp -r polo-system/mockup_data/frontend/charts polo-system/polo-front/public/
```

### 2. React에서 HTML 호출

```typescript
// Upload.tsx에 버튼 추가
const handleIntegratedView = () => {
  window.open("/integrated-result.html", "_blank");
};
```

### 3. 실제 API 연동

- HTML의 JavaScript에서 실제 API 호출하도록 수정
- Easy/Math 모델 결과를 통합 API로 받아오기
- `frontend/data/integrated_result.jsonl` 대신 API에서 데이터 로드

## 문제 해결

### 데이터 파일 404 에러

데이터 파일이 자동으로 생성되지 않는 경우:

```bash
# 수동으로 통합 JSONL 생성
cd polo-system/mockup_data/
python run.py
```

**확인사항:**

- `frontend/data/integrated_result.jsonl` 파일이 존재하는지 확인
- `frontend/data/` 폴더가 생성되었는지 확인

### MathJax 렌더링 문제

- 브라우저에서 MathJax 스크립트가 로드되는지 확인
- 수식이 `$$...$$` 형태로 감싸져 있는지 확인
- 개발자 도구(F12) → Console에서 MathJax 에러 확인

### 시각화 이미지 404 에러

```bash
# 시각화 이미지가 없는 경우
# polo-system/viz/charts/ 폴더에서 이미지들을 복사
cp polo-system/viz/charts/*.png polo-system/mockup_data/frontend/charts/
```
