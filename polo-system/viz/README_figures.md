# PDF → PNG 렌더링 및 Figure 통합 시스템

원본 PDF 파일을 PNG로 렌더링하고 [Figure] 토큰에 자동으로 첨부하는 시스템입니다.

## 🔧 주요 기능

### 1. PDF → PNG 렌더링 (중복 방지)
- **PyMuPDF** 사용한 고품질 PDF 렌더링 (DPI 200)
- **내용 기반 해시 파일명**: 동일한 내용이면 재생성 스킵
- **멀티페이지 지원**: PDF의 각 페이지를 개별 PNG로 변환
- **이미지 파일 복사**: PNG/JPG 등 이미지 파일 자동 복사

### 2. [Figure] 토큰 자동 매칭
- **라벨 우선 매칭**: `\ref{fig:model}`, `Figure 1` 등 참조 기반
- **키워드 매칭**: 텍스트 내용과 figure caption 유사도 기반
- **순서 기반 Fallback**: 라벨 매칭 실패 시 순서대로 할당
- **토큰 제거**: 매칭 후 [Figure] 토큰 자동 제거

### 3. 웹 경로 정규화
- **Windows → Web**: `C:\path\file.png` → `/static/viz/figures/file.png`
- **캐시 버스팅**: `?v=hash` 쿼리스트링으로 버전 관리
- **정적 파일 서빙**: `/static/*` 경로로 브라우저 접근

## 📁 파일 구조

```
polo-system/
├── viz/
│   ├── assets_mapper.py      # PDF 렌더링 및 인덱스 구축
│   ├── integrate_figures.py  # [Figure] 토큰 통합
│   └── save_png.py          # 중복 방지 PNG 저장 (기존)
├── server/
│   └── app.py               # Figure 처리 엔드포인트
└── polo-front/src/
    ├── types/index.ts       # TypeScript 타입 정의
    └── pages/Result.tsx     # Figure 렌더링 컴포넌트
```

## 🚀 사용법

### 1. 서버 측 Figure 처리

```python
# 자동 처리 (권장)
POST /api/build-figures?paper_id=1506.02640

# 수동 경로 지정
POST /api/build-figures
{
    "paper_id": "1506.02640",
    "assets_path": "/path/to/assets.jsonl",
    "integrated_path": "/path/to/integrated_result.json"
}

# Figure 목록 조회
GET /api/results/1506.02640/figures
```

### 2. 프로그래밍 방식 사용

```python
from viz.assets_mapper import build_figure_index, get_figure_web_paths
from viz.integrate_figures import attach_figures

# 1. Figure 인덱스 구축
figures = build_figure_index(
    assets_jsonl=Path("assets.jsonl"),
    source_dir=Path("source/"),
    png_root=Path("outputs/")
)

# 2. 웹 경로 추가
figures_with_web = get_figure_web_paths(figures, "/static")

# 3. [Figure] 토큰에 첨부
attach_figures(
    integrated_json_path=Path("integrated_result.json"),
    out_path=Path("integrated_result.with_figures.json"),
    figures=figures_with_web,
    static_prefix="/static"
)
```

### 3. 프론트엔드 렌더링

```tsx
import type { FigureMeta } from "../types";

// Figure 컴포넌트 사용
<FigureView 
  figure={paragraph.figure} 
  openImage={openImageHandler} 
  className="paragraph-figure"
/>

// 멀티페이지 지원
{figure.all_pages?.map((pageUrl, idx) => (
  <button onClick={() => openImage(pageUrl)}>
    Page {idx + 1}
  </button>
))}
```

## 📊 파일명 예시

### 입력
```
source/
├── model.pdf          # 3페이지 PDF
├── architecture.png   # 단일 이미지
└── flow.jpg          # 단일 이미지
```

### 출력 (해시 기반)
```
outputs/viz/figures/
├── model/
│   ├── model_p1__a1b2c3d4e5.png
│   ├── model_p2__a1b2c3d4e5.png
│   └── model_p3__a1b2c3d4e5.png
├── architecture/
│   └── architecture__f6g7h8i9j0.png
└── flow/
    └── flow__k1l2m3n4o5.jpg
```

### 웹 URL
```
/static/viz/figures/model/model_p1__a1b2c3d4e5.png?v=a1b2c3d4e5
/static/viz/figures/architecture/architecture__f6g7h8i9j0.png?v=f6g7h8i9j0
```

## 🔄 처리 흐름

1. **assets.jsonl 파싱** → figure 환경 추출
2. **원본 파일 검색** → PDF/이미지 파일 찾기
3. **PNG 렌더링/복사** → 해시 기반 파일명으로 저장
4. **인덱스 구축** → 메타데이터 생성
5. **토큰 매칭** → [Figure] 위치에 적절한 figure 할당
6. **JSON 업데이트** → 통합 결과에 figure 정보 첨부

## ⚙️ 설정

### PyMuPDF 설치
```bash
pip install PyMuPDF
```

### 서버 정적 파일 마운트
```python
app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="static")
```

### 프론트엔드 프록시 (Vite)
```js
// vite.config.js
export default {
  server: {
    proxy: {
      '/static': 'http://localhost:8000'
    }
  }
}
```

## 🎯 장점

- **성능**: 동일 내용 재생성 방지로 속도 향상
- **저장 공간**: 중복 파일 생성 방지
- **호환성**: Windows/Linux/Web 모든 환경 지원
- **자동화**: 수동 개입 없이 완전 자동 처리
- **확장성**: 새로운 figure 형식 쉽게 추가 가능

## 🐛 문제 해결

### PyMuPDF 설치 오류
```bash
# Windows
pip install --upgrade pip
pip install PyMuPDF

# Linux
sudo apt-get install python3-dev
pip install PyMuPDF
```

### 경로 문제
- Windows 역슬래시 → 자동으로 슬래시 변환
- 상대/절대 경로 → 자동 정규화
- 정적 파일 서빙 → `/static/*` 마운트 확인

### 메모리 사용량
- 대용량 PDF → DPI 조정 (기본 200 → 150)
- 멀티페이지 → 페이지별 개별 처리로 메모리 절약
