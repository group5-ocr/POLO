# Figure 사이드카 맵 시스템 사용 가이드

통합 JSON 구조를 건드리지 않고 별도 `figures_map.json`으로 Figure 정보를 제공하는 시스템입니다.

## 🚀 실행 순서

### 0) 사전 준비
```bash
# PyMuPDF 설치 (PDF → PNG 렌더링용)
pip install pymupdf
```

### 1) Figure 인덱스 생성 및 PNG 렌더링
```bash
# 프로젝트 루트에서 실행
cd C:\POLO\POLO\polo-system
python build_figures_map.py
```

**실행 결과:**
- `C:\POLO\POLO\polo-system\server\data\outputs\viz\figures\` - 렌더링된 PNG 파일들
- `C:\POLO\POLO\polo-system\server\data\outputs\viz\figures_map.json` - 사이드카 맵
- `C:\POLO\POLO\polo-system\server\data\outputs\viz\README_figures.txt` - 안내 파일

### 2) 서버 정적 파일 서빙 확인
서버에 이미 `/static` 마운트가 추가되어 있습니다:
```python
# server/app.py
app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="static")
```

서버 실행:
```bash
# 서버가 실행 중이 아니라면
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3) 프론트엔드 프록시 설정 (개발 환경)
Vite 개발 서버에서 `/static` 경로를 백엔드로 프록시:
```js
// polo-front/vite.config.js
export default {
  server: {
    proxy: {
      '/static': 'http://localhost:8000',
      '/api': 'http://localhost:8000'
    }
  }
}
```

### 4) 테스트
브라우저에서 직접 접근 테스트:
- `http://localhost:8000/static/viz/figures_map.json` - 사이드카 맵
- `http://localhost:8000/static/viz/figures/model/model_p1__해시.png` - 렌더링된 PNG

## 📊 파일 구조

```
outputs/
└── viz/
    ├── figures_map.json          # 사이드카 맵 (핵심)
    ├── README_figures.txt        # 안내 파일
    └── figures/                  # PNG 파일들
        ├── model/
        │   ├── model_p1__a1b2c3d4e5.png
        │   └── model_p2__a1b2c3d4e5.png
        └── architecture/
            └── architecture__f6g7h8i9j0.png
```

## 🔧 사이드카 맵 구조

```json
{
  "figures": [
    {
      "order": 1,
      "label": "fig:model",
      "caption": "YOLO network architecture",
      "graphics": "model",
      "src_file": "C:/POLO/.../source/model.pdf",
      "image_path": "/static/viz/figures/model/model_p1__a1b2c3d4e5.png?v=a1b2c3d4e5",
      "all_pages": [
        "/static/viz/figures/model/model_p1__a1b2c3d4e5.png?v=a1b2c3d4e5",
        "/static/viz/figures/model/model_p2__a1b2c3d4e5.png?v=a1b2c3d4e5"
      ],
      "hash": "a1b2c3d4e5"
    }
  ],
  "metadata": {
    "total_count": 1,
    "generated_at": "auto",
    "source_assets": "C:/POLO/.../assets.jsonl",
    "source_dir": "C:/POLO/.../source",
    "static_root": "C:/POLO/.../outputs"
  }
}
```

## 🎯 작동 원리

### 1. Figure 인덱스 구축
```python
# assets.jsonl 파싱
{
  "env": "figure",
  "graphics": "model",
  "label": "fig:model",
  "caption": "YOLO network architecture"
}

# PDF 찾기 및 렌더링
source/model.pdf → figures/model/model_p1__hash.png
```

### 2. 프론트엔드에서 토큰 교체
```typescript
// 사이드카 맵 로드
const figures = await loadFigureQueue();

// [Figure] 토큰을 실제 Figure로 교체
const chunks = replaceFigureTokens(
  "This is [Figure] showing the architecture.",
  figureQueue,
  figures
);

// 렌더링
{chunks.map(chunk => 
  typeof chunk === 'string' ? 
    <span>{chunk}</span> : 
    <SidecarFigureView figure={chunk} />
)}
```

### 3. 매칭 전략
1. **라벨 우선**: `\ref{fig:model}`, `Figure 1` 등 참조 기반
2. **키워드 매칭**: 텍스트 내용과 figure caption 유사도
3. **순서 기반 Fallback**: 위 방법 실패 시 순서대로

## 🔍 디버깅

### Figure 맵 로드 확인
```javascript
// 브라우저 콘솔에서
fetch('/static/viz/figures_map.json')
  .then(r => r.json())
  .then(console.log);
```

### 토큰 교체 로그 확인
브라우저 개발자 도구 콘솔에서:
```
🔄 [Figure] 토큰 교체: [Figure] → model
🎯 Figure 라벨 매칭: fig:model → model
📊 [Figure] 사이드카 맵 로드: 3개 figures
📈 [Figure] 통계: 5 토큰, 3 figures
```

## ⚠️ 주의사항

### 1. 경로 설정
`build_figures_map.py`의 경로들이 실제 환경과 맞는지 확인:
```python
ASSETS_JSONL = Path(r"C:\POLO\POLO\polo-system\server\data\out\source\assets.jsonl")
SOURCE_DIR   = Path(r"C:\POLO\POLO\polo-system\server\data\arxiv\1506.02640\source")
STATIC_ROOT  = Path(r"C:\POLO\POLO\polo-system\server\data\outputs")
```

### 2. PyMuPDF 설치 오류 시
```bash
# Windows
pip install --upgrade pip
pip install pymupdf

# 또는 conda 사용
conda install -c conda-forge pymupdf
```

### 3. 정적 파일 서빙 확인
- 서버: `http://localhost:8000/static/viz/figures_map.json`
- 프론트: `http://localhost:5173/static/viz/figures_map.json` (프록시 통해)

## 🎉 장점

1. **통합 JSON 보존**: 기존 구조 완전 유지
2. **완전 자동화**: 수동 개입 없이 전체 파이프라인
3. **중복 방지**: 동일 내용 재생성 스킵
4. **지능적 매칭**: 라벨 우선 + 키워드 매칭
5. **멀티페이지 지원**: PDF 모든 페이지 개별 PNG
6. **캐시 최적화**: 해시 기반 버전 관리

## 🔄 업데이트 시
Figure가 변경되면 다시 실행:
```bash
python build_figures_map.py  # 새로운 맵 생성
# 브라우저 새로고침으로 업데이트된 맵 로드
```
