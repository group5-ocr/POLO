# Figure 사이드카 시스템 사용 가이드

**"기존 건 손대지 않고 추가만"** 방식으로 구현된 Figure 사이드카 시스템입니다.

## 🎯 핵심 개념

- **기존 파이프라인 보존**: 메인 시스템 코드 수정 없음
- **옵션 방식**: `figures_map.json`이 있으면 사용, 없으면 기존 렌더링
- **추가 파일만**: `tools/` 폴더에 새 파일 2개 + `run_system.bat` 몇 줄 추가
- **자동 처리**: 배치 실행 시 자동으로 PDF → PNG → 맵 생성

## 📋 설치 및 실행

### 1) 사전 준비 (한 번만)
```bash
pip install pymupdf
```

### 2) 시스템 실행
기존과 동일하게 실행:
```bash
run_system.bat
```

**새로 추가된 동작:**
- Frontend 시작 후 자동으로 `build_figures_map.py` 실행
- `assets.jsonl` → PDF 렌더링 → `figures_map.json` 생성
- 프론트엔드에서 자동으로 사이드카 맵 로드

### 3) 확인
브라우저에서 다음 URL로 확인:
- `http://localhost:8000/static/viz/figures_map.json` - 사이드카 맵
- `http://localhost:8000/static/viz/figures/model/model_p1__해시.png` - 렌더링된 PNG

## 🔧 추가된 파일들

### 1. `tools/build_figures_map.py`
```python
# assets.jsonl 파싱 → PDF 렌더링 → figures_map.json 생성
# 자동으로 run_system.bat에서 실행됨
```

### 2. `tools/figsidecar_app.py` (선택적)
```python
# 메인 서버에 /static이 없을 때 사용하는 별도 정적 서버
# 포트 8010에서 /static/* 서빙
```

### 3. `run_system.bat` 추가 부분
```batch
REM [FIGURES] Build figures_map.json (assets.jsonl → PNG → map)
set FIG_BUILD_PY=%~dp0tools\build_figures_map.py
if exist "%FIG_BUILD_PY%" (
  echo [FIG] Generating figures_map.json ...
  python "%FIG_BUILD_PY%"
) else (
  echo [FIG][WARN] build_figures_map.py not found: %FIG_BUILD_PY%
)
```

## 🎨 프론트엔드 동작

### 기존 동작 (변경 없음)
- `[Figure]` 토큰이 있어도 그대로 텍스트로 표시
- 기존 visualization 시스템 정상 동작

### 새 동작 (추가됨)
- `figures_map.json` 로드 성공 시:
  - `[Figure]` 토큰을 실제 이미지로 교체
  - 순서대로 매칭 (문서 등장 순서)
  - 클릭으로 확대 가능

### 로드 우선순위
1. 메인 서버: `/static/viz/figures_map.json`
2. 사이드카: `http://localhost:8010/static/viz/figures_map.json`
3. 실패 시: 기존 렌더링 유지

## 📊 파일 구조

```
polo-system/
├── tools/                           # 새 파일들
│   ├── build_figures_map.py         # PDF → PNG → 맵 생성
│   └── figsidecar_app.py           # 선택적 정적 서버
├── server/data/outputs/viz/         # 생성 결과
│   ├── figures_map.json            # 사이드카 맵
│   └── figures/                    # PNG 파일들
│       ├── model/
│       │   ├── model_p1__해시.png
│       │   └── model_p2__해시.png
│       └── architecture/
│           └── architecture__해시.png
└── run_system.bat                  # 몇 줄 추가됨
```

## 🔍 디버깅

### 콘솔 로그 확인
```javascript
// 브라우저 개발자 도구에서
✅ [FIG] 메인 서버에서 로드: 3
🔄 [FIG] 토큰 교체: [Figure] → model
```

### 수동 맵 생성
```bash
cd polo-system
python tools/build_figures_map.py
```

### 사이드카 서버 수동 실행
```bash
python tools/figsidecar_app.py --port 8010
```

## ⚠️ 주의사항

### 1. 경로 설정
`tools/build_figures_map.py`의 경로가 실제 환경과 맞는지 확인:
```python
ASSETS_JSONL = Path(r"C:\POLO\POLO\polo-system\server\data\out\source\assets.jsonl")
SOURCE_DIR   = Path(r"C:\POLO\POLO\polo-system\server\data\arxiv\1506.02640\source")
STATIC_ROOT  = Path(r"C:\POLO\POLO\polo-system\server\data\outputs")
```

### 2. 메인 서버 /static 마운트
`server/app.py`에 이미 다음이 있다면 사이드카 서버 불필요:
```python
app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="static")
```

### 3. PyMuPDF 설치 오류 시
```bash
# Windows
pip install --upgrade pip
pip install pymupdf

# 또는
conda install -c conda-forge pymupdf
```

## 🎉 장점

1. **기존 시스템 보존**: 메인 코드 한 줄도 수정 안 함
2. **점진적 적용**: 맵 파일이 없어도 정상 동작
3. **자동화**: 배치 실행만으로 전체 처리
4. **중복 방지**: 동일 PDF 재렌더링 안 함
5. **호환성**: 기존 visualization과 공존

## 🔄 업데이트 시

새로운 PDF나 assets.jsonl 변경 시:
```bash
# 다시 실행하면 자동으로 업데이트됨
run_system.bat

# 또는 맵만 다시 생성
python tools/build_figures_map.py
```

## 🚀 결과 확인

성공적으로 설정되면:
- `[Figure]` 토큰이 실제 PDF 이미지로 교체됨
- 멀티페이지 PDF는 첫 번째 페이지 표시
- 클릭하면 확대 모달 열림
- 콘솔에 로드/교체 로그 출력

### 콘솔 로그 예시
```
✅ [FIG] 메인 서버에서 로드: 3
🔄 [FIG] 토큰 교체: [Figure] → model
🔄 [FIG] 토큰 교체: [Figure] → architecture
```

### 실제 동작
1. **기존 텍스트**: `"This is [Figure] showing the network architecture."`
2. **변환 후**: `"This is "` + `<img src="/static/viz/figures/model/model_p1_abc123.png">` + `" showing the network architecture."`

## 🔧 문제 해결

### figures_map.json이 생성되지 않는 경우
1. `assets.jsonl` 파일 존재 확인
2. `source/` 폴더에 PDF/이미지 파일 존재 확인
3. PyMuPDF 설치 확인: `pip install pymupdf`
4. 수동 실행: `python tools/build_figures_map.py`

### Figure가 표시되지 않는 경우
1. 브라우저 개발자 도구에서 콘솔 로그 확인
2. Network 탭에서 `/static/viz/figures_map.json` 요청 확인
3. 이미지 URL 직접 접근 테스트

### 경로 오류가 발생하는 경우
`tools/build_figures_map.py`의 경로 설정을 실제 환경에 맞게 수정

**이제 기존 시스템을 건드리지 않고도 완전한 Figure 시스템을 사용할 수 있습니다!** 🎊
