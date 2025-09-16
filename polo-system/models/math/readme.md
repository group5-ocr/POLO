# 이 순서대로 진행하시면 됩니다!

pip install transformers==4.44.2
pip install bitsandbytes==0.44.1
python install accelerate==0.33.0
후 프로그램 종료 후 다시 벤브

# GCP 설정

model 안에 stone-booking-466716-n6-f6fff7380e05.json 을 넣는다!!

SERVICE_ACCOUNT_PATH 상의 경로를 본인에 맞게 수정한다

# 실행

cd polo-system/models/math
uvicorn app:app --port 5004

# 호출

- 수식만 세기
  http://127.0.0.1:5004/count/C:/POLO/POLO/polo-system/models/math/yolo.tex
  http://127.0.0.1:5004/count/C:\POLO\POLO\polo-system\models\math\iclr2022_conference.tex

- JSON/tex 파일 받기
  http://127.0.0.1:5004/math/C:/POLO/POLO/polo-system/models/math/yolo.tex
  http://127.0.0.1:5004/math/C:\POLO\POLO\polo-system\models\math\iclr2022_conference.tex

- 건강상태? 확인
  http://127.0.0.1:5004/health

  이때 결과가
  {
  "status": "ok",
  "python": "3.11.9",
  "torch": "2.5.1+cu121",
  "cuda": true,
  "device": "cuda",
  "model_loaded": true,
  "gcp_translate_ready": true,
  "gcp_parent": "projects/stone-booking-466716-n6/locations/global"
  }

이렇게 나오면 성공!

# 훤용

주석 확인할 것
OUT_DIR
SERVICE_ACCOUNT_PATH

http://127.0.0.1:5004/count/C:/POLO/polo-system/models/math/yolo.tex
http://127.0.0.1:5004/math/C:/POLO/polo-system/models/math/yolo.tex

---

# POLO Math Explainer API

LaTeX 문서에서 수식을 자동으로 추출‧분류하고, \*\*영어 요약(Overview)\*\*과 \*\*영어 수식 해설(Example → Explanation → Conclusion 형식)\*\*을 생성하여 JSON/TeX 리포트로 저장하는 FastAPI 서비스입니다.

- 모델: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- 프롬프트: **영어**(요청에 따라 조정 가능)
- 로그: 콘솔에 **즉시 출력**되도록 설정
- 경고 회피: `pad_token`/`attention_mask` 처리로 주의 메시지 최소화

---

## 주요 기능

- **수식 자동 추출**: $…$, $…$, $…$, inline `$…$`, 그리고 `equation`, `align` 등 환경 검색
- **난이도 분류**: 간단한 휴리스틱(특정 토큰, 길이, 첨자 수 등)으로 “중학생 이상” 수식 필터링
- **문서 개요 생성** (영어): 문서 앞/중/뒤 슬라이스 기반 요약
- **수식 해설 생성** (영어): Example → Explanation → Conclusion 포맷
- **산출물 저장**: JSON(`equations_explained.json`), LaTeX 리포트(`yolo_math_report.tex`)
- **경량 API**: `/count`(개수만), `/math`(풀 파이프라인), `/health`(상태)

---

## 시스템 요구 사항

- Python 3.11.9
- PyTorch (CUDA 권장)
  예: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  (현재 예시는 CUDA 12.1 기반을 염두에 두었습니다.)
- 기타 라이브러리: `fastapi`, `uvicorn`, `transformers`, `pydantic`

> GPU가 없으셔도 동작은 가능하지만, LLM 생성 속도가 느릴 수 있습니다.

## 코드 구조 & 처리 흐름

```
app.py
 ├─ (셀 1) 환경 준비 & 모델 로드
 │    ├─ 토크나이저/모델 로드, PAD 토큰/attention_mask 설정
 │    └─ 생성 파라미터(GEN_KW) 설정
 ├─ (공통) 라인 오프셋 유틸
 ├─ (셀 2) 수식 추출
 │    └─ $$, \[ \], \( \), inline $ $ , 환경(equation, align,...) 탐지
 ├─ (셀 3) 난이도 휴리스틱
 │    └─ 특정 토큰/길이/첨자/줄바꿈 기반으로 “중학생 이상” 판정
 ├─ (셀 4) 문서 개요 LLM
 │    └─ 앞/중/뒤 슬라이스를 영어로 간단 요약
 ├─ (셀 5) 수식 해설 LLM
 │    └─ Example → Explanation → Conclusion 형식(영어)
 ├─ (셀 6) 리포트 생성
 │    └─ JSON/TeX 저장
 └─ FastAPI 엔드포인트 정의
```

---

## 사용 방법

### 1) 건강 상태 확인

- `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### 2) 수식 개수만 빠르게 확인

- `GET /count/{file_path:path}`

```bash
# Windows PowerShell 예시(백슬래시 이스케이프 주의)
curl "http://127.0.0.1:8000/count/C:%5CPOLO%5Cpolo-system%5Cmodels%5Cmath%5Cyolo.tex"
```

- `POST /count`

```bash
curl -X POST "http://127.0.0.1:8000/count" ^
  -H "Content-Type: application/json" ^
  -d "{\"path\": \"C:\\\\POLO\\\\polo-system\\\\models\\\\math\\\\yolo.tex\"}"
```

**응답 예시**

```json
{
  "총 수식": 42,
  "중학생 수준 이상": 19
}
```

### 3) 전체 파이프라인 실행

- `GET /math/{file_path:path}`

```bash
curl "http://127.0.0.1:8000/math/C:%5CPOLO%5Cpolo-system%5Cmodels%5Cmath%5Cyolo.tex"
```

- `POST /math`

```bash
curl -X POST "http://127.0.0.1:8000/math" ^
  -H "Content-Type: application/json" ^
  -d "{\"path\": \"C:\\\\POLO\\\\polo-system\\\\models\\\\math\\\\yolo.tex\"}"
```

**응답 예시**

```json
{
  "input": "C:\\POLO\\polo-system\\models\\math\\yolo.tex",
  "counts": {
    "총 수식": 42,
    "중학생 수준 이상": 19
  },
  "outputs": {
    "json": "C:/POLO/polo-system/models/math/_build/equations_explained.json",
    "report_tex": "C:/POLO/polo-system/models/math/_build/yolo_math_report.tex",
    "out_dir": "C:/POLO/polo-system/models/math/_build"
  }
}
```

---

## 산출물

- **`_build/equations_explained.json`**

  - `overview`: 문서 개요(영어)
  - `items[]`: 각 고난도 수식에 대한 해설(영어), 라인 범위/종류/환경 등 메타 포함

- **`_build/yolo_math_report.tex`**

  - 문서 개요 + 수식별 해설 섹션으로 구성된 LaTeX 리포트

> `kotex`를 사용하므로 Windows에서는 MiKTeX에 `kotex` 패키지가 필요합니다. 안되면 XeLaTeX/xeCJK로 컴파일해 보시길 권장드립니다.

---

## 추후 개선 아이디어

- 문서 구조(섹션/하위섹션) 파싱 강화 및 섹션별 해설 연결
- 수식 렌더링(preview)용 PNG/SVG 산출 옵션 추가
- 긴 문서에 대한 분산 배치 처리 및 재시도 로직
- 한국어 해설 템플릿 옵션(토글) 제공
