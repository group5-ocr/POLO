# POLO 로컬 실행 가이드

## 전체 구조
- **도커**: 모델 파인튜닝만 (easy-train, easy-llm)
- **로컬**: 서버(backend), 프론트엔드(polo-front)

## 1. 도커 서비스 실행 (모델 파인튜닝)

### 파인튜닝 실행
```bash
# 도커 빌드
docker compose build --no-cache easy-train

# 파인튜닝 실행
docker compose up easy-train
```

### 파인튜닝된 모델 서빙 (선택사항)
```bash
# easy-llm 서비스 실행
docker compose up easy-llm
```

## 2. 로컬 서버 실행 (Backend)

### 환경 설정
```bash
cd polo-system/server

# 가상환경 생성 (Python 3.8+)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 의존성 설치
pip install -r requirements.api.txt
```

### 서버 실행
```bash
# FastAPI 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

서버가 http://localhost:8000 에서 실행됩니다.

## 3. 로컬 프론트엔드 실행 (React)

### 환경 설정
```bash
cd polo-system/polo-front

# 의존성 설치
npm install
```

### 개발 서버 실행
```bash
# 개발 서버 실행
npm run dev
```

프론트엔드가 http://localhost:5173 에서 실행됩니다.

## 4. 전체 시스템 실행 순서

1. **도커 서비스 시작** (모델 파인튜닝)
   ```bash
   docker compose up easy-train
   ```

2. **백엔드 서버 시작** (새 터미널)
   ```bash
   cd polo-system/server
   venv\Scripts\activate  # Windows
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **프론트엔드 시작** (새 터미널)
   ```bash
   cd polo-system/polo-front
   npm run dev
   ```

## 5. 파인튜닝 결과 확인

현재 파인튜닝된 모델은 다음 위치에 있습니다:
- `polo-system/models/fine-tuning/outputs/llama32-3b-qlora/checkpoint-600/`

이 모델을 사용하려면:
1. `easy-llm` 서비스를 실행하거나
2. 서버에서 직접 모델을 로드하여 사용

## 6. 환경 변수 설정

`.env` 파일을 생성하여 필요한 환경 변수를 설정하세요:
```bash
# polo-system/server/.env
HUGGINGFACE_TOKEN=your_token_here
```

## 7. 포트 정보

- **프론트엔드**: http://localhost:5173
- **백엔드 API**: http://localhost:8000
- **Easy LLM**: http://localhost:5003 (도커)
- **Math LLM**: http://localhost:5001 (도커, 필요시)

## 8. 문제 해결

### 도커 관련
```bash
# 도커 컨테이너 상태 확인
docker ps

# 로그 확인
docker compose logs easy-train
```

### Python 관련
```bash
# 의존성 재설치
pip install -r requirements.api.txt --force-reinstall
```

### Node.js 관련
```bash
# node_modules 재설치
rm -rf node_modules package-lock.json
npm install
```
