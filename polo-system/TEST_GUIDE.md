# POLO 시스템 테스트 가이드

## 🚀 빠른 시작

### 1. 시스템 실행
```bash
# Windows (PowerShell)
.\run_system.ps1

# Windows (CMD)
run_system.bat
```

### 2. 수동 실행 (단계별)
```bash
# 1. 도커 서비스 시작
docker compose up -d easy-llm

# 2. 백엔드 서버 시작 (새 터미널)
cd server
venv\Scripts\activate  # Windows
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 3. 프론트엔드 시작 (새 터미널)
cd polo-front
npm run dev
```

## 🔍 서비스 상태 확인

### 도커 서비스 확인
```bash
# 컨테이너 상태 확인
docker ps

# Easy LLM 로그 확인
docker compose logs easy-llm

# 모델 로딩 상태 확인
curl http://localhost:5003/health
```

### 백엔드 API 확인
```bash
# API 상태 확인
curl http://localhost:8000/health

# 모델 연결 상태 확인
curl http://localhost:8000/api/model-status
```

## 📄 PDF 테스트

### 1. 웹 인터페이스 테스트
1. 브라우저에서 http://localhost:5173 접속
2. "Upload" 페이지로 이동
3. "AI 모델 상태 확인" 버튼 클릭
4. PDF 파일 업로드 및 변환 테스트

### 2. API 직접 테스트
```bash
# PDF 파일 업로드 테스트
curl -X POST "http://localhost:8000/api/convert" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_test_file.pdf"
```

## 🐛 문제 해결

### 도커 관련 문제
```bash
# 컨테이너 재시작
docker compose restart easy-llm

# 컨테이너 재빌드
docker compose build --no-cache easy-llm
docker compose up -d easy-llm

# 컨테이너 내부 접속
docker exec -it polo-system-easy-llm-1 bash
```

### 모델 로딩 문제
- GPU 메모리 부족: `docker compose logs easy-llm`으로 확인
- 모델 파일 경로: `outputs/llama32-3b-qlora/checkpoint-600/` 확인
- Hugging Face 토큰: `.env` 파일에 `HUGGINGFACE_TOKEN` 설정

### 백엔드 연결 문제
```bash
# Python 의존성 재설치
cd server
pip install -r requirements.api.txt --force-reinstall

# 포트 충돌 확인
netstat -an | findstr :8000
```

### 프론트엔드 문제
```bash
# Node.js 의존성 재설치
cd polo-front
rm -rf node_modules package-lock.json
npm install

# 포트 충돌 확인
netstat -an | findstr :5173
```

## 📊 성능 모니터링

### 도커 리소스 사용량
```bash
# 컨테이너 리소스 사용량 확인
docker stats

# GPU 사용량 확인 (NVIDIA)
nvidia-smi
```

### API 응답 시간
```bash
# API 응답 시간 측정
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/model-status"
```

## 🔧 설정 파일

### 환경 변수 (.env)
```bash
# polo-system/server/.env
HUGGINGFACE_TOKEN=your_token_here
```

### 도커 설정
- `docker-compose.yml`: 모델 서비스 설정
- `models/fine-tuning/dockerfile`: 모델 컨테이너 설정

## 📝 로그 확인

### 도커 로그
```bash
# 실시간 로그 확인
docker compose logs -f easy-llm

# 특정 시간대 로그
docker compose logs --since="2024-01-01T00:00:00" easy-llm
```

### 백엔드 로그
- FastAPI 자동 리로드 시 콘솔에 로그 출력
- 에러 발생 시 상세한 스택 트레이스 확인

### 프론트엔드 로그
- 브라우저 개발자 도구 콘솔에서 확인
- 네트워크 탭에서 API 요청/응답 확인

## ✅ 테스트 체크리스트

- [ ] 도커 서비스가 정상적으로 시작됨
- [ ] Easy LLM 모델이 로드됨 (checkpoint-600)
- [ ] 백엔드 API가 8000 포트에서 실행됨
- [ ] 프론트엔드가 5173 포트에서 실행됨
- [ ] AI 모델 상태 확인 API 응답
- [ ] PDF 업로드 및 텍스트 추출 작동
- [ ] AI 모델을 통한 텍스트 변환 작동
- [ ] 에러 처리 및 사용자 피드백 작동

## 🎯 예상 결과

정상적으로 작동하면:
1. PDF 파일 업로드 시 텍스트 추출
2. 추출된 텍스트를 AI 모델로 변환
3. 사용자 친화적인 형태로 결과 표시
4. 에러 발생 시 적절한 메시지 표시
