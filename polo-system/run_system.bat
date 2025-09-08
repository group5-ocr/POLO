@echo off
echo ========================================
echo POLO 시스템 실행 스크립트
echo ========================================

echo.
echo 1. 도커 서비스 시작 (AI 모델 파인튜닝)
echo ========================================
echo Easy LLM 서비스를 시작합니다...
docker compose up -d easy-llm

echo.
echo 2. 백엔드 서버 시작
echo ========================================
echo 백엔드 서버를 시작합니다...
cd server
start "POLO Backend" cmd /k "venv\Scripts\activate && uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
cd ..

echo.
echo 3. 프론트엔드 시작
echo ========================================
echo 프론트엔드를 시작합니다...
cd polo-front
start "POLO Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ========================================
echo 모든 서비스가 시작되었습니다!
echo ========================================
echo.
echo 서비스 접속 주소:
echo - 프론트엔드: http://localhost:5173
echo - 백엔드 API: http://localhost:8000
echo - Easy LLM: http://localhost:5003
echo.
echo 도커 서비스 상태 확인:
echo docker ps
echo.
echo 도커 로그 확인:
echo docker compose logs easy-llm
echo.
pause
