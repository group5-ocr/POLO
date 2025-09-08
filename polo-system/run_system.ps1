# POLO 시스템 실행 스크립트 (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "POLO 시스템 실행 스크립트" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "1. 도커 서비스 시작 (AI 모델 파인튜닝)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Easy LLM 서비스를 시작합니다..." -ForegroundColor Green

docker compose up -d easy-llm

Write-Host ""
Write-Host "2. 백엔드 서버 시작" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "백엔드 서버를 시작합니다..." -ForegroundColor Green

Set-Location server
Start-Process powershell -ArgumentList "-NoExit", "-Command", "venv\Scripts\Activate.ps1; uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
Set-Location ..

Write-Host ""
Write-Host "3. 프론트엔드 시작" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "프론트엔드를 시작합니다..." -ForegroundColor Green

Set-Location polo-front
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev"
Set-Location ..

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "모든 서비스가 시작되었습니다!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "서비스 접속 주소:" -ForegroundColor White
Write-Host "- 프론트엔드: http://localhost:5173" -ForegroundColor Green
Write-Host "- 백엔드 API: http://localhost:8000" -ForegroundColor Green
Write-Host "- Easy LLM: http://localhost:5003" -ForegroundColor Green
Write-Host ""
Write-Host "도커 서비스 상태 확인:" -ForegroundColor White
Write-Host "docker ps" -ForegroundColor Gray
Write-Host ""
Write-Host "도커 로그 확인:" -ForegroundColor White
Write-Host "docker compose logs easy-llm" -ForegroundColor Gray
Write-Host ""

Read-Host "Press Enter to continue"
