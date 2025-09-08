@echo off
setlocal ENABLEDELAYEDEXPANSION
title POLO Launcher (CMD)

echo ========================================
echo POLO - Run all (Backend, Easy Model, Frontend)
echo ========================================
echo.

REM 1) Backend (8000)
echo [1/3] Start Backend (8000)
pushd "%~dp0server"
if not exist venv (python -m venv venv)
start "POLO Backend" cmd /k "call venv\Scripts\activate.bat && pip install -r requirements.api.txt && uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
popd
echo   -> launched
echo.

REM 2) Easy Model (5003)
echo [2/3] Start Easy Model (5003)
pushd "%~dp0models\easy"
if not exist venv (python -m venv venv)
start "POLO Easy Model" cmd /k "call venv\Scripts\activate.bat && pip install -r requirements.easy.txt && uvicorn app:app --host 0.0.0.0 --port 5003"
popd
echo   -> launched
echo.

REM 3) Frontend (5173)
echo [3/3] Start Frontend (5173)
pushd "%~dp0polo-front"
start "POLO Frontend" cmd /k "npm install && npm run dev"
popd
echo   -> launched
echo.

echo ========================================
echo All services launched.
echo - Frontend : http://localhost:5173
echo - Backend  : http://localhost:8000
echo - Easy LLM : http://localhost:5003
echo ========================================
echo.
echo (Optional) Start fine-tuning in Docker:
echo   cd %~dp0 ^&^& docker compose up -d easy-train
echo.
pause
