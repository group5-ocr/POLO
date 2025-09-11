@echo off
setlocal EnableExtensions EnableDelayedExpansion
title POLO Launcher (Easy Docker + Others Local)

goto :main

:CHECK_PORT_IN_USE
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%1 " /C:":%1$" /C:":%1:" 2^>NUL') do ( exit /b 0 )
exit /b 1

:FIND_FREE_PORT
set "_p=%1"
:__find_loop
call :CHECK_PORT_IN_USE !_p!
if %errorlevel%==0 (
  set /a _p=!_p!+1
  goto :__find_loop
)
call set "%2=!_p!"
exit /b 0

:PATCH_ENV_FILE
if not exist "%ENV_FILE%" (
  echo [ENV] %ENV_FILE% not found. Creating...
  >"%ENV_FILE%" echo SERVER_PORT=%SERVER_PORT%
)

set "TMP_FILE=%ENV_FILE%.tmp"
findstr /V /B /C:"SERVER_PORT=" /C:"PREPROCESS_URL=" /C:"EASY_MODEL_URL=" /C:"MATH_MODEL_URL=" /C:"VIZ_MODEL_URL=" /C:"CALLBACK_URL=" "%ENV_FILE%" > "%TMP_FILE%"

>>"%TMP_FILE%" echo SERVER_PORT=%SERVER_PORT%
>>"%TMP_FILE%" echo PREPROCESS_URL=http://localhost:%PREPROCESS_PORT%
>>"%TMP_FILE%" echo EASY_MODEL_URL=http://localhost:%EASY_PORT%
>>"%TMP_FILE%" echo MATH_MODEL_URL=http://localhost:%MATH_PORT%
>>"%TMP_FILE%" echo VIZ_MODEL_URL=http://localhost:%VIZ_PORT%
>>"%TMP_FILE%" echo CALLBACK_URL=http://localhost:%SERVER_PORT%

move /Y "%TMP_FILE%" "%ENV_FILE%" >nul
echo [ENV] Patched %ENV_FILE% with dynamic ports.
exit /b 0

:RESOLVE_VENV
if not exist "%1" (
  echo [VENV] Creating %1
  python -m venv "%1"
)
if exist "%1\Scripts\activate.bat" (
  set "ACTIVATE=%1\Scripts\activate.bat"
) else (
  set "ACTIVATE=%1\Scripts\activate"
)
exit /b 0

:LAUNCH_PY_APP
set "TITLE=%1"
set "WORK_DIR=%2"
set "REQ_FILE=%3"
set "MODULE_APP=%4"
set "PORT=%5"
set "NEED_HF=%6"
set "FALLBACK_PKGS=%7"

call :RESOLVE_VENV "%WORK_DIR%\venv"
if not exist "%ACTIVATE%" (
  echo [ERROR] Virtual environment not found: %ACTIVATE%
  exit /b 1
)

echo [%TITLE%] Starting on port %PORT%...
if exist "%WORK_DIR%\%REQ_FILE%" (
  echo [%TITLE%] Installing requirements...
  call "%ACTIVATE%" && pip install -r "%REQ_FILE%" --quiet
) else (
  echo [%TITLE%] Installing fallback packages: %FALLBACK_PKGS%
  call "%ACTIVATE%" && pip install %FALLBACK_PKGS% --quiet
)

if "%NEED_HF%"=="yes" (
  if defined HUGGINGFACE_TOKEN (
    call "%ACTIVATE%" && set HUGGINGFACE_TOKEN=%HUGGINGFACE_TOKEN% && uvicorn %MODULE_APP% --host 0.0.0.0 --port %PORT%
  ) else (
    echo [WARNING] HUGGINGFACE_TOKEN not set for %TITLE%
    call "%ACTIVATE%" && uvicorn %MODULE_APP% --host 0.0.0.0 --port %PORT%
  )
) else (
  call "%ACTIVATE%" && uvicorn %MODULE_APP% --host 0.0.0.0 --port %PORT%
)
exit /b 0

:WAIT_FOR_HTTP
set "URL=%1"
set "SERVICE=%2"
set "MAX_ATTEMPTS=%3"
set "ATTEMPT=0"

:__wait_loop
set /a ATTEMPT=!ATTEMPT!+1
if !ATTEMPT! GTR %MAX_ATTEMPTS% (
  echo [%SERVICE%] Timeout waiting for %URL%
  exit /b 1
)

curl -s "%URL%" >nul 2>&1
if %errorlevel%==0 (
  echo [%SERVICE%] Ready at %URL%
  exit /b 0
)

echo [%SERVICE%] Waiting... (!ATTEMPT!/%MAX_ATTEMPTS%)
timeout /t 2 /nobreak >nul
goto :__wait_loop

:main
set "ROOT=%~dp0"
set "ENV_FILE=%ROOT%server\.env"

echo ========================================
echo POLO System Launcher (Easy Docker + Others Local)
echo ========================================

call :FIND_FREE_PORT 8000 SERVER_PORT
call :FIND_FREE_PORT 5002 PREPROCESS_PORT
call :FIND_FREE_PORT 5003 EASY_PORT
call :FIND_FREE_PORT 5004 MATH_PORT
call :FIND_FREE_PORT 5005 VIZ_PORT

echo [PORTS] Server:%SERVER_PORT% Preprocess:%PREPROCESS_PORT% Easy:%EASY_PORT% Math:%MATH_PORT% Viz:%VIZ_PORT%

call :PATCH_ENV_FILE

set "SERVER_DIR=%ROOT%server"
set "PREPROCESS_DIR=%ROOT%preprocessing\texprep"
set "MATH_DIR=%ROOT%models\math"
set "VIZ_DIR=%ROOT%viz"
set "FRONTEND_DIR=%ROOT%polo-front"

set "REQ_SERVER=requirements.api.txt"
set "REQ_PREPROCESS=pyproject.toml"
set "REQ_MATH=requirements.math.txt"
set "REQ_VIZ=requirements.viz.txt"

set "FALLBACK_SERVER=fastapi uvicorn httpx pydantic asyncpg aiosqlite sqlalchemy bcrypt anyio arxiv"
set "FALLBACK_PREPROCESS=fastapi uvicorn pydantic"
set "FALLBACK_MATH=fastapi uvicorn transformers torch"
set "FALLBACK_VIZ=fastapi uvicorn matplotlib seaborn plotly"

echo [1/6] Starting Backend Server (Local, port %SERVER_PORT%)
if exist "%SERVER_DIR%" (
  call :LAUNCH_PY_APP "POLO Server" "%SERVER_DIR%" %REQ_SERVER% app:app %SERVER_PORT% no "%FALLBACK_SERVER%"
) else (
  echo [ERROR] Server directory not found: %SERVER_DIR%
  exit /b 1
)

call :WAIT_FOR_HTTP http://localhost:%SERVER_PORT%/openapi.json Backend 40

echo [2/6] Start Easy Model (Docker, port %EASY_PORT%)
pushd "%ROOT%models\fine-tuning"
set "EASY_PORT=%EASY_PORT%"
docker compose up -d --build easy-llm
popd

echo [3/6] Start Preprocess (%PREPROCESS_PORT%)
if exist "%PREPROCESS_DIR%" (
  call :LAUNCH_PY_APP "POLO Preprocess" "%PREPROCESS_DIR%" %REQ_PREPROCESS% app:app %PREPROCESS_PORT% no "%FALLBACK_PREPROCESS%"
) else (
  echo [ERROR] Preprocess directory not found: %PREPROCESS_DIR%
  exit /b 1
)

echo [4/6] Start Math Model (%MATH_PORT%)
if exist "%MATH_DIR%" (
  call :LAUNCH_PY_APP "POLO Math" "%MATH_DIR%" %REQ_MATH% app:app %MATH_PORT% yes "%FALLBACK_MATH%"
) else (
  echo [ERROR] Math directory not found: %MATH_DIR%
  exit /b 1
)

echo [5/6] Start Viz Model (%VIZ_PORT%)
if exist "%VIZ_DIR%" (
  call :LAUNCH_PY_APP "POLO Viz" "%VIZ_DIR%" %REQ_VIZ% app:app %VIZ_PORT% no "%FALLBACK_VIZ%"
) else (
  echo [ERROR] Viz directory not found: %VIZ_DIR%
  exit /b 1
)

echo [6/6] Start Frontend
if exist "%FRONTEND_DIR%" (
  pushd "%FRONTEND_DIR%"
  if not exist "node_modules" (
    echo [FRONTEND] Installing dependencies...
    npm install
  )
  echo [FRONTEND] Starting development server...
  start "POLO Frontend" cmd /k "npm run dev"
  popd
) else (
  echo [ERROR] Frontend directory not found: %FRONTEND_DIR%
  exit /b 1
)

echo.
echo ========================================
echo POLO System Started Successfully!
echo ========================================
echo Backend:  http://localhost:%SERVER_PORT%
echo Frontend: http://localhost:5173
echo.
echo Press Ctrl+C to stop all services
echo.

:wait_loop
timeout /t 1 /nobreak >nul
goto :wait_loop