@echo off
setlocal EnableExtensions EnableDelayedExpansion
title POLO Launcher (Easy Docker + Others Local)
goto :main

:: ================================
:: UTIL: Check if port is in use
:: ================================
:CHECK_PORT_IN_USE
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%1 " /C:":%1$" /C:":%1:" 2^>NUL') do exit /b 0
exit /b 1

:: ================================
:: UTIL: Find a free port from base
:: ================================
:FIND_FREE_PORT
set "_p=%1"
:__find_loop
call :CHECK_PORT_IN_USE !_p!
if !errorlevel! == 0 (
  set /a _p=!_p!+1
  goto :__find_loop
)
call set "%2=!_p!"
exit /b 0

:: ==========================================
:: Patch server\.env with dynamically chosen ports
:: ==========================================
:PATCH_ENV_FILE
if not exist "%ENV_FILE%" (
  echo [ENV] %ENV_FILE% not found. Creating...
  >"%ENV_FILE%" echo SERVER_PORT=%SERVER_PORT%
)

set "TMP_FILE=%ENV_FILE%.tmp"

:: 배치 파서가 깨지지 않도록 한 줄씩 유지 (줄연속 캐럿 주의)
> "%TMP_FILE%" (
  for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
    if not "%%a"=="" if /I not "%%a"=="SERVER_PORT" if /I not "%%a"=="PREPROCESS_URL" if /I not "%%a"=="EASY_MODEL_URL" if /I not "%%a"=="MATH_MODEL_URL" if /I not "%%a"=="VIZ_MODEL_URL" if /I not "%%a"=="CALLBACK_URL" (
      echo %%a=%%b
    )
  )
  echo SERVER_PORT=%SERVER_PORT%
  echo PREPROCESS_URL=http://localhost:%PREPROCESS_PORT%
  echo EASY_MODEL_URL=http://localhost:%EASY_PORT%
  echo MATH_MODEL_URL=http://localhost:%MATH_PORT%
  echo VIZ_MODEL_URL=http://localhost:%VIZ_PORT%
  echo CALLBACK_URL=http://localhost:%SERVER_PORT%
)

move /Y "%TMP_FILE%" "%ENV_FILE%" >nul
echo [ENV] Patched %ENV_FILE% with dynamic ports.
exit /b 0

:: =================================
:: Resolve / create venv (safe quotes)
:: =================================
:RESOLVE_VENV
set "VENV_PATH=%~1"
if not exist "%VENV_PATH%" (
  echo [VENV] Creating "%VENV_PATH%"
  python -m venv "%VENV_PATH%"
)
if exist "%VENV_PATH%\Scripts\activate.bat" (
  set "ACTIVATE=%VENV_PATH%\Scripts\activate.bat"
) else (
  set "ACTIVATE=%VENV_PATH%\Scripts\activate"
)
echo [DEBUG] VENV_PATH=%VENV_PATH%
echo [DEBUG] ACTIVATE=%ACTIVATE%
exit /b 0

:: =================================
:: Launch a Python app (uvicorn)
:: =================================
:LAUNCH_PY_APP
set "TITLE=%~1"
set "WORK_DIR=%~2"
set "REQ_FILE=%~3"
set "MODULE_APP=%~4"
set "PORT=%~5"
set "NEED_HF=%~6"
set "FALLBACK_PKGS=%~7"

echo [DEBUG] LAUNCH_PY_APP called with:
echo [DEBUG]   TITLE=%TITLE%
echo [DEBUG]   WORK_DIR=%WORK_DIR%
echo [DEBUG]   REQ_FILE=%REQ_FILE%
echo [DEBUG]   MODULE_APP=%MODULE_APP%
echo [DEBUG]   PORT=%PORT%
echo [DEBUG]   NEED_HF=%NEED_HF%

:: venv 구성 (인자에 따옴표 포함해서 전달 → 내부에서 %~1로 제거)
call :RESOLVE_VENV "%WORK_DIR%\venv"
echo [DEBUG] RESOLVE_VENV completed, ACTIVATE=%ACTIVATE%
if not exist "%ACTIVATE%" (
  echo [ERROR] Virtual environment not found: %ACTIVATE%
  exit /b 1
)

echo [%TITLE%] Starting on port %PORT%...
set "REQ_PATH=%WORK_DIR%\%REQ_FILE%"
echo [DEBUG] Looking for requirements file: %REQ_PATH%

if exist "%REQ_PATH%" (
  echo [%TITLE%] Requirements file found: %REQ_FILE%
  if /I "%REQ_FILE%"=="pyproject.toml" (
    pushd "%WORK_DIR%"
    call "%ACTIVATE%" ^&^& python -m pip install . --quiet
    popd
  ) else (
    call "%ACTIVATE%" ^&^& python -m pip install -r "%REQ_PATH%" --quiet
  )
) else (
  echo [%TITLE%] Requirements file not found: %REQ_PATH%
  if not defined FALLBACK_PKGS set "FALLBACK_PKGS=fastapi uvicorn"
  call "%ACTIVATE%" ^&^& python -m pip install %FALLBACK_PKGS% --quiet
)

echo [%TITLE%] Launching in separate window...
set "TEMP_BAT=%TEMP%\polo_%TITLE%_%PORT%.bat"
> "%TEMP_BAT%" (
  echo @echo off
  echo setlocal
  echo cd /d "%WORK_DIR%"
  echo call "%ACTIVATE%"
  if /I "%NEED_HF%"=="yes" (
    echo set HF_HOME=%%TEMP%%\hf_cache
    echo set TRANSFORMERS_CACHE=%%TEMP%%\hf_cache
    echo set HF_DATASETS_CACHE=%%TEMP%%\hf_cache
    echo if defined HUGGINGFACE_TOKEN set HUGGINGFACE_TOKEN=%%HUGGINGFACE_TOKEN%%
  )
  echo echo [%TITLE%] Environment ready, starting uvicorn...
  echo uvicorn --app-dir "%WORK_DIR%" %MODULE_APP% --host 0.0.0.0 --port %PORT%
  echo pause
)

start "POLO %TITLE% (Port %PORT%)" cmd /k "%TEMP_BAT%"
exit /b 0

:: ================================
:: MAIN
:: ================================
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

if exist "%ENV_FILE%" (
  echo [ENV] Loading environment variables from %ENV_FILE%...
  for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" (
      set "%%a=%%b"
      echo   %%a=%%b
    )
  )
) else (
  echo [WARNING] .env file not found: %ENV_FILE%
)

set "SERVER_DIR=%ROOT%server"
set "EASY_DIR=%ROOT%models\easy"
set "MATH_DIR=%ROOT%models\math"
set "FRONTEND_DIR=%ROOT%polo-front"

set "REQ_SERVER=requirements.api.txt"
set "REQ_EASY=requirements.easy.txt"
set "REQ_MATH=requirements.math.txt"

set "FALLBACK_SERVER=fastapi uvicorn httpx pydantic asyncpg aiosqlite sqlalchemy bcrypt anyio arxiv"
set "FALLBACK_MATH=fastapi uvicorn transformers torch"

echo [1/6] Start Easy Model (Port %EASY_PORT%)
if exist "%EASY_DIR%" (
  call :LAUNCH_PY_APP "Easy" "%EASY_DIR%" "%REQ_EASY%" "app:app" "%EASY_PORT%" "yes" "fastapi uvicorn transformers torch peft"
) else (
  echo [ERROR] Easy directory not found: %EASY_DIR%
  exit /b 1
)

echo [2/6] Start Math Model (Port %MATH_PORT%)
if exist "%MATH_DIR%" (
  call :LAUNCH_PY_APP "Math" "%MATH_DIR%" "%REQ_MATH%" "app:app" "%MATH_PORT%" "yes" "%FALLBACK_MATH%"
) else (
  echo [ERROR] Math directory not found: %MATH_DIR%
  exit /b 1
)

echo [3/6] Start Backend Server (Port %SERVER_PORT%)
if exist "%SERVER_DIR%" (
  call :LAUNCH_PY_APP "Server" "%SERVER_DIR%" "%REQ_SERVER%" "app:app" "%SERVER_PORT%" "no" "%FALLBACK_SERVER%"
) else (
  echo [ERROR] Server directory not found: %SERVER_DIR%
  exit /b 1
)

echo [6/6] Start Frontend
if exist "%FRONTEND_DIR%" (
  pushd "%FRONTEND_DIR%"
  if not exist "node_modules" (
    echo [FRONTEND] Installing dependencies...
    call npm install
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
