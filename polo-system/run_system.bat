@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title POLO Launcher (Easy + Math + Server + Frontend)

REM ==============================
REM safety: find script root
REM ==============================
set "ROOT=%~dp0"
if "%ROOT%"=="" set "ROOT=."

goto :main

REM ==============================
REM UTIL: check if port is in use
REM ==============================
:CHECK_PORT_IN_USE
REM return 0 (in use) / 1 (free)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%~1 " /C:":%~1$" /C:":%~1:" 2^>NUL') do exit /b 0
exit /b 1

REM ==============================
REM UTIL: find free port >= base
REM ==============================
:FIND_FREE_PORT
set /a _p=%~1
:__find_loop
call :CHECK_PORT_IN_USE !_p!
if !errorlevel! EQU 0 (
  set /a _p=!_p!+1
  goto :__find_loop
)
set "%~2=!_p!"
exit /b 0

REM ==============================
REM Patch server\.env with ports
REM ==============================
:PATCH_ENV_FILE
set "ENV_FILE=%~1"
set "SERVER_PORT=%~2"
set "PREPROCESS_PORT=%~3"
set "EASY_PORT=%~4"
set "MATH_PORT=%~5"
set "VIZ_PORT=%~6"

if not exist "%ENV_FILE%" (
  echo [ENV] %ENV_FILE% not found. Creating...
  >"%ENV_FILE%" echo SERVER_PORT=%SERVER_PORT%
)

set "TMP_FILE=%ENV_FILE%.tmp"
if exist "%TMP_FILE%" del /f /q "%TMP_FILE%" >nul 2>&1

REM keep unknown keys, overwrite known keys
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

REM ==============================
REM venv resolve/create
REM ==============================
:RESOLVE_VENV
set "VENV_PATH=%~1"
if not exist "%VENV_PATH%" (
  echo [VENV] Creating "%VENV_PATH%"
  python -m venv "%VENV_PATH%"
  if errorlevel 1 (
    echo [ERROR] venv creation failed: %VENV_PATH%
    exit /b 1
  )
)
if exist "%VENV_PATH%\Scripts\activate.bat" (
  set "ACTIVATE=%VENV_PATH%\Scripts\activate.bat"
) else (
  set "ACTIVATE=%VENV_PATH%\Scripts\activate"
)
exit /b 0

REM ==============================
REM Launch a Python app (uvicorn)
REM ==============================
:LAUNCH_PY_APP
set "TITLE=%~1"
set "WORK_DIR=%~2"
set "REQ_FILE=%~3"
set "MODULE_APP=%~4"
set "PORT=%~5"
set "NEED_HF=%~6"
set "FALLBACK_PKGS=%~7"

echo [DEBUG] LAUNCH_PY_APP:
echo [DEBUG]   TITLE=%TITLE%
echo [DEBUG]   WORK_DIR=%WORK_DIR%
echo [DEBUG]   REQ_FILE=%REQ_FILE%
echo [DEBUG]   MODULE_APP=%MODULE_APP%
echo [DEBUG]   PORT=%PORT%
echo [DEBUG]   NEED_HF=%NEED_HF%

if not exist "%WORK_DIR%" (
  echo [ERROR] Work dir not found: %WORK_DIR%
  exit /b 1
)

call :RESOLVE_VENV "%WORK_DIR%\venv"
if not exist "%ACTIVATE%" (
  echo [ERROR] venv activate not found: %ACTIVATE%
  exit /b 1
)

set "REQ_PATH=%WORK_DIR%\%REQ_FILE%"
echo [%TITLE%] Preparing dependencies...
if exist "%REQ_PATH%" (
  if /I "%REQ_FILE%"=="pyproject.toml" (
    pushd "%WORK_DIR%"
    call "%ACTIVATE%" ^&^& python -m pip install --upgrade pip --quiet
    call "%ACTIVATE%" ^&^& python -m pip install . --quiet
    popd
  ) else (
    call "%ACTIVATE%" ^&^& python -m pip install --upgrade pip --quiet
    call "%ACTIVATE%" ^&^& python -m pip install -r "%REQ_PATH%" --quiet
  )
) else (
  if not defined FALLBACK_PKGS set "FALLBACK_PKGS=fastapi uvicorn"
  call "%ACTIVATE%" ^&^& python -m pip install --upgrade pip --quiet
  call "%ACTIVATE%" ^&^& python -m pip install %FALLBACK_PKGS% --quiet
)

echo [%TITLE%] Launching (port %PORT%)...
set "TEMP_BAT=%TEMP%\polo_%TITLE%_%PORT%.bat"
> "%TEMP_BAT%" (
  echo @echo off
  echo setlocal EnableExtensions EnableDelayedExpansion
  echo chcp 65001 ^>nul
  echo cd /d "%WORK_DIR%"
  echo call "%ACTIVATE%"
  if /I "%NEED_HF%"=="yes" (
    REM 안전 캐시 폴더 (원하면 주석 해제)
    REM echo set HF_HOME=%%TEMP%%\hf_cache
    REM echo set TRANSFORMERS_CACHE=%%TEMP%%\hf_cache
    REM echo set HF_DATASETS_CACHE=%%TEMP%%\hf_cache
    REM echo set HUGGINGFACE_HUB_CACHE=%%TEMP%%\hf_cache
    echo if defined HUGGINGFACE_TOKEN set HUGGINGFACE_TOKEN=%%HUGGINGFACE_TOKEN%%
  )
  echo uvicorn --app-dir "%WORK_DIR%" %MODULE_APP% --host 0.0.0.0 --port %PORT%
  echo echo.
  echo echo [%TITLE%] stopped. Press any key to close...
  echo pause ^>nul
)
start "POLO %TITLE% (Port %PORT%)" cmd /k "%TEMP_BAT%"
exit /b 0

REM ==============================
REM MAIN
REM ==============================
:main
echo ========================================
echo POLO System Launcher
echo ========================================

REM sanity: python / npm
where python >nul 2>&1 || (echo [ERROR] Python not found in PATH & exit /b 1)
where npm    >nul 2>&1 || (echo [WARN] npm not found in PATH. Frontend may not start.)

set "ENV_FILE=%ROOT%server\.env"

REM free ports
call :FIND_FREE_PORT 8000 SERVER_PORT
call :FIND_FREE_PORT 5002 PREPROCESS_PORT
call :FIND_FREE_PORT 5003 EASY_PORT
call :FIND_FREE_PORT 5004 MATH_PORT
call :FIND_FREE_PORT 5005 VIZ_PORT
echo [PORTS] Server:%SERVER_PORT% Preprocess:%PREPROCESS_PORT% Easy:%EASY_PORT% Math:%MATH_PORT% Viz:%VIZ_PORT%

REM patch env
call :PATCH_ENV_FILE "%ENV_FILE%" "%SERVER_PORT%" "%PREPROCESS_PORT%" "%EASY_PORT%" "%MATH_PORT%" "%VIZ_PORT%"

REM also export to current process (so child windows inherit)
for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
  if not "%%a"=="" if not "%%a:~0,1%"=="#" (
    set "%%a=%%b"
  )
)

REM dirs
set "SERVER_DIR=%ROOT%server"
set "EASY_DIR=%ROOT%models\easy"
set "MATH_DIR=%ROOT%models\math"
set "FRONTEND_DIR=%ROOT%polo-front"

set "REQ_SERVER=requirements.api.txt"
set "REQ_EASY=requirements.easy.txt"
set "REQ_MATH=requirements.math.txt"

set "FALLBACK_SERVER=fastapi uvicorn httpx pydantic asyncpg aiosqlite sqlalchemy bcrypt anyio arxiv"
set "FALLBACK_MATH=fastapi uvicorn transformers torch"

echo [1/4] Start Easy Model (Port %EASY_PORT%)
if exist "%EASY_DIR%" (
  call :LAUNCH_PY_APP "Easy" "%EASY_DIR%" "%REQ_EASY%" "app:app" "%EASY_PORT%" "yes" "fastapi uvicorn transformers torch peft"
) else (
  echo [ERROR] Easy directory not found: %EASY_DIR%
  goto :end
)

echo [2/4] Start Math Model (Port %MATH_PORT%)
if exist "%MATH_DIR%" (
  call :LAUNCH_PY_APP "Math" "%MATH_DIR%" "%REQ_MATH%" "app:app" "%MATH_PORT%" "yes" "%FALLBACK_MATH%"
) else (
  echo [ERROR] Math directory not found: %MATH_DIR%
  goto :end
)

echo [3/4] Start Backend Server (Port %SERVER_PORT%)
if exist "%SERVER_DIR%" (
  call :LAUNCH_PY_APP "Server" "%SERVER_DIR%" "%REQ_SERVER%" "app:app" "%SERVER_PORT%" "no" "%FALLBACK_SERVER%"
) else (
  echo [ERROR] Server directory not found: %SERVER_DIR%
  goto :end
)

echo [4/4] Start Frontend (Vite)
if exist "%FRONTEND_DIR%" (
  pushd "%FRONTEND_DIR%"
  if not exist ".env.local" (
    echo VITE_API_BASE=http://localhost:%SERVER_PORT%> ".env.local"
    echo [FRONTEND] wrote .env.local (VITE_API_BASE=http://localhost:%SERVER_PORT%)
  )
  if not exist "node_modules" (
    echo [FRONTEND] Installing dependencies...
    call npm install
  )
  echo [FRONTEND] Starting dev server...
  start "POLO Frontend" cmd /k "npm run dev"
  popd
) else (
  echo [WARN] Frontend directory not found: %FRONTEND_DIR%
)

echo.
echo ========================================
echo POLO System Started Successfully!
echo ========================================
echo Backend:  http://localhost:%SERVER_PORT%
echo Frontend: http://localhost:5173
echo.

:wait_loop
timeout /t 1 /nobreak >nul
goto :wait_loop

:end
echo.
echo [DONE] Some components failed to start. Check messages above.
pause
