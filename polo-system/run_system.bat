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
set "VENV_PATH=%~1"
if not exist "%VENV_PATH%" (
  echo [VENV] Creating %VENV_PATH%
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

:LAUNCH_PY_APP
set "TITLE=%1"
set "WORK_DIR=%2"
set "REQ_FILE=%3"
set "MODULE_APP=%4"
set "PORT=%5"
set "NEED_HF=%6"
set "FALLBACK_PKGS=%7"

echo [DEBUG] LAUNCH_PY_APP called with:
echo [DEBUG]   TITLE=%TITLE%
echo [DEBUG]   WORK_DIR=%WORK_DIR%
echo [DEBUG]   REQ_FILE=%REQ_FILE%
echo [DEBUG]   MODULE_APP=%MODULE_APP%
echo [DEBUG]   PORT=%PORT%
echo [DEBUG]   NEED_HF=%NEED_HF%

call :RESOLVE_VENV "%WORK_DIR%\venv"
echo [DEBUG] RESOLVE_VENV completed, ACTIVATE=%ACTIVATE%
if not exist "%ACTIVATE%" (
  echo [ERROR] Virtual environment not found: %ACTIVATE%
  echo [DEBUG] WORK_DIR=%WORK_DIR%
  echo [DEBUG] Looking for venv in: %WORK_DIR%\venv
  exit /b 1
)

echo [%TITLE%] Starting on port %PORT%...
set "REQ_PATH=%WORK_DIR%\%REQ_FILE%"
echo [DEBUG] Looking for requirements file: %REQ_PATH%

REM 패키지 설치 여부 확인
set "INSTALL_NEEDED=0"
if exist "%REQ_PATH%" (
  echo [%TITLE%] Requirements file found: %REQ_FILE%
  if /I "%REQ_FILE%"=="pyproject.toml" (
    pushd "%WORK_DIR%"
    call "%ACTIVATE%" && python -m pip install . --quiet
    popd
  ) else (
    echo [%TITLE%] Installing requirements (quiet mode)...
    call "%ACTIVATE%" && python -m pip install -r "%REQ_PATH%" --quiet
  )
) else (
  echo [%TITLE%] Requirements file not found: %REQ_PATH%
  echo [%TITLE%] Installing fallback packages (quiet mode)...
  call "%ACTIVATE%" && python -m pip install %FALLBACK_PKGS% --quiet
)

echo [%TITLE%] Launching in separate window...
echo [DEBUG] WORK_DIR=%WORK_DIR%
echo [DEBUG] ACTIVATE=%ACTIVATE%

REM 임시 배치 파일 생성
set "TEMP_BAT=%TEMP%\polo_%TITLE%_%PORT%.bat"
echo @echo off > "%TEMP_BAT%"
echo echo [%TITLE%] Starting service... >> "%TEMP_BAT%"
echo cd /d "%WORK_DIR%" >> "%TEMP_BAT%"
echo call "%ACTIVATE%" >> "%TEMP_BAT%"

REM Hugging Face 환경변수 설정
if "%NEED_HF%"=="yes" (
  echo set HF_HOME=%TEMP%\hf_cache >> "%TEMP_BAT%"
  echo set TRANSFORMERS_CACHE=%TEMP%\hf_cache >> "%TEMP_BAT%"
  echo set HF_DATASETS_CACHE=%TEMP%\hf_cache >> "%TEMP_BAT%"
  if defined HUGGINGFACE_TOKEN (
    echo set HUGGINGFACE_TOKEN=%HUGGINGFACE_TOKEN% >> "%TEMP_BAT%"
  )
)

echo echo [%TITLE%] Environment ready, starting uvicorn... >> "%TEMP_BAT%"
echo uvicorn --app-dir "%WORK_DIR%" %MODULE_APP% --host 0.0.0.0 --port %PORT% >> "%TEMP_BAT%"
echo pause >> "%TEMP_BAT%"

start "POLO %TITLE% (Port %PORT%)" cmd /k "%TEMP_BAT%"
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

REM 환경변수 로드
if exist "%ENV_FILE%" (
  echo [ENV] Loading environment variables from %ENV_FILE%...
  for /f "usebackq tokens=1,2 delims==" %%a in ("%ENV_FILE%") do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" if not "%%a:~0,1%"=="S" (
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
set "PREPROCESS_DIR=%ROOT%preprocessing\texprep"
set "VIZ_DIR=%ROOT%viz"
set "FRONTEND_DIR=%ROOT%polo-front"

set "REQ_SERVER=requirements.api.txt"
set "REQ_EASY=requirements.easy.txt"
set "REQ_MATH=requirements.math.txt"
set "REQ_PREPROCESS=pyproject.toml"
set "REQ_VIZ=requirements.viz.txt"

set "FALLBACK_SERVER=fastapi uvicorn httpx pydantic asyncpg aiosqlite sqlalchemy bcrypt anyio arxiv"
set "FALLBACK_PREPROCESS=fastapi uvicorn pydantic"
set "FALLBACK_MATH=fastapi uvicorn transformers torch"
set "FALLBACK_VIZ=fastapi uvicorn matplotlib seaborn plotly"

echo [1/6] Start Easy Model (Local, port %EASY_PORT%)
if exist "%EASY_DIR%" (
  echo [DEBUG] Easy directory found, calling LAUNCH_PY_APP...
  call :LAUNCH_PY_APP Easy "%EASY_DIR%" %REQ_EASY% app:app %EASY_PORT% yes "fastapi uvicorn transformers torch peft"
  echo [DEBUG] LAUNCH_PY_APP for Easy completed
) else (
  echo [ERROR] Easy directory not found: %EASY_DIR%
  exit /b 1
)

echo [2/6] Start Math Model (%MATH_PORT%)
if exist "%MATH_DIR%" (
  call :LAUNCH_PY_APP Math "%MATH_DIR%" %REQ_MATH% app:app %MATH_PORT% yes "%FALLBACK_MATH%"
) else (
  echo [ERROR] Math directory not found: %MATH_DIR%
  exit /b 1
)

echo [3/6] Starting Backend Server (Local, port %SERVER_PORT%)
if exist "%SERVER_DIR%" (
  call :LAUNCH_PY_APP Server "%SERVER_DIR%" %REQ_SERVER% app:app %SERVER_PORT% no "%FALLBACK_SERVER%"
) else (
  echo [ERROR] Server directory not found: %SERVER_DIR%
  exit /b 1
)

echo [4/6] Install Viz & Preprocess dependencies (서버 내부 실행)
echo [Viz] Installing global dependencies...
python -m pip install matplotlib Pillow numpy fastapi uvicorn
echo [Preprocess] Installing global dependencies...
python -m pip install PyYAML fastapi uvicorn
echo [INFO] Viz와 Preprocess는 서버에서 내부적으로 실행됩니다.

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