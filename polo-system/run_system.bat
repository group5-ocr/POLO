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

REM Math 모델의 경우 의존성 설치를 다시 수행
if "%TITLE%"=="Math" (
  echo [MATH] Reinstalling dependencies for Math model
  echo [MATH] venv path: %WORK_DIR%\venv
  set "REQ_PATH=%WORK_DIR%\%REQ_FILE%"
  if exist "%REQ_PATH%" (
    call "%ACTIVATE%" ^&^& python -m pip install --upgrade pip --quiet
    call "%ACTIVATE%" ^&^& python -m pip install -r "%REQ_PATH%" --force-reinstall --quiet
  )
  goto :skip_install
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

:skip_install
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
  REM PostgreSQL 환경변수 전달
  echo if defined POSTGRES_HOST set POSTGRES_HOST=%%POSTGRES_HOST%%
  echo if defined POSTGRES_PORT set POSTGRES_PORT=%%POSTGRES_PORT%%
  echo if defined POSTGRES_DB set POSTGRES_DB=%%POSTGRES_DB%%
  echo if defined POSTGRES_USER set POSTGRES_USER=%%POSTGRES_USER%%
  echo if defined POSTGRES_PASSWORD set POSTGRES_PASSWORD=%%POSTGRES_PASSWORD%%
  echo if defined LOCAL_DB_PATH set LOCAL_DB_PATH=%%LOCAL_DB_PATH%%
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

REM 고정 포트 사용
set "SERVER_PORT=8000"
set "PREPROCESS_PORT=5002"
set "EASY_PORT=5003"
set "MATH_PORT=5004"
set "VIZ_PORT=5005"
echo [PORTS] Server:%SERVER_PORT% Preprocess:%PREPROCESS_PORT% Easy:%EASY_PORT% Math:%MATH_PORT% Viz:%VIZ_PORT%

REM patch env
call :PATCH_ENV_FILE "%ENV_FILE%" "%SERVER_PORT%" "%PREPROCESS_PORT%" "%EASY_PORT%" "%MATH_PORT%" "%VIZ_PORT%"

REM also export to current process (so child windows inherit)
if exist "%ENV_FILE%" (
  for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" (
      set "%%a=%%b"
      echo [ENV] Loaded: %%a=%%b
    )
  )
) else (
  echo [WARN] .env file not found: %ENV_FILE%
)

REM dirs
set "SERVER_DIR=%ROOT%server"
set "EASY_DIR=%ROOT%models\easy"
set "MATH_DIR=%ROOT%models\math"
set "PREPROCESS_DIR=%ROOT%preprocessing\texprep"
set "VIZ_DIR=%ROOT%viz"
set "FRONTEND_DIR=%ROOT%polo-front"

set "REQ_SERVER=requirements.api.txt"
set "REQ_EASY=requirements.easy.txt"
set "REQ_MATH=requirements.math.txt"
set "REQ_PREPROCESS=requirements.pre.txt"
set "REQ_VIZ=requirements.viz.txt"

set "FALLBACK_SERVER=fastapi uvicorn httpx pydantic asyncpg aiosqlite sqlalchemy bcrypt anyio arxiv"
set "FALLBACK_MATH=fastapi uvicorn transformers torch"
set "FALLBACK_PREPROCESS=fastapi uvicorn httpx pydantic pyyaml"
set "FALLBACK_VIZ=fastapi uvicorn matplotlib numpy torch"

echo [1/6] Start Preprocess Service (Port %PREPROCESS_PORT%)
if exist "%PREPROCESS_DIR%" (
  call :LAUNCH_PY_APP "Preprocess" "%PREPROCESS_DIR%" "%REQ_PREPROCESS%" "app:app" "%PREPROCESS_PORT%" "no" "%FALLBACK_PREPROCESS%"
) else (
  echo [ERROR] Preprocess directory not found: %PREPROCESS_DIR%
  goto :end
)

echo [2/6] Start Viz Service (Port %VIZ_PORT%)
if exist "%VIZ_DIR%" (
  call :LAUNCH_PY_APP "Viz" "%VIZ_DIR%" "%REQ_VIZ%" "app:app" "%VIZ_PORT%" "no" "%FALLBACK_VIZ%"
) else (
  echo [ERROR] Viz directory not found: %VIZ_DIR%
  goto :end
)

echo [3/6] Start Easy Model (Port %EASY_PORT%)
if exist "%EASY_DIR%" (
  call :LAUNCH_PY_APP "Easy" "%EASY_DIR%" "%REQ_EASY%" "app:app" "%EASY_PORT%" "yes" "fastapi uvicorn transformers torch peft"
) else (
  echo [ERROR] Easy directory not found: %EASY_DIR%
  goto :end
)

echo [4/6] Start Math Model (Port %MATH_PORT%)
if exist "%MATH_DIR%" (
  call :LAUNCH_PY_APP "Math" "%MATH_DIR%" "%REQ_MATH%" "app:app" "%MATH_PORT%" "yes" "%FALLBACK_MATH%"
) else (
  echo [ERROR] Math directory not found: %MATH_DIR%
  goto :end
)

echo [5/7] Start Backend Server (Port %SERVER_PORT%)
if exist "%SERVER_DIR%" (
  REM 서버 의존성에 email-validator 추가
  set "FALLBACK_SERVER_WITH_EMAIL=fastapi uvicorn httpx pydantic[email] asyncpg aiosqlite sqlalchemy bcrypt anyio arxiv"
  call :LAUNCH_PY_APP "Server" "%SERVER_DIR%" "%REQ_SERVER%" "app:app" "%SERVER_PORT%" "no" "%FALLBACK_SERVER_WITH_EMAIL%"
) else (
  echo [ERROR] Server directory not found: %SERVER_DIR%
  goto :end
)

echo [6/7] Start Frontend (Vite)
if exist "%FRONTEND_DIR%" (
  pushd "%FRONTEND_DIR%"
  
  REM 루트의 .env 파일에서 VITE_API_BASE 읽기
  echo [FRONTEND] Using VITE_API_BASE from root .env file
  
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

REM ============================================================
REM [VIZ API] Check external VIZ API connection status
set VIZ_API_CHECK_PY=%~dp0tools\check_viz_api.py
if exist "%VIZ_API_CHECK_PY%" (
  echo [VIZ API] Checking external API connection...
  python "%VIZ_API_CHECK_PY%"
  if errorlevel 1 (
    echo [VIZ API][WARN] External API connection failed. PNG conversion may not work.
    echo [VIZ API][INFO] You can still use the system, but external PNG conversion will be unavailable.
  ) else (
    echo [VIZ API] External API connection successful!
  )
) else (
  echo [VIZ API][WARN] check_viz_api.py not found: %VIZ_API_CHECK_PY%
)

REM [FIGURES] Build figures_map.json (assets.jsonl → PNG → map)
set FIG_BUILD_PY=%~dp0tools\build_figures_map.py
if exist "%FIG_BUILD_PY%" (
  echo [FIG] Generating figures_map.json ...
  python "%FIG_BUILD_PY%"
) else (
  echo [FIG][WARN] build_figures_map.py not found: %FIG_BUILD_PY%
)

REM [FIG] Build figure mapping template (integrated_result.json → figureMapTemplate.ts)
set FIG_TEMPLATE_PY=%~dp0tools\build_figure_map_template.py
if exist "%FIG_TEMPLATE_PY%" (
  echo [FIG] Generating figure mapping template ...
  python "%FIG_TEMPLATE_PY%"
) else (
  echo [FIG][WARN] build_figure_map_template.py not found: %FIG_TEMPLATE_PY%
)

REM [PDF TO PNG] External API conversion tool (optional usage)
set PDF_TO_PNG_PY=%~dp0tools\convert_pdf_to_png.py
if exist "%PDF_TO_PNG_PY%" (
  echo [PDF TO PNG] PDF to PNG conversion tool available at: %PDF_TO_PNG_PY%
  echo [PDF TO PNG] Usage: python "%PDF_TO_PNG_PY%" [arxiv_id] --output [output_dir]
  echo [PDF TO PNG] Example: python "%PDF_TO_PNG_PY%" 1506.02640
) else (
  echo [PDF TO PNG][WARN] convert_pdf_to_png.py not found: %PDF_TO_PNG_PY%
)

REM [FIG] 사이드카 정적 서버 실행 (선택적 - 메인 서버에 /static이 없을 때만)
set FIG_SIDECAR_PY=%~dp0tools\figsidecar_app.py
set FIG_STATIC_ROOT=C:\POLO\POLO\polo-system\server\data\outputs
set FIG_SIDECAR_PORT=8020

if exist "%FIG_SIDECAR_PY%" (
  echo [FIG] Start static sidecar on port %FIG_SIDECAR_PORT% ...
  start "FigStatic" /min cmd /c python "%FIG_SIDECAR_PY%" --root "%FIG_STATIC_ROOT%" --port %FIG_SIDECAR_PORT% >nul 2>&1
) else (
  echo [FIG][WARN] figsidecar_app.py not found: %FIG_SIDECAR_PY%
)
REM ============================================================

echo.
echo ========================================
echo POLO System Started Successfully!
echo ========================================
echo Preprocess: http://localhost:%PREPROCESS_PORT%
echo Viz:        http://localhost:%VIZ_PORT%
echo Easy:       http://localhost:%EASY_PORT%
echo Math:       http://localhost:%MATH_PORT%
echo Backend:    http://localhost:%SERVER_PORT%
echo Frontend:   http://localhost:5173
echo.

:wait_loop
timeout /t 1 /nobreak >nul
goto :wait_loop

:end
echo.
echo [DONE] Some components failed to start. Check messages above.
pause
