#Requires -Version 5.1
param(
  [switch]$ServerOnly
)

$ErrorActionPreference = "Stop"

function Get-FreePort([int]$StartPort){
  $p = $StartPort
  while($true){
    $inUse = (Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue) -ne $null
    if(-not $inUse){ return $p }
    $p++
  }
}

function Test-DockerCompose {
  try {
    docker compose version | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Patch-EnvFile($EnvPath, $ServerPort, $PrePort, $EasyPort, $MathPort, $VizPort){
  if(!(Test-Path $EnvPath)){ 
    Write-Host "[ENV] Creating $EnvPath" -ForegroundColor Yellow
    New-Item -ItemType File -Path $EnvPath -Force | Out-Null 
  }
  
  $lines = Get-Content $EnvPath -Raw -Encoding utf8 -ErrorAction SilentlyContinue
  if([string]::IsNullOrWhiteSpace($lines)){ $lines = "" }

  $keep = @()
  foreach($line in $lines -split "`r?`n"){
    if($line -match '^(SERVER_PORT|PREPROCESS_URL|EASY_MODEL_URL|MATH_MODEL_URL|VIZ_MODEL_URL|CALLBACK_URL)='){ continue }
    if($line.Trim().Length -gt 0){ $keep += $line }
  }
  
  $new = @()
  $new += "SERVER_PORT=$ServerPort"
  $new += "PREPROCESS_URL=http://localhost:$PrePort"
  $new += "EASY_MODEL_URL=http://localhost:$EasyPort"
  $new += "MATH_MODEL_URL=http://localhost:$MathPort"
  $new += "VIZ_MODEL_URL=http://localhost:$VizPort"
  $new += "CALLBACK_URL=http://localhost:$ServerPort"

  ($keep + $new) -join "`r`n" | Set-Content -Path $EnvPath -Encoding UTF8
  Write-Host "[ENV] Patched $EnvPath with dynamic ports" -ForegroundColor Cyan
}

function Resolve-Venv($WorkDir){
  $v1 = Join-Path $WorkDir "venv\Scripts\activate.bat"
  $v2 = Join-Path $WorkDir ".venv\Scripts\activate.bat"
  if(Test-Path $v1){ return "venv" }
  if(Test-Path $v2){ return ".venv" }
  # 없으면 만들어줌
  Write-Host "[ENV] Creating venv in $WorkDir" -ForegroundColor Yellow
  & py -3 -m venv (Join-Path $WorkDir "venv") 2>$null
  if($LASTEXITCODE -ne 0){
    & python -m venv (Join-Path $WorkDir "venv") 2>$null
  }
  return "venv"
}

function Launch-PyApp(
  [string]$Title, [string]$WorkDir, [string]$ReqFile,
  [string]$ModuleApp, [int]$Port, [bool]$NeedHF, [string]$FallbackPkgs = ""
){
  $venvDir = Resolve-Venv $WorkDir
  $act = Join-Path $WorkDir "$venvDir\Scripts\activate.bat"
  $cmd = @()
  $cmd += "call `"$act`""
  
  # 의존성 설치
  if($ReqFile){
    $reqPath = Join-Path $WorkDir $ReqFile
    if(Test-Path $reqPath){ 
      $cmd += "pip install -q -r `"$ReqFile`"" 
    } elseif($FallbackPkgs){ 
      $cmd += "pip install -q $FallbackPkgs" 
    }
  } elseif($FallbackPkgs){
    $cmd += "pip install -q $FallbackPkgs"
  }
  
  # 환경 변수 설정
  if($NeedHF){ $cmd += "set HUGGINGFACE_TOKEN=$env:HUGGINGFACE_TOKEN" }
  $cmd += "set ORCH_BASE=http://localhost:$SERVER_PORT"
  
  # 서비스 실행
  $cmd += "uvicorn $ModuleApp --host 0.0.0.0 --port $Port"

  $args = "/k " + ($cmd -join " && ")
  Start-Process -FilePath "cmd.exe" -WorkingDirectory $WorkDir -ArgumentList $args -WindowStyle Normal -Wait:$false -PassThru | Out-Null
  Write-Host " -> launched $Title on port $Port" -ForegroundColor Green
}

# ---------------- MAIN ----------------

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$EnvFile = Join-Path $Root "server\.env"

# 기본 포트 (빈 포트 자동 할당)
$SERVER_PORT    = Get-FreePort 8000
$PREPROCESS_PORT= Get-FreePort 5002
$EASY_PORT      = Get-FreePort 5003
$MATH_PORT      = Get-FreePort 5004
$VIZ_PORT       = Get-FreePort 5005
$FRONT_PORT     = 5173

Write-Host "========================================"
Write-Host "POLO - Assigned Ports"
Write-Host ("  Backend   : {0}" -f $SERVER_PORT)
Write-Host ("  Preprocess: {0}" -f $PREPROCESS_PORT)
Write-Host ("  Easy      : {0}" -f $EASY_PORT)
Write-Host ("  Math      : {0}" -f $MATH_PORT)
Write-Host ("  Viz       : {0}" -f $VIZ_PORT)
Write-Host ("  Frontend  : {0}" -f $FRONT_PORT)
Write-Host "========================================`n"

# HF 토큰 없으면 입력
if([string]::IsNullOrWhiteSpace($env:HUGGINGFACE_TOKEN)){
  $env:HUGGINGFACE_TOKEN = Read-Host "Enter HUGGINGFACE_TOKEN"
}

# server/.env 패치
Patch-EnvFile -EnvPath $EnvFile -ServerPort $SERVER_PORT -PrePort $PREPROCESS_PORT -EasyPort $EASY_PORT -MathPort $MATH_PORT -VizPort $VIZ_PORT

# 서버 로컬 기동
$env:SERVER_PORT = "$SERVER_PORT"
Write-Host "[1/6] Starting Backend Server (Local, port $SERVER_PORT)" -ForegroundColor Cyan

$SERVER_DIR = Join-Path $Root "server"
if(Test-Path $SERVER_DIR){
  Launch-PyApp -Title "POLO Server" -WorkDir $SERVER_DIR -ReqFile "requirements.api.txt" `
    -ModuleApp "app:app" -Port $SERVER_PORT -NeedHF:$false -FallbackPkgs "fastapi uvicorn httpx pydantic asyncpg aiosqlite sqlalchemy bcrypt anyio"
} else {
  Write-Host "[ERROR] Server directory not found: $SERVER_DIR" -ForegroundColor Red
  exit 1
}

# 헬스체크
Write-Host "[HEALTH] Waiting for backend to be ready..." -ForegroundColor Yellow
try{
  for($i=0;$i -lt 40;$i++){
    Start-Sleep -Seconds 2
    try{
      $r = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$SERVER_PORT/openapi.json" -TimeoutSec 3
      if($r.StatusCode -eq 200){ 
        Write-Host "[UP] Backend ready" -ForegroundColor Green
        break 
      }
    } catch {}
  }
}catch{
  Write-Host "[WARN] Backend health check failed, continuing..." -ForegroundColor Yellow
}

# ServerOnly 모드가 아니면 로컬 서비스들도 실행
if(-not $ServerOnly){
  # 로컬 서비스 경로
  $PRE_DIR = Join-Path $Root "preprocessing\texprep"
  $EASY_DIR= Join-Path $Root "models\easy"
  $MATH_DIR= Join-Path $Root "models\math"
  $VIZ_DIR = Join-Path $Root "viz"
  $FRONT_DIR=Join-Path $Root "polo-front"

  # 실행 (requirements 파일 이름 정책 반영)
  if(Test-Path $PRE_DIR){
    Write-Host "[2/6] Starting Preprocess (port $PREPROCESS_PORT)" -ForegroundColor Cyan
    Launch-PyApp -Title "POLO Preprocess" -WorkDir $PRE_DIR -ReqFile "requirements.preprocess.txt" `
      -ModuleApp "app:app" -Port $PREPROCESS_PORT -NeedHF:$false -FallbackPkgs "fastapi uvicorn httpx pydantic"
  } else {
    Write-Host "[SKIP] Preprocess directory not found: $PRE_DIR" -ForegroundColor Yellow
  }
  
  if(Test-Path $EASY_DIR){
    Write-Host "[3/6] Starting Easy Model (port $EASY_PORT)" -ForegroundColor Cyan
    Launch-PyApp -Title "POLO Easy" -WorkDir $EASY_DIR -ReqFile "requirements.easy.txt" `
      -ModuleApp "app:app" -Port $EASY_PORT -NeedHF:$true
  } else {
    Write-Host "[SKIP] Easy directory not found: $EASY_DIR" -ForegroundColor Yellow
  }
  
  if(Test-Path $MATH_DIR){
    Write-Host "[4/6] Starting Math Model (port $MATH_PORT)" -ForegroundColor Cyan
    Launch-PyApp -Title "POLO Math" -WorkDir $MATH_DIR -ReqFile "requirements.math.txt" `
      -ModuleApp "app:app" -Port $MATH_PORT -NeedHF:$true
  } else {
    Write-Host "[SKIP] Math directory not found: $MATH_DIR" -ForegroundColor Yellow
  }
  
  if(Test-Path $VIZ_DIR){
    Write-Host "[5/6] Starting Viz Model (port $VIZ_PORT)" -ForegroundColor Cyan
    Launch-PyApp -Title "POLO Viz" -WorkDir $VIZ_DIR -ReqFile "requirements.viz.txt" `
      -ModuleApp "app:app" -Port $VIZ_PORT -NeedHF:$false -FallbackPkgs "fastapi uvicorn httpx pydantic pillow"
  } else {
    Write-Host "[SKIP] Viz directory not found: $VIZ_DIR" -ForegroundColor Yellow
  }

  # 프론트엔드
  if(Test-Path $FRONT_DIR){
    Write-Host "[6/6] Starting Frontend (port $FRONT_PORT)" -ForegroundColor Cyan
    Start-Process cmd -WorkingDirectory $FRONT_DIR -ArgumentList '/k npm install && npm run dev -- --port ' + $FRONT_PORT
  } else {
    Write-Host "[SKIP] Frontend directory not found: $FRONT_DIR" -ForegroundColor Yellow
  }
} else {
  Write-Host "[INFO] ServerOnly mode - skipping local services" -ForegroundColor Cyan
}

Write-Host "`n========================================"
Write-Host "POLO System Launch Complete"
Write-Host "========================================"
Write-Host ("Backend   : http://localhost:{0}" -f $SERVER_PORT)
if(-not $ServerOnly){
  Write-Host ("Preprocess: http://localhost:{0}" -f $PREPROCESS_PORT)
  Write-Host ("Easy      : http://localhost:{0}" -f $EASY_PORT)
  Write-Host ("Math      : http://localhost:{0}" -f $MATH_PORT)
  Write-Host ("Viz       : http://localhost:{0}" -f $VIZ_PORT)
  Write-Host ("Frontend  : http://localhost:{0}" -f $FRONT_PORT)
}
Write-Host "========================================"
Write-Host "`nUsage:"
Write-Host "  .\run_system.ps1           # Full system"
Write-Host "  .\run_system.ps1 -ServerOnly  # Backend only"
Write-Host "========================================"
