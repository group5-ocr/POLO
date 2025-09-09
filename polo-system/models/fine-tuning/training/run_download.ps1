# POLO 데이터셋 다운로드 및 변환 스크립트 (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "POLO 데이터셋 다운로드 및 변환 스크립트" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "사용 가능한 데이터셋:" -ForegroundColor Yellow
Write-Host "1. WikiLarge (영어 문장 단순화, ~15만개)" -ForegroundColor Green
Write-Host "2. ASSET (문장 단순화, 더 큰 규모)" -ForegroundColor Green
Write-Host "3. 테스트용 WikiLarge (1만개 샘플)" -ForegroundColor Magenta
Write-Host ""

$choice = Read-Host "데이터셋을 선택하세요 (1, 2, 또는 3)"

$pythonPath = "C:\POLO\POLO\polo-system\models\easy\venv\Scripts\python.exe"
$scriptPath = "C:\POLO\POLO\polo-system\models\easy\training\download_and_convert.py"
$outputPath = "C:\POLO\POLO\polo-system\models\easy\training\train.jsonl"

switch ($choice) {
    "1" {
        Write-Host "WikiLarge 데이터셋 다운로드 중..." -ForegroundColor Green
        & $pythonPath $scriptPath --dataset wikilarge --output $outputPath
    }
    "2" {
        Write-Host "ASSET 데이터셋 다운로드 중..." -ForegroundColor Green
        & $pythonPath $scriptPath --dataset asset --output $outputPath
    }
    "3" {
        Write-Host "테스트용 WikiLarge 데이터셋 다운로드 중 (1만개)..." -ForegroundColor Magenta
        & $pythonPath $scriptPath --dataset wikilarge --sample_size 10000 --output $outputPath
    }
    default {
        Write-Host "잘못된 선택입니다. 1, 2, 또는 3을 입력하세요." -ForegroundColor Red
        Read-Host "계속하려면 Enter를 누르세요"
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "다운로드 완료!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "다음 단계:" -ForegroundColor Yellow
Write-Host "1. Docker 컨테이너 빌드: docker compose build easy-train" -ForegroundColor White
Write-Host "2. 학습 시작: docker compose run --rm --gpus all easy-train" -ForegroundColor White
Write-Host ""
Read-Host "계속하려면 Enter를 누르세요"
