@echo off
echo ========================================
echo POLO 데이터셋 다운로드 및 변환 스크립트
echo ========================================
echo.

echo 사용 가능한 데이터셋:
echo 1. WikiLarge (영어 문장 단순화, ~15만개)
echo 2. ASSET (문장 단순화, 더 큰 규모)
echo.

set /p choice="데이터셋을 선택하세요 (1 또는 2): "

if "%choice%"=="1" (
    echo WikiLarge 데이터셋 다운로드 중...
    "C:\POLO\POLO\polo-system\models\easy\venv\Scripts\python.exe" "C:\POLO\POLO\polo-system\models\easy\training\download_and_convert.py" --dataset wikilarge --output "C:\POLO\POLO\polo-system\models\easy\training\train.jsonl"
) else if "%choice%"=="2" (
    echo ASSET 데이터셋 다운로드 중...
    "C:\POLO\POLO\polo-system\models\easy\venv\Scripts\python.exe" "C:\POLO\POLO\polo-system\models\easy\training\download_and_convert.py" --dataset asset --output "C:\POLO\POLO\polo-system\models\easy\training\train.jsonl"
) else (
    echo 잘못된 선택입니다. 1 또는 2를 입력하세요.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 다운로드 완료!
echo ========================================
echo.
echo 다음 단계:
echo 1. Docker 컨테이너 빌드: docker compose build easy-train
echo 2. 학습 시작: docker compose run --rm --gpus all easy-train
echo.
pause
