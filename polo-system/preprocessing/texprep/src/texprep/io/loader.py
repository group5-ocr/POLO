# src/texprep/io/loader.py
from __future__ import annotations
from pathlib import Path

MAX_READ_MB = 20  
NEWLINE = "\n"

def read_text(path: str) -> str:
    """
    TeX 소스 안전 로드:
    - 파일 크기 제한
    - UTF-8 우선, 실패 시 latin-1 폴백
    - BOM 제거
    - 개행 통일(\n)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {p}")

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_READ_MB:
        raise ValueError(f"파일 너무 큼({size_mb:.1f} MB): {p}")

    # 인코딩 시도: utf-8 → latin-1 폴백
    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = p.read_text(encoding="latin-1", errors="ignore")

    # BOM 제거
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")

    # 개행 통일
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    return raw
