#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv1 논문 통합 분석 실행 스크립트

이 스크립트는 목업 데이터를 사용하여 통합 JSONL을 생성합니다.
"""

import sys
import os
from pathlib import Path

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from integrated_generator import main

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv1 논문 통합 분석 시스템")
    print("=" * 60)
    
    # 통합 JSONL 생성
    main()
    
    print("\n" + "=" * 60)
    print("실행 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. 프론트엔드 실행: python -m http.server 8000")
    print("2. 브라우저에서 확인: http://localhost:8000")
    print("\n생성된 파일:")
    print("- data/integrated_result.jsonl")
    print("- frontend/index.html")
