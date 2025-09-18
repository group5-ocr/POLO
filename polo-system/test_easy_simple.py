#!/usr/bin/env python3
"""
간단한 Easy 모델 테스트 스크립트
사용법: python test_easy_simple.py
"""

import sys
import os
from pathlib import Path

# 경로 설정
POLO_ROOT = Path(__file__).parent
EASY_DIR = POLO_ROOT / "models" / "easy"

def main():
    print("🧪 간단한 Easy 모델 테스트...")
    
    # Easy 모델 디렉토리로 이동
    os.chdir(EASY_DIR)
    sys.path.insert(0, str(EASY_DIR))
    
    try:
        from app import load_model, _translate_to_korean, _normalize_bracket_tokens
        
        print("📥 모델 로딩 중...")
        load_model()
        print("✅ 모델 로딩 완료!")
        
        # 테스트 텍스트
        test_text = "This is a test paper about machine learning. LRB lrb RRB rrb are bracket tokens."
        
        print(f"\n📝 원본 텍스트: {test_text}")
        
        # 괄호 정규화 테스트
        normalized = _normalize_bracket_tokens(test_text)
        print(f"🔧 괄호 정규화: {normalized}")
        
        # 번역 테스트
        print("\n🌐 한국어 번역 중...")
        korean_text = _translate_to_korean(normalized)
        print(f"✅ 한국어 번역: {korean_text}")
        
        # 한국어 비율 확인
        hangul_count = len([c for c in korean_text if '가' <= c <= '힣'])
        total_chars = len([c for c in korean_text if c.isalpha()])
        korean_ratio = hangul_count / max(1, total_chars)
        
        print(f"\n📊 한국어 비율: {korean_ratio:.2%} ({hangul_count}/{total_chars})")
        
        if korean_ratio > 0.5:
            print("✅ 한국어 번역 성공!")
        else:
            print("⚠️ 한국어 번역이 부족합니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
