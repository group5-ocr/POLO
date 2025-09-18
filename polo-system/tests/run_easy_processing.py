#!/usr/bin/env python3
"""
Easy 모델 처리 스크립트
사용법: python run_easy_processing.py
"""

import sys
import os
from pathlib import Path

# 경로 설정
POLO_ROOT = Path(__file__).parent
EASY_DIR = POLO_ROOT / "models" / "easy"
TRANSPORT_DIR = POLO_ROOT / "server" / "data" / "out" / "transformer" / "source"
OUTPUT_ROOT = POLO_ROOT / "server" / "data" / "outputs"

def main():
    print("🚀 Easy 모델 처리 시작...")
    print(f"소스 디렉토리(from-transport): {TRANSPORT_DIR}")
    print(f"출력 루트: {OUTPUT_ROOT}")
    
    # 출력 루트 생성
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Easy 모델 디렉토리로 이동
    os.chdir(EASY_DIR)
    sys.path.insert(0, str(EASY_DIR))
    
    try:
        from app import from_transport, TransportRequest, load_model
        import asyncio
        
        async def process_paper():
            print("📥 모델 로딩 중...")
            load_model()
            print("✅ 모델 로딩 완료!")
            
            print("🔄 논문 처리 시작(from-transport)...")

            request = TransportRequest(
                paper_id='transformer_v2',
                transport_path=str(TRANSPORT_DIR),
                output_dir=str(OUTPUT_ROOT),
                style='three_para_ko'
            )

            result = await from_transport(request)
            
            print("✅ 처리 완료!")
            print(f"성공: {result.success}개, 실패: {result.failed}개")
            print(f"출력 디렉토리: {result.out_dir}")
            
            # 결과 파일 확인
            out_dir = Path(result.out_dir)
            json_file = out_dir / 'easy_results.json'
            html_file = out_dir / 'easy_results.html'
            
            if json_file.exists():
                print(f"📄 JSON 결과: {json_file}")
            if html_file.exists():
                print(f"🌐 HTML 결과: {html_file}")
            
            # 결과 미리보기
            if json_file.exists():
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print("\n📋 결과 미리보기:")
                    for i, section in enumerate(data.get('sections', [])[:2]):
                        print(f"\n섹션 {i+1}: {section.get('title', 'Unknown')}")
                        korean_text = section.get('korean_translation', '')
                        if korean_text:
                            preview = korean_text[:200] + "..." if len(korean_text) > 200 else korean_text
                            print(f"한국어: {preview}")
                        else:
                            print("한국어 번역 없음")
        
        # 비동기 실행
        asyncio.run(process_paper())
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
