#!/usr/bin/env python3
"""
캐시된 결과 이어서 처리 테스트 스크립트
"""
import asyncio
import json
from pathlib import Path

# Easy 모델의 resume_from_cache 함수를 테스트
async def test_resume():
    # 캐시된 결과 로드
    cache_file = Path("server/data/db/yolo/doc_-6816425174722030295_2767669_caa97616c4a9f51b.json")
    
    if not cache_file.exists():
        print("❌ 캐시 파일이 없습니다:", cache_file)
        return
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        cached_result = json.load(f)
    
    print("✅ 캐시된 결과 로드 완료")
    print(f"📊 총 섹션 수: {len(cached_result.get('easy_sections', []))}")
    
    # 빈 섹션 찾기
    empty_sections = []
    for i, section in enumerate(cached_result.get('easy_sections', [])):
        if not section.get('easy_content', '').strip():
            empty_sections.append(i)
            print(f"🔍 빈 섹션 발견: {i+1} - {section.get('easy_section_title', 'Unknown')}")
    
    print(f"📝 빈 섹션 수: {len(empty_sections)}")
    
    if empty_sections:
        print("🔄 이어서 처리 가능한 섹션들:")
        for idx in empty_sections:
            section = cached_result['easy_sections'][idx]
            print(f"  - {idx+1}: {section.get('easy_section_title', 'Unknown')}")
    else:
        print("✅ 모든 섹션이 완성되어 있습니다!")

if __name__ == "__main__":
    asyncio.run(test_resume())
