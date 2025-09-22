#!/usr/bin/env python3
"""
캐시 동작 테스트 스크립트
"""
import sys
sys.path.append('models/easy')
from app import get_cache_key, load_from_cache, check_cache_completeness

# YOLO 논문 테스트
paper_id = "doc_8898966966363258996_2767669"  # 현재 처리 중인 논문 ID
sections = [
    {'title': 'Abstract', 'content': 'We present YOLO, a new approach to object detection...'},
    {'title': 'Introduction', 'content': 'Humans glance at an image and instantly know...'},
    {'title': 'Unified Detection', 'content': 'Our system divides the input image...'}
]  # YOLO 내용이 포함된 테스트용 섹션

# 캐시 키 생성
cache_key = get_cache_key(paper_id, sections)
print(f"📝 논문 ID: {paper_id}")
print(f"🔑 캐시 키: {cache_key}")

# 캐시 로드
cached_result = load_from_cache(cache_key)
if cached_result:
    print("✅ 캐시에서 결과 로드 성공!")
    print(f"📊 총 섹션 수: {len(cached_result.get('easy_sections', []))}")
    
    # 완성도 체크
    is_complete, missing_indices = check_cache_completeness(cached_result, sections)
    print(f"🔍 완성도: {is_complete}")
    print(f"📝 누락 섹션 수: {len(missing_indices)}")
    
    if missing_indices:
        print("🔄 이어서 처리할 섹션들:")
        for idx in missing_indices:
            section = cached_result['easy_sections'][idx] if idx < len(cached_result['easy_sections']) else None
            title = section.get('easy_section_title', f'Section {idx+1}') if section else f'Section {idx+1}'
            print(f"  - {idx+1}: {title}")
else:
    print("❌ 캐시에서 결과를 찾을 수 없습니다.")
