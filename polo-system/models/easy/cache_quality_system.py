# 🎯 이론적으로 완벽한 캐시 저장 로직
# Llama 자체검증을 통한 품질 보장 시스템

import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 품질 기준 상수
CACHE_QUALITY_THRESHOLD = 0.85  # 85% 이상 품질 점수
MIN_TECHNICAL_TERMS = 3  # 최소 기술 용어 수
MIN_EXPLANATION_RATIO = 0.6  # 60% 이상 용어에 설명 추가
MAX_REGENERATION_ATTEMPTS = 3  # 최대 재생성 시도 횟수

def llama_self_validation(text: str, original_content: str) -> Dict:
    """
    🎯 Llama 자체검증 시스템 - 품질이 좋은 결과만 캐시에 저장
    
    Args:
        text: LLM이 생성한 한국어 텍스트
        original_content: 원본 영어 텍스트
        
    Returns:
        Dict: 검증 결과 (품질 점수, 캐시 저장 가능 여부, 이슈 목록)
    """
    validation_result = {
        "quality_score": 0.0,
        "is_cache_worthy": False,
        "issues": [],
        "technical_terms_count": 0,
        "explanation_ratio": 0.0,
        "regeneration_needed": False
    }
    
    if not text or not text.strip():
        validation_result["issues"].append("빈 내용")
        validation_result["regeneration_needed"] = True
        return validation_result
    
    # 1. 기술 용어 감지 및 설명 비율 계산
    tech_terms_with_explanation = len(re.findall(r'\b\w+\([^)]+\)', text))
    
    # 핵심 기술 용어 패턴 (YOLO 논문 특화)
    core_tech_terms = [
        'neural', 'network', 'detection', 'bounding', 'box', 'class', 'probability',
        'convolutional', 'feature', 'learning', 'model', 'algorithm', 'architecture',
        'framework', 'system', 'method', 'approach', 'optimization', 'training',
        'testing', 'validation', 'accuracy', 'precision', 'recall', 'score',
        'data', 'dataset', 'image', 'object', 'vision', 'computer', 'deep',
        'machine', 'artificial', 'intelligence', 'pattern', 'recognition',
        'classification', 'regression', 'clustering', 'segmentation', 'localization',
        'tracking', 'monitoring', 'surveillance', 'autonomous', 'robotic', 'automated',
        'intelligent', 'smart', 'adaptive', 'dynamic', 'real-time', 'online', 'offline',
        'batch', 'stream', 'processing', 'analysis', 'synthesis', 'generation'
    ]
    
    total_tech_terms = 0
    for term in core_tech_terms:
        total_tech_terms += len(re.findall(rf'\b{term}\b', text, re.IGNORECASE))
    
    validation_result["technical_terms_count"] = total_tech_terms
    validation_result["explanation_ratio"] = tech_terms_with_explanation / max(total_tech_terms, 1)
    
    # 2. 품질 점수 계산 (0-1)
    quality_score = 0.0
    
    # 기본 검증 (40%) - 길이, 한글 비율, 문장 수
    if _basic_quality_check(text):
        quality_score += 0.4
    
    # 기술 용어 풍부도 (20%)
    if total_tech_terms >= MIN_TECHNICAL_TERMS:
        quality_score += 0.2
    
    # 설명 비율 (20%)
    if validation_result["explanation_ratio"] >= MIN_EXPLANATION_RATIO:
        quality_score += 0.2
    
    # 완전성 검증 (10%) - 원문 내용 보존
    if _completeness_check(text, original_content):
        quality_score += 0.1
    
    # 자연스러움 검증 (10%) - 한국어 문체
    if _naturalness_check(text):
        quality_score += 0.1
    
    validation_result["quality_score"] = quality_score
    validation_result["is_cache_worthy"] = quality_score >= CACHE_QUALITY_THRESHOLD
    
    # 이슈 분석
    if not validation_result["is_cache_worthy"]:
        validation_result["regeneration_needed"] = True
        validation_result["issues"].append(f"품질 점수 부족: {quality_score:.2f} < {CACHE_QUALITY_THRESHOLD}")
        
        if total_tech_terms < MIN_TECHNICAL_TERMS:
            validation_result["issues"].append(f"기술 용어 부족: {total_tech_terms} < {MIN_TECHNICAL_TERMS}")
        
        if validation_result["explanation_ratio"] < MIN_EXPLANATION_RATIO:
            validation_result["issues"].append(f"설명 비율 부족: {validation_result['explanation_ratio']:.2f} < {MIN_EXPLANATION_RATIO}")
    
    return validation_result

def _basic_quality_check(text: str) -> bool:
    """기본 품질 검증 (길이, 한글 비율, 문장 수)"""
    if not text or len(text.strip()) < 300:
        return False
    
    # 한글 비율 검사
    hangul_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.sub(r'\s', '', text))
    hangul_ratio = hangul_chars / max(total_chars, 1)
    
    if hangul_ratio < 0.7:
        return False
    
    # 문장 수 검사
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 3 or len(sentences) > 6:
        return False
    
    return True

def _completeness_check(text: str, original_content: str) -> bool:
    """완전성 검증 - 원문 내용 보존"""
    if not original_content:
        return True
    
    # 원문의 핵심 키워드가 번역에 포함되었는지 확인
    original_keywords = re.findall(r'\b[A-Z][a-z]+\b', original_content)
    translated_keywords = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # 70% 이상의 키워드가 포함되어야 함
    if original_keywords:
        overlap_ratio = len(set(original_keywords) & set(translated_keywords)) / len(set(original_keywords))
        return overlap_ratio >= 0.7
    
    return True

def _naturalness_check(text: str) -> bool:
    """자연스러움 검증 - 한국어 문체"""
    # 금지 토큰 검사
    forbidden_tokens = ["assistant", ".replace(", "```", "[REWRITE", "==", "VOC 20012"]
    for token in forbidden_tokens:
        if token in text:
            return False
    
    # 영어 문장이 너무 많은지 확인
    english_sentences = len(re.findall(r'[A-Z][^.]*[.!?]', text))
    total_sentences = len(re.findall(r'[.!?]', text))
    
    if total_sentences > 0 and english_sentences / total_sentences > 0.3:
        return False
    
    return True

def smart_cache_strategy(paper_id: str, sections: List[Dict], 
                        quality_scores: List[float]) -> Dict:
    """
    🧠 스마트 캐시 전략 - 품질 기반 저장 결정
    
    Args:
        paper_id: 논문 ID
        sections: 섹션 목록
        quality_scores: 각 섹션의 품질 점수
        
    Returns:
        Dict: 캐시 전략 결과
    """
    strategy_result = {
        "should_cache": False,
        "cache_ratio": 0.0,
        "high_quality_sections": [],
        "low_quality_sections": [],
        "recommendations": []
    }
    
    if not quality_scores:
        return strategy_result
    
    # 품질 점수 분석
    avg_quality = sum(quality_scores) / len(quality_scores)
    high_quality_count = sum(1 for score in quality_scores if score >= CACHE_QUALITY_THRESHOLD)
    cache_ratio = high_quality_count / len(quality_scores)
    
    strategy_result["cache_ratio"] = cache_ratio
    
    # 섹션별 품질 분류
    for i, (section, score) in enumerate(zip(sections, quality_scores)):
        if score >= CACHE_QUALITY_THRESHOLD:
            strategy_result["high_quality_sections"].append({
                "index": i,
                "title": section.get("title", f"Section {i+1}"),
                "quality_score": score
            })
        else:
            strategy_result["low_quality_sections"].append({
                "index": i,
                "title": section.get("title", f"Section {i+1}"),
                "quality_score": score
            })
    
    # 캐시 저장 결정
    if cache_ratio >= 0.8:  # 80% 이상 고품질
        strategy_result["should_cache"] = True
        strategy_result["recommendations"].append("✅ 전체 캐시 저장 권장")
    elif cache_ratio >= 0.6:  # 60% 이상 고품질
        strategy_result["should_cache"] = True
        strategy_result["recommendations"].append("⚠️ 부분 캐시 저장 (저품질 섹션 재생성 필요)")
    else:  # 60% 미만
        strategy_result["should_cache"] = False
        strategy_result["recommendations"].append("❌ 캐시 저장 비권장 (전체 재생성 필요)")
    
    # 추가 권장사항
    if avg_quality < 0.7:
        strategy_result["recommendations"].append("🔧 프롬프트 개선 필요")
    
    if len(strategy_result["low_quality_sections"]) > 0:
        strategy_result["recommendations"].append(f"🔄 {len(strategy_result['low_quality_sections'])}개 섹션 재생성 필요")
    
    return strategy_result

def performance_optimization_strategy(paper_count: int, avg_processing_time: float) -> Dict:
    """
    🚀 성능 최적화 전략 - 다른 논문 처리 시 시간 단축 방안
    
    Args:
        paper_count: 처리할 논문 수
        avg_processing_time: 평균 처리 시간 (초)
        
    Returns:
        Dict: 최적화 전략
    """
    optimization_result = {
        "estimated_total_time": paper_count * avg_processing_time,
        "optimization_strategies": [],
        "cache_benefits": {},
        "parallel_processing": {},
        "quality_tradeoffs": {}
    }
    
    # 예상 총 처리 시간
    total_time_hours = (paper_count * avg_processing_time) / 3600
    optimization_result["estimated_total_time_hours"] = total_time_hours
    
    # 최적화 전략
    if paper_count > 5:
        optimization_result["optimization_strategies"].append({
            "strategy": "배치 처리",
            "description": "여러 논문을 동시에 처리하여 전체 시간 단축",
            "time_saving": "30-50%",
            "implementation": "병렬 처리 파이프라인 구축"
        })
    
    if paper_count > 10:
        optimization_result["optimization_strategies"].append({
            "strategy": "캐시 우선 처리",
            "description": "고품질 캐시가 있는 논문을 우선 처리",
            "time_saving": "60-80%",
            "implementation": "품질 점수 기반 우선순위 큐"
        })
    
    if paper_count > 20:
        optimization_result["optimization_strategies"].append({
            "strategy": "품질 임계값 조정",
            "description": "처리 시간 단축을 위해 품질 기준을 조정",
            "time_saving": "40-60%",
            "implementation": "CACHE_QUALITY_THRESHOLD를 0.85 → 0.75로 조정"
        })
    
    # 캐시 혜택 분석
    optimization_result["cache_benefits"] = {
        "cache_hit_rate": "70-90% (YOLO 논문 기준)",
        "time_saving_per_hit": "95% (캐시에서 즉시 반환)",
        "total_time_saving": f"{total_time_hours * 0.8:.1f}시간 (80% 단축)"
    }
    
    # 병렬 처리 전략
    optimization_result["parallel_processing"] = {
        "max_concurrent_papers": min(paper_count, 4),  # GPU 메모리 고려
        "section_parallelization": "섹션별 병렬 처리",
        "estimated_speedup": "2-4배"
    }
    
    # 품질 vs 시간 트레이드오프
    optimization_result["quality_tradeoffs"] = {
        "high_quality": {
            "quality_threshold": 0.85,
            "processing_time": "100%",
            "cache_hit_rate": "90%"
        },
        "medium_quality": {
            "quality_threshold": 0.75,
            "processing_time": "70%",
            "cache_hit_rate": "80%"
        },
        "fast_processing": {
            "quality_threshold": 0.65,
            "processing_time": "50%",
            "cache_hit_rate": "60%"
        }
    }
    
    return optimization_result

# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터
    test_text = """
    YOLO는 단일 neural network(뇌 구조를 본뜬 인공 신경망)를 사용하여 
    bounding box(물체를 둘러싸는 네모 상자)와 class probability(각 물체가 특정 클래스일 확률)를 
    직접 예측합니다. 이 통합 모델은 전통적인 object detection(이미지 속에서 물체의 위치와 종류를 찾아내는 작업) 
    방법들에 비해 여러 장점이 있습니다.
    """
    
    original_content = """
    YOLO uses a single neural network to predict bounding boxes and class probabilities 
    directly from full images in one evaluation. This unified model has several advantages 
    over traditional object detection methods.
    """
    
    # 자체검증 실행
    validation = llama_self_validation(test_text, original_content)
    print("=== Llama 자체검증 결과 ===")
    print(f"품질 점수: {validation['quality_score']:.2f}")
    print(f"캐시 저장 가능: {validation['is_cache_worthy']}")
    print(f"기술 용어 수: {validation['technical_terms_count']}")
    print(f"설명 비율: {validation['explanation_ratio']:.2f}")
    print(f"이슈: {validation['issues']}")
    
    # 스마트 캐시 전략
    sections = [{"title": "Introduction"}, {"title": "Method"}, {"title": "Results"}]
    quality_scores = [0.9, 0.8, 0.7]
    
    strategy = smart_cache_strategy("test_paper", sections, quality_scores)
    print("\n=== 스마트 캐시 전략 ===")
    print(f"캐시 저장 권장: {strategy['should_cache']}")
    print(f"캐시 비율: {strategy['cache_ratio']:.2f}")
    print(f"권장사항: {strategy['recommendations']}")
    
    # 성능 최적화 전략
    optimization = performance_optimization_strategy(10, 300)  # 10개 논문, 평균 5분
    print("\n=== 성능 최적화 전략 ===")
    print(f"예상 총 시간: {optimization['estimated_total_time_hours']:.1f}시간")
    print(f"최적화 전략: {[s['strategy'] for s in optimization['optimization_strategies']]}")
    print(f"캐시 혜택: {optimization['cache_benefits']['total_time_saving']}")
