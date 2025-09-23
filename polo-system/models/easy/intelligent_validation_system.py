# 🧠 지능적 검증 시스템 - 주관적 판단 기반
# 수치가 아닌 의미적 품질 평가

import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def intelligent_quality_assessment(text: str, original_content: str) -> Dict:
    """
    🧠 지능적 품질 평가 - 주관적 판단 기반
    
    Args:
        text: LLM이 생성한 한국어 텍스트
        original_content: 원본 영어 텍스트
        
    Returns:
        Dict: 지능적 평가 결과
    """
    assessment = {
        "is_well_translated": False,
        "translation_quality": "unknown",
        "key_issues": [],
        "strengths": [],
        "needs_regeneration": False,
        "confidence_score": 0.0
    }
    
    if not text or not text.strip():
        assessment["key_issues"].append("빈 내용")
        assessment["needs_regeneration"] = True
        return assessment
    
    # 1. 의미적 완전성 검증 (가장 중요)
    completeness_score = _assess_semantic_completeness(text, original_content)
    
    # 2. 이해도 검증
    understandability_score = _assess_understandability(text)
    
    # 3. 자연스러움 검증
    naturalness_score = _assess_naturalness(text)
    
    # 4. 전문성 검증
    expertise_score = _assess_technical_expertise(text)
    
    # 종합 판단
    overall_score = (completeness_score + understandability_score + 
                    naturalness_score + expertise_score) / 4
    
    assessment["confidence_score"] = overall_score
    
    # 품질 등급 결정
    if overall_score >= 0.8:
        assessment["translation_quality"] = "excellent"
        assessment["is_well_translated"] = True
        assessment["strengths"].append("완벽한 번역")
    elif overall_score >= 0.6:
        assessment["translation_quality"] = "good"
        assessment["is_well_translated"] = True
        assessment["strengths"].append("양호한 번역")
    elif overall_score >= 0.4:
        assessment["translation_quality"] = "fair"
        assessment["key_issues"].append("부분적 개선 필요")
    else:
        assessment["translation_quality"] = "poor"
        assessment["needs_regeneration"] = True
        assessment["key_issues"].append("전체 재생성 필요")
    
    return assessment

def _assess_semantic_completeness(text: str, original_content: str) -> float:
    """
    의미적 완전성 검증 - 원문의 핵심 의미가 보존되었는가?
    """
    if not original_content:
        return 1.0
    
    # 원문의 핵심 개념 추출
    original_concepts = _extract_key_concepts(original_content)
    translated_concepts = _extract_key_concepts(text)
    
    # 개념 보존도 계산
    preserved_concepts = 0
    for concept in original_concepts:
        if _concept_preserved(concept, text):
            preserved_concepts += 1
    
    completeness_ratio = preserved_concepts / max(len(original_concepts), 1)
    
    # 추가 검증: 원문의 논리적 구조 보존
    logical_structure_preserved = _check_logical_structure(original_content, text)
    
    return (completeness_ratio + logical_structure_preserved) / 2

def _extract_key_concepts(text: str) -> List[str]:
    """핵심 개념 추출"""
    concepts = []
    
    # 기술적 개념
    tech_concepts = re.findall(r'\b(?:neural|network|detection|bounding|box|class|probability|convolutional|feature|learning|model|algorithm|architecture|framework|system|method|approach|optimization|training|testing|validation|accuracy|precision|recall|score|data|dataset|image|object|vision|computer|deep|machine|artificial|intelligence|pattern|recognition|classification|regression|clustering|segmentation)\b', text, re.IGNORECASE)
    concepts.extend(tech_concepts)
    
    # 수치적 개념
    numerical_concepts = re.findall(r'\b\d+\.?\d*%|\d+\.?\d*fps|\d+\.?\d*mAP|\d+\.?\d*ms\b', text)
    concepts.extend(numerical_concepts)
    
    # 모델명/데이터셋명
    proper_nouns = re.findall(r'\b(?:YOLO|R-CNN|Fast R-CNN|Faster R-CNN|DPM|PASCAL VOC|ImageNet|COCO|Titan X|GPU)\b', text)
    concepts.extend(proper_nouns)
    
    return list(set(concepts))

def _concept_preserved(concept: str, translated_text: str) -> bool:
    """개념이 번역에서 보존되었는지 확인"""
    # 직접 매칭
    if concept.lower() in translated_text.lower():
        return True
    
    # 한국어 번역 매칭
    korean_translations = {
        'neural': '신경망', 'network': '네트워크', 'detection': '탐지', 'bounding': '경계',
        'box': '상자', 'class': '클래스', 'probability': '확률', 'convolutional': '합성곱',
        'feature': '특징', 'learning': '학습', 'model': '모델', 'algorithm': '알고리즘',
        'architecture': '구조', 'framework': '프레임워크', 'system': '시스템',
        'method': '방법', 'approach': '접근', 'optimization': '최적화',
        'training': '훈련', 'testing': '테스트', 'validation': '검증',
        'accuracy': '정확도', 'precision': '정밀도', 'recall': '재현율',
        'score': '점수', 'data': '데이터', 'dataset': '데이터셋',
        'image': '이미지', 'object': '객체', 'vision': '비전',
        'computer': '컴퓨터', 'deep': '딥', 'machine': '머신',
        'artificial': '인공', 'intelligence': '지능', 'pattern': '패턴',
        'recognition': '인식', 'classification': '분류', 'regression': '회귀',
        'clustering': '클러스터링', 'segmentation': '분할'
    }
    
    if concept.lower() in korean_translations:
        korean_term = korean_translations[concept.lower()]
        if korean_term in translated_text:
            return True
    
    return False

def _check_logical_structure(original: str, translated: str) -> float:
    """논리적 구조 보존 확인"""
    # 원문의 문장 수
    original_sentences = len([s for s in original.split('.') if s.strip()])
    translated_sentences = len([s for s in translated.split('.') if s.strip()])
    
    # 문장 수 비율이 비슷한지 확인
    if original_sentences == 0:
        return 1.0
    
    sentence_ratio = min(translated_sentences, original_sentences) / max(translated_sentences, original_sentences)
    
    # 논리적 연결어 확인
    logical_connectors = ['따라서', '그러므로', '또한', '또한', '그러나', '하지만', '그리고', '또는']
    connector_preserved = any(connector in translated for connector in logical_connectors)
    
    return (sentence_ratio + (0.5 if connector_preserved else 0)) / 1.5

def _assess_understandability(text: str) -> float:
    """
    이해도 검증 - 중학생도 이해할 수 있는가?
    """
    # 복잡한 문장 구조 검사
    complex_sentences = len(re.findall(r'[^.!?]*[,;][^.!?]*[,;][^.!?]*[.!?]', text))
    total_sentences = len([s for s in text.split('.') if s.strip()])
    
    if total_sentences == 0:
        return 0.0
    
    complexity_ratio = complex_sentences / total_sentences
    
    # 설명이 포함된 용어 비율
    explained_terms = len(re.findall(r'\b\w+\([^)]+\)', text))
    total_terms = len(re.findall(r'\b(?:neural|network|detection|bounding|box|class|probability|convolutional|feature|learning|model|algorithm|architecture|framework|system|method|approach|optimization|training|testing|validation|accuracy|precision|recall|score|data|dataset|image|object|vision|computer|deep|machine|artificial|intelligence|pattern|recognition|classification|regression|clustering|segmentation)\b', text, re.IGNORECASE))
    
    explanation_ratio = explained_terms / max(total_terms, 1)
    
    # 이해도 점수 계산
    understandability_score = 1.0 - complexity_ratio + explanation_ratio * 0.5
    
    return min(understandability_score, 1.0)

def _assess_naturalness(text: str) -> float:
    """
    자연스러움 검증 - 한국어로 자연스러운가?
    """
    # 한국어 비율
    hangul_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.sub(r'\s', '', text))
    hangul_ratio = hangul_chars / max(total_chars, 1)
    
    # 영어 문장 검사
    english_sentences = len(re.findall(r'[A-Z][^.]*[.!?]', text))
    total_sentences = len([s for s in text.split('.') if s.strip()])
    english_ratio = english_sentences / max(total_sentences, 1)
    
    # 자연스러운 한국어 표현 검사
    natural_korean_patterns = [
        r'[가-힣]+은\s+[가-힣]+',  # "YOLO는 단일"
        r'[가-힣]+을\s+[가-힣]+',  # "네트워크를 사용하여"
        r'[가-힣]+에서\s+[가-힣]+',  # "이미지에서"
        r'[가-힣]+로\s+[가-힣]+',  # "직접로 예측"
    ]
    
    natural_patterns = sum(len(re.findall(pattern, text)) for pattern in natural_korean_patterns)
    
    # 자연스러움 점수 계산
    naturalness_score = (hangul_ratio * 0.4 + 
                        (1 - english_ratio) * 0.3 + 
                        min(natural_patterns / max(total_sentences, 1), 1.0) * 0.3)
    
    return min(naturalness_score, 1.0)

def _assess_technical_expertise(text: str) -> float:
    """
    전문성 검증 - 기술적 정확성이 있는가?
    """
    # 기술 용어의 정확성
    tech_terms = re.findall(r'\b(?:neural|network|detection|bounding|box|class|probability|convolutional|feature|learning|model|algorithm|architecture|framework|system|method|approach|optimization|training|testing|validation|accuracy|precision|recall|score|data|dataset|image|object|vision|computer|deep|machine|artificial|intelligence|pattern|recognition|classification|regression|clustering|segmentation)\b', text, re.IGNORECASE)
    
    # 설명이 포함된 용어
    explained_terms = len(re.findall(r'\b\w+\([^)]+\)', text))
    
    # 전문성 점수 계산
    if len(tech_terms) == 0:
        return 0.0
    
    expertise_score = explained_terms / len(tech_terms)
    
    # 추가: 수치의 정확성
    numerical_accuracy = _check_numerical_accuracy(text)
    
    return (expertise_score + numerical_accuracy) / 2

def _check_numerical_accuracy(text: str) -> float:
    """수치 정확성 검사"""
    # 수치가 포함된 문장에서 정확성 확인
    numerical_sentences = re.findall(r'[^.!?]*\d+[^.!?]*[.!?]', text)
    
    if not numerical_sentences:
        return 1.0
    
    # 수치가 올바르게 번역되었는지 확인
    accurate_sentences = 0
    for sentence in numerical_sentences:
        # 수치가 한국어로 자연스럽게 표현되었는지 확인
        if re.search(r'\d+[가-힣]+', sentence) or re.search(r'[가-힣]+\d+', sentence):
            accurate_sentences += 1
    
    return accurate_sentences / len(numerical_sentences)

def smart_cache_decision(assessments: List[Dict]) -> Dict:
    """
    🧠 스마트 캐시 결정 - 지능적 판단 기반
    """
    decision = {
        "should_cache": False,
        "cache_quality": "unknown",
        "recommendations": [],
        "confidence": 0.0
    }
    
    if not assessments:
        return decision
    
    # 전체 품질 평가
    excellent_count = sum(1 for a in assessments if a["translation_quality"] == "excellent")
    good_count = sum(1 for a in assessments if a["translation_quality"] == "good")
    fair_count = sum(1 for a in assessments if a["translation_quality"] == "fair")
    poor_count = sum(1 for a in assessments if a["translation_quality"] == "poor")
    
    total_count = len(assessments)
    
    # 품질 분포
    excellent_ratio = excellent_count / total_count
    good_ratio = good_count / total_count
    fair_ratio = fair_count / total_count
    poor_ratio = poor_count / total_count
    
    # 캐시 결정 로직
    if excellent_ratio >= 0.8:
        decision["should_cache"] = True
        decision["cache_quality"] = "excellent"
        decision["recommendations"].append("✅ 완벽한 품질 - 전체 캐시 저장 권장")
    elif excellent_ratio + good_ratio >= 0.8:
        decision["should_cache"] = True
        decision["cache_quality"] = "good"
        decision["recommendations"].append("✅ 양호한 품질 - 전체 캐시 저장 권장")
    elif excellent_ratio + good_ratio >= 0.6:
        decision["should_cache"] = True
        decision["cache_quality"] = "mixed"
        decision["recommendations"].append("⚠️ 혼재된 품질 - 부분 캐시 저장 (저품질 섹션 재생성)")
    else:
        decision["should_cache"] = False
        decision["cache_quality"] = "poor"
        decision["recommendations"].append("❌ 낮은 품질 - 전체 재생성 필요")
    
    # 신뢰도 계산
    decision["confidence"] = (excellent_ratio * 1.0 + good_ratio * 0.8 + 
                            fair_ratio * 0.5 + poor_ratio * 0.2)
    
    # 추가 권장사항
    if poor_count > 0:
        decision["recommendations"].append(f"🔄 {poor_count}개 섹션 재생성 필요")
    
    if fair_count > 0:
        decision["recommendations"].append(f"🔧 {fair_count}개 섹션 부분 개선 필요")
    
    return decision

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
    
    # 지능적 평가 실행
    assessment = intelligent_quality_assessment(test_text, original_content)
    print("=== 지능적 품질 평가 ===")
    print(f"번역 품질: {assessment['translation_quality']}")
    print(f"잘 번역됨: {assessment['is_well_translated']}")
    print(f"신뢰도: {assessment['confidence_score']:.2f}")
    print(f"강점: {assessment['strengths']}")
    print(f"이슈: {assessment['key_issues']}")
    
    # 스마트 캐시 결정
    assessments = [assessment, assessment, assessment]  # 3개 섹션 가정
    decision = smart_cache_decision(assessments)
    print("\n=== 스마트 캐시 결정 ===")
    print(f"캐시 저장: {decision['should_cache']}")
    print(f"캐시 품질: {decision['cache_quality']}")
    print(f"신뢰도: {decision['confidence']:.2f}")
    print(f"권장사항: {decision['recommendations']}")
