#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv1 논문 통합 분석 JSONL 생성기

이 스크립트는 Easy 모델 결과와 Math 모델 결과를 받아서
논문의 자연스러운 흐름에 따라 통합된 JSONL을 생성합니다.
"""

import json
import os
from pathlib import Path
from config import Config


def read_jsonl(file_path):
    """JSONL 파일을 읽어서 리스트로 반환"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return []
    
    return data


def save_jsonl(data, file_path):
    """데이터를 JSONL 파일로 저장"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"JSONL 파일 저장 완료: {file_path}")
    except Exception as e:
        print(f"파일 저장 오류: {e}")


def save_integrated_jsonl(integrated_result, file_path):
    """통합 결과를 JSONL 파일로 저장"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 논문 정보 먼저 저장
            f.write(json.dumps(integrated_result["paper_info"], ensure_ascii=False) + '\n')
            
            # 각 섹션별로 저장
            for section in integrated_result["sections"]:
                f.write(json.dumps(section, ensure_ascii=False) + '\n')
        
        print(f"통합 JSONL 파일 저장 완료: {file_path}")
    except Exception as e:
        print(f"파일 저장 오류: {e}")


def find_easy_content(section_name, easy_sections):
    """Easy 모델 결과에서 해당 섹션의 내용 찾기"""
    for section in easy_sections:
        if section.get('section_name') == section_name:
            return section.get('easy_content', '')
    return ''


def find_math_equations(section_range, math_equations):
    """해당 섹션 범위에 포함되는 수식들 찾기"""
    section_equations = []
    for equation in math_equations:
        line_start = equation.get('line_start', 0)
        if section_range['start'] <= line_start <= section_range['end']:
            # Math 모델 결과를 통합 결과 형식에 맞게 변환
            integrated_equation = {
                'index': equation.get('index', 0),
                'line_number': line_start,
                'equation': equation.get('equation', ''),
                'explanation': equation.get('explanation', ''),
                'kind': equation.get('kind', 'equation'),
                'env': equation.get('env', 'align')
            }
            section_equations.append(integrated_equation)
    
    return section_equations


def find_visualizations(section_name, viz_sections, easy_content=""):
    """섹션에 해당하는 시각화들 찾기 (용어 기반)"""
    section_visualizations = []
    
    # 1. 기존 시각화 데이터에서 찾기
    for viz_section in viz_sections:
        if viz_section.get('section_name') == section_name:
            visualizations = viz_section.get('visualizations', [])
            for viz in visualizations:
                integrated_viz = {
                    'id': viz.get('id', ''),
                    'type': viz.get('type', ''),
                    'title': viz.get('title', ''),
                    'description': viz.get('description', ''),
                    'line_start': viz.get('line_start', ''),
                    'line_end': viz.get('line_end', ''),
                    'image_path': viz.get('image_path', ''),
                    'spec': viz.get('spec', {})
                }
                section_visualizations.append(integrated_viz)
    
    # 2. 용어 기반 시각화 생성 (실제 프로젝트 로직 시뮬레이션)
    term_based_viz = generate_term_based_visualizations(section_name, easy_content)
    section_visualizations.extend(term_based_viz)
    
    return section_visualizations

def generate_term_based_visualizations(section_name, easy_content):
    """용어 기반 시각화 생성 (실제 프로젝트 로직 시뮬레이션)"""
    visualizations = []
    
    # 용어-시각화 매핑 (실제 프로젝트의 glossary 기반)
    term_viz_mapping = {
        "격자": {
            "type": "cell_scale",
            "title": "YOLO 격자 구조",
            "description": "이미지를 격자로 나누어 객체를 탐지하는 방식",
            "image_path": "viz/charts/cell_scale_cell_scale.png"
        },
        "신뢰도": {
            "type": "confidence_plot",
            "title": "신뢰도 점수 분포",
            "description": "객체 탐지 신뢰도 점수의 분포",
            "image_path": "viz/charts/confidence_distribution.png"
        },
        "mAP": {
            "type": "metric_table",
            "title": "성능 지표 비교",
            "description": "다양한 방법의 mAP 성능 비교",
            "image_path": "viz/charts/map_detection_rubric_metric_table.png"
        },
        "FPS": {
            "type": "bar_group",
            "title": "처리 속도 비교",
            "description": "초당 프레임 수 성능 비교",
            "image_path": "viz/charts/bar_map_detection_bar_group.png"
        },
        "정확도": {
            "type": "metric_table",
            "title": "정확도 성능 비교",
            "description": "다양한 방법의 정확도 비교",
            "image_path": "viz/charts/accuracy_rubric_metric_table.png"
        },
        "활성화": {
            "type": "activations_panel",
            "title": "네트워크 활성화 패턴",
            "description": "신경망 레이어의 활성화 패턴",
            "image_path": "viz/charts/activations_panel_activations_panel.png"
        }
    }
    
    # 쉬운 설명에서 용어 감지 및 시각화 생성
    for term, viz_info in term_viz_mapping.items():
        if term in easy_content:
            viz_id = f"term_{term}_{section_name}"
            visualization = {
                'id': viz_id,
                'type': viz_info['type'],
                'title': viz_info['title'],
                'description': viz_info['description'],
                'line_start': 0,  # 용어 기반이므로 라인 번호 없음
                'line_end': 0,
                'image_path': viz_info['image_path'],
                'spec': {
                    'title': viz_info['title'],
                    'description': viz_info['description'],
                    'trigger_term': term
                }
            }
            visualizations.append(visualization)
    
    return visualizations


def create_integrated_section(section_name, section_range, easy_sections, math_equations, viz_sections):
    """섹션별 통합 데이터 생성"""
    # Easy 모델 결과에서 해당 섹션 찾기
    easy_content = find_easy_content(section_name, easy_sections)
    
    # Math 모델 결과에서 해당 섹션 범위의 수식들 찾기
    math_equations_list = find_math_equations(section_range, math_equations)
    
    # Viz 모델 결과에서 해당 섹션의 시각화들 찾기 (용어 기반 포함)
    visualizations_list = find_visualizations(section_name, viz_sections, easy_content)
    
    # 통합된 섹션 데이터 생성
    integrated_section = {
        'paper_id': Config.PAPER_INFO['paper_id'],
        'section_name': section_name,
        'section_title': Config.get_section_title(section_name),
        'section_order': Config.get_section_order(section_name),
        'line_start': section_range['start'],
        'line_end': section_range['end'],
        'easy_content': easy_content,
        'math_equations': math_equations_list,
        'visualizations': visualizations_list
    }
    
    return integrated_section


def create_integrated_result(easy_sections, math_equations, viz_sections):
    """통합 결과 생성"""
    integrated_sections = []
    
    # 논문 구조에 따라 섹션별로 통합
    for section_name in Config.SECTION_ORDER:
        if section_name in Config.PAPER_STRUCTURE:
            section_range = Config.PAPER_STRUCTURE[section_name]
            integrated_section = create_integrated_section(
                section_name, section_range, easy_sections, math_equations, viz_sections
            )
            integrated_sections.append(integrated_section)
    
    # 논문 정보를 Easy 모델 결과에서 가져오기 (첫 번째 섹션에서)
    paper_info = {}
    if easy_sections and len(easy_sections) > 0:
        first_section = easy_sections[0]
        paper_info = {
            "paper_id": first_section.get("paper_id", ""),
            "paper_title": first_section.get("paper_title", ""),
            "paper_authors": first_section.get("paper_authors", ""),
            "paper_date": first_section.get("paper_date", ""),
            "paper_venue": first_section.get("paper_venue", ""),
            "paper_doi": first_section.get("paper_doi", ""),
            "total_sections": len(integrated_sections),
            "total_equations": sum(len(section.get("math_equations", [])) for section in integrated_sections)
        }
    else:
        # Fallback: Config에서 가져오기
        paper_info = {
            "paper_id": Config.PAPER_INFO["paper_id"],
            "paper_title": Config.PAPER_INFO["paper_title"],
            "paper_authors": Config.PAPER_INFO["paper_authors"],
            "paper_date": Config.PAPER_INFO["paper_date"],
            "paper_venue": Config.PAPER_INFO["paper_venue"],
            "paper_doi": Config.PAPER_INFO["paper_doi"],
            "total_sections": len(integrated_sections),
            "total_equations": sum(len(section.get("math_equations", [])) for section in integrated_sections)
        }
    
    return {
        "paper_info": paper_info,
        "sections": integrated_sections
    }


def main():
    """메인 함수"""
    print("YOLOv1 논문 통합 분석 JSONL 생성기 시작...")
    
    # 설정
    config = Config()
    
    # 파일 경로 설정
    easy_jsonl_path = config.EASY_JSONL_PATH
    math_jsonl_path = config.MATH_JSONL_PATH
    integrated_jsonl_path = config.INTEGRATED_JSONL_PATH
    
    # 데이터 디렉토리 생성
    os.makedirs(os.path.dirname(integrated_jsonl_path), exist_ok=True)
    
    print(f"Easy 모델 결과 읽기: {easy_jsonl_path}")
    easy_sections = read_jsonl(easy_jsonl_path)
    
    print(f"Math 모델 결과 읽기: {math_jsonl_path}")
    math_equations = read_jsonl(math_jsonl_path)
    
    print(f"Viz 모델 결과 읽기: {config.VIZ_JSONL_PATH}")
    viz_sections = read_jsonl(config.VIZ_JSONL_PATH)
    
    if not easy_sections:
        print("Easy 모델 결과를 읽을 수 없습니다.")
        return
    
    if not math_equations:
        print("Math 모델 결과를 읽을 수 없습니다.")
        return
    
    print("통합 결과 생성 중...")
    integrated_result = create_integrated_result(easy_sections, math_equations, viz_sections)
    
    print(f"통합 결과 저장: {integrated_jsonl_path}")
    save_integrated_jsonl(integrated_result, integrated_jsonl_path)
    
    # 프론트엔드용 데이터 저장
    frontend_data_path = "frontend/data/integrated_result.jsonl"
    os.makedirs(os.path.dirname(frontend_data_path), exist_ok=True)
    print(f"프론트엔드용 데이터 저장: {frontend_data_path}")
    save_integrated_jsonl(integrated_result, frontend_data_path)
    
    # 결과 요약
    print("\n=== 생성 완료 ===")
    print(f"논문 ID: {integrated_result['paper_info']['paper_id']}")
    print(f"논문 제목: {integrated_result['paper_info']['paper_title']}")
    print(f"통합 섹션 수: {integrated_result['paper_info']['total_sections']}")
    print(f"총 수식 수: {integrated_result['paper_info']['total_equations']}")
    
    # 섹션별 수식 분포
    print("\n=== 섹션별 수식 분포 ===")
    for section in integrated_result['sections']:
        equation_count = len(section['math_equations'])
        print(f"{section['section_title']}: {equation_count}개 수식")
    
    print(f"\n통합 JSONL 파일: {integrated_jsonl_path}")


if __name__ == "__main__":
    main()
