"""
[Figure] 토큰 처리 및 라벨 매칭으로 figure 통합
- 순서 기반 토큰 교체
- 라벨 우선 매칭
- JSON 데이터 구조 업데이트
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional


# [Figure] 토큰 패턴 (대소문자 무관)
FIG_TOKEN = re.compile(r'\[Figure[^\]]*\]', re.IGNORECASE)


def find_figure_by_label(figures: List[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    """
    라벨로 figure 찾기
    
    Args:
        figures: figure 정보 리스트
        label: 찾을 라벨
        
    Returns:
        매칭된 figure 정보 또는 None
    """
    if not label:
        return None
    
    for fig in figures:
        fig_label = fig.get("label", "")
        if fig_label and (label in fig_label or fig_label in label):
            return fig
    
    return None


def find_figure_by_key(figures: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    """
    키로 figure 찾기 (label이 없을 때 fallback)
    
    Args:
        figures: figure 정보 리스트
        key: 찾을 키
        
    Returns:
        매칭된 figure 정보 또는 None
    """
    if not key:
        return None
    
    for fig in figures:
        fig_key = fig.get("key", "")
        if fig_key and (key in fig_key or fig_key in key):
            return fig
    
    return None


def extract_figure_references(text: str) -> List[str]:
    """
    텍스트에서 figure 참조 추출
    Figure~\ref{fig:model}, Figure 1, fig:model 등 패턴 검색
    
    Args:
        text: 검색할 텍스트
        
    Returns:
        추출된 참조 리스트
    """
    references = []
    
    # LaTeX 참조 패턴: \ref{fig:...}, \label{fig:...}
    latex_refs = re.findall(r'\\(?:ref|label)\{(fig:[^}]+)\}', text, re.IGNORECASE)
    references.extend(latex_refs)
    
    # 일반적인 figure 참조: Figure 1, Fig. 2 등
    fig_nums = re.findall(r'(?:Figure|Fig\.?)\s*(\d+)', text, re.IGNORECASE)
    references.extend([f"fig:{num}" for num in fig_nums])
    
    # 직접적인 라벨 언급
    direct_labels = re.findall(r'(fig:[a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    references.extend(direct_labels)
    
    return list(set(references))  # 중복 제거


def match_figure_to_text(text: str, figures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    텍스트 내용을 기반으로 가장 적합한 figure 찾기
    
    Args:
        text: 분석할 텍스트
        figures: figure 정보 리스트
        
    Returns:
        매칭된 figure 정보 또는 None
    """
    # 1. 텍스트에서 figure 참조 추출
    references = extract_figure_references(text)
    
    # 2. 참조와 라벨 매칭
    for ref in references:
        matched = find_figure_by_label(figures, ref)
        if matched:
            print(f"✅ [MATCH] 라벨 매칭: {ref} → {matched.get('key')}")
            return matched
    
    # 3. 키워드 기반 매칭 (figure caption이나 key에 텍스트의 주요 단어가 포함된 경우)
    text_lower = text.lower()
    keywords = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)  # 4글자 이상 단어
    
    for fig in figures:
        fig_caption = (fig.get("caption") or "").lower()
        fig_key = (fig.get("key") or "").lower()
        
        # caption이나 key에 키워드가 많이 포함된 figure 우선
        matches = sum(1 for kw in keywords if kw in fig_caption or kw in fig_key)
        if matches >= 2:  # 2개 이상 키워드 매칭
            print(f"✅ [MATCH] 키워드 매칭: {matches} keywords → {fig.get('key')}")
            return fig
    
    return None


def create_figure_metadata(figure_info: Dict[str, Any], static_prefix: str = "/static") -> Dict[str, Any]:
    """
    figure 정보를 프론트엔드용 메타데이터로 변환
    
    Args:
        figure_info: build_figure_index 결과의 figure 정보
        static_prefix: 정적 파일 URL 프리픽스
        
    Returns:
        프론트엔드용 figure 메타데이터
    """
    web_url = f"{static_prefix}/{figure_info['rel_url']}"
    
    # 모든 페이지 URL 생성
    all_pages = []
    for png_path in figure_info.get("pngs", []):
        png_name = Path(png_path).name
        fig_stem = Path(figure_info["key"]).stem
        page_url = f"{static_prefix}/viz/figures/{fig_stem}/{png_name}?v={figure_info['hash']}"
        all_pages.append(page_url)
    
    return {
        "image_path": web_url,
        "caption": figure_info.get("caption"),
        "label": figure_info.get("label"),
        "src_file": figure_info.get("src_file"),
        "all_pages": all_pages,
        "order": figure_info.get("order"),
        "key": figure_info.get("key")
    }


def attach_figures(
    integrated_json_path: Path,
    out_path: Path,
    figures: List[Dict[str, Any]],
    static_prefix: str = "/static"
) -> None:
    """
    integrated_result.json에 figure 정보 첨부
    
    Args:
        integrated_json_path: 입력 JSON 파일 경로
        out_path: 출력 JSON 파일 경로
        figures: build_figure_index 결과
        static_prefix: 정적 파일 URL 프리픽스
    """
    if not integrated_json_path.exists():
        print(f"❌ 통합 JSON 파일 없음: {integrated_json_path}")
        return
    
    try:
        # JSON 데이터 로드
        data = json.loads(integrated_json_path.read_text(encoding='utf-8'))
        
        # figure iterator (순서 기반 fallback용)
        available_figures = figures.copy()
        used_figures = set()
        
        total_tokens = 0
        matched_tokens = 0
        
        # 1. 섹션별 처리
        for sec in data.get("easy_sections", []):
            # 섹션 본문에서 [Figure] 토큰 처리
            sec_content = sec.get("easy_content", "")
            if isinstance(sec_content, str) and FIG_TOKEN.search(sec_content):
                # 라벨 기반 매칭 시도
                matched_fig = match_figure_to_text(sec_content, available_figures)
                
                if not matched_fig and available_figures:
                    # fallback: 순서 기반
                    matched_fig = available_figures[0]
                
                if matched_fig:
                    # 섹션에 figure 첨부
                    sec.setdefault("figures", []).append(create_figure_metadata(matched_fig, static_prefix))
                    
                    # 사용된 figure 제거
                    if matched_fig in available_figures:
                        available_figures.remove(matched_fig)
                    used_figures.add(matched_fig.get("key"))
                    matched_tokens += 1
                    
                    print(f"✅ [ATTACH] 섹션 figure: {sec.get('easy_section_title', 'Unknown')} → {matched_fig.get('key')}")
                
                # [Figure] 토큰 제거 (첫 번째만)
                sec["easy_content"] = FIG_TOKEN.sub("", sec_content, count=1)
                total_tokens += 1
            
            # 2. 문단별 처리
            for p in sec.get("easy_paragraphs", []):
                p_text = p.get("easy_paragraph_text", "")
                if not isinstance(p_text, str):
                    continue
                
                if FIG_TOKEN.search(p_text):
                    total_tokens += 1
                    
                    # 라벨 기반 매칭 시도
                    matched_fig = match_figure_to_text(p_text, available_figures)
                    
                    if not matched_fig and available_figures:
                        # fallback: 순서 기반
                        matched_fig = available_figures[0]
                    
                    if matched_fig:
                        # 문단에 figure 첨부
                        p["figure"] = create_figure_metadata(matched_fig, static_prefix)
                        
                        # 사용된 figure 제거
                        if matched_fig in available_figures:
                            available_figures.remove(matched_fig)
                        used_figures.add(matched_fig.get("key"))
                        matched_tokens += 1
                        
                        print(f"✅ [ATTACH] 문단 figure: {p.get('easy_paragraph_id', 'Unknown')} → {matched_fig.get('key')}")
                    
                    # [Figure] 토큰 제거 (첫 번째만)
                    p["easy_paragraph_text"] = FIG_TOKEN.sub("", p_text, count=1)
        
        # 결과 저장
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        print(f"✅ [INTEGRATE] 완료: {matched_tokens}/{total_tokens} 토큰 매칭, {len(used_figures)} figures 사용")
        print(f"✅ [INTEGRATE] 출력: {out_path}")
        
    except Exception as e:
        print(f"❌ [INTEGRATE] figure 통합 실패: {e}")


def validate_figure_integration(json_path: Path) -> Dict[str, Any]:
    """
    figure 통합 결과 검증
    
    Args:
        json_path: 통합된 JSON 파일 경로
        
    Returns:
        검증 결과 딕셔너리
    """
    if not json_path.exists():
        return {"valid": False, "error": "파일 없음"}
    
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
        
        stats = {
            "sections": 0,
            "paragraphs": 0,
            "section_figures": 0,
            "paragraph_figures": 0,
            "remaining_tokens": 0
        }
        
        for sec in data.get("easy_sections", []):
            stats["sections"] += 1
            
            # 섹션 figure 카운트
            if sec.get("figures"):
                stats["section_figures"] += len(sec["figures"])
            
            # 남은 토큰 카운트
            sec_content = sec.get("easy_content", "")
            if isinstance(sec_content, str):
                stats["remaining_tokens"] += len(FIG_TOKEN.findall(sec_content))
            
            # 문단 처리
            for p in sec.get("easy_paragraphs", []):
                stats["paragraphs"] += 1
                
                if p.get("figure"):
                    stats["paragraph_figures"] += 1
                
                p_text = p.get("easy_paragraph_text", "")
                if isinstance(p_text, str):
                    stats["remaining_tokens"] += len(FIG_TOKEN.findall(p_text))
        
        return {
            "valid": True,
            "stats": stats,
            "total_figures": stats["section_figures"] + stats["paragraph_figures"]
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}
