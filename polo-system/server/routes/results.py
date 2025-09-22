from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from services.database import db as DB
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pathlib import Path

router = APIRouter()


class ResultItem(BaseModel):
    index: int
    text: Optional[str] = None
    rewritten_text: Optional[str] = None
    image_path: Optional[str] = None
    original_images: Optional[List[Dict[str, Any]]] = None


class MathFileItem(BaseModel):
    math_id: int
    file_addr: Optional[str] = None


class EasyFileItem(BaseModel):
    easy_id: int
    filename: Optional[str] = None
    file_addr: Optional[str] = None


class FinalResult(BaseModel):
    paper_id: int
    total_chunks: int
    easy_done: int
    viz_done: int
    math_done: bool
    items: List[ResultItem]
    math: Dict[str, List[MathFileItem]]
    easy: Dict[str, List[EasyFileItem]]


# ---- 통합 결과용 새로운 스키마들 ----
class PaperInfo(BaseModel):
    paper_id: str
    paper_title: Optional[str] = None
    paper_authors: Optional[str] = None
    paper_venue: Optional[str] = None
    total_sections: int = 0
    total_equations: int = 0
    total_visualizations: int = 0


class EasyParagraph(BaseModel):
    easy_paragraph_id: str
    easy_paragraph_text: Optional[str] = None
    easy_visualization_trigger: bool = False


class EasyVisualization(BaseModel):
    viz_id: int
    viz_type: Optional[str] = None
    viz_title: Optional[str] = None
    viz_description: Optional[str] = None
    viz_image_path: Optional[str] = None
    viz_metadata: Optional[Dict[str, Any]] = None


class EasySection(BaseModel):
    easy_section_id: str
    easy_section_title: Optional[str] = None
    easy_section_type: Optional[str] = None
    easy_section_order: Optional[int] = None
    easy_section_level: Optional[int] = None
    easy_section_parent: Optional[str] = None
    easy_content: Optional[str] = None
    easy_paragraphs: List[EasyParagraph] = []
    easy_visualizations: List[EasyVisualization] = []


class MathEquation(BaseModel):
    math_equation_id: str
    math_equation_latex: Optional[str] = None
    math_equation_explanation: Optional[str] = None
    math_equation_section_ref: Optional[str] = None


class IntegratedResult(BaseModel):
    paper_info: PaperInfo
    easy_sections: List[EasySection] = []
    math_equations: List[MathEquation] = []
    processing_status: str = "pending"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.get("/{tex_id}", response_model=FinalResult)
async def get_results(tex_id: int):
    """
    프론트 템플릿에서 바로 사용할 수 있는 최종 결과 JSON (기존 방식)
    """
    data = await DB.fetch_results(tex_id)
    if not data:
        raise HTTPException(status_code=404, detail="result not ready")
    return data


@router.get("/{tex_id}/integrated", response_model=IntegratedResult)
async def get_integrated_results(tex_id: int):
    """
    통합 결과 조회 (새로운 구조)
    """
    data = await DB.get_integrated_result(tex_id)
    if not data:
        raise HTTPException(status_code=404, detail="integrated result not found")
    return data


@router.get("/db/status")
async def get_database_status():
    """
    데이터베이스 연결 상태 및 테이블 정보 조회
    """
    try:
        # 연결 테스트
        connection_ok = await DB.test_connection()
        
        # 테이블 정보 조회
        table_info = await DB.get_table_info()
        
        return {
            "connection": "ok" if connection_ok else "failed",
            "mode": DB.mode,
            "table_info": table_info
        }
    except Exception as e:
        return {
            "connection": "failed",
            "error": str(e)
        }


@router.get("/{tex_id}/status")
async def get_status(tex_id: int):
    """
    진행률(퍼센트, 카운터) 조회용 보조 엔드포인트
    """
    st = await DB.get_state(tex_id)
    if not st:
        raise HTTPException(404, "not found")

    pct_easy = (st.easy_done / st.total_chunks * 100.0) if st.total_chunks else 0.0
    pct_viz  = (st.viz_done / st.total_chunks * 100.0) if st.total_chunks else 0.0

    return {
        "paper_id": tex_id,
        "total_chunks": st.total_chunks,
        "easy_done": st.easy_done,
        "viz_done": st.viz_done,
        "math_done": st.math_done,
        "progress": {
            "easy_pct": round(pct_easy, 2),
            "viz_pct": round(pct_viz, 2),
        }
    }

@router.get("/{paper_id}/html")
async def get_html_result(paper_id: str):
    """
    Easy 모델 결과 HTML 파일 제공
    """
    # HTML 파일 경로 찾기 (신규/레거시 경로 모두 시도)
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    # 신규 기본: data/outputs/{paper_id}/easy_results.html
    html_path = server_dir / "data" / "outputs" / str(paper_id) / "easy_results.html"
    if not html_path.exists():
        # 레거시: data/outputs/{paper_id}/easy_outputs/easy_results.html
        legacy_path = server_dir / "data" / "outputs" / str(paper_id) / "easy_outputs" / "easy_results.html"
        if legacy_path.exists():
            html_path = legacy_path
        else:
            # 폴백: outputs 하위에서 재귀적으로 탐색
            candidates = list((server_dir / "data" / "outputs").rglob(f"{paper_id}/easy_results.html"))
            if not candidates:
                candidates = list((server_dir / "data" / "outputs").rglob(f"{paper_id}*/easy_results.html"))
            if candidates:
                html_path = candidates[0]
    
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML result not found")
    
    # 브라우저에서 바로 열리도록 inline으로 제공 (다운로드 강제 방지)
    return FileResponse(
        path=str(html_path),
        media_type="text/html",
        headers={"Content-Disposition": f"inline; filename=polo_easy_explanation_{paper_id}.html"}
    )

@router.get("/{paper_id}/ready")
async def is_result_ready(paper_id: str):
    """
    Easy 결과 파일 존재 여부를 반환 (DB 연동 없이 로컬 파일만 확인)
    { ok: bool, html: str|None, json: str|None }
    """
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    outputs_root = server_dir / "data" / "outputs"
    # 신규 기본 경로
    out_dir_new = outputs_root / str(paper_id)
    html_path = out_dir_new / "easy_results.html"
    json_path = out_dir_new / "easy_results.json"
    # 레거시 경로 보조
    if not html_path.exists() and not json_path.exists():
        out_dir_legacy = outputs_root / str(paper_id) / "easy_outputs"
        if (out_dir_legacy / "easy_results.html").exists() or (out_dir_legacy / "easy_results.json").exists():
            html_path = out_dir_legacy / "easy_results.html"
            json_path = out_dir_legacy / "easy_results.json"
            out_dir = out_dir_legacy
        else:
            # 재귀 탐색 (폴백)
            found_html = list(outputs_root.rglob(f"{paper_id}/easy_results.html"))
            found_json = list(outputs_root.rglob(f"{paper_id}/easy_results.json"))
            out_dir = outputs_root
            if found_html:
                html_path = found_html[0]
            if found_json:
                json_path = found_json[0]
    else:
        out_dir = out_dir_new

    # 결과 파일 존재 여부 확인
    if not out_dir.exists():
        return {
            "ok": False,
            "status": "not_found",
            "html": None,
            "json": None,
        }

    status = "idle"
    if (out_dir / ".started").exists():
        status = "processing"
    if html_path.exists() or json_path.exists():
        status = "ready"
    
    # 디버깅을 위한 로그 (너무 자주 출력되지 않도록 제한)
    import time
    if not hasattr(is_result_ready, '_last_log_time'):
        is_result_ready._last_log_time = 0
    
    current_time = time.time()
    if current_time - is_result_ready._last_log_time > 30:  # 30초마다 로그
        print(f"🔍 [READY] paper_id={paper_id}, status={status}, html={html_path.exists()}, json={json_path.exists()}")
        is_result_ready._last_log_time = current_time
    
    return {
        "ok": html_path.exists() or json_path.exists(),
        "status": status,
        "html": str(html_path) if html_path.exists() else None,
        "json": str(json_path) if json_path.exists() else None,
    }

@router.get("/{paper_id}/original-images")
async def get_original_images(paper_id: str):
    """
    원본 논문 이미지들을 ZIP으로 압축하여 다운로드
    """
    import zipfile
    import tempfile
    
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    
    # 전처리 결과 디렉토리에서 원본 이미지 찾기
    source_dir = server_dir / "data" / "out" / "source"
    arxiv_dir = server_dir / "data" / "arxiv"
    
    # 이미지 파일들 찾기
    image_files = []
    
    # 1. source 디렉토리에서 찾기
    if source_dir.exists():
        for ext in ['*.pdf', '*.jpg', '*.jpeg', '*.png', '*.eps']:
            image_files.extend(list(source_dir.glob(ext)))
    
    # 2. arxiv 디렉토리에서 찾기
    if arxiv_dir.exists():
        for paper_dir in arxiv_dir.iterdir():
            if paper_dir.is_dir():
                for ext in ['*.pdf', '*.jpg', '*.jpeg', '*.png', '*.eps']:
                    image_files.extend(list(paper_dir.rglob(ext)))
    
    if not image_files:
        raise HTTPException(status_code=404, detail="원본 이미지를 찾을 수 없습니다")
    
    # ZIP 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_file in image_files:
                # 상대 경로로 ZIP에 추가
                arcname = img_file.name
                zipf.write(img_file, arcname)
        
        return FileResponse(
            path=tmp_file.name,
            filename=f"{paper_id}_original_images.zip",
            media_type="application/zip"
        )

@router.get("/{paper_id}/image/{filename}")
async def get_single_image(paper_id: str, filename: str):
    """
    단일 원본 이미지 파일 다운로드
    """
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    
    # 여러 위치에서 이미지 파일 찾기
    search_paths = [
        server_dir / "data" / "out" / "source" / filename,
        server_dir / "data" / "arxiv" / "1506.02640" / "source" / filename,
    ]
    
    # arxiv 디렉토리에서 재귀 검색
    arxiv_dir = server_dir / "data" / "arxiv"
    if arxiv_dir.exists():
        for paper_dir in arxiv_dir.iterdir():
            if paper_dir.is_dir():
                search_paths.append(paper_dir / "source" / filename)
                search_paths.extend(list(paper_dir.rglob(filename)))
    
    for search_path in search_paths:
        if search_path.exists():
            return FileResponse(
                path=str(search_path),
                filename=filename,
                media_type="application/octet-stream"
            )
    
    raise HTTPException(status_code=404, detail=f"이미지 파일을 찾을 수 없습니다: {filename}")

