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


class FinalResult(BaseModel):
    paper_id: int
    total_chunks: int
    easy_done: int
    viz_done: int
    math_done: bool
    items: List[ResultItem]
    math: Dict[str, List[MathFileItem]]


@router.get("/{tex_id}", response_model=FinalResult)
async def get_results(tex_id: int):
    """
    프론트 템플릿에서 바로 사용할 수 있는 최종 결과 JSON
    """
    data = await DB.fetch_results(tex_id)
    if not data:
        raise HTTPException(status_code=404, detail="result not ready")
    return data


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
    # HTML 파일 경로 찾기
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    html_path = server_dir / "data" / "outputs" / str(paper_id) / "easy_outputs" / "easy_results.html"
    
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
    out_dir = server_dir / "data" / "outputs" / str(paper_id) / "easy_outputs"
    html_path = out_dir / "easy_results.html"
    json_path = out_dir / "easy_results.json"

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

