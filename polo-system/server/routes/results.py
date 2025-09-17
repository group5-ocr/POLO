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

