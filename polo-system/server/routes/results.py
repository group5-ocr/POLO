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
async def get_html_result(paper_id: int):
    """
    Easy 모델 결과 HTML 파일 제공
    """
    # HTML 파일 경로 찾기
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    html_path = server_dir / "data" / "outputs" / str(paper_id) / "easy_outputs" / "easy_results.html"
    
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML result not found")
    
    return FileResponse(
        path=str(html_path),
        media_type="text/html",
        filename=f"polo_easy_explanation_{paper_id}.html"
    )

