# server/routes/upload.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from services.database import db as DB
from services import arxiv_client, preprocess_client

router = APIRouter()


class UploadFromArxiv(BaseModel):
    user_id: int = Field(..., description="업로드한 사용자 ID")
    arxiv_id: str = Field(..., description="예: '2408.12345'")
    title: str = Field(..., description="논문 제목 (origin_file.filename 저장용)")


@router.post("/from-arxiv")
async def upload_from_arxiv(body: UploadFromArxiv, bg: BackgroundTasks):
    """
    1) origin_file 생성
    2) arXiv tex 소스 다운로드/추출 (벤더 스크립트 활용)
    3) tex 레코드 생성 (원본 source tar 경로 저장)
    4) 전처리 서비스 호출 (완료 시 /generate/preprocess/callback)
    """
    # 1) origin_file
    origin_id = await DB.create_origin_file(user_id=body.user_id, filename=body.title)

    # 2) arXiv fetch & extract
    try:
        res = await arxiv_client.fetch_and_extract(arxiv_id=body.arxiv_id, out_root="data/arxiv")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"arXiv fetch failed: {e}")

    # 3) tex
    tex_id = await DB.create_tex(origin_id=origin_id, file_addr=res["src_tar"])

    # 4) preprocess 시작
    bg.add_task(
        preprocess_client.run,
        str(tex_id),
        res["source_dir"],
        "/generate/preprocess/callback"
    )

    return {
        "ok": True,
        "db_mode": DB.mode,  # "pg" or "local"
        "origin_id": origin_id,
        "tex_id": tex_id,
        "paths": {
            "pdf": res["pdf_path"],
            "src_tar": res["src_tar"],
            "source_dir": res["source_dir"],
            "main_tex": res["main_tex"],
        }
    }