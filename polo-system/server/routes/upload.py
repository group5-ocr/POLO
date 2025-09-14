# server/routes/upload.py
from __future__ import annotations

import os, re, unicodedata
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
import tempfile
import shutil
from pathlib import Path

from services.database.db import DB
from services import arxiv_client, preprocess_client

router = APIRouter()
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")

def slugify_filename(name: str) -> str:
    # 간단한 파일명 안전화 (공백/특수문자 제거)
    value = unicodedata.normalize("NFKC", name).strip()
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value[:200] or "paper"

class UploadFromArxiv(BaseModel):
    user_id: int = Field(..., description="업로드한 사용자 ID")
    arxiv_id: str = Field(..., description="예: '2408.12345'")
    title: str = Field(..., description="논문 제목 (origin_file.filename 저장용)")

class PreprocessCallback(BaseModel):
    paper_id: str
    transport_path: str
    status: str

class ConvertResponse(BaseModel):
    filename: str
    file_size: int
    extracted_text_length: int
    extracted_text_preview: str
    easy_text: str
    status: str
    doc_id: Optional[str] = None
    json_file_path: Optional[str] = None

@router.post("/convert", response_model=ConvertResponse)
async def convert_pdf(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하고 변환하는 엔드포인트
    """
    try:
        # 파일 크기 체크 (50MB)
        if file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일은 50MB 이하만 가능합니다.")
        
        # PDF 파일인지 체크
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")
        
        # 임시 디렉토리에 파일 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 간단한 텍스트 추출 (실제로는 PDF 파싱 라이브러리 사용해야 함)
            try:
                # 여기서는 간단한 시뮬레이션
                extracted_text = f"업로드된 논문: {file.filename}\n\n이것은 PDF에서 추출된 텍스트의 예시입니다. 실제 구현에서는 PyPDF2나 pdfplumber 같은 라이브러리를 사용하여 PDF에서 텍스트를 추출해야 합니다."
                extracted_text_length = len(extracted_text)
                extracted_text_preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            except Exception as e:
                extracted_text = f"텍스트 추출 실패: {str(e)}"
                extracted_text_length = len(extracted_text)
                extracted_text_preview = extracted_text
            
            # Easy 모델 호출 시뮬레이션
            try:
                # 실제로는 Easy 모델 API를 호출해야 함
                easy_text = f"이것은 AI가 변환한 쉬운 버전의 논문입니다.\n\n원본: {file.filename}\n\n복잡한 학술 용어들이 일반인도 이해할 수 있는 쉬운 말로 바뀌었습니다. 실제 구현에서는 Easy 모델 API를 호출하여 텍스트를 변환해야 합니다."
            except Exception as e:
                easy_text = f"변환 실패: {str(e)}"
            
            # 성공 응답
            return ConvertResponse(
                filename=file.filename,
                file_size=file.size,
                extracted_text_length=extracted_text_length,
                extracted_text_preview=extracted_text_preview,
                easy_text=easy_text,
                status="success",
                doc_id=f"doc_{hash(file.filename)}_{file.size}",
                json_file_path=f"/api/download/{file.filename}.json"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/model-status")
async def get_model_status():
    """
    AI 모델 상태 확인
    """
    try:
        # Easy 모델 상태 확인
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:5003/health", timeout=5.0)
                easy_available = response.status_code == 200
            except:
                easy_available = False
        
        # Math 모델 상태 확인
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:5004/health", timeout=5.0)
                math_available = response.status_code == 200
        except:
            math_available = False
        
        return {
            "model_available": easy_available and math_available,
            "easy_model": easy_available,
            "math_model": math_available,
            "status": "healthy" if (easy_available and math_available) else "unhealthy"
        }
    except Exception as e:
        return {
            "model_available": False,
            "easy_model": False,
            "math_model": False,
            "status": "error",
            "error": str(e)
        }

@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    파일 다운로드 (임시 구현)
    """
    raise HTTPException(status_code=501, detail="다운로드 기능은 아직 구현되지 않았습니다.")

@router.post("/from-arxiv")
async def upload_from_arxiv(body: UploadFromArxiv, bg: BackgroundTasks):
    """
    1) origin_file 생성
    2) arXiv tex 소스 다운로드/추출 (벤더 스크립트)
    3) tex 레코드 생성 (원본 tar 경로 저장)
    4) 전처리 서비스 호출 (완료 시 /generate/preprocess/callback)
    """
    if not ARXIV_ID_RE.match(body.arxiv_id):
        raise HTTPException(status_code=400, detail="Invalid arXiv id format")

    safe_title = slugify_filename(body.title)

    # 1) origin_file
    origin_id = await DB.create_origin_file(user_id=body.user_id, filename=safe_title)

    # 2) arXiv fetch & extract (항상 먼저 거침)
    try:
        res = await arxiv_client.fetch_and_extract(
            arxiv_id=body.arxiv_id,
            out_root=os.getenv("ARXIV_OUT_ROOT", "server/data/arxiv"),
            corp_ca_pem=os.getenv("CORP_CA_PEM") or None,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"arXiv fetch failed: {e}")

    # 3) tex
    tex_id = await DB.create_tex(origin_id=origin_id, file_addr=res["src_tar"])

    # 4) preprocess 시작 (콜백은 절대 URL로)
    base_cb = os.getenv("CALLBACK_URL", "http://localhost:8000").rstrip("/")
    callback_url = f"{base_cb}/generate/preprocess/callback"

    # preprocess_client.run 이 동기라면 BackgroundTasks로 충분
    # 비동기라면 asyncio.create_task를 써야 함 → 두 경우 모두 대응
    run_fn = getattr(preprocess_client, "run", None)
    run_async = getattr(preprocess_client, "run_async", None)

    if callable(run_fn):
        # sync
        bg.add_task(run_fn, str(tex_id), res["source_dir"], callback_url)
    elif callable(run_async):
        # async
        import asyncio
        asyncio.create_task(run_async(str(tex_id), res["source_dir"], callback_url))
    else:
        raise HTTPException(status_code=500, detail="No preprocess client entrypoint found")

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
        },
    }

@router.post("/preprocess/callback")
async def preprocess_callback(body: PreprocessCallback):
    """
    전처리 완료 콜백
    """
    try:
        # TODO: DB 업데이트 로직 (paper_id 기준 상태 갱신)
        print(f"✅ 전처리 완료: paper_id={body.paper_id}, transport_path={body.transport_path}, status={body.status}")
        return {"ok": True, "paper_id": body.paper_id, "status": "callback_received"}
    except Exception as e:
        print(f"❌ 콜백 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Callback processing failed: {e}")
