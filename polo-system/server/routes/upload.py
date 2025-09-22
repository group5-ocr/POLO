# server/routes/upload.py
from __future__ import annotations

import os, re, unicodedata, time
import httpx
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import tempfile
import shutil
from pathlib import Path
import json

from services.database.db import DB, Tex
from sqlalchemy import select
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

class ModelSendRequest(BaseModel):
    paper_id: str

class ConvertResponse(BaseModel):
    filename: str
    file_size: int
    extracted_text_length: int
    extracted_text_preview: str
    easy_text: str
    status: str
    doc_id: Optional[str] = None
    json_file_path: Optional[str] = None
    arxiv_id: Optional[str] = None
    is_arxiv_paper: bool = False

@router.post("/upload/convert", response_model=ConvertResponse)
async def convert_pdf(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하고 변환하는 엔드포인트
    arXiv ID 자동 추출 기능 포함
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
            
            # arXiv ID 자동 추출 시도
            extracted_arxiv_id = None
            try:
                from services.external.arxiv_downloader_back import extract_arxiv_id_from_pdf
                extracted_arxiv_id = extract_arxiv_id_from_pdf(temp_path, left_margin_px=120)
                if extracted_arxiv_id:
                    print(f"[PDF] arXiv ID 자동 추출됨: {extracted_arxiv_id}")
            except Exception as e:
                print(f"[PDF] arXiv ID 추출 실패: {e}")
            
            # 텍스트 추출 (실제로는 PDF 파싱 라이브러리 사용해야 함)
            try:
                if extracted_arxiv_id:
                    extracted_text = f"업로드된 논문: {file.filename}\n\narXiv ID: {extracted_arxiv_id}\n\n이 PDF에서 arXiv ID가 자동으로 추출되었습니다. arXiv 논문으로 처리할 수 있습니다."
                else:
                    extracted_text = f"업로드된 논문: {file.filename}\n\n이것은 PDF에서 추출된 텍스트의 예시입니다. 실제 구현에서는 PyPDF2나 pdfplumber 같은 라이브러리를 사용하여 PDF에서 텍스트를 추출해야 합니다."
                
                extracted_text_length = len(extracted_text)
                extracted_text_preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            except Exception as e:
                extracted_text = f"텍스트 추출 실패: {str(e)}"
                extracted_text_length = len(extracted_text)
                extracted_text_preview = extracted_text
            
            # Easy 모델 호출 시뮬레이션
            try:
                if extracted_arxiv_id:
                    easy_text = f"이것은 arXiv 논문의 쉬운 버전입니다.\n\narXiv ID: {extracted_arxiv_id}\n원본: {file.filename}\n\n복잡한 학술 용어들이 일반인도 이해할 수 있는 쉬운 말로 바뀌었습니다."
                else:
                    easy_text = f"이것은 AI가 변환한 쉬운 버전의 논문입니다.\n\n원본: {file.filename}\n\n복잡한 학술 용어들이 일반인도 이해할 수 있는 쉬운 말로 바뀌었습니다."
            except Exception as e:
                easy_text = f"변환 실패: {str(e)}"
            
            # 파일을 로컬에 저장
            current_file = Path(__file__).resolve()
            server_dir = current_file.parent.parent  # polo-system/server
            data_dir = server_dir / "data" / "raw"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 고유한 파일명 생성 (타임스탬프 + 원본 파일명)
            timestamp = int(time.time() * 1000)
            safe_filename = f"{timestamp}_{file.filename}"
            file_path = data_dir / safe_filename
            
            # 파일 저장
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # DB에 저장 (ERD 구조에 따라)
            try:
                # 1. origin_file 테이블에 저장
                origin_id = await DB.create_origin_file(user_id=1, filename=file.filename)
                
                # 2. tex 테이블에 저장 (원본 파일 경로)
                tex_id = await DB.create_tex(origin_id=origin_id, file_addr=str(file_path))
                
                # 논문 ID는 tex_id 사용
                doc_id = str(tex_id)
                
            except Exception as db_error:
                print(f"[DB] 데이터베이스 저장 실패: {db_error}")
                # DB 저장 실패 시 파일명 해시 사용
                doc_id = f"doc_{hash(safe_filename)}_{file.size}"
            
            # 성공 응답
            return ConvertResponse(
                filename=file.filename,
                file_size=file.size,
                extracted_text_length=extracted_text_length,
                extracted_text_preview=extracted_text_preview,
                easy_text=easy_text,
                status="success",
                doc_id=doc_id,
                json_file_path=f"/api/download/{doc_id}.json",
                # arXiv ID 정보 추가
                arxiv_id=extracted_arxiv_id,
                is_arxiv_paper=extracted_arxiv_id is not None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/upload/model-status")
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

def _find_file_by_pattern(base_dir: Path, pattern: str) -> Optional[Path]:
    """패턴으로 파일 찾기"""
    if not base_dir.exists():
        return None
    matches = list(base_dir.rglob(pattern))
    return matches[0] if matches else None

def _get_file_info(file_path: Path) -> dict:
    """파일 정보 반환"""
    if not file_path.exists():
        return {"exists": False}
    
    stat = file_path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "path": str(file_path)
    }

@router.get("/upload/download/{filename}")
async def download_file(filename: str):
    """
    일반 파일 다운로드 (JSON, 텍스트 등)
    """
    # 여러 위치에서 파일 찾기
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    search_paths = [
        server_dir / "data" / "outputs" / filename,
        server_dir / "data" / "local" / filename,
        server_dir / "data" / filename,
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            return FileResponse(
                path=str(search_path),
                filename=filename,
                media_type="application/octet-stream"
            )
    
    raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {filename}")

@router.get("/upload/download/easy/{paper_id}")
async def download_easy_file(paper_id: str):
    """
    Easy 모델 출력 파일 다운로드 (이미지들을 ZIP으로 압축)
    """
    import zipfile
    import tempfile
    
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    easy_output_dir = server_dir / "data" / "outputs" / paper_id / "easy_outputs"
    
    if not easy_output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Easy 출력 디렉토리를 찾을 수 없습니다: {paper_id}")
    
    # 이미지 파일들 찾기
    image_files = list(easy_output_dir.glob("*.png")) + list(easy_output_dir.glob("*.jpg"))
    
    if not image_files:
        raise HTTPException(status_code=404, detail=f"Easy 모델 출력 이미지를 찾을 수 없습니다: {paper_id}")
    
    # ZIP 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_file in image_files:
                zipf.write(img_file, img_file.name)
        
        return FileResponse(
            path=tmp_file.name,
            filename=f"{paper_id}_viz_images.zip",
            media_type="application/zip"
        )

@router.get("/upload/download/easy-json/{paper_id}")
async def download_easy_json(paper_id: str):
    """
    Easy 모델 JSON 결과 파일 다운로드
    """
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    easy_output_dir = server_dir / "data" / "outputs" / paper_id / "easy_outputs"
    json_file = easy_output_dir / "easy_results.json"
    
    if not json_file.exists():
        raise HTTPException(status_code=404, detail=f"Easy 모델 JSON 결과 파일을 찾을 수 없습니다: {paper_id}")
    
    return FileResponse(
        path=str(json_file),
        filename=f"{paper_id}_easy_results.json",
        media_type="application/json"
    )

@router.get("/upload/download/math/{paper_id}")
async def download_math_file(paper_id: str):
    """
    Math 모델 출력 파일 다운로드 (JSON, TeX)
    """
    # Math 모델 출력 디렉토리 찾기 (우선순위: outputs > models/math/_build)
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    
    # 1. outputs 디렉토리에서 찾기 (새로운 처리 결과)
    math_output_dir = server_dir / "data" / "outputs" / paper_id / "math_outputs"
    if math_output_dir.exists():
        # JSON 파일 찾기
        json_file = math_output_dir / "equations_explained.json"
        if json_file.exists():
            return FileResponse(
                path=str(json_file),
                filename=f"{paper_id}_math_equations.json",
                media_type="application/json"
            )
        
        # TeX 파일 찾기
        tex_file = math_output_dir / "yolo_math_report.tex"
        if tex_file.exists():
            return FileResponse(
                path=str(tex_file),
                filename=f"{paper_id}_math_report.tex",
                media_type="text/plain"
            )
    
    # 2. models/math/_build 디렉토리에서 찾기 (기존 결과)
    math_output_dir = server_dir.parent / "models" / "math" / "_build"
    if math_output_dir.exists():
        # JSON 파일 찾기
        json_file = math_output_dir / "equations_explained.json"
        if json_file.exists():
            return FileResponse(
                path=str(json_file),
                filename=f"{paper_id}_math_equations.json",
                media_type="application/json"
            )
        
        # TeX 파일 찾기
        tex_file = math_output_dir / "yolo_math_report.tex"
        if tex_file.exists():
            return FileResponse(
                path=str(tex_file),
                filename=f"{paper_id}_math_report.tex",
                media_type="text/plain"
            )
    
    raise HTTPException(status_code=404, detail=f"Math 모델 출력 파일을 찾을 수 없습니다: {paper_id}")

@router.get("/upload/download/math-html/{paper_id}")
async def download_math_html(paper_id: str):
    """
    Math 모델 HTML 결과 파일 다운로드
    """
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    math_output_dir = server_dir / "data" / "outputs" / paper_id / "math_outputs"
    html_file = math_output_dir / f"math_results_{paper_id}.html"
    
    if not html_file.exists():
        raise HTTPException(status_code=404, detail=f"Math 모델 HTML 결과 파일을 찾을 수 없습니다: {paper_id}")
    
    return FileResponse(
        path=str(html_file),
        filename=f"{paper_id}_math_results.html",
        media_type="text/html"
    )

@router.get("/upload/math-status/{paper_id}")
async def get_math_status(paper_id: str):
    """
    Math 모델 처리 상태 및 결과 정보 조회
    """
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    
    # Math 모델 출력 디렉토리 찾기
    math_output_dir = server_dir / "data" / "outputs" / paper_id / "math_outputs"
    
    if not math_output_dir.exists():
        return {
            "status": "not_started",
            "message": "Math 모델 처리가 시작되지 않았습니다",
            "files": []
        }
    
    # 시작 마커 파일 확인
    started_flag = math_output_dir / ".started"
    if started_flag.exists():
        # JSON, TeX, HTML 파일 확인
        json_file = math_output_dir / "equations_explained.json"
        tex_file = math_output_dir / "yolo_math_report.tex"
        html_file = math_output_dir / f"math_results_{paper_id}.html"
        
        if json_file.exists() and tex_file.exists() and html_file.exists():
            return {
                "status": "completed",
                "message": "Math 모델 처리가 완료되었습니다",
                "files": [
                    {
                        "name": "equations_explained.json",
                        "path": str(json_file),
                        "size": json_file.stat().st_size,
                        "type": "json"
                    },
                    {
                        "name": "yolo_math_report.tex", 
                        "path": str(tex_file),
                        "size": tex_file.stat().st_size,
                        "type": "tex"
                    },
                    {
                        "name": f"math_results_{paper_id}.html",
                        "path": str(html_file),
                        "size": html_file.stat().st_size,
                        "type": "html"
                    }
                ]
            }
        else:
            return {
                "status": "processing",
                "message": "Math 모델 처리가 진행 중입니다",
                "files": []
            }
    else:
        return {
            "status": "not_started",
            "message": "Math 모델 처리가 시작되지 않았습니다",
            "files": []
        }

@router.get("/upload/download/raw/{filename}")
async def download_raw_file(filename: str):
    """
    원본 파일 다운로드 (업로드된 파일들)
    """
    # 업로드된 파일들 찾기
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent  # polo-system/server
    raw_dir = server_dir / "data" / "raw"
    arxiv_dir = server_dir / "data" / "arxiv"
    
    # 1. data/raw에서 찾기
    if raw_dir.exists():
        found_file = _find_file_by_pattern(raw_dir, f"*{filename}*")
        if found_file:
            return FileResponse(
                path=str(found_file),
                filename=filename,
                media_type="application/octet-stream"
            )
    
    # 2. data/arxiv에서 찾기
    if arxiv_dir.exists():
        found_file = _find_file_by_pattern(arxiv_dir, f"*{filename}*")
        if found_file:
            return FileResponse(
                path=str(found_file),
                filename=filename,
                media_type="application/octet-stream"
            )
    
    raise HTTPException(status_code=404, detail=f"원본 파일을 찾을 수 없습니다: {filename}")

@router.get("/upload/download/info/{paper_id}")
async def get_download_info(paper_id: str):
    """
    특정 논문의 다운로드 가능한 파일 목록 조회 (DB 기반)
    """
    info = {
        "paper_id": paper_id,
        "files": {
            "easy": [],
            "math": [],
            "raw": [],
            "preprocess": []
        }
    }
    
    try:
        # DB에서 논문 정보 조회
        from services.database.db import get_state, fetch_results
        
        # tex_id로 상태 조회
        tex_id = int(paper_id)
        state = await get_state(tex_id)
        
        if state:
            # 원본 파일 경로 (tex.file_addr)
            if state.file_addr:
                raw_file = Path(state.file_addr)
                if raw_file.exists():
                    info["files"]["raw"].append({
                        "name": raw_file.name,
                        "size": raw_file.stat().st_size,
                        "type": "original"
                    })
            
            # 처리 결과 조회
            results = await fetch_results(tex_id)
            if results:
                # Easy 파일들 (이미지)
                for item in results.get("items", []):
                    if item.get("image_path"):
                        img_path = Path(item["image_path"])
                        if img_path.exists():
                            info["files"]["easy"].append({
                                "name": img_path.name,
                                "size": img_path.stat().st_size,
                                "type": "image"
                            })
                
                # Math 파일들
                for math_file in results.get("math", {}).get("files", []):
                    if math_file.get("file_addr"):
                        math_path = Path(math_file["file_addr"])
                        if math_path.exists():
                            info["files"]["math"].append({
                                "name": math_path.name,
                                "size": math_path.stat().st_size,
                                "type": "math_output"
                            })
        
    except Exception as e:
        print(f"[DOWNLOAD_INFO] DB 조회 실패: {e}")
        # DB 조회 실패 시 기존 방식으로 폴더 스캔
        pass
    
    # 폴더 스캔 방식 (fallback)
    if not any(info["files"].values()):
        # Easy 모델 출력 (이미지들)
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent  # polo-system/server
        easy_dir = server_dir / "data" / "outputs" / paper_id / "easy_outputs"
        if easy_dir.exists():
            image_files = list(easy_dir.glob("*.png")) + list(easy_dir.glob("*.jpg"))
            info["files"]["easy"] = [{"name": f.name, "size": f.stat().st_size, "type": "image"} for f in image_files]
        
        # Math 모델 출력
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent  # polo-system/server
        math_dir = server_dir.parent / "models" / "math" / "_build"
        if math_dir.exists():
            json_file = math_dir / "equations_explained.json"
            tex_file = math_dir / "yolo_math_report.tex"
            
            if json_file.exists():
                info["files"]["math"].append({
                    "name": "equations_explained.json",
                    "size": json_file.stat().st_size,
                    "type": "json"
                })
            
            if tex_file.exists():
                info["files"]["math"].append({
                    "name": "yolo_math_report.tex", 
                    "size": tex_file.stat().st_size,
                    "type": "tex"
                })
        
        # 전처리 출력
        preprocess_dir = server_dir / "data" / "outputs" / paper_id
        if preprocess_dir.exists():
            preprocess_files = list(preprocess_dir.glob("*.jsonl*")) + list(preprocess_dir.glob("*.tex"))
            info["files"]["preprocess"] = [{"name": f.name, "size": f.stat().st_size, "type": "preprocess"} for f in preprocess_files]
        
        # 원본 파일
        arxiv_dir = server_dir / "data" / "arxiv"
        if arxiv_dir.exists():
            raw_files = list(arxiv_dir.rglob(f"*{paper_id}*"))
            info["files"]["raw"] = [{"name": f.name, "size": f.stat().st_size, "type": "original"} for f in raw_files]
    
    return info

@router.post("/upload/from-arxiv")
async def upload_from_arxiv(body: UploadFromArxiv, bg: BackgroundTasks):
    """
    1) origin_file 생성
    2) arXiv tex 소스 다운로드/추출 (arxiv_downloader_back.py 활용)
    3) tex 레코드 생성 (원본 tar 경로 저장)
    4) 전처리 서비스 호출 (완료 시 /api/preprocess/callback)
    """
    if not ARXIV_ID_RE.match(body.arxiv_id):
        raise HTTPException(status_code=400, detail="Invalid arXiv id format")

    safe_title = slugify_filename(body.title)

    # 1) origin_file 생성
    origin_id = await DB.create_origin_file(user_id=body.user_id, filename=safe_title)

    # 2) arXiv fetch & extract (arxiv_downloader_back.py 활용)
    try:
        print(f"[ARXIV] 논문 다운로드 시작: {body.arxiv_id}")
        # 절대 경로로 arxiv 디렉토리 설정
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent  # polo-system/server
        arxiv_dir = server_dir / "data" / "arxiv"
        
        res = await arxiv_client.fetch_and_extract(
            arxiv_id=body.arxiv_id,
            out_root=str(arxiv_dir),
            corp_ca_pem=os.getenv("CORP_CA_PEM") or None,
            left_margin_px=120,  # PDF 왼쪽 여백 설정
            preview_lines=40,    # 미리보기 줄 수
        )
        print(f"[ARXIV] 다운로드 완료: {res['arxiv_id']}")
        print(f"[ARXIV] PDF: {res['pdf_path']}")
        print(f"[ARXIV] 소스: {res['src_tar']}")
        print(f"[ARXIV] 메인 TeX: {res['main_tex']}")
    except Exception as e:
        print(f"[ARXIV] 다운로드 실패: {e}")
        raise HTTPException(status_code=502, detail=f"arXiv 다운로드 실패: {e}")

    # 3) tex 레코드 생성
    tex_id = await DB.create_tex(origin_id=origin_id, file_addr=res["src_tar"])

    # 4) 전처리 서비스 호출 (비동기)
    base_cb = os.getenv("CALLBACK_URL", "http://localhost:8000").rstrip("/")
    callback_url = f"{base_cb}/api/upload/preprocess/callback"

    # preprocess_client.run_async 사용 (비동기)
    import asyncio
    asyncio.create_task(preprocess_client.run_async(str(tex_id), res["source_dir"], callback_url))

    return {
        "ok": True,
        "db_mode": DB.mode,  # "pg" or "local"
        "origin_id": origin_id,
        "tex_id": tex_id,
        "arxiv_id": res["arxiv_id"],
        "paths": {
            "pdf": res["pdf_path"],
            "src_tar": res["src_tar"],
            "source_dir": res["source_dir"],
            "main_tex": res["main_tex"],
        },
        "status": "processing",
        "message": "논문이 다운로드되고 처리 중입니다."
    }

@router.post("/upload/preprocess/callback")
async def preprocess_callback(body: PreprocessCallback):
    """
    전처리 완료 콜백 (DB 업데이트)
    """
    try:
        # DB 업데이트 로직
        from services.database.db import init_pipeline_state
        
        try:
            tex_id = int(body.paper_id)
        except Exception:
            print(f"⚠️ preprocess_callback: invalid paper_id '{body.paper_id}' → skip DB update")
            return {"ok": True, "paper_id": body.paper_id, "status": "ignored"}
        transport_path = Path(body.transport_path)
        
        # transport_path가 파일(transport.json)인 경우 부모 디렉터리를 사용
        base_dir = transport_path if transport_path.is_dir() else transport_path.parent
        
        # 전처리 결과 파일들 찾기
        jsonl_files = list(base_dir.glob("*.jsonl*"))
        tex_files = list(base_dir.glob("*.tex"))
        
        # 파이프라인 상태 초기화
        if jsonl_files:
            jsonl_path = str(jsonl_files[0])
        else:
            jsonl_path = ""
            
        if tex_files:
            math_text_path = str(tex_files[0])
        else:
            math_text_path = ""
        
        # chunks 수 계산 (JSONL 파일에서)
        total_chunks = 0
        if jsonl_files:
            try:
                with open(jsonl_files[0], 'r', encoding='utf-8') as f:
                    total_chunks = sum(1 for line in f if line.strip())
            except:
                total_chunks = 0
        
        # DB 상태 업데이트
        await init_pipeline_state(tex_id, total_chunks, jsonl_path, math_text_path)
        
        # Easy 배치 트리거 (하드코딩 보강)
        try:
            if jsonl_files:
                import httpx, os
                easy_url = os.getenv("EASY_MODEL_URL", "http://localhost:5003")
                print(f"🔍 [DEBUG] Easy 배치 트리거 시작")
                print(f"🔍 [DEBUG] easy_url: {easy_url}")
                print(f"🔍 [DEBUG] jsonl_files: {jsonl_files}")
                
                # 고정 입력/출력 경로로 강제 설정
                server_dir = Path(__file__).resolve().parent.parent
                fixed_tex = server_dir / "data" / "out" / "source" / "merged_body.tex"
                out_dir = server_dir / "data" / "outputs"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"🔍 [DEBUG] out_dir: {out_dir}")
                print(f"🔍 [DEBUG] 전송할 데이터:")
                print(f"  - paper_id: {str(tex_id)}")
                print(f"  - chunks_jsonl: {str(jsonl_files[0])}")
                print(f"  - output_dir: {str(out_dir)}")
                
                async with httpx.AsyncClient(timeout=60) as client:
                    print(f"🔍 [DEBUG] HTTP 요청 시작: {easy_url}/from-transport")
                    r = await client.post(f"{easy_url}/from-transport", json={
                        "paper_id": str(tex_id),
                        "transport_path": str(fixed_tex),
                        "output_dir": str(out_dir),
                    })
                    print(f"🔍 [DEBUG] Easy 배치 응답: {r.status_code}")
                    print(f"🔍 [DEBUG] 응답 내용: {r.text[:500]}...")
                    
                    if r.status_code != 200:
                        print(f"❌ [ERROR] Easy 배치 실패: {r.status_code}")
                        print(f"❌ [ERROR] 응답 내용: {r.text}")
            else:
                print(f"⚠️ [WARNING] jsonl_files가 없어서 Easy 배치 트리거 스킵")
        except httpx.ConnectError as e:
            print(f"❌ [ERROR] Easy 모델 연결 실패: {e}")
            print(f"❌ [ERROR] Easy 모델이 실행 중인지 확인하세요: {easy_url}")
        except httpx.TimeoutException as e:
            print(f"❌ [ERROR] Easy 모델 타임아웃: {e}")
        except Exception as e:
            print(f"❌ [ERROR] Easy 배치 트리거 실패: {e}")
            print(f"❌ [ERROR] 에러 타입: {type(e).__name__}")
            import traceback
            traceback.print_exc()

        print(f"✅ 전처리 완료: paper_id={body.paper_id}, transport_path={body.transport_path}, status={body.status}")
        print(f"📊 총 청크 수: {total_chunks}")
        return {"ok": True, "paper_id": body.paper_id, "status": "callback_received", "total_chunks": total_chunks}
    except Exception as e:
        print(f"❌ 콜백 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Callback processing failed: {e}")

@router.post("/upload/api/preprocess/callback")
async def api_preprocess_callback(body: PreprocessCallback):
    """
    API 전처리 완료 콜백 (DB 업데이트)
    """
    try:
        # DB 업데이트 로직
        from services.database.db import init_pipeline_state
        
        try:
            tex_id = int(body.paper_id)
        except Exception:
            print(f"⚠️ api_preprocess_callback: invalid paper_id '{body.paper_id}' → skip DB update")
            return {"ok": True, "paper_id": body.paper_id, "status": "ignored"}
        transport_path = Path(body.transport_path)
        
        # transport_path가 파일(transport.json)인 경우 부모 디렉터리를 사용
        base_dir = transport_path if transport_path.is_dir() else transport_path.parent
        
        # 전처리 결과 파일들 찾기
        jsonl_files = list(base_dir.glob("*.jsonl*"))
        tex_files = list(base_dir.glob("*.tex"))
        
        # 파이프라인 상태 초기화
        if jsonl_files:
            jsonl_path = str(jsonl_files[0])
        else:
            jsonl_path = ""
            
        if tex_files:
            math_text_path = str(tex_files[0])
        else:
            math_text_path = ""
        
        # chunks 수 계산 (JSONL 파일에서)
        total_chunks = 0
        if jsonl_files:
            try:
                with open(jsonl_files[0], 'r', encoding='utf-8') as f:
                    total_chunks = sum(1 for line in f if line.strip())
            except:
                total_chunks = 0
        
        # DB 상태 업데이트
        await init_pipeline_state(tex_id, total_chunks, jsonl_path, math_text_path)
        
        print(f"✅ API 전처리 완료: paper_id={body.paper_id}, transport_path={body.transport_path}, status={body.status}")
        print(f"📊 총 청크 수: {total_chunks}")
        return {"ok": True, "paper_id": body.paper_id, "status": "callback_received", "total_chunks": total_chunks}
    except Exception as e:
        print(f"❌ API 콜백 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API Callback processing failed: {e}")

@router.post("/upload/send-to-easy")
async def send_to_easy(request: ModelSendRequest, bg: BackgroundTasks):
    """
    Easy 모델로 chunks.jsonl 전송
    """
    try:
        paper_id = request.paper_id
        print(f"🚀 [SERVER] Easy 모델 전송 요청: paper_id={paper_id}")
        
        # 전처리 결과 파일 경로 찾기
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent  # polo-system/server
        # 고정 입력 경로: out/source/merged_body.tex
        tex_path = server_dir / "data" / "out" / "source" / "merged_body.tex"
        if not tex_path.exists():
            print(f"❌ [SERVER] merged_body.tex 파일 없음: {tex_path}")
            raise HTTPException(status_code=404, detail="merged_body.tex 파일을 찾을 수 없습니다")
        
        # Easy 모델 URL (5003으로 통일)
        easy_url = os.getenv("EASY_MODEL_URL", "http://localhost:5003")
        # 출력 루트는 상위 outputs로 고정 (Easy가 내부에서 paper_id 하위로 생성)
        output_dir = server_dir / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 [SERVER] Easy 모델 전송 준비 완료:")
        print(f"  - easy_url: {easy_url}")
        print(f"  - tex_path: {str(tex_path)}")
        print(f"  - output_dir: {str(output_dir)}")
        
        # Easy 모델로 전송 (비동기 백그라운드 실행, 즉시 202 반환)

        async def _run_easy_batch():
            try:
                print(f"🔄 [SERVER] Easy 모델 백그라운드 작업 시작...")
                
                # Easy 모델 연결 테스트
                try:
                    async with httpx.AsyncClient(timeout=10) as test_client:
                        test_response = await test_client.get(f"{easy_url}/health")
                        if test_response.status_code != 200:
                            print(f"❌ [SERVER] Easy 모델 연결 실패: {test_response.status_code}")
                            return
                        print(f"✅ [SERVER] Easy 모델 연결 확인됨")
                except Exception as e:
                    print(f"❌ [SERVER] Easy 모델 연결 테스트 실패: {e}")
                    return
                
                async with httpx.AsyncClient(timeout=1200) as client:  # 20분 허용
                    print(f"📤 [SERVER] Easy 모델로 전송 시작...")
                    # Easy 모델의 새로운 /from-transport 엔드포인트 사용
                    response = await client.post(f"{easy_url}/from-transport", json={
                        "paper_id": paper_id,
                        "transport_path": str(tex_path),
                        "output_dir": str(output_dir)
                    })
                    print(f"📥 [SERVER] Easy 모델 응답: {response.status_code}")
                    if response.status_code != 200:
                        print(f"❌ [SERVER] Easy 모델 응답 실패: {response.status_code} - {response.text}")
                        return
                    print(f"✅ [SERVER] Easy 모델 처리 완료")

                    # 처리 후 결과 파일을 DB에 기록(가능한 경우)
                    try:
                        # paper_id가 doc_ 형태인 경우 DB 저장 스킵 (로컬 파일만 사용)
                        if paper_id.startswith("doc_"):
                            print(f"⚠️ doc_ 형태의 paper_id는 DB 저장 스킵: {paper_id}")
                        else:
                            tex_id = int(paper_id)
                            origin_id = await DB.get_origin_id_from_tex(tex_id)
                            if origin_id:
                                easy_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
                                for easy_file in easy_files:
                                    await DB.create_easy_file(
                                        tex_id=tex_id,
                                        origin_id=origin_id,
                                        filename=easy_file.name,
                                        file_addr=str(easy_file)
                                    )
                                print(f"✅ Easy 파일들 DB에 저장 완료: {len(easy_files)}개 파일")
                            else:
                                print(f"⚠️ origin_id를 찾을 수 없어 Easy 파일 DB 저장 스킵")
                    except Exception as db_error:
                        print(f"❌ Easy 파일 DB 저장 실패: {db_error}")
            except Exception as e:
                print(f"❌ [ERROR] Easy 백그라운드 작업 실패: {e}")

        # 백그라운드 작업 시작
        task = asyncio.create_task(_run_easy_batch())
        
        # 처리 시작 마커 파일 생성 → 결과 폴더 폴링 시 'processing' 상태 표시 가능
        try:
            started_flag = output_dir / ".started"
            started_flag.write_text("started", encoding="utf-8")
        except Exception as e:
            print(f"❌ [SERVER] 시작 마커 파일 생성 실패: {e}")
        
        print(f"✅ [SERVER] Easy 모델 백그라운드 작업 시작됨")
        return JSONResponse(status_code=202, content={"ok": True, "message": "Easy 모델 전송을 시작했습니다", "paper_id": paper_id, "status": "processing"})
                
    except httpx.ConnectError as e:
        print(f"❌ [ERROR] Easy 모델 연결 실패: {e}")
        raise HTTPException(status_code=503, detail=f"Easy 모델 연결 실패: {e}")
    except Exception as e:
        print(f"❌ [ERROR] Easy 모델 전송 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Easy 모델 전송 실패: {e}")

@router.post("/upload/send-to-math")
async def send_to_math(request: ModelSendRequest, bg: BackgroundTasks):
    """
    Math 모델로 merged_body.tex 전송 및 실제 처리 실행
    """
    try:
        paper_id = request.paper_id
        print(f"🚀 [SERVER] Math 모델 처리 요청: paper_id={paper_id}")
        
        # 전처리 결과 파일 경로 찾기
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent  # polo-system/server
        source_dir = server_dir / "data" / "out" / "source"
        
        if not source_dir.exists():
            print(f"❌ [SERVER] 전처리 결과 디렉토리 없음: {source_dir}")
            raise HTTPException(status_code=404, detail="전처리 결과를 찾을 수 없습니다")
        
        # merged_body.tex 파일 찾기
        tex_path = source_dir / "merged_body.tex"
        
        if not tex_path.exists():
            print(f"❌ [SERVER] merged_body.tex 파일 없음: {tex_path}")
            raise HTTPException(status_code=404, detail="merged_body.tex 파일을 찾을 수 없습니다")
        
        # Math 모델 URL
        math_url = os.getenv("MATH_MODEL_URL", "http://localhost:5004")
        output_dir = server_dir / "data" / "outputs" / paper_id / "math_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 [SERVER] Math 모델 처리 준비 완료:")
        print(f"  - math_url: {math_url}")
        print(f"  - tex_path: {str(tex_path)}")
        print(f"  - output_dir: {str(output_dir)}")
        
        # Math 모델로 실제 처리 실행 (Easy 결과 기반)
        async def _run_math_processing():
            try:
                print(f"🔄 [SERVER] Math 모델 백그라운드 작업 시작...")
                
                # Easy 결과 파일 확인 (Math는 Easy 결과를 기반으로 처리)
                easy_output_dir = server_dir / "data" / "outputs" / paper_id
                easy_json_path = easy_output_dir / "easy_results.json"
                
                if not easy_json_path.exists():
                    print(f"❌ [SERVER] Easy 결과 파일 없음: {easy_json_path}")
                    print(f"🔄 [SERVER] Easy 결과 대기 중...")
                    # Easy 결과 대기 (최대 10분)
                    for i in range(120):  # 120 * 5초 = 10분
                        await asyncio.sleep(5)
                        if easy_json_path.exists():
                            print(f"✅ [SERVER] Easy 결과 파일 발견: {easy_json_path}")
                            break
                        print(f"⏳ [SERVER] Easy 결과 대기 중... ({i+1}/120)")
                    else:
                        print(f"❌ [SERVER] Easy 결과 대기 타임아웃")
                        return
                
                # Math 모델 연결 테스트
                try:
                    async with httpx.AsyncClient(timeout=10) as test_client:
                        test_response = await test_client.get(f"{math_url}/health")
                        if test_response.status_code != 200:
                            print(f"❌ [SERVER] Math 모델 연결 실패: {test_response.status_code}")
                            return
                        print(f"✅ [SERVER] Math 모델 연결 확인됨")
                except Exception as e:
                    print(f"❌ [SERVER] Math 모델 연결 테스트 실패: {e}")
                    return
                
                # Math 모델로 실제 처리 실행 (Easy 결과 기반)
                async with httpx.AsyncClient(timeout=1800) as client:  # 30분 허용
                    print(f"📤 [SERVER] Math 모델로 처리 시작...")
                    
                    # Easy 결과를 Math 모델에 전달
                    # 1. Easy 결과에서 수식이 포함된 섹션들 추출
                    import json
                    with open(easy_json_path, 'r', encoding='utf-8') as f:
                        easy_data = json.load(f)
                    
                    # 2. Math 모델에 Easy 결과 전달 (새로운 엔드포인트 사용)
                    response = await client.post(f"{math_url}/math-with-easy", json={
                        "path": str(tex_path),
                        "easy_results": easy_data,  # Easy 결과 전달
                        "paper_id": paper_id
                    })
                    print(f"📥 [SERVER] Math 모델 응답: {response.status_code}")
                    if response.status_code != 200:
                        print(f"❌ [SERVER] Math 모델 응답 실패: {response.status_code} - {response.text}")
                        return
                    
                    # Math 결과 저장
                    math_result = response.json()
                    
                    # Math 결과를 올바른 경로에 저장
                    math_output_dir = server_dir / "data" / "outputs" / paper_id
                    math_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # math_results.json으로 저장 (Result.tsx에서 읽을 수 있도록)
                    math_json_file = math_output_dir / "math_results.json"
                    math_json_file.write_text(json.dumps(math_result, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"✅ [SERVER] Math JSON 결과 생성 완료: {math_json_file}")
                    
                    # 기존 경로에도 저장 (호환성 유지)
                    json_file = output_dir / "equations_explained.json"
                    json_file.write_text(json.dumps(math_result, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"✅ [SERVER] Math JSON 결과 생성 완료 (호환성): {json_file}")
                    
                    # HTML 생성 (선택사항)
                    try:
                        html_response = await client.get(f"{math_url}/html/{str(tex_path)}")
                        if html_response.status_code == 200:
                            html_file = output_dir / f"math_results_{paper_id}.html"
                            html_file.write_text(html_response.text, encoding="utf-8")
                            print(f"✅ [SERVER] Math HTML 결과 생성 완료: {html_file}")
                    except Exception as e:
                        print(f"⚠️ [SERVER] Math HTML 생성 실패 (무시): {e}")

                        # TeX 파일 복사 (math_build_dir 변수 정의 필요)
                        try:
                            math_build_dir = Path("/tmp/math_build")  # 임시 경로
                            tex_file = math_build_dir / "yolo_math_report.tex"
                            if tex_file.exists():
                                import shutil
                                shutil.copy2(tex_file, output_dir / "yolo_math_report.tex")
                                print(f"✅ [SERVER] Math TeX 결과 복사 완료")
                        except Exception as e:
                            print(f"⚠️ [SERVER] Math TeX 복사 실패 (무시): {e}")
                        
                        # 처리 후 결과 파일을 DB에 기록(가능한 경우)
                        try:
                            # paper_id가 doc_ 형태인 경우 DB 저장 스킵 (로컬 파일만 사용)
                            if paper_id.startswith("doc_"):
                                print(f"⚠️ doc_ 형태의 paper_id는 DB 저장 스킵: {paper_id}")
                            else:
                                tex_id = int(paper_id)
                                origin_id = await DB.get_origin_id_from_tex(tex_id)
                                if origin_id:
                                    await DB.save_math_result(
                                        tex_id=tex_id,
                                        origin_id=origin_id,
                                        result_path=str(output_dir / "equations_explained.json"),
                                        sections=None
                                    )
                                    await DB.set_flag(tex_id=tex_id, field="math_done", value=True)
                                    print(f"✅ Math 결과 DB에 저장 완료")
                                else:
                                    print(f"⚠️ origin_id를 찾을 수 없어 Math 결과 DB 저장 스킵")
                        except Exception as db_error:
                            print(f"❌ Math 결과 DB 저장 실패: {db_error}")
                        
                    except Exception as copy_error:
                        print(f"❌ [SERVER] Math 결과 파일 복사 실패: {copy_error}")
                        
            except Exception as e:
                print(f"❌ [ERROR] Math 백그라운드 작업 실패: {e}")

        # 백그라운드 작업 시작
        task = asyncio.create_task(_run_math_processing())
        
        # 처리 시작 마커 파일 생성
        try:
            started_flag = output_dir / ".started"
            started_flag.write_text("started", encoding="utf-8")
        except Exception as e:
            print(f"❌ [SERVER] 시작 마커 파일 생성 실패: {e}")
        
        print(f"✅ [SERVER] Math 모델 백그라운드 작업 시작됨")
        return JSONResponse(status_code=202, content={"ok": True, "message": "Math 모델 처리를 시작했습니다", "paper_id": paper_id, "status": "processing"})
                
    except Exception as e:
        print(f"❌ [ERROR] Math 모델 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Math 모델 처리 실패: {e}")

@router.post("/upload/send-to-viz")
async def send_to_viz(request: ModelSendRequest, bg: BackgroundTasks):
    """
    Viz 모델로 Easy 결과 전송 및 시각화 생성
    """
    try:
        paper_id = request.paper_id
        print(f"🚀 [SERVER] Viz 모델 처리 요청: paper_id={paper_id}")
        
        # Easy 결과 파일 경로 찾기
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        # Easy 결과 파일을 여러 위치에서 찾기
        easy_json_path = server_dir / "data" / "outputs" / paper_id / "easy_results.json"
        if not easy_json_path.exists():
            # easy_outputs 디렉토리에서 찾기
            easy_json_path = server_dir / "data" / "outputs" / paper_id / "easy_outputs" / "easy_results.json"
        
        if not easy_json_path.exists():
            print(f"❌ [SERVER] Easy 결과 파일 없음: {easy_json_path}")
            raise HTTPException(status_code=404, detail="Easy 결과를 찾을 수 없습니다")
        
        # Viz 모델 URL
        viz_url = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
        output_dir = server_dir / "data" / "outputs" / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 [SERVER] Viz 모델 처리 준비 완료:")
        print(f"  - viz_url: {viz_url}")
        print(f"  - easy_json_path: {str(easy_json_path)}")
        print(f"  - output_dir: {str(output_dir)}")
        
        # Viz 모델로 처리 실행
        async def _run_viz_processing():
            try:
                print(f"🔄 [SERVER] Viz 모델 백그라운드 작업 시작...")
                
                # Viz 모델 연결 테스트
                try:
                    async with httpx.AsyncClient(timeout=10) as test_client:
                        test_response = await test_client.get(f"{viz_url}/health")
                        if test_response.status_code != 200:
                            print(f"❌ [SERVER] Viz 모델 연결 실패: {test_response.status_code}")
                            return
                        print(f"✅ [SERVER] Viz 모델 연결 확인됨")
                except Exception as e:
                    print(f"❌ [SERVER] Viz 모델 연결 테스트 실패: {e}")
                    return
                
                async with httpx.AsyncClient(timeout=1800) as client:  # 30분 허용
                    print(f"📤 [SERVER] Viz 모델로 처리 시작...")
                    
                    # Easy 결과를 Viz 모델에 전달
                    import json
                    with open(easy_json_path, 'r', encoding='utf-8') as f:
                        easy_data = json.load(f)
                    
                    # Viz 모델에 Easy 결과 전달
                    response = await client.post(f"{viz_url}/generate-visualizations", json={
                        "paper_id": paper_id,
                        "easy_results": easy_data,
                        "output_dir": str(output_dir)
                    })
                    print(f"📥 [SERVER] Viz 모델 응답: {response.status_code}")
                    if response.status_code != 200:
                        print(f"❌ [SERVER] Viz 모델 응답 실패: {response.status_code} - {response.text}")
                        return
                    
                    # Viz 결과 저장
                    viz_result = response.json()
                    viz_json_file = output_dir / "viz_results.json"
                    viz_json_file.write_text(json.dumps(viz_result, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"✅ [SERVER] Viz JSON 결과 생성 완료: {viz_json_file}")
                    
            except Exception as e:
                print(f"❌ [SERVER] Viz 모델 처리 실패: {e}")
        
        # 백그라운드에서 실행
        bg.add_task(_run_viz_processing)
        
        return {
            "message": "Viz 모델 처리 시작됨",
            "paper_id": paper_id,
            "status": "processing"
        }
        
    except Exception as e:
        print(f"❌ [SERVER] Viz 모델 전송 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Viz 모델 전송 실패: {e}")

@router.post("/upload/send-to-viz-api")
async def send_to_viz_api(request: ModelSendRequest, bg: BackgroundTasks):
    """
    Viz API 모델로 고급 시각화 생성
    """
    try:
        paper_id = request.paper_id
        print(f"🚀 [SERVER] Viz API 모델 처리 요청: paper_id={paper_id}")
        
        # Easy 결과 파일 경로 찾기
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        # Easy 결과 파일을 여러 위치에서 찾기
        easy_json_path = server_dir / "data" / "outputs" / paper_id / "easy_results.json"
        if not easy_json_path.exists():
            # easy_outputs 디렉토리에서 찾기
            easy_json_path = server_dir / "data" / "outputs" / paper_id / "easy_outputs" / "easy_results.json"
        
        if not easy_json_path.exists():
            print(f"❌ [SERVER] Easy 결과 파일 없음: {easy_json_path}")
            raise HTTPException(status_code=404, detail="Easy 결과를 찾을 수 없습니다")
        
        # Viz API 모델 URL
        viz_api_url = os.getenv("VIZ_API_URL", "http://localhost:5006")
        output_dir = server_dir / "data" / "outputs" / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 [SERVER] Viz API 모델 처리 준비 완료:")
        print(f"  - viz_api_url: {viz_api_url}")
        print(f"  - easy_json_path: {str(easy_json_path)}")
        print(f"  - output_dir: {str(output_dir)}")
        
        # Viz API 모델로 처리 실행
        async def _run_viz_api_processing():
            try:
                print(f"🔄 [SERVER] Viz API 모델 백그라운드 작업 시작...")
                
                # Viz API 모델 연결 테스트
                try:
                    async with httpx.AsyncClient(timeout=10) as test_client:
                        test_response = await test_client.get(f"{viz_api_url}/health")
                        if test_response.status_code != 200:
                            print(f"❌ [SERVER] Viz API 모델 연결 실패: {test_response.status_code}")
                            return
                        print(f"✅ [SERVER] Viz API 모델 연결 확인됨")
                except Exception as e:
                    print(f"❌ [SERVER] Viz API 모델 연결 테스트 실패: {e}")
                    return
                
                async with httpx.AsyncClient(timeout=1800) as client:  # 30분 허용
                    print(f"📤 [SERVER] Viz API 모델로 처리 시작...")
                    
                    # Easy 결과를 Viz API 모델에 전달
                    import json
                    with open(easy_json_path, 'r', encoding='utf-8') as f:
                        easy_data = json.load(f)
                    
                    # Viz API 모델에 Easy 결과 전달
                    response = await client.post(f"{viz_api_url}/generate-advanced-visualizations", json={
                        "paper_id": paper_id,
                        "easy_results": easy_data,
                        "output_dir": str(output_dir)
                    })
                    print(f"📥 [SERVER] Viz API 모델 응답: {response.status_code}")
                    if response.status_code != 200:
                        print(f"❌ [SERVER] Viz API 모델 응답 실패: {response.status_code} - {response.text}")
                        return
                    
                    # Viz API 결과 저장
                    viz_api_result = response.json()
                    viz_api_json_file = output_dir / "viz_api_results.json"
                    viz_api_json_file.write_text(json.dumps(viz_api_result, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"✅ [SERVER] Viz API JSON 결과 생성 완료: {viz_api_json_file}")
                    
            except Exception as e:
                print(f"❌ [SERVER] Viz API 모델 처리 실패: {e}")
        
        # 백그라운드에서 실행
        bg.add_task(_run_viz_api_processing)
        
        return {
            "message": "Viz API 모델 처리 시작됨",
            "paper_id": paper_id,
            "status": "processing"
        }
        
    except Exception as e:
        print(f"❌ [SERVER] Viz API 모델 전송 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Viz API 모델 전송 실패: {e}")


@router.post("/integrated-result/create")
async def create_integrated_result(request: ModelSendRequest):
    """
    통합 결과 생성 (데이터베이스 저장)
    """
    try:
        paper_id = request.paper_id
        print(f"🚀 [SERVER] 통합 결과 생성 요청: paper_id={paper_id}")
        
        # tex_id 찾기
        tex_id = int(paper_id)
        
        # origin_id 찾기
        async with DB.session()() as s:
            tex_result = await s.execute(select(Tex).where(Tex.tex_id == tex_id))
            tex_record = tex_result.scalar_one_or_none()
            if not tex_record:
                raise HTTPException(status_code=404, detail="Tex record not found")
            origin_id = tex_record.origin_id
        
        # 통합 결과 생성
        result_id = await DB.create_integrated_result(
            tex_id=tex_id,
            origin_id=origin_id,
            paper_id=paper_id,
            processing_status="processing"
        )
        
        # Easy 결과 로드 및 저장
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        easy_file = server_dir / "data" / "outputs" / paper_id / "easy_outputs" / "easy_results.json"
        
        if easy_file.exists():
            with open(easy_file, 'r', encoding='utf-8') as f:
                easy_data = json.load(f)
            
            # Easy 섹션들 저장
            if "easy_sections" in easy_data:
                section_ids = await DB.save_easy_sections(result_id, easy_data["easy_sections"])
                print(f"✅ [SERVER] Easy 섹션 {len(section_ids)}개 저장 완료")
            
            # 논문 정보 업데이트
            paper_info = easy_data.get("paper_info", {})
            await DB.update_integrated_result(
                result_id=result_id,
                paper_title=paper_info.get("paper_title"),
                paper_authors=paper_info.get("paper_authors"),
                paper_venue=paper_info.get("paper_venue"),
                total_sections=len(easy_data.get("easy_sections", []))
            )
        
        # Math 결과 로드 및 저장
        math_file = server_dir / "data" / "outputs" / paper_id / "math_outputs" / "equations_explained.json"
        
        if math_file.exists():
            with open(math_file, 'r', encoding='utf-8') as f:
                math_data = json.load(f)
            
            # Math 수식들 저장
            if "items" in math_data:
                equations = []
                for i, item in enumerate(math_data["items"]):
                    equation = {
                        "math_equation_id": f"eq_{i+1}",
                        "math_equation_latex": item.get("latex"),
                        "math_equation_explanation": item.get("explanation"),
                        "math_equation_section_ref": item.get("section_ref")
                    }
                    equations.append(equation)
                
                equation_ids = await DB.save_math_equations(result_id, equations)
                print(f"✅ [SERVER] Math 수식 {len(equation_ids)}개 저장 완료")
                
                # 수식 개수 업데이트
                await DB.update_integrated_result(
                    result_id=result_id,
                    total_equations=len(equations)
                )
        
        # Viz 결과 로드 및 저장
        viz_file = server_dir / "data" / "outputs" / paper_id / "viz_outputs" / "visualizations.json"
        
        if viz_file.exists():
            with open(viz_file, 'r', encoding='utf-8') as f:
                viz_data = json.load(f)
            
            # 시각화들 저장
            if "generated_visualizations" in viz_data:
                visualizations = []
                for viz in viz_data["generated_visualizations"]:
                    viz_item = {
                        "viz_type": viz.get("visualization_type"),
                        "viz_title": viz.get("title"),
                        "viz_description": viz.get("description"),
                        "viz_image_path": viz.get("image_path"),
                        "viz_metadata": viz.get("metadata")
                    }
                    visualizations.append(viz_item)
                
                viz_ids = await DB.save_visualizations(result_id, visualizations)
                print(f"✅ [SERVER] 시각화 {len(viz_ids)}개 저장 완료")
                
                # 시각화 개수 업데이트
                await DB.update_integrated_result(
                    result_id=result_id,
                    total_visualizations=len(visualizations)
                )
        
        # 완료 상태로 업데이트
        await DB.update_integrated_result(
            result_id=result_id,
            processing_status="completed"
        )
        
        # tex 테이블 업데이트
        async with DB.session()() as s:
            tex_record.integrated_done = True
            tex_record.integrated_result_id = result_id
            await s.commit()
        
        print(f"✅ [SERVER] 통합 결과 생성 완료: result_id={result_id}")
        
        return {
            "success": True,
            "result_id": result_id,
            "paper_id": paper_id,
            "message": "통합 결과가 성공적으로 생성되었습니다"
        }
        
    except Exception as e:
        print(f"❌ [SERVER] 통합 결과 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통합 결과 생성 실패: {e}")

@router.get("/integrated-result/{paper_id}")
async def get_integrated_result(paper_id: str):
    """
    통합 결과 조회 (Easy + Math + 시각화) - 파일 기반 (기존 방식)
    """
    try:
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        output_dir = server_dir / "data" / "outputs" / paper_id
        
        # Easy 결과 로드
        easy_data = None
        easy_error = None
        try:
            easy_file = output_dir / "easy_outputs" / "easy_results.json"
            if easy_file.exists():
                with open(easy_file, 'r', encoding='utf-8') as f:
                    easy_data = json.load(f)
            else:
                easy_error = "Easy 결과 파일이 없습니다"
        except Exception as e:
            easy_error = f"Easy 결과 로드 실패: {str(e)}"
        
        # Math 결과 로드
        math_data = None
        math_error = None
        try:
            math_file = output_dir / "math_outputs" / "equations_explained.json"
            if math_file.exists():
                with open(math_file, 'r', encoding='utf-8') as f:
                    math_data = json.load(f)
            else:
                math_error = "Math 결과 파일이 없습니다"
        except Exception as e:
            math_error = f"Math 결과 로드 실패: {str(e)}"
        
        # 통합 데이터 생성
        integrated_data = {
            "paper_info": {
                "paper_id": paper_id,
                "paper_title": easy_data.get("paper_info", {}).get("paper_title", f"논문 {paper_id}") if easy_data else f"논문 {paper_id}",
                "paper_authors": easy_data.get("paper_info", {}).get("paper_authors", "Unknown") if easy_data else "Unknown",
                "paper_venue": easy_data.get("paper_info", {}).get("paper_venue", "Unknown") if easy_data else "Unknown",
                "total_sections": easy_data.get("paper_info", {}).get("total_sections", 0) if easy_data else 0,
                "total_equations": len(math_data.get("items", [])) if math_data else 0
            },
            "easy_sections": easy_data.get("easy_sections", []) if easy_data else [],
            "math_equations": [],
            "model_errors": {
                "easy_model_error": easy_error,
                "math_model_error": math_error,
                "viz_api_error": None
            },
            "processing_logs": []
        }
        
        # Math 데이터 변환 및 Easy 섹션과 매핑
        if math_data and "items" in math_data:
            try:
                easy_sections = easy_data.get("easy_sections", []) if easy_data else []
                
                for i, item in enumerate(math_data["items"]):
                    try:
                        # Math 수식이 어느 Easy 섹션에 속하는지 결정
                        # 기본적으로 순서대로 매핑하되, 섹션 수를 초과하지 않도록 함
                        section_index = i % len(easy_sections) if easy_sections else 0
                        section_ref = easy_sections[section_index]["easy_section_id"] if easy_sections else f"easy_section_{i+1}"
                        
                        equation = {
                            "math_equation_id": f"math_equation_{i+1}",
                            "math_equation_index": f"({i+1})",
                            "math_equation_latex": item.get("equation", ""),
                            "math_equation_explanation": item.get("explanation", ""),
                            "math_equation_context": f"수식 {i+1}",
                            "math_equation_section_ref": section_ref  # Easy 섹션 ID와 매핑
                        }
                        integrated_data["math_equations"].append(equation)
                    except Exception as e:
                        integrated_data["processing_logs"].append(f"수식 {i+1} 변환 실패: {str(e)}")
                        # 에러가 발생해도 빈 수식으로 추가
                        equation = {
                            "math_equation_id": f"math_equation_{i+1}",
                            "math_equation_index": f"({i+1})",
                            "math_equation_latex": item.get("equation", ""),
                            "math_equation_explanation": f"수식 변환 중 오류 발생: {str(e)}",
                            "math_equation_context": f"수식 {i+1}",
                            "math_equation_section_ref": f"easy_section_{i+1}"
                        }
                        integrated_data["math_equations"].append(equation)
            except Exception as e:
                integrated_data["processing_logs"].append(f"Math 데이터 변환 실패: {str(e)}")
        else:
            integrated_data["processing_logs"].append("Math 데이터가 없거나 형식이 올바르지 않습니다")
        
        return integrated_data
        
    except Exception as e:
        print(f"❌ [ERROR] 통합 결과 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통합 결과 조회 실패: {e}")

@router.get("/integrated-result/{paper_id}/download")
async def download_integrated_result(paper_id: str):
    """
    통합 결과 HTML 다운로드
    """
    try:
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        output_dir = server_dir / "data" / "outputs" / paper_id
        
        # 통합 결과 HTML 생성
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>통합 결과 - 논문 {paper_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ padding: 30px; }}
        .section {{ margin-bottom: 30px; padding: 25px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }}
        .section-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; }}
        .paragraph {{ margin-bottom: 15px; line-height: 1.6; }}
        .equation {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; font-family: 'Courier New', monospace; }}
        .visualization {{ text-align: center; margin: 20px 0; }}
        .visualization img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI 논문 분석 결과</h1>
            <p>논문 ID: {paper_id}</p>
        </div>
        <div class="content">
            <p>이 결과는 AI가 논문을 분석하여 생성한 통합 결과입니다.</p>
            <p>쉬운 설명, 수식 해설, 시각화가 포함되어 있습니다.</p>
        </div>
    </div>
</body>
</html>
        """
        
        # HTML 파일로 저장
        html_file = output_dir / "integrated_result.html"
        html_file.write_text(html_content, encoding="utf-8")
        
        return FileResponse(
            path=str(html_file),
            filename=f"integrated_result_{paper_id}.html",
            media_type="text/html"
        )
        
    except Exception as e:
        print(f"❌ [ERROR] 통합 결과 다운로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통합 결과 다운로드 실패: {e}")

@router.get("/results/{paper_id}/easy_results.json")
async def get_easy_results(paper_id: str):
    """Easy 모델 결과 직접 반환"""
    try:
        results_dir = Path("data/outputs") / paper_id
        easy_file = results_dir / "easy_results.json"
        
        if not easy_file.exists():
            raise HTTPException(status_code=404, detail="Easy 결과 파일이 없습니다")
        
        return FileResponse(
            path=str(easy_file),
            media_type="application/json",
            filename=f"easy_results_{paper_id}.json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Easy 결과 로드 실패: {str(e)}")

@router.get("/results/{paper_id}/ready")
async def check_results_ready(paper_id: str):
    """결과 파일이 준비되었는지 확인"""
    try:
        # 결과 디렉토리 확인
        results_dir = Path("data/outputs") / paper_id
        if not results_dir.exists():
            return {"status": "not_found", "ok": False, "message": "결과 디렉토리가 없습니다"}
        
        # easy_results.json 파일 확인
        easy_file = results_dir / "easy_results.json"
        if not easy_file.exists():
            return {"status": "processing", "ok": False, "message": "Easy 모델 처리 중"}
        
        # math_results.json 파일 확인
        math_file = results_dir / "math_results.json"
        if not math_file.exists():
            return {"status": "processing", "ok": False, "message": "Math 모델 처리 중"}
        
        return {"status": "ready", "ok": True, "message": "모든 결과가 준비되었습니다"}
    except Exception as e:
        return {"status": "error", "ok": False, "message": f"오류 발생: {str(e)}"}
