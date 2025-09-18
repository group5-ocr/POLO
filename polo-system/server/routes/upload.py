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
        # JSON 파일 확인
        json_file = math_output_dir / "equations_explained.json"
        tex_file = math_output_dir / "yolo_math_report.tex"
        
        if json_file.exists() and tex_file.exists():
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
                
                out_dir = (transport_path if transport_path.is_dir() else transport_path.parent).parent / "outputs" / str(tex_id) / "easy_outputs"
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
                        "transport_path": str(jsonl_files[0]),
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
        source_dir = server_dir / "data" / "out" / "source"
        
        if not source_dir.exists():
            print(f"❌ [SERVER] 전처리 결과 디렉토리 없음: {source_dir}")
            raise HTTPException(status_code=404, detail="전처리 결과를 찾을 수 없습니다")
        
        # merged_body.tex 파일 찾기 (Easy 모델이 섹션 기반으로 변경됨)
        tex_path = source_dir / "merged_body.tex"
        
        if not tex_path.exists():
            print(f"❌ [SERVER] merged_body.tex 파일 없음: {tex_path}")
            raise HTTPException(status_code=404, detail="merged_body.tex 파일을 찾을 수 없습니다")
        
        # Easy 모델 URL (5003으로 통일)
        easy_url = os.getenv("EASY_MODEL_URL", "http://localhost:5003")
        output_dir = server_dir / "data" / "outputs" / paper_id / "easy_outputs"
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
        
        # Math 모델로 실제 처리 실행 (비동기 백그라운드 실행, 즉시 202 반환)
        async def _run_math_processing():
            try:
                print(f"🔄 [SERVER] Math 모델 백그라운드 작업 시작...")
                
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
                
                # Math 모델로 실제 처리 실행
                async with httpx.AsyncClient(timeout=1800) as client:  # 30분 허용
                    print(f"📤 [SERVER] Math 모델로 처리 시작...")
                    response = await client.post(f"{math_url}/math", json={
                        "path": str(tex_path)
                    })
                    print(f"📥 [SERVER] Math 모델 응답: {response.status_code}")
                    if response.status_code != 200:
                        print(f"❌ [SERVER] Math 모델 응답 실패: {response.status_code} - {response.text}")
                        return
                    
                    result = response.json()
                    print(f"✅ [SERVER] Math 모델 처리 완료")
                    print(f"📊 [SERVER] Math 결과: {result}")
                    
                    # 결과 파일을 지정된 output_dir로 복사
                    try:
                        outputs = result.get("outputs", {})
                        json_path = outputs.get("json")
                        report_tex = outputs.get("report_tex")
                        math_out_dir = outputs.get("out_dir")
                        
                        if json_path and Path(json_path).exists():
                            import shutil
                            shutil.copy2(json_path, output_dir / "equations_explained.json")
                            print(f"✅ [SERVER] Math JSON 결과 복사 완료")
                        
                        if report_tex and Path(report_tex).exists():
                            import shutil
                            shutil.copy2(report_tex, output_dir / "yolo_math_report.tex")
                            print(f"✅ [SERVER] Math TeX 결과 복사 완료")
                        
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
