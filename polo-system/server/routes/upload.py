# server/routes/upload.py
from __future__ import annotations

import os, re, unicodedata, time
from datetime import datetime
import httpx
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import tempfile
import shutil
from pathlib import Path
import json

from services.database.db import DB
from services import arxiv_client, preprocess_client

router = APIRouter()

# === TeX 섹션 파싱/매핑 유틸 ===
SECTION_PATTERNS = [
    (re.compile(r"^\\section\{(?P<title>.+?)\}"), "section"),
    (re.compile(r"^\\subsection\{(?P<title>.+?)\}"), "subsection"),
    (re.compile(r"^\\subsubsection\{(?P<title>.+?)\}"), "subsubsection"),
]

def _normalize_title(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\\\w+", "", t)  # TeX commands 제거
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()

def _build_tex_section_ranges(tex_path: Path) -> list[dict]:
    ranges: list[dict] = []
    try:
        lines = tex_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ranges
    for idx, raw in enumerate(lines, start=1):
        for rx, typ in SECTION_PATTERNS:
            m = rx.match(raw.strip())
            if m:
                title = m.group("title")
                ranges.append({
                    "title": _normalize_title(title),
                    "type": typ,
                    "start_line": idx,
                    "end_line": None,
                })
                break
    # end lines
    if not ranges:
        return ranges
    for i in range(len(ranges) - 1):
        ranges[i]["end_line"] = ranges[i+1]["start_line"] - 1
    ranges[-1]["end_line"] = len(lines)
    return ranges

def _map_easy_sections_to_tex(easy_data: dict, tex_ranges: list[dict]) -> dict[str, dict]:
    """Return mapping: easy_section_id -> tex_range dict with start/end."""
    mapped: dict[str, dict] = {}
    if not easy_data:
        return mapped
    used = set()
    for section in easy_data.get("easy_sections", []):
        sid = section.get("easy_section_id")
        title = _normalize_title(section.get("easy_section_title", ""))
        level = section.get("easy_section_level", 1)
        typ = "section" if level == 1 else "subsection"
        # exact match by title+type
        best = None
        for i, r in enumerate(tex_ranges):
            if i in used:
                continue
            if r["type"] == typ and (_normalize_title(r.get("title")) == title or title in _normalize_title(r.get("title"))):
                best = (i, r)
                break
        # fallback by order
        if best is None:
            for i, r in enumerate(tex_ranges):
                if i in used:
                    continue
                if (typ == r["type"]):
                    best = (i, r)
                    break
        if best:
            used.add(best[0])
            mapped[sid] = {
                "start_line": best[1]["start_line"],
                "end_line": best[1]["end_line"],
            }
    return mapped

# === Viz 헬퍼 함수들 ===
def _should_create_visualization(text: str) -> bool:
    """
    실제 Viz 트리거 시스템을 사용하여 시각화가 필요한지 판단
    """
    if not text or len(text.strip()) < 50:  # 너무 짧은 텍스트
        return False
    
    try:
        # Viz 모듈의 실제 트리거 시스템 사용
        import sys
        from pathlib import Path
        
        # Viz 모듈 경로 추가
        viz_dir = Path(__file__).parent.parent.parent / "viz"
        if str(viz_dir) not in sys.path:
            sys.path.insert(0, str(viz_dir))
        
        # 현재 작업 디렉토리를 viz 디렉토리로 변경 (상대 경로 문제 해결)
        original_cwd = Path.cwd()
        try:
            os.chdir(viz_dir)
            
            from text_to_spec import auto_build_spec_from_text
            # 실제 Viz 트리거로 스펙 생성 시도
            spec = auto_build_spec_from_text(text)
            print(f"🔍 [VIZ] 트리거 분석 결과: {len(spec)}개 스펙 생성")
            
            # 스펙이 생성되면 시각화 필요
            return len(spec) > 0
            
        except ImportError as e:
            print(f"⚠️ [VIZ] text_to_spec 모듈을 찾을 수 없습니다: {e}")
            return False
        except Exception as e:
            print(f"⚠️ [VIZ] 트리거 분석 실패: {e}")
            return False
        finally:
            # 원래 작업 디렉토리로 복원
            os.chdir(original_cwd)
        
    except Exception as e:
        print(f"⚠️ [VIZ] 트리거 확인 실패: {e}")
        # 실패 시 기본 조건으로 폴백
        has_numbers = len(re.findall(r'\d+', text)) >= 3
        has_math = len(re.findall(r'[+\-*/=<>]', text)) >= 2
        has_keywords = any(keyword in text.lower() for keyword in [
            'chart', 'graph', 'plot', 'table', 'figure', 'diagram',
            'visualization', '시각화', '그래프', '차트', '표', '도표'
        ])
        return has_numbers or has_math or has_keywords

async def _generate_paragraph_visualization(
    paper_id: str, 
    section_id: str, 
    paragraph_id: str, 
    text: str, 
    output_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    개별 문단에 대한 시각화 생성
    """
    try:
        viz_url = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
        
        # 고유한 파일명 생성 (중복 방지)
        timestamp = int(time.time() * 1000)
        safe_paragraph_id = re.sub(r'[^\w\-_]', '_', paragraph_id)
        unique_id = f"{safe_paragraph_id}_{timestamp}"
        
        async with httpx.AsyncClient(timeout=30) as client:
            # 실제 Viz API 엔드포인트 사용
            response = await client.post(f"{viz_url}/viz", json={
                "paper_id": paper_id,
                "index": -1,  # 전체 처리
                "rewritten_text": text,
                "target_lang": "ko",
                "bilingual": "missing"
            })
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # 이미지 원본 경로 또는 base64
                    src_path_str = result.get("image_path", "")
                    if src_path_str:
                        src_path = Path(src_path_str)
                        # 타깃 경로: outputs/{paper_id}/viz/{section_id}/{paragraph_id}/{filename}
                        try:
                            dest_dir = output_dir / "viz" / section_id / paragraph_id
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest_path = dest_dir / (src_path.name or f"{unique_id}.png")
                            # 원본이 존재하면 복사, 없으면 스킵
                            try:
                                if src_path.exists():
                                    shutil.copy2(src_path, dest_path)
                                else:
                                    # viz/charts 폴더에서 대체 소스 시도
                                    charts_candidate = (Path(__file__).resolve().parent.parent.parent / "viz" / "charts" / src_path.name)
                                    if charts_candidate.exists():
                                        shutil.copy2(charts_candidate, dest_path)
                            except Exception as ce:
                                print(f"⚠️ [VIZ] 원본 이미지 복사 실패(무시): {ce}")
                            # outputs/{paper_id} 기준 상대 경로로 저장
                            rel_path = Path("viz") / section_id / paragraph_id / dest_path.name
                            return {
                                "success": True,
                                "image_path": str(rel_path).replace("\\", "/"),
                                "viz_type": "auto_generated",
                                "created_at": datetime.now().isoformat()
                            }
                        except Exception as e:
                            print(f"⚠️ [VIZ] 대상 경로 저장 실패: {e}")
                            # 실패 시 outputs/{paper_id} 기준 파일명만 반환
                            fallback_rel = Path("viz") / section_id / paragraph_id / (src_path.name or f"{unique_id}.png")
                            return {
                                "success": True,
                                "image_path": str(fallback_rel).replace("\\", "/"),
                                "viz_type": "auto_generated",
                                "created_at": datetime.now().isoformat()
                            }
                    # base64 이미지가 제공되는 경우 저장
                    b64 = result.get("image_base64") or result.get("image_bytes")
                    if b64:
                        try:
                            import base64
                            img_bytes = base64.b64decode(b64)
                            dest_dir = output_dir / "viz" / section_id / paragraph_id
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest_path = dest_dir / f"{unique_id}.png"
                            with open(dest_path, "wb") as wf:
                                wf.write(img_bytes)
                            rel_path = Path("viz") / section_id / paragraph_id / dest_path.name
                            return {
                                "success": True,
                                "image_path": str(rel_path).replace("\\", "/"),
                                "viz_type": "auto_generated",
                                "created_at": datetime.now().isoformat()
                            }
                        except Exception as e:
                            print(f"⚠️ [VIZ] base64 이미지 저장 실패: {e}")
            
            return None
            
    except Exception as e:
        print(f"❌ [VIZ] 문단 시각화 생성 실패: {e}")
        return None
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
                
                # Viz 완료 대기
                print(f"⏳ [SERVER] Viz 모델 완료 대기 중...")
                max_wait_time = 1800  # 30분
                wait_interval = 5     # 5초마다 확인
                waited_time = 0
                
                while waited_time < max_wait_time:
                    viz_complete_flag = output_dir / ".viz_complete"
                    viz_error_flag = output_dir / ".viz_error"
                    
                    if viz_error_flag.exists():
                        error_msg = viz_error_flag.read_text(encoding="utf-8")
                        print(f"❌ [SERVER] Viz 모델 에러로 인해 Math 모델 실행 중단: {error_msg}")
                        return
                    
                    if viz_complete_flag.exists():
                        print(f"✅ [SERVER] Viz 모델 완료 확인됨")
                        break
                    
                    await asyncio.sleep(wait_interval)
                    waited_time += wait_interval
                    print(f"⏳ [SERVER] Viz 대기 중... ({waited_time}s/{max_wait_time}s)")
                
                if waited_time >= max_wait_time:
                    print(f"⚠️ [SERVER] Viz 모델 대기 타임아웃, Math 모델 계속 실행")
                
                # Easy 결과 파일 확인 (Math는 Easy 결과를 기반으로 처리)
                easy_output_dir = server_dir / "data" / "outputs" / paper_id
                easy_json_path = easy_output_dir / "easy_results.json"
                
                if not easy_json_path.exists():
                    print(f"❌ [SERVER] Easy 결과 파일 없음: {easy_json_path}")
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
                    
                    # Math 완료 마커 파일 생성
                    math_complete_flag = output_dir / ".math_complete"
                    math_complete_flag.write_text("completed", encoding="utf-8")
                    print(f"✅ [SERVER] Math 모델 처리 완료 마커 생성")
                    
                    # Math 결과를 올바른 형식으로 변환
                    if "math_equations" in math_result:
                        converted_equations = []
                        for i, equation in enumerate(math_result["math_equations"]):
                            converted_equation = {
                                "math_equation_id": f"math_equation_{i+1}",
                                "math_equation_index": f"({i+1})",
                                "math_equation_latex": equation.get("equation", ""),
                                "math_equation_explanation": equation.get("explanation", ""),
                                "math_equation_context": f"수식 {i+1}",
                                "math_equation_section_ref": equation.get("math_equation_section_ref", "easy_section_1"),
                                "math_equation_type": equation.get("kind", "inline"),
                                "math_equation_env": equation.get("env", ""),
                                "math_equation_line_start": equation.get("line_start", 0),
                                "math_equation_line_end": equation.get("line_end", 0),
                                "math_equation_variables": [],
                                "math_equation_importance": "medium",
                                "math_equation_difficulty": "intermediate"
                            }
                            converted_equations.append(converted_equation)
                        
                        math_result["math_equations"] = converted_equations
                        print(f"✅ [SERVER] Math 결과 변환 완료: {len(converted_equations)}개 수식")
                    
                    # Easy 결과와 Math 수식 통합
                    try:
                        easy_json_path = output_dir / "easy_results.json"
                        if easy_json_path.exists():
                            with open(easy_json_path, 'r', encoding='utf-8') as f:
                                easy_data = json.load(f)
                            
                            # Math 수식을 Easy 문단에 통합
                            integrated_sections = []
                            for section in easy_data.get("easy_sections", []):
                                section_id = section.get("easy_section_id", "")
                                
                                # 해당 섹션의 Math 수식들 찾기
                                section_equations = [
                                    eq for eq in math_result.get("math_equations", [])
                                    if eq.get("math_equation_section_ref") == section_id
                                ]
                                
                                # 문단별로 수식 통합
                                integrated_paragraphs = []
                                for paragraph in section.get("easy_paragraphs", []):
                                    # 문단 추가
                                    integrated_paragraphs.append(paragraph)
                                    
                                    # 해당 문단 다음에 수식들 추가
                                    for equation in section_equations:
                                        # 수식을 문단 형태로 변환
                                        equation_paragraph = {
                                            "easy_paragraph_id": f"math_{equation.get('math_equation_id', '')}",
                                            "easy_paragraph_text": f"**{equation.get('math_equation_context', '수식')}**",
                                            "math_equation": {
                                                "equation_id": equation.get("math_equation_id", ""),
                                                "equation_index": equation.get("math_equation_index", ""),
                                                "equation_latex": equation.get("math_equation_latex", ""),
                                                "equation_explanation": equation.get("math_equation_explanation", ""),
                                                "equation_context": equation.get("math_equation_context", ""),
                                                "equation_variables": equation.get("math_equation_variables", []),
                                                "equation_importance": equation.get("math_equation_importance", "medium"),
                                                "equation_difficulty": equation.get("math_equation_difficulty", "intermediate")
                                            },
                                            "paragraph_type": "math_equation"
                                        }
                                        integrated_paragraphs.append(equation_paragraph)
                                
                                # 통합된 섹션 생성
                                integrated_section = {
                                    **section,
                                    "easy_paragraphs": integrated_paragraphs
                                }
                                integrated_sections.append(integrated_section)
                            
                            # 통합된 Easy 결과 저장
                            easy_data["easy_sections"] = integrated_sections
                            with open(easy_json_path, 'w', encoding='utf-8') as f:
                                json.dump(easy_data, f, ensure_ascii=False, indent=2)
                            
                            print(f"✅ [SERVER] Easy 결과에 Math 수식 통합 완료")
                    except Exception as e:
                        print(f"⚠️ [SERVER] Math 수식 통합 실패: {e}")
                    
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
        # Math 에러 마커 파일 생성
        try:
            math_error_flag = output_dir / ".math_error"
            math_error_flag.write_text(str(e), encoding="utf-8")
        except Exception as flag_error:
            print(f"❌ [ERROR] Math 에러 마커 파일 생성 실패: {flag_error}")
        raise HTTPException(status_code=500, detail=f"Math 모델 처리 실패: {e}")

@router.post("/upload/run-all-models")
async def run_all_models_sequentially(request: ModelSendRequest, bg: BackgroundTasks):
    """
    Easy → Math → Viz 순차 실행 (완전한 파이프라인)
    """
    try:
        paper_id = request.paper_id
        print(f"🚀 [SERVER] 전체 모델 순차 실행 시작: paper_id={paper_id}")
        
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        output_dir = server_dir / "data" / "outputs" / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1단계: Easy 모델 실행
        print(f"📝 [PIPELINE] 1단계: Easy 모델 실행")
        easy_url = os.getenv("EASY_MODEL_URL", "http://localhost:5003")
        tex_path = server_dir / "data" / "out" / "source" / "merged_body.tex"
        
        if not tex_path.exists():
            raise HTTPException(status_code=404, detail="merged_body.tex 파일을 찾을 수 없습니다")
        
        async with httpx.AsyncClient(timeout=1200) as client:
            easy_response = await client.post(f"{easy_url}/from-transport", json={
                "paper_id": paper_id,
                "transport_path": str(tex_path),
                "output_dir": str(output_dir)  # paper_id 디렉토리 포함
            })
            
            if easy_response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Easy 모델 실행 실패: {easy_response.text}")
            
            print(f"✅ [PIPELINE] Easy 모델 전송 완료")
            
            # Easy 결과 파일 생성 대기 (여러 가능한 경로 확인)
            possible_paths = [
                output_dir / "easy_results.json",  # 직접 경로
                output_dir / paper_id / "easy_results.json",  # Easy 모델이 paper_id 추가
                output_dir.parent / paper_id / "easy_results.json"  # Easy 모델이 상위에 생성
            ]
            
            easy_json_path = None
            print(f"⏳ [PIPELINE] Easy 결과 파일 대기 중...")
            for path in possible_paths:
                print(f"  - 확인 중: {path}")
            
            max_wait_time = 300  # 5분
            wait_interval = 2    # 2초마다 확인
            waited_time = 0
            
            while waited_time < max_wait_time:
                for path in possible_paths:
                    if path.exists():
                        easy_json_path = path
                        print(f"✅ [PIPELINE] Easy 결과 파일 발견: {easy_json_path}")
                        break
                
                if easy_json_path:
                    break
                
                await asyncio.sleep(wait_interval)
                waited_time += wait_interval
                print(f"⏳ [PIPELINE] Easy 결과 대기 중... ({waited_time}s/{max_wait_time}s)")
            
            if not easy_json_path:
                raise HTTPException(status_code=500, detail="Easy 모델 결과 파일 생성 실패")
            
            # Easy 결과를 올바른 위치로 복사
            target_easy_path = output_dir / "easy_results.json"
            if easy_json_path != target_easy_path:
                import shutil
                shutil.copy2(easy_json_path, target_easy_path)
                print(f"✅ [PIPELINE] Easy 결과 파일 복사: {easy_json_path} → {target_easy_path}")
            
            # 섹션 개수 제한(최대 16개) - 캐시 결과가 17개인 경우 트림
            try:
                with open(target_easy_path, 'r', encoding='utf-8') as f:
                    _easy = json.load(f)
                sections = _easy.get("easy_sections", [])
                if isinstance(sections, list) and len(sections) > 16:
                    _easy["easy_sections"] = sections[:16]
                    if isinstance(_easy.get("paper_info"), dict):
                        _easy["paper_info"]["total_sections"] = 16
                    with open(target_easy_path, 'w', encoding='utf-8') as f:
                        json.dump(_easy, f, ensure_ascii=False, indent=2)
                    print(f"✂️ [PIPELINE] Easy 섹션을 16개로 트림: 기존 {len(sections)} → 16")
            except Exception as e:
                print(f"⚠️ [PIPELINE] Easy 섹션 트림 실패(무시): {e}")

            print(f"✅ [PIPELINE] Easy 모델 완료")
        
        # 2단계: Viz 모델 실행 (easy_results.json 기반)
        print(f"🎨 [PIPELINE] 2단계: Viz 모델 실행")
        viz_url = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
        
        # Easy 결과 로드 (이미 위에서 확인됨)
        easy_json_path = output_dir / "easy_results.json"
        with open(easy_json_path, 'r', encoding='utf-8') as f:
            easy_data = json.load(f)
        
        # Viz 처리 (중복 방지: 기존 visualization이 있으면 스킵)
        updated_sections = []
        for section in easy_data.get("easy_sections", []):
            updated_paragraphs = []
            for paragraph in section.get("easy_paragraphs", []):
                paragraph_text = paragraph.get("easy_paragraph_text", "")
                needs_visualization = _should_create_visualization(paragraph_text)
                
                # 이미 시각화가 있고 파일도 존재하면 스킵
                existing = paragraph.get("visualization")
                if existing:
                    try:
                        existing_path = existing.get("image_path")
                        if existing_path:
                            candidate = (output_dir / existing_path).resolve()
                            if candidate.exists():
                                needs_visualization = False
                    except Exception:
                        pass

                if needs_visualization:
                    try:
                        viz_result = await _generate_paragraph_visualization(
                            paper_id, section.get("easy_section_id", ""), 
                            paragraph.get("easy_paragraph_id", ""), 
                            paragraph_text, output_dir
                        )
                        
                        if viz_result and viz_result.get("success"):
                            paragraph["visualization"] = {
                                "image_path": viz_result["image_path"],
                                "viz_type": viz_result.get("viz_type", "chart"),
                                "created_at": viz_result.get("created_at")
                            }
                            print(f"✅ [PIPELINE] 문단 시각화 생성: {viz_result['image_path']}")
                    except Exception as e:
                        print(f"❌ [PIPELINE] 문단 시각화 오류: {e}")
                
                updated_paragraphs.append(paragraph)
            
            section["easy_paragraphs"] = updated_paragraphs
            updated_sections.append(section)
        
        # 업데이트된 Easy 결과 저장
        easy_data["easy_sections"] = updated_sections
        with open(easy_json_path, 'w', encoding='utf-8') as f:
            json.dump(easy_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ [PIPELINE] Viz 모델 완료")

        # 3단계: Math 모델 실행
        print(f"🔢 [PIPELINE] 3단계: Math 모델 실행")
        math_url = os.getenv("MATH_MODEL_URL", "http://localhost:5004")
        
        # Math 모델에 전달할 Easy 결과 준비 (섹션 정보 포함)
        math_easy_data = {
            "paper_info": easy_data.get("paper_info", {}),
            "easy_sections": easy_data.get("easy_sections", []),
            "sections_mapping": {
                section.get("easy_section_id", ""): {
                    "title": section.get("easy_section_title", ""),
                    "order": section.get("easy_section_order", 0),
                    "level": section.get("easy_section_level", 1)
                }
                for section in easy_data.get("easy_sections", [])
            }
        }
        
        async with httpx.AsyncClient(timeout=1800) as client:
            math_response = await client.post(f"{math_url}/math-with-easy", json={
                "path": str(tex_path),
                "easy_results": math_easy_data,
                "paper_id": paper_id,
                "output_dir": str(output_dir)  # Math 모델도 올바른 경로 지정
            })
            
            if math_response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Math 모델 실행 실패: {math_response.text}")
            
            math_result = math_response.json()

            # 한국어 설명 우선 적용을 위해 .ko.json 병합 시도
            try:
                # 우선순위: outputs/{paper_id}/math_outputs > models/math/_build > models/math/_build_yolo
                ko_candidates = [
                    output_dir / "math_outputs" / "equations_explained.ko.json",
                    server_dir.parent / "models" / "math" / "_build" / "equations_explained.ko.json",
                    server_dir.parent / "models" / "math" / "_build_yolo" / "equations_explained.ko.json",
                ]
                ko_path = next((p for p in ko_candidates if p.exists()), None)
                ko_items_by_key: Dict[str, Any] = {}
                if ko_path:
                    with open(ko_path, 'r', encoding='utf-8') as f:
                        ko_doc = json.load(f)
                    for it in ko_doc.get("items", []):
                        # 매칭 키: (line_start,line_end,env,kind) 또는 index 기반
                        key = (
                            it.get("line_start", -1),
                            it.get("line_end", -1),
                            it.get("env", ""),
                            it.get("kind", "")
                        )
                        ko_items_by_key[str(key)] = it
                        if "index" in it:
                            ko_items_by_key[f"idx:{it['index']}"] = it
                # math_result에 한국어 설명/변수 병합
                if "math_equations" in math_result and ko_items_by_key:
                    for i, eq in enumerate(math_result["math_equations"]):
                        key = (
                            eq.get("line_start", -1),
                            eq.get("line_end", -1),
                            eq.get("env", ""),
                            eq.get("kind", "")
                        )
                        ko_it = ko_items_by_key.get(str(key)) or ko_items_by_key.get(f"idx:{i+1}")
                        if ko_it:
                            # 한국어 설명 우선
                            if ko_it.get("explanation"):
                                eq["explanation"] = ko_it.get("explanation")
                            # 변수 목록 병합 (가능한 키들을 탐색)
                            vars_candidate = (
                                ko_it.get("variables")
                                or ko_it.get("variable_list")
                                or ko_it.get("symbols")
                                or []
                            )
                            if vars_candidate and not eq.get("equation_variables"):
                                eq["equation_variables"] = vars_candidate
            except Exception as e:
                print(f"⚠️ [PIPELINE] 한국어 Math 결과 병합 실패: {e}")
            
            # Math 결과를 올바른 형식으로 변환
            if "math_equations" in math_result:
                sections_list = list(easy_data.get("easy_sections", []))
                converted_equations = []
                
                for i, equation in enumerate(math_result["math_equations"]):
                    # Math 모델의 JSON 구조를 서버 형식으로 변환
                    converted_equation = {
                        "math_equation_id": f"math_equation_{i+1}",
                        "math_equation_index": f"({i+1})",
                        "math_equation_latex": equation.get("equation", ""),
                        "math_equation_explanation": equation.get("explanation", ""),
                        "math_equation_context": f"수식 {i+1}",
                        "math_equation_section_ref": equation.get("math_equation_section_ref", "easy_section_1"),
                        "math_equation_type": equation.get("kind", "inline"),
                        "math_equation_env": equation.get("env", ""),
                        "math_equation_line_start": equation.get("line_start", 0),
                        "math_equation_line_end": equation.get("line_end", 0),
                        # Math 모델이 제공한 변수 정보 사용 (fallback 빈 배열)
                        "math_equation_variables": equation.get("equation_variables", []) or equation.get("variables", []) or [],
                        "math_equation_importance": "medium",
                        "math_equation_difficulty": "intermediate"
                    }
                    
                    # 섹션 참조 수정
                    current_ref = converted_equation["math_equation_section_ref"]
                    if not any(section.get("easy_section_id") == current_ref for section in sections_list):
                        # 잘못된 참조인 경우, 라인 번호 기반으로 올바른 섹션 찾기
                        line_start = equation.get("line_start", 0)
                        best_section = None
                        
                        # 라인 번호가 가장 가까운 섹션 찾기
                        for section in sections_list:
                            section_order = section.get("easy_section_order", 0)
                            if section_order > 0 and section_order <= line_start:
                                best_section = section
                        
                        if best_section:
                            converted_equation["math_equation_section_ref"] = best_section.get("easy_section_id", "easy_section_1")
                            print(f"🔧 [PIPELINE] 수식 섹션 참조 수정: {current_ref} → {converted_equation['math_equation_section_ref']}")
                        else:
                            converted_equation["math_equation_section_ref"] = "easy_section_1"
                            print(f"⚠️ [PIPELINE] 수식 섹션 참조를 기본값으로 설정: {current_ref} → easy_section_1")
                    
                    converted_equations.append(converted_equation)
                
                # 변환된 수식으로 교체
                math_result["math_equations"] = converted_equations
                print(f"✅ [PIPELINE] Math 결과 변환 완료: {len(converted_equations)}개 수식")
            
            # Math 결과 저장
            math_json_file = output_dir / "math_results.json"
            math_json_file.write_text(json.dumps(math_result, ensure_ascii=False, indent=2), encoding="utf-8")
            
            print(f"✅ [PIPELINE] Math 모델 완료")
        
        # 4단계: 통합 결과 생성 (Easy 문단 + Math 수식 통합)
        print(f"🔗 [PIPELINE] 4단계: 통합 결과 생성")
        
        # Math 결과 로드
        with open(math_json_file, 'r', encoding='utf-8') as f:
            math_data = json.load(f)
        
        # Math 수식을 Easy 문단에 통합 (문단 기준 삽입: paragraph_id 매칭 우선, 섹션 기준은 말미에 1회)
        integrated_sections = []
        for section in easy_data.get("easy_sections", []):
            section_id = section.get("easy_section_id", "")
            
            # 해당 섹션의 Math 수식들 찾기
            section_equations = [
                eq for eq in math_data.get("math_equations", [])
                if eq.get("math_equation_section_ref") == section_id or str(eq.get("math_equation_section_ref", "")).startswith("easy_paragraph_")
            ]
            
            # 문단별로 수식 통합
            integrated_paragraphs = []
            appended_ids = set()
            for paragraph in section.get("easy_paragraphs", []):
                paragraph_id = paragraph.get("easy_paragraph_id", "")
                
                # 문단 추가
                integrated_paragraphs.append(paragraph)
                
                # 해당 문단 다음에, 참조가 이 문단을 가리키는 수식만 추가
                for equation in section_equations:
                    if str(equation.get("math_equation_section_ref", "")) != paragraph_id:
                        continue
                    if equation.get("math_equation_id") in appended_ids:
                        continue
                    # 수식을 문단 형태로 변환
                    equation_paragraph = {
                        "easy_paragraph_id": f"math_{equation.get('math_equation_id', '')}",
                        "easy_paragraph_text": f"**{equation.get('math_equation_context', '수식')}**",
                        "math_equation": {
                            "equation_id": equation.get("math_equation_id", ""),
                            "equation_index": equation.get("math_equation_index", ""),
                            "equation_latex": equation.get("math_equation_latex", ""),
                            "equation_explanation": equation.get("math_equation_explanation", ""),
                            "equation_context": equation.get("math_equation_context", ""),
                            "equation_variables": equation.get("math_equation_variables", []),
                            "equation_importance": equation.get("math_equation_importance", "medium"),
                            "equation_difficulty": equation.get("math_equation_difficulty", "intermediate")
                        },
                        "paragraph_type": "math_equation"
                    }
                    integrated_paragraphs.append(equation_paragraph)
                    appended_ids.add(equation.get("math_equation_id"))

            # 섹션 참조만 가진(문단 참조가 아닌) 수식은 섹션 내 첫 문단 뒤에 1회만 추가(가독성)
            for equation in section_equations:
                ref = str(equation.get("math_equation_section_ref", ""))
                if ref != section_id:
                    continue
                if equation.get("math_equation_id") in appended_ids:
                    continue
                equation_paragraph = {
                    "easy_paragraph_id": f"math_{equation.get('math_equation_id', '')}",
                    "easy_paragraph_text": f"**{equation.get('math_equation_context', '수식')}**",
                    "math_equation": {
                        "equation_id": equation.get("math_equation_id", ""),
                        "equation_index": equation.get("math_equation_index", ""),
                        "equation_latex": equation.get("math_equation_latex", ""),
                        "equation_explanation": equation.get("math_equation_explanation", ""),
                        "equation_context": equation.get("math_equation_context", ""),
                        "equation_variables": equation.get("math_equation_variables", []),
                        "equation_importance": equation.get("math_equation_importance", "medium"),
                        "equation_difficulty": equation.get("math_equation_difficulty", "intermediate")
                    },
                    "paragraph_type": "math_equation"
                }
                # 첫 문단 뒤 삽입
                if integrated_paragraphs:
                    integrated_paragraphs.insert(1, equation_paragraph)
                else:
                    integrated_paragraphs.append(equation_paragraph)
                appended_ids.add(equation.get("math_equation_id"))
            
            # 통합된 섹션 생성
            integrated_section = {
                **section,
                "easy_paragraphs": integrated_paragraphs
            }
            integrated_sections.append(integrated_section)
        
        # 통합 결과 생성
        integrated_result = {
            "paper_info": easy_data.get("paper_info", {}),
            "easy_sections": integrated_sections,
            "math_equations": math_data.get("math_equations", []),
            "processing_status": "completed",
            "completed_at": datetime.now().isoformat()
        }
        
        # 통합 결과 저장
        integrated_file = output_dir / "integrated_result.json"
        integrated_file.write_text(json.dumps(integrated_result, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print(f"✅ [PIPELINE] 전체 파이프라인 완료")
        
        return {
            "message": "전체 모델 파이프라인 완료",
            "paper_id": paper_id,
            "status": "completed",
            "integrated_file": str(integrated_file),
            "ready_for_result": True
        }
        
    except Exception as e:
        print(f"❌ [PIPELINE] 전체 파이프라인 실패: {e}")
        raise HTTPException(status_code=500, detail=f"전체 파이프라인 실패: {e}")

@router.get("/upload/status/{paper_id}")
async def get_processing_status(paper_id: str):
    """
    모델 처리 상태 확인 (Result.tsx 이동 버튼용)
    """
    try:
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        output_dir = server_dir / "data" / "outputs" / paper_id
        
        # 각 모델의 완료 상태 확인
        easy_complete = (output_dir / "easy_results.json").exists()
        viz_complete = (output_dir / ".viz_complete").exists()
        math_complete = (output_dir / ".math_complete").exists()
        integrated_complete = (output_dir / "integrated_result.json").exists()
        
        # 에러 상태 확인
        viz_error = (output_dir / ".viz_error").exists()
        math_error = (output_dir / ".math_error").exists()
        
        status = "processing"
        if integrated_complete:
            status = "completed"
        elif math_complete:
            status = "math_completed"
        elif viz_complete:
            status = "viz_completed"
        elif easy_complete:
            status = "easy_completed"
        
        # 에러가 있으면 실패 상태
        if viz_error or math_error:
            status = "error"
        
        # Result.tsx 이동 가능 여부
        ready_for_result = integrated_complete or (easy_complete and viz_complete and math_complete)
        
        return {
            "paper_id": paper_id,
            "status": status,
            "ready_for_result": ready_for_result,
            "models": {
                "easy": {
                    "completed": easy_complete,
                    "error": False
                },
                "viz": {
                    "completed": viz_complete,
                    "error": viz_error
                },
                "math": {
                    "completed": math_complete,
                    "error": math_error
                }
            },
            "integrated_result": integrated_complete
        }
        
    except Exception as e:
        print(f"❌ [STATUS] 상태 확인 실패: {e}")
        return {
            "paper_id": paper_id,
            "status": "error",
            "ready_for_result": False,
            "error": str(e)
        }

@router.post("/upload/send-to-viz")
async def send_to_viz(request: ModelSendRequest, bg: BackgroundTasks):
    """
    Viz 모델로 Easy 결과 전송 및 시각화 생성
    """
    try:
        paper_id = request.paper_id
        print(f"🚀 [SERVER] Viz 모델 처리 요청: paper_id={paper_id}")
        
        # Easy 결과 파일 경로 찾기 (실제 경로 확인됨)
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        easy_json_path = server_dir / "data" / "outputs" / paper_id / "easy_results.json"
        
        print(f"🔍 [DEBUG] Easy 결과 파일 경로: {easy_json_path}")
        print(f"🔍 [DEBUG] 파일 존재: {easy_json_path.exists()}")
        
        if not easy_json_path.exists():
            raise HTTPException(status_code=404, detail="Easy 결과를 찾을 수 없습니다")
        
        # Easy 결과 로드
        with open(easy_json_path, 'r', encoding='utf-8') as f:
            easy_data = json.load(f)
        
        # Viz 이미지 생성 및 Easy 결과에 통합 (실제 경로 사용)
        viz_output_dir = server_dir / "data" / "outputs" / paper_id
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 [DEBUG] Viz 출력 디렉토리: {viz_output_dir}")
        
        # Viz 모델 백그라운드 처리
        async def _run_viz_processing():
            try:
                print(f"🔄 [SERVER] Viz 모델 백그라운드 작업 시작...")
                
                # 각 섹션의 문단에 대해 시각화 생성
                updated_sections = []
                for section in easy_data.get("easy_sections", []):
                    updated_paragraphs = []
                    for paragraph in section.get("easy_paragraphs", []):
                        # 시각화 트리거 확인
                        paragraph_text = paragraph.get("easy_paragraph_text", "")
                        needs_visualization = _should_create_visualization(paragraph_text)
                        
                        # 시각화가 필요하지 않으면 건너뛰기
                        if not needs_visualization:
                            print(f"⏭️ [VIZ] 문단 건너뛰기 (트리거 없음): {paragraph.get('easy_paragraph_id', '')}")
                            updated_paragraphs.append(paragraph)
                            continue
                        
                        # 시각화 생성
                        try:
                            # Viz 모델로 이미지 생성
                            viz_result = await _generate_paragraph_visualization(
                                paper_id, section.get("easy_section_id", ""), 
                                paragraph.get("easy_paragraph_id", ""), 
                                paragraph_text, viz_output_dir
                            )
                            
                            if viz_result and viz_result.get("success"):
                                # 이미지 경로를 문단에 추가
                                paragraph["visualization"] = {
                                    "image_path": viz_result["image_path"],
                                    "viz_type": viz_result.get("viz_type", "chart"),
                                    "created_at": viz_result.get("created_at")
                                }
                                print(f"✅ [VIZ] 문단 시각화 생성: {viz_result['image_path']}")
                            else:
                                print(f"⚠️ [VIZ] 문단 시각화 실패: {paragraph.get('easy_paragraph_id', '')}")

                        except Exception as e:
                            print(f"❌ [VIZ] 문단 시각화 오류: {e}")

                        updated_paragraphs.append(paragraph)
                    
                    section["easy_paragraphs"] = updated_paragraphs
                    updated_sections.append(section)
                
                # 업데이트된 Easy 결과 저장
                easy_data["easy_sections"] = updated_sections
                with open(easy_json_path, 'w', encoding='utf-8') as f:
                    json.dump(easy_data, f, ensure_ascii=False, indent=2)
                
                # Viz 완료 마커 파일 생성
                viz_complete_flag = viz_output_dir / ".viz_complete"
                viz_complete_flag.write_text("completed", encoding="utf-8")
                
                print(f"✅ [VIZ] Easy 결과에 시각화 통합 완료")
                
            except Exception as e:
                print(f"❌ [VIZ] 백그라운드 처리 실패: {e}")
                # 에러 마커 파일 생성
                viz_error_flag = viz_output_dir / ".viz_error"
                viz_error_flag.write_text(str(e), encoding="utf-8")
        
        # 백그라운드에서 실행
        bg.add_task(_run_viz_processing)
        
        return {
            "message": "Viz 모델 처리 시작됨",
            "paper_id": paper_id,
            "status": "processing"
        }
        
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
        # Easy 결과 파일을 여러 위치에서 찾기 (직접 경로를 먼저 확인)
        easy_json_path = server_dir / "data" / "outputs" / paper_id / "easy_results.json"
        if not easy_json_path.exists():
            # easy_outputs 디렉토리에서 찾기 (백업)
            easy_json_path = server_dir / "data" / "outputs" / paper_id / "easy_outputs" / "easy_results.json"
        
        if not easy_json_path.exists():
            print(f"❌ [SERVER] Easy 결과 파일 없음:")
            print(f"  - 시도한 경로 1: {server_dir / 'data' / 'outputs' / paper_id / 'easy_results.json'}")
            print(f"  - 시도한 경로 2: {server_dir / 'data' / 'outputs' / paper_id / 'easy_outputs' / 'easy_results.json'}")
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

@router.get("/integrated-result/{paper_id}")
async def get_integrated_result(paper_id: str):
    """
    통합 결과 조회 (Easy + Math + 시각화)
    """
    try:
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent
        output_dir = server_dir / "data" / "outputs" / paper_id
        
        # 0) 이미 생성된 통합 결과 파일이 있으면 그대로 반환 (파이프라인 산출물 우선)
        integrated_path = output_dir / "integrated_result.json"
        if integrated_path.exists():
            try:
                with open(integrated_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ [INTEGRATED] 기존 통합 결과 로드 실패, 재구성 시도: {e}")

        # Easy 결과 로드 (실제 경로 확인됨)
        easy_data = None
        easy_error = None
        try:
            # 실제 경로: outputs/{paper_id}/easy_results.json
            easy_file = output_dir / "easy_results.json"
            
            if easy_file.exists():
                with open(easy_file, 'r', encoding='utf-8') as f:
                    easy_data = json.load(f)
            else:
                easy_error = "Easy 결과 파일이 없습니다"
        except Exception as e:
            easy_error = f"Easy 결과 로드 실패: {str(e)}"
        
        # Math 결과 로드 (우선순위: math_results.json > math_outputs/equations_explained.json)
        math_data = None
        math_error = None
        try:
            math_results_path = output_dir / "math_results.json"
            math_items_path = output_dir / "math_outputs" / "equations_explained.json"
            if math_results_path.exists():
                with open(math_results_path, 'r', encoding='utf-8') as f:
                    math_data = json.load(f)
            elif math_items_path.exists():
                with open(math_items_path, 'r', encoding='utf-8') as f:
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
        elif math_data and "math_equations" in math_data:
            # 이미 서버 형식으로 변환된 math_results.json인 경우 그대로 삽입
            integrated_data["math_equations"] = math_data.get("math_equations", [])
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
