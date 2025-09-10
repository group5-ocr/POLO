from fastapi import APIRouter, UploadFile, File, HTTPException
import fitz  # PyMuPDF
import os
import tempfile
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from services.llm_client import easy_llm
from services.file_manager import file_manager
from services.environment import env_manager

router = APIRouter(tags=["easy-upload"])
logger = logging.getLogger(__name__)

# ===== 경로/환경 =====
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # polo-system 루트
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "50"))

# ===== 유틸 =====
def _sanitize_filename(name: str) -> str:
    name = Path(name).stem  # 확장자 제거
    name = name.strip()[:200]
    return re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name) or "doc"

def _extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출. 기본 text, 보조 blocks 경로 지원."""
    try:
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted:
                try:
                    if not doc.authenticate(""):
                        raise RuntimeError("암호화된 PDF이며 열 수 없습니다.")
                except Exception:
                    raise RuntimeError("암호화된 PDF이며 열 수 없습니다.")
            parts = []
            for i, page in enumerate(doc):
                t = page.get_text("text") or ""
                if not t.strip():
                    # 스캔/비텍스트 PDF 보조 시도
                    t = page.get_text("blocks") or ""
                    # blocks는 튜플 목록일 수 있어 문자열로 변환
                    if isinstance(t, list):
                        t = "\n".join([b[4] for b in t if isinstance(b, (list, tuple)) and len(b) >= 5 and isinstance(b[4], str)])
                if t.strip():
                    parts.append(f"--- 페이지 {i+1} ---\n{t}")
            return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 실패: {e}")
        raise RuntimeError(f"PDF 텍스트 추출 실패: {e}")

def _minimize_easy_json(data: dict) -> dict:
    try:
        result = dict(data)
        for sec_name in ["abstract","introduction","methods","results","discussion","conclusion"]:
            sec = (result.get(sec_name) or {})
            if isinstance(sec, dict) and "original" in sec:
                sec.pop("original", None)
                result[sec_name] = sec
        return result
    except Exception as e:
        logger.warning(f"경량화 실패, 원본 유지: {e}")
        return data

# ===== 엔드포인트 =====
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF 업로드 → data/raw에 저장 → 바로 모델 변환까지 수행 후 data/outputs에 JSON 저장.
    프론트에서 '업로드 즉시 변환'을 원할 때 사용.
    환경에 따라 DB 또는 로컬 파일 시스템 사용.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 모델 상태 선확인(큰 파일 낭비 방지)
    if not easy_llm.health_check():
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다. /health 확인 요망")

    # 읽기
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > UPLOAD_MAX_MB:
        raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다(>{UPLOAD_MAX_MB}MB).")

    # 텍스트 추출(임시파일로)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"PDF 텍스트 추출 시작: {file.filename}")
        extracted = _extract_text_from_pdf(tmp_path)
        if not extracted.strip():
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다(스캔본 가능성).")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 모델 변환
    logger.info("AI 모델로 JSON 변환 시작")
    easy_json = easy_llm.generate(extracted)
    if easy_json is None:
        raise HTTPException(status_code=500, detail="AI 모델 처리 중 오류가 발생했습니다.")

    # 메타데이터 보강
    easy_json.setdefault("metadata", {})
    easy_json["metadata"].update({
        "original_filename": file.filename,
        "processed_at": datetime.now().isoformat(),
        "file_size": len(content),
        "extracted_text_length": len(extracted),
    })

    # 경량 저장
    minimized = _minimize_easy_json(easy_json)

    # 통합 파일 관리자 사용
    try:
        # 원본 파일 저장
        origin_result = file_manager.save_origin_file(
            filename=file.filename,
            content=content,
            user_id=None  # TODO: 사용자 인증 연동 시 user_id 전달
        )
        
        # Easy 변환 결과 저장
        easy_result = file_manager.save_easy_file(
            origin_filename=file.filename,
            easy_json=minimized,
            user_id=None  # TODO: 사용자 인증 연동 시 user_id 전달
        )
        
        logger.info(f"파일 저장 완료 - 원본: {origin_result['filename']}, Easy: {easy_result['filename']}")
        
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장에 실패했습니다: {e}")

    return {
        "status": "success",
        "doc_id": origin_result["filename"],
        "filename": file.filename,
        "raw_file_path": origin_result["file_path"],
        "json_file_path": easy_result["file_path"],
        "file_size": len(content),
        "extracted_text_length": len(extracted),
        "extracted_text_preview": (extracted[:500] + "...") if len(extracted) > 500 else extracted,
        "easy_json": minimized,
        "storage_info": env_manager.get_storage_info(),
    }


@router.get("/upload-status")
async def get_upload_status():
    """업로드/결과 파일 개요"""
    # 통합 파일 관리자 사용
    try:
        files = file_manager.get_file_list("all")
        origin_files = [f for f in files if f["file_type"] == "origin"]
        easy_files = [f for f in files if f["file_type"] == "easy"]
        math_files = [f for f in files if f["file_type"] == "math"]
        
        return {
            "raw_files_count": len(origin_files),
            "output_files_count": len(easy_files),
            "math_files_count": len(math_files),
            "raw_dir": str(file_manager.raw_dir),
            "outputs_dir": str(file_manager.outputs_dir),
            "storage_info": env_manager.get_storage_info(),
            "status": "ok",
        }
    except Exception as e:
        logger.error(f"파일 상태 조회 실패: {e}")
        return {
            "raw_files_count": 0,
            "output_files_count": 0,
            "math_files_count": 0,
            "raw_dir": str(file_manager.raw_dir),
            "outputs_dir": str(file_manager.outputs_dir),
            "storage_info": env_manager.get_storage_info(),
            "status": "error",
            "error": str(e)
        }
