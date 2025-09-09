from fastapi import APIRouter, UploadFile, File, HTTPException, Query
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

router = APIRouter(tags=["easy-convert"])
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "50"))

def _sanitize_filename(name: str) -> str:
    name = Path(name).stem
    name = name.strip()[:200]
    return re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name) or "doc"

def _extract_text_from_pdf(pdf_path: str) -> str:
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
                    t = page.get_text("blocks") or ""
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

@router.post("/convert")
async def convert_pdf(
    file: Optional[UploadFile] = File(None),
    doc_id: Optional[str] = Query(default=None, description="이전에 업로드된 raw PDF 파일명(예: 20250909_123456_myfile.pdf)")
):
    """
    두 모드 지원:
    1) 업로드 즉시 변환: file 업로드만 제공
    2) 업로드 후 나중에 변환: doc_id만 제공 (data/raw의 PDF를 찾아서 변환)
    """
    # 모델 상태 선확인
    if not easy_llm.health_check():
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다. /health 확인 요망")

    # ---- 모드 결정 ----
    content: Optional[bytes] = None
    original_filename = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if file and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > UPLOAD_MAX_MB:
            raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다(>{UPLOAD_MAX_MB}MB).")
        original_filename = file.filename
        safe_base = _sanitize_filename(original_filename)
        raw_pdf_name = f"{timestamp}_{safe_base}.pdf"
        raw_file_path = RAW_DIR / raw_pdf_name
        # 원본 저장
        try:
            with open(raw_file_path, "wb") as f:
                f.write(content)
            logger.info(f"[convert:file] 원본 PDF 저장: {raw_file_path}")
        except Exception as e:
            logger.error(f"원본 저장 실패: {e}")
            raise HTTPException(status_code=500, detail="원본 파일 저장에 실패했습니다.")
    elif doc_id:
        # 이전 업로드 파일 사용
        raw_file_path = RAW_DIR / doc_id
        if not raw_file_path.exists():
            raise HTTPException(status_code=404, detail="doc_id에 해당하는 원본 PDF가 존재하지 않습니다.")
        original_filename = doc_id
    else:
        raise HTTPException(status_code=400, detail="file 또는 doc_id 중 하나는 반드시 제공해야 합니다.")

    # ---- 텍스트 추출 ----
    # 파일 모드에선 content에서 임시파일. doc_id 모드에선 raw_file_path 직접 사용.
    if content is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        pdf_path_to_use = tmp_path
    else:
        pdf_path_to_use = str(raw_file_path)

    try:
        extracted = _extract_text_from_pdf(pdf_path_to_use)
        if not extracted.strip():
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다(스캔본 가능성).")
    finally:
        if content is not None and os.path.exists(pdf_path_to_use):
            os.unlink(pdf_path_to_use)

    # ---- 모델 변환 ----
    logger.info("AI 모델로 JSON 변환 시작")
    easy_json = easy_llm.generate(extracted)
    if easy_json is None:
        raise HTTPException(status_code=500, detail="AI 모델 처리 중 오류가 발생했습니다.")

    easy_json.setdefault("metadata", {})
    easy_json["metadata"].update({
        "original_filename": original_filename,
        "processed_at": datetime.now().isoformat(),
        "file_size": len(content) if content is not None else None,
        "extracted_text_length": len(extracted),
        "doc_id": raw_file_path.name if 'raw_file_path' in locals() else None,
    })

    minimized = _minimize_easy_json(easy_json)
    safe_base_for_json = _sanitize_filename(original_filename or "document")
    json_file_path = OUTPUTS_DIR / f"{timestamp}_{safe_base_for_json}.json"
    try:
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(minimized, f, ensure_ascii=False, indent=2)
        logger.info(f"변환된 JSON 저장: {json_file_path}")
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        raise HTTPException(status_code=500, detail="결과 파일 저장에 실패했습니다.")

    return {
        "status": "success",
        "doc_id": (raw_file_path.name if 'raw_file_path' in locals() else None),
        "raw_file_path": (str(raw_file_path) if 'raw_file_path' in locals() else None),
        "json_file_path": str(json_file_path),
        "extracted_text_length": len(extracted),
        "extracted_text_preview": (extracted[:500] + "...") if len(extracted) > 500 else extracted,
        "easy_json": minimized,
    }


@router.get("/model-status")
async def get_model_status():
    """AI 모델 서비스 상태"""
    is_healthy = easy_llm.health_check()
    model_info = easy_llm.get_model_info() if is_healthy else None
    return {
        "model_available": is_healthy,
        "model_info": model_info,
        "status": "healthy" if is_healthy else "unavailable",
    }
