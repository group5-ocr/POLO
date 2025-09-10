# server/routes/convert.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import fitz, tempfile, os, logging
from services.llm_client import easy_llm
from services.file_manager import file_manager
from services.environment import env_manager

router = APIRouter()
logger = logging.getLogger(__name__)

def _extract_text(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes); tmp_path = tmp.name
    try:
        with fitz.open(tmp_path) as doc:
            parts = []
            for i in range(len(doc)):
                t = doc[i].get_text("text")
                if t.strip():
                    parts.append(f"--- 페이지 {i+1} ---\n{t}")
        return "\n\n".join(parts)
    finally:
        try: os.unlink(tmp_path)
        except: pass

@router.post("/convert")
async def convert_pdf(
    file: Optional[UploadFile] = File(None),
    doc_id: Optional[str] = Query(default=None, description="이전에 업로드된 raw PDF 파일명(예: 20250909_123456_myfile.pdf)")
):
    """
    두 모드 지원:
    1) 업로드 즉시 변환: file 업로드만 제공
    2) 업로드 후 나중에 변환: doc_id만 제공 (data/raw의 PDF를 찾아서 변환)
    환경에 따라 DB 또는 로컬 파일 시스템 사용.
    """
    # 모델 상태 선확인
    if not easy_llm.health_check():
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다. /health 확인 요망")

    # ---- 모드 결정 ----
    content: Optional[bytes] = None
    original_filename = None
    raw_file_path = None

    if file and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > UPLOAD_MAX_MB:
            raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다(>{UPLOAD_MAX_MB}MB).")
        original_filename = file.filename
    elif doc_id:
        # 이전 업로드 파일 사용
        raw_file_path = file_manager.raw_dir / doc_id
        if not raw_file_path.exists():
            raise HTTPException(status_code=404, detail="doc_id에 해당하는 원본 PDF가 존재하지 않습니다.")
        original_filename = doc_id
    else:
        raise HTTPException(status_code=400, detail="file 또는 doc_id 중 하나는 반드시 제공해야 합니다.")

    # ---- 텍스트 추출 ----
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
    })

    minimized = _minimize_easy_json(easy_json)

    # 통합 파일 관리자 사용
    try:
        # 파일 업로드 모드인 경우 원본 파일도 저장
        if content is not None:
            origin_result = file_manager.save_origin_file(
                filename=original_filename,
                content=content,
                user_id=None  # TODO: 사용자 인증 연동 시 user_id 전달
            )
            raw_file_path = Path(origin_result["file_path"])
        
        # Easy 변환 결과 저장
        easy_result = file_manager.save_easy_file(
            origin_filename=original_filename,
            easy_json=minimized,
            user_id=None  # TODO: 사용자 인증 연동 시 user_id 전달
        )
        
        logger.info(f"변환 완료 - Easy: {easy_result['filename']}")
        
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장에 실패했습니다: {e}")

    return {
        "status": "success",
        "doc_id": raw_file_path.name if raw_file_path else None,
        "raw_file_path": str(raw_file_path) if raw_file_path else None,
        "json_file_path": easy_result["file_path"],
        "extracted_text_length": len(extracted),
        "extracted_text_preview": (extracted[:500] + "...") if len(extracted) > 500 else extracted,
        "easy_json": minimized,
        "storage_info": env_manager.get_storage_info(),
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
