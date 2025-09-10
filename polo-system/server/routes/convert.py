# server/routes/convert.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import fitz, tempfile, os, logging
from services.llm_client import easy_llm
from services.storage import save_conversion

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
async def convert_pdf(file: UploadFile = File(...), user_id: int = Query(0)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    if not easy_llm.health_check():
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다.")

    try:
        content = await file.read()
        text = _extract_text(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        easy_json = easy_llm.generate(text)
        if easy_json is None:
            raise HTTPException(status_code=500, detail="AI 모델 처리 중 오류가 발생했습니다.")

        saved = save_conversion(
            original_filename=file.filename,
            pdf_bytes=content,
            extracted_text=text,
            easy_json=easy_json,
            processing_info=easy_json.get("processing_info", {}),
            user_id=user_id,
        )

        # 공통 응답
        resp = {
            "status": "success",
            "filename": file.filename,
            "file_size": len(content),
            "extracted_text_length": len(text),
            "extracted_text_preview": text[:500] + ("..." if len(text) > 500 else ""),
            "easy_json": easy_json,
        }
        resp.update(saved)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF 변환 실패")
        raise HTTPException(status_code=500, detail=f"PDF 변환 실패: {str(e)}")