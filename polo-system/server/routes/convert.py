from fastapi import APIRouter, UploadFile, File, HTTPException
import fitz  # PyMuPDF
import os
import tempfile
import logging
from services.llm_client import easy_llm

router = APIRouter()
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출"""
    try:
        with fitz.open(pdf_path) as doc:
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():  # 빈 페이지 제외
                    text_parts.append(f"--- 페이지 {page_num + 1} ---\n{text}")
            return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 실패: {e}")
        raise RuntimeError(f"PDF 텍스트 추출 실패: {e}")

@router.post("/convert")
async def convert_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드하고 AI로 변환"""
    
    # 파일 형식 검증
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    # Easy LLM 서비스 상태 확인
    if not easy_llm.health_check():
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다.")
    
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # PDF 텍스트 추출
            logger.info(f"PDF 텍스트 추출 시작: {file.filename}")
            extracted_text = extract_text_from_pdf(tmp_path)
            
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")
            
            # AI 모델로 JSON 변환
            logger.info("AI 모델로 JSON 변환 시작")
            easy_json = easy_llm.generate(extracted_text)
            
            if easy_json is None:
                raise HTTPException(status_code=500, detail="AI 모델 처리 중 오류가 발생했습니다.")
            
            return {
                "filename": file.filename,
                "file_size": len(content),
                "extracted_text_length": len(extracted_text),
                "extracted_text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "easy_json": easy_json,
                "status": "success"
            }
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF 변환 실패: {e}")
        raise HTTPException(status_code=500, detail=f"PDF 변환 실패: {str(e)}")

@router.get("/model-status")
async def get_model_status():
    """AI 모델 서비스 상태 확인"""
    is_healthy = easy_llm.health_check()
    model_info = easy_llm.get_model_info() if is_healthy else None
    
    return {
        "model_available": is_healthy,
        "model_info": model_info,
        "status": "healthy" if is_healthy else "unavailable"
    }
