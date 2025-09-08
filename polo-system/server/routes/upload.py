from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import fitz  # PyMuPDF
import os
import tempfile
import logging
import json
from datetime import datetime
from pathlib import Path
from services.llm_client import easy_llm

router = APIRouter()
logger = logging.getLogger(__name__)

# 데이터 디렉토리 경로 설정
BASE_DIR = Path(__file__).parent.parent.parent  # polo-system 루트
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

# 디렉토리 생성
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일 업로드 및 처리 시작"""
    
    # 파일 형식 검증
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    # Easy LLM 서비스 상태 확인
    if not easy_llm.health_check():
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다.")
    
    try:
        # 파일 내용 읽기
        content = await file.read()
        
        # 타임스탬프로 고유한 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(file.filename).stem
        safe_filename = f"{timestamp}_{base_name}.pdf"
        
        # data/raw에 원본 PDF 저장
        raw_file_path = RAW_DIR / safe_filename
        with open(raw_file_path, "wb") as f:
            f.write(content)
        logger.info(f"원본 PDF 저장: {raw_file_path}")
        
        # 임시 파일로 텍스트 추출용 복사본 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
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
            
            # JSON에 메타데이터 추가
            easy_json["metadata"] = {
                "original_filename": file.filename,
                "processed_at": datetime.now().isoformat(),
                "file_size": len(content),
                "extracted_text_length": len(extracted_text)
            }
            
            # data/outputs에 JSON 저장
            json_filename = f"{timestamp}_{base_name}.json"
            json_file_path = OUTPUTS_DIR / json_filename
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(easy_json, f, ensure_ascii=False, indent=2)
            logger.info(f"변환된 JSON 저장: {json_file_path}")
            
            return {
                "filename": file.filename,
                "raw_file_path": str(raw_file_path),
                "json_file_path": str(json_file_path),
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

@router.get("/upload-status")
async def get_upload_status():
    """업로드 상태 확인"""
    return {
        "raw_files_count": len(list(RAW_DIR.glob("*.pdf"))),
        "output_files_count": len(list(OUTPUTS_DIR.glob("*.json"))),
        "raw_dir": str(RAW_DIR),
        "outputs_dir": str(OUTPUTS_DIR)
    }
