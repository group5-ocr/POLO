from fastapi import APIRouter, UploadFile, File, HTTPException
import fitz  # PyMuPDF
from services.llm_client import easy_llm

router = APIRouter()

def extract_text_from_pdf(pdf_path: str):
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        raise RuntimeError(f"PDF 텍스트 추출 실패: {e}")

@router.post("/convert")
async def convert_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(pdf_path)
    result = easy_llm.generate(text)

    return {
        "filename": file.filename,
        "extracted_text": text[:500] + "...",
        "easy_text": result
    }
