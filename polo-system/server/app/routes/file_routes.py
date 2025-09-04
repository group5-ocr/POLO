import os
import fitz
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app import crud, llm_loader
from app.embedding import embed_text
from app.vector_store import store_vector
from datetime import datetime

router = APIRouter()

math_llm = llm_loader.load_math_llm()
summary_llm = llm_loader.load_summary_llm()
easy_llm = llm_loader.load_easy_llm()

@router.post("/upload")
async def upload_pdf(user_id: int, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    if not file.filename.endswith(".pdf"):
        return {"error": "PDF only"}

    pdf_bytes = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)

    doc = fitz.open("temp.pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # 1. Math Explanation
    math_text = await math_llm.summarize(full_text.encode())

    # 2. Summary
    summary_text = await summary_llm.summarize(full_text.encode())

    # 3. Easy Paper (based on summary)
    easy_text = await easy_llm.summarize(summary_text.encode())

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base = f"{user_id}_{timestamp}"

    paths = {
        "math": f"results/{base}_math.txt",
        "summary": f"results/{base}_summary.txt",
        "easy": f"results/{base}_easy.txt"
    }

    with open(paths["math"], "w", encoding="utf-8") as f:
        f.write(math_text)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        f.write(summary_text)
    with open(paths["easy"], "w", encoding="utf-8") as f:
        f.write(easy_text)

    await crud.create_paper(db, filename=file.filename, result_path=paths["summary"], user_id=user_id)

    return {
        "math_url": f"/files/download/{os.path.basename(paths['math'])}",
        "summary_url": f"/files/download/{os.path.basename(paths['summary'])}",
        "easy_url": f"/files/download/{os.path.basename(paths['easy'])}"
    }

embedding = embed_text(full_text)
store_vector(
    user_id=user_id,
    original_text=full_text,
    embedding=embedding,
    metadata={"filename": file.filename}
)
