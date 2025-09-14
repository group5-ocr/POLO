# utils/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class PreprocessCallback(BaseModel):
    paper_id: str
    jsonl_path: str
    math_text_path: str
    total_chunks: int

class EasyChunkDone(BaseModel):
    paper_id: str
    index: int
    rewritten_text: str

class VizDone(BaseModel):
    paper_id: str
    index: int
    image_path: str

# ✅ math 콜백(두 형태 중 하나 선택)
class MathSection(BaseModel):
    equation_id: Optional[str] = None
    title: Optional[str] = None
    explanation: str
    refs: Optional[List[str]] = None

class MathDone(BaseModel):
    paper_id: str
    math_result_path: Optional[str] = None
    sections: Optional[List[MathSection]] = None

class FinalResult(BaseModel):
    paper_id: str
    items: List[dict]   # [{index, text, image_url}]
    math: dict          # {"path": "..."} or {"sections":[...]}