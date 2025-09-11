"""
POLO Math Service
- 전처리에서 합쳐진 수학 텍스트를 받아 섹션/설명/참조를 생성
- 완료 시 orchestrator에 콜백
"""
import os, json, time
from pathlib import Path
from typing import Optional, List, Dict, Any
import uvicorn, httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ORCH_BASE = os.getenv("ORCH_BASE", "http://localhost:8000")
SAVE_ROOT = Path(os.getenv("MATH_SAVE_ROOT", "data/math")).absolute()
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="POLO Math", version="1.0.0")

class GenReq(BaseModel):
    paper_id: str
    math_text_path: str

@app.get("/health")
def health():
    return {"ok": True}

def dummy_math_generator(text: str) -> Dict[str, Any]:
    # 실제 구현: LLM/수식 파서 등
    sections = [
        {"equation_id": "eq1", "title": "Definition", "explanation": "Definition of loss function ...", "refs": []},
        {"equation_id": "eq2", "title": "Theorem", "explanation": "Main convergence theorem ...", "refs": ["eq1"]},
    ]
    return {"sections": sections}

@app.post("/generate")
async def generate(req: GenReq):
    p = Path(req.math_text_path)
    if not p.exists():
        raise HTTPException(400, f"math_text_path not found: {p}")
    text = p.read_text(encoding="utf-8")

    # 1) 생성
    result = dummy_math_generator(text)

    # 2) 산출 저장
    out_dir = SAVE_ROOT / req.paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "math_sections.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) 콜백
    callback_url = f"{ORCH_BASE}/generate/math-callback"
    payload = {
        "paper_id": req.paper_id,
        "math_result_path": str(result_path),
        "sections": result.get("sections"),
    }
    async with httpx.AsyncClient(timeout=30) as cx:
        await cx.post(callback_url, json=payload)

    return {"ok": True, "result_path": str(result_path), "sections": result.get("sections")}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5004, reload=False)