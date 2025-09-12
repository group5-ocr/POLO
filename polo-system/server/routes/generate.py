# server/routes/generate.py
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from typing import Optional

from services.database.db import DB
from services import llm_client, math_client, viz_client

# 프로젝트에 공통 스키마가 있으면 사용하고,
# 없으면 아래 Fallback 스키마 사용.
try:
    from utils.schemas import PreprocessCallback, EasyChunkDone, VizDone, MathDone
except Exception:
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

    class MathDone(BaseModel):
        paper_id: str
        math_result_path: Optional[str] = None
        sections: Optional[list] = None


router = APIRouter()


# (선택) 멱등키/서명 검증 훅 – 필요 시 구현
async def _verify(idempotency_key: Optional[str], x_signature: Optional[str]) -> None:
    return


@router.post("/preprocess/callback")
async def preprocess_done(
    payload: PreprocessCallback,
    bg: BackgroundTasks,
    idempotency_key: Optional[str] = Header(default=None),
    x_signature: Optional[str] = Header(default=None),
):
    """
    전처리 완료 → 상태 초기화 → easy & math 병렬 시작
    """
    await _verify(idempotency_key, x_signature)

    tex_id = int(payload.paper_id)

    # 1) 상태 초기화
    await DB.init_pipeline_state(
        tex_id=tex_id,
        total_chunks=payload.total_chunks,
        jsonl_path=payload.jsonl_path,
        math_text_path=payload.math_text_path,
    )

    # 2) easy & math 시작 (백그라운드)
    bg.add_task(llm_client.ingest_jsonl, payload.paper_id, payload.jsonl_path)   # easy → /generate/easy-callback
    bg.add_task(math_client.ingest_text, payload.paper_id, payload.math_text_path)  # math → /generate/math-callback

    return {"ok": True, "tex_id": tex_id}


@router.post("/easy-callback")
async def easy_chunk_done(
    payload: EasyChunkDone,
    bg: BackgroundTasks,
    idempotency_key: Optional[str] = Header(default=None),
    x_signature: Optional[str] = Header(default=None),
):
    """
    easy가 index 단위로 재해석 완료 시 호출
    - chunk 저장 / easy_done += 1
    - viz 생성 트리거
    - 완료 조건 검사
    """
    await _verify(idempotency_key, x_signature)
    tex_id = int(payload.paper_id)

    await DB.save_easy_chunk(tex_id=tex_id, index=payload.index, rewritten_text=payload.rewritten_text)
    await DB.bump_counter(tex_id=tex_id, field="easy_done")

    # viz 생성 트리거
    bg.add_task(viz_client.generate, payload.paper_id, payload.index, payload.rewritten_text)  # 완료 시 /generate/viz-callback

    await _maybe_assemble(tex_id, bg)
    return {"ok": True}


@router.post("/viz-callback")
async def viz_done(
    payload: VizDone,
    bg: BackgroundTasks,
    idempotency_key: Optional[str] = Header(default=None),
    x_signature: Optional[str] = Header(default=None),
):
    """
    viz가 index 이미지 생성 완료 시 호출
    - 이미지 경로 저장 / viz_done += 1
    - 완료 조건 검사
    """
    await _verify(idempotency_key, x_signature)
    tex_id = int(payload.paper_id)

    await DB.save_viz_image(tex_id=tex_id, index=payload.index, image_path=payload.image_path)
    await DB.bump_counter(tex_id=tex_id, field="viz_done")

    await _maybe_assemble(tex_id, bg)
    return {"ok": True}


@router.post("/math-callback")
async def math_done(
    payload: MathDone,
    bg: BackgroundTasks,
    idempotency_key: Optional[str] = Header(default=None),
    x_signature: Optional[str] = Header(default=None),
):
    """
    math 보조설명 완료 시 호출
    - math 결과 저장 / math_done=true
    - 완료 조건 검사
    """
    await _verify(idempotency_key, x_signature)
    tex_id = int(payload.paper_id)

    await DB.save_math_result(tex_id=tex_id, result_path=payload.math_result_path, sections=payload.sections)
    await DB.set_flag(tex_id=tex_id, field="math_done", value=True)

    await _maybe_assemble(tex_id, bg)
    return {"ok": True}


# ---------- 완료 조건 검사 & 최종 조립 ----------
async def _maybe_assemble(tex_id: int, bg: BackgroundTasks) -> None:
    st = await DB.get_state(tex_id)
    if not st:
        return
    if st.total_chunks == 0:
        return
    if st.easy_done == st.total_chunks and st.viz_done == st.total_chunks and st.math_done:
        bg.add_task(DB.assemble_final, tex_id)