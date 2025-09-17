# server/routes/generate.py
from __future__ import annotations

import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from typing import Optional, List, Dict, Any

from services.database.db import DB
from services import llm_client
# math는 나중에 붙일 예정이면 추후 import
# from services import math_client

try:
    # 프로젝트 공용 스키마가 있으면 사용
    from utils.schemas import PreprocessCallback
except Exception:
    # Fallback 스키마
    from pydantic import BaseModel

    class PreprocessCallback(BaseModel):
        paper_id: str
        jsonl_path: str
        total_chunks: int
        # math_text_path는 지금은 미사용 (수학 단계 나중에)

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
    전처리 완료 콜백 → 상태 초기화 → Easy 배치(/batch) 시작
    Easy는 내부에서 Viz까지 처리하고, 이미지 경로 요약을 반환.
    """
    await _verify(idempotency_key, x_signature)

    tex_id = int(payload.paper_id)

    # 1) 상태 초기화
    await DB.init_pipeline_state(
        tex_id=tex_id,
        total_chunks=payload.total_chunks,
        jsonl_path=payload.jsonl_path,
        math_text_path=None,   # 수학은 나중에
    )

    # 2) Easy 배치 시작 (백그라운드)
    out_dir = os.path.join("server", "data", "outputs", str(tex_id), "easy_outputs")
    bg.add_task(_run_easy_batch_and_record, tex_id, payload.jsonl_path, out_dir)

    return {"ok": True, "tex_id": tex_id}


async def _run_easy_batch_and_record(tex_id: int, jsonl_path: str, output_dir: str) -> None:
    """
    Easy /batch 실행 → 결과(images)를 DB에 반영 → 완료 검사
    """
    try:
        res: Dict[str, Any] = await llm_client.run_batch(
            paper_id=str(tex_id),
            jsonl_path=jsonl_path,
            output_dir=output_dir,
        )
    except Exception as e:
        # 실패 시 파이프라인 상태 기록 (원한다면 전용 필드/로그 추가)
        print(f"❌ Easy batch failed for tex_id={tex_id}: {e}")
        return

    # res: { ok, paper_id, count, success, failed, out_dir, images:[{index, image_path, ok, error?}, ...] }
    images: List[Dict[str, Any]] = res.get("images", [])

    # easy_done / viz_done 카운트 반영
    for item in images:
        idx = int(item.get("index", -1))
        if idx < 0:
            continue

        # easy 단계는 재서술 텍스트를 보존하지 않았으므로, 카운트만 올림
        await DB.bump_counter(tex_id=tex_id, field="easy_done")

        # viz 결과 저장
        if item.get("ok") and item.get("image_path"):
            await DB.save_viz_image(tex_id=tex_id, index=idx, image_path=item["image_path"])
            await DB.bump_counter(tex_id=tex_id, field="viz_done")

    # 수학은 아직 안 했으니 False 유지 (math_done)
    await _maybe_assemble(tex_id)


# ---------- (구) 청크 콜백 엔드포인트: 더 이상 사용하지 않음 ----------
@router.post("/easy-callback")
async def easy_chunk_done_deprecated(*args, **kwargs):
    raise HTTPException(status_code=410, detail="Deprecated: Easy now runs in batch and calls Viz internally.")

@router.post("/viz-callback")
async def viz_done_deprecated(*args, **kwargs):
    raise HTTPException(status_code=410, detail="Deprecated: Viz is triggered by Easy internally.")

@router.post("/math-callback")
async def math_done_deprecated(*args, **kwargs):
    raise HTTPException(status_code=410, detail="Deferred: math integration is not enabled yet.")


# ---------- 완료 조건 검사 & 최종 조립 ----------
async def _maybe_assemble(tex_id: int) -> None:
    st = await DB.get_state(tex_id)
    if not st:
        return
    if st.total_chunks == 0:
        return
    # 수학은 나중에: math_done 없이도 조립하려면 아래 조건을 완화해도 됨.
    # 지금은 easy + viz 가 모두 끝난 경우만 조립 시도.
    if st.easy_done == st.total_chunks and st.viz_done == st.total_chunks:
        await DB.assemble_final(tex_id)
