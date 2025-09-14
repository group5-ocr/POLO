# server/routes/math_generate.py
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel, Field

from services.database.db import DB
from services import math_client

router = APIRouter()


# ==== 스키마 ====
class MathStart(BaseModel):
    paper_id: str = Field(..., description="tex_id (문서 식별자)")
    tex_path: str = Field(..., description="Math 모델에 넘길 TeX 파일 경로")


class MathDone(BaseModel):
    paper_id: str
    # math 모델 결과물(우리 쪽에서 주로 참조할 산출물 경로)
    math_result_path: Optional[str] = None  # e.g. equations_explained.json
    report_tex_path: Optional[str] = None   # e.g. yolo_math_report.tex
    out_dir: Optional[str] = None           # e.g. .../_build
    # 원하면 세부 섹션/메타를 추가해도 됨
    sections: Optional[list] = None


# 선택: 멱등키/서명 검증 훅 – 필요 시 구현
async def _verify(idempotency_key: Optional[str], x_signature: Optional[str]) -> None:
    return


# ==== 라우트 ====
@router.post("/start")
async def math_start(
    payload: MathStart,
    bg: BackgroundTasks,
    idempotency_key: Optional[str] = Header(default=None),
    x_signature: Optional[str] = Header(default=None),
):
    """
    Math 파이프라인 시작 엔드포인트.
    - Math 모델 API(/math) 호출을 백그라운드로 시작
    - 완료되면 본 라우터의 /callback 으로 콜백(내부적으로 math_client가 호출)
    """
    await _verify(idempotency_key, x_signature)

    # tex_id는 정수로 저장되므로 변환
    tex_id = int(payload.paper_id)

    # DB 상태가 없다면 초기화 (preprocess 이전에 단독 호출되는 상황 방지용 안전망)
    st = await DB.get_state(tex_id)
    if not st:
        await DB.init_pipeline_state(
            tex_id=tex_id,
            total_chunks=0,            # 알 수 없으면 0
            jsonl_path="",             # 알 수 없으면 빈 값
            math_text_path=payload.tex_path,
        )

    # 콜백 URL 절대값 구성 (본 라우터 기준: /math/callback)
    base_cb = os.getenv("CALLBACK_URL", "http://localhost:8000").rstrip("/")
    callback_url = f"{base_cb}/math/callback"

    # Math 호출 시작 (백그라운드)
    bg.add_task(math_client.run, payload.paper_id, payload.tex_path, callback_url)

    return {"ok": True, "paper_id": payload.paper_id, "tex_path": payload.tex_path}


@router.post("/callback")
async def math_callback(
    payload: MathDone,
    idempotency_key: Optional[str] = Header(default=None),
    x_signature: Optional[str] = Header(default=None),
):
    """
    Math 모델 처리 완료 콜백.
    - DB에 결과 저장
    - math_done 플래그 true
    - (조건 충족 시) 최종 조립 시도
    """
    await _verify(idempotency_key, x_signature)
    tex_id = int(payload.paper_id)

    # 결과 저장
    await DB.save_math_result(
        tex_id=tex_id,
        result_path=payload.math_result_path,
        sections=payload.sections,
        report_tex_path=payload.report_tex_path if hasattr(DB, "save_math_result") else None,  # DB 구현체에 따라 무시될 수 있음
    )
    await DB.set_flag(tex_id=tex_id, field="math_done", value=True)

    # 완료 조건 체크 → 최종 조립
    await _maybe_assemble(tex_id)

    return {"ok": True, "paper_id": payload.paper_id}


# ---- 완료 조건 검사 & 최종 조립 ----
async def _maybe_assemble(tex_id: int) -> None:
    st = await DB.get_state(tex_id)
    if not st:
        return
    # total_chunks가 0인 경우(Easy/Viz 파이프라인 미실행)에는 수학만으로도 조립할지 정책에 따라 결정
    # 여기서는: easy+viz 파이프라인이 존재한다면 그 완료까지 기다리고, 없으면 math_done만으로도 조립
    if st.total_chunks > 0:
        if st.easy_done == st.total_chunks and st.viz_done == st.total_chunks and st.math_done:
            await DB.assemble_final(tex_id)
    else:
        if st.math_done:
            await DB.assemble_final(tex_id)
