# server/services/math_client.py
from __future__ import annotations

import os
import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("math_client")
logger.setLevel(logging.INFO)

MATH_MODEL_URL = os.getenv("MATH_MODEL_URL", "http://localhost:5004")
TIMEOUT = int(os.getenv("MATH_TIMEOUT", "900"))  # 수학은 좀 길 수 있으니 넉넉히
# 콜백 URL은 호출 시점에 인자로 주입 (기본값은 /math/callback으로 셋업 가능)


async def run(paper_id: str, tex_path: str, callback_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Math 모델 API 호출 → 결과를 오케스트레이터 콜백으로 전달.
    - 모델 API (POST {MATH_MODEL_URL}/math, body={"path": tex_path})
    - 응답(JSON)에서 결과물 경로(outputs.json, report_tex, out_dir)를 추출
    - 콜백 URL(기본: http://localhost:8000/math/callback)에 POST
    """
    # 기본 콜백
    if not callback_url:
        base_cb = os.getenv("CALLBACK_URL", "http://localhost:8000").rstrip("/")
        callback_url = f"{base_cb}/math/callback"

    # 1) 모델 호출
    api = f"{MATH_MODEL_URL.rstrip('/')}/math"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(api, json={"path": tex_path})
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        logger.error(f"❌ Math 모델 호출 실패: {e}")
        # 실패 시에도 콜백으로 실패 사실을 알리고 싶다면 아래 주석 해제
        # await _post_callback(callback_url, paper_id, None, None, None)
        raise

    # 2) 결과 파싱
    outputs = (result or {}).get("outputs", {})
    json_path = outputs.get("json")          # equations_explained.json
    report_tex = outputs.get("report_tex")   # yolo_math_report.tex
    out_dir = outputs.get("out_dir")

    logger.info(
        "✅ Math 완료: paper_id=%s json=%s tex=%s out_dir=%s",
        paper_id, json_path, report_tex, out_dir
    )

    # 3) 콜백 전송
    await _post_callback(
        callback_url=callback_url,
        paper_id=paper_id,
        math_result_path=json_path,
        report_tex_path=report_tex,
        out_dir=out_dir,
    )

    return {
        "ok": True,
        "paper_id": paper_id,
        "math_result_path": json_path,
        "report_tex_path": report_tex,
        "out_dir": out_dir,
        "raw": result,
    }


async def _post_callback(
    callback_url: str,
    paper_id: str,
    math_result_path: Optional[str],
    report_tex_path: Optional[str],
    out_dir: Optional[str],
) -> None:
    payload = {
        "paper_id": str(paper_id),
        "math_result_path": math_result_path,
        "report_tex_path": report_tex_path,
        "out_dir": out_dir,
        # sections 등 세부 항목을 추가로 싣고 싶으면 여기서 확장
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(callback_url, json=payload)
        logger.info("📨 Math 콜백 전송 완료: %s", callback_url)
    except Exception as e:
        logger.error("❌ Math 콜백 전송 실패: %s", e)
